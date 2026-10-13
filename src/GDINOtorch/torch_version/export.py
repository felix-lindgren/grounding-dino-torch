"""Export Grounding DINO's image path to ONNX for TensorRT.

The exported graph is everything after the text tower: Swin backbone, fusion
encoder, two-stage query selection, decoder and heads, taking a caption already
encoded by :meth:`GroundingDINO.encode_text` as input. BERT stays in PyTorch:
it runs once per caption, and its phrase masks use ``cummax``, which ONNX lacks.

The image size is fixed at export (Swin's padding and shifted-window masks are
traced as constants); batch size and caption length are dynamic axes, so one
engine serves any vocabulary. Deformable attention is exported as TensorRT's
fused ``MultiscaleDeformableAttnPlugin_TRT`` instead of four ``GridSample`` per
layer, about 10x faster in the engine; the ONNX therefore only runs under
TensorRT, not onnxruntime.

TensorRT 11 builds every network strongly typed: each layer runs in the type
the ONNX gives it, and there is no builder flag to lower precision. An fp16
engine therefore needs an fp16 ONNX, which ``fp16=True`` traces under
``torch.autocast`` so the graph carries autocast's casts (fp16 matmuls and
convolutions, fp32 normalization, softmax and position embeddings).
"""

import argparse
import contextlib
import warnings
from pathlib import Path

import torch
import torch.nn as nn

from GDINOtorch.torch_version import gdino
from GDINOtorch.torch_version.gdino import GroundingDINO, TextFeatures

INPUT_NAMES = [
    "pixel_values", "pixel_mask", "text_features", "text_token_mask", "text_self_attn_mask", "text_position_ids",
]
OUTPUT_NAMES = ["pred_logits", "pred_boxes"]

_torch_deformable_attention = gdino.multi_scale_deformable_attention


class _DeformableAttentionPlugin(torch.autograd.Function):
    """PyTorch MSDA while tracing; ``MultiscaleDeformableAttnPlugin_TRT`` in the ONNX.

    The plugin takes the tensors of :func:`gdino.multi_scale_deformable_attention`
    plus int32 ``spatial_shapes`` ``[levels, 2]`` (h, w) and ``level_start_index``
    ``[levels]``, and returns ``[B, Q, heads, head_dim]``.
    """

    @staticmethod
    def forward(ctx, value, spatial_shapes, level_start_index, sampling_locations, attention_weights):
        shapes = [tuple(s) for s in spatial_shapes.tolist()]
        out = _torch_deformable_attention(value, shapes, sampling_locations, attention_weights)
        return out.view(*sampling_locations.shape[:3], value.shape[-1])

    @staticmethod
    def symbolic(g, value, spatial_shapes, level_start_index, sampling_locations, attention_weights):
        return g.op(
            "trt::MultiscaleDeformableAttnPlugin_TRT",
            value, spatial_shapes, level_start_index, sampling_locations, attention_weights,
            plugin_version_s="1",
        )


def _plugin_deformable_attention(value, spatial_shapes_list, sampling_locations, attention_weights):
    # The plugin takes its three float inputs in one type; under autocast the
    # sampling locations come out fp32 while value is fp16.
    sampling_locations = sampling_locations.to(value.dtype)
    attention_weights = attention_weights.to(value.dtype)
    sizes = [h * w for h, w in spatial_shapes_list]
    spatial_shapes = torch.tensor(spatial_shapes_list, dtype=torch.int32, device=value.device)
    level_start_index = torch.tensor(
        [sum(sizes[:i]) for i in range(len(sizes))], dtype=torch.int32, device=value.device
    )
    out = _DeformableAttentionPlugin.apply(
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights
    )
    return out.flatten(2)


@contextlib.contextmanager
def _deformable_attention_as_plugin():
    gdino.multi_scale_deformable_attention = _plugin_deformable_attention
    try:
        yield
    finally:
        gdino.multi_scale_deformable_attention = _torch_deformable_attention


class ImagePath(nn.Module):
    """``forward_object_detection_from_text`` with :class:`TextFeatures` unpacked into tensors."""

    def __init__(self, model: GroundingDINO):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, pixel_mask, text_features, text_token_mask, text_self_attn_mask,
                text_position_ids):
        text = TextFeatures(text_features, text_token_mask, text_self_attn_mask, text_position_ids)
        # Called unbound, so a TensorRT subclass's override is not what gets traced.
        logits, boxes = GroundingDINO.forward_object_detection_from_text(self.model, pixel_values, text, pixel_mask)
        # Outputs stay fp32 whatever precision the graph runs in.
        return logits.float(), boxes.float()


def export_image_path(
    model: GroundingDINO,
    onnx_path: str | Path,
    image_size: tuple[int, int] = (800, 1333),
    device: str | torch.device | None = None,
    simplify: bool = False,
    fp16: bool = False,
):
    """Trace ``model``'s image path at ``image_size`` (h, w) and write it to ``onnx_path``.

    Tracing runs on ``device`` (CUDA when available), keeping the trace's
    activations off host RAM; the model is moved back afterwards. ``fp16``
    traces under CUDA fp16 autocast, for an fp16 engine. ``simplify``
    runs onnx-simplifier, which TensorRT does not need (it folds constants
    itself) and which takes minutes on the larger models.
    """
    height, width = image_size
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if fp16 and torch.device(device).type != "cuda":
        raise ValueError("An fp16 export traces under CUDA autocast and needs a CUDA device.")
    original_device = next(model.parameters()).device
    was_training = model.training
    model.eval().to(device)
    try:
        input_ids, attention_mask = model.tokenize("a cat. a remote control.")
        # Batch 1: the tracer keeps every activation until it finishes, and
        # large at batch 2 does not fit a 12 GB GPU. The batch axis stays dynamic.
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            text = model.encode_text(input_ids.to(device), attention_mask.to(device))
            pixel_values = torch.randn(1, 3, height, width, device=device)
            pixel_mask = torch.ones(1, height, width, dtype=torch.bool, device=device)
            with _deformable_attention_as_plugin(), torch.autocast("cuda", torch.float16, enabled=fp16):
                torch.onnx.export(
                    ImagePath(model),
                    (pixel_values, pixel_mask, *text),
                    str(onnx_path),
                    dynamo=False,  # the torch.export exporter cannot take the plugin symbolic
                    opset_version=17,
                    custom_opsets={"trt": 1},
                    input_names=INPUT_NAMES,
                    output_names=OUTPUT_NAMES,
                    dynamic_axes={
                        "pixel_values": {0: "batch"},
                        "pixel_mask": {0: "batch"},
                        "text_features": {0: "batch", 1: "text_len"},
                        "text_token_mask": {0: "batch", 1: "text_len"},
                        "text_self_attn_mask": {0: "batch", 1: "text_len", 2: "text_len"},
                        "text_position_ids": {0: "batch", 1: "text_len"},
                        "pred_logits": {0: "batch", 2: "text_len"},
                        "pred_boxes": {0: "batch"},
                    },
                )
    finally:
        model.to(original_device).train(was_training)
        if torch.device(device).type == "cuda":
            # Hand the trace's memory back to the driver; an engine build right
            # after needs it and cannot take it from torch's cache.
            torch.cuda.empty_cache()

    if simplify:
        import onnx
        from onnxsim import simplify as onnxsim_simplify

        try:
            model_simp, check = onnxsim_simplify(onnx.load(str(onnx_path)))
        except Exception as e:  # noqa: BLE001 - keep the valid unsimplified export
            print(f"WARNING: onnx-simplifier failed ({e}); keeping the unsimplified model.")
            return
        if check:
            onnx.save_model(model_simp, str(onnx_path))
        else:
            print("WARNING: onnx-simplifier validation failed; keeping the unsimplified model.")


def _parse_args():
    p = argparse.ArgumentParser(description="Export the Grounding DINO image path to ONNX for TensorRT.")
    p.add_argument("--model-type", default="tiny", help='"tiny", "base", or a Hub repo id')
    p.add_argument("--onnx", default="gdino_image.onnx")
    p.add_argument("--image-size", type=int, nargs=2, default=(800, 1333), metavar=("H", "W"))
    p.add_argument("--simplify", action="store_true")
    p.add_argument("--fp16", action="store_true", help="export for an fp16 engine (needs CUDA)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export_image_path(
        GroundingDINO(args.model_type), args.onnx, tuple(args.image_size), simplify=args.simplify, fp16=args.fp16
    )
    print(f"Exported {args.onnx}")
