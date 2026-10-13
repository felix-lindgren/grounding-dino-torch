"""
TensorRT runtime for Grounding DINO's image path.

``GroundingDINOTRT`` is a drop-in :class:`GroundingDINO` whose
``forward_object_detection_from_text`` (and so ``forward``) runs the Swin
backbone, fusion encoder, query selection and decoder as one TensorRT engine.
BERT stays in PyTorch; encode a caption once with ``encode_text`` and reuse it.

Missing artifacts are exported (``export.py``) and built (``build_engine.py``)
into ``output_dir`` on construction. An engine has a fixed image size: smaller
inputs are zero-padded bottom/right into it, which needs no output adjustment
since boxes are normalized to each image's unpadded area; larger inputs, CPU
inputs, and captions longer than the engine allows fall back to PyTorch. Build
at the size your images resize to (e.g. 750x1333 for 16:9) to avoid padding,
which slightly changes scores.

The runner executes on a private CUDA stream ordered against the caller's
current stream, so no CPU sync happens until the caller reads results back.
"""

import argparse
import warnings
from pathlib import Path

import torch
from PIL import Image

from GDINOtorch.torch_version.gdino import MODEL_ALIASES, GroundingDINO, TextFeatures
from GDINOtorch.utils.tokenizer import build_caption

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep
    trt = None
    TRT_AVAILABLE = False


if TRT_AVAILABLE:
    _TRT_TO_TORCH = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.bool: torch.bool,
    }

    class TRTRunner:
        """Minimal TensorRT 11 runtime for a serialized engine, on torch tensors.

        TensorRT allocates the context's scratch memory itself, once, for the
        largest shapes in the engine's profile, outside torch's caching
        allocator; sharing torch's cache fragments it when engine and PyTorch
        work alternate on a full GPU. Outputs are fresh tensors on every call.
        """

        def __init__(self, engine_path, device="cuda"):
            self.logger = trt.Logger(trt.Logger.WARNING)
            trt.init_libnvinfer_plugins(self.logger, "")  # the engine uses MultiscaleDeformableAttnPlugin_TRT
            self.device = torch.device(device)
            with torch.cuda.device(self.device):
                self.runtime = trt.Runtime(self.logger)
                with open(engine_path, "rb") as f:
                    self.engine = self.runtime.deserialize_cuda_engine(f.read())
                if self.engine is None:
                    raise RuntimeError(
                        f"Failed to deserialize TRT engine: {engine_path}. Engines only load in the "
                        "TensorRT version and on the GPU they were built for; delete it to rebuild."
                    )
                self.context = self.engine.create_execution_context()
                if self.context is None:  # its scratch may be sitting in torch's cache
                    torch.cuda.empty_cache()
                    self.context = self.engine.create_execution_context()
                if self.context is None:
                    raise RuntimeError(
                        f"Could not create a TensorRT execution context: it needs "
                        f"{self.engine.device_memory_size_v2 / 2**30:.2f} GiB of scratch; "
                        "rebuild with a smaller max_batch."
                    )
                self.stream = torch.cuda.Stream(self.device)
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            self.input_names = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            self.output_names = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

        def _torch_dtype(self, name):
            trt_dtype = self.engine.get_tensor_dtype(name)
            try:
                return _TRT_TO_TORCH[trt_dtype]
            except KeyError as e:
                raise TypeError(f"Unsupported TRT dtype: {trt_dtype}") from e

        def profile_max(self, name):
            """Largest shape the engine's (single) optimization profile accepts for ``name``."""
            return tuple(self.engine.get_tensor_profile_shape(name, 0)[2])

        def __call__(self, **inputs) -> dict[str, torch.Tensor]:
            caller = torch.cuda.current_stream(self.device)
            # Tensors allocated here on the caller's stream may be freed on return:
            # the caller waits for the engine below, so their reuse is ordered after it.
            kept = []
            for name in self.input_names:
                t = inputs[name].to(self.device, self._torch_dtype(name)).contiguous()
                kept.append(t)
                self.context.set_input_shape(name, tuple(t.shape))
                self.context.set_tensor_address(name, t.data_ptr())

            outputs = {}
            for name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                outputs[name] = torch.empty(shape, dtype=self._torch_dtype(name), device=self.device)
                self.context.set_tensor_address(name, outputs[name].data_ptr())

            # Everything above was issued on the caller's stream; run after it,
            # and make the caller's later work wait for the engine.
            self.stream.wait_stream(caller)
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v3 returned False.")
            caller.wait_stream(self.stream)
            return outputs

else:  # pragma: no cover - only hit when tensorrt isn't installed
    TRTRunner = None


def _model_slug(model_type: str) -> str:
    if model_type in MODEL_ALIASES:
        return model_type
    return model_type.rstrip("/").split("/")[-1]


class GroundingDINOTRT(GroundingDINO):
    """Grounding DINO whose image path runs as a TensorRT engine when available."""

    def __init__(
        self,
        model_type: str = "tiny",
        engine_path: str | Path | None = None,
        *,
        onnx_path: str | Path | None = None,
        output_dir: str | Path = ".",
        image_size: tuple[int, int] = (800, 1333),
        build_missing: bool = True,
        min_batch: int = 1,
        opt_batch: int = 1,
        max_batch: int = 2,
        opt_text_len: int = 32,
        workspace_gb: int = 8,
        fp16: bool = True,
        tf32: bool = True,
        simplify: bool = False,
    ):
        """``image_size`` is the engine's fixed (h, w); ignored when loading an existing engine.

        ``fp16`` exports an fp16 ONNX (TensorRT 11 takes precision from the
        graph), so fp16 and fp32 engines each have their own ONNX.
        """
        self.output_dir = Path(output_dir)
        self.image_size = tuple(image_size)
        precision = "fp16" if fp16 else ("tf32" if tf32 else "fp32")
        artifact_stem = f"gdino_{_model_slug(model_type)}_{self.image_size[0]}x{self.image_size[1]}"
        self.engine_path = (
            Path(engine_path) if engine_path is not None
            else self.output_dir / f"{artifact_stem}_{precision}.engine"
        )
        self.onnx_path = (
            Path(onnx_path) if onnx_path is not None
            else self.output_dir / f"{artifact_stem}{'_fp16' if fp16 else ''}.onnx"
        )
        self.fp16 = fp16
        self.build_missing = build_missing
        self.simplify = simplify
        self.engine_build_kwargs = {
            "min_batch": min_batch,
            "opt_batch": opt_batch,
            "max_batch": max_batch,
            "opt_text_len": opt_text_len,
            "workspace_gb": workspace_gb,
            "tf32": tf32,
        }
        self.trt = None
        super().__init__(model_type=model_type)

    def _ensure_artifacts(self):
        if self.engine_path.exists():
            return

        self.onnx_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.onnx_path.exists():
            from GDINOtorch.torch_version.export import export_image_path

            print(f"Exporting Grounding DINO image-path ONNX to {self.onnx_path}")
            if self.fp16 and not torch.cuda.is_available():
                print("CUDA is not available; an fp16 ONNX needs it to export, so nothing was built.")
                return
            export_image_path(self, self.onnx_path, self.image_size, simplify=self.simplify, fp16=self.fp16)

        if not TRT_AVAILABLE:
            print("TensorRT is not installed; ONNX was exported but engine build was skipped.")
            return
        if not torch.cuda.is_available():
            print("CUDA is not available; ONNX was exported but engine build was skipped.")
            return

        from GDINOtorch.torch_version.build_engine import build_engine

        print(f"Building Grounding DINO TensorRT engine to {self.engine_path}")
        built = build_engine(
            onnx_file_path=self.onnx_path,
            engine_file_path=self.engine_path,
            max_text_len=self.max_text_len,
            **self.engine_build_kwargs,
        )
        if built is None:
            warnings.warn(
                f"TensorRT engine build failed (see the log above); {self.engine_path} was not written "
                "and the model runs in PyTorch. Out-of-memory errors usually mean max_batch is too "
                "large for this GPU."
            )

    def _load_model(self, model_path):
        super()._load_model(model_path)
        if self.build_missing:
            self._ensure_artifacts()
        self.engine_ready = TRT_AVAILABLE and torch.cuda.is_available() and self.engine_path.exists()

    def _runner_for(self, device: torch.device):
        """The TensorRT runner on ``device``, created on first use there; None if unusable."""
        if not (self.engine_ready and device.type == "cuda"):
            return None
        if self.trt is None or self.trt.device != device:
            self.trt = TRTRunner(self.engine_path, device)
        return self.trt

    @staticmethod
    def _pad_to(pixel_values, pixel_mask, size):
        """Zero-pad images bottom/right to ``size`` (h, w); None if any side is too large."""
        B, C, height, width = pixel_values.shape
        if height > size[0] or width > size[1]:
            return None
        if (height, width) == size:
            return pixel_values, pixel_mask
        padded = pixel_values.new_zeros(B, C, *size)
        padded_mask = pixel_mask.new_zeros(B, *size)
        padded[..., :height, :width] = pixel_values
        padded_mask[..., :height, :width] = pixel_mask
        return padded, padded_mask

    def forward_object_detection_from_text(self, pixel_values, text: TextFeatures, pixel_mask=None):
        runner = self._runner_for(pixel_values.device)
        if runner is None:
            return super().forward_object_detection_from_text(pixel_values, text, pixel_mask)

        B, _, height, width = pixel_values.shape
        engine_size = runner.profile_max("pixel_values")[2:]
        max_batch = runner.profile_max("pixel_values")[0]
        max_text_len = runner.profile_max("text_features")[1]
        if pixel_mask is None:
            pixel_mask = torch.ones((B, height, width), dtype=torch.bool, device=pixel_values.device)
        fitted = self._pad_to(pixel_values, pixel_mask.bool(), engine_size)
        if fitted is None or text.features.shape[1] > max_text_len:
            warnings.warn(
                f"Input ({height}x{width} image, {text.features.shape[1]} tokens) exceeds the TensorRT "
                f"engine ({engine_size[0]}x{engine_size[1]}, {max_text_len} tokens); running in PyTorch."
            )
            return super().forward_object_detection_from_text(pixel_values, text, pixel_mask)
        pixel_values, pixel_mask = fitted
        text = self._expand_text(text, B)

        logits, boxes = [], []
        for start in range(0, B, max_batch):
            chunk = slice(start, start + max_batch)
            out = runner(
                pixel_values=pixel_values[chunk],
                pixel_mask=pixel_mask[chunk],
                text_features=text.features[chunk],
                text_token_mask=text.token_mask[chunk],
                text_self_attn_mask=text.self_attn_mask[chunk],
                text_position_ids=text.position_ids[chunk],
            )
            logits.append(out["pred_logits"])
            boxes.append(out["pred_boxes"])
        if len(logits) == 1:
            return logits[0], boxes[0]
        return torch.cat(logits), torch.cat(boxes)


def _parse_args():
    p = argparse.ArgumentParser(description="Run Grounding DINO with a TensorRT image path.")
    p.add_argument("--model-type", default="tiny", help='"tiny", "base", or a Hub repo id')
    p.add_argument("--output-dir", default=".")
    p.add_argument("--onnx", default=None)
    p.add_argument("--engine", default=None)
    p.add_argument("--image-size", type=int, nargs=2, default=(800, 1333), metavar=("H", "W"))
    p.add_argument("--image", default=None)
    p.add_argument("--phrases", nargs="+", default=["a cat", "a remote control"])
    p.add_argument("--min-batch", type=int, default=1)
    p.add_argument("--opt-batch", type=int, default=1)
    p.add_argument("--max-batch", type=int, default=2)
    p.add_argument("--workspace-gb", type=int, default=8)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--no-tf32", action="store_true")
    p.add_argument("--no-build", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    model = GroundingDINOTRT(
        args.model_type,
        engine_path=args.engine,
        onnx_path=args.onnx,
        output_dir=args.output_dir,
        image_size=tuple(args.image_size),
        build_missing=not args.no_build,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        workspace_gb=args.workspace_gb,
        fp16=not args.no_fp16,
        tf32=not args.no_tf32,
    )
    if args.image is None or not Path(args.image).exists():
        print("No --image given; artifact setup is complete.")
        raise SystemExit(0)

    model.eval().cuda()
    image = Image.open(args.image)
    pixel_values, pixel_mask = model.preprocess_image(image)
    caption, spans = build_caption(args.phrases)
    input_ids, attention_mask = model.tokenize(caption)
    pixel_values, pixel_mask = pixel_values.cuda(), pixel_mask.cuda()
    with torch.inference_mode():
        text = model.encode_text(input_ids.cuda(), attention_mask.cuda())
        pred_logits, pred_boxes = model.forward_object_detection_from_text(pixel_values, text, pixel_mask)
    token_indices = model.tokenizer.phrase_token_indices(caption, spans)
    scores, labels = GroundingDINO.phrase_logits(pred_logits, token_indices).sigmoid().max(-1)
    boxes = GroundingDINO.postprocess_boxes(pred_boxes, [image.size[::-1]])
    for i in scores[0].topk(5).indices.tolist():
        box = [round(v, 1) for v in boxes[0, i].tolist()]
        print(f"{args.phrases[labels[0, i]]:20s} {scores[0, i].item():.3f} {box}")

    # Quick micro-benchmark of the image path.
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        for _ in range(3):
            model.forward_object_detection_from_text(pixel_values, text, pixel_mask)
        start.record()
        for _ in range(10):
            model.forward_object_detection_from_text(pixel_values, text, pixel_mask)
        end.record()
    torch.cuda.synchronize()
    print(f"image path: {start.elapsed_time(end) / 10:.1f} ms/batch ({'TensorRT' if model.trt else 'PyTorch'})")
