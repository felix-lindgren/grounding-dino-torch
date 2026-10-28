"""Build a TensorRT engine from the ONNX image path written by ``export.py``.

The image size is fixed by the ONNX; the optimization profile spans image
batch sizes ``min_batch..max_batch`` and caption lengths ``1..max_text_len``.
An execution context reserves scratch memory for the largest profile shapes:
at 800x1333 and 256 tokens, ~1.5 GiB for tiny/base plus ~0.17 GiB per image
of ``max_batch``, and ~4.6 GiB plus ~0.3 GiB per image for Swin-L. Batching
barely raises throughput (0-7% per image), so keep ``max_batch`` small.

Requires TensorRT 11, which builds every network strongly typed: precision
comes from the ONNX, so build an fp16 engine from an ``export.py --fp16`` ONNX.
"""

import argparse
from pathlib import Path

import tensorrt as trt

DEFAULT_MIN_BATCH = 1
DEFAULT_OPT_BATCH = 1
DEFAULT_MAX_BATCH = 2
DEFAULT_OPT_TEXT_LEN = 32
DEFAULT_MAX_TEXT_LEN = 256
DEFAULT_WORKSPACE_GB = 8
# Image (h, w) range of a dynamic-size engine: any orientation of the default
# resize (short side 800, long side at most 1333), tuned for 4:3 landscape.
# Below ~210x210 an image has fewer than the 900 proposals query selection keeps.
DEFAULT_MIN_SIZE = (256, 256)
DEFAULT_OPT_SIZE = (800, 1066)
DEFAULT_MAX_SIZE = (1333, 1333)


def build_engine(
    onnx_file_path: str | Path,
    engine_file_path: str | Path,
    min_batch: int = DEFAULT_MIN_BATCH,
    opt_batch: int = DEFAULT_OPT_BATCH,
    max_batch: int = DEFAULT_MAX_BATCH,
    opt_text_len: int = DEFAULT_OPT_TEXT_LEN,
    max_text_len: int = DEFAULT_MAX_TEXT_LEN,
    workspace_gb: int = DEFAULT_WORKSPACE_GB,
    tf32: bool = True,
    min_size: tuple[int, int] = DEFAULT_MIN_SIZE,
    opt_size: tuple[int, int] = DEFAULT_OPT_SIZE,
    max_size: tuple[int, int] = DEFAULT_MAX_SIZE,
):
    """Build and save the engine; returns its path, or None on failure.

    ``min_size`` / ``opt_size`` / ``max_size`` (h, w) bound the image size of
    an ONNX exported with ``dynamic_size``; a fixed-size ONNX ignores them.
    Context memory grows with ``max_size``, and kernels are chosen for ``opt_size``.

    ``tf32`` (TensorRT's default) speeds up the layers the ONNX keeps in fp32 at
    a small accuracy cost: an fp32 engine drifts ~0.004 in score with it, ~0.002 without.
    """
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")  # registers MultiscaleDeformableAttnPlugin_TRT
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_file_path)):
        print("ERROR: Failed to parse the ONNX file.")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        return None

    config = builder.create_builder_config()
    # Workspace is a ceiling, not a reservation; larger -> more tactics considered.
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
    if not tf32:
        config.clear_flag(trt.BuilderFlag.TF32)

    inputs = {network.get_input(i).name: network.get_input(i) for i in range(network.num_inputs)}
    _, channels, height, width = inputs["pixel_values"].shape
    d_model = inputs["text_features"].shape[-1]
    text_shapes = {
        "text_features": lambda b, n: (b, n, d_model),
        "text_token_mask": lambda b, n: (b, n),
        "text_self_attn_mask": lambda b, n: (b, n, n),
        "text_position_ids": lambda b, n: (b, n),
    }
    profile = builder.create_optimization_profile()
    batches = (min_batch, opt_batch, max_batch)
    sizes = [min_size, opt_size, max_size] if height < 0 or width < 0 else [(height, width)] * 3
    profile.set_shape("pixel_values", *[(b, channels, *hw) for b, hw in zip(batches, sizes)])
    profile.set_shape("pixel_mask", *[(b, *hw) for b, hw in zip(batches, sizes)])
    for name, shape in text_shapes.items():
        profile.set_shape(name, shape(min_batch, 1), shape(opt_batch, opt_text_len), shape(max_batch, max_text_len))
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("Failed to create engine")
        return None
    with open(engine_file_path, "wb") as f:
        f.write(engine_bytes)
    return Path(engine_file_path)


def _parse_args():
    p = argparse.ArgumentParser(description="Build a Grounding DINO image-path TensorRT engine.")
    p.add_argument("--onnx", default="gdino_image.onnx")
    p.add_argument("--engine", default="gdino_image.engine")
    p.add_argument("--min-batch", type=int, default=DEFAULT_MIN_BATCH)
    p.add_argument("--opt-batch", type=int, default=DEFAULT_OPT_BATCH)
    p.add_argument("--max-batch", type=int, default=DEFAULT_MAX_BATCH)
    p.add_argument("--opt-text-len", type=int, default=DEFAULT_OPT_TEXT_LEN)
    p.add_argument("--max-text-len", type=int, default=DEFAULT_MAX_TEXT_LEN)
    p.add_argument("--workspace-gb", type=int, default=DEFAULT_WORKSPACE_GB)
    p.add_argument("--no-tf32", action="store_true")
    p.add_argument("--min-size", type=int, nargs=2, default=DEFAULT_MIN_SIZE, metavar=("H", "W"))
    p.add_argument("--opt-size", type=int, nargs=2, default=DEFAULT_OPT_SIZE, metavar=("H", "W"))
    p.add_argument("--max-size", type=int, nargs=2, default=DEFAULT_MAX_SIZE, metavar=("H", "W"))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    engine = build_engine(
        onnx_file_path=args.onnx,
        engine_file_path=args.engine,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        opt_text_len=args.opt_text_len,
        max_text_len=args.max_text_len,
        workspace_gb=args.workspace_gb,
        tf32=not args.no_tf32,
        min_size=tuple(args.min_size),
        opt_size=tuple(args.opt_size),
        max_size=tuple(args.max_size),
    )
    if engine:
        print(f"TensorRT engine has been created and saved to {args.engine}")
    else:
        print("Failed to create TensorRT engine.")
