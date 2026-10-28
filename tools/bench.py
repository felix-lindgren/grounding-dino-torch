"""Latency of Grounding DINO at batch 1, end to end and by stage.

Times four COCO val images (two landscape, two portrait) from
decoded PIL image to host detections, then breaks one forward down by module
with the CUDA profiler. Run in the ``evals`` venv for TensorRT 11:

    cd evals && PYTHONPATH=../gdino-torch/src .venv/bin/python ../gdino-torch/tools/bench.py --model tiny --device cuda:1
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

from GDINOtorch import GroundingDINO, build_caption

COCO_IMAGES = Path("/mnt/datasets/coco/val2017")
# 640x426, 586x640, 640x483, 375x500 (W x H): COCO's common shapes.
DEFAULT_IMAGES = ["000000000139.jpg", "000000000285.jpg", "000000000632.jpg", "000000000724.jpg"]
COCO_80 = [name.replace("_", " ") for name in (
    "person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign "
    "parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag "
    "tie suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard "
    "tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot "
    "hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
    "cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear hair_drier toothbrush"
).split(" ")]


def detect(model, image, text, token_idx, args):
    """PIL image -> top-k (scores, labels, xyxy boxes) on the host."""
    pixel_values, pixel_mask = model.preprocess_image(image, size=args.size, max_size=args.max_size)
    with torch.autocast("cuda", dtype=torch.float16, enabled=args.autocast):
        logits, boxes = model.forward_object_detection_from_text(pixel_values, text, pixel_mask)
    cls = GroundingDINO.phrase_logits(logits.float(), token_idx).sigmoid()
    best, label = cls.max(-1)
    scores, query = best.topk(300, dim=1)
    xyxy = GroundingDINO.postprocess_boxes(boxes.float(), [image.size[::-1]])
    picked = xyxy.gather(1, query[..., None].expand(-1, -1, 4))
    return scores.cpu(), label.gather(1, query).cpu(), picked.cpu()


def stage_breakdown(model, image, text, args):
    """GPU time per top-level stage, from CUDA events around each module call (synchronizing)."""
    timings = defaultdict(list)
    stages = {
        "backbone": model.backbone,
        "input_proj": model.input_proj,
        "encoder.fusion": [l.fusion_layer for l in model.encoder.layers],
        "encoder.text": [l.text_enhancer_layer for l in model.encoder.layers],
        "encoder.deformable": [l.deformable_layer for l in model.encoder.layers],
        "decoder": model.decoder,
    }
    handles = []
    for name, modules in stages.items():
        for m in modules if isinstance(modules, list) else [modules]:
            def pre(mod, inp, name=name):
                torch.cuda.synchronize()
                mod._t0 = time.perf_counter()

            def post(mod, inp, out, name=name):
                torch.cuda.synchronize()
                timings[name].append((time.perf_counter() - mod._t0) * 1000)

            handles += [m.register_forward_pre_hook(pre), m.register_forward_hook(post)]
    pixel_values, pixel_mask = model.preprocess_image(image, size=args.size, max_size=args.max_size)
    for _ in range(3):
        timings.clear()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16, enabled=args.autocast):
            model.forward_object_detection_from_text(pixel_values, text, pixel_mask)
        torch.cuda.synchronize()
        total = (time.perf_counter() - t0) * 1000
    for h in handles:
        h.remove()
    out = {name: round(sum(v), 2) for name, v in timings.items()}
    out["other (masks, pos, proposals, heads)"] = round(total - sum(out.values()), 2)
    out["total"] = round(total, 2)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="tiny")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--images", nargs="+", default=DEFAULT_IMAGES)
    p.add_argument("--size", type=int, default=800)
    p.add_argument("--max-size", type=int, default=1333)
    p.add_argument("--num-queries", type=int, default=None, help="decoder queries (default: the checkpoint's 900)")
    p.add_argument("--no-autocast", dest="autocast", action="store_false")
    p.add_argument("--precast-weights", action="store_true", help="GroundingDINO.precast_weights() (fp16, autocast only)")
    p.add_argument("--trt-engine", default=None)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--profile", action="store_true", help="also print the top CUDA kernels")
    p.add_argument("--no-stages", dest="stages", action="store_false")
    args = p.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    if args.trt_engine:
        from GDINOtorch import GroundingDINOTRT

        model = GroundingDINOTRT(args.model, engine_path=args.trt_engine, build_missing=False)
    else:
        model = GroundingDINO(args.model)
    model = model.eval().to(device)
    if args.num_queries:
        model.num_queries = args.num_queries
    if args.precast_weights:
        model.precast_weights()
    caption, spans = build_caption(COCO_80)
    input_ids, attention_mask = model.tokenize(caption)
    token_idx = model.tokenizer.phrase_token_indices(caption, spans)
    images = [Image.open(COCO_IMAGES / name).convert("RGB") for name in args.images]

    with torch.inference_mode():
        text = model.encode_text(input_ids.to(device), attention_mask.to(device))
        results = {}
        for name, image in zip(args.images, images):
            for _ in range(3):
                detect(model, image, text, token_idx, args)
            times = []
            for _ in range(args.iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                detect(model, image, text, token_idx, args)
                times.append((time.perf_counter() - t0) * 1000)
            times.sort()
            results[f"{name} {image.size[0]}x{image.size[1]}"] = round(times[len(times) // 2], 2)
        print(json.dumps({"model": args.model, "trt": args.trt_engine, "size": args.size, "queries": model.num_queries,
                          "precast_weights": args.precast_weights,
                          "median_ms": results,
                          "mean_ms": round(sum(results.values()) / len(results), 2)}))
        if args.stages and not args.trt_engine:
            print(json.dumps(stage_breakdown(model, images[0], text, args)))
        if args.profile:
            from torch.profiler import ProfilerActivity, profile

            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
                for _ in range(3):
                    detect(model, images[0], text, token_idx, args)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))


if __name__ == "__main__":
    main()
