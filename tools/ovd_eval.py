"""Zero-shot box mAP and throughput of Grounding DINO on COCO, Fashionpedia or DIOR.

The counterpart of owlv2-torch's ``tools/ovd_eval.py``: one caption of every
class name, the standard COCO metric (maxDets 100), and the inference knobs
being evaluated. Runs in the ``evals`` venv, which has ``faster-coco-eval``,
``datasets`` and TensorRT 11:

    cd evals && PYTHONPATH=../gdino-torch/src .venv/bin/python ../gdino-torch/tools/ovd_eval.py \
        --dataset coco --model tiny --device cuda:0

COCO defaults to val2017's first 500 images (``instances_val2017_mini500.json``).
``ms_per_image_model`` is decoded PIL image to host detections (resize,
forward, selection, copy back) for the first post-processing variant.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from GDINOtorch import GroundingDINO, build_caption

COCO_ROOT = Path("/mnt/datasets/coco")
COCO_ANNS = {
    "mini500": COCO_ROOT / "annotations/instances_val2017_mini500.json",
    "val": COCO_ROOT / "annotations/instances_val2017.json",
}


def build_dataset(args):
    if args.dataset == "coco":
        from openvocab_evals.datasets import CocoFileDetectionDataset

        ann = COCO_ANNS.get(args.split or "mini500", args.split)
        return CocoFileDetectionDataset(ann, COCO_ROOT / "val2017", limit=args.limit)
    if args.dataset == "fashionpedia":
        from openvocab_evals.datasets import FashionpediaDetectionDataset

        return FashionpediaDetectionDataset(split=args.split or "val", limit=args.limit)
    if args.dataset == "dior":
        from openvocab_evals.datasets import DiorDetectionDataset

        return DiorDetectionDataset(split=args.split or "test", limit=args.limit)
    raise ValueError(args.dataset)


def build_model(args):
    if args.trt_engine or args.tensorrt:
        from GDINOtorch import GroundingDINOTRT

        kwargs = dict(output_dir=args.trt_dir, image_size=tuple(args.engine_size), max_batch=max(2, args.batch_size))
        if args.trt_engine:
            kwargs["engine_path"] = args.trt_engine
        model = GroundingDINOTRT(args.model, **kwargs)
    else:
        model = GroundingDINO(args.model)
    return model.eval().to(args.device)


def coco_metrics(coco_gt: dict, detections: list[dict]) -> dict:
    from faster_coco_eval import COCO, COCOeval_faster

    gt = COCO(coco_gt)
    ev = COCOeval_faster(gt, gt.loadRes(detections), iouType="bbox", print_function=lambda *a: None)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    names = ["map", "map_50", "map_75", "map_small", "map_medium", "map_large", "mar_1", "mar_10", "mar_100"]
    return {name: float(v) for name, v in zip(names, ev.stats)}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", default="coco", choices=["coco", "fashionpedia", "dior"])
    p.add_argument("--split", default=None, help="coco: mini500 (default), val, or an annotation path")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--model", default="tiny", help="tiny, base or a Hub repo id")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=1, help="batches are sorted by aspect ratio")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--size", type=int, default=800, help="short side")
    p.add_argument("--max-size", type=int, default=1333, help="long side cap")
    p.add_argument("--num-queries", type=int, default=None, help="decoder queries (default: the checkpoint's 900)")
    p.add_argument("--no-autocast", dest="autocast", action="store_false")
    p.add_argument("--precast-weights", action="store_true", help="GroundingDINO.precast_weights() (fp16, autocast only)")
    p.add_argument("--tensorrt", action="store_true", help="GroundingDINOTRT at --engine-size, built if missing")
    p.add_argument("--trt-engine", default=None)
    p.add_argument("--trt-dir", default="trt_engines")
    p.add_argument("--engine-size", type=int, nargs=2, default=(800, 1333), metavar=("H", "W"))
    p.add_argument("--template", default="{name}")
    p.add_argument("--pool", default="max", choices=["max", "mean"],
                   help="token logits -> class: max logit, or mean token probability (the original repo's COCO eval)")
    p.add_argument("--multi-label", action="store_true", help="top-k (query, class) pairs instead of one class per query")
    p.add_argument("--top-k", type=int, default=300)
    p.add_argument("--all-post", action="store_true", help="also score the other pool / multi-label combinations")
    p.add_argument("--save-metrics", default=None)
    p.add_argument("--tag", default=None)
    args = p.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    dataset = build_dataset(args)
    class_names = dataset.class_names
    model = build_model(args)
    if args.num_queries:
        model.num_queries = args.num_queries
    if args.precast_weights:
        model.precast_weights()
    caption, spans = build_caption([args.template.format(name=n.lower().replace("_", " ")) for n in class_names])
    input_ids, attention_mask = model.tokenize(caption)
    token_idx = model.tokenizer.phrase_token_indices(caption, spans)
    print(f"{len(dataset)} images, {len(class_names)} classes, caption {input_ids.shape[1]} tokens")
    with torch.inference_mode():
        text = model.encode_text(input_ids.to(device), attention_mask.to(device))

    # [C, L] token -> class map, for mean pooling.
    class_map = torch.zeros(len(class_names), input_ids.shape[1], device=device)
    for c, idx in enumerate(token_idx):
        class_map[c, idx] = 1.0

    order = list(range(len(dataset)))
    if args.batch_size > 1 and hasattr(dataset, "items"):  # group similar shapes so a batch pads little
        order.sort(key=lambda i: dataset.items[i][0]["width"] / dataset.items[i][0]["height"])
    batches = [order[i : i + args.batch_size] for i in range(0, len(order), args.batch_size)]
    loader = DataLoader(_BatchView(dataset, batches), batch_size=None, num_workers=args.num_workers,
                        prefetch_factor=4 if args.num_workers else None)

    variants = [(args.pool, args.multi_label)]
    if args.all_post:  # every pooling x labelling combination from the same forward passes
        variants += [(pool, ml) for pool in ("max", "mean") for ml in (False, True) if (pool, ml) != variants[0]]
    detections, model_ms, n_images = {v: [] for v in variants}, 0.0, 0
    torch.cuda.synchronize(device)
    t_start = time.perf_counter()
    for samples in tqdm(loader, desc="eval"):
        images = [s["image"] for s in samples]
        with torch.inference_mode():
            t0 = time.perf_counter()
            pixel_values, pixel_mask = model.preprocess_image(images, size=args.size, max_size=args.max_size, device=device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.autocast):
                logits, boxes = model.forward_object_detection_from_text(pixel_values, text, pixel_mask)
            logits = logits.float()
            xyxy = GroundingDINO.postprocess_boxes(boxes.float(), [s["target_size"] for s in samples])
            picks = {variants[0]: select(logits, xyxy, token_idx, class_map, *variants[0], args.top_k)}
            model_ms += (time.perf_counter() - t0) * 1000
            picks.update({v: select(logits, xyxy, token_idx, class_map, *v, args.top_k) for v in variants[1:]})
        n_images += len(samples)
        for v, (scores, label, picked) in picks.items():
            for s, sc, lb, bx in zip(samples, scores, label, picked):
                x0, y0 = s["offset"]
                for score, cat, (a, b, c, d) in zip(sc.tolist(), lb.tolist(), bx.tolist()):
                    detections[v].append({"image_id": s["image_id"], "category_id": cat, "score": score,
                                          "bbox": [a + x0, b + y0, c - a, d - b]})
    wall = time.perf_counter() - t_start

    gt = dataset.build_coco_gt()
    for (pool, multi_label), dets in detections.items():
        run = {"tag": args.tag, "model": args.model, "dataset": args.dataset, "split": args.split,
               "images": n_images, "size": args.size, "max_size": args.max_size, "batch_size": args.batch_size,
               "autocast": args.autocast, "precast_weights": args.precast_weights,
               "tensorrt": bool(args.tensorrt or args.trt_engine), "engine": args.trt_engine,
               "pool": pool, "multi_label": multi_label, "top_k": args.top_k, "num_queries": model.num_queries,
               "ms_per_image_model": model_ms / n_images, "wall_s": wall, **coco_metrics(gt, dets)}
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in run.items()}))
        if args.save_metrics:
            path = Path(args.save_metrics)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(json.dumps(run) + "\n")


def select(logits, xyxy, token_idx, class_map, pool, multi_label, top_k):
    """Per image: top-k (score, class, box) from per-token logits [B, Q, L]."""
    if pool == "max":
        cls = GroundingDINO.phrase_logits(logits, token_idx).sigmoid()
    else:
        cls = (logits.sigmoid() @ class_map.T) / class_map.sum(-1)
    B, Q, C = cls.shape
    if multi_label:
        scores, flat = cls.flatten(1).topk(min(top_k, Q * C), dim=1)
        query, label = flat // C, flat % C
    else:
        best, best_label = cls.max(-1)
        scores, query = best.topk(min(top_k, Q), dim=1)
        label = best_label.gather(1, query)
    picked = xyxy.gather(1, query[..., None].expand(-1, -1, 4))
    return scores.cpu().numpy(), label.cpu().numpy(), picked.cpu().numpy()


class _BatchView(torch.utils.data.Dataset):
    """Each item is one pre-ordered batch, so workers decode whole batches."""

    def __init__(self, dataset, batches):
        self.dataset, self.batches = dataset, batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        return [self.dataset[j] for j in self.batches[i]]


if __name__ == "__main__":
    main()
