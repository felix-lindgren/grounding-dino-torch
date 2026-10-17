"""Few-shot Grounding DINO adaptation by learned text embeddings (arXiv:2504.07252).

Samples ``--shots`` instances per class from a training split, trains only the
class embeddings of ``GDINOtorch.few_shot.LearnedTextEmbeddings``, and
evaluates COCO mAP on a held-out split, next to the zero-shot text-prompt
baseline. Run with an environment that has ``faster-coco-eval``, ``scipy`` and
``datasets`` (the ``evals`` venv), e.g.

    cd evals && PYTHONPATH=../gdino-torch/src .venv/bin/python \
        ../gdino-torch/tools/few_shot_adapt.py --dataset dior --shots 8 --size base
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from GDINOtorch import GroundingDINO, build_caption
from GDINOtorch.few_shot import FewShotGDINO
from openvocab_evals.metrics import coco_eval_with_custom_sizes


class Record:
    """One image: a lazy loader plus xyxy pixel boxes and 0-based labels."""

    def __init__(self, image_id, loader, size, boxes, labels):
        self.image_id, self.loader, self.size = image_id, loader, size
        self.boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        self.labels = np.asarray(labels, dtype=np.int64)


def load_dior(split, limit):
    from datasets import load_dataset

    ds = load_dataset("HichTala/dior", split=split)
    names = ds.features["objects"]["category"].feature.names
    # Columns without the image decode cheaply; images are decoded on demand.
    meta = ds.remove_columns("image")
    records = []
    for i in range(len(ds) if limit is None else min(limit, len(ds))):
        row = meta[i]
        xywh = np.asarray(row["objects"]["bbox"], dtype=np.float32).reshape(-1, 4)
        xyxy = np.concatenate([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], 1)
        records.append(Record(
            int(row["image_id"]), lambda i=i: ds[i]["image"].convert("RGB"),
            (row["height"], row["width"]), xyxy, row["objects"]["category"],
        ))
    return records, names


def load_coco(ann_file, image_root, limit):
    coco = json.load(open(ann_file))
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_index = {c["id"]: i for i, c in enumerate(cats)}
    by_image = {}
    for a in coco["annotations"]:
        if not a.get("iscrowd", 0):
            by_image.setdefault(a["image_id"], []).append(a)
    records = []
    for im in coco["images"][: limit]:
        anns = [a for a in by_image.get(im["id"], []) if a["bbox"][2] > 1 and a["bbox"][3] > 1]
        xywh = np.asarray([a["bbox"] for a in anns], dtype=np.float32).reshape(-1, 4)
        xyxy = np.concatenate([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], 1)
        path = Path(image_root) / im["file_name"]
        records.append(Record(
            im["id"], lambda p=path: Image.open(p).convert("RGB"), (im["height"], im["width"]),
            xyxy, [cat_index[a["category_id"]] for a in anns],
        ))
    return records, [c["name"] for c in cats]


def sample_shots(records, num_classes, shots, seed):
    """Per class, random images until it has at least ``shots`` instances."""
    rng = random.Random(seed)
    chosen, counts = [], np.zeros(num_classes, dtype=int)
    for c in range(num_classes):
        pool = [r for r in records if (r.labels == c).any()]
        rng.shuffle(pool)
        for r in pool:
            if counts[c] >= shots:
                break
            if r in chosen:
                continue
            chosen.append(r)
            counts += np.bincount(r.labels, minlength=num_classes)
    return chosen


@torch.no_grad()
def evaluate(model, records, class_names, args, device, embeds=None, title=""):
    """COCO mAP; ``embeds=None`` evaluates the zero-shot text-prompt baseline."""
    if embeds is None:
        caption, spans = build_caption([args.query_template.format(name=n.lower()) for n in class_names])
        input_ids, attention_mask = model.tokenize(caption)
        text = model.encode_text(input_ids.to(device), attention_mask.to(device))
        token_idx = model.tokenizer.phrase_token_indices(caption, spans)
    else:
        text = embeds.text_features(model.text_projection)
    dets = []
    for i in tqdm(range(0, len(records), args.eval_batch_size), desc=f"eval {title}"):
        chunk = records[i : i + args.eval_batch_size]
        pixel_values, pixel_mask = model.preprocess_image([r.loader() for r in chunk], size=args.image_size)
        with torch.autocast("cuda", dtype=torch.float16):
            logits, boxes = model.forward_object_detection_from_text(pixel_values, text, pixel_mask)
        logits = logits.float()
        scores = (embeds.class_logits(logits) if embeds is not None
                  else GroundingDINO.phrase_logits(logits, token_idx)).sigmoid()
        xyxy = GroundingDINO.postprocess_boxes(boxes.float(), [r.size for r in chunk])
        for r, s, b in zip(chunk, scores, xyxy):
            top = s.flatten().topk(min(args.top_k, s.numel()))
            q, c = top.indices // s.shape[1], top.indices % s.shape[1]
            bq = b[q].cpu().numpy()
            for score, cls, box in zip(top.values.cpu().tolist(), c.cpu().tolist(), bq):
                dets.append({"image_id": r.image_id, "category_id": cls, "score": score,
                             "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]})
    gt = {"images": [{"id": r.image_id, "height": r.size[0], "width": r.size[1]} for r in records],
          "annotations": [], "categories": [{"id": i, "name": n} for i, n in enumerate(class_names)]}
    for r in records:
        for box, cls in zip(r.boxes, r.labels):
            gt["annotations"].append({"id": len(gt["annotations"]) + 1, "image_id": r.image_id, "category_id": int(cls),
                                      "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                                      "area": float((box[2] - box[0]) * (box[3] - box[1])), "iscrowd": 0})
    return coco_eval_with_custom_sizes(gt, dets, class_names, title=title)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dior", help="'dior' or 'coco'")
    p.add_argument("--train-ann"), p.add_argument("--train-images")
    p.add_argument("--test-ann"), p.add_argument("--test-images")
    p.add_argument("--size", default="base", help="tiny, base or a Hub repo id (e.g. openmmlab-community/mm_grounding_dino_large_all)")
    p.add_argument("--shots", type=int, default=8, help="instances per class")
    p.add_argument("--tokens-per-class", type=int, default=4)
    p.add_argument("--init-std", type=float, default=0.5)
    p.add_argument("--init", default="random", choices=["random", "text"], help="text: start from the class names' BERT features")
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--micro-batch", type=int, default=1, help="images per forward; gradients accumulate to --batch-size")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--image-size", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-limit", type=int, default=500)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--top-k", type=int, default=300)
    p.add_argument("--query-template", default="{name}")
    p.add_argument("--zero-shot", action="store_true", help="also evaluate the text-prompt baseline")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", help="save the learned embeddings here")
    args = p.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    if args.dataset == "dior":
        train_records, names = load_dior("train", None)
        test_records, _ = load_dior("test", args.eval_limit)
    else:
        train_records, names = load_coco(args.train_ann, args.train_images, None)
        test_records, _ = load_coco(args.test_ann, args.test_images, args.eval_limit)
    shots = sample_shots(train_records, len(names), args.shots, args.seed)
    print(f"{len(names)} classes, {len(shots)} shot images, {sum(len(r.labels) for r in shots)} instances, "
          f"{len(test_records)} eval images")

    model = GroundingDINO(args.size).eval().to(device)
    for prm in model.parameters():
        prm.requires_grad_(False)
    if args.zero_shot:
        evaluate(model, test_records, names, args, device, title=f"{args.dataset} zero-shot")
    adapter = FewShotGDINO(model, names, args.tokens_per_class, args.init_std, args.image_size, args.seed, args.init)
    for r in shots:
        adapter.add_example(r.loader(), r.boxes, r.labels)
    embeds = adapter.embeds
    t0 = time.time()
    adapter.fit(args.iters, args.batch_size, args.micro_batch, args.lr, args.weight_decay)
    print(f"trained {embeds.embeddings.numel()} params in {time.time() - t0:.0f}s")
    evaluate(model, test_records, names, args, device, embeds, title=f"{args.dataset} {args.shots}-shot")
    if args.out:
        adapter.save(args.out)


if __name__ == "__main__":
    main()
