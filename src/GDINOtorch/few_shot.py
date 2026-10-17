"""Few-shot adaptation of Grounding DINO by learning text embeddings.

Use :class:`FewShotGDINO` for the add-examples / ``fit`` / ``detect`` workflow:

    adapter = FewShotGDINO(model, ["chimney", "dam"])
    adapter.add_example("a.jpg", boxes=[[x1, y1, x2, y2], ...], labels=[0, 1, ...])
    adapter.fit()
    boxes, scores, labels = adapter.detect("test.jpg")

Reimplements Singh et al., "Few-Shot Adaptation of Grounding DINO for
Agricultural Domain" (arXiv:2504.07252). BERT is dropped and replaced by a
trainable ``(C*T + 2, 768)`` tensor in BERT's output space: ``T`` tokens per
class plus dummy start/end tokens. Everything else (Swin, fusion encoder, query
selection, decoder, and the frozen text projection) stays untouched; only the
embeddings receive gradients.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision.ops import batched_nms, generalized_box_iou, sigmoid_focal_loss
from torchvision.ops.boxes import box_convert

from .torch_version.gdino import GroundingDINO, TextFeatures, phrase_masks_and_positions
from .utils.tokenizer import build_caption


class LearnedTextEmbeddings(nn.Module):
    """Class prompts as free parameters standing in for BERT's output."""

    def __init__(self, model: GroundingDINO, num_classes: int, tokens_per_class: int = 4, init_std: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.tokens_per_class = tokens_per_class
        hidden = model.text_backbone.hidden_size
        self.embeddings = nn.Parameter(torch.randn(num_classes * tokens_per_class + 2, hidden) * init_std)

        # Mimic BERT's layout "[CLS] class0 tokens ... classC tokens [SEP]" with
        # the phrase-wise masking of Grounding DINO: tokens attend only to their
        # own class, positions restart at 0 for each class, dummies stand alone.
        length = self.embeddings.shape[0]
        block = torch.zeros(length, dtype=torch.long)
        block[1:-1] = torch.arange(num_classes).repeat_interleave(tokens_per_class) + 1
        block[-1] = num_classes + 1
        position = torch.zeros(length, dtype=torch.long)
        position[1:-1] = torch.arange(tokens_per_class).repeat(num_classes)
        self.register_buffer("self_attn_mask", block[:, None] == block[None, :], persistent=False)
        self.register_buffer("position_ids", position, persistent=False)

    @property
    def length(self) -> int:
        return self.embeddings.shape[0]

    def text_features(self, projection: nn.Module, batch_size: int = 1) -> TextFeatures:
        """Project through the frozen ``text_projection`` and wrap as ``TextFeatures``."""
        feats = projection(self.embeddings.to(projection.weight.dtype))[None]
        mask = torch.ones(1, self.length, dtype=torch.bool, device=feats.device)
        return TextFeatures(
            features=feats.expand(batch_size, -1, -1),
            token_mask=mask.expand(batch_size, -1),
            self_attn_mask=self.self_attn_mask[None].expand(batch_size, -1, -1),
            position_ids=self.position_ids[None].expand(batch_size, -1),
        )

    def class_logits(self, token_logits: torch.Tensor) -> torch.Tensor:
        """``[B, Q, L]`` token logits -> ``[B, Q, C]``: best of each class's tokens."""
        per_class = token_logits[..., 1:-1].unflatten(-1, (self.num_classes, self.tokens_per_class))
        return per_class.amax(-1)


@torch.no_grad()
def text_init_embeddings(model: GroundingDINO, class_names, tokens_per_class: int = 4) -> torch.Tensor:
    """BERT features of the class names, laid out like :class:`LearnedTextEmbeddings`.

    Each class's wordpieces fill its ``tokens_per_class`` slots (cycled, or cut if
    longer), so before any training the adapter scores close to the zero-shot
    text prompt and training starts from there instead of from noise.
    """
    caption, spans = build_caption([n.lower() for n in class_names])
    device = next(model.parameters()).device
    input_ids, _ = model.tokenize(caption)
    input_ids = input_ids[:, : model.max_text_len].to(device)
    mask, position = phrase_masks_and_positions(input_ids)
    feats = model.text_backbone(input_ids, mask, position)[0]
    indices = model.tokenizer.phrase_token_indices(caption, spans)
    rows = [feats[0]]
    for idx in indices:
        rows += [feats[idx[t % len(idx)]] for t in range(tokens_per_class)]
    rows.append(feats[input_ids.shape[1] - 1])
    return torch.stack(rows)


@torch.no_grad()
def hungarian_match(class_logits, pred_boxes, targets, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0,
                    alpha=0.25, gamma=2.0):
    """DETR-style one-to-one matching; returns per-image ``(query_idx, target_idx)``."""
    out = []
    for logits, boxes, tgt in zip(class_logits.float(), pred_boxes.float(), targets):
        if len(tgt["labels"]) == 0:
            out.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
            continue
        prob = logits.sigmoid()
        neg = (1 - alpha) * prob**gamma * -(1 - prob + 1e-8).log()
        pos = alpha * (1 - prob) ** gamma * -(prob + 1e-8).log()
        c_cls = (pos - neg)[:, tgt["labels"]]
        c_l1 = torch.cdist(boxes, tgt["boxes"], p=1)
        c_giou = -generalized_box_iou(box_convert(boxes, "cxcywh", "xyxy"), box_convert(tgt["boxes"], "cxcywh", "xyxy"))
        cost = cost_class * c_cls + cost_bbox * c_l1 + cost_giou * c_giou
        q, t = linear_sum_assignment(cost.cpu().numpy())
        out.append((torch.as_tensor(q, dtype=torch.long), torch.as_tensor(t, dtype=torch.long)))
    return out


def detection_loss(embeds: LearnedTextEmbeddings, token_logits, pred_boxes, targets,
                   w_cls=1.0, w_l1=5.0, w_giou=2.0):
    """Focal + L1 + GIoU over Hungarian-matched pairs.

    The classification target is 1 on *every* token of the matched object's
    class (the paper's ``P_gt``), 0 elsewhere; the dummy start/end tokens are
    left out of the loss.
    """
    token_logits = token_logits.float()
    pred_boxes = pred_boxes.float()
    logits = token_logits[..., 1:-1]
    class_logits = embeds.class_logits(token_logits)
    matches = hungarian_match(class_logits, pred_boxes, targets)

    gt = torch.zeros_like(logits)
    box_pred, box_gt = [], []
    for b, ((q, t), tgt) in enumerate(zip(matches, targets)):
        if len(q) == 0:
            continue
        q, t = q.to(logits.device), t.to(logits.device)
        gt[b, q] = F.one_hot(tgt["labels"][t], embeds.num_classes).repeat_interleave(embeds.tokens_per_class, -1).float()
        box_pred.append(pred_boxes[b, q])
        box_gt.append(tgt["boxes"][t])
    num_boxes = max(sum(len(t["labels"]) for t in targets), 1)
    loss_cls = sigmoid_focal_loss(logits, gt, alpha=0.25, gamma=2.0, reduction="sum") / num_boxes
    if box_pred:
        bp, bg = torch.cat(box_pred), torch.cat(box_gt)
        loss_l1 = F.l1_loss(bp, bg, reduction="sum") / num_boxes
        giou = torch.diag(generalized_box_iou(box_convert(bp, "cxcywh", "xyxy"), box_convert(bg, "cxcywh", "xyxy")))
        loss_giou = (1 - giou).sum() / num_boxes
    else:
        loss_l1 = loss_giou = logits.sum() * 0
    total = w_cls * loss_cls + w_l1 * loss_l1 + w_giou * loss_giou
    return total, {"cls": loss_cls.item(), "l1": loss_l1.item(), "giou": loss_giou.item()}


def augment(image, boxes, labels, rng):
    """Random horizontal flip and random crop (box centres must stay inside)."""
    w, h = image.size
    if rng.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        boxes = boxes.copy()
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
    if rng.random() < 0.5 and len(boxes):
        scale = rng.uniform(0.6, 1.0)
        cw, ch = int(w * scale), int(h * scale)
        x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
        centres = (boxes[:, :2] + boxes[:, 2:]) / 2
        keep = ((centres > [x0, y0]) & (centres < [x0 + cw, y0 + ch])).all(1)
        if keep.any():
            image = image.crop((x0, y0, x0 + cw, y0 + ch))
            boxes = boxes[keep] - [x0, y0, x0, y0]
            boxes = boxes.clip(0, [cw, ch, cw, ch])
            labels = labels[keep]
    return image, boxes, labels


def to_target(image, boxes, labels, device):
    """Pixel xyxy boxes -> the normalized cxcywh targets :func:`detection_loss` expects."""
    w, h = image.size
    b = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    cxcywh = torch.stack([(b[:, 0] + b[:, 2]) / 2 / w, (b[:, 1] + b[:, 3]) / 2 / h,
                          (b[:, 2] - b[:, 0]) / w, (b[:, 3] - b[:, 1]) / h], 1)
    return {"boxes": cxcywh.to(device), "labels": torch.as_tensor(labels, dtype=torch.long, device=device)}


def _load(image):
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    return image.convert("RGB")


class FewShotGDINO:
    """Few-shot detector: Grounding DINO / MM-Grounding-DINO with learned class prompts.

    ``class_names`` only name the classes (and size the embedding table); the
    model is never shown their text. Boxes given to :meth:`add_example` must
    cover every instance of every class in that image, since unboxed objects
    are trained as background. Works with any checkpoint ``GroundingDINO``
    loads (tiny/base/large, MM-Grounding-DINO included).
    """

    def __init__(self, model: GroundingDINO, class_names, tokens_per_class: int = 4, init_std: float = 0.5,
                 image_size: int = 800, seed: int = 0, init: str = "random"):
        self.model = model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.class_names = list(class_names)
        self.image_size, self.seed = image_size, seed
        device = next(model.parameters()).device
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(seed)
            self.embeds = LearnedTextEmbeddings(model, len(self.class_names), tokens_per_class, init_std).to(device)
        if init == "text":
            self.embeds.embeddings.data.copy_(text_init_embeddings(model, self.class_names, tokens_per_class))
        elif init != "random":
            raise ValueError("init must be 'random' or 'text'")
        self.examples: list[tuple] = []  # (image or path, boxes xyxy px, labels)

    def add_example(self, image, boxes, labels=None):
        """Add one image with every instance boxed (``xyxy`` pixels); ``labels`` index ``class_names``.

        ``boxes=[]`` adds an image that contains none of the classes. ``labels``
        may be omitted when there is a single class.
        """
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if labels is None:
            if len(self.class_names) != 1 and len(boxes):
                raise ValueError("labels are required with more than one class")
            labels = np.zeros(len(boxes), dtype=np.int64)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if len(labels) != len(boxes) or (len(labels) and not 0 <= labels.min() <= labels.max() < len(self.class_names)):
            raise ValueError("labels must give one class index per box")
        self.examples.append((image if isinstance(image, (str, Path)) else _load(image), boxes, labels))
        return self

    def fit(self, iters: int = 400, batch_size: int = 4, micro_batch: int = 1, lr: float = 1e-2,
            weight_decay: float = 0.0, progress: bool = True, checkpoint: bool | None = None):
        """Train the class embeddings on the added examples; returns the final loss.

        ``checkpoint`` recomputes encoder layers in backward to save memory. The
        default turns it on for 5-level (large) models, whose stride-4 map makes
        the encoder's activations ~4x bigger (9.8 GB vs ~3 GB at 800x1200).
        """
        if not self.examples:
            raise ValueError("add_example() before fit()")
        device = self.embeds.embeddings.device
        if checkpoint is None:
            checkpoint = self.model.level_embed.shape[0] >= 5
        self.model.encoder.grad_checkpoint = checkpoint
        rng = random.Random(self.seed)
        opt = torch.optim.AdamW([self.embeds.embeddings], lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        projection = self.model.text_projection
        bar = range(iters)
        if progress:
            from tqdm import tqdm
            bar = tqdm(bar, desc="fit")
        for step in bar:
            opt.zero_grad(set_to_none=True)
            for _ in range(0, batch_size, micro_batch):
                images, targets = [], []
                for src, boxes, labels in (rng.choice(self.examples) for _ in range(micro_batch)):
                    img, boxes, labels = augment(_load(src), boxes, labels, rng)
                    images.append(img)
                    targets.append(to_target(img, boxes, labels, device))
                pixel_values, pixel_mask = self.model.preprocess_image(images, size=self.image_size)
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    text = self.embeds.text_features(projection, len(images))
                    logits, pred = self.model.forward_object_detection_from_text(pixel_values, text, pixel_mask)
                loss, parts = detection_loss(self.embeds, logits, pred, targets)
                (loss * micro_batch / batch_size).backward()
            opt.step()
            sched.step()
            if progress and step % 10 == 0:
                bar.set_postfix(loss=f"{loss.item():.3f}", **{k: f"{v:.2f}" for k, v in parts.items()})
        self.model.encoder.grad_checkpoint = False
        return loss.item()

    @torch.no_grad()
    def scores(self, pixel_values, pixel_mask=None):
        """``(class scores [B, 900, C], boxes cxcywh [B, 900, 4])`` for a preprocessed batch."""
        device = self.embeds.embeddings.device
        text = self.embeds.text_features(self.model.text_projection)
        with torch.autocast(device.type, dtype=torch.float16):
            logits, boxes = self.model.forward_object_detection_from_text(pixel_values, text, pixel_mask)
        return self.embeds.class_logits(logits.float()).sigmoid(), boxes.float()

    @torch.no_grad()
    def detect(self, images, threshold: float = 0.3, top_k: int = 100, nms: float | None = None, batch_size: int = 8):
        """Return ``(boxes xyxy px, scores, labels)`` for one image, or a list of them for a list.

        Every query keeps its best class; ``nms`` (IoU, per class) is off by
        default because the matching loss already trains the queries to be
        one-to-one.
        """
        single = not isinstance(images, (list, tuple))
        images = [_load(i) for i in ([images] if single else images)]
        out = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            pixel_values, pixel_mask = self.model.preprocess_image(chunk, size=self.image_size)
            scores, boxes = self.scores(pixel_values, pixel_mask)
            xyxy = GroundingDINO.postprocess_boxes(boxes, [im.size[::-1] for im in chunk])
            for s, b in zip(scores, xyxy):
                s, labels = s.max(-1)
                keep = (s >= threshold).nonzero()[:, 0]
                if nms:
                    keep = keep[batched_nms(b[keep], s[keep], labels[keep], nms)]
                keep = keep[s[keep].argsort(descending=True)[:top_k]]
                out.append((b[keep].cpu(), s[keep].cpu(), labels[keep].cpu()))
        return out[0] if single else out

    def save(self, path):
        """Save the learned embeddings (not the examples, nor the model)."""
        torch.save({"embeddings": self.embeds.embeddings.detach().cpu(), "class_names": self.class_names,
                    "tokens_per_class": self.embeds.tokens_per_class, "image_size": self.image_size}, path)

    @classmethod
    def load(cls, path, model: GroundingDINO, **kwargs):
        state = torch.load(path, weights_only=True)
        adapter = cls(model, state["class_names"], state["tokens_per_class"],
                      image_size=state.get("image_size", 800), **kwargs)
        adapter.embeds.embeddings.data.copy_(state["embeddings"])
        return adapter
