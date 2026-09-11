# GDINO-Torch

A standalone PyTorch implementation of [Grounding DINO](https://arxiv.org/abs/2303.05499)
for open-vocabulary object detection, the counterpart of `owlv2-torch`.

It loads the `IDEA-Research/grounding-dino-{tiny,base}` safetensors, and the
`openmmlab-community/mm_grounding_dino_*` ones (MM-Grounding-DINO), straight
from the Hub and runs without `transformers`. The whole model — Swin backbone,
BERT text tower, fusion encoder, query selection, decoder — is plain
`nn.Module`s in one file, `src/GDINOtorch/torch_version/gdino.py`. Tokenization
uses the checkpoint's own `tokenizer.json` through the standalone `tokenizers`
library.

## Install

```bash
uv sync
```

Requires Python 3.11+. Weights and the tokenizer are downloaded on first use
and read from the Hugging Face cache after that.

## Quick start

```python
import torch
from PIL import Image
from GDINOtorch import GroundingDINO, build_caption

model = GroundingDINO("tiny").eval().cuda()  # or "base"

image = Image.open("photo.jpg")
pixel_values, pixel_mask = model.preprocess_image(image)  # on the model's device

phrases = ["a cat", "a remote control"]
caption, spans = build_caption(phrases)          # "a cat. a remote control."
input_ids, attention_mask = model.tokenize(caption)

with torch.inference_mode():
    pred_logits, pred_boxes = model(
        pixel_values.cuda(), input_ids.cuda(), attention_mask.cuda(), pixel_mask.cuda()
    )

# Logits are per caption *token*; max-pool them back to one score per phrase.
token_indices = model.tokenizer.phrase_token_indices(caption, spans)
scores, labels = GroundingDINO.phrase_logits(pred_logits, token_indices).sigmoid().max(-1)
boxes = GroundingDINO.postprocess_boxes(pred_boxes, [image.size[::-1]])  # (x0, y0, x1, y1) px
```

`pred_logits` is `[B, 900, L]`, one logit per query per caption token (`-inf`
on padding). `pred_boxes` is `[B, 900, 4]`, `(cx, cy, w, h)` normalized to
each image's *unpadded* area, so `postprocess_boxes` only needs the original
image size, even for images padded in a mixed-size batch.

### Reusing the text encoding

BERT only sees the caption, so for a fixed vocabulary encode it once and reuse
it for every image batch. A single encoded caption is broadcast across the
image batch:

```python
with torch.inference_mode():
    text = model.encode_text(input_ids.cuda(), attention_mask.cuda())
    for pixel_values, pixel_mask in batches:
        pred_logits, pred_boxes = model.forward_object_detection_from_text(
            pixel_values, text, pixel_mask
        )
```

### Precision

`torch.autocast("cuda", dtype=torch.float16)` and `model.half()` both work, and
match fp32 closely (top detections within ~0.002 in score). On an RTX 3060, tiny
at 800x1066 runs in about 165 ms/image under autocast, on par with the
`transformers` implementation.

### TensorRT

`GroundingDINOTRT` is a drop-in `GroundingDINO` that runs everything after the
text tower (Swin, fusion encoder, query selection, decoder) as one TensorRT
engine. BERT stays in PyTorch, so `encode_text` and caption reuse work as
above. Install the extra first (`uv sync --extra trt`, TensorRT 10.x).

```python
from GDINOtorch import GroundingDINOTRT

# Exports <output_dir>/gdino_tiny_800x1333.onnx and builds
# gdino_tiny_800x1333_fp16.engine on first use (~2-4 min), then reuses them.
model = GroundingDINOTRT("tiny", output_dir="trt_engines", image_size=(800, 1333)).eval().cuda()
pred_logits, pred_boxes = model(pixel_values.cuda(), input_ids.cuda(), attention_mask.cuda(), pixel_mask.cuda())
```

The engine's image size is fixed; its batch size (up to `max_batch`, larger
batches are split) and caption length (up to 256 tokens) are not. Smaller
images are zero-padded into it; larger images, CPU tensors and longer captions
fall back to PyTorch. Padding shifts scores a little (~0.02 on a test image
padded from 800x1066 to 800x1333, more than the engine's own error), so
set `image_size` to the size your images resize to, e.g. `(750, 1333)` for
16:9. Deformable attention runs as TensorRT's fused
`MultiscaleDeformableAttnPlugin_TRT`.

On an RTX 3060, batch 1 at 800x1333, compared with fp16 autocast:

| model | autocast | TensorRT fp16 |
|---|---|---|
| tiny | 197 ms | 61 ms |
| base | 286 ms | 83 ms |
| `mm_grounding_dino_large_all` | ~920 ms | 233 ms |

Top detections stay within ~0.01 in score of fp32 for tiny and base; on the one
image checked, large's top detection moved by 0.02 (autocast: 0.002), so check
large on your own eval before relying on it. `fp16=False` builds an fp32
engine (TF32 on by default; `tf32=False` matches PyTorch fp32 within ~0.002).
The standalone steps are `python -m GDINOtorch.torch_version.export` and
`python -m GDINOtorch.torch_version.build_engine`.

## Model variants

`GroundingDINO(model_type)` takes `"tiny"`, `"base"`, or any Hub repo id of a
Grounding DINO / MM-Grounding-DINO checkpoint in `transformers` format; the
architecture is read from the repo's `config.json`.

| `model_type` | Backbone | Levels | Decoder heads |
|---|---|---|---|
| `"tiny"` (`IDEA-Research/grounding-dino-tiny`) | Swin-T | 4 | shared |
| `"base"` (`IDEA-Research/grounding-dino-base`) | Swin-B | 4 | shared |
| `openmmlab-community/mm_grounding_dino_tiny_*` | Swin-T | 4 | per layer |
| `openmmlab-community/mm_grounding_dino_base_*` | Swin-B | 4 | per layer |
| `openmmlab-community/mm_grounding_dino_large_*` | Swin-L | 5 | per layer |

MM-Grounding-DINO keeps the architecture and changes the heads: each decoder
layer has its own box head, and query-token logits are scaled by 1/sqrt(256)
plus a learned per-layer bias. Its repos ship `vocab.txt` rather than
`tokenizer.json`; the tokenizer is built from it and gives the same ids.

All use `bert-base-uncased`, 900 queries, a 256-token caption limit, and
resize images to a short side of 800 (long side at most 1333).

## Differences from `transformers`

Checked against `transformers` 5.16.1 on the same inputs, for IDEA tiny/base
and MM-Grounding-DINO tiny/base/large: preprocessing and token ids match
exactly, encoder outputs agree to ~4e-6 relative error, both select the same
900 proposals, and the top-100 detections agree to within 1e-3 in score.
Top-900 selection can reorder near-tied proposals, and since each query slot
has its own learned content embedding, reordered slots can differ. That
comparison needed three `transformers` 5.16.1 bugs patched first, and this
implementation has none of them:

- **Text position embeddings are truncated to integers** (every checkpoint).
  `encode_sinusoidal_position_embedding` casts its output back to the input
  dtype, and the encoder feeds it integer `position_ids`, so almost every
  sin/cos value becomes 0. Here they stay float, as in the original model.
- **`grounding-dino-base` loads with its decoder box head at init.** That
  checkpoint stores the shared head as `model.decoder.bbox_embed.0`, which
  `transformers` does not map to the head it uses. The head's last layer stays
  all zeros, so boxes are never refined past the encoder proposals. `tiny`
  stores it as `bbox_embed.0` and loads correctly.
- **MM-Grounding-DINO's per-layer heads all get layer 0's weights.** The
  checkpoints store head 0 as `bbox_embed.0` / `class_embed.0` and heads 1-5
  under `model.decoder.`; `transformers` ties all six to head 0, both the box
  heads and the class-logit biases.

Other deliberate differences:

- Logits are not padded out to 256 tokens; they stop at the caption length.
- Dropout and drop-path are omitted; this is an inference implementation.
- Padding a batch changes the image features: zero-padding reaches Swin's
  windows and the GroupNorm statistics. The same holds in `transformers`, so
  an image scores slightly differently alone than beside a larger one.

## Layout

```
src/GDINOtorch/
  torch_version/
    gdino.py        # the whole model, preprocessing and post-processing
    gdino_tensorrt.py  # GroundingDINOTRT: the image path as a TensorRT engine
    export.py       # image path -> ONNX (fused deformable-attention plugin)
    build_engine.py # ONNX -> TensorRT engine
  utils/
    tokenizer.py    # BERT tokenization, caption building, phrase->token spans
    hf_hub_utils.py # cache-first weight lookup
```
