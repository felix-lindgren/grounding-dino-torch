# Inference speed — Grounding DINO tiny / base, COCO

Date: 2026-10-28. The counterpart of owlv2-torch's
`docs/arch-speedup-findings.md` and `docs/inference-speed-profile.md`: an
accuracy harness, a latency benchmark, and the speedups that keep COCO mAP.
RTX 5070 Ti (torch `cuda:0`) unless noted, RTX 3060 (`cuda:1`) for the full
val2017 runs; PyTorch 2.14.0+cu130, TensorRT 11.4, batch 1, fp16 autocast,
`IDEA-Research/grounding-dino-{tiny,base}`. "Before" is `4e83077`.

## Harness

- `tools/ovd_eval.py`: zero-shot box mAP on COCO (mini500 = val2017's first
  500 images, or full val), Fashionpedia or DIOR, through the `evals` repo's
  datasets and faster-coco-eval. One caption of all class names (`{name}`),
  standard metric (maxDets 100), top 300 detections. `--all-post` scores four
  post-processings from the same forward passes: max-pooled or mean-pooled
  token logits, one class per query (argmax) or top-300 (query, class) pairs
  (multi-label, what the original repo's COCO eval does).
  `ms_per_image_model` is decoded PIL image to host detections.
- `tools/bench.py`: median latency on four COCO images (640x426, 586x640,
  640x483, 375x500), PIL image to host detections, plus a per-stage breakdown
  (synchronizing after every stage, so the stages add up to more than the
  end-to-end time).

Both run in the `evals` venv:

```bash
cd evals && PYTHONPATH=../gdino-torch/src .venv/bin/python ../gdino-torch/tools/ovd_eval.py --model tiny --all-post
cd evals && PYTHONPATH=../gdino-torch/src .venv/bin/python ../gdino-torch/tools/bench.py --model base
```

## Summary

The PyTorch path is about **2x faster** at batch 1 with unchanged COCO mAP,
and TensorRT gains a dynamic-size engine that runs every image at its own size,
portrait included.

PIL image to host detections, median of 30 runs, mean over the four bench
images:

| | tiny 5070 Ti | base 5070 Ti | tiny 3060 | base 3060 |
|---|---:|---:|---:|---:|
| PyTorch, before | 70.2 | 90.1 | 195.3 | 271.6 |
| PyTorch, now | **32.1** | **43.4** | **91.3** | **139.1** |
| + `precast_weights()` | 29.7 | 41.0 | 89.3 | 135.6 |
| TensorRT, fixed 800x1333 (before) | 47.2 | 60.3 | 143.0 | — |
| TensorRT, `dynamic_size=True` | **27.4** | **35.6** | — | — |

PyTorch is 2.2x faster for tiny and 2.0-2.1x for base. The fixed engine's mean
mixes engine runs for the two landscape images (tiny 29.0 ms, base 36.9) with
PyTorch fallbacks for the two portrait ones (see [TensorRT](#tensorrt)); the
dynamic engine runs all four.

COCO box mAP, argmax / multi-label, max-pooled token logits:

| | tiny mini500 | base mini500 | tiny val2017 | base val2017 |
|---|---:|---:|---:|---:|
| PyTorch, before | 0.5194 / 0.5371 | 0.5978 / 0.6053 | 0.4702 / 0.4860 | 0.5633 / 0.5690 |
| PyTorch, now | 0.5185 / 0.5361 | 0.5965 / 0.6043 | 0.4701 / 0.4860 | 0.5635 / 0.5691 |
| TensorRT, fixed 800x1333 (before) | 0.5186 / 0.5349 | 0.5976 / 0.6052 | — | — |
| TensorRT, `dynamic_size=True` | 0.5189 / 0.5371 | 0.5966 / 0.6048 | — | — |

The mini500 differences (up to 0.0013) are 500-image noise: on all 5000
val2017 images (on the 3060) the two agree within 0.0002, while the eval loop
went from 207.8 to 99.8 ms/image for tiny and from 287.5 to 150.1 for base.
`precast_weights()` is bit-identical, so its mAP is the "now" row's.

Forward only at 800x1333 on the RTX 3060, the README's TensorRT comparison
size, in ms:

| model | before | now | + `precast_weights()` |
|---|---:|---:|---:|
| tiny | 236.8 | 106.5 | 104.4 |
| base | 324.7 | 162.4 | 159.0 |
| `mm_grounding_dino_large_all` | 1124.9 | 431.7 | 426.6 |

The fixed 800x1333 tiny engine takes 97.6 ms on the same GPU. The README's
earlier 3060 numbers (tiny 197 ms in autocast, 61 ms in TensorRT) do not
reproduce on today's setup, where the "before" code measures 236.8 ms; why was
not investigated. The README now quotes today's measurements.

## Where the time went

Tiny on the 5070 Ti by stage, in ms, synchronizing around each (so a column
sums to more than the end-to-end time: 84.5 ms against 70.2 before):

| stage | before | now | + `precast_weights()` |
|---|---:|---:|---:|
| Swin-T backbone | 9.8 | 8.1 | 7.4 |
| encoder fusion layers (6) | 18.4 | 8.0 | 7.8 |
| encoder text layers (6) | 2.8 | 2.7 | 2.3 |
| encoder deformable layers (6) | 38.2 | 8.8 | 8.7 |
| decoder (6) | 10.7 | 9.9 | 8.7 |
| other (masks, positions, proposals, heads) | 4.6 | 4.5 | 4.5 |
| sum | 84.5 | 42.0 | 39.4 |

Base differs only in its Swin-B backbone: 35.0 ms before, 23.3 now.

Deformable attention was nearly half the time: inference sampled each level
with `grid_sample` and summed in PyTorch, because the fused Triton kernel
(`a7ee3c3`) was only taken when gradients were needed. The fusion layers were
next: their bi-directional attention built the full image-by-token similarity
in fp16 and softmaxed it twice.

## Changes

Tiny on the 5070 Ti, bench mean as each change landed. The bench then pooled
token logits with one masked max of its own instead of `phrase_logits`, whose
per-phrase loop (item 5) is the 2.8 ms between 67.3 here and 70.2 above:

| step | ms |
|---|---:|
| before | 67.3 |
| Triton deformable attention for inference | 39.9 |
| fusion attention through SDPA, unmasked when nothing is padded | 33.2 |
| Swin attention mask contiguous | 32.7 |

1. **Triton deformable attention for inference.** `msda_triton.py` gains a
   forward-only entry, `multi_scale_deformable_attention_triton_forward`
   (no autograd bookkeeping, value kept in its dtype), and
   `multi_scale_deformable_attention` takes it on CUDA whenever no gradient is
   needed. It samples at fp32 locations, where the `grid_sample` path rounded
   them to fp16 (a few hundredths of a pixel), so outputs are not
   bit-identical; val2017 mAP is unchanged. The `grid_sample` path stays for
   CPU and for tracing (export uses TensorRT's plugin).
2. **Fusion attention as two SDPA calls.** `BiMultiHeadAttention` computed
   `q k^T` once and softmaxed it along both axes. It now calls
   `F.scaled_dot_product_attention` twice (vision attends to text, text to
   vision), computing the similarity twice instead of materializing it. The old
   global-max shift and ±50000 clamp go: softmax is shift-invariant and SDPA
   subtracts the row max itself. With a padding mask SDPA uses the
   memory-efficient kernel; without one, flash: 0.28 ms per direction against
   0.66 / 1.44 ms masked. So `forward_object_detection_from_text` checks once
   whether any pixel or token is padding (one host sync per forward) and passes
   no mask when nothing is, the case for a single image and a single caption.
3. **Contiguous Swin attention masks.** The relative-position bias was a
   permuted view. SDPA's fused kernels need a unit-stride mask, so every
   window attention fell back to the unfused math path. Base's backbone went
   from 35.0 to 23.3 ms.
4. **Float masks in the query's dtype.** `MultiheadAttention` casts an
   additive mask to `q`'s dtype (fp16 under autocast) for the same reason.
5. **`phrase_logits` as one gather.** It indexed and reduced each phrase
   separately, 80 small kernels for COCO: 2.81 ms. Phrases are now padded to
   the longest with a `-inf` column and gathered once: 0.11 ms, bit-identical.
6. **`precast_weights()`, opt-in.** Under fp16 autocast every `Linear` and
   `Conv2d` casts its fp32 weight to fp16 on each call, a few hundred small
   kernels per forward. `model.precast_weights()` stores the image path's
   weights in fp16 once: outputs are bit-identical (checked on the four bench
   images, and mini500 mAP is identical), the forward is ~2.4 ms faster for
   tiny and base, and those weights take half the memory. It is opt-in
   because detection must then run under autocast; BERT keeps fp32 weights, so
   `encode_text` still works outside it.

The model also traces with a dynamic image size now (next section), and
`model.num_queries` can be lowered (see [resolution and
queries](#resolution-and-queries)); eager outputs at the defaults are
bit-identical to before both changes.

## TensorRT

Before, an engine had one fixed image size, 800x1333 by default. COCO's
landscape images (4:3) resize to about 800x1066 and were zero-padded into it,
about 15% padding, which shifts scores a little. Portrait images, a quarter of
COCO, resize to about 1066x800, do not fit, and fell back to PyTorch with a
warning.

`GroundingDINOTRT(..., dynamic_size=True)` exports height and width as dynamic
axes and builds one engine for every size from 256x256 to `max_size` (default
1333x1333), with kernels tuned for `image_size`. Each image runs at its own
size, unpadded, so scores match PyTorch's (tiny mini500 0.5189 / 0.5371 against
0.5185 / 0.5361 in PyTorch). Making the graph size-dynamic needed:

- Swin's window padding and `PatchMerging`'s odd-size padding computed with
  integer ops from the traced size (padding by zero when the size is even);
- `arange` instead of `linspace` for the reference points and proposal grids
  (the same values, and `linspace` traces its length as a constant);
- the plugin's `spatial_shapes` and `level_start_index` built in the graph
  from the level sizes.

**TensorRT 11 bug: InstanceNormalization on small maps.** The first dynamic
engine scored 0.245 mAP. Bisecting outputs layer by layer pointed at the input
projections' GroupNorm, which the exporter lowers to InstanceNormalization:
on the stride-32 and stride-64 levels, hundreds of output rows came back as
garbage with dynamic shapes (the fixed-size engine was fine). Export now
writes GroupNorm as plain mean / variance reductions in fp32, which is exact.

Engine-only GPU time for tiny (no preprocessing, no copies):

| engine | input | ms |
|---|---|---:|
| fixed 800x1333 | 800x1333 | 24.8 |
| fixed 800x1066 | 800x1066 | 23.2 |
| dynamic | 800x1066 | 25.2 |
| dynamic | 1066x800 | 25.8 |
| dynamic | 800x1333 | 32.8 |

A dynamic engine costs ~8% against one fixed at the same size. Against the
padded 800x1333 engine it is about even for 4:3 images (end to end, 27.7 against
29.0 ms on the 640x483 bench image) and slower for wider ones (31.6 against
29.0 on 640x426, which resizes to 800x1201). For a single
known input size (a video stream), a fixed engine at that size is still the
fastest option. Per-layer profiling shows no hot spot: time is spread over
~1200 layers, many of them reshapes, transposes and the exported softmax.

Building the dynamic engine at 1333x1333 with `max_batch=2` asked for an
8.6 GB tactic workspace and failed on the 16 GB 5070 Ti; `max_batch=1` builds
in about two minutes.

## Resolution and queries

PyTorch, mini500, argmax / multi-label (`--size` is the short side, the long
side cap scales with it):

| short side | tiny, 900 queries | tiny, 300 | base, 900 | base, 300 |
|---|---:|---:|---:|---:|
| 800 | 0.5185 / 0.5361 | 0.5145 / 0.5334 | 0.5965 / 0.6043 | 0.5937 / 0.6016 |
| 704 | 0.5157 / 0.5333 | 0.5110 / 0.5280 | 0.5898 / 0.5988 | 0.5871 / 0.5955 |
| 640 | 0.5199 / 0.5327 | 0.5142 / 0.5301 | 0.5855 / 0.5922 | 0.5847 / 0.5915 |

- **Fewer queries cost 0.003-0.005 mAP and save nothing measurable** in
  PyTorch, which is launch-bound at batch 1. Not recommended.
- **Smaller images cost base 0.007 (704) and 0.011 (640) mAP; tiny holds up
  better** (argmax unchanged at 640, multi-label −0.0035). But tiny's PyTorch
  path is launch-bound at batch 1 and barely gets faster: 32.1 / 31.9 / 31.7 ms
  at 800 / 704 / 640. Base gets 43.4 / 38.6 / 38.3.
- In TensorRT, which is not launch-bound, smaller images pay off: tiny's
  dynamic engine runs 27.4 / 22.2 / 18.5 ms at 800 / 704 / 640, with mini500
  0.5189 / 0.5371, 0.5162 / 0.5338 and 0.5185 / 0.5326.

## Other findings

- **Mean vs max pooling** of a class's token logits: within 0.0006 mAP
  everywhere; max stays the default.
- **Multi-label selection** (top-300 (query, class) pairs, the original
  repo's COCO protocol) scores +0.018 mAP for tiny and +0.0075 for base over
  one class per query. It is an evaluation choice, not a model change, but
  matters when comparing against published numbers.
- **CUDA graphs (not applied).** Capturing the forward (after replacing its
  two host round trips, the padding check and the `spatial_shapes` upload,
  with precomputed values) is bit-identical and, with `precast_weights()`,
  takes tiny's forward from 27.9 to 24.5 ms and base's from 39.1 to 37.3,
  about TensorRT's speed for tiny without an engine build. A graph is tied to
  one input shape, and COCO val resizes to 515 distinct shapes, so it only
  suits fixed-size inputs; it would need an API for that.

## Recommendations

- PyTorch: nothing to change; the speedups are on by default on CUDA. Add
  `model.precast_weights()` when all detection runs under fp16 autocast.
- TensorRT with mixed image shapes: `dynamic_size=True` (and `max_batch=1` at
  the default 1333x1333 on GPUs of 16 GB or less). For one known size, a fixed
  engine at exactly that size.
- On a latency budget, prefer TensorRT at a smaller size over fewer queries;
  tiny loses at most 0.0035 mAP at a 640 short side, base 0.011.

## Reproduce

From `evals/`; "before" numbers ran the same tools with `PYTHONPATH` on a
checkout of `4e83077`.

```bash
export PYTHONPATH=../gdino-torch/src
T=../gdino-torch/tools
# accuracy (mini500, all four post-processings); --split val for all 5000 images
.venv/bin/python $T/ovd_eval.py --model tiny --all-post
.venv/bin/python $T/ovd_eval.py --model tiny --all-post --precast-weights
.venv/bin/python $T/ovd_eval.py --model tiny --all-post --size 640 --max-size 1066 --num-queries 300

# latency and stage breakdown
.venv/bin/python $T/bench.py --model tiny --iters 30
.venv/bin/python $T/bench.py --model tiny --iters 30 --precast-weights --no-stages

# TensorRT: build the dynamic engine once, then pass it to either tool
.venv/bin/python -c "from GDINOtorch import GroundingDINOTRT; \
    GroundingDINOTRT('tiny', output_dir='trt_engines', dynamic_size=True, image_size=(800, 1066), max_batch=1)"
.venv/bin/python $T/ovd_eval.py --model tiny --all-post --trt-engine trt_engines/gdino_tiny_dyn1333x1333_fp16.engine
.venv/bin/python $T/bench.py --model tiny --iters 30 --trt-engine trt_engines/gdino_tiny_dyn1333x1333_fp16.engine
```
