"""Triton multi-scale deformable attention, forward for inference and a backward for training.

Same math as ``multi_scale_deformable_attention`` in ``gdino.py`` (bilinear
sampling at ``loc * size - 0.5``, zeros outside the map), but fused: one
program handles ``BLOCK`` (batch, query, head) rows, loops over levels/points,
and the backward scatters value gradients with coalesced channel-wise atomics
instead of ``grid_sampler_2d_backward``'s per-element ones.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK = 16


@triton.jit
def _corner(value, b, S, st, Wl, Hl, h, H, xi, yi, d, m, D: tl.constexpr):
    ok = m & (xi >= 0) & (xi < Wl) & (yi >= 0) & (yi < Hl)
    row = (b.to(tl.int64) * S + st + yi * Wl + xi) * H + h
    off = row[:, None] * D + d[None, :]
    v = tl.load(value + off, mask=ok[:, None], other=0.0).to(tl.float32)
    return v, off, ok


@triton.jit
def _fwd_kernel(value, shapes, starts, loc, wt, out, N, QH, S, H,
                L: tl.constexpr, P: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    b, h, d = n // QH, n % H, tl.arange(0, D)
    acc = tl.zeros([BLOCK, D], tl.float32)
    for l in tl.static_range(L):
        Hl, Wl, st = tl.load(shapes + 2 * l), tl.load(shapes + 2 * l + 1), tl.load(starts + l)
        for p in tl.static_range(P):
            k = (n * L + l) * P + p
            x = tl.load(loc + 2 * k, mask=m, other=0.0).to(tl.float32) * Wl - 0.5
            y = tl.load(loc + 2 * k + 1, mask=m, other=0.0).to(tl.float32) * Hl - 0.5
            w = tl.load(wt + k, mask=m, other=0.0).to(tl.float32)
            x0f, y0f = tl.floor(x), tl.floor(y)
            fx, fy = x - x0f, y - y0f
            x0, y0 = x0f.to(tl.int32), y0f.to(tl.int32)
            v00, _, _ = _corner(value, b, S, st, Wl, Hl, h, H, x0, y0, d, m, D)
            v10, _, _ = _corner(value, b, S, st, Wl, Hl, h, H, x0 + 1, y0, d, m, D)
            v01, _, _ = _corner(value, b, S, st, Wl, Hl, h, H, x0, y0 + 1, d, m, D)
            v11, _, _ = _corner(value, b, S, st, Wl, Hl, h, H, x0 + 1, y0 + 1, d, m, D)
            s = ((1 - fx) * (1 - fy))[:, None] * v00 + (fx * (1 - fy))[:, None] * v10 \
                + ((1 - fx) * fy)[:, None] * v01 + (fx * fy)[:, None] * v11
            acc += w[:, None] * s
    tl.store(out + n[:, None].to(tl.int64) * D + d[None, :], acc.to(out.dtype.element_ty), mask=m[:, None])


@triton.jit
def _bwd_kernel(value, shapes, starts, loc, wt, gout, gvalue, gloc, gwt, N, QH, S, H,
                L: tl.constexpr, P: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    b, h, d = n // QH, n % H, tl.arange(0, D)
    go = tl.load(gout + n[:, None].to(tl.int64) * D + d[None, :], mask=m[:, None], other=0.0).to(tl.float32)
    for l in tl.static_range(L):
        Hl, Wl, st = tl.load(shapes + 2 * l), tl.load(shapes + 2 * l + 1), tl.load(starts + l)
        for p in tl.static_range(P):
            k = (n * L + l) * P + p
            x = tl.load(loc + 2 * k, mask=m, other=0.0).to(tl.float32) * Wl - 0.5
            y = tl.load(loc + 2 * k + 1, mask=m, other=0.0).to(tl.float32) * Hl - 0.5
            w = tl.load(wt + k, mask=m, other=0.0).to(tl.float32)
            x0f, y0f = tl.floor(x), tl.floor(y)
            fx, fy = x - x0f, y - y0f
            x0, y0 = x0f.to(tl.int32), y0f.to(tl.int32)
            v00, o00, k00 = _corner(value, b, S, st, Wl, Hl, h, H, x0, y0, d, m, D)
            v10, o10, k10 = _corner(value, b, S, st, Wl, Hl, h, H, x0 + 1, y0, d, m, D)
            v01, o01, k01 = _corner(value, b, S, st, Wl, Hl, h, H, x0, y0 + 1, d, m, D)
            v11, o11, k11 = _corner(value, b, S, st, Wl, Hl, h, H, x0 + 1, y0 + 1, d, m, D)
            c00, c10 = (1 - fx) * (1 - fy), fx * (1 - fy)
            c01, c11 = (1 - fx) * fy, fx * fy
            gw = (w[:, None] * go)
            tl.atomic_add(gvalue + o00, gw * c00[:, None], mask=k00[:, None])
            tl.atomic_add(gvalue + o10, gw * c10[:, None], mask=k10[:, None])
            tl.atomic_add(gvalue + o01, gw * c01[:, None], mask=k01[:, None])
            tl.atomic_add(gvalue + o11, gw * c11[:, None], mask=k11[:, None])
            s = c00[:, None] * v00 + c10[:, None] * v10 + c01[:, None] * v01 + c11[:, None] * v11
            ds_dx = (1 - fy)[:, None] * (v10 - v00) + fy[:, None] * (v11 - v01)
            ds_dy = (1 - fx)[:, None] * (v01 - v00) + fx[:, None] * (v11 - v10)
            tl.store(gwt + k, tl.sum(go * s, 1).to(gwt.dtype.element_ty), mask=m)
            tl.store(gloc + 2 * k, (tl.sum(go * ds_dx, 1) * w * Wl).to(gloc.dtype.element_ty), mask=m)
            tl.store(gloc + 2 * k + 1, (tl.sum(go * ds_dy, 1) * w * Hl).to(gloc.dtype.element_ty), mask=m)


_shape_cache: dict = {}


def _level_tensors(spatial_shapes_list, device):
    """Level sizes ``[L * 2]`` and start offsets ``[L]`` as int32 on ``device``.

    Cached in eager mode. Under ``torch.compile`` they are built in the graph
    instead, as constants: a cached tensor first made inside a CUDA-graph
    memory pool would be left there untracked.
    """
    def build():
        sizes = [h * w for h, w in spatial_shapes_list]
        starts = [sum(sizes[:i]) for i in range(len(sizes))]
        return (
            torch.tensor([list(s) for s in spatial_shapes_list], dtype=torch.int32, device=device).flatten(),
            torch.tensor(starts, dtype=torch.int32, device=device),
        )

    if torch.compiler.is_compiling():
        return build()
    key = (tuple(map(tuple, spatial_shapes_list)), device)
    if key not in _shape_cache:
        _shape_cache[key] = build()
    return _shape_cache[key]


# Registered as torch.library custom ops, so torch.compile traces through them
# (an autograd.Function around Triton launches breaks the graph).
@torch.library.custom_op("gdinotorch::msda", mutates_args=())
def _msda(value: torch.Tensor, shapes: torch.Tensor, starts: torch.Tensor, loc: torch.Tensor,
          wt: torch.Tensor) -> torch.Tensor:
    """``[B, Q, H, D]`` in ``value``'s dtype; sampling and the weighted sum run in fp32."""
    B, S, H, D = value.shape
    _, Q, _, L, P, _ = loc.shape
    value, loc, wt = value.contiguous(), loc.contiguous(), wt.contiguous()
    out = torch.empty(B, Q, H, D, device=value.device, dtype=value.dtype)
    N = B * Q * H
    with torch.cuda.device(value.device):  # Triton launches on the current device
        _fwd_kernel[(triton.cdiv(N, BLOCK),)](value, shapes, starts, loc, wt, out, N, Q * H, S, H,
                                              L=L, P=P, D=D, BLOCK=BLOCK)
    return out


@_msda.register_fake
def _(value, shapes, starts, loc, wt):
    B, _, H, D = value.shape
    return value.new_empty(B, loc.shape[1], H, D)


@torch.library.custom_op("gdinotorch::msda_backward", mutates_args=())
def _msda_backward(value: torch.Tensor, shapes: torch.Tensor, starts: torch.Tensor, loc: torch.Tensor,
                   wt: torch.Tensor, gout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, S, H, D = value.shape
    _, Q, _, L, P, _ = loc.shape
    value, loc, wt = value.contiguous(), loc.contiguous(), wt.contiguous()
    gvalue = torch.zeros(value.shape, device=value.device, dtype=torch.float32)
    gloc, gwt = torch.empty_like(loc), torch.empty_like(wt)
    N = B * Q * H
    with torch.cuda.device(value.device):
        _bwd_kernel[(triton.cdiv(N, BLOCK),)](value, shapes, starts, loc, wt, gout.contiguous(), gvalue, gloc,
                                              gwt, N, Q * H, S, H, L=L, P=P, D=D, BLOCK=BLOCK)
    return gvalue.to(value.dtype), gloc, gwt


@_msda_backward.register_fake
def _(value, shapes, starts, loc, wt, gout):
    return torch.empty_like(value), torch.empty_like(loc), torch.empty_like(wt)


def _setup_context(ctx, inputs, output):
    value, shapes, starts, loc, wt = inputs
    ctx.save_for_backward(value, shapes, starts, loc, wt)


def _backward(ctx, gout):
    value, shapes, starts, loc, wt = ctx.saved_tensors
    gvalue, gloc, gwt = _msda_backward(value, shapes, starts, loc, wt, gout)
    return gvalue, None, None, gloc, gwt


_msda.register_autograd(_backward, setup_context=_setup_context)


def multi_scale_deformable_attention_triton(value, spatial_shapes_list, sampling_locations, attention_weights):
    """Drop-in for ``multi_scale_deformable_attention``; differentiable, CUDA only.

    Under autocast the inputs are cast to fp32 (``custom_fwd(cast_inputs=torch.float32)``'s rule).
    """
    if torch.is_autocast_enabled("cuda"):
        value, sampling_locations, attention_weights = (
            value.float(), sampling_locations.float(), attention_weights.float()
        )
    shapes, starts = _level_tensors(spatial_shapes_list, value.device)
    return _msda(value, shapes, starts, sampling_locations, attention_weights).flatten(2)


def multi_scale_deformable_attention_triton_forward(value, spatial_shapes_list, sampling_locations, attention_weights):
    """Inference-only forward, no autograd. ``value`` keeps its dtype (fp16 under
    autocast) and the output takes it; sampling and the weighted sum run in fp32."""
    shapes, starts = _level_tensors(spatial_shapes_list, value.device)
    return _msda(value, shapes, starts, sampling_locations, attention_weights).flatten(2)
