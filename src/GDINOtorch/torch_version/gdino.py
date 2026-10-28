"""Grounding DINO in plain PyTorch.

Swin backbone + BERT text tower, a six-layer encoder that fuses image and text
(bi-directional cross-attention, text self-attention, multi-scale deformable
attention), two-stage query selection, and a six-layer decoder that refines
900 boxes against both modalities. Weights load directly from the Hub
safetensors of ``IDEA-Research/grounding-dino-{tiny,base}`` and of the
``openmmlab-community/mm_grounding_dino_*`` (MM-Grounding-DINO) conversions.
"""

import json
import math
import re
import warnings
from typing import NamedTuple, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.v2.functional as TF
from PIL import Image
from safetensors import safe_open

from GDINOtorch.utils.hf_hub_utils import resolve_hub_file
from GDINOtorch.utils.tokenizer import Tokenizer

IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

# [CLS], [SEP], "." and "?": the tokens that delimit phrases in a caption.
SPECIAL_TOKENS: list[int] = [101, 102, 1012, 1029]

MODEL_ALIASES = {
    "tiny": "IDEA-Research/grounding-dino-tiny",
    "base": "IDEA-Research/grounding-dino-base",
}


def load_config(model_id: str) -> dict:
    """The architecture fields this implementation reads from a Hub ``config.json``.

    Fields a config omits fall back to the transformers defaults, which is how
    the IDEA checkpoints leave the Swin-T backbone unspecified.
    """
    with open(resolve_hub_file(model_id, "config.json")) as f:
        cfg = json.load(f)
    backbone = cfg.get("backbone_config") or {}
    model_type = cfg.get("model_type", "grounding-dino")
    if model_type not in ("grounding-dino", "mm-grounding-dino"):
        raise ValueError(f"{model_id} is a {model_type!r} checkpoint, not Grounding DINO")
    if backbone.get("model_type", "swin") != "swin":
        raise ValueError(f"{model_id} uses a {backbone['model_type']!r} backbone; only Swin is implemented")
    if not cfg.get("two_stage", True):
        raise ValueError(f"{model_id} is not two-stage; only two-stage query selection is implemented")
    return dict(
        embed_dim=backbone.get("embed_dim", 96),
        depths=tuple(backbone.get("depths", (2, 2, 6, 2))),
        num_heads=tuple(backbone.get("num_heads", (3, 6, 12, 24))),
        window_size=backbone.get("window_size", 7),
        # transformers counts the patch embedding as stage 0.
        out_indices=tuple(i - 1 for i in backbone.get("out_indices", (2, 3, 4))),
        d_model=cfg.get("d_model", 256),
        num_feature_levels=cfg.get("num_feature_levels", 4),
        encoder_layers=cfg.get("encoder_layers", 6),
        decoder_layers=cfg.get("decoder_layers", 6),
        encoder_attention_heads=cfg.get("encoder_attention_heads", 8),
        decoder_attention_heads=cfg.get("decoder_attention_heads", 8),
        encoder_ffn_dim=cfg.get("encoder_ffn_dim", 2048),
        decoder_ffn_dim=cfg.get("decoder_ffn_dim", 2048),
        encoder_n_points=cfg.get("encoder_n_points", 4),
        decoder_n_points=cfg.get("decoder_n_points", 4),
        num_queries=cfg.get("num_queries", 900),
        max_text_len=cfg.get("max_text_len", 256),
        layer_norm_eps=cfg.get("layer_norm_eps", 1e-5),
        positional_embedding_temperature=cfg.get("positional_embedding_temperature", 20),
        decoder_bbox_embed_share=cfg.get("decoder_bbox_embed_share", True),
        # MM-Grounding-DINO scales its text-query logits and adds a learned bias.
        scaled_class_logits=model_type == "mm-grounding-dino",
    )


class TextFeatures(NamedTuple):
    """A caption encoded by BERT, reusable across any number of images."""

    features: torch.Tensor  # [B, L, d_model] projected BERT output
    token_mask: torch.Tensor  # [B, L] bool, True for real (non-pad) tokens
    self_attn_mask: torch.Tensor  # [B, L, L] bool, True where tokens share a phrase
    position_ids: torch.Tensor  # [B, L] position of each token within its phrase


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """ReLU MLP used for box regression and the query-position head."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip(dims, dims[1:] + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class MultiheadAttention(nn.Module):
    """Batch-first multi-head attention over SDPA.

    ``attn_mask`` follows SDPA: bool (True attends) or additive float,
    broadcastable to ``[B, heads, Lq, Lk]``.
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _heads(self, x):
        return x.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, queries, keys, values, attn_mask=None):
        batch_size, query_len, _ = queries.shape
        q = self._heads(self.query(queries))
        k = self._heads(self.key(keys))
        v = self._heads(self.value(values))
        if attn_mask is not None and attn_mask.is_floating_point():
            # SDPA's fused kernels need an additive mask in q's dtype (fp16
            # under autocast); any other falls back to the unfused math path.
            attn_mask = attn_mask.to(q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(batch_size, query_len, -1)
        return self.out_proj(out)


def sine_embedding(pos: torch.Tensor, num_pos_feats: int = 128, temperature: int = 10000) -> torch.Tensor:
    """Sin/cos features for each coordinate of ``pos[..., n]``, concatenated.

    With two or more coordinates the x and y blocks are swapped into DETR's
    ``(y, x, ...)`` order. Always returns float32: the text path feeds integer
    position ids through here, and casting back to the input dtype would
    truncate the embedding to integers.
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
    embeddings = []
    for coord in pos.float().unbind(-1):
        e = coord[..., None] * scale / dim_t
        embeddings.append(torch.stack((e[..., 0::2].sin(), e[..., 1::2].cos()), dim=-1).flatten(-2))
    if len(embeddings) >= 2:
        embeddings[0], embeddings[1] = embeddings[1], embeddings[0]
    return torch.cat(embeddings, dim=-1)


# ---------------------------------------------------------------------------
# Swin backbone
# ---------------------------------------------------------------------------


def window_partition(x, window_size):
    """[B, H, W, C] -> [B * num_windows, window_size**2, C]"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.transpose(2, 3).reshape(-1, window_size * window_size, C)


def window_reverse(windows, window_size, height, width):
    """[B * num_windows, window_size**2, C] -> [B, H, W, C]"""
    C = windows.shape[-1]
    x = windows.view(-1, height // window_size, width // window_size, window_size, window_size, C)
    return x.transpose(2, 3).reshape(-1, height, width, C)


def shifted_window_mask(height, width, window_size, shift_size, device):
    """Additive mask keeping shifted-window attention inside each original region.

    Returns ``[num_windows, window_size**2, window_size**2]`` with -100 between
    tokens that the cyclic shift brought together from different regions.
    """
    h = torch.arange(height, device=device)
    w = torch.arange(width, device=device)
    h_region = (h >= height - window_size).long() + (h >= height - shift_size).long()
    w_region = (w >= width - window_size).long() + (w >= width - shift_size).long()
    regions = (h_region[:, None] * 3 + w_region[None, :]).float()
    windows = window_partition(regions[None, :, :, None], window_size).squeeze(-1)
    diff = windows[:, None, :] - windows[:, :, None]
    return torch.zeros_like(diff).masked_fill(diff != 0, -100.0)


class WindowAttention(MultiheadAttention):
    def __init__(self, dim, num_heads, window_size):
        super().__init__(dim, num_heads)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        coords = torch.stack(
            torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij")
        ).flatten(1)
        relative = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + (window_size - 1)
        index = relative[..., 0] * (2 * window_size - 1) + relative[..., 1]
        self.register_buffer("relative_position_index", index.flatten(), persistent=False)
        self.window_area = window_size * window_size

    def forward(self, windows, shift_mask=None):
        n = self.window_area
        bias = self.relative_position_bias_table[self.relative_position_index]
        # Contiguous: SDPA's fused kernels need a unit-stride mask, and fall back to math otherwise.
        mask = bias.view(n, n, -1).permute(2, 0, 1).contiguous().unsqueeze(0)  # [1, heads, n, n]
        if shift_mask is not None:
            num_windows = shift_mask.shape[0]
            mask = (mask + shift_mask.unsqueeze(1)).repeat(windows.shape[0] // num_windows, 1, 1, 1)
        return super().forward(windows, windows, windows, attn_mask=mask)


class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, 4 * dim)

    def forward(self, x, hw, shift_mask):
        height, width = hw
        B, _, C = x.shape
        ws = self.window_size
        shortcut = x

        x = self.norm1(x).view(B, height, width, C)
        pad_r = (ws - width % ws) % ws
        pad_b = (ws - height % ws) % ws
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        if self.shift_size:
            x = torch.roll(x, (-self.shift_size, -self.shift_size), dims=(1, 2))

        x = self.attn(window_partition(x, ws), shift_mask if self.shift_size else None)
        x = window_reverse(x, ws, height + pad_b, width + pad_r)

        if self.shift_size:
            x = torch.roll(x, (self.shift_size, self.shift_size), dims=(1, 2))
        x = shortcut + x[:, :height, :width].reshape(B, height * width, C)
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    """2x2 neighbourhood concat + linear: halves resolution, doubles channels."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, hw):
        height, width = hw
        B, _, C = x.shape
        x = x.view(B, height, width, C)
        if torch.jit.is_tracing() or height % 2 or width % 2:  # a traced graph pads whatever the size
            x = F.pad(x, (0, 0, 0, width % 2, 0, height % 2))
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)
        return self.reduction(self.norm(x.view(B, -1, 4 * C)))


class SwinStage(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, downsample):
        super().__init__()
        self.window_size = window_size
        self.blocks = nn.ModuleList(
            SwinBlock(dim, num_heads, window_size, 0 if i % 2 == 0 else window_size // 2)
            for i in range(depth)
        )
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x, hw):
        ws = self.window_size
        # Integer ops only, so a traced graph keeps them dynamic in the image size.
        padded_h = (hw[0] + ws - 1) // ws * ws
        padded_w = (hw[1] + ws - 1) // ws * ws
        # Every shifted block in a stage sees the same padded grid.
        shift_mask = shifted_window_mask(padded_h, padded_w, ws, ws // 2, x.device)
        for block in self.blocks:
            x = block(x, hw, shift_mask)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, pixel_values):
        height, width = pixel_values.shape[-2:]
        p = self.patch_size
        pixel_values = F.pad(pixel_values, (0, (p - width % p) % p, 0, (p - height % p) % p))
        x = self.proj(pixel_values)
        hw = tuple(x.shape[-2:])
        return self.norm(x.flatten(2).transpose(1, 2)), hw


class SwinBackbone(nn.Module):
    """Swin transformer returning normalized feature maps from the last three stages."""

    def __init__(self, embed_dim, depths, num_heads, window_size, out_indices=(1, 2, 3)):
        super().__init__()
        self.patch_embed = PatchEmbed(embed_dim)
        self.stages = nn.ModuleList(
            SwinStage(
                embed_dim * 2**i, depths[i], num_heads[i], window_size, downsample=i < len(depths) - 1
            )
            for i in range(len(depths))
        )
        self.out_indices = tuple(out_indices)
        self.channels = [embed_dim * 2**i for i in self.out_indices]
        self.out_norms = nn.ModuleList(nn.LayerNorm(c) for c in self.channels)

    def forward(self, pixel_values):
        x, hw = self.patch_embed(pixel_values)
        features = []
        for i, stage in enumerate(self.stages):
            x = stage(x, hw)
            if i in self.out_indices:
                out = self.out_norms[self.out_indices.index(i)](x)
                features.append(out.view(x.shape[0], *hw, -1).permute(0, 3, 1, 2).contiguous())
            if stage.downsample is not None:
                x = stage.downsample(x, hw)
                hw = ((hw[0] + 1) // 2, (hw[1] + 1) // 2)
        return features


# ---------------------------------------------------------------------------
# BERT text tower
# ---------------------------------------------------------------------------


class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_positions, type_vocab_size):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_positions, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, input_ids, position_ids, token_type_ids):
        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.norm(x)


class BertLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.attn = MultiheadAttention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.mlp = FeedForward(hidden_size, intermediate_size)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, x, attn_mask):
        x = self.norm1(x + self.attn(x, x, x, attn_mask))
        return self.norm2(x + self.mlp(x))


class BertModel(nn.Module):
    """bert-base-uncased encoder, no pooler."""

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=3072,
        max_positions=512,
        type_vocab_size=2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_positions, type_vocab_size)
        self.layers = nn.ModuleList(
            BertLayer(hidden_size, num_heads, intermediate_size) for _ in range(num_layers)
        )

    def forward(self, input_ids, attn_mask, position_ids, token_type_ids=None):
        """``attn_mask``: bool ``[B, L, L]``, True where a token may attend."""
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        x = self.embeddings(input_ids, position_ids, token_type_ids)
        attn_mask = attn_mask[:, None]
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x


def phrase_masks_and_positions(input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-diagonal text self-attention mask and phrase-relative position ids.

    Tokens attend only within their own phrase (the run up to the next
    [CLS]/[SEP]/"."/"?"), and positions restart at 0 after each delimiter, so
    ``"cat. dog."`` encodes "dog" as if it were a caption of its own.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    special = torch.isin(input_ids, torch.tensor(SPECIAL_TOKENS, device=device))
    indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    prev_special = torch.where(special, indices, torch.full((), -1, device=device))
    prev_special = torch.cummax(prev_special, dim=1)[0]
    next_special = torch.where(special, indices, torch.full((), seq_len, device=device))
    next_special = torch.flip(torch.cummin(torch.flip(next_special, dims=[1]), dim=1)[0], dims=[1])

    # A block closed by the leading [CLS] or trailing [SEP] (or by nothing, for
    # padding) is not a phrase.
    valid = (next_special != 0) & (next_special != seq_len - 1) & (next_special != seq_len)
    mask = (next_special.unsqueeze(2) == next_special.unsqueeze(1)) & valid.unsqueeze(1)
    mask = mask | torch.eye(seq_len, device=device, dtype=torch.bool).unsqueeze(0)

    position_ids = torch.where(valid, indices - prev_special - 1, torch.zeros_like(indices))
    return mask, position_ids.clamp(min=0).long()


# ---------------------------------------------------------------------------
# Deformable attention
# ---------------------------------------------------------------------------


try:  # fused Triton kernels: ~10x faster than grid_sample forward, ~3x in backward
    from .msda_triton import multi_scale_deformable_attention_triton as _msda_triton
    from .msda_triton import multi_scale_deformable_attention_triton_forward as _msda_triton_forward
except ImportError:
    _msda_triton = _msda_triton_forward = None


def multi_scale_deformable_attention(value, spatial_shapes_list, sampling_locations, attention_weights):
    """MSDeformAttn: bilinear-sample each level, then weight and sum.

    value: [B, S, heads, head_dim]; sampling_locations: [B, Q, heads, levels,
    points, 2] in [0, 1]; attention_weights: [B, Q, heads, levels, points].
    Returns [B, Q, heads * head_dim].

    On CUDA this is one Triton kernel that samples at fp32 locations; the
    pure-PyTorch path below (CPU) rounds them to ``value``'s dtype, a few
    hundredths of a pixel at fp16.
    """
    if _msda_triton is not None and value.is_cuda and not torch.jit.is_tracing():
        if torch.is_grad_enabled() and (
            value.requires_grad or sampling_locations.requires_grad or attention_weights.requires_grad
        ):
            return _msda_triton(value, spatial_shapes_list, sampling_locations, attention_weights)
        return _msda_triton_forward(value, spatial_shapes_list, sampling_locations, attention_weights)
    batch_size, _, num_heads, head_dim = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([h * w for h, w in spatial_shapes_list], dim=1)
    sampling_grids = (2 * sampling_locations - 1).to(value.dtype)
    sampled = []
    for level, (height, width) in enumerate(spatial_shapes_list):
        # [B, h*w, heads, head_dim] -> [B*heads, head_dim, h, w]
        value_l = value_list[level].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, head_dim, height, width)
        # [B, Q, heads, points, 2] -> [B*heads, Q, points, 2]
        grid_l = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled.append(F.grid_sample(value_l, grid_l, mode="bilinear", padding_mode="zeros", align_corners=False))
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_levels * num_points
    )
    out = (torch.stack(sampled, dim=-2).flatten(-2) * attention_weights).sum(-1)
    return out.view(batch_size, num_heads * head_dim, num_queries).transpose(1, 2)


class MultiscaleDeformableAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_levels, num_points):
        super().__init__()
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(d_model, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(d_model, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, queries, reference_points, value, value_mask, spatial_shapes, spatial_shapes_list):
        """
        queries: [B, Q, d] (position embedding already added)
        reference_points: [B, Q, levels, 2] points or [B, Q, levels, 4] boxes, in [0, 1]
        value: [B, S, d] flattened multi-scale features; value_mask: [B, S] True for valid pixels
        """
        B, Q, _ = queries.shape
        H, L, P = self.num_heads, self.num_levels, self.num_points
        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(~value_mask[..., None], 0.0)
        value = value.view(B, value.shape[1], H, -1)

        offsets = self.sampling_offsets(queries).view(B, Q, H, L, P, 2)
        weights = self.attention_weights(queries).view(B, Q, H, L * P).softmax(-1).view(B, Q, H, L, P)
        if reference_points.shape[-1] == 2:
            normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            locations = reference_points[:, :, None, :, None, :] + offsets / normalizer[None, None, None, :, None, :]
        else:
            # Offsets are scaled by the box size, so points spread with the box.
            locations = (
                reference_points[:, :, None, :, None, :2]
                + offsets / P * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        out = multi_scale_deformable_attention(value, spatial_shapes_list, locations, weights)
        return self.output_proj(out)


# ---------------------------------------------------------------------------
# Encoder: image-text fusion + text self-attention + deformable self-attention
# ---------------------------------------------------------------------------


class BiMultiHeadAttention(nn.Module):
    """Image->text and text->image cross-attention sharing one similarity matrix."""

    def __init__(self, dim, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.scale = self.head_dim**-0.5
        self.vision_proj = nn.Linear(dim, embed_dim)
        self.text_proj = nn.Linear(dim, embed_dim)
        self.values_vision_proj = nn.Linear(dim, embed_dim)
        self.values_text_proj = nn.Linear(dim, embed_dim)
        self.out_vision_proj = nn.Linear(embed_dim, dim)
        self.out_text_proj = nn.Linear(embed_dim, dim)

    def _heads(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x):
        B, _, L, _ = x.shape
        return x.transpose(1, 2).reshape(B, L, self.embed_dim)

    def forward(self, vision, text, vision_pad_mask=None, text_pad_mask=None):
        """Pad masks are True where the vision pixel / text token is padding."""
        q = self._heads(self.vision_proj(vision) * self.scale)
        k = self._heads(self.text_proj(text))
        v_vision = self._heads(self.values_vision_proj(vision))
        v_text = self._heads(self.values_text_proj(text))

        # Both directions share the similarity q k^T [B, heads, N_img, N_txt]:
        # softmax over text for the vision update, over pixels for the text one.
        # Fused attention computes it twice rather than materializing it.
        text_keep = None if text_pad_mask is None else ~text_pad_mask[:, None, None, :]
        vision_keep = None if vision_pad_mask is None else ~vision_pad_mask[:, None, None, :]
        vision_out = F.scaled_dot_product_attention(q, k, v_text, attn_mask=text_keep, scale=1.0)
        text_out = F.scaled_dot_product_attention(k, q, v_vision, attn_mask=vision_keep, scale=1.0)
        return self.out_vision_proj(self._merge(vision_out)), self.out_text_proj(self._merge(text_out))


class FusionLayer(nn.Module):
    def __init__(self, d_model, embed_dim, num_heads, layer_norm_eps):
        super().__init__()
        self.layer_norm_vision = nn.LayerNorm(d_model, layer_norm_eps)
        self.layer_norm_text = nn.LayerNorm(d_model, layer_norm_eps)
        self.attn = BiMultiHeadAttention(d_model, embed_dim, num_heads)
        # Layer scale: the fused update starts near zero.
        self.vision_param = nn.Parameter(1e-4 * torch.ones(d_model))
        self.text_param = nn.Parameter(1e-4 * torch.ones(d_model))

    def forward(self, vision, text, vision_pad_mask, text_pad_mask):
        vision = self.layer_norm_vision(vision)
        text = self.layer_norm_text(text)
        delta_v, delta_t = self.attn(vision, text, vision_pad_mask, text_pad_mask)
        return vision + self.vision_param * delta_v, text + self.text_param * delta_t


class TextEnhancerLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, layer_norm_eps):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads)
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.layer_norm_before = nn.LayerNorm(d_model, layer_norm_eps)
        self.layer_norm_after = nn.LayerNorm(d_model, layer_norm_eps)

    def forward(self, text, attn_mask, position_embeddings):
        q = k = text + position_embeddings
        text = self.layer_norm_before(text + self.self_attn(q, k, text, attn_mask))
        return self.layer_norm_after(text + self.fc2(F.relu(self.fc1(text))))


class DeformableEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, num_levels, num_points, ffn_dim, layer_norm_eps):
        super().__init__()
        self.self_attn = MultiscaleDeformableAttention(d_model, num_heads, num_levels, num_points)
        self.self_attn_layer_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model, layer_norm_eps)

    def forward(self, x, pos, reference_points, valid_mask, spatial_shapes, spatial_shapes_list):
        attn = self.self_attn(x + pos, reference_points, x, valid_mask, spatial_shapes, spatial_shapes_list)
        x = self.self_attn_layer_norm(x + attn)
        return self.final_layer_norm(x + self.fc2(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_dim, num_heads, num_levels, num_points, layer_norm_eps):
        super().__init__()
        self.text_enhancer_layer = TextEnhancerLayer(d_model, num_heads // 2, ffn_dim // 2, layer_norm_eps)
        self.fusion_layer = FusionLayer(d_model, ffn_dim // 2, num_heads // 2, layer_norm_eps)
        self.deformable_layer = DeformableEncoderLayer(
            d_model, num_heads, num_levels, num_points, ffn_dim, layer_norm_eps
        )

    def forward(self, vision, vision_pos, reference_points, vision_valid, spatial_shapes, spatial_shapes_list,
                text, text_pos, text_token_mask, text_self_attn_mask, vision_pad_mask, text_pad_mask):
        vision, text = self.fusion_layer(vision, text, vision_pad_mask, text_pad_mask)
        text = self.text_enhancer_layer(text, text_self_attn_mask[:, None], text_pos)
        vision = self.deformable_layer(
            vision, vision_pos, reference_points, vision_valid, spatial_shapes, spatial_shapes_list
        )
        return vision, text


class Encoder(nn.Module):
    grad_checkpoint = False  # recompute each layer in backward: ~1/6 the activation memory, ~1.3x the time

    def __init__(self, num_layers, **layer_kwargs):
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(**layer_kwargs) for _ in range(num_layers))

    @staticmethod
    def reference_points(spatial_shapes_list, valid_ratios, device):
        """Pixel centres of every level, normalized to the valid (unpadded) area."""
        points = []
        for level, (height, width) in enumerate(spatial_shapes_list):
            # arange, not linspace: the same values, and it traces with a dynamic size.
            ref_y, ref_x = torch.meshgrid(
                torch.arange(height, dtype=torch.float32, device=device) + 0.5,
                torch.arange(width, dtype=torch.float32, device=device) + 0.5,
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * width)
            points.append(torch.stack((ref_x, ref_y), -1))
        points = torch.cat(points, 1)
        return points[:, :, None] * valid_ratios[:, None]

    def forward(self, vision, vision_pos, vision_valid, spatial_shapes, spatial_shapes_list, valid_ratios,
                text, text_token_mask, text_self_attn_mask, text_position_ids, vision_padded=True, text_padded=True):
        """``vision_padded`` / ``text_padded`` False promise that every pixel /
        token is valid, so the fusion layers' attention can run unmasked."""
        reference_points = self.reference_points(spatial_shapes_list, valid_ratios, vision.device)
        text_pos = sine_embedding(text_position_ids[..., None], num_pos_feats=text.shape[-1]).to(text.dtype)
        vision_pad_mask = ~vision_valid if vision_padded else None
        text_pad_mask = ~text_token_mask if text_padded else None
        for layer in self.layers:
            args = (vision, vision_pos, reference_points, vision_valid, spatial_shapes, spatial_shapes_list,
                    text, text_pos, text_token_mask, text_self_attn_mask, vision_pad_mask, text_pad_mask)
            if self.grad_checkpoint and torch.is_grad_enabled():
                vision, text = torch.utils.checkpoint.checkpoint(layer, *args, use_reentrant=False)
            else:
                vision, text = layer(*args)
        return vision, text


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, num_levels, num_points, ffn_dim, layer_norm_eps):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.encoder_attn_text = MultiheadAttention(d_model, num_heads)
        self.encoder_attn_text_layer_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.encoder_attn = MultiscaleDeformableAttention(d_model, num_heads, num_levels, num_points)
        self.encoder_attn_layer_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model, layer_norm_eps)

    def forward(self, x, query_pos, reference_points, vision, vision_valid, spatial_shapes, spatial_shapes_list,
                text, text_attn_mask):
        q = k = x + query_pos
        x = self.self_attn_layer_norm(x + self.self_attn(q, k, x))
        x = self.encoder_attn_text_layer_norm(x + self.encoder_attn_text(x + query_pos, text, text, text_attn_mask))
        attn = self.encoder_attn(
            x + query_pos, reference_points, vision, vision_valid, spatial_shapes, spatial_shapes_list
        )
        x = self.encoder_attn_layer_norm(x + attn)
        return self.final_layer_norm(x + self.fc2(F.relu(self.fc1(x))))


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, **layer_kwargs):
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer(d_model=d_model, **layer_kwargs) for _ in range(num_layers))
        self.layer_norm = nn.LayerNorm(d_model, layer_kwargs["layer_norm_eps"])
        self.reference_points_head = MLP(2 * d_model, d_model, d_model, 2)

    def forward(self, queries, reference_points, vision, vision_valid, spatial_shapes, spatial_shapes_list,
                valid_ratios, text, text_token_mask, bbox_embed):
        """Run all layers, refining the boxes after each with ``bbox_embed[i]``.

        Returns the layer-normed hidden state of every layer and the reference
        boxes each layer was *given* (``references[i]`` is layer ``i``'s input),
        plus the final refinement at the end.
        """
        text_attn_mask = text_token_mask[:, None, None, :]
        hidden_states, references = [], [reference_points]
        x = queries
        for layer, box_head in zip(self.layers, bbox_embed):
            points_input = reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            query_pos = self.reference_points_head(
                sine_embedding(points_input[:, :, 0, :], num_pos_feats=x.shape[-1] // 2).to(x.dtype)
            )
            x = layer(x, query_pos, points_input, vision, vision_valid, spatial_shapes, spatial_shapes_list,
                      text, text_attn_mask)
            # Iterative box refinement predicts from the un-normed hidden state.
            reference_points = (box_head(x) + torch.logit(reference_points, eps=1e-5)).sigmoid().detach()
            hidden_states.append(self.layer_norm(x))
            references.append(reference_points)
        return hidden_states, references


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class SinePositionEmbedding(nn.Module):
    """2D sine embedding over the valid (unpadded) area of a feature map."""

    def __init__(self, embedding_dim=128, temperature=20):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature

    def forward(self, mask):
        """mask: [B, H, W] bool, True for valid pixels. Returns [B, 2*dim, H, W]."""
        scale = 2 * math.pi
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        # Clamp rather than add an epsilon: fully padded rows/columns total 0,
        # and a TensorRT fp16 engine flushes 1e-6 to zero, turning 0/eps into NaN.
        y_embed = y_embed / y_embed[:, -1:, :].clamp(min=1) * scale
        x_embed = x_embed / x_embed[:, :, -1:].clamp(min=1) * scale

        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class ContrastiveHead(nn.Module):
    """Dot product of each query with each text token; -inf on padding tokens.

    Grounding DINO uses the raw dot product. MM-Grounding-DINO (``scaled``)
    divides by sqrt(d) and adds a learned bias, one per head.
    """

    def __init__(self, scaled=False):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(())) if scaled else None

    def forward(self, queries, text, text_token_mask):
        logits = queries @ text.transpose(-1, -2)
        if self.bias is not None:
            logits = logits / math.sqrt(queries.shape[-1]) + self.bias
        return logits.masked_fill(~text_token_mask[:, None, :], float("-inf"))


def resize_shape(height, width, size=800, max_size=1333):
    """Output (h, w) with the short side at ``size`` unless the long side would pass ``max_size``."""
    raw_size = None
    min_side, max_side = float(min(height, width)), float(max(height, width))
    if max_side / min_side * size > max_size:
        raw_size = max_size * min_side / max_side
        size = int(round(raw_size))
    if (height <= width and height == size) or (width <= height and width == size):
        return height, width
    if width < height:
        return int((raw_size or size) * height / width), size
    return size, int((raw_size or size) * width / height)


_KEY_RENAMES = [
    ("backbone.conv_encoder.model.embeddings.patch_embeddings.projection.", "backbone.patch_embed.proj."),
    ("backbone.conv_encoder.model.embeddings.norm.", "backbone.patch_embed.norm."),
    ("backbone.conv_encoder.model.encoder.layers.", "backbone.stages."),
    ("text_backbone.encoder.layer.", "text_backbone.layers."),
    ("embeddings.LayerNorm.", "embeddings.norm."),
    (".attention.self.", ".attn."),
    (".attention.output.dense.", ".attn.out_proj."),
    (".attention.output.LayerNorm.", ".norm1."),
    (".intermediate.dense.", ".mlp.fc1."),
    (".output.dense.", ".mlp.fc2."),
    (".output.LayerNorm.", ".norm2."),
    (".layernorm_before.", ".norm1."),
    (".layernorm_after.", ".norm2."),
    ("input_proj_vision.", "input_proj."),
]


def _remap_key(key: str, out_indices: Sequence[int]) -> Optional[str]:
    """Map an HF checkpoint key onto this module tree (None drops the key)."""
    if key.endswith("relative_position_index"):
        return None  # rebuilt from the window size
    key = key.removeprefix("model.")
    # Per-layer decoder heads are split across two prefixes: head 0 as
    # ``bbox_embed.0`` / ``class_embed.0``, heads 1+ under ``decoder.``. A
    # shared box head is stored once, under either prefix.
    if m := re.fullmatch(r"(?:decoder\.)?(bbox_embed|class_embed)\.(\d+)\.(.+)", key):
        return f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
    if m := re.fullmatch(r"backbone\.conv_encoder\.model\.hidden_states_norms\.stage(\d)\.(.+)", key):
        return f"backbone.out_norms.{list(out_indices).index(int(m.group(1)) - 1)}.{m.group(2)}"
    for old, new in _KEY_RENAMES:
        key = key.replace(old, new)
    return key


class GroundingDINO(nn.Module):
    def __init__(self, model_type="tiny"):
        """``model_type``: ``"tiny"``, ``"base"``, or a Hub repo id of a
        Grounding DINO or MM-Grounding-DINO checkpoint in transformers format."""
        super().__init__()
        self.model_string = MODEL_ALIASES.get(model_type, model_type)
        cfg = load_config(self.model_string)

        d_model = cfg["d_model"]
        num_levels = cfg["num_feature_levels"]
        layer_norm_eps = cfg["layer_norm_eps"]
        self.d_model = d_model
        self.num_queries = cfg["num_queries"]
        self.max_text_len = cfg["max_text_len"]

        self.backbone = SwinBackbone(
            cfg["embed_dim"], cfg["depths"], cfg["num_heads"], cfg["window_size"], cfg["out_indices"]
        )
        self.position_embedding = SinePositionEmbedding(d_model // 2, cfg["positional_embedding_temperature"])
        in_channels = self.backbone.channels
        if num_levels <= len(in_channels):
            raise ValueError(f"Expected more feature levels ({num_levels}) than backbone outputs ({len(in_channels)})")
        # Levels past the backbone outputs are strided convs, the first on the
        # last backbone stage and any further ones on the level before.
        extra_in = [in_channels[-1]] + [d_model] * (num_levels - len(in_channels) - 1)
        self.input_proj = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(c, d_model, kernel_size=1), nn.GroupNorm(32, d_model)) for c in in_channels]
            + [
                nn.Sequential(nn.Conv2d(c, d_model, 3, stride=2, padding=1), nn.GroupNorm(32, d_model))
                for c in extra_in
            ]
        )
        self.level_embed = nn.Parameter(torch.zeros(num_levels, d_model))

        self.text_backbone = BertModel()
        self.text_projection = nn.Linear(self.text_backbone.hidden_size, d_model)

        self.encoder = Encoder(
            cfg["encoder_layers"], d_model=d_model, num_heads=cfg["encoder_attention_heads"],
            num_levels=num_levels, num_points=cfg["encoder_n_points"], ffn_dim=cfg["encoder_ffn_dim"],
            layer_norm_eps=layer_norm_eps,
        )
        self.decoder = Decoder(
            cfg["decoder_layers"], d_model, num_heads=cfg["decoder_attention_heads"],
            num_levels=num_levels, num_points=cfg["decoder_n_points"], ffn_dim=cfg["decoder_ffn_dim"],
            layer_norm_eps=layer_norm_eps,
        )

        # Two-stage query selection: every encoder pixel proposes a box, the
        # best-matching 900 seed the decoder.
        scaled = cfg["scaled_class_logits"]
        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.encoder_output_bbox_embed = MLP(d_model, d_model, 4, 3)
        self.encoder_output_class_embed = ContrastiveHead(scaled)
        self.query_position_embeddings = nn.Embedding(self.num_queries, d_model)

        # One box and one class head per decoder layer; a shared box head is
        # stored once and reused by every layer.
        num_box_heads = 1 if cfg["decoder_bbox_embed_share"] else cfg["decoder_layers"]
        self.bbox_embed = nn.ModuleList(MLP(d_model, d_model, 4, 3) for _ in range(num_box_heads))
        self.class_embed = nn.ModuleList(ContrastiveHead(scaled) for _ in range(cfg["decoder_layers"]))

        self.tokenizer = Tokenizer(self.model_string, max_length=self.max_text_len)
        self._load_model(self.model_string)

    def _load_model(self, model_path):
        path = resolve_hub_file(model_path, "model.safetensors")
        state_dict = {}
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                new_key = _remap_key(key, self.backbone.out_indices)
                if new_key is None:
                    continue
                tensor = f.get_tensor(key)
                if new_key in state_dict and not torch.equal(state_dict[new_key], tensor):
                    raise ValueError(f"Checkpoint holds conflicting weights for {new_key}")
                state_dict[new_key] = tensor
        # Assign rather than copy: nothing holds parameter references yet.
        self.load_state_dict(state_dict, strict=True, assign=True)

    def _box_heads(self):
        return list(self.bbox_embed) * (len(self.decoder.layers) // len(self.bbox_embed))

    def precast_weights(self, dtype=torch.float16):
        """Store the image path's Linear and Conv2d weights in ``dtype``, for autocast.

        ``torch.autocast(dtype=dtype)`` casts these weights on every call, so
        storing them cast gives bit-identical outputs without those cast
        kernels: ~2.4 ms per image for tiny and base on an RTX 5070 Ti, and
        half the weights' memory. Detection must then run under that autocast
        (fp32 inputs would meet ``dtype`` weights). BERT is left alone, so
        :meth:`encode_text` still runs in fp32 outside it. Returns ``self``.
        """
        for name, child in self.named_children():
            if name in ("text_backbone", "text_projection"):
                continue
            for module in child.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    module.to(dtype)
        return self

    # ------------------------------------------------------------------ inputs

    def preprocess_image(self, image, size=800, max_size=1333, device=None):
        """Resize, normalize and pad one or more images into a batch.

        Accepts a PIL image, a file path, a ``[3, H, W]`` uint8 tensor, or a
        list/tuple of those. Each image is resized with its short side at
        ``size`` (long side capped at ``max_size``), then the batch is
        zero-padded bottom/right to a common size.

        Resizing stays on the CPU, in torchvision's uint8 path (the GPU's float
        resize rounds ~1 grey level differently); the batch is then moved to
        ``device`` as uint8 and normalized there, a quarter of the transfer.

        Returns ``pixel_values`` ``[B, 3, H, W]`` and ``pixel_mask`` ``[B, H, W]``
        (bool, True on real pixels), on ``device`` (default: the model's).
        """
        images = image if isinstance(image, (list, tuple)) else [image]
        device = torch.device(device) if device is not None else self.level_embed.device
        resized = [self._resize(img, size, max_size) for img in images]
        height = max(t.shape[-2] for t in resized)
        width = max(t.shape[-1] for t in resized)
        pixels = torch.zeros(len(resized), 3, height, width, dtype=torch.uint8, pin_memory=device.type == "cuda")
        pixel_mask = torch.zeros(len(resized), height, width, dtype=torch.bool, device=device)
        for i, t in enumerate(resized):
            pixels[i, :, : t.shape[-2], : t.shape[-1]] = t
            pixel_mask[i, : t.shape[-2], : t.shape[-1]] = True
        pixels = pixels.to(device, non_blocking=True)
        # A tensor divisor keeps CUDA from multiplying by a rounded 1/255
        # instead, so the result is bit-identical to normalizing on the CPU.
        pixel_values = TF.normalize(
            pixels.float().div_(torch.tensor(255.0, device=device)), IMAGENET_MEAN, IMAGENET_STD
        )
        return pixel_values.masked_fill_(~pixel_mask[:, None], 0.0), pixel_mask

    @staticmethod
    def _resize(image, size, max_size):
        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            # Viewing PIL's buffer skips the copy (and fresh-page faults) that
            # pil_to_tensor pays, 7x slower on a 4K frame. It is only read,
            # so torch's warning about it being read-only doesn't apply.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
                image = torch.from_numpy(np.asarray(image)).permute(2, 0, 1)
        height, width = image.shape[-2:]
        return TF.resize(
            image.cpu(), list(resize_shape(height, width, size, max_size)),
            interpolation=TF.InterpolationMode.BILINEAR, antialias=True,
        )

    def tokenize(self, captions):
        """Tokenize caption(s); returns ``input_ids``, ``attention_mask`` ``[B, L]``."""
        return self.tokenizer(captions)

    # ------------------------------------------------------------------ text

    def encode_text(self, input_ids, attention_mask=None) -> TextFeatures:
        """Run BERT once for a caption batch; reuse the result across images.

        A single caption (batch 1) is broadcast to every image it is used with.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        input_ids = input_ids[:, : self.max_text_len]
        attention_mask = attention_mask[:, : self.max_text_len]
        self_attn_mask, position_ids = phrase_masks_and_positions(input_ids)
        features = self.text_backbone(input_ids, self_attn_mask, position_ids)
        return TextFeatures(
            features=self.text_projection(features),
            token_mask=attention_mask.bool(),
            self_attn_mask=self_attn_mask,
            position_ids=position_ids,
        )

    @staticmethod
    def _expand_text(text: TextFeatures, batch_size) -> TextFeatures:
        if text.features.shape[0] == batch_size:
            return text
        if text.features.shape[0] != 1:
            raise ValueError(
                f"Text batch {text.features.shape[0]} matches neither the image batch "
                f"{batch_size} nor 1 (a caption shared by every image)"
            )
        return TextFeatures(*(t.expand(batch_size, *t.shape[1:]) for t in text))

    # ------------------------------------------------------------------ detection

    def forward(self, pixel_values, input_ids, attention_mask=None, pixel_mask=None):
        return self.forward_object_detection(pixel_values, input_ids, attention_mask, pixel_mask)

    def forward_object_detection(self, pixel_values, input_ids, attention_mask=None, pixel_mask=None):
        """Detect in ``pixel_values`` against tokenized caption(s).

        Returns ``pred_logits`` ``[B, 900, L]`` (one logit per query per caption
        token, -inf on padding tokens) and ``pred_boxes`` ``[B, 900, 4]``
        as ``(cx, cy, w, h)`` normalized to each image's unpadded area.
        """
        text = self.encode_text(input_ids, attention_mask)
        return self.forward_object_detection_from_text(pixel_values, text, pixel_mask)

    def forward_object_detection_from_text(self, pixel_values, text: TextFeatures, pixel_mask=None):
        """Detection with a caption already encoded by :meth:`encode_text`."""
        B, _, height, width = pixel_values.shape
        text = self._expand_text(text, B)
        if pixel_mask is None:
            pixel_mask = torch.ones((B, height, width), dtype=torch.bool, device=pixel_values.device)
        pixel_mask = pixel_mask.bool()

        # Multi-scale features at d_model channels, each with its valid-pixel mask.
        features = self.backbone(pixel_values)
        sources = [proj(f) for proj, f in zip(self.input_proj, features)]
        extra = features[-1]
        for proj in self.input_proj[len(features):]:
            extra = proj(extra)
            sources.append(extra)
        masks = [F.interpolate(pixel_mask[None].float(), size=s.shape[-2:])[0].bool() for s in sources]

        spatial_shapes_list = [tuple(s.shape[-2:]) for s in sources]
        vision = torch.cat([s.flatten(2).transpose(1, 2) for s in sources], 1)
        vision_valid = torch.cat([m.flatten(1) for m in masks], 1)
        vision_pos = torch.cat(
            [
                self.position_embedding(m).to(s.dtype).flatten(2).transpose(1, 2) + self.level_embed[level]
                for level, (s, m) in enumerate(zip(sources, masks))
            ],
            1,
        )
        if torch.jit.is_tracing():  # the level sizes are traced tensors; keep them dynamic
            spatial_shapes = torch.stack([torch.stack(hw) for hw in spatial_shapes_list]).to(vision.device)
        else:
            spatial_shapes = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device=vision.device)
        valid_ratios = torch.stack([self._valid_ratio(m) for m in masks], 1)

        # The fusion layers' attention runs ~3x faster unmasked, so drop the
        # masks when nothing is padded (images of one size, one caption). The
        # check is a sync; a traced graph keeps them, for zero-padded inputs.
        vision_padded = text_padded = True
        if not torch.jit.is_tracing():
            vision_padded, text_padded = (~torch.stack([pixel_mask.all(), text.token_mask.all()])).tolist()
        vision, text_hidden = self.encoder(
            vision, vision_pos, vision_valid, spatial_shapes, spatial_shapes_list, valid_ratios,
            text.features, text.token_mask, text.self_attn_mask, text.position_ids, vision_padded, text_padded,
        )

        # Two-stage: score every encoder pixel against the text, keep the top 900.
        object_query, proposals = self._encoder_proposals(vision, vision_valid, spatial_shapes_list)
        enc_logits = self.encoder_output_class_embed(object_query, text_hidden, text.token_mask)
        enc_coords = self.encoder_output_bbox_embed(object_query) + proposals
        topk = enc_logits.max(-1).values.topk(self.num_queries, dim=1).indices
        reference_points = torch.gather(enc_coords, 1, topk[..., None].expand(-1, -1, 4)).detach().sigmoid()
        # Slot i holds the i-th best proposal, so fewer queries keep the best ones.
        queries = self.query_position_embeddings.weight[: self.num_queries].unsqueeze(0).expand(B, -1, -1)

        hidden_states, references = self.decoder(
            queries, reference_points, vision, vision_valid, spatial_shapes, spatial_shapes_list,
            valid_ratios, text_hidden, text.token_mask, self._box_heads(),
        )
        # The last layer's box is a delta on the reference it was given.
        last = hidden_states[-1]
        pred_logits = self.class_embed[-1](last, text_hidden, text.token_mask)
        pred_boxes = (self._box_heads()[-1](last) + torch.logit(references[-2], eps=1e-5)).sigmoid()
        return pred_logits, pred_boxes

    @staticmethod
    def _valid_ratio(mask):
        _, height, width = mask.shape
        valid_h = mask[:, :, 0].sum(1).float() / height
        valid_w = mask[:, 0, :].sum(1).float() / width
        return torch.stack([valid_w, valid_h], -1)

    def _encoder_proposals(self, vision, vision_valid, spatial_shapes_list):
        """A fixed-size box at every encoder pixel (larger on coarser levels), as logits."""
        B = vision.shape[0]
        proposals = []
        start = 0
        for level, (height, width) in enumerate(spatial_shapes_list):
            valid = vision_valid[:, start : start + height * width].view(B, height, width)
            valid_h = valid[:, :, 0].sum(1)
            valid_w = valid[:, 0, :].sum(1)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, dtype=torch.float32, device=vision.device),
                torch.arange(width, dtype=torch.float32, device=vision.device),
                indexing="ij",
            )
            grid = torch.stack([grid_x, grid_y], -1)
            scale = torch.stack([valid_w, valid_h], -1).view(B, 1, 1, 2)
            grid = (grid.unsqueeze(0) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**level)
            proposals.append(torch.cat((grid, wh), -1).view(B, -1, 4))
            start += height * width
        proposals = torch.cat(proposals, 1)
        in_range = ((proposals > 0.01) & (proposals < 0.99)).all(-1, keepdim=True)
        keep = in_range & vision_valid[..., None]
        proposals = torch.log(proposals / (1 - proposals)).masked_fill(~keep, float("inf"))

        object_query = vision.masked_fill(~keep, 0.0)
        return self.enc_output_norm(self.enc_output(object_query)), proposals

    # ------------------------------------------------------------------ outputs

    @staticmethod
    def postprocess_boxes(boxes, target_sizes):
        """``(cx, cy, w, h)`` in [0, 1] -> pixel ``(x0, y0, x1, y1)``.

        ``target_sizes`` is ``[B, 2]`` ``(height, width)`` of the original images.
        """
        target_sizes = torch.as_tensor(target_sizes, device=boxes.device, dtype=boxes.dtype)
        if target_sizes.ndim != 2 or target_sizes.shape != (boxes.shape[0], 2):
            raise ValueError(
                f"target_sizes must be [{boxes.shape[0]}, 2] (height, width), got {tuple(target_sizes.shape)}"
            )
        cx, cy, w, h = boxes.unbind(-1)
        boxes_xyxy = torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)
        image_h, image_w = target_sizes.unbind(1)
        scale = torch.stack([image_w, image_h, image_w, image_h], dim=1).unsqueeze(1)
        return boxes_xyxy * scale

    @staticmethod
    def phrase_logits(pred_logits, phrase_token_indices: Sequence[Sequence[int]]):
        """Max-pool per-token logits to one logit per phrase: ``[B, Q, num_phrases]``.

        ``phrase_token_indices`` comes from ``Tokenizer.phrase_token_indices``.
        """
        # One gather for every phrase rather than an index + amax each (80 for
        # COCO): phrases are padded to the longest with a -inf column.
        width = max(len(idx) for idx in phrase_token_indices)
        pad = pred_logits.shape[-1]
        index = torch.tensor(
            [list(idx) + [pad] * (width - len(idx)) for idx in phrase_token_indices], device=pred_logits.device
        )
        padded = F.pad(pred_logits, (0, 1), value=float("-inf"))
        return padded[..., index].amax(-1)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GroundingDINO("tiny").eval().to(device)
    with torch.no_grad():
        pixel_values = torch.randn(1, 3, 800, 1067, device=device)
        input_ids, attention_mask = model.tokenize("a cat. a remote control.")
        logits, boxes = model(pixel_values, input_ids.to(device), attention_mask.to(device))
        print("pred_logits", logits.shape)
        print("pred_boxes", boxes.shape)
