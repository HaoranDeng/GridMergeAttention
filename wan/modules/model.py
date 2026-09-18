# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import repeat

from ..rmsnorm_ops import rmsnorm
from .._rocm_runtime import is_rocm

from attention import flash_attention, token_merging_attn
from attention.flash_attn_wan21_1_3b_480p_triton import flash_attention_triton
from attention.gma_output_calibration import (
    apply_query_conditioned_gma_calibration,
    maybe_capture_gma_output_calibration,
)
from attention.runtime_timing import ATTENTION_RUNTIME_RECORDER

__all__ = ['WanModel']

_FIRST_QKV_SAVED = False

_TEMPORAL_ROUTER_CYCLES = {
    2: (5, 15),
    4: (0, 7, 13, 20),
}

_WEIGHTED_SPAN_PER_FRAME_LAYERS = {
    3: frozenset((9, 12, 25)),
}


def resolve_temporal_router_cycle(seed_mode, layer_idx, call_index):
    """Resolve one cheap router frame for this layer and denoising call."""

    seed_mode = str(seed_mode)
    matches = list(re.finditer(r"_lyrc(2|4)(?=_|$)", seed_mode))
    if not matches:
        return seed_mode
    if len(matches) != 1:
        raise ValueError("seed_mode must contain at most one temporal router cycle")
    match = matches[0]
    cycle_size = int(match.group(1))
    frames = _TEMPORAL_ROUTER_CYCLES[cycle_size]
    frame = frames[(int(layer_idx) + int(call_index)) % cycle_size]
    return seed_mode[: match.start()] + f"_rf{frame}" + seed_mode[match.end() :]


def resolve_weighted_span_layer_schedule(seed_mode, layer_idx):
    """Use per-frame routing only in QKV-ranked sensitive layers."""

    seed_mode = str(seed_mode)
    matches = list(re.finditer(r"_lpf(3)(?=_|$)", seed_mode))
    if not matches:
        return seed_mode
    if len(matches) != 1:
        raise ValueError("seed_mode must contain at most one per-frame layer schedule")
    if not seed_mode.startswith("weighted_span_"):
        raise ValueError("per-frame layer schedules require weighted-span routing")
    match = matches[0]
    count = int(match.group(1))
    replacement = (
        "_hg1o0_j4_p16_v8"
        if int(layer_idx) in _WEIGHTED_SPAN_PER_FRAME_LAYERS[count]
        else "_t1_j4"
    )
    return seed_mode[: match.start()] + replacement + seed_mode[match.end() :]


def maybe_save_first_attention_qkv(q, k, v, tag):
    global _FIRST_QKV_SAVED
    path = os.environ.get("TOKEN_MERGING_SAVE_FIRST_QKV_PATH")
    if _FIRST_QKV_SAVED or not path:
        return
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    _FIRST_QKV_SAVED = True
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "q": q.detach().transpose(1, 2).contiguous().to(torch.float16).cpu().numpy(),
        "k": k.detach().transpose(1, 2).contiguous().to(torch.float16).cpu().numpy(),
        "v": v.detach().transpose(1, 2).contiguous().to(torch.float16).cpu().numpy(),
        "tag": tag,
        "shape": tuple(q.shape),
    }
    import numpy as np
    np.savez(path, **payload)
    print(f"[qkv_capture] saved first attention q/k/v to {path} tag={tag} shape={tuple(q.shape)}")
    if os.environ.get("TOKEN_MERGING_EXIT_AFTER_QKV_SAVE", "").lower() in {"1", "true", "yes"}:
        raise SystemExit(0)


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


# @amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

@torch.no_grad()
def build_cos_sin_cache_3d(freqs_t, freqs_h, freqs_w, f, h, w, bsz, start_frame=0, device="cuda"):
    S = f * h * w
    cis = torch.cat([
        freqs_t[start_frame:start_frame+f][:, None, None, :].expand(f, h, w, -1),
        freqs_h[:h][None, :, None, :].expand(f, h, w, -1),
        freqs_w[:w][None, None, :, :].expand(f, h, w, -1),
    ], dim=-1).reshape(S, -1)
    cos_sin_cache = torch.cat([cis.real, cis.imag], dim=-1).to(device=device, dtype=torch.float32).contiguous()
    positions = torch.arange(S, device=device, dtype=torch.int32).repeat(bsz)
    return cos_sin_cache, positions

def _rope_rotate_heads(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleave: bool,
) -> torch.Tensor:
    """Rotate the first ``2 * cos.shape[-1]`` channels; matches FlashInfer ``_rotate``."""
    rotary_dim = cos.shape[-1] * 2
    x_rot = x[..., :rotary_dim].to(torch.float32)
    x_pass = x[..., rotary_dim:]
    if interleave:
        x1 = x_rot[..., 0::2]
        x2 = x_rot[..., 1::2]
        rotated_1 = x1 * cos - x2 * sin
        rotated_2 = x2 * cos + x1 * sin
        interleaved = torch.stack([rotated_1, rotated_2], dim=-1)
        rotated = interleaved.reshape(*x_rot.shape)
    else:
        half = rotary_dim // 2
        x1 = x_rot[..., :half]
        x2 = x_rot[..., half:]
        rotated_1 = x1 * cos - x2 * sin
        rotated_2 = x2 * cos + x1 * sin
        rotated = torch.cat([rotated_1, rotated_2], dim=-1)
    if x_pass.numel() == 0:
        return rotated.to(x.dtype)
    return torch.cat([rotated.to(x.dtype), x_pass], dim=-1)


def _apply_rope_video_tokens_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch RoPE fallback applied independently to every attention head."""
    B, S, H, D = q.shape
    half = D // 2
    positions = positions.to(device=cos_sin_cache.device, dtype=torch.long)
    gathered = cos_sin_cache.index_select(0, positions)
    cos = gathered[:, :half].to(dtype=q.dtype).view(B * S, 1, half)
    sin = gathered[:, half:].to(dtype=q.dtype).view(B * S, 1, half)

    q_view = q.reshape(B * S, H, D)
    k_view = k.reshape(B * S, H, D)

    if is_neox:
        q1, q2 = q_view[..., :half], q_view[..., half:]
        k1, k2 = k_view[..., :half], k_view[..., half:]
        q_out = torch.cat(
            [q1 * cos - q2 * sin, q1 * sin + q2 * cos],
            dim=-1,
        )
        k_out = torch.cat(
            [k1 * cos - k2 * sin, k1 * sin + k2 * cos],
            dim=-1,
        )
    else:
        q_even, q_odd = q_view[..., ::2], q_view[..., 1::2]
        k_even, k_odd = k_view[..., ::2], k_view[..., 1::2]
        q_out = torch.stack(
            [q_even * cos - q_odd * sin,
             q_even * sin + q_odd * cos],
            dim=-1,
        ).flatten(-2)
        k_out = torch.stack(
            [k_even * cos - k_odd * sin,
             k_even * sin + k_odd * cos],
            dim=-1,
        ).flatten(-2)

    return q_out.view(B, S, H, D), k_out.view(B, S, H, D)


def apply_rope_video_tokens(
    q: torch.Tensor,
    k: torch.Tensor,
    rope_cache: tuple[torch.Tensor, torch.Tensor],
    *,
    is_neox: bool = False,
):
    cos_sin_cache, positions = rope_cache
    B, S, H, D = q.shape

    if torch.is_grad_enabled() or q.requires_grad or k.requires_grad or is_rocm():
        return _apply_rope_video_tokens_torch(
            q, k, cos_sin_cache, positions, is_neox=is_neox
        )

    q2 = q.reshape(B * S, H * D).contiguous()
    k2 = k.reshape(B * S, H * D).contiguous()
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

    apply_rope_with_cos_sin_cache_inplace(
        positions=positions.to(device=q.device, dtype=torch.int32),
        query=q2,
        key=k2,
        head_size=D,
        cos_sin_cache=cos_sin_cache.to(device=q.device, dtype=torch.float32).contiguous(),
        is_neox=is_neox,
    )
    return q2.view(B, S, H, D), k2.view(B, S, H, D)

class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.enable_token_merging = False
        self.token_merging_h_reduce = 1
        self.token_merging_w_reduce = 1
        self.token_merging_f_tied = 1
        self.token_merging_q_init = None
        self.token_merging_query_outer_chunk = None
        self.token_merging_layout = "hard_route"
        self.token_merging_num_merge_tokens = 4
        self.token_merging_seed_mode = "local_with_ith"
        self.token_merging_seed = 0
        self.token_merging_hard_route_backend = "sdpa"
        self.token_merging_head_chunk = None
        self.token_merging_exact_intra_frame_attention = False
        self.token_merging_exact_neighbor_window_columns = 0
        self.token_merging_exact_query_rows = 0
        self.token_merging_first_frame_full_attention = False
        self.token_merging_full_attention_anchor_time_count = 0
        self.token_merging_full_attention_anchor_row_count = 0
        self.token_merging_full_attention_anchor_width_count = 0
        self.token_merging_full_attention_anchor_scale = 1.0
        self.token_merging_seed_table_cache = {}
        self.gma_output_calibration_timestep = None
        self.gma_output_calibration_weights = None
        self.gma_query_calibration_parameters = None
        self.token_merging_dense_heads = ()
        self.token_merging_vmonarch_iterations = 2
        self.token_merging_vsa_sparsity = 0.9
        self.sparsepr_options = {}
        self.attention_backend = "sdpa"
        self.layer_idx = -1
        self.seed_query_proj = None

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, grid_size, rope_cache):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            grid_size(Sequence): List of (F, H, W)
        """
        full_start = ATTENTION_RUNTIME_RECORDER.start()
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            if torch.is_grad_enabled() or x.requires_grad:
                q = self.norm_q(self.q(x)).view(b, s, n, d)
                k = self.norm_k(self.k(x)).view(b, s, n, d)
            else:
                q = rmsnorm(self.q(x), self.norm_q.weight, eps=self.norm_q.eps).view(b, s, n, d)
                k = rmsnorm(self.k(x), self.norm_k.weight, eps=self.norm_k.eps).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        # q/k/v: [batch, sequence, head, head_dim].
        # Default 480p/81-frame Wan: [B, 21 * 30 * 52, H, D].
        roped_query, roped_key = apply_rope_video_tokens(q, k, rope_cache)
        seed_query = None
        if self.enable_token_merging and self.seed_query_proj is not None:
            seed_query = rmsnorm(
                self.seed_query_proj(x),
                self.norm_q.weight,
                eps=self.norm_q.eps,
            ).view(b, s, n, d)
            seed_query, _ = apply_rope_video_tokens(
                seed_query, seed_query.clone(), rope_cache
            )
        # roped_query/roped_key keep [batch, sequence, head, head_dim].
        maybe_save_first_attention_qkv(roped_query, roped_key, v, "wan_self_attn")

        core_start = ATTENTION_RUNTIME_RECORDER.start()
        if self.enable_token_merging:
            f, h, w = grid_size
            b, s, _, d = q.shape
            if f % self.token_merging_f_tied != 0:
                raise ValueError("frame count must be divisible by token_merging_f_tied.")
            seed_mode = self.token_merging_seed_mode
            if "_lpf" in str(seed_mode):
                seed_mode = resolve_weighted_span_layer_schedule(
                    seed_mode, self.layer_idx
                )
            if "_lyrc" in str(seed_mode):
                router_call_index = int(
                    getattr(self, "_gma_temporal_router_call_index", 0)
                )
                seed_mode = resolve_temporal_router_cycle(
                    seed_mode,
                    self.layer_idx,
                    router_call_index,
                )
                self._gma_temporal_router_call_index = router_call_index + 1
            x = token_merging_attn(
                roped_query,
                roped_key,
                v,
                self.token_merging_f_tied,
                self.token_merging_h_reduce,
                self.token_merging_w_reduce,
                h,
                w,
                q_init=self.token_merging_q_init,
                query_outer_chunk=self.token_merging_query_outer_chunk,
                layout=self.token_merging_layout,
                num_merge_tokens=self.token_merging_num_merge_tokens,
                seed_mode=seed_mode,
                seed=self.token_merging_seed,
                hard_route_backend=self.token_merging_hard_route_backend,
                head_chunk=self.token_merging_head_chunk,
                exact_intra_frame_attention=(
                    self.token_merging_exact_intra_frame_attention
                ),
                exact_neighbor_window_columns=(
                    self.token_merging_exact_neighbor_window_columns
                ),
                exact_query_rows=self.token_merging_exact_query_rows,
                first_frame_full_attention=(
                    self.token_merging_first_frame_full_attention
                ),
                full_attention_anchor_time_count=(
                    self.token_merging_full_attention_anchor_time_count
                ),
                full_attention_anchor_row_count=(
                    self.token_merging_full_attention_anchor_row_count
                ),
                full_attention_anchor_width_count=(
                    self.token_merging_full_attention_anchor_width_count
                ),
                full_attention_anchor_scale=(
                    self.token_merging_full_attention_anchor_scale
                ),
                dense_heads=self.token_merging_dense_heads,
                seed_query=seed_query,
                seed_table_cache=self.token_merging_seed_table_cache,
                cache_timestep=self.gma_output_calibration_timestep,
            )
        else:
            backend = self.attention_backend.lower()
            if backend == "flash_attn_triton":
                if self.window_size != (-1, -1):
                    raise ValueError("flash_attn_triton backend supports dense self-attention only.")
                x = flash_attention_triton(roped_query, roped_key, v, scale=d**-0.5)
            elif backend == "sdpa":
                x = flash_attention(
                    q=roped_query,
                    k=roped_key,
                    v=v,
                    window_size=self.window_size)
            elif backend in {"svg_all", "svg2_all", "ear_all"}:
                from attention.svg_family_wan import svg_family_attention

                x = svg_family_attention(
                    roped_query,
                    roped_key,
                    v,
                    backend,
                    self.layer_idx,
                    grid_size,
                )
            elif backend == "vmonarch_all":
                from attention.vmonarch_wan import vmonarch_attention

                x = vmonarch_attention(
                    roped_query,
                    roped_key,
                    v,
                    grid_size,
                    iterations=self.token_merging_vmonarch_iterations,
                    head_chunk=self.token_merging_head_chunk or 1,
                    first_frame_full_attention=(
                        self.token_merging_first_frame_full_attention
                    ),
                )
            elif backend == "vsa_training_free_all":
                from attention.vsa_training_free_wan import (
                    vsa_training_free_attention,
                )

                x = vsa_training_free_attention(
                    roped_query,
                    roped_key,
                    v,
                    grid_size,
                    sparsity=self.token_merging_vsa_sparsity,
                    head_chunk=self.token_merging_head_chunk or 1,
                )
            elif backend == "sparsepr_all":
                from attention.sparsepr_wan import sparsepr_wan21_attention

                x = sparsepr_wan21_attention(
                    self,
                    roped_query,
                    roped_key,
                    v,
                    layer_idx=self.layer_idx,
                    options=self.sparsepr_options,
                )
            else:
                raise ValueError(f"unknown Wan self-attention backend: {self.attention_backend}")
        ATTENTION_RUNTIME_RECORDER.stop_core(core_start)

        if self.enable_token_merging:
            maybe_capture_gma_output_calibration(
                x,
                roped_query,
                roped_key,
                v,
                layer_idx=self.layer_idx,
                timestep=self.gma_output_calibration_timestep,
            )

        query_calibration = self.gma_query_calibration_parameters
        if query_calibration is not None:
            timestep = int(self.gma_output_calibration_timestep)
            if timestep not in query_calibration:
                raise ValueError(
                    "no query-conditioned GMA calibration for timestep "
                    f"{timestep}; available={sorted(query_calibration)}"
                )
            x = apply_query_conditioned_gma_calibration(
                x,
                roped_query,
                query_calibration[timestep],
            )

        # output
        x = x.flatten(2)
        calibration_weights = self.gma_output_calibration_weights
        if query_calibration is not None or calibration_weights is None:
            x = self.o(x)
        else:
            timestep = int(self.gma_output_calibration_timestep)
            if timestep not in calibration_weights:
                raise ValueError(
                    "no GMA output calibration for timestep "
                    f"{timestep}; available={sorted(calibration_weights)}"
                )
            x = F.linear(x, calibration_weights[timestep], self.o.bias)
        ATTENTION_RUNTIME_RECORDER.stop_full(full_start)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanGanCrossAttention(WanSelfAttention):

    def forward(self, x, context, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        qq = self.norm_q(self.q(context)).view(b, 1, -1, d)

        kk = self.norm_k(self.k(x)).view(b, -1, n, d)
        vv = self.v(x).view(b, -1, n, d)

        # compute attention
        x = flash_attention(qq, kk, vv)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(
            dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        grid_size,
        context,
        context_lens,
        rope_cache,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            grid_size(Sequence): List of (F, H, W)
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation + e).chunk(6, dim=1)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0], grid_size, rope_cache)
        # with amp.autocast(dtype=torch.float32):
        x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
            # with amp.autocast(dtype=torch.float32):
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class GanAttentionBlock(nn.Module):

    def __init__(self,
                 dim=1536,
                 ffn_dim=8192,
                 num_heads=12,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        # self.norm1 = WanLayerNorm(dim, eps)
        # self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
        #   eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()

        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        self.cross_attn = WanGanCrossAttention(dim, num_heads,
                                               (-1, -1),
                                               qk_norm,
                                               eps)

        # modulation
        # self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        context,
        # seq_lens,
        # grid_sizes,
        # freqs,
        # context,
        # context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        # e = (self.modulation + e).chunk(6, dim=1)
        # assert e[0].dtype == torch.float32

        # # self-attention
        # y = self.self_attn(
        #     self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes,
        #     freqs)
        # # with amp.autocast(dtype=torch.float32):
        # x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context):
            token = context + self.cross_attn(self.norm3(x), context)
            y = self.ffn(self.norm2(token)) + token  # * (1 + e[4]) + e[3])
            # with amp.autocast(dtype=torch.float32):
            # x = x + y * e[5]
            return y

        x = cross_attn_ffn(x, context)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class RegisterTokens(nn.Module):
    def __init__(self, num_registers: int, dim: int):
        super().__init__()
        self.register_tokens = nn.Parameter(torch.randn(num_registers, dim) * 0.02)
        self.rms_norm = WanRMSNorm(dim, eps=1e-6)

    def forward(self):
        return self.rms_norm(self.register_tokens)

    def reset_parameters(self):
        nn.init.normal_(self.register_tokens, std=0.02)


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.local_attn_size = 21
        self._token_merging_args = {}

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs_t = rope_params(1024, d - 4 * (d // 6))
        self.freqs_h = rope_params(1024, 2 * (d // 6))
        self.freqs_w = rope_params(1024, 2 * (d // 6))

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False
    
    @property
    def token_merging_args(self):
        return self._token_merging_args

    @token_merging_args.setter
    def token_merging_args(self, args: dict):
        self._token_merging_args = args
        enable = args.get("enable", False)
        enabled_layers = args.get("enabled_layers", None)
        if enabled_layers is not None:
            enabled_layers = list(enabled_layers)
            if any(
                isinstance(layer_idx, bool) or not isinstance(layer_idx, int)
                for layer_idx in enabled_layers
            ):
                raise TypeError(
                    "token_merging_args.enabled_layers must contain integers."
                )
            if len(set(enabled_layers)) != len(enabled_layers):
                raise ValueError(
                    "token_merging_args.enabled_layers contains duplicates."
                )
            invalid_layers = [
                layer_idx
                for layer_idx in enabled_layers
                if layer_idx < 0 or layer_idx >= len(self.blocks)
            ]
            if invalid_layers:
                raise ValueError(
                    "token_merging_args.enabled_layers contains out-of-range "
                    f"indices {invalid_layers}; valid range is "
                    f"[0, {len(self.blocks) - 1}]."
                )
            enabled_layers = set(enabled_layers)
        f_tied = args.get("f_tied", 1)
        h_reduce = args.get("h_reduce", 1)
        w_reduce = args.get("w_reduce", 1)
        q_init = args.get("q_init", None)
        query_outer_chunk = args.get("query_outer_chunk", None)
        layout = args.get("layout", "hard_route")
        num_merge_tokens = args.get("num_merge_tokens", 4)
        seed_mode = args.get("seed_mode", "local_with_ith")
        singleton_layer_schedule = args.get("singleton_layer_schedule", None)
        if singleton_layer_schedule is not None:
            singleton_layer_schedule = list(singleton_layer_schedule)
            if seed_mode != "singleton_leverage_layerwise":
                raise ValueError(
                    "singleton_layer_schedule requires "
                    "seed_mode='singleton_leverage_layerwise'."
                )
            if len(singleton_layer_schedule) != len(self.blocks):
                raise ValueError(
                    "singleton_layer_schedule must contain one budget per layer."
                )
            supported_singleton_budgets = {
                0,
                2,
                4,
                6,
                8,
                12,
                16,
                20,
                24,
                28,
                32,
            }
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value not in supported_singleton_budgets
                for value in singleton_layer_schedule
            ):
                raise ValueError(
                    "singleton_layer_schedule supports only "
                    "0, 2, 4, 6, 8, 12, 16, 20, 24, 28, or 32 atoms "
                    "per layer."
                )
        elif seed_mode == "singleton_leverage_layerwise":
            raise ValueError(
                "seed_mode='singleton_leverage_layerwise' requires "
                "singleton_layer_schedule."
            )
        dense_head_schedule = args.get("dense_head_schedule", None)
        if dense_head_schedule is not None:
            dense_head_schedule = [tuple(heads) for heads in dense_head_schedule]
            if len(dense_head_schedule) != len(self.blocks):
                raise ValueError(
                    "dense_head_schedule must contain one head set per layer."
                )
            num_heads = self.blocks[0].self_attn.num_heads
            for layer_heads in dense_head_schedule:
                if len(layer_heads) > num_heads:
                    raise ValueError(
                        "dense_head_schedule cannot exceed the model head count."
                    )
                if any(
                    isinstance(head, bool) or not isinstance(head, int)
                    for head in layer_heads
                ):
                    raise TypeError("dense_head_schedule must contain integers.")
                if len(set(layer_heads)) != len(layer_heads):
                    raise ValueError(
                        "dense_head_schedule must not contain duplicate heads."
                    )
                if any(head < 0 or head >= num_heads for head in layer_heads):
                    raise ValueError(
                        f"dense_head_schedule heads must be in [0, {num_heads - 1}]."
                    )
        seed = args.get("seed", 0)
        hard_route_backend = args.get("hard_route_backend", "sdpa")
        head_chunk = args.get("head_chunk", None)
        exact_intra_frame_attention = args.get(
            "exact_intra_frame_attention", False
        )
        exact_neighbor_window_columns = args.get(
            "exact_neighbor_window_columns", 0
        )
        exact_query_rows = args.get("exact_query_rows", 0)
        first_frame_full_attention = args.get(
            "first_frame_full_attention", False
        )
        full_attention_anchor_time_count = args.get(
            "full_attention_anchor_time_count", 0
        )
        full_attention_anchor_row_count = args.get(
            "full_attention_anchor_row_count", 0
        )
        full_attention_anchor_width_count = args.get(
            "full_attention_anchor_width_count", 0
        )
        full_attention_anchor_scale = args.get("full_attention_anchor_scale", 1.0)
        full_attention_anchor_layers = args.get(
            "full_attention_anchor_layers", None
        )
        if full_attention_anchor_layers is not None:
            full_attention_anchor_layers = list(full_attention_anchor_layers)
            if any(
                isinstance(layer_idx, bool) or not isinstance(layer_idx, int)
                for layer_idx in full_attention_anchor_layers
            ):
                raise TypeError(
                    "token_merging_args.full_attention_anchor_layers must "
                    "contain integers."
                )
            if len(set(full_attention_anchor_layers)) != len(
                full_attention_anchor_layers
            ):
                raise ValueError(
                    "token_merging_args.full_attention_anchor_layers contains "
                    "duplicates."
                )
            invalid_anchor_layers = [
                layer_idx
                for layer_idx in full_attention_anchor_layers
                if layer_idx < 0 or layer_idx >= len(self.blocks)
            ]
            if invalid_anchor_layers:
                raise ValueError(
                    "token_merging_args.full_attention_anchor_layers contains "
                    f"out-of-range indices {invalid_anchor_layers}; valid range "
                    f"is [0, {len(self.blocks) - 1}]."
                )
            full_attention_anchor_layers = set(full_attention_anchor_layers)
        vmonarch_iterations = args.get("vmonarch_iterations", 2)
        vsa_sparsity = args.get("vsa_sparsity", 0.9)
        sparsepr_options = dict(args.get("sparsepr_options", {}))
        attention_backend = args.get("attention_backend", "sdpa")

        for layer_idx, block in enumerate(self.blocks):
            use_full_attention_anchors = (
                full_attention_anchor_layers is None
                or layer_idx in full_attention_anchor_layers
            )
            block.self_attn.layer_idx = layer_idx
            block.self_attn.enable_token_merging = enable and (
                enabled_layers is None or layer_idx in enabled_layers
            )
            block.self_attn.token_merging_f_tied = f_tied
            block.self_attn.token_merging_h_reduce = h_reduce
            block.self_attn.token_merging_w_reduce = w_reduce
            block.self_attn.token_merging_q_init = q_init
            block.self_attn.token_merging_query_outer_chunk = query_outer_chunk
            block.self_attn.token_merging_layout = layout
            block.self_attn.token_merging_num_merge_tokens = num_merge_tokens
            layer_seed_mode = seed_mode
            if singleton_layer_schedule is not None:
                singleton_budget = singleton_layer_schedule[layer_idx]
                layer_seed_mode = (
                    "local_with_ith"
                    if singleton_budget == 0
                    else f"singleton_leverage_top{singleton_budget}"
                )
            block.self_attn.token_merging_seed_mode = layer_seed_mode
            block.self_attn.token_merging_seed_table_cache = {}
            block.self_attn.token_merging_seed = seed
            block.self_attn.token_merging_hard_route_backend = hard_route_backend
            block.self_attn.token_merging_head_chunk = head_chunk
            block.self_attn.token_merging_exact_intra_frame_attention = (
                exact_intra_frame_attention
            )
            block.self_attn.token_merging_exact_neighbor_window_columns = (
                exact_neighbor_window_columns
            )
            block.self_attn.token_merging_exact_query_rows = exact_query_rows
            block.self_attn.token_merging_first_frame_full_attention = (
                first_frame_full_attention
            )
            block.self_attn.token_merging_full_attention_anchor_time_count = (
                full_attention_anchor_time_count
                if use_full_attention_anchors
                else 0
            )
            block.self_attn.token_merging_full_attention_anchor_row_count = (
                full_attention_anchor_row_count
                if use_full_attention_anchors
                else 0
            )
            block.self_attn.token_merging_full_attention_anchor_width_count = (
                full_attention_anchor_width_count
                if use_full_attention_anchors
                else 0
            )
            block.self_attn.token_merging_full_attention_anchor_scale = (
                full_attention_anchor_scale
            )
            block.self_attn.token_merging_dense_heads = (
                tuple(sorted(dense_head_schedule[layer_idx]))
                if dense_head_schedule is not None
                else ()
            )
            block.self_attn.token_merging_vmonarch_iterations = (
                vmonarch_iterations
            )
            block.self_attn.token_merging_vsa_sparsity = vsa_sparsity
            block.self_attn.sparsepr_options = dict(sparsepr_options)
            block.self_attn.attention_backend = attention_backend

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        *args,
        **kwargs
    ):
        # if kwargs.get('classify_mode', False) is True:
        # kwargs.pop('classify_mode')
        # return self._forward_classify(*args, **kwargs)
        # else:
        return self._forward(*args, **kwargs)

    def _forward(
        self,
        x,
        t,
        context,
        seq_len,
        classify_mode=False,
        concat_time_embeddings=False,
        register_tokens=None,
        cls_pred_branch=None,
        gan_ca_blocks=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs_t.device != device:
            self.freqs_t = self.freqs_t.to(device)
            self.freqs_h = self.freqs_h.to(device)
            self.freqs_w = self.freqs_w.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        assert x.size(1) <= seq_len

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)
        
        rope_cache = build_cos_sin_cache_3d(self.freqs_t, self.freqs_h, self.freqs_w, *grid_size, x.shape[0], device=device)

        # arguments
        kwargs = dict(
            e=e0,
            grid_size=grid_size,
            context=context,
            context_lens=context_lens,
            rope_cache=rope_cache)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        # TODO: Tune the number of blocks for feature extraction
        final_x = None
        if classify_mode:
            assert register_tokens is not None
            assert gan_ca_blocks is not None
            assert cls_pred_branch is not None

            final_x = []
            registers = repeat(register_tokens(), "n d -> b n d", b=x.shape[0])
            # x = torch.cat([registers, x], dim=1)

        gan_idx = 0
        calibration_timestep = None
        if t.numel() == 1 or torch.all(t == t.flatten()[0]):
            calibration_timestep = int(
                round(float(t.detach().flatten()[0].item()))
            )
        for ii, block in enumerate(self.blocks):
            block.self_attn.gma_output_calibration_timestep = calibration_timestep
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

            if classify_mode and ii in [13, 21, 29]:
                gan_token = registers[:, gan_idx: gan_idx + 1]
                final_x.append(gan_ca_blocks[gan_idx](x, gan_token))
                gan_idx += 1

        if classify_mode:
            final_x = torch.cat(final_x, dim=1)
            if concat_time_embeddings:
                final_x = cls_pred_branch(torch.cat([final_x, 10 * e[:, None, :]], dim=1).view(final_x.shape[0], -1))
            else:
                final_x = cls_pred_branch(final_x.view(final_x.shape[0], -1))

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_size)

        if classify_mode:
            return x, final_x

        return x

    def _forward_classify(
        self,
        x,
        t,
        context,
        seq_len,
        register_tokens,
        cls_pred_branch,
        clip_fea=None,
        y=None,
    ):
        r"""
        Feature extraction through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of video features with original input shapes [C_block, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs_t.device != device:
            self.freqs_t = self.freqs_t.to(device)
            self.freqs_h = self.freqs_h.to(device)
            self.freqs_w = self.freqs_w.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        assert x.size(1) <= seq_len

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            grid_size=grid_size,
            context=context,
            context_lens=context_lens,
            rope_cache=build_cos_sin_cache_3d(
                self.freqs_t,
                self.freqs_h,
                self.freqs_w,
                *grid_size,
                x.shape[0],
                device=device,
            ))

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        # TODO: Tune the number of blocks for feature extraction
        for block in self.blocks[:16]:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        # unpatchify
        x = self.unpatchify(x, grid_size, c=self.dim // 4)
        return x

    def unpatchify(self, x, grid_size, c=None):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_size (Sequence):
                3 dimensions correspond to F_patches, H_patches, W_patches

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim if c is None else c
        x = x[:math.prod(grid_size)].view(-1, *grid_size, *self.patch_size, c)
        x = torch.einsum('bfhwpqrc->bcfphqwr', x)
        x = x.reshape(-1, c, *[i * j for i, j in zip(grid_size, self.patch_size)])
        return x

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
