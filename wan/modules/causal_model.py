"""Causal, blockwise Wan backbone used by Self-Forcing inference.

This module keeps the learned parameter names and shapes of :class:`WanModel`
but changes the execution graph: only the current temporal block is queried,
and every self-attention layer reads/writes a persistent K/V cache.  Token
Merging is applied to that current-block-Q / visible-prefix-KV attention.

The causal structure is adapted from the sibling MonarchRT implementation;
Monarch attention itself is deliberately replaced with this repository's
GridMerge token-merging implementation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from attention import flash_attention, token_merging_attn_with_kv_cache

from ..rmsnorm_ops import rmsnorm
from .model import (
    MLPProj,
    WAN_CROSSATTENTION_CLASSES,
    WanLayerNorm,
    WanRMSNorm,
    apply_rope_video_tokens,
    build_cos_sin_cache_3d,
    maybe_save_first_attention_qkv,
    rope_params,
    sinusoidal_embedding_1d,
)

__all__ = [
    "CausalWanSelfAttention",
    "CausalWanAttentionBlock",
    "CausalHead",
    "CausalWanModel",
]


class CausalWanSelfAttention(nn.Module):
    """Wan self-attention over a current block and its visible cached prefix."""

    def __init__(
        self,
        dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        eps=1e-6,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads.")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = int(local_attn_size)
        self.sink_size = int(sink_size)
        self.qk_norm = qk_norm
        self.eps = eps
        self.num_frame_per_block = 1

        # These fields intentionally mirror WanSelfAttention's Token Merging
        # controls.  They contain no learned parameters, preserving checkpoint
        # compatibility with the ordinary Wan model.
        self.enable_token_merging = False
        self.token_merging_h_reduce = 1
        self.token_merging_w_reduce = 1
        self.token_merging_f_tied = 1
        self.token_merging_q_init = None
        self.token_merging_query_outer_chunk = None
        self.token_merging_layout = "hard_route"
        self.token_merging_num_merge_tokens = 1
        self.token_merging_seed_mode = "random"
        self.token_merging_seed = 0
        self.token_merging_hard_route_backend = "sdpa"
        self.token_merging_hard_route_routing_backend = "sdpa"

        # Keep these module names exactly aligned with WanSelfAttention.
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def _project_qkv(self, x):
        batch, sequence = x.shape[:2]
        if (
            not torch.is_grad_enabled()
            and not x.requires_grad
            and isinstance(self.norm_q, WanRMSNorm)
            and isinstance(self.norm_k, WanRMSNorm)
        ):
            q = rmsnorm(
                self.q(x), self.norm_q.weight, eps=self.norm_q.eps
            ).view(batch, sequence, self.num_heads, self.head_dim)
            k = rmsnorm(
                self.k(x), self.norm_k.weight, eps=self.norm_k.eps
            ).view(batch, sequence, self.num_heads, self.head_dim)
        else:
            q = self.norm_q(self.q(x)).view(
                batch, sequence, self.num_heads, self.head_dim
            )
            k = self.norm_k(self.k(x)).view(
                batch, sequence, self.num_heads, self.head_dim
            )
        v = self.v(x).view(batch, sequence, self.num_heads, self.head_dim)
        return q, k, v

    def _cache_interval(self, kv_cache, current_end, num_new_tokens, frame_seq_len):
        """Resolve visible cache indices, rolling a local cache when required."""

        cache_capacity = kv_cache["k"].shape[1]
        sink_tokens = self.sink_size * frame_seq_len
        global_end = int(kv_cache["global_end_index"])
        previous_local_end = int(kv_cache["local_end_index"])

        # Repeated denoising evaluations of one block have current_end equal to
        # global_end. They overwrite the same slots. Only the first evaluation
        # of a new block is allowed to roll/extend the cache.
        needs_roll = (
            self.local_attn_size != -1
            and current_end > global_end
            and num_new_tokens + previous_local_end > cache_capacity
        )
        if needs_roll:
            num_evicted = num_new_tokens + previous_local_end - cache_capacity
            num_rolled = previous_local_end - num_evicted - sink_tokens
            if num_rolled < 0:
                raise ValueError(
                    "KV cache is too small for the configured block and sink sizes."
                )
            if num_rolled:
                kv_cache["k"][:, sink_tokens : sink_tokens + num_rolled] = (
                    kv_cache["k"][
                        :,
                        sink_tokens + num_evicted : sink_tokens + num_evicted + num_rolled,
                    ].clone()
                )
                kv_cache["v"][:, sink_tokens : sink_tokens + num_rolled] = (
                    kv_cache["v"][
                        :,
                        sink_tokens + num_evicted : sink_tokens + num_evicted + num_rolled,
                    ].clone()
                )
            local_end = previous_local_end + current_end - global_end - num_evicted
            cache_start = max(
                0, local_end - self.local_attn_size * frame_seq_len
            )
        else:
            local_end = previous_local_end + current_end - global_end
            cache_start = 0

        local_start = local_end - num_new_tokens
        if not 0 <= cache_start <= local_start <= local_end <= cache_capacity:
            raise ValueError(
                "Invalid causal KV-cache interval: "
                f"visible_start={cache_start}, write=[{local_start}, {local_end}), "
                f"capacity={cache_capacity}."
            )
        return cache_start, local_start, local_end

    def forward(
        self,
        x,
        grid_size,
        block_mask=None,
        rope_cache=None,
        kv_cache=None,
        current_start_frame=0,
    ):
        del block_mask  # Cache visibility implements block causality in inference.
        if kv_cache is None:
            raise ValueError(
                "CausalWanSelfAttention requires a KV cache during generation; "
                "a cacheless call would not be Self-Forcing inference."
            )
        if kv_cache["k"].shape != kv_cache["v"].shape:
            raise ValueError("causal K/V cache tensors must have identical shapes.")

        num_frames, frame_height, frame_width = map(int, grid_size)
        frame_seq_len = frame_height * frame_width
        expected_tokens = num_frames * frame_seq_len
        if x.size(1) != expected_tokens:
            raise ValueError(
                f"current block has {x.size(1)} tokens but grid_size implies "
                f"{expected_tokens}."
            )

        q, k, v = self._project_qkv(x)
        roped_query, roped_key = apply_rope_video_tokens(q, k, rope_cache)
        maybe_save_first_attention_qkv(
            roped_query, roped_key, v, "causal_wan_self_attn"
        )

        current_end = int(current_start_frame) * frame_seq_len + roped_query.size(1)
        cache_start, local_start, local_end = self._cache_interval(
            kv_cache,
            current_end=current_end,
            num_new_tokens=roped_query.size(1),
            frame_seq_len=frame_seq_len,
        )

        if self.enable_token_merging:
            if (
                num_frames % self.token_merging_f_tied
                or frame_height % self.token_merging_h_reduce
                or frame_width % self.token_merging_w_reduce
            ):
                raise ValueError(
                    "current causal block dimensions must be divisible by the "
                    "configured Token Merging reductions."
                )
            # Work with a view beginning at the visible local-cache boundary.
            # start/end are therefore relative to that view.
            visible_k = kv_cache["k"][:, cache_start:local_end]
            visible_v = kv_cache["v"][:, cache_start:local_end]
            output = token_merging_attn_with_kv_cache(
                roped_query,
                visible_k,
                visible_v,
                roped_key,
                v,
                local_start - cache_start,
                local_end - cache_start,
                f_tied=self.token_merging_f_tied,
                h_reduce=self.token_merging_h_reduce,
                w_reduce=self.token_merging_w_reduce,
                h=frame_height,
                w=frame_width,
                q_init=self.token_merging_q_init,
                query_outer_chunk=self.token_merging_query_outer_chunk,
                layout=self.token_merging_layout,
                num_merge_tokens=self.token_merging_num_merge_tokens,
                seed_mode=self.token_merging_seed_mode,
                seed=self.token_merging_seed,
                hard_route_backend=self.token_merging_hard_route_backend,
                hard_route_routing_backend=(
                    self.token_merging_hard_route_routing_backend
                ),
            )
        else:
            kv_cache["k"][:, local_start:local_end].copy_(roped_key)
            kv_cache["v"][:, local_start:local_end].copy_(v)
            output = flash_attention(
                roped_query,
                kv_cache["k"][:, cache_start:local_end],
                kv_cache["v"][:, cache_start:local_end],
            )

        kv_cache["global_end_index"] = current_end
        kv_cache["local_end_index"] = local_end
        return self.o(output.flatten(2))


class CausalWanAttentionBlock(nn.Module):
    """Wan transformer block with per-frame modulation and cached self-attention."""

    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Module names and shapes intentionally match WanAttentionBlock.
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim,
            num_heads,
            local_attn_size,
            sink_size,
            qk_norm,
            eps,
        )
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        grid_size,
        context,
        context_lens,
        block_mask,
        rope_cache,
        kv_cache=None,
        crossattn_cache=None,
        current_start_frame=0,
    ):
        num_frames = e.shape[1]
        if x.shape[1] % num_frames:
            raise ValueError("token count must be divisible by current block frames.")
        frame_seq_len = x.shape[1] // num_frames
        modulation = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        normalized = self.norm1(x).unflatten(1, (num_frames, frame_seq_len))
        attention_input = (normalized * (1 + modulation[1]) + modulation[0]).flatten(
            1, 2
        )
        y = self.self_attn(
            attention_input,
            grid_size,
            block_mask,
            rope_cache,
            kv_cache,
            current_start_frame,
        )
        x = x + (
            y.unflatten(1, (num_frames, frame_seq_len)) * modulation[2]
        ).flatten(1, 2)

        x = x + self.cross_attn(
            self.norm3(x),
            context,
            context_lens,
            crossattn_cache=crossattn_cache,
        )
        y = self.ffn(
            (
                self.norm2(x).unflatten(1, (num_frames, frame_seq_len))
                * (1 + modulation[4])
                + modulation[3]
            ).flatten(1, 2)
        )
        return x + (
            y.unflatten(1, (num_frames, frame_seq_len)) * modulation[5]
        ).flatten(1, 2)


class CausalHead(nn.Module):
    """Wan output head with a separate time modulation for every frame."""

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        num_frames = e.shape[1]
        frame_seq_len = x.shape[1] // num_frames
        modulation = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = self.norm(x).unflatten(1, (num_frames, frame_seq_len))
        return self.head(x * (1 + modulation[1]) + modulation[0])


class CausalWanModel(ModelMixin, ConfigMixin):
    """Checkpoint-compatible Wan model for cached Self-Forcing generation."""

    ignore_for_config = [
        "patch_size",
        "cross_attn_norm",
        "qk_norm",
        "text_dim",
    ]
    _no_split_modules = ["CausalWanAttentionBlock"]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
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
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    ):
        super().__init__()
        if model_type not in {"t2v", "i2v"}:
            raise ValueError("model_type must be 't2v' or 'i2v'.")

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
        self.local_attn_size = int(local_attn_size)
        self.sink_size = int(sink_size)
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        cross_attn_type = (
            "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        )
        self.blocks = nn.ModuleList(
            [
                CausalWanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    self.local_attn_size,
                    self.sink_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        if dim % num_heads or (dim // num_heads) % 2:
            raise ValueError("Wan head dimension must be even.")
        head_dim = dim // num_heads
        self.freqs_t = rope_params(1024, head_dim - 4 * (head_dim // 6))
        self.freqs_h = rope_params(1024, 2 * (head_dim // 6))
        self.freqs_w = rope_params(1024, 2 * (head_dim // 6))

        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim)

        self._token_merging_args = {}
        self._num_frame_per_block = 1
        self.independent_first_frame = False
        self.gradient_checkpointing = False
        self.init_weights()

    @property
    def token_merging_args(self):
        return self._token_merging_args

    @token_merging_args.setter
    def token_merging_args(self, args):
        args = dict(args or {})
        self._token_merging_args = args
        attributes = {
            "enable_token_merging": args.get("enable", False),
            "token_merging_f_tied": args.get("f_tied", 1),
            "token_merging_h_reduce": args.get("h_reduce", 1),
            "token_merging_w_reduce": args.get("w_reduce", 1),
            "token_merging_q_init": args.get("q_init"),
            "token_merging_query_outer_chunk": args.get("query_outer_chunk"),
            "token_merging_layout": args.get("layout", "hard_route"),
            "token_merging_num_merge_tokens": args.get("num_merge_tokens", 1),
            "token_merging_seed_mode": args.get("seed_mode", "random"),
            "token_merging_seed": args.get("seed", 0),
            "token_merging_hard_route_backend": args.get(
                "hard_route_backend", "sdpa"
            ),
            "token_merging_hard_route_routing_backend": args.get(
                "hard_route_routing_backend", "sdpa"
            ),
        }
        for block in self.blocks:
            for name, value in attributes.items():
                setattr(block.self_attn, name, value)

    @property
    def num_frame_per_block(self):
        return self._num_frame_per_block

    @num_frame_per_block.setter
    def num_frame_per_block(self, value):
        value = int(value)
        if value <= 0:
            raise ValueError("num_frame_per_block must be positive.")
        self._num_frame_per_block = value
        for block in self.blocks:
            block.self_attn.num_frame_per_block = value

    def _set_gradient_checkpointing(self, module, value=False):
        del module
        self.gradient_checkpointing = value

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache=None,
        crossattn_cache=None,
        current_start_frame=0,
    ):
        if kv_cache is None or crossattn_cache is None:
            raise ValueError(
                "CausalWanModel generation requires both per-layer self-attention "
                "and cross-attention caches."
            )
        if len(kv_cache) != self.num_layers or len(crossattn_cache) != self.num_layers:
            raise ValueError(
                "cache lists must contain one entry for every transformer layer."
            )
        if self.model_type != "t2v":
            raise NotImplementedError(
                "This inference port currently supports Self-Forcing T2V only."
            )
        if y is not None or clip_fea is not None:
            raise ValueError("T2V Self-Forcing does not accept y or clip_fea.")
        if t.ndim != 2 or t.shape[:1] != x.shape[:1] or t.shape[1] != x.shape[2]:
            raise ValueError(
                "causal timesteps must have shape [batch, current_block_frames]."
            )

        device = self.patch_embedding.weight.device
        if self.freqs_t.device != device:
            self.freqs_t = self.freqs_t.to(device)
            self.freqs_h = self.freqs_h.to(device)
            self.freqs_w = self.freqs_w.to(device)

        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        if x.size(1) > seq_len:
            raise ValueError("current block token count exceeds configured sequence length.")

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)
        )
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(
            0, t.shape
        )

        padded_context = []
        for embedding in context:
            if embedding.size(0) > self.text_len:
                raise ValueError("prompt embedding exceeds the configured text length.")
            padded_context.append(
                torch.cat(
                    [
                        embedding,
                        embedding.new_zeros(
                            self.text_len - embedding.size(0), embedding.size(1)
                        ),
                    ]
                )
            )
        context = self.text_embedding(torch.stack(padded_context))
        rope_cache = build_cos_sin_cache_3d(
            self.freqs_t,
            self.freqs_h,
            self.freqs_w,
            *grid_size,
            x.shape[0],
            start_frame=int(current_start_frame),
            device=device,
        )

        for layer_index, block in enumerate(self.blocks):
            x = block(
                x,
                e=e0,
                grid_size=grid_size,
                context=context,
                context_lens=None,
                block_mask=None,
                rope_cache=rope_cache,
                kv_cache=kv_cache[layer_index],
                crossattn_cache=crossattn_cache[layer_index],
                current_start_frame=current_start_frame,
            )

        x = self.head(x, e.unflatten(0, t.shape).unsqueeze(2))
        return self.unpatchify(x, grid_size)

    def unpatchify(self, x, grid_size):
        channels = self.out_dim
        x = x[: math.prod(grid_size)].view(
            -1, *grid_size, *self.patch_size, channels
        )
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        return x.reshape(
            -1,
            channels,
            *[size * patch for size, patch in zip(grid_size, self.patch_size)],
        )

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for module in self.text_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
        for module in self.time_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
        nn.init.zeros_(self.head.head.weight)
