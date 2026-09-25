"""Diffusers attention processor that routes HunyuanVideo to Token Merging."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from . import hunyuan_token_merging


class HunyuanTokenMergingAttnProcessor2_0:
    """Drop-in replacement for Diffusers' HunyuanVideo attention processor.

    Projection, QK normalization, video-only RoPE, stream splitting, and output
    projection intentionally follow ``HunyuanVideoAttnProcessor2_0``. Only the
    joint attention primitive itself is replaced.
    """

    def __init__(
        self,
        *,
        h,
        w,
        f_tied=1,
        h_reduce=1,
        w_reduce=1,
        q_init="ith",
        query_outer_chunk=1,
        layout="hard_route",
        num_merge_tokens=1,
        seed_mode="random_with_ith",
        seed=0,
        layer_index=None,
        block_kind=None,
    ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "Hunyuan Token Merging requires PyTorch 2.0 or newer."
            )
        self.h = int(h)
        self.w = int(w)
        self.f_tied = int(f_tied)
        self.h_reduce = int(h_reduce)
        self.w_reduce = int(w_reduce)
        self.q_init = q_init
        self.query_outer_chunk = int(query_outer_chunk)
        self.layout = layout
        self.num_merge_tokens = int(num_merge_tokens)
        self.seed_mode = seed_mode
        self.seed = int(seed)
        self.layer_index = layer_index
        self.block_kind = block_kind

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ):
        if encoder_hidden_states is None:
            raise ValueError(
                "Hunyuan Token Merging requires the joint text stream."
            )

        video_length = hidden_states.shape[1]
        text_length = encoder_hidden_states.shape[1]
        single_stream = attn.add_q_proj is None
        if single_stream:
            projection_input = torch.cat(
                (hidden_states, encoder_hidden_states), dim=1
            )
        else:
            projection_input = hidden_states

        # 1. Video QKV projections (joint video+text projections in the
        # single-stream blocks).
        query = attn.to_q(projection_input)
        key = attn.to_k(projection_input)
        value = attn.to_v(projection_input)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 2. QK normalization.
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 3. RoPE applies only to video tokens. Text tokens in single-stream
        # blocks share the projection but must not receive video coordinates.
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            if single_stream:
                query = torch.cat(
                    (
                        apply_rotary_emb(
                            query[:, :, :video_length], image_rotary_emb
                        ),
                        query[:, :, video_length:],
                    ),
                    dim=2,
                )
                key = torch.cat(
                    (
                        apply_rotary_emb(
                            key[:, :, :video_length], image_rotary_emb
                        ),
                        key[:, :, video_length:],
                    ),
                    dim=2,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        # 4. Dual-stream blocks use separate text projections and norms.
        if not single_stream:
            text_query = attn.add_q_proj(encoder_hidden_states)
            text_key = attn.add_k_proj(encoder_hidden_states)
            text_value = attn.add_v_proj(encoder_hidden_states)

            text_query = text_query.unflatten(
                2, (attn.heads, -1)
            ).transpose(1, 2)
            text_key = text_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            text_value = text_value.unflatten(
                2, (attn.heads, -1)
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                text_query = attn.norm_added_q(text_query)
            if attn.norm_added_k is not None:
                text_key = attn.norm_added_k(text_key)

            query = torch.cat((query, text_query), dim=2)
            key = torch.cat((key, text_key), dim=2)
            value = torch.cat((value, text_value), dim=2)

        if query.shape[2] != video_length + text_length:
            raise ValueError(
                "projected Hunyuan sequence does not match video+text lengths."
            )

        # 5. Convert Diffusers [B, H, S, D] tensors to the repository's public
        # [B, S, H, D] contract and replace only the attention primitive.
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        video_output, text_output = (
            hunyuan_token_merging.hunyuan_token_merging_attn(
                query[:, :video_length],
                key[:, :video_length],
                value[:, :video_length],
                query[:, video_length:],
                key[:, video_length:],
                value[:, video_length:],
                f_tied=self.f_tied,
                h_reduce=self.h_reduce,
                w_reduce=self.w_reduce,
                h=self.h,
                w=self.w,
                attention_mask=attention_mask,
                sm_scale=getattr(attn, "scale", None),
                q_init=self.q_init,
                query_outer_chunk=self.query_outer_chunk,
                layout=self.layout,
                num_merge_tokens=self.num_merge_tokens,
                seed_mode=self.seed_mode,
                seed=self.seed,
            )
        )

        # 6. Preserve Diffusers' per-stream output projections.
        hidden_states = video_output.flatten(2, 3).to(query.dtype)
        encoder_hidden_states = text_output.flatten(2, 3).to(query.dtype)

        if getattr(attn, "to_out", None) is not None:
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
        if getattr(attn, "to_add_out", None) is not None:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


__all__ = ["HunyuanTokenMergingAttnProcessor2_0"]
