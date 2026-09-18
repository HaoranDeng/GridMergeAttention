"""Tests that causal Wan self-attention actually dispatches to Token Merging.

The spy records the semantic frame counts rather than only checking that a
function was called. A correct cached run has a fixed current-block Q length
and a K/V prefix that grows as blocks are committed.
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from tests._support import require_symbol


class CausalAttentionDispatchTests(unittest.TestCase):
    def test_current_block_queries_attend_to_growing_prefix_through_gridmerge(self) -> None:
        """Expected trace for three 2-frame blocks is (2,2), (2,4), (2,6)."""

        attention_class = require_symbol(
            "wan.modules.causal_model", "CausalWanSelfAttention"
        )
        causal_module = __import__(
            "wan.modules.causal_model", fromlist=["CausalWanSelfAttention"]
        )
        require_symbol(
            "wan.modules.causal_model", "token_merging_attn_with_kv_cache"
        )

        dim, heads = 8, 2
        height, width = 2, 3
        frame_tokens = height * width
        block_frames = 2
        capacity_frames = 6
        traces = []

        def identity_rope(q, k, rope_cache):
            del rope_cache
            return q, k

        def cache_attention_spy(
            q,
            k_cache,
            v_cache,
            new_k,
            new_v,
            start_idx,
            end_idx,
            *args,
            **kwargs,
        ):
            del args, kwargs
            k_cache[:, start_idx:end_idx] = new_k
            v_cache[:, start_idx:end_idx] = new_v
            traces.append(
                (
                    q.size(1) // frame_tokens,
                    end_idx // frame_tokens,
                )
            )
            return torch.zeros_like(q)

        attention = attention_class(
            dim=dim,
            num_heads=heads,
            local_attn_size=-1,
            sink_size=0,
        ).eval()
        attention.num_frame_per_block = block_frames
        attention.enable_token_merging = True
        attention.token_merging_f_tied = 1
        attention.token_merging_h_reduce = 1
        attention.token_merging_w_reduce = 1
        attention.token_merging_q_init = "ith"
        attention.token_merging_query_outer_chunk = None
        attention.token_merging_layout = "hard_route"
        attention.token_merging_num_merge_tokens = 1
        attention.token_merging_seed_mode = "random"
        attention.token_merging_seed = 0
        attention.token_merging_hard_route_backend = "sdpa"
        attention.token_merging_hard_route_routing_backend = "sdpa"

        cache = {
            "k": torch.zeros(
                1,
                capacity_frames * frame_tokens,
                heads,
                dim // heads,
            ),
            "v": torch.zeros(
                1,
                capacity_frames * frame_tokens,
                heads,
                dim // heads,
            ),
            "global_end_index": 0,
            "local_end_index": 0,
        }

        with mock.patch.object(
            causal_module,
            "apply_rope_video_tokens",
            side_effect=identity_rope,
        ), mock.patch.object(
            causal_module,
            "token_merging_attn_with_kv_cache",
            side_effect=cache_attention_spy,
        ):
            with torch.no_grad():
                for start_frame in (0, 2, 4):
                    x = torch.randn(
                        1,
                        block_frames * frame_tokens,
                        dim,
                    )
                    output = attention(
                        x,
                        grid_size=(block_frames, height, width),
                        block_mask=None,
                        rope_cache=None,
                        kv_cache=cache,
                        current_start_frame=start_frame,
                    )
                    self.assertEqual(output.shape, x.shape)

        self.assertEqual(traces, [(2, 2), (2, 4), (2, 6)])
        self.assertEqual(cache["global_end_index"], capacity_frames * frame_tokens)
        self.assertEqual(cache["local_end_index"], capacity_frames * frame_tokens)


if __name__ == "__main__":
    unittest.main(verbosity=2)
