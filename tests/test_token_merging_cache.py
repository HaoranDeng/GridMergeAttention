"""Unit tests for cache-aware one-token GridMerge attention.

These are small CPU/FP32 tests. They prove that the attention calculation can
use current-block queries together with a longer cached K/V prefix, which is
the central shape difference between ordinary Wan and Self-Forcing inference.
"""

from __future__ import annotations

import unittest

import torch

from attention.common import blocks_to_video, video_to_blocks
from attention.gridmerge_one import gridmerge_one
from attention.token_merging import token_merging_attn
from tests._support import gridmerge_one_reference, require_symbol


class GridMergeUnequalLengthTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.batch = 1
        self.height = 2
        self.width = 3
        self.heads = 2
        self.dim = 4
        self.query_frames = 2
        self.key_frames = 6
        self.frame_tokens = self.height * self.width
        self.q = torch.randn(
            self.batch,
            self.query_frames * self.frame_tokens,
            self.heads,
            self.dim,
            dtype=torch.float32,
        )
        self.k = torch.randn(
            self.batch,
            self.key_frames * self.frame_tokens,
            self.heads,
            self.dim,
            dtype=torch.float32,
        )
        self.v = torch.randn_like(self.k)

    def test_gridmerge_core_already_supports_short_q_and_long_kv(self) -> None:
        """The low-level GridMerge math should not require Q and K frame counts to match."""

        q_blocks = video_to_blocks(
            self.q, 1, 1, 1, self.height, self.width
        )
        k_blocks = video_to_blocks(
            self.k, 1, 1, 1, self.height, self.width
        )
        v_blocks = video_to_blocks(
            self.v, 1, 1, 1, self.height, self.width
        )
        actual = blocks_to_video(
            gridmerge_one(
                q_blocks,
                k_blocks,
                v_blocks,
                self.dim**-0.5,
                "ith",
            ),
            1,
            1,
            1,
        )
        expected = gridmerge_one_reference(
            self.q,
            self.k,
            self.v,
            height=self.height,
            width=self.width,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_dispatcher_accepts_short_q_and_long_kv(self) -> None:
        """The public dispatcher must remove the current equal-length restriction."""

        try:
            actual = token_merging_attn(
                self.q,
                self.k,
                self.v,
                f_tied=1,
                h_reduce=1,
                w_reduce=1,
                h=self.height,
                w=self.width,
                q_init="ith",
                layout="hard_route",
                num_merge_tokens=1,
            )
        except ValueError as exc:
            self.fail(
                "token_merging_attn still rejects the short-Q/long-KV shape "
                f"required by causal caching: {exc}"
            )
        expected = gridmerge_one_reference(
            self.q,
            self.k,
            self.v,
            height=self.height,
            width=self.width,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


class CacheAwareGridMergeTests(unittest.TestCase):
    def _cache_call(self, q, k_cache, v_cache, new_k, new_v, start, end):
        """Call the cache API that the causal attention model is expected to expose."""

        cache_attention = require_symbol(
            "attention.token_merging", "token_merging_attn_with_kv_cache"
        )
        return cache_attention(
            q,
            k_cache,
            v_cache,
            new_k,
            new_v,
            start,
            end,
            f_tied=1,
            h_reduce=1,
            w_reduce=1,
            h=2,
            w=3,
            q_init="ith",
            layout="hard_route",
            num_merge_tokens=1,
        )

    def test_cache_write_and_attention_match_independent_reference(self) -> None:
        """New K/V must overwrite the current slots before attending to the prefix."""

        torch.manual_seed(11)
        batch, heads, dim = 1, 2, 4
        height, width = 2, 3
        frame_tokens = height * width
        prefix_frames, new_frames, capacity_frames = 4, 2, 8
        start = prefix_frames * frame_tokens
        end = (prefix_frames + new_frames) * frame_tokens

        prefix_k = torch.randn(batch, start, heads, dim)
        prefix_v = torch.randn_like(prefix_k)
        new_k = torch.randn(batch, new_frames * frame_tokens, heads, dim)
        new_v = torch.randn_like(new_k)
        q = torch.randn_like(new_k)

        k_cache = torch.full(
            (batch, capacity_frames * frame_tokens, heads, dim),
            float("nan"),
        )
        v_cache = torch.full_like(k_cache, float("nan"))
        k_cache[:, :start] = prefix_k
        v_cache[:, :start] = prefix_v

        actual = self._cache_call(
            q, k_cache, v_cache, new_k, new_v, start, end
        )
        expected = gridmerge_one_reference(
            q,
            torch.cat((prefix_k, new_k), dim=1),
            torch.cat((prefix_v, new_v), dim=1),
            height=height,
            width=width,
        )

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(k_cache[:, start:end], new_k)
        torch.testing.assert_close(v_cache[:, start:end], new_v)
        self.assertTrue(torch.isnan(k_cache[:, end:]).all())
        self.assertTrue(torch.isnan(v_cache[:, end:]).all())

    def test_unused_future_cache_cannot_change_current_output(self) -> None:
        """Values beyond end_idx are future tokens and must never be read."""

        torch.manual_seed(13)
        batch, heads, dim = 1, 2, 4
        frame_tokens = 2 * 3
        prefix_frames, new_frames, capacity_frames = 2, 2, 8
        start = prefix_frames * frame_tokens
        end = (prefix_frames + new_frames) * frame_tokens

        q = torch.randn(batch, new_frames * frame_tokens, heads, dim)
        new_k = torch.randn_like(q)
        new_v = torch.randn_like(q)
        prefix_k = torch.randn(batch, start, heads, dim)
        prefix_v = torch.randn_like(prefix_k)

        cache_shape = (batch, capacity_frames * frame_tokens, heads, dim)
        k_cache_a = torch.full(cache_shape, 1.0e6)
        v_cache_a = torch.full(cache_shape, 1.0e6)
        k_cache_b = torch.full(cache_shape, -1.0e6)
        v_cache_b = torch.full(cache_shape, -1.0e6)
        for cache, prefix in (
            (k_cache_a, prefix_k),
            (k_cache_b, prefix_k),
            (v_cache_a, prefix_v),
            (v_cache_b, prefix_v),
        ):
            cache[:, :start] = prefix

        output_a = self._cache_call(
            q, k_cache_a, v_cache_a, new_k, new_v, start, end
        )
        output_b = self._cache_call(
            q, k_cache_b, v_cache_b, new_k, new_v, start, end
        )

        torch.testing.assert_close(output_a, output_b, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(output_a).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
