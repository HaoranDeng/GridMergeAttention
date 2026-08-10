"""Regression tests for upstream main's hard-C4 path after branch integration."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from attention.hard_route_pytorch import hard_route_attention_pytorch
from attention.token_merging import token_merging_attn


class WanHardC4CompatibilityTests(unittest.TestCase):
    def test_dispatcher_preserves_upstream_pytorch_hard_c4(self):
        """Adding causal C1 must not change main's maintained hard-C4 output."""

        torch.manual_seed(20260810)
        batch, frames, height, width, heads, dim = 1, 2, 6, 7, 2, 4
        shape = (batch, frames * height * width, heads, dim)
        query = torch.randn(shape)
        key = torch.randn(shape)
        value = torch.randn(shape)
        scale = dim**-0.5

        actual = token_merging_attn(
            query,
            key,
            value,
            f_tied=1,
            h_reduce=1,
            w_reduce=1,
            h=height,
            w=width,
            sm_scale=scale,
            query_outer_chunk=1,
            layout="hard_route",
            num_merge_tokens=4,
            seed_mode="local_with_ith",
            hard_route_backend="sdpa",
        )

        query_blocks = query.reshape(
            batch, frames, height, width, heads, dim
        )
        key_blocks = key.reshape(batch, frames, height, width, heads, dim)
        value_blocks = value.reshape(
            batch, frames, height, width, heads, dim
        )
        expected_blocks = torch.empty_like(query_blocks)
        for frame in range(frames):
            expected_blocks[:, frame : frame + 1] = (
                hard_route_attention_pytorch(
                    query_blocks[:, frame : frame + 1],
                    key_blocks,
                    value_blocks,
                    scale,
                    query_outer_offset=frame,
                )
            )
        expected = expected_blocks.reshape_as(query)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_dispatcher_preserves_new_fused_hard_c4_backend(self):
        """Main's fused C4 selection must survive the causal-C1 dispatcher."""

        query = torch.randn(1, 2 * 6 * 7, 1, 4)

        def fill_fused_output(q, _k, _v, _scale, output=None):
            self.assertIsNotNone(output)
            output.fill_(3.25)
            return output

        with mock.patch(
            "attention.token_merging.hard_route_attention_triton_fused",
            side_effect=fill_fused_output,
        ) as fused:
            actual = token_merging_attn(
                query,
                query,
                query,
                f_tied=1,
                h_reduce=1,
                w_reduce=1,
                h=6,
                w=7,
                query_outer_chunk=1,
                num_merge_tokens=4,
                hard_route_backend="hard_c4_fused",
            )

        self.assertEqual(fused.call_count, 2)
        torch.testing.assert_close(actual, torch.full_like(actual, 3.25))

    def test_removed_wan_c2_c6_paths_stay_rejected(self):
        """Only causal C1 and upstream hard C4 remain public Wan paths."""

        query = torch.randn(1, 6 * 7, 1, 4)
        for merge_tokens in (2, 6):
            with self.subTest(num_merge_tokens=merge_tokens):
                with self.assertRaisesRegex(ValueError, "num_merge_tokens=4"):
                    token_merging_attn(
                        query,
                        query,
                        query,
                        f_tied=1,
                        h_reduce=1,
                        w_reduce=1,
                        h=6,
                        w=7,
                        num_merge_tokens=merge_tokens,
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
