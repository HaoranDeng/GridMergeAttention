"""Regression tests for upstream main's hard-C4 path after branch integration."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from attention.hard_route_pytorch import (
    hard_route_attention_pytorch,
    wide_local_c4_seed_rows,
)
from attention.core import flash_attention
from attention.hard_c4_wan21_14b_720p_triton import (
    HARD_720_C,
    HARD_720_CENTROID_CANDIDATES,
    HARD_720_MERGE_BLOCK_J,
    HARD_720_MERGE_BLOCK_L,
    HARD_720_MERGE_LOOP_STAGES,
    HARD_720_MERGE_NUM_STAGES,
    HARD_720_MERGE_NUM_WARPS,
    HARD_720_QUERY_OUTER_GROUP,
    HARD_720_ROUTE_NUM_STAGES,
    hard_c4_720p_attention_triton_blocks,
    hard_c4_720p_exact_query_top3_attention_triton_blocks,
    hard_c4_720p_exact_intra_frame_attention_triton_blocks,
    hard_c4_720p_query_centroid_attention_triton_blocks,
    hard_c4_720p_query_response_attention_triton_blocks,
    hard_c4_720p_responsibility_centroid_attention_triton_blocks,
    hard_c4_720p_attention_triton_head_major_to_token,
    hard_c4_720p_attention_triton_head_major_no_entropy_bias,
)
from attention.hard_c4_wan21_1_3b_480p_triton import hard_c4_rt_attention_triton_blocks
from attention.hard_c5_c6_wan21_14b_720p_triton import (
    CENTROID_CANDIDATES,
    hard_c5_720p_query_centroid_norm_attention_triton_blocks,
)
from attention.singleton_c4_wan21_14b_720p_triton import (
    hard_c4_720p_singleton_leverage_top3_exact_intra_frame_attention_triton_blocks,
)
from attention.token_merging import (
    _hard_head_chunks,
    _interpolate_anchor_grid_residual,
    _uniform_anchor_indices,
    token_merging_attn,
)


class WanHardC4CompatibilityTests(unittest.TestCase):
    def test_dense_head_rescue_preserves_contiguous_c4_chunks(self):
        self.assertEqual(
            _hard_head_chunks(10, 4, dense_heads=(2, 5, 6, 9)),
            [(0, 2), (3, 2), (7, 2)],
        )

    def test_dense_rescue_rejects_non_c4_dispatch(self):
        query = torch.randn(1, 2 * 3 * 4, 1, 4)
        with self.assertRaisesRegex(ValueError, "only for Hard C4"):
            token_merging_attn(
                query,
                query,
                query,
                f_tied=1,
                h_reduce=1,
                w_reduce=1,
                h=3,
                w=4,
                num_merge_tokens=5,
                dense_heads=(0,),
            )

    def test_wide_seed_reference_matches_720p_kernel_boundaries(self):
        seed_rows = wide_local_c4_seed_rows(
            torch.arange(1),
            torch.arange(1),
            key_rows=45,
            query_columns=1,
            query_rows=45,
        )[0, 0, :, 0]

        torch.testing.assert_close(seed_rows[0], torch.tensor([0, 1, 2, 3]))
        torch.testing.assert_close(seed_rows[2], torch.tensor([0, 2, 3, 5]))
        torch.testing.assert_close(seed_rows[41], torch.tensor([39, 41, 42, 44]))
        torch.testing.assert_close(seed_rows[42], torch.tensor([41, 42, 43, 44]))

    def test_anchor_grid_reconstructs_a_linear_residual_field(self):
        frames, rows = 5, 7
        time_anchors = _uniform_anchor_indices(frames, 3, torch.device("cpu"))
        row_anchors = _uniform_anchor_indices(rows, 4, torch.device("cpu"))
        anchor_residual = (
            time_anchors.float().view(1, -1, 1, 1, 1, 1)
            + 2.0 * row_anchors.float().view(1, 1, -1, 1, 1, 1)
        )
        actual = _interpolate_anchor_grid_residual(
            anchor_residual,
            time_anchors,
            row_anchors,
            frames=frames,
            rows=rows,
        )
        expected = (
            torch.arange(frames).float().view(1, -1, 1, 1, 1, 1)
            + 2.0 * torch.arange(rows).float().view(1, 1, -1, 1, 1, 1)
        )
        torch.testing.assert_close(actual, expected)

    def test_anchor_grid_reconstructs_a_trilinear_residual_field(self):
        frames, rows, width = 5, 7, 9
        time_anchors = _uniform_anchor_indices(frames, 3, torch.device("cpu"))
        row_anchors = _uniform_anchor_indices(rows, 4, torch.device("cpu"))
        width_anchors = _uniform_anchor_indices(width, 5, torch.device("cpu"))
        anchor_residual = (
            time_anchors.float().view(1, -1, 1, 1, 1, 1)
            + 2.0 * row_anchors.float().view(1, 1, -1, 1, 1, 1)
            - 0.5 * width_anchors.float().view(1, 1, 1, -1, 1, 1)
        )
        actual = _interpolate_anchor_grid_residual(
            anchor_residual,
            time_anchors,
            row_anchors,
            frames=frames,
            rows=rows,
            width_anchors=width_anchors,
            width=width,
        )
        expected = (
            torch.arange(frames).float().view(1, -1, 1, 1, 1, 1)
            + 2.0 * torch.arange(rows).float().view(1, 1, -1, 1, 1, 1)
            - 0.5 * torch.arange(width).float().view(1, 1, 1, -1, 1, 1)
        )
        torch.testing.assert_close(actual, expected)

    def test_anchor_grid_requires_both_axis_counts(self):
        query = torch.randn(1, 2 * 3 * 4, 1, 4)
        with self.assertRaisesRegex(ValueError, "both time and row counts"):
            token_merging_attn(
                query,
                query,
                query,
                f_tied=1,
                h_reduce=1,
                w_reduce=1,
                h=3,
                w=4,
                query_outer_chunk=1,
                num_merge_tokens=4,
                hard_route_backend="sdpa",
                full_attention_anchor_time_count=2,
            )

    def test_width_anchors_require_time_and_row_anchors(self):
        query = torch.randn(1, 2 * 3 * 4, 1, 4)
        with self.assertRaisesRegex(ValueError, "width anchors require"):
            token_merging_attn(
                query,
                query,
                query,
                f_tied=1,
                h_reduce=1,
                w_reduce=1,
                h=3,
                w=4,
                query_outer_chunk=1,
                num_merge_tokens=4,
                hard_route_backend="sdpa",
                full_attention_anchor_width_count=2,
            )

    def test_720p_tensorized_loop_schedule(self):
        self.assertEqual(HARD_720_C * HARD_720_MERGE_BLOCK_J, 64)
        self.assertEqual(HARD_720_MERGE_BLOCK_L, 128)
        self.assertEqual(HARD_720_MERGE_NUM_WARPS, 4)
        self.assertEqual(HARD_720_MERGE_NUM_STAGES, 1)
        self.assertEqual(HARD_720_MERGE_LOOP_STAGES, 2)
        self.assertEqual(HARD_720_QUERY_OUTER_GROUP, 21)
        self.assertEqual(HARD_720_ROUTE_NUM_STAGES, 1)
        self.assertEqual(HARD_720_CENTROID_CANDIDATES, 11)

    def test_720p_shape_dispatches_to_specialized_kernel(self):
        query = torch.randn(1, 1, 45, 80, 1, 4)
        expected = torch.full_like(query, 7.0)

        with mock.patch(
            "attention.hard_c4_wan21_14b_720p_triton.hard_c4_720p_attention_triton_blocks",
            return_value=expected,
        ) as kernel_720p:
            actual = hard_c4_rt_attention_triton_blocks(query, query, query)

        kernel_720p.assert_called_once()
        self.assertIs(actual, expected)

    def test_720p_kernel_rejects_partial_temporal_groups(self):
        query = torch.empty(1, 19, 45, 80, 1, 4)
        with self.assertRaisesRegex(ValueError, "divisible by 21"):
            hard_c4_720p_attention_triton_blocks(query, query, query)

    def test_720p_kernel_forwards_an_independent_seed_query(self):
        query = torch.randn(1, 21, 45, 80, 1, 4)
        seed_query = torch.randn_like(query)
        output = torch.empty_like(query)

        with mock.patch(
            "attention.hard_c4_wan21_14b_720p_triton._run_hard_c4_720p",
            return_value=output,
        ) as run_kernel:
            actual = hard_c4_720p_attention_triton_blocks(
                query, query, query, output=output, seed_q=seed_query
            )

        self.assertIs(actual, output)
        run_kernel.assert_called_once_with(
            query,
            query,
            query,
            seed_q=seed_query,
            scale=None,
            output=output,
            head_start=0,
            head_count=None,
        )

    def test_separate_seed_query_rejects_reference_backend(self):
        query = torch.randn(1, 2 * 3 * 4, 1, 4)
        with self.assertRaisesRegex(ValueError, "separate seed_query requires"):
            token_merging_attn(
                query,
                query,
                query,
                f_tied=1,
                h_reduce=1,
                w_reduce=1,
                h=3,
                w=4,
                query_outer_chunk=1,
                num_merge_tokens=4,
                hard_route_backend="sdpa",
                seed_query=query.clone(),
            )

    def test_720p_no_entropy_ablation_keeps_only_bias_flag_different(self):
        query = torch.randn(1, 2, 12, 128)
        output = torch.empty_like(query)

        with mock.patch(
            "attention.hard_c4_wan21_14b_720p_triton._run_hard_c4_720p",
            return_value=output,
        ) as run_kernel:
            actual = hard_c4_720p_attention_triton_head_major_no_entropy_bias(
                query,
                query,
                query,
                output=output,
                head_start=2,
                head_count=4,
            )

        self.assertIs(actual, output)
        run_kernel.assert_called_once_with(
            query,
            query,
            query,
            scale=None,
            output=output,
            head_start=2,
            head_count=4,
            head_major=True,
            use_entropy_bias=False,
        )

    def test_720p_token_output_keeps_only_output_layout_different(self):
        query = torch.randn(1, 2, 12, 128)
        output = torch.empty(1, 12, 256)

        with mock.patch(
            "attention.hard_c4_wan21_14b_720p_triton._run_hard_c4_720p",
            return_value=output,
        ) as run_kernel:
            actual = hard_c4_720p_attention_triton_head_major_to_token(
                query,
                query,
                query,
                output=output,
                head_start=2,
                head_count=4,
            )

        self.assertIs(actual, output)
        run_kernel.assert_called_once_with(
            query,
            query,
            query,
            scale=None,
            output=output,
            head_start=2,
            head_count=4,
            head_major=True,
            output_head_major=False,
        )

    def test_exact_intra_frame_wrapper_enables_only_the_route_variant(self):
        query = torch.randn(1, 2, 12, 128)
        output = torch.empty_like(query)

        with mock.patch(
            "attention.hard_c4_wan21_14b_720p_triton._run_hard_c4_720p",
            return_value=output,
        ) as run_kernel:
            actual = hard_c4_720p_exact_intra_frame_attention_triton_blocks(
                query,
                query,
                query,
                output=output,
                head_start=0,
                head_count=1,
                query_outer_offset=0,
            )

        self.assertIs(actual, output)
        run_kernel.assert_called_once_with(
            query,
            query,
            query,
            scale=None,
            output=output,
            head_start=0,
            head_count=1,
            exact_intra_frame=True,
            exact_neighbor_window_columns=0,
            query_outer_offset=0,
        )

    def test_exact_query_wrapper_enables_only_three_row_refinement(self):
        query = torch.randn(1, 2, 12, 128)
        output = torch.empty_like(query)

        with mock.patch(
            "attention.hard_c4_wan21_14b_720p_triton._run_hard_c4_720p",
            return_value=output,
        ) as run_kernel:
            actual = hard_c4_720p_exact_query_top3_attention_triton_blocks(
                query,
                query,
                query,
                output=output,
                head_start=0,
                head_count=1,
            )

        self.assertIs(actual, output)
        run_kernel.assert_called_once_with(
            query,
            query,
            query,
            scale=None,
            output=output,
            head_start=0,
            head_count=1,
            exact_query_rows=3,
        )

    def test_query_centroid_wrapper_enables_only_p11_seed_preprocessing(self):
        query = torch.randn(1, 2, 12, 128)
        output = torch.empty_like(query)

        with mock.patch(
            "attention.hard_c4_wan21_14b_720p_triton._run_hard_c4_720p",
            return_value=output,
        ) as run_kernel:
            actual = hard_c4_720p_query_centroid_attention_triton_blocks(
                query,
                query,
                query,
                output=output,
                head_start=1,
                head_count=1,
            )

        self.assertIs(actual, output)
        run_kernel.assert_called_once_with(
            query,
            query,
            query,
            scale=None,
            output=output,
            head_start=1,
            head_count=1,
            query_centroid_window=11,
        )

    def test_query_centroid_seed_mode_dispatches_to_specialized_kernel(self):
        query = torch.randn(1, 21 * 45 * 80, 1, 4)

        def fill_output(*args, output, **kwargs):
            output.fill_(3.0)
            return output

        with mock.patch(
            "attention.token_merging."
            "hard_c4_720p_query_centroid_attention_triton_blocks",
            side_effect=fill_output,
        ) as centroid_kernel:
            actual = token_merging_attn(
                query,
                query,
                query,
                f_tied=1,
                h_reduce=1,
                w_reduce=1,
                h=45,
                w=80,
                query_outer_chunk=21,
                head_chunk=1,
                num_merge_tokens=4,
                seed_mode="local_centroid_p11",
                hard_route_backend="hard_c4",
            )

        centroid_kernel.assert_called_once()
        torch.testing.assert_close(actual, torch.full_like(actual, 3.0))

    def test_query_centroid_seed_mode_requires_native_triton_path(self):
        query = torch.randn(1, 45 * 80, 1, 4)
        with self.assertRaisesRegex(ValueError, "plain 45x80 Triton"):
            token_merging_attn(
                query,
                query,
                query,
                f_tied=1,
                h_reduce=1,
                w_reduce=1,
                h=45,
                w=80,
                query_outer_chunk=1,
                num_merge_tokens=4,
                seed_mode="local_centroid_p11",
                hard_route_backend="sdpa",
            )

    def test_query_response_wrapper_enables_selector_and_bias_scale(self):
        query = torch.randn(1, 21, 45, 80, 1, 128)
        output = torch.empty_like(query)

        with mock.patch(
            "attention.hard_c4_wan21_14b_720p_triton._run_hard_c4_720p",
            return_value=output,
        ) as run_kernel:
            actual = hard_c4_720p_query_response_attention_triton_blocks(
                query,
                query,
                query,
                output=output,
                head_start=0,
                head_count=1,
            )

        self.assertIs(actual, output)
        run_kernel.assert_called_once_with(
            query,
            query,
            query,
            scale=None,
            output=output,
            head_start=0,
            head_count=1,
            query_response_seeds=True,
            route_bias_scale=1.2,
        )

    def test_query_response_seed_mode_dispatches_to_specialized_kernel(self):
        query = torch.randn(1, 21 * 45 * 80, 1, 4)

        def fill_output(*args, output, **kwargs):
            output.fill_(5.0)
            return output

        with mock.patch(
            "attention.token_merging."
            "hard_c4_720p_query_response_attention_triton_blocks",
            side_effect=fill_output,
        ) as response_kernel:
            actual = token_merging_attn(
                query,
                query,
                query,
                f_tied=1,
                h_reduce=1,
                w_reduce=1,
                h=45,
                w=80,
                query_outer_chunk=21,
                head_chunk=1,
                num_merge_tokens=4,
                seed_mode="query_response_span15_s8",
                hard_route_backend="hard_c4",
            )

        response_kernel.assert_called_once()
        torch.testing.assert_close(actual, torch.full_like(actual, 5.0))

    def test_responsibility_centroid_wrapper_enables_top_two_refinements(self):
        query = torch.randn(1, 2, 12, 128)
        output = torch.empty_like(query)

        with mock.patch(
            "attention.hard_c4_wan21_14b_720p_triton._run_hard_c4_720p",
            return_value=output,
        ) as run_kernel:
            actual = hard_c4_720p_responsibility_centroid_attention_triton_blocks(
                query,
                query,
                query,
                output=output,
                head_start=1,
                head_count=1,
            )

        self.assertIs(actual, output)
        run_kernel.assert_called_once_with(
            query,
            query,
            query,
            scale=None,
            output=output,
            head_start=1,
            head_count=1,
            responsibility_top_r=2,
        )

    def test_responsibility_centroid_seed_mode_dispatches_to_specialized_kernel(self):
        query = torch.randn(1, 21 * 45 * 80, 1, 4)

        def fill_output(*args, output, **kwargs):
            output.fill_(4.0)
            return output

        with mock.patch(
            "attention.token_merging."
            "hard_c4_720p_responsibility_centroid_attention_triton_blocks",
            side_effect=fill_output,
        ) as centroid_kernel:
            actual = token_merging_attn(
                query,
                query,
                query,
                f_tied=1,
                h_reduce=1,
                w_reduce=1,
                h=45,
                w=80,
                query_outer_chunk=21,
                head_chunk=1,
                num_merge_tokens=4,
                seed_mode="responsibility_centroid_top2",
                hard_route_backend="hard_c4",
            )

        centroid_kernel.assert_called_once()
        torch.testing.assert_close(actual, torch.full_like(actual, 4.0))

    def test_hierarchical_mean_split_dispatches_to_specialized_kernel(self):
        query = torch.randn(1, 21 * 45 * 80, 1, 4)

        def fill_output(*args, output, **kwargs):
            output.fill_(6.0)
            return output

        with mock.patch(
            "attention.token_merging."
            "hard_c4_720p_hierarchical_mean_split_attention_triton_blocks",
            side_effect=fill_output,
        ) as split_kernel:
            actual = token_merging_attn(
                query,
                query,
                query,
                f_tied=1,
                h_reduce=1,
                w_reduce=1,
                h=45,
                w=80,
                query_outer_chunk=21,
                head_chunk=1,
                num_merge_tokens=4,
                seed_mode="hierarchical_mean_split_top2",
                hard_route_backend="hard_c4",
            )

        split_kernel.assert_called_once()
        torch.testing.assert_close(actual, torch.full_like(actual, 6.0))

    def test_singleton_leverage_dispatches_to_specialized_kernel(self):
        query = torch.randn(1, 21 * 45 * 80, 1, 4)

        def fill_output(*args, output, **kwargs):
            output.fill_(7.0)
            return output

        with mock.patch(
            "attention.token_merging."
            "hard_c4_720p_singleton_leverage_attention_triton_blocks",
            side_effect=fill_output,
        ) as singleton_kernel:
            actual = token_merging_attn(
                query,
                query,
                query,
                f_tied=1,
                h_reduce=1,
                w_reduce=1,
                h=45,
                w=80,
                query_outer_chunk=21,
                head_chunk=1,
                num_merge_tokens=4,
                seed_mode="singleton_leverage_top2",
                hard_route_backend="hard_c4",
            )

        singleton_kernel.assert_called_once()
        torch.testing.assert_close(actual, torch.full_like(actual, 7.0))

    def test_singleton_leverage_top6_dispatches_to_specialized_kernel(self):
        query = torch.randn(1, 21 * 45 * 80, 1, 4)

        def fill_output(*args, output, **kwargs):
            output.fill_(8.0)
            return output

        with mock.patch(
            "attention.token_merging."
            "hard_c4_720p_singleton_leverage_top6_attention_triton_blocks",
            side_effect=fill_output,
        ) as singleton_kernel:
            actual = token_merging_attn(
                query,
                query,
                query,
                f_tied=1,
                h_reduce=1,
                w_reduce=1,
                h=45,
                w=80,
                query_outer_chunk=21,
                head_chunk=1,
                num_merge_tokens=4,
                seed_mode="singleton_leverage_top6",
                hard_route_backend="hard_c4",
            )

        singleton_kernel.assert_called_once()
        torch.testing.assert_close(actual, torch.full_like(actual, 8.0))

    def test_same_frame_exact_singleton_top3_dispatches_to_combined_kernel(self):
        query = torch.randn(1, 21 * 45 * 80, 1, 4)

        def fill_output(q, _k, _v, _scale, output=None, **kwargs):
            self.assertEqual(q.size(1), 21)
            self.assertEqual(kwargs["query_outer_offset"], 0)
            output.fill_(8.5)
            return output

        with mock.patch(
            "attention.token_merging."
            "hard_c4_720p_singleton_leverage_top3_exact_intra_frame_attention_triton_blocks",
            side_effect=fill_output,
        ) as combined_kernel:
            actual = token_merging_attn(
                query,
                query,
                query,
                f_tied=1,
                h_reduce=1,
                w_reduce=1,
                h=45,
                w=80,
                query_outer_chunk=21,
                head_chunk=1,
                num_merge_tokens=4,
                seed_mode="singleton_leverage_top3",
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=True,
            )

        combined_kernel.assert_called_once()
        torch.testing.assert_close(actual, torch.full_like(actual, 8.5))

    def test_same_frame_exact_singleton_top3_wrapper_sets_joint_route(self):
        query = torch.randn(1, 21, 45, 80, 1, 128)
        output = torch.empty_like(query)

        with mock.patch(
            "attention.singleton_c4_wan21_14b_720p_triton."
            "_hard_c4_720p_singleton_leverage_attention_triton_blocks",
            return_value=output,
        ) as run_kernel:
            actual = (
                hard_c4_720p_singleton_leverage_top3_exact_intra_frame_attention_triton_blocks(
                    query,
                    query,
                    query,
                    output=output,
                    query_outer_offset=0,
                )
            )

        self.assertIs(actual, output)
        run_kernel.assert_called_once_with(
            query,
            query,
            query,
            top_r=3,
            scale=None,
            output=output,
            head_start=0,
            head_count=None,
            exact_intra_frame=True,
            query_outer_offset=0,
        )

    def test_singleton_leverage_top16_dispatches_budget_to_generic_kernel(self):
        query = torch.randn(1, 21 * 45 * 80, 1, 4)

        def fill_output(*args, output, **kwargs):
            self.assertEqual(kwargs["top_r"], 16)
            output.fill_(9.0)
            return output

        with mock.patch(
            "attention.token_merging."
            "hard_c4_720p_singleton_leverage_attention_triton_blocks",
            side_effect=fill_output,
        ) as singleton_kernel:
            actual = token_merging_attn(
                query,
                query,
                query,
                f_tied=1,
                h_reduce=1,
                w_reduce=1,
                h=45,
                w=80,
                query_outer_chunk=21,
                head_chunk=1,
                num_merge_tokens=4,
                seed_mode="singleton_leverage_top16",
                hard_route_backend="hard_c4",
            )

        singleton_kernel.assert_called_once()
        torch.testing.assert_close(actual, torch.full_like(actual, 9.0))

    def test_c5_norm_centroid_wrapper_enables_only_p11_preprocessing(self):
        query = torch.randn(1, 2, 12, 128)
        output = torch.empty_like(query)

        with mock.patch(
            "attention.hard_c5_c6_wan21_14b_720p_triton._hard_c_attention",
            return_value=output,
        ) as run_kernel:
            actual = hard_c5_720p_query_centroid_norm_attention_triton_blocks(
                query,
                query,
                query,
                output=output,
                head_start=1,
                head_count=1,
            )

        self.assertIs(actual, output)
        run_kernel.assert_called_once_with(
            query,
            query,
            query,
            5,
            None,
            output,
            1,
            1,
            head_major=False,
            query_centroid_norm_window=CENTROID_CANDIDATES,
        )

    def test_c5_norm_centroid_seed_mode_dispatches_to_specialized_kernel(self):
        query = torch.randn(1, 21 * 45 * 80, 1, 4)

        def fill_output(*args, output, **kwargs):
            output.fill_(5.0)
            return output

        with mock.patch(
            "attention.token_merging."
            "hard_c5_720p_query_centroid_norm_attention_triton_blocks",
            side_effect=fill_output,
        ) as centroid_kernel:
            actual = token_merging_attn(
                query,
                query,
                query,
                f_tied=1,
                h_reduce=1,
                w_reduce=1,
                h=45,
                w=80,
                query_outer_chunk=21,
                head_chunk=1,
                num_merge_tokens=5,
                seed_mode="local_centroid_norm_p11",
                hard_route_backend="hard_c4",
            )

        centroid_kernel.assert_called_once()
        torch.testing.assert_close(actual, torch.full_like(actual, 5.0))

    def test_winner_probability_reuse_matches_cluster_softmax(self):
        """A routed seed's global log-normalizer cancels inside its cluster."""

        torch.manual_seed(20260814)
        scores = torch.randn(2, 9, 4, 13, dtype=torch.float32)
        # Exact all-seed ties route left to seed 0 and force seeds 1..3 to be
        # empty for this row, covering both maintained edge cases explicitly.
        scores[0, 0].zero_()
        seed_logprob = torch.log_softmax(scores, dim=-1)
        route = seed_logprob.argmax(dim=-2)
        self.assertTrue(
            torch.equal(route[0, 0], torch.zeros(13, dtype=torch.long))
        )
        winner_logprob = torch.gather(
            seed_logprob, -2, route.unsqueeze(-2)
        ).squeeze(-2)

        for seed_id in range(4):
            with self.subTest(seed_id=seed_id):
                route_mask = route == seed_id
                has_key = route_mask.any(dim=-1)
                seed_scores = scores[:, :, seed_id]

                masked_scores = seed_scores.masked_fill(
                    ~route_mask, -float("inf")
                )
                row_max = masked_scores.amax(dim=-1)
                safe_max = torch.where(has_key, row_max, 0.0)
                old_probs = torch.where(
                    route_mask,
                    torch.exp(seed_scores - safe_max.unsqueeze(-1)),
                    0.0,
                )
                old_sum = old_probs.sum(dim=-1)
                old_safe_sum = torch.where(has_key, old_sum, 1.0)
                old_weights = old_probs / old_safe_sum.unsqueeze(-1)
                old_bias = (old_weights * seed_scores).sum(dim=-1) - (
                    safe_max + old_safe_sum.log()
                )
                old_bias = torch.where(
                    has_key, old_bias, float("inf")
                )

                reused_probs = torch.where(
                    route_mask, winner_logprob.exp(), 0.0
                )
                reused_sum = reused_probs.sum(dim=-1)
                self.assertTrue(torch.all(reused_sum[has_key] > 0.0))
                reused_safe_sum = torch.where(has_key, reused_sum, 1.0)
                reused_weights = reused_probs / reused_safe_sum.unsqueeze(-1)
                reused_bias = (
                    reused_weights * winner_logprob
                ).sum(dim=-1) - reused_safe_sum.log()
                reused_bias = torch.where(
                    has_key, reused_bias, float("inf")
                )

                torch.testing.assert_close(
                    reused_weights, old_weights, rtol=2e-6, atol=2e-7
                )
                self.assertTrue(
                    torch.equal(
                        torch.isfinite(reused_bias), torch.isfinite(old_bias)
                    )
                )
                torch.testing.assert_close(
                    reused_bias[has_key],
                    old_bias[has_key],
                    rtol=2e-6,
                    atol=2e-7,
                )

    def test_entropy_expectation_ignores_zero_probability_masked_keys(self):
        probabilities = torch.tensor([0.0, 0.25, 0.75])
        log_probabilities = torch.tensor([-float("inf"), -2.0, -0.5])
        safe_log_probabilities = torch.where(
            probabilities > 0, log_probabilities, 0.0
        )
        expectation = (probabilities * safe_log_probabilities).sum()

        self.assertTrue(torch.isfinite(expectation))
        torch.testing.assert_close(expectation, torch.tensor(-0.875))

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

    def test_first_frame_recomputation_overwrites_only_first_query_frame(self):
        torch.manual_seed(20260819)
        batch, frames, height, width, heads, dim = 1, 2, 3, 4, 2, 4
        shape = (batch, frames * height * width, heads, dim)
        query = torch.randn(shape)
        key = torch.randn(shape)
        value = torch.randn(shape)
        scale = dim**-0.5

        baseline = token_merging_attn(
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
            num_merge_tokens=4,
            hard_route_backend="sdpa",
        )
        recomputed = token_merging_attn(
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
            num_merge_tokens=4,
            hard_route_backend="sdpa",
            first_frame_full_attention=True,
        )
        frame_tokens = height * width
        expected_first = flash_attention(
            query[:, :frame_tokens],
            key,
            value,
            softmax_scale=scale,
        )

        torch.testing.assert_close(
            recomputed[:, :frame_tokens], expected_first, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            recomputed[:, frame_tokens:],
            baseline[:, frame_tokens:],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_first_frame_recomputation_composes_with_dense_head_rescue(self):
        frames, height, width, heads, dim = 21, 45, 80, 2, 2
        frame_tokens = height * width
        query = torch.zeros(1, frames * frame_tokens, heads, dim)

        def fill_c4(_q, _k, _v, _scale, output=None, **_kwargs):
            output.fill_(3.0)
            return output

        def fill_dense(q, _k, _v, **_kwargs):
            value = 9.0 if q.size(1) == frame_tokens else 5.0
            return torch.full_like(q, value)

        with mock.patch(
            "attention.token_merging.hard_c4_rt_attention_triton_blocks",
            side_effect=fill_c4,
        ), mock.patch(
            "attention.token_merging.flash_attention", side_effect=fill_dense
        ):
            actual = token_merging_attn(
                query,
                query,
                query,
                f_tied=1,
                h_reduce=1,
                w_reduce=1,
                h=height,
                w=width,
                query_outer_chunk=frames,
                head_chunk=1,
                num_merge_tokens=4,
                hard_route_backend="hard_c4",
                dense_heads=(1,),
                first_frame_full_attention=True,
            )

        torch.testing.assert_close(
            actual[:, :frame_tokens], torch.full_like(actual[:, :frame_tokens], 9.0)
        )
        torch.testing.assert_close(
            actual[:, frame_tokens:, 0],
            torch.full_like(actual[:, frame_tokens:, 0], 3.0),
        )
        torch.testing.assert_close(
            actual[:, frame_tokens:, 1],
            torch.full_like(actual[:, frame_tokens:, 1], 5.0),
        )

    def test_dispatcher_preserves_fused_hard_c4_backend_contract(self):
        """Default and canonical C4 labels must select main's fused kernel."""

        query = torch.randn(1, 2 * 6 * 7, 1, 4)

        def fill_fused_output(q, _k, _v, _scale, output=None, **_kwargs):
            self.assertIsNotNone(output)
            output.fill_(3.25)
            return output

        backend_cases = {
            "default": {},
            "canonical": {"hard_route_backend": "hard_c4"},
        }
        for case_name, backend_options in backend_cases.items():
            with self.subTest(case=case_name):
                with mock.patch(
                    "attention.token_merging.hard_c4_rt_attention_triton_blocks",
                    side_effect=fill_fused_output,
                ) as triton_backend:
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
                        **backend_options,
                    )

                self.assertEqual(triton_backend.call_count, 2)
                torch.testing.assert_close(
                    actual, torch.full_like(actual, 3.25)
                )

    def test_exact_intra_frame_dispatches_all_21_frames_together(self):
        query = torch.randn(1, 21 * 45 * 80, 1, 4)

        def fill_output(q, _k, _v, _scale, output=None, **kwargs):
            self.assertEqual(q.size(1), 21)
            self.assertEqual(kwargs["query_outer_offset"], 0)
            output.fill_(4.5)
            return output

        with mock.patch(
            "attention.token_merging."
            "hard_c4_720p_exact_intra_frame_attention_triton_blocks",
            side_effect=fill_output,
        ) as exact_kernel:
            actual = token_merging_attn(
                query,
                query,
                query,
                f_tied=1,
                h_reduce=1,
                w_reduce=1,
                h=45,
                w=80,
                query_outer_chunk=21,
                num_merge_tokens=4,
                hard_route_backend="hard_c4",
                head_chunk=1,
                exact_intra_frame_attention=True,
            )

        exact_kernel.assert_called_once()
        torch.testing.assert_close(actual, torch.full_like(actual, 4.5))

    def test_exact_neighbor_windows_reach_the_specialized_kernel(self):
        query = torch.randn(1, 21 * 45 * 80, 1, 4)

        def fill_output(q, _k, _v, _scale, output=None, **kwargs):
            self.assertEqual(q.size(1), 21)
            self.assertEqual(kwargs["exact_neighbor_window_columns"], 30)
            output.fill_(5.5)
            return output

        with mock.patch(
            "attention.token_merging."
            "hard_c4_720p_exact_intra_frame_attention_triton_blocks",
            side_effect=fill_output,
        ) as exact_kernel:
            actual = token_merging_attn(
                query,
                query,
                query,
                f_tied=1,
                h_reduce=1,
                w_reduce=1,
                h=45,
                w=80,
                query_outer_chunk=21,
                num_merge_tokens=4,
                hard_route_backend="hard_c4",
                head_chunk=1,
                exact_intra_frame_attention=True,
                exact_neighbor_window_columns=30,
            )

        exact_kernel.assert_called_once()
        torch.testing.assert_close(actual, torch.full_like(actual, 5.5))

    def test_exact_query_rows_dispatch_to_budgeted_kernel(self):
        query = torch.randn(1, 21 * 45 * 80, 1, 4)

        def fill_output(q, _k, _v, _scale, output=None, **_kwargs):
            self.assertEqual(q.size(1), 21)
            output.fill_(6.5)
            return output

        with mock.patch(
            "attention.token_merging."
            "hard_c4_720p_exact_query_top3_attention_triton_blocks",
            side_effect=fill_output,
        ) as exact_kernel:
            actual = token_merging_attn(
                query,
                query,
                query,
                f_tied=1,
                h_reduce=1,
                w_reduce=1,
                h=45,
                w=80,
                query_outer_chunk=21,
                num_merge_tokens=4,
                hard_route_backend="hard_c4",
                head_chunk=1,
                exact_query_rows=3,
            )

        exact_kernel.assert_called_once()
        torch.testing.assert_close(actual, torch.full_like(actual, 6.5))

    def test_exact_neighbor_window_budget_is_bounded(self):
        query = torch.randn(1, 12, 1, 4)
        with self.assertRaisesRegex(ValueError, "0 to 30"):
            token_merging_attn(
                query,
                query,
                query,
                f_tied=1,
                h_reduce=1,
                w_reduce=1,
                h=3,
                w=4,
                exact_intra_frame_attention=True,
                exact_neighbor_window_columns=31,
            )

    def test_exact_intra_frame_rejects_other_full_attention_corrections(self):
        query = torch.randn(1, 12, 1, 4)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            token_merging_attn(
                query,
                query,
                query,
                f_tied=1,
                h_reduce=1,
                w_reduce=1,
                h=3,
                w=4,
                exact_intra_frame_attention=True,
                first_frame_full_attention=True,
            )

    def test_h_reduced_hard_c4_dispatches_finer_gma_grids(self):
        query = torch.randn(1, 21 * 45 * 80, 2, 4)

        def fill_grid_output(
            _q,
            _k,
            _v,
            *,
            query_block_rows,
            key_block_cols,
            output,
            head_start,
            head_count,
            **_kwargs,
        ):
            self.assertEqual(key_block_cols, 80)
            output[:, head_start : head_start + head_count].fill_(
                float(query_block_rows)
            )
            return output

        for h_reduce, query_block_rows in ((2, 24), (4, 12)):
            with self.subTest(h_reduce=h_reduce):
                with mock.patch(
                    "attention.token_merging."
                    "hard_c4_grid_720p_attention_triton_head_major",
                    side_effect=fill_grid_output,
                ) as grid_backend:
                    actual = token_merging_attn(
                        query,
                        query,
                        query,
                        f_tied=1,
                        h_reduce=h_reduce,
                        w_reduce=1,
                        h=45,
                        w=80,
                        query_outer_chunk=21,
                        num_merge_tokens=4,
                        hard_route_backend="hard_c4",
                        head_chunk=1,
                    )

                self.assertEqual(grid_backend.call_count, 2)
                torch.testing.assert_close(
                    actual, torch.full_like(actual, float(query_block_rows))
                )

    def test_monarch_rt_head_chunk_preserves_output(self):
        query = torch.randn(1, 2 * 3 * 4, 3, 4)

        def add_one(q, _k, _v, _scale):
            self.assertEqual(q.size(-2), 1)
            return q + 1

        with mock.patch(
            "attention.token_merging.monarch_rt_attention_triton_blocks",
            side_effect=add_one,
        ) as monarch_kernel:
            actual = token_merging_attn(
                query,
                query,
                query,
                f_tied=1,
                h_reduce=1,
                w_reduce=1,
                h=3,
                w=4,
                query_outer_chunk=2,
                num_merge_tokens=1,
                q_init="ith",
                hard_route_backend="monarch_rt",
                head_chunk=1,
            )

        self.assertEqual(monarch_kernel.call_count, 3)
        torch.testing.assert_close(actual, query + 1)

    def test_monarch_rt_h_reduce_padding_passes_key_valid_mask(self):
        query = torch.randn(1, 3 * 4, 1, 4)

        def preserve_query(q, _k, _v, _scale, *, key_valid):
            self.assertEqual(tuple(key_valid.shape), (1, 2, 2, 4))
            self.assertEqual(key_valid.dtype, torch.bool)
            self.assertEqual(key_valid.sum().item(), 12)
            self.assertFalse(key_valid[0, 1, 1].any())
            return q

        with mock.patch(
            "attention.token_merging.monarch_rt_attention_triton_blocks",
            side_effect=preserve_query,
        ) as monarch_kernel:
            actual = token_merging_attn(
                query,
                query,
                query,
                f_tied=1,
                h_reduce=2,
                w_reduce=1,
                h=3,
                w=4,
                query_outer_chunk=2,
                num_merge_tokens=1,
                q_init="ith",
                hard_route_backend="monarch_rt",
                head_chunk=1,
            )

        monarch_kernel.assert_called_once()
        torch.testing.assert_close(actual, query)

    def test_dispatcher_supports_hard_c6_and_c8(self):
        query = torch.randn(1, 2 * 6 * 7, 1, 4)

        def fill_output(q, _k, _v, _scale, output=None, **_kwargs):
            self.assertIsNotNone(output)
            output.fill_(6.5)
            return output

        kernels = {
            6: "attention.token_merging.hard_c6_720p_attention_triton_blocks",
            8: "attention.token_merging.hard_c8_720p_attention_triton_blocks",
        }
        for merge_tokens, kernel_path in kernels.items():
            with self.subTest(num_merge_tokens=merge_tokens):
                with mock.patch(kernel_path, side_effect=fill_output) as kernel:
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
                        num_merge_tokens=merge_tokens,
                        hard_route_backend="hard_c4",
                    )

                self.assertEqual(kernel.call_count, 2)
                torch.testing.assert_close(actual, torch.full_like(actual, 6.5))

    def test_removed_wan_c2_path_stays_rejected(self):
        """Unsupported multi-seed counts remain rejected."""

        query = torch.randn(1, 6 * 7, 1, 4)
        with self.assertRaisesRegex(ValueError, "4, 5, 6, or 8"):
            token_merging_attn(
                query,
                query,
                query,
                f_tied=1,
                h_reduce=1,
                w_reduce=1,
                h=6,
                w=7,
                num_merge_tokens=2,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
