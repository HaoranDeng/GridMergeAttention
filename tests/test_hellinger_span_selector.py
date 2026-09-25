import unittest
from unittest.mock import patch

import torch

from attention.adaptive_seed_pytorch import (
    multiscale_candidate_table,
    parse_hellinger_span_seed_mode,
)
from attention.singleton_c4_wan21_14b_720p_triton import (
    hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks,
)
from attention.token_merging import _hard_c4_attention, _hard_c568_attention


class HellingerSpanSelectorTest(unittest.TestCase):
    def test_compressed_fp8_reaches_global_mass_wrapper(self):
        sentinel = object()
        with patch(
            "attention.singleton_c4_wan21_14b_720p_triton."
            "_hard_c4_720p_temporal_shared_global_mass_attention_triton_blocks",
            return_value=sentinel,
        ) as fused:
            output = (
                hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks(
                    None,
                    None,
                    None,
                    None,
                    selector="query_column",
                    compressed_fp8_kv=True,
                )
            )

        self.assertIs(output, sentinel)
        self.assertTrue(fused.call_args.kwargs["compressed_fp8_kv"])

    def test_parse_seed_mode(self):
        self.assertEqual(
            parse_hellinger_span_seed_mode(
                "hellinger_span_alpha0p3_c22_p16_a2_j4"
            ),
            (0.3, 22, 16, 2, 4, 1, 0, 0),
        )
        self.assertEqual(
            parse_hellinger_span_seed_mode(
                "hellinger_span_alpha0p3_c22_p16_a2_j4_g8_r2"
            ),
            (0.3, 22, 16, 2, 4, 8, 2, 0),
        )
        self.assertEqual(
            parse_hellinger_span_seed_mode(
                "hellinger_span_alpha0p3_c22_p16_a2_j4_t3"
            ),
            (0.3, 22, 16, 2, 4, 1, 0, 3),
        )
        self.assertEqual(
            parse_hellinger_span_seed_mode(
                "hellinger_span_alpha0p3_c22_p16_a2_j4_t4"
            ),
            (0.3, 22, 16, 2, 4, 1, 0, 4),
        )
        with self.assertRaises(ValueError):
            parse_hellinger_span_seed_mode(
                "hellinger_span_alpha0p3_c22_p16_a2_j4_g8_r8"
            )
        self.assertIsNone(parse_hellinger_span_seed_mode("local_with_ith"))
        with self.assertRaises(ValueError):
            parse_hellinger_span_seed_mode("hellinger_span_bad")

    def test_multiscale_table_is_unique_and_keeps_anchor(self):
        rows, anchors = multiscale_candidate_table(45, 22, "cpu")
        self.assertEqual(rows.shape, (45, 22))
        self.assertEqual(anchors.shape, (45,))
        self.assertEqual(rows.dtype, torch.int32)
        self.assertTrue(torch.all((rows >= 0) & (rows < 45)))
        self.assertTrue(all(torch.unique(row).numel() == 22 for row in rows))
        key_rows = torch.arange(45, dtype=torch.int32)
        self.assertTrue(
            torch.equal(
                rows.gather(1, anchors.long()[:, None]).squeeze(1),
                key_rows,
            )
        )

    def test_c16_cg22_table_is_a_strict_c22_subset(self):
        compact, anchors = multiscale_candidate_table(
            45, 16, "cpu", geometry_count=22
        )
        full, _ = multiscale_candidate_table(45, 22, "cpu")
        self.assertEqual(compact.shape, (45, 16))
        self.assertTrue(torch.all((compact >= 0) & (compact < 45)))
        self.assertTrue(all(torch.unique(row).numel() == 16 for row in compact))
        self.assertTrue(
            torch.equal(
                compact.gather(1, anchors.long()[:, None]).squeeze(1),
                torch.arange(45, dtype=torch.int32),
            )
        )
        for compact_row, full_row in zip(compact, full):
            self.assertTrue(
                set(compact_row.tolist()).issubset(set(full_row.tolist()))
            )

    def test_routing_heads_dispatch_all_frames_to_optimized_hard_c4(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, frames, frames, rows, columns, 1, 4), dtype=torch.uint8
        )

        def fake_hard_c4(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertEqual(seed_rows.shape, (1, 21, 80, 8, 21, 45, 4))
            kwargs["output"].fill_(1)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ) as selector, patch(
            "attention.token_merging.hard_c4_720p_seed_rows_attention_triton_blocks",
            side_effect=fake_hard_c4,
        ) as fused:
            output = _hard_c4_attention(
                q,
                q,
                q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode="hellinger_span_alpha0p3_c22_p16_a2_j4_g8_r2",
            )

        self.assertTrue(torch.all(output == 1))
        self.assertEqual(selector.call_count, 1)
        self.assertEqual(selector.call_args.kwargs["output_dtype"], torch.uint8)
        self.assertEqual(fused.call_count, 1)

    def test_dynamic_routing_dispatches_singleton_residual_budget(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, frames, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_singleton(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertEqual(seed_rows.shape, (1, 21, 80, 8, 21, 45, 4))
            self.assertEqual(kwargs["top_r"], 4)
            kwargs["output"].fill_(4)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
            side_effect=fake_singleton,
        ) as singleton:
            output = _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode="hellinger_span_alpha0p3_c22_p16_a2_j4_sr4",
            )

        self.assertTrue(torch.all(output == 4))
        self.assertEqual(selector.call_count, 1)
        self.assertEqual(singleton.call_count, 1)

    def test_scaled_fp8_suffix_reaches_singleton_kernel(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, frames, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_singleton(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertEqual(kwargs["top_r"], 1)
            self.assertEqual(kwargs["selector"], "query_column")
            self.assertTrue(kwargs["scaled_fp8_kv"])
            kwargs["output"].fill_(8)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ), patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
            side_effect=fake_singleton,
        ) as singleton:
            output = _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode="hellinger_span_alpha0p3_c22_p16_a2_j4_t1_sq1_fp8",
            )

        self.assertTrue(torch.all(output == 8))
        self.assertEqual(singleton.call_count, 1)

    def test_scaled_fp8_and_fp8_score_suffixes_compose(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, frames, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_singleton(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertEqual(kwargs["top_r"], 1)
            self.assertEqual(kwargs["selector"], "query_column")
            self.assertTrue(kwargs["scaled_fp8_kv"])
            self.assertTrue(kwargs["fp8_score_dot"])
            self.assertTrue(kwargs["query_column_major"])
            self.assertTrue(kwargs["seed_rows_native_layout"])
            self.assertEqual(kwargs["merge_loop_stages"], 2)
            self.assertFalse(kwargs["merge_disable_licm"])
            kwargs["output"].fill_(12)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ), patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
            side_effect=fake_singleton,
        ) as singleton:
            output = _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode=(
                    "hellinger_span_alpha0p3_c22_p16_a2_j4_t1_"
                    "sq1_qm_fp8_fp8score_nl_ml2"
                ),
            )

        self.assertTrue(torch.all(output == 12))
        self.assertEqual(singleton.call_count, 1)

    def test_compressed_fp8_and_fp8_score_suffixes_compose(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, frames, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_singleton(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertEqual(kwargs["top_r"], 1)
            self.assertEqual(kwargs["selector"], "query_column")
            self.assertTrue(kwargs["compressed_fp8_kv"])
            self.assertFalse(kwargs["scaled_fp8_kv"])
            self.assertTrue(kwargs["fp8_score_dot"])
            self.assertTrue(kwargs["split_kv_accum"])
            self.assertTrue(kwargs["query_column_major"])
            self.assertTrue(kwargs["seed_rows_native_layout"])
            kwargs["output"].fill_(13)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ), patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
            side_effect=fake_singleton,
        ) as singleton:
            output = _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode=(
                    "hellinger_span_alpha0p3_c22_p16_a2_j4_t1_"
                    "sq1_qm_cfp8_fp8score_skv_nl_ml2"
                ),
            )

        self.assertTrue(torch.all(output == 13))
        self.assertEqual(singleton.call_count, 1)

    def test_fp8_merge_dot_suffixes_reach_singleton_kernel(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, frames, frames, rows, columns, heads, 4), dtype=torch.uint8
        )
        modes = {
            "fp8mkv": (True, True),
            "fp8mk": (True, False),
            "fp8mv": (False, True),
        }
        for suffix, expected in modes.items():
            with self.subTest(suffix=suffix):
                def fake_singleton(
                    q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs
                ):
                    self.assertEqual(kwargs["top_r"], 1)
                    self.assertEqual(kwargs["selector"], "query_column")
                    self.assertTrue(kwargs["fp8_score_dot"])
                    self.assertTrue(kwargs["split_kv_accum"])
                    self.assertEqual(kwargs["fp8_merge_k_dot"], expected[0])
                    self.assertEqual(kwargs["fp8_merge_v_dot"], expected[1])
                    kwargs["output"].fill_(14)
                    return kwargs["output"]

                with patch(
                    "attention.token_merging.hellinger_span_seed_rows_triton",
                    return_value=router_table,
                ), patch(
                    "attention.token_merging."
                    "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
                    side_effect=fake_singleton,
                ) as singleton:
                    output = _hard_c4_attention(
                        q, q, q,
                        h=rows,
                        w=columns,
                        h_reduce=1,
                        w_reduce=1,
                        sm_scale=1.0,
                        query_outer_chunk=21,
                        head_chunk=8,
                        hard_route_backend="hard_c4",
                        exact_intra_frame_attention=False,
                        exact_neighbor_window_columns=0,
                        exact_query_rows=0,
                        first_frame_full_attention=False,
                        full_attention_anchor_time_count=0,
                        full_attention_anchor_row_count=0,
                        full_attention_anchor_scale=1.0,
                        seed_mode=(
                            "hellinger_span_alpha0p3_c22_p16_a2_j4_t1_"
                            f"sq1_qm_{suffix}_fp8score_skv_nl_ml2"
                        ),
                    )

                self.assertTrue(torch.all(output == 14))
                self.assertEqual(singleton.call_count, 1)

    def test_mean_routing_suffix_reaches_singleton_kernel(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, frames, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_singleton(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertEqual(kwargs["top_r"], 1)
            self.assertEqual(kwargs["selector"], "query_column")
            self.assertTrue(kwargs["mean_centered_routing"])
            self.assertTrue(kwargs["fp8_score_dot"])
            kwargs["output"].fill_(9)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ), patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
            side_effect=fake_singleton,
        ) as singleton:
            output = _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode=(
                    "hellinger_span_alpha0p3_c22_p16_a2_j4_t1_"
                    "sq1_meanroute_fp8score"
                ),
            )

        self.assertTrue(torch.all(output == 9))
        self.assertEqual(singleton.call_count, 1)

    def test_gaussian_routing_suffix_reaches_singleton_kernel(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, frames, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_singleton(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertEqual(kwargs["top_r"], 1)
            self.assertEqual(kwargs["selector"], "query_column")
            self.assertFalse(kwargs["mean_centered_routing"])
            self.assertTrue(kwargs["gaussian_partition_routing"])
            self.assertTrue(kwargs["fp8_score_dot"])
            kwargs["output"].fill_(10)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ), patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
            side_effect=fake_singleton,
        ) as singleton:
            output = _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode=(
                    "hellinger_span_alpha0p3_c22_p16_a2_j4_t1_"
                    "sq1_gaussianroute_fp8score"
                ),
            )

        self.assertTrue(torch.all(output == 10))
        self.assertEqual(singleton.call_count, 1)

    def test_reuse_partition_exp_suffix_reaches_singleton_kernel(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, frames, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_singleton(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertEqual(kwargs["top_r"], 1)
            self.assertEqual(kwargs["selector"], "query_column")
            self.assertTrue(kwargs["reuse_partition_exp"])
            self.assertTrue(kwargs["fp8_score_dot"])
            kwargs["output"].fill_(11)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ), patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
            side_effect=fake_singleton,
        ) as singleton:
            output = _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode=(
                    "hellinger_span_alpha0p3_c22_p16_a2_j4_t1_"
                    "sq1_reuseexp_fp8score"
                ),
            )

        self.assertTrue(torch.all(output == 11))
        self.assertEqual(singleton.call_count, 1)

    def test_direct_bulk_suffix_reaches_singleton_kernel(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, frames, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_singleton(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertEqual(kwargs["top_r"], 1)
            self.assertEqual(kwargs["selector"], "query_column")
            self.assertTrue(kwargs["direct_bulk_normalization"])
            self.assertTrue(kwargs["fp8_score_dot"])
            kwargs["output"].fill_(12)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ), patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
            side_effect=fake_singleton,
        ) as singleton:
            output = _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode=(
                    "hellinger_span_alpha0p3_c22_p16_a2_j4_t1_"
                    "sq1_qm_directbulk_fp8score"
                ),
            )

        self.assertTrue(torch.all(output == 12))
        self.assertEqual(singleton.call_count, 1)

    def test_query_frame_group_suffix_reaches_singleton_kernel(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, frames, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_singleton(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertEqual(kwargs["top_r"], 1)
            self.assertEqual(kwargs["selector"], "query_column")
            self.assertEqual(kwargs["query_frame_group"], 3)
            kwargs["output"].fill_(3)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ), patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
            side_effect=fake_singleton,
        ) as singleton:
            output = _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode=(
                    "hellinger_span_alpha0p3_c22_p16_a2_j4_t1_sq1_qg3"
                ),
            )

        self.assertTrue(torch.all(output == 3))
        self.assertEqual(singleton.call_count, 1)

    def test_split_merge_suffix_reaches_singleton_kernel(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, frames, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_singleton(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertEqual(kwargs["top_r"], 1)
            self.assertEqual(kwargs["selector"], "query_column")
            self.assertTrue(kwargs["split_merge"])
            kwargs["output"].fill_(6)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ), patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
            side_effect=fake_singleton,
        ) as singleton:
            output = _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode=(
                    "hellinger_span_alpha0p3_c22_p16_a2_j4_t1_"
                    "sq1_splitmerge"
                ),
            )

        self.assertTrue(torch.all(output == 6))
        self.assertEqual(singleton.call_count, 1)

    def test_materialized_singleton_suffix_reaches_singleton_kernel(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, frames, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_singleton(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertEqual(kwargs["top_r"], 1)
            self.assertEqual(kwargs["selector"], "query_column")
            self.assertTrue(kwargs["materialize_singleton"])
            kwargs["output"].fill_(7)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ), patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
            side_effect=fake_singleton,
        ) as singleton:
            output = _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode=(
                    "hellinger_span_alpha0p3_c22_p16_a2_j4_t1_sq1_mat"
                ),
            )

        self.assertTrue(torch.all(output == 7))
        self.assertEqual(singleton.call_count, 1)

    def test_packed_singleton_suffix_reaches_fused_kernel(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, 1, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_singleton(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertTrue(kwargs["packed_singleton_kv"])
            self.assertTrue(kwargs["query_column_major"])
            self.assertEqual(kwargs["selector"], "query_column")
            kwargs["output"].fill_(11)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
            side_effect=fake_singleton,
        ) as singleton:
            output = _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode="hellinger_span_alpha0p3_c22_p16_a2_j4_t1_sq1_pack",
            )

        self.assertTrue(torch.all(output == 11))
        self.assertEqual(selector.call_count, 1)
        self.assertEqual(singleton.call_count, 1)
        router_query, router_key = selector.call_args.args[:2]
        self.assertEqual(
            router_query.untyped_storage().data_ptr(), q.untyped_storage().data_ptr()
        )
        self.assertEqual(
            router_key.untyped_storage().data_ptr(), q.untyped_storage().data_ptr()
        )

    def test_query_column_major_suffix_keeps_raw_singletons(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, 1, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_singleton(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertTrue(kwargs["query_column_major"])
            self.assertFalse(kwargs["packed_singleton_kv"])
            self.assertEqual(kwargs["selector"], "query_column")
            kwargs["output"].fill_(13)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ), patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
            side_effect=fake_singleton,
        ) as singleton:
            output = _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode="hellinger_span_alpha0p3_c22_p16_a2_j4_t1_sq1_qm",
            )

        self.assertTrue(torch.all(output == 13))
        self.assertEqual(singleton.call_count, 1)

    def test_native_layout_suffix_passes_selector_table_without_transpose(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, 1, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_singleton(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertIs(seed_rows, router_table)
            self.assertTrue(kwargs["seed_rows_native_layout"])
            self.assertTrue(kwargs["query_column_major"])
            self.assertEqual(kwargs["merge_loop_stages"], 2)
            self.assertFalse(kwargs["merge_disable_licm"])
            self.assertEqual(kwargs["selector"], "query_column")
            kwargs["output"].fill_(19)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ), patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
            side_effect=fake_singleton,
        ) as singleton:
            output = _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode=(
                    "hellinger_span_alpha0p3_c22_p16_a2_j4_t1_sq1_qm_nl_ml2"
                ),
            )

        self.assertTrue(torch.all(output == 19))
        self.assertEqual(singleton.call_count, 1)

    def test_first_frame_full_key_router_dispatches_p80(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        frame_ids = torch.arange(frames, dtype=torch.float32).repeat_interleave(
            rows * columns
        )
        q = frame_ids[None, :, None, None].expand(-1, -1, heads, 1).clone()
        router_table = torch.zeros(
            (1, 1, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_selector(router_q, *args, **kwargs):
            self.assertEqual(router_q.shape, (1, 1, rows, columns, heads, 1))
            self.assertTrue(torch.all(router_q == 0))
            self.assertEqual(kwargs["top_probes"], 40)
            self.assertEqual(kwargs["uniform_probes"], 40)
            self.assertEqual(kwargs["block_j"], 1)
            return router_table

        def fake_singleton(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            kwargs["output"].fill_(29)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            side_effect=fake_selector,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
            side_effect=fake_singleton,
        ):
            output = _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode=(
                    "hellinger_span_alpha0p2_c32_p80_a2_j1_t1_rf0_"
                    "cg45_sq1_qm_fp8score_skv_nl_ml2"
                ),
            )

        self.assertTrue(torch.all(output == 29))
        self.assertEqual(selector.call_count, 1)

    def test_split_kv_accum_suffix_reaches_exact_native_merge(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, 1, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_singleton(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertIs(seed_rows, router_table)
            self.assertTrue(kwargs["seed_rows_native_layout"])
            self.assertTrue(kwargs["query_column_major"])
            self.assertTrue(kwargs["split_kv_accum"])
            self.assertEqual(kwargs["merge_loop_stages"], 2)
            self.assertEqual(kwargs["selector"], "query_column")
            kwargs["output"].fill_(23)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ), patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
            side_effect=fake_singleton,
        ) as singleton:
            output = _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode=(
                    "hellinger_span_alpha0p3_c22_p16_a2_j4_t1_sq1_"
                    "qm_skv_nl_ml2"
                ),
            )

        self.assertTrue(torch.all(output == 23))
        self.assertEqual(singleton.call_count, 1)

    def test_p16_uniform_probe_policy_reaches_selector(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, 1, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_singleton(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertIs(seed_rows, router_table)
            kwargs["output"].fill_(37)
            return kwargs["output"]

        cases = (
            ("", 0, 16),
            ("_u16", 0, 16),
            ("_a8u8", 8, 8),
        )
        for policy_suffix, expected_top, expected_uniform in cases:
            with self.subTest(policy_suffix=policy_suffix or "default"), patch(
                "attention.token_merging.hellinger_span_seed_rows_triton",
                return_value=router_table,
            ) as selector, patch(
                "attention.token_merging."
                "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
                side_effect=fake_singleton,
            ):
                output = _hard_c4_attention(
                    q, q, q,
                    h=rows,
                    w=columns,
                    h_reduce=1,
                    w_reduce=1,
                    sm_scale=1.0,
                    query_outer_chunk=21,
                    head_chunk=8,
                    hard_route_backend="hard_c4",
                    exact_intra_frame_attention=False,
                    exact_neighbor_window_columns=0,
                    exact_query_rows=0,
                    first_frame_full_attention=False,
                    full_attention_anchor_time_count=0,
                    full_attention_anchor_row_count=0,
                    full_attention_anchor_scale=1.0,
                    seed_mode=(
                        "hellinger_span_alpha0p2_c32_p16_a2_j4_t1"
                        f"{policy_suffix}_cg45_sq1_qm_fp8score_skv_nl_ml2"
                    ),
                )

            self.assertTrue(torch.all(output == 37))
            self.assertEqual(
                selector.call_args.kwargs["top_probes"], expected_top
            )
            self.assertEqual(
                selector.call_args.kwargs["uniform_probes"], expected_uniform
            )

    def test_selective_query_column_r2_suffix_reaches_exact_native_merge(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, 1, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_singleton(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertIs(seed_rows, router_table)
            self.assertEqual(kwargs["top_r"], 2)
            self.assertEqual(kwargs["selector"], "query_column")
            self.assertTrue(kwargs["seed_rows_native_layout"])
            self.assertTrue(kwargs["query_column_major"])
            self.assertTrue(kwargs["split_kv_accum"])
            self.assertEqual(kwargs["query_column_secondary_frame_stride"], 2)
            kwargs["output"].fill_(29)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ), patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
            side_effect=fake_singleton,
        ) as singleton:
            output = _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode=(
                    "hellinger_span_alpha0p3_c16_p16_a2_j4_t1_sq2_sf2_"
                    "qm_skv_nl_ml2"
                ),
            )

        self.assertTrue(torch.all(output == 29))
        self.assertEqual(singleton.call_count, 1)

    def test_selective_query_column_r1_row_mask_reaches_exact_native_merge(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, 1, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_singleton(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertIs(seed_rows, router_table)
            self.assertEqual(kwargs["top_r"], 1)
            self.assertEqual(kwargs["selector"], "query_column")
            self.assertTrue(kwargs["seed_rows_native_layout"])
            self.assertTrue(kwargs["query_column_major"])
            self.assertTrue(kwargs["split_kv_accum"])
            self.assertTrue(kwargs["query_column_primary_row_mod4_omit2"])
            kwargs["output"].fill_(31)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ), patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
            side_effect=fake_singleton,
        ) as singleton:
            output = _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode=(
                    "hellinger_span_alpha0p3_c22_p16_a2_j4_t1_sq1_"
                    "pr4o2_qm_skv_nl_ml2"
                ),
            )

        self.assertTrue(torch.all(output == 31))
        self.assertEqual(singleton.call_count, 1)

    def test_tuned_selector_launch_suffix_reaches_selector(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, 1, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_singleton(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertTrue(kwargs["query_column_major"])
            kwargs["output"].fill_(17)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
            side_effect=fake_singleton,
        ):
            output = _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode=(
                    "hellinger_span_alpha0p3_c22_p16_a2_j4_t1_sq1_qm_gt32gw8"
                ),
            )

        self.assertTrue(torch.all(output == 17))
        self.assertEqual(selector.call_args.kwargs["group_tile"], 32)
        self.assertEqual(selector.call_args.kwargs["gs_num_warps"], 8)

    def test_temporal_routing_expands_three_frames_and_recomputes_first(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, 3, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_hard_c4(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertEqual(seed_rows.shape, (1, 3, 80, 8, 21, 45, 4))
            kwargs["output"].fill_(1)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_attention_triton_blocks",
            side_effect=fake_hard_c4,
        ) as fused, patch(
            "attention.token_merging.flash_attention",
            return_value=torch.full((1, rows * columns, heads, 1), 2.0),
        ) as dense_first:
            output = _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=True,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode="hellinger_span_alpha0p3_c22_p16_a2_j4_t3",
            )

        selected = selector.call_args.args[0]
        self.assertEqual(selected.shape[1], 3)
        self.assertTrue(torch.all(output[:, : rows * columns] == 2))
        self.assertTrue(torch.all(output[:, rows * columns :] == 1))
        self.assertEqual(fused.call_count, 1)
        self.assertEqual(dense_first.call_count, 1)

    def test_temporal_routing_dispatches_four_frames_to_shared_kernel(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = (
            torch.arange(frames, dtype=torch.float32)
            .repeat_interleave(rows * columns)
            .view(1, -1, 1, 1)
            .expand(-1, -1, heads, -1)
            .contiguous()
        )
        router_table = torch.zeros(
            (1, 4, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_hard_c4(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertEqual(seed_rows.shape, (1, 4, 80, 8, 21, 45, 4))
            kwargs["output"].fill_(4)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_attention_triton_blocks",
            side_effect=fake_hard_c4,
        ) as fused:
            output = _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode="hellinger_span_alpha0p3_c22_p16_a2_j4_t4_lm1",
            )

        self.assertTrue(torch.all(output == 4))
        selected = selector.call_args.args[0]
        self.assertEqual(selected.shape[1], 4)
        self.assertEqual(selected[0, :, 0, 0, 0, 0].tolist(), [0.0, 7.0, 13.0, 20.0])
        self.assertTrue(selector.call_args.kwargs["force_local_minus_one"])
        self.assertEqual(fused.call_count, 1)

    def test_temporal_simpson_router_uses_fixed_three_point_rule(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        frame_values = torch.arange(frames, dtype=torch.float32).square()
        q = (
            frame_values.repeat_interleave(rows * columns)
            .view(1, -1, 1, 1)
            .expand(-1, -1, heads, -1)
            .contiguous()
        )
        router_table = torch.zeros(
            (1, 1, frames, rows, columns, heads, 4), dtype=torch.uint8
        )
        expected = (frame_values[0] + 4 * frame_values[10] + frame_values[20]) / 6
        expected_router = torch.full(
            (1, 1, rows, columns, heads, 1), expected
        )

        def fake_hard_c4(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            kwargs["output"].fill_(3)
            return kwargs["output"]

        with patch(
            "attention.token_merging.temporal_simpson_query_triton",
            return_value=expected_router,
        ) as simpson, patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_attention_triton_blocks",
            side_effect=fake_hard_c4,
        ) as fused:
            output = _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode="hellinger_span_alpha0p2_c32_p16_a2_j4_t1_ts3_cg45",
            )

        self.assertTrue(torch.allclose(selector.call_args.args[0], expected_router))
        self.assertTrue(torch.all(output == 3))
        simpson.assert_called_once()
        self.assertEqual(simpson.call_args.kwargs["center_frame"], 10)
        self.assertEqual(fused.call_count, 1)

    def test_temporal_routing_dispatches_singleton_residual_budget(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, 1, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_singleton(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertEqual(seed_rows.shape, (1, 1, 80, 8, 21, 45, 4))
            self.assertEqual(kwargs["top_r"], 4)
            self.assertEqual(kwargs["selector"], "cluster_mass")
            kwargs["output"].fill_(4)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
            side_effect=fake_singleton,
        ) as singleton:
            output = _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode="hellinger_span_alpha0p3_c22_p16_a2_j4_t1_sc4",
            )

        self.assertTrue(torch.all(output == 4))
        self.assertEqual(selector.call_count, 1)
        self.assertEqual(singleton.call_count, 1)

    def test_head_and_temporal_sharing_keep_singleton_fused_path(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, 1, frames, rows, columns, heads // 2, 4), dtype=torch.uint8
        )

        def fake_singleton(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertEqual(seed_rows.shape, (1, 1, 80, 8, 21, 45, 4))
            self.assertEqual(kwargs["top_r"], 1)
            self.assertEqual(kwargs["selector"], "query_column")
            kwargs["output"].fill_(2)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
            side_effect=fake_singleton,
        ) as singleton:
            output = _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode=(
                    "hellinger_span_alpha0p3_c22_p16_a2_j4_g2_r0_t1_sq1"
                ),
            )

        self.assertTrue(torch.all(output == 2))
        self.assertEqual(selector.call_args.args[0].shape[4], heads // 2)
        self.assertEqual(singleton.call_count, 1)

    def test_temporal_routing_parses_logit_variance_bias(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, 1, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_hard_c4(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertEqual(kwargs["logit_variance_bias_scale"], 0.5)
            kwargs["output"].fill_(1)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ), patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_attention_triton_blocks",
            side_effect=fake_hard_c4,
        ) as fused:
            output = _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode="hellinger_span_alpha0p3_c22_p16_a2_j4_t1_lv0p5",
            )

        self.assertTrue(torch.all(output == 1))
        self.assertEqual(fused.call_count, 1)

    def test_temporal_routing_dispatches_simplified_selector(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, 1, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
        ) as fused:
            _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode=(
                    "hellinger_span_alpha0p3_c22_p16_a2_j4_t1_"
                    "two_stage_sq1"
                ),
            )

        self.assertEqual(selector.call_args.kwargs["selection_mode"], "two_stage")
        self.assertEqual(fused.call_count, 1)

    def test_temporal_routing_dispatches_pivoted_cholesky_selector(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, 1, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
        ) as fused:
            _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode=(
                    "hellinger_span_alpha0p3_c22_p16_a2_j4_t1_"
                    "chol_sq1"
                ),
            )

        self.assertEqual(
            selector.call_args.kwargs["selection_mode"],
            "pivoted_cholesky",
        )
        self.assertEqual(fused.call_count, 1)

    def test_temporal_routing_dispatches_pair_first_selector(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, 1, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
        ) as fused:
            _hard_c4_attention(
                q, q, q,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode=(
                    "hellinger_span_alpha0p3_c22_p16_a2_j4_t1_"
                    "pair_m8_sq1"
                ),
            )

        self.assertEqual(
            selector.call_args.kwargs["selection_mode"],
            "pair_first_m8",
        )
        self.assertEqual(fused.call_count, 1)

    def test_temporal_routing_dispatches_compact_hard_c5(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, 3, frames, rows, columns, heads, 5), dtype=torch.uint8
        )

        def fake_hard_c5(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertEqual(seed_rows.shape, (1, 3, 80, 8, 21, 45, 5))
            kwargs["output"].fill_(5)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c5_720p_temporal_shared_seed_rows_attention_triton_blocks",
            side_effect=fake_hard_c5,
        ) as fused:
            output = _hard_c568_attention(
                q,
                q,
                q,
                h=rows,
                w=columns,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                num_merge_tokens=5,
                seed_mode=(
                    "hellinger_span_alpha0p3_c22_p16_a2_j4_t3_gt32gw8"
                ),
            )

        self.assertTrue(torch.all(output == 5))
        self.assertEqual(selector.call_args.kwargs["num_seeds"], 5)
        self.assertEqual(selector.call_args.kwargs["group_tile"], 32)
        self.assertEqual(selector.call_args.kwargs["gs_num_warps"], 8)
        self.assertEqual(selector.call_args.args[0].shape[1], 3)
        self.assertEqual(fused.call_count, 1)

    def test_common_c4_dispatches_generalized_external_seed_kernel(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))
        router_table = torch.zeros(
            (1, 1, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_common(q_blocks, k_blocks, v_blocks, seed_rows, c, *args, **kwargs):
            self.assertEqual(seed_rows.shape, router_table.shape)
            self.assertEqual(c, 4)
            kwargs["output"].fill_(4)
            return kwargs["output"]

        with patch(
            "attention.token_merging.hellinger_span_seed_rows_triton",
            return_value=router_table,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c_720p_native_temporal_seed_rows_attention_triton_blocks",
            side_effect=fake_common,
        ) as fused:
            output = _hard_c4_attention(
                q,
                q,
                q,
                seed_query=None,
                h=rows,
                w=columns,
                h_reduce=1,
                w_reduce=1,
                sm_scale=1.0,
                query_outer_chunk=21,
                head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode=(
                    "hellinger_span_alpha0p2_c32_p16_a2_j4_t1_rf0_"
                    "cg45_commonc"
                ),
            )

        self.assertTrue(torch.all(output == 4))
        self.assertEqual(selector.call_args.kwargs["candidate_geometry_count"], 45)
        self.assertEqual(fused.call_count, 1)

    def test_common_c6_c8_dispatch_generalized_external_seed_kernel(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.zeros((1, frames * rows * columns, heads, 1))

        for c in (6, 8):
            with self.subTest(c=c):
                router_table = torch.zeros(
                    (1, 1, frames, rows, columns, heads, c),
                    dtype=torch.uint8,
                )

                def fake_common(
                    q_blocks,
                    k_blocks,
                    v_blocks,
                    seed_rows,
                    actual_c,
                    *args,
                    **kwargs,
                ):
                    self.assertEqual(seed_rows.shape, router_table.shape)
                    self.assertEqual(actual_c, c)
                    kwargs["output"].fill_(c)
                    return kwargs["output"]

                with patch(
                    "attention.token_merging.hellinger_span_seed_rows_triton",
                    return_value=router_table,
                ) as selector, patch(
                    "attention.token_merging."
                    "hard_c_720p_native_temporal_seed_rows_attention_triton_blocks",
                    side_effect=fake_common,
                ) as fused:
                    output = _hard_c568_attention(
                        q,
                        q,
                        q,
                        h=rows,
                        w=columns,
                        sm_scale=1.0,
                        query_outer_chunk=21,
                        head_chunk=8,
                        num_merge_tokens=c,
                        seed_mode=(
                            "hellinger_span_alpha0p2_c32_p16_a2_j4_t1_rf0_"
                            "cg45_commonc"
                        ),
                    )

                self.assertTrue(torch.all(output == c))
                self.assertEqual(selector.call_args.kwargs["num_seeds"], c)
                self.assertEqual(
                    selector.call_args.kwargs["candidate_geometry_count"], 45
                )
                self.assertEqual(fused.call_count, 1)


if __name__ == "__main__":
    unittest.main()
