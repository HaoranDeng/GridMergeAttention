import unittest
from unittest.mock import patch

import torch

from attention.token_merging import _hard_c4_attention


class WeightedSpanTemporalTest(unittest.TestCase):
    def test_exact_selector_mode_preserves_reference_launch_geometry(self):
        frames, rows, columns, heads = 21, 45, 80, 4
        q = torch.zeros(1, frames * rows * columns, heads, 1)
        router_table = torch.zeros(
            (1, frames, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_selector(*args, **kwargs):
            self.assertIsNone(kwargs["key_outer_tile"])
            self.assertEqual(kwargs["gs_num_warps"], 4)
            self.assertEqual(kwargs["block_j"], 2)
            self.assertEqual(kwargs["value_dims"], 32)
            return router_table

        with patch(
            "attention.token_merging.span_seed_rows_triton_split",
            side_effect=fake_selector,
        ), patch(
            "attention.token_merging."
            "hard_c4_720p_native_seed_rows_attention_triton_blocks",
            side_effect=lambda *args, **kwargs: kwargs["output"].fill_(5),
        ):
            output = _hard_c4_attention(
                q, q, q,
                h=rows, w=columns, h_reduce=1, w_reduce=1,
                sm_scale=1.0, query_outer_chunk=21, head_chunk=4,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0, exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode=(
                    "weighted_span_alpha0p3_pool45_"
                    "sx_hg1o0_j2_p16_v32"
                ),
            )

        self.assertTrue(torch.all(output == 5))

    def test_routing_head_table_is_cached_across_denoise_steps(self):
        frames, rows, columns, heads = 21, 45, 80, 4
        q = torch.zeros(1, frames * rows * columns, heads, 1)
        router_table = torch.zeros(
            (1, frames, frames, rows, columns, 2, 4), dtype=torch.uint8
        )
        cache = {}

        def fake_attention(*args, **kwargs):
            kwargs["output"].fill_(6)
            return kwargs["output"]

        with patch(
            "attention.token_merging.span_seed_rows_triton_split",
            return_value=router_table,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c4_720p_native_seed_rows_attention_triton_blocks",
            side_effect=fake_attention,
        ):
            for timestep in (1000, 750):
                output = _hard_c4_attention(
                    q, q, q,
                    h=rows, w=columns, h_reduce=1, w_reduce=1,
                    sm_scale=1.0, query_outer_chunk=21, head_chunk=4,
                    hard_route_backend="hard_c4",
                    exact_intra_frame_attention=False,
                    exact_neighbor_window_columns=0, exact_query_rows=0,
                    first_frame_full_attention=False,
                    full_attention_anchor_time_count=0,
                    full_attention_anchor_row_count=0,
                    full_attention_anchor_scale=1.0,
                    seed_mode=(
                        "weighted_span_alpha0p3_pool45_"
                        "hg2o1_j4_p16_v8_dc4"
                    ),
                    seed_table_cache=cache,
                    cache_timestep=timestep,
                )
                self.assertTrue(torch.all(output == 6))

        self.assertEqual(selector.call_count, 1)
        self.assertIs(cache["weighted_span_routing_heads"], router_table)

    def test_two_step_cache_refreshes_at_midpoint(self):
        frames, rows, columns, heads = 21, 45, 80, 10
        q = torch.zeros(1, frames * rows * columns, heads, 1)
        router_table = torch.zeros(
            (1, frames, frames, rows, columns, 2, 4), dtype=torch.uint8
        )
        cache = {}

        with patch(
            "attention.token_merging.span_seed_rows_triton_split",
            return_value=router_table,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c4_720p_native_seed_rows_attention_triton_blocks",
            side_effect=lambda *args, **kwargs: kwargs["output"].fill_(2),
        ):
            for timestep in (1000, 750, 500, 250):
                _hard_c4_attention(
                    q, q, q,
                    h=rows, w=columns, h_reduce=1, w_reduce=1,
                    sm_scale=1.0, query_outer_chunk=21, head_chunk=10,
                    hard_route_backend="hard_c4",
                    exact_intra_frame_attention=False,
                    exact_neighbor_window_columns=0, exact_query_rows=0,
                    first_frame_full_attention=False,
                    full_attention_anchor_time_count=0,
                    full_attention_anchor_row_count=0,
                    full_attention_anchor_scale=1.0,
                    seed_mode=(
                        "weighted_span_alpha0p3_pool45_"
                        "hg5o3_j4_p16_v8_dc2"
                    ),
                    seed_table_cache=cache,
                    cache_timestep=timestep,
                )

        self.assertEqual(selector.call_count, 2)

    def test_three_step_cache_reuses_only_final_route(self):
        frames, rows, columns, heads = 21, 45, 80, 10
        q = torch.zeros(1, frames * rows * columns, heads, 1)
        router_table = torch.zeros(
            (1, frames, frames, rows, columns, 2, 4), dtype=torch.uint8
        )
        cache = {}

        with patch(
            "attention.token_merging.span_seed_rows_triton_split",
            return_value=router_table,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c4_720p_native_seed_rows_attention_triton_blocks",
            side_effect=lambda *args, **kwargs: kwargs["output"].fill_(3),
        ):
            for timestep in (1000, 750, 500, 250):
                _hard_c4_attention(
                    q, q, q,
                    h=rows, w=columns, h_reduce=1, w_reduce=1,
                    sm_scale=1.0, query_outer_chunk=21, head_chunk=10,
                    hard_route_backend="hard_c4",
                    exact_intra_frame_attention=False,
                    exact_neighbor_window_columns=0, exact_query_rows=0,
                    first_frame_full_attention=False,
                    full_attention_anchor_time_count=0,
                    full_attention_anchor_row_count=0,
                    full_attention_anchor_scale=1.0,
                    seed_mode=(
                        "weighted_span_alpha0p3_pool45_"
                        "hg5o3_j4_p16_v8_dc3"
                    ),
                    seed_table_cache=cache,
                    cache_timestep=timestep,
                )

        self.assertEqual(selector.call_count, 3)

    def test_two_step_late_cache_refreshes_final_step(self):
        frames, rows, columns, heads = 21, 45, 80, 10
        q = torch.zeros(1, frames * rows * columns, heads, 1)
        router_table = torch.zeros(
            (1, frames, frames, rows, columns, 2, 4), dtype=torch.uint8
        )
        cache = {}

        with patch(
            "attention.token_merging.span_seed_rows_triton_split",
            return_value=router_table,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c4_720p_native_seed_rows_attention_triton_blocks",
            side_effect=lambda *args, **kwargs: kwargs["output"].fill_(2),
        ):
            call_counts = []
            for timestep in (1000, 938, 833, 625):
                _hard_c4_attention(
                    q, q, q,
                    h=rows, w=columns, h_reduce=1, w_reduce=1,
                    sm_scale=1.0, query_outer_chunk=21, head_chunk=10,
                    hard_route_backend="hard_c4",
                    exact_intra_frame_attention=False,
                    exact_neighbor_window_columns=0, exact_query_rows=0,
                    first_frame_full_attention=False,
                    full_attention_anchor_time_count=0,
                    full_attention_anchor_row_count=0,
                    full_attention_anchor_scale=1.0,
                    seed_mode=(
                        "weighted_span_alpha0p3_pool45_"
                        "hg5o3_j4_p16_v8_dc2l"
                    ),
                    seed_table_cache=cache,
                    cache_timestep=timestep,
                )
                call_counts.append(selector.call_count)

        self.assertEqual(selector.call_count, 2)
        self.assertEqual(call_counts, [1, 1, 1, 2])

    def test_three_step_cache_follows_warped_timestep_order(self):
        frames, rows, columns, heads = 21, 45, 80, 10
        q = torch.zeros(1, frames * rows * columns, heads, 1)
        router_table = torch.zeros(
            (1, frames, frames, rows, columns, 2, 4), dtype=torch.uint8
        )
        cache = {}

        with patch(
            "attention.token_merging.span_seed_rows_triton_split",
            return_value=router_table,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c4_720p_native_seed_rows_attention_triton_blocks",
            side_effect=lambda *args, **kwargs: kwargs["output"].fill_(3),
        ):
            for timestep in (1000, 938, 833, 625):
                _hard_c4_attention(
                    q, q, q,
                    h=rows, w=columns, h_reduce=1, w_reduce=1,
                    sm_scale=1.0, query_outer_chunk=21, head_chunk=10,
                    hard_route_backend="hard_c4",
                    exact_intra_frame_attention=False,
                    exact_neighbor_window_columns=0, exact_query_rows=0,
                    first_frame_full_attention=False,
                    full_attention_anchor_time_count=0,
                    full_attention_anchor_row_count=0,
                    full_attention_anchor_scale=1.0,
                    seed_mode=(
                        "weighted_span_alpha0p3_pool45_"
                        "hg5o3_j4_p16_v8_dc3"
                    ),
                    seed_table_cache=cache,
                    cache_timestep=timestep,
                )

        self.assertEqual(selector.call_count, 3)

    def test_routing_heads_expand_native_per_frame_seed_table(self):
        frames, rows, columns, heads = 21, 45, 80, 20
        q = torch.zeros(1, frames * rows * columns, heads, 1)
        router_table = torch.zeros(
            (1, frames, frames, rows, columns, 2, 4), dtype=torch.uint8
        )
        router_table[..., 0, :] = 1
        router_table[..., 1, :] = 2

        def fake_selector(q_blocks, k_blocks, v_blocks, *args, **kwargs):
            self.assertEqual(q_blocks.shape[4], 2)
            self.assertEqual(kwargs["candidate_count"], 45)
            self.assertEqual(kwargs["block_j"], 4)
            self.assertEqual(kwargs["top_probes"], 8)
            self.assertEqual(kwargs["uniform_probes"], 8)
            self.assertEqual(kwargs["value_dims"], 8)
            self.assertEqual(kwargs["key_outer_tile"], 7)
            self.assertEqual(kwargs["output_dtype"], torch.uint8)
            self.assertEqual(kwargs["gs_num_warps"], 8)
            return router_table

        def fake_attention(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertEqual(
                seed_rows.shape,
                (1, frames, frames, rows, columns, heads, 4),
            )
            self.assertTrue(torch.all(seed_rows[..., :10, :] == 1))
            self.assertTrue(torch.all(seed_rows[..., 10:, :] == 2))
            kwargs["output"].fill_(8)
            return kwargs["output"]

        with patch(
            "attention.token_merging.span_seed_rows_triton_split",
            side_effect=fake_selector,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c4_720p_native_seed_rows_attention_triton_blocks",
            side_effect=fake_attention,
        ) as attention:
            output = _hard_c4_attention(
                q, q, q,
                h=rows, w=columns, h_reduce=1, w_reduce=1, sm_scale=1.0,
                query_outer_chunk=21, head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0, exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode=(
                    "weighted_span_alpha0p3_pool45_hg10o4_j4_p16_v8"
                ),
            )

        self.assertTrue(torch.all(output == 8))
        self.assertEqual(selector.call_count, 1)
        self.assertEqual(attention.call_count, 3)

    def test_key_frame_routing_expands_to_all_key_frames(self):
        frames, rows, columns, heads = 21, 45, 80, 4
        q = torch.zeros(1, frames * rows * columns, heads, 1)
        compact_table = torch.arange(7, dtype=torch.uint8).reshape(
            1, 1, 7, 1, 1, 1, 1
        ).expand(1, frames, 7, rows, columns, heads, 4)

        def fake_selector(q_blocks, k_blocks, v_blocks, *args, **kwargs):
            self.assertEqual(q_blocks.shape[1], frames)
            self.assertEqual(k_blocks.shape[1], 7)
            self.assertEqual(v_blocks.shape[1], 7)
            return compact_table

        def fake_attention(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertEqual(seed_rows.shape[2], frames)
            self.assertTrue(torch.all(seed_rows[:, :, 0] == 0))
            self.assertTrue(torch.all(seed_rows[:, :, -1] == 6))
            return kwargs["output"].fill_(9)

        with patch(
            "attention.token_merging.span_seed_rows_triton_split",
            side_effect=fake_selector,
        ), patch(
            "attention.token_merging."
            "hard_c4_720p_native_seed_rows_attention_triton_blocks",
            side_effect=fake_attention,
        ):
            output = _hard_c4_attention(
                q, q, q,
                h=rows, w=columns, h_reduce=1, w_reduce=1,
                sm_scale=1.0, query_outer_chunk=21, head_chunk=4,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0, exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode=(
                    "weighted_span_alpha0p3_pool45_"
                    "hg1o0_j4_p16_v8_kf7_dc2"
                ),
                seed_table_cache={},
                cache_timestep=1000,
            )

        self.assertTrue(torch.all(output == 9))

    def test_temporal_probes_keep_per_frame_native_tables(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.arange(frames, dtype=torch.float32).view(
            1, frames, 1, 1, 1, 1
        )
        q = q.expand(1, frames, rows, columns, heads, 1).reshape(
            1, frames * rows * columns, heads, 1
        )
        native_table = torch.zeros(
            (1, frames, frames, rows, columns, heads, 4),
            dtype=torch.uint8,
        )

        def fake_selector(q_blocks, k_blocks, v_blocks, *args, **kwargs):
            self.assertEqual(q_blocks.shape[1], frames)
            self.assertEqual(kwargs["candidate_count"], 45)
            self.assertEqual(kwargs["top_probes"], 8)
            self.assertEqual(kwargs["uniform_probes"], 8)
            self.assertEqual(kwargs["value_dims"], 8)
            self.assertEqual(kwargs["block_j"], 4)
            self.assertEqual(kwargs["router_frames"], 3)
            return native_table

        def fake_attention(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertIs(seed_rows, native_table)
            self.assertEqual(seed_rows.dtype, torch.uint8)
            self.assertEqual(
                seed_rows.shape,
                (1, frames, frames, rows, columns, heads, 4),
            )
            kwargs["output"].fill_(4)
            return kwargs["output"]

        with patch(
            "attention.token_merging."
            "span_seed_rows_triton_temporal_probes_per_frame",
            side_effect=fake_selector,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c4_720p_native_seed_rows_attention_triton_blocks",
            side_effect=fake_attention,
        ) as attention:
            output = _hard_c4_attention(
                q, q, q,
                h=rows, w=columns, h_reduce=1, w_reduce=1, sm_scale=1.0,
                query_outer_chunk=21, head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0, exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode=(
                    "weighted_span_alpha0p3_pool45_tp3_j4_p16_v8"
                ),
            )

        self.assertTrue(torch.all(output == 4))
        self.assertEqual(selector.call_count, 1)
        self.assertEqual(attention.call_count, 1)

    def test_middle_frame_table_is_shared_by_all_query_frames(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.arange(frames, dtype=torch.float32).view(1, frames, 1, 1, 1, 1)
        q = q.expand(1, frames, rows, columns, heads, 1).reshape(
            1, frames * rows * columns, heads, 1
        )
        router_table = torch.zeros(
            (1, 1, frames, rows, columns, heads, 4), dtype=torch.int32
        )

        def fake_selector(q_blocks, k_blocks, v_blocks, *args, **kwargs):
            self.assertEqual(q_blocks.shape[1], 1)
            self.assertTrue(torch.all(q_blocks == 10))
            self.assertEqual(kwargs["candidate_count"], 45)
            self.assertEqual(kwargs["block_j"], 4)
            return router_table

        def fake_attention(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertEqual(seed_rows.shape, (1, 1, columns, heads, frames, rows, 4))
            kwargs["output"].fill_(9)
            return kwargs["output"]

        with patch(
            "attention.token_merging.span_seed_rows_triton_split",
            side_effect=fake_selector,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_attention_triton_blocks",
            side_effect=fake_attention,
        ) as attention:
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
                seed_mode="weighted_span_alpha0p3_pool45_t1_j4_rf10",
            )

        self.assertTrue(torch.all(output == 9))
        self.assertEqual(selector.call_count, 1)
        self.assertEqual(attention.call_count, 1)

    def test_two_router_consensus_emits_one_shared_table(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.arange(frames, dtype=torch.float32).view(1, frames, 1, 1, 1, 1)
        q = q.expand(1, frames, rows, columns, heads, 1).reshape(
            1, frames * rows * columns, heads, 1
        )
        router_table = torch.zeros(
            (1, 1, frames, rows, columns, heads, 4), dtype=torch.int32
        )

        def fake_selector(q_blocks, *args, **kwargs):
            self.assertEqual(q_blocks.shape[1], 2)
            self.assertTrue(torch.all(q_blocks[:, 0] == 5))
            self.assertTrue(torch.all(q_blocks[:, 1] == 15))
            self.assertEqual(kwargs["top_probes"], 8)
            self.assertEqual(kwargs["uniform_probes"], 8)
            self.assertEqual(kwargs["value_dims"], 8)
            self.assertEqual(kwargs["block_j"], 4)
            return router_table

        def fake_attention(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertEqual(seed_rows.shape, (1, 1, columns, heads, frames, rows, 4))
            kwargs["output"].fill_(6)
            return kwargs["output"]

        with patch(
            "attention.token_merging.span_seed_rows_triton_temporal_consensus2",
            side_effect=fake_selector,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_attention_triton_blocks",
            side_effect=fake_attention,
        ) as attention:
            output = _hard_c4_attention(
                q, q, q,
                h=rows, w=columns, h_reduce=1, w_reduce=1, sm_scale=1.0,
                query_outer_chunk=21, head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0, exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode="weighted_span_alpha0p3_pool45_tc2_j4_p16_v8",
            )

        self.assertTrue(torch.all(output == 6))
        self.assertEqual(selector.call_count, 1)
        self.assertEqual(attention.call_count, 1)

    def test_two_router_frames_are_selected_in_one_batched_call(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.arange(frames, dtype=torch.float32).view(1, frames, 1, 1, 1, 1)
        q = q.expand(1, frames, rows, columns, heads, 1).reshape(
            1, frames * rows * columns, heads, 1
        )
        router_table = torch.zeros(
            (1, 2, frames, rows, columns, heads, 4), dtype=torch.int32
        )

        def fake_selector(q_blocks, k_blocks, v_blocks, *args, **kwargs):
            self.assertEqual(q_blocks.shape[1], 2)
            self.assertTrue(torch.all(q_blocks[:, 0] == 5))
            self.assertTrue(torch.all(q_blocks[:, 1] == 15))
            self.assertEqual(kwargs["candidate_count"], 45)
            self.assertEqual(kwargs["block_j"], 4)
            return router_table

        def fake_attention(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertEqual(seed_rows.shape, (1, 2, columns, heads, frames, rows, 4))
            kwargs["output"].fill_(7)
            return kwargs["output"]

        with patch(
            "attention.token_merging.span_seed_rows_triton_split",
            side_effect=fake_selector,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_attention_triton_blocks",
            side_effect=fake_attention,
        ) as attention:
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
                seed_mode="weighted_span_alpha0p3_pool45_t2_j4",
            )

        self.assertTrue(torch.all(output == 7))
        self.assertEqual(selector.call_count, 1)
        self.assertEqual(attention.call_count, 1)

    def test_three_router_frames_use_centered_temporal_coverage(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.arange(frames, dtype=torch.float32).view(1, frames, 1, 1, 1, 1)
        q = q.expand(1, frames, rows, columns, heads, 1).reshape(
            1, frames * rows * columns, heads, 1
        )
        router_table = torch.zeros(
            (1, 3, frames, rows, columns, heads, 4), dtype=torch.int32
        )

        def fake_selector(q_blocks, *args, **kwargs):
            self.assertEqual(q_blocks.shape[1], 3)
            self.assertTrue(torch.all(q_blocks[:, 0] == 3))
            self.assertTrue(torch.all(q_blocks[:, 1] == 10))
            self.assertTrue(torch.all(q_blocks[:, 2] == 17))
            self.assertEqual(kwargs["value_dims"], 8)
            return router_table

        def fake_attention(*args, **kwargs):
            kwargs["output"].fill_(5)
            return kwargs["output"]

        with patch(
            "attention.token_merging.span_seed_rows_triton_split",
            side_effect=fake_selector,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_attention_triton_blocks",
            side_effect=fake_attention,
        ) as attention:
            output = _hard_c4_attention(
                q, q, q,
                h=rows, w=columns, h_reduce=1, w_reduce=1, sm_scale=1.0,
                query_outer_chunk=21, head_chunk=8,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0, exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode="weighted_span_alpha0p3_pool45_t3_j4_p16_v8",
            )

        self.assertTrue(torch.all(output == 5))
        self.assertEqual(selector.call_count, 1)
        self.assertEqual(attention.call_count, 1)

    def test_low_cost_selector_suffix_is_forwarded(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.randn(1, frames * rows * columns, heads, 1)
        router_table = torch.zeros(
            (1, 4, frames, rows, columns, heads, 4), dtype=torch.int32
        )

        def fake_selector(*args, **kwargs):
            self.assertEqual(kwargs["top_probes"], 4)
            self.assertEqual(kwargs["uniform_probes"], 4)
            self.assertEqual(kwargs["value_dims"], 16)
            self.assertEqual(kwargs["block_j"], 4)
            return router_table

        def fake_attention(*args, **kwargs):
            kwargs["output"].fill_(3)
            return kwargs["output"]

        with patch(
            "attention.token_merging.span_seed_rows_triton_split",
            side_effect=fake_selector,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_attention_triton_blocks",
            side_effect=fake_attention,
        ):
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
                seed_mode="weighted_span_alpha0p3_pool45_t4_j4_p8_v16",
            )

        self.assertTrue(torch.all(output == 3))
        self.assertEqual(selector.call_count, 1)

    def test_temporal_mean_router_uses_one_shared_table(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.arange(frames, dtype=torch.float32).view(1, frames, 1, 1, 1, 1)
        q = q.expand(1, frames, rows, columns, heads, 1).reshape(
            1, frames * rows * columns, heads, 1
        )
        router_table = torch.zeros(
            (1, 1, frames, rows, columns, heads, 4), dtype=torch.int32
        )

        def fake_selector(q_blocks, k_blocks, v_blocks, *args, **kwargs):
            self.assertEqual(q_blocks.shape[1], 1)
            self.assertTrue(torch.all(q_blocks == 10))
            return router_table

        def fake_attention(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertEqual(seed_rows.shape, (1, 1, columns, heads, frames, rows, 4))
            kwargs["output"].fill_(5)
            return kwargs["output"]

        with patch(
            "attention.token_merging.span_seed_rows_triton_split",
            side_effect=fake_selector,
        ) as selector, patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_attention_triton_blocks",
            side_effect=fake_attention,
        ) as attention:
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
                seed_mode="weighted_span_alpha0p3_pool45_tm_j4",
            )

        self.assertTrue(torch.all(output == 5))
        self.assertEqual(selector.call_count, 1)
        self.assertEqual(attention.call_count, 1)

    def test_query_row_tracking_is_forwarded_to_shared_merge(self):
        frames, rows, columns, heads = 21, 45, 80, 8
        q = torch.arange(frames, dtype=torch.float32).view(1, frames, 1, 1, 1, 1)
        q = q.expand(1, frames, rows, columns, heads, 1).reshape(
            1, frames * rows * columns, heads, 1
        )
        router_table = torch.zeros(
            (1, 1, frames, rows, columns, heads, 4), dtype=torch.int32
        )
        row_map = torch.zeros(
            (1, frames, columns, heads, rows), dtype=torch.uint8
        )

        def fake_attention(q_blocks, k_blocks, v_blocks, seed_rows, *args, **kwargs):
            self.assertIs(kwargs["temporal_query_row_map"], row_map)
            kwargs["output"].fill_(3)
            return kwargs["output"]

        with patch(
            "attention.token_merging.span_seed_rows_triton_split",
            return_value=router_table,
        ), patch(
            "attention.token_merging.temporal_query_row_map_triton",
            return_value=row_map,
        ) as tracker, patch(
            "attention.token_merging."
            "hard_c4_720p_temporal_shared_seed_rows_attention_triton_blocks",
            side_effect=fake_attention,
        ):
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
                seed_mode="weighted_span_alpha0p3_pool45_t1_j4_tr2_rf10",
            )

        self.assertTrue(torch.all(output == 3))
        tracker.assert_called_once()
        self.assertEqual(tracker.call_args.kwargs["router_frame"], 10)
        self.assertEqual(tracker.call_args.kwargs["radius"], 2)


if __name__ == "__main__":
    unittest.main()
