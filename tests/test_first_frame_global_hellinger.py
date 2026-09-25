import unittest
from unittest.mock import patch

import torch

from attention.adaptive_seed_pytorch import (
    first_frame_global_hellinger_seed_rows,
)
from attention.token_merging import _hard_c4_attention


class FirstFrameGlobalHellingerTest(unittest.TestCase):
    def test_selector_uses_global_keys_and_shares_across_key_frames(self):
        torch.manual_seed(0)
        q = torch.randn(1, 1, 5, 3, 2, 4)
        k = torch.randn(1, 2, 5, 3, 2, 4)
        table = first_frame_global_hellinger_seed_rows(
            q,
            k,
            candidate_count=5,
            candidate_geometry_count=5,
            num_seeds=4,
            head_chunk=1,
            column_chunk=2,
        )

        self.assertEqual(table.shape, (1, 1, 2, 5, 3, 2, 4))
        self.assertEqual(table.dtype, torch.uint8)
        torch.testing.assert_close(table[:, :, 0], table[:, :, 1])
        expected_anchor = torch.arange(5, dtype=torch.uint8).view(1, 1, 5, 1, 1)
        torch.testing.assert_close(
            table[:, :, 0, ..., 0], expected_anchor.expand(1, 1, 5, 3, 2)
        )
        flattened = table[:, :, 0].reshape(-1, 4)
        self.assertTrue(all(torch.unique(rows).numel() == 4 for rows in flattened))

    def test_ffg_dispatches_frame_zero_to_global_selector_without_dense_output(self):
        frames, rows, columns, heads = 21, 45, 80, 2
        frame_ids = torch.arange(frames, dtype=torch.float32).repeat_interleave(
            rows * columns
        )
        q = frame_ids[None, :, None, None].expand(-1, -1, heads, 1).clone()
        router_table = torch.zeros(
            (1, 1, frames, rows, columns, heads, 4), dtype=torch.uint8
        )

        def fake_selector(router_q, router_k, *args, **kwargs):
            self.assertEqual(router_q.shape, (1, 1, rows, columns, heads, 1))
            self.assertTrue(torch.all(router_q == 0))
            self.assertEqual(router_k.shape, (1, frames, rows, columns, heads, 1))
            return router_table

        def fake_merge(*args, **kwargs):
            kwargs["output"].fill_(17)
            return kwargs["output"]

        with patch(
            "attention.token_merging.first_frame_global_hellinger_seed_rows",
            side_effect=fake_selector,
        ) as selector, patch(
            "attention.token_merging.hard_c4_720p_temporal_shared_seed_rows_attention_triton_blocks",
            side_effect=fake_merge,
        ), patch("attention.token_merging.flash_attention") as dense_output:
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
                head_chunk=2,
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                exact_query_rows=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode=(
                    "hellinger_span_alpha0p2_c32_p80_a2_j4_t1_rf0_ffg_"
                    "cg45_nl_ml2"
                ),
            )

        self.assertTrue(torch.all(output == 17))
        self.assertEqual(selector.call_count, 1)
        selector_kwargs = selector.call_args.kwargs
        self.assertEqual(selector_kwargs["head_chunk"], min(16, heads))
        self.assertEqual(selector_kwargs["column_chunk"], 16)
        self.assertEqual(selector_kwargs["profile_dtype"], torch.float16)
        dense_output.assert_not_called()


if __name__ == "__main__":
    unittest.main()
