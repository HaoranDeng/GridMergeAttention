import unittest

import torch

from analysis.evaluate_query_only_seed_kl import (
    C4,
    FRAMES,
    ROWS,
    evaluate_attention_kl,
    select_query_only_rows,
)


class QueryOnlySeedSelectorTest(unittest.TestCase):
    def test_selectors_return_unique_real_rows_with_anchor(self):
        generator = torch.Generator().manual_seed(7)
        query = torch.randn(3, ROWS, 16, generator=generator)
        rows, _ = select_query_only_rows(query, candidate_count=11)

        self.assertIn("fixed_wide", rows)
        for name, table in rows.items():
            self.assertEqual(table.shape, (3, FRAMES * ROWS, C4), name)
            self.assertTrue(bool(((table >= 0) & (table < ROWS)).all()), name)
            self.assertTrue(
                bool((table.sort(dim=-1).values.diff(dim=-1) > 0).all()), name
            )
            if name not in {"fixed_wide", "query_global_c4_d32"}:
                key_row = torch.arange(FRAMES * ROWS) % ROWS
                self.assertTrue(
                    bool((table == key_row.view(1, -1, 1)).any(dim=-1).all()),
                    name,
                )

    def test_global_c4_is_shared_and_global_pool_refinement_is_row_specific(self):
        query = torch.randn(2, ROWS, 32, generator=torch.Generator().manual_seed(13))
        rows, _ = select_query_only_rows(query, candidate_count=6)

        global_c4 = rows["query_global_c4_d32"].reshape(2, FRAMES, ROWS, C4)
        torch.testing.assert_close(
            global_c4,
            global_c4[:, :, :1].expand_as(global_c4),
        )
        pool_refined = rows["query_global_pool8_local_c4_d32"]
        key_row = torch.arange(FRAMES * ROWS) % ROWS
        torch.testing.assert_close(
            pool_refined[..., 0],
            key_row.view(1, -1).expand(pool_refined.size(0), -1),
        )

    def test_selectors_are_deterministic(self):
        query = torch.randn(2, ROWS, 8, generator=torch.Generator().manual_seed(11))
        first, _ = select_query_only_rows(query, candidate_count=9)
        second, _ = select_query_only_rows(query, candidate_count=9)
        for name in first:
            torch.testing.assert_close(first[name], second[name])

    def test_kl_decomposition_is_finite_with_empty_clusters(self):
        generator = torch.Generator().manual_seed(17)
        query = torch.randn(2, ROWS, 8, generator=generator)
        key = torch.randn(FRAMES * ROWS, 3, 8, generator=generator)
        rows, _ = select_query_only_rows(query, candidate_count=9)
        metrics = evaluate_attention_kl(
            query,
            key,
            rows["fixed_wide"],
            column_chunk=1,
        )
        for name in ("mean_key_kl", "mean_cluster_kl", "mean_conditional_kl"):
            self.assertTrue(torch.isfinite(torch.tensor(metrics[name])).item(), name)
        self.assertLessEqual(metrics["mean_cluster_kl"], metrics["mean_key_kl"] + 1e-5)


if __name__ == "__main__":
    unittest.main()
