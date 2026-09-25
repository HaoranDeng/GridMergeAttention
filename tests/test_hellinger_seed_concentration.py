import unittest

import torch

from analysis.analyze_hellinger_seed_concentration import (
    _histogram_summary,
    _vote_shared_rows,
)


class HellingerSeedConcentrationTest(unittest.TestCase):
    def test_histogram_summary_reports_effective_support(self):
        summary = _histogram_summary(torch.tensor([2, 2, 0, 0]))
        self.assertAlmostEqual(summary["effective_support"], 2.0)
        self.assertAlmostEqual(summary["top1_mass"], 0.5)

    def test_vote_uses_adaptive_slots_only(self):
        table = torch.zeros((2, 3, 1, 1, 4), dtype=torch.long)
        table[..., 0] = torch.arange(3).view(1, 3, 1, 1)
        table[..., 1] = 7
        table[..., 2] = 11
        table[..., 3] = 19
        table[0, 0, 0, 0, 3] = 23
        rows = _vote_shared_rows(table)
        self.assertEqual(tuple(rows.shape), (1, 1, 4))
        self.assertTrue({7, 11, 19}.issubset(set(rows.flatten().tolist())))


if __name__ == "__main__":
    unittest.main()
