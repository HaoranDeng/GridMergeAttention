"""Small numerical checks for the cumulant-corrected GMA derivation."""

from __future__ import annotations

import unittest

import torch

from analysis.probe_cumulant_gma_real_qkv import channel_groups, flops


class CumulantGmaReferenceTests(unittest.TestCase):
    def test_second_order_term_improves_small_query_shift(self) -> None:
        keys = torch.tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        seed = torch.zeros(2)
        query = torch.tensor([0.1, 0.0])
        seed_logits = keys @ seed
        weights = torch.softmax(seed_logits, dim=0)
        mean_key = weights @ keys
        entropy = -(weights * weights.log()).sum()
        base_log_mass = query @ mean_key + entropy
        exact_log_mass = torch.logsumexp(keys @ query, dim=0)
        centered = keys - mean_key
        variance = weights @ ((centered @ query).square())
        corrected_log_mass = base_log_mass + 0.5 * variance

        self.assertLess(
            abs(float(exact_log_mass - corrected_log_mass)),
            abs(float(exact_log_mass - base_log_mass)),
        )

    def test_rope_groups_cover_each_channel_once(self) -> None:
        groups = channel_groups(128, "rope3")
        covered = [index for start, end in groups for index in range(start, end)]
        self.assertEqual(covered, list(range(128)))
        self.assertEqual([end - start for start, end in groups], [44, 42, 42])

    def test_grouped_mass_and_value_candidate_stays_under_budget(self) -> None:
        estimate = flops(0.37906453310046345, groups=8, include_value=True)
        self.assertTrue(estimate["passes_70_percent_svg_ear_limit"])
        self.assertLess(estimate["ratio_vs_svg_ear"], 0.70)


if __name__ == "__main__":
    unittest.main()
