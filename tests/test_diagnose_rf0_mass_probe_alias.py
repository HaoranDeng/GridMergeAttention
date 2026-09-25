"""Pure CPU NumPy checks; no Torch import, captured files or GPU execution."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("probe_alias", ROOT / "analysis/diagnose_rf0_mass_probe_alias.py")
alias = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alias)


def lse(x):
    maximum = x.max(axis=1)
    return maximum + np.log(np.exp(x - maximum[:, None]).sum(axis=1))


def partitions(logits, block_start=0):
    phase, probes = alias.historical_indices()
    local = np.arange(block_start, block_start + alias.COLS)
    return (lse(logits), lse(logits[:, phase]), lse(logits[:, probes]),
            lse(logits[:, local]), lse(logits[:, local[::5]]))


class ProbeAliasTests(unittest.TestCase):
    def test_historical_phase_identity(self):
        phase, probes = alias.historical_indices()
        self.assertEqual(len(phase), 15120)
        self.assertEqual(probes.tolist(), (np.arange(256) * 295).tolist())
        self.assertTrue(np.isin(probes, phase).all())
        self.assertEqual(int(probes[-1]), 75225)

    def test_uniform_logits_exact_quadrature(self):
        logs = partitions(np.zeros((3, alias.N)))
        result = alias.diagnose_bank(*logs, [0, 22, 44])
        self.assertLess(result["pi_tv"], 1e-14)
        self.assertLess(result["candidate_centered_log_error_rms"], 1e-14)
        self.assertLess(abs(result["mean_log_error"]), 1e-12)
        self.assertEqual(result["estimated_mass_above_one_count"], 0)
        self.assertAlmostEqual(np.exp(result["max_log_estimated_mass"]), 80 / 75600)

    def test_direct_mass_decomposition_and_pi(self):
        logits = np.random.default_rng(123).normal(size=(4, alias.N)) * 2
        logs = partitions(logits, (10 * 45 + 22) * 80)
        result = alias.diagnose_bank(*logs, [2, 5, 12, 44])
        z, z0, zg, zb, zb0 = logs
        rho, estimated = np.exp(zb - z), (256 / 15120) * np.exp(zb0 - zg)
        expected_tv = .5 * abs(rho / rho.sum() - estimated / estimated.sum()).sum()
        self.assertAlmostEqual(result["pi_tv"], expected_tv, places=13)
        self.assertLess(result["decomposition_max_abs_error"], 1e-12)
        np.testing.assert_allclose(np.log(estimated / rho),
            np.log(np.exp(zb0 - zb) / np.exp(z0 - z)) - np.log((15120 / 256) * np.exp(zg - z0)), atol=1e-12)

    def test_per_query_logit_shift_invariance(self):
        logits = np.random.default_rng(4).normal(size=(3, alias.N))
        a = alias.diagnose_bank(*partitions(logits), [0, 3, 9])
        b = alias.diagnose_bank(*partitions(logits + np.array([-500., 0., 500.])[:, None]), [0, 3, 9])
        for key in ("pi_tv", "candidate_centered_log_error_rms", "mean_log_error"):
            self.assertAlmostEqual(a[key], b[key], places=11)

    def test_common_error_can_be_huge_without_candidate_ranking_error(self):
        logits = np.zeros((3, alias.N))
        # All candidate queries miss the same off-phase peak.
        logits[:, 1001] = 20
        result = alias.diagnose_bank(*partitions(logits), [0, 22, 44])
        self.assertGreater(result["mean_log_error"], 5)
        self.assertLess(result["candidate_centered_log_error_rms"], 1e-14)
        self.assertLess(result["pi_tv"], 1e-14)
        self.assertFalse(result["estimated_top_outside_exact_max_set"])

    def test_out_of_phase_global_peak_reverses_mass_rank(self):
        logits = np.zeros((2, alias.N))
        logits[0, :80] = 2
        logits[0, 1001] = 20
        result = alias.diagnose_bank(*partitions(logits), [2, 7])
        self.assertEqual(result["exact_top_row"], 7)
        self.assertEqual(result["estimated_top_row"], 2)
        self.assertTrue(result["top_row_flip"])
        self.assertGreater(result["pi_tv"], .7)

    def test_unseen_local_phase_peak_can_make_estimated_mass_exceed_one(self):
        logits = np.zeros((2, alias.N))
        logits[0, 5] = 20  # In P16, but not in G256.
        result = alias.diagnose_bank(*partitions(logits), [0, 1])
        self.assertEqual(result["estimated_mass_above_one_count"], 1)
        self.assertGreater(result["max_log_estimated_mass"], 0)

    def test_shared_phase_error_can_cancel(self):
        logits = np.zeros((2, alias.N))
        # Suppress the same exact proportion of phase/nonphase mass locally
        # and globally. G is uniform on the phase, so mass ratios stay exact.
        mask = np.arange(alias.N) % 5 != 0
        logits[0, mask], logits[1, mask] = 3, -2
        result = alias.diagnose_bank(*partitions(logits), [0, 1])
        self.assertLess(abs(result["mean_log_error"]), 1e-12)
        self.assertLess(result["pi_tv"], 1e-12)

    def test_streaming_chunk_subsets_cover_each_partition_once(self):
        gathered = [[] for _ in range(3 + 2 * len(alias.BLOCKS))]
        for start in range(0, alias.N, 4093):
            stop = min(start + 4093, alias.N)
            for index, subset in enumerate(alias.key_chunk_plan(start, stop)):
                gathered[index].extend((start + subset).tolist())
        phase, probes = alias.historical_indices()
        expected = [np.arange(alias.N), phase, probes]
        for frame, row in alias.BLOCKS:
            block = np.arange((frame * 45 + row) * 80, (frame * 45 + row + 1) * 80)
            expected.extend((block, block[::5]))
        for actual, wanted in zip(gathered, expected):
            np.testing.assert_array_equal(actual, wanted)

    def test_streaming_lse_matches_direct_partitions(self):
        logits = np.random.default_rng(9).normal(size=(2, alias.N)) * 4
        cache = np.full((2, 3 + 2 * len(alias.BLOCKS)), -np.inf)
        for start in range(0, alias.N, 4093):
            stop = min(start + 4093, alias.N)
            for index, subset in enumerate(alias.key_chunk_plan(start, stop)):
                if len(subset):
                    cache[:, index] = np.logaddexp(cache[:, index], lse(logits[:, start + subset]))
        for block_index, (frame, row) in enumerate(alias.BLOCKS):
            direct = partitions(logits, (frame * 45 + row) * 80)
            for actual_column, expected in zip((0, 1, 2, 3 + 2 * block_index, 4 + 2 * block_index), direct):
                np.testing.assert_allclose(cache[:, actual_column], expected, atol=1e-12)

    def test_aggregate_preserves_counts_and_scopes(self):
        uniform = alias.diagnose_bank(*partitions(np.zeros((2, alias.N))), [0, 1])
        peaked = np.zeros((2, alias.N))
        peaked[0, 5] = 20
        outlier = alias.diagnose_bank(*partitions(peaked), [0, 1])
        result = alias.aggregate([uniform, outlier])
        self.assertEqual(result["sampled_banks"], 2)
        self.assertEqual(result["candidate_block_observations"], 4)
        self.assertEqual(result["estimated_mass_above_one_fraction"], .25)
        self.assertAlmostEqual(result["mean_pi_tv"], outlier["pi_tv"] / 2)
        self.assertIn("mean_log_error", result)
        self.assertNotIn("mean_mean_log_error", result)

    def test_invalid_partitions_rejected(self):
        logs = list(partitions(np.zeros((2, alias.N))))
        with self.assertRaises(ValueError):
            alias.diagnose_bank(*logs, [0, 0])
        logs[2] = logs[0] + 1
        with self.assertRaises(ValueError):
            alias.diagnose_bank(*logs, [0, 1])


if __name__ == "__main__":
    unittest.main()
