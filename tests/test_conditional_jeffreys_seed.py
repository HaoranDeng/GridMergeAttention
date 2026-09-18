"""CPU-only independent identities for the isolated NumPy seed reference."""

import unittest

import numpy as np

from analysis.conditional_jeffreys_seed import partition_metrics, select_jeffreys_seeds


def independent_projection(target_logits, seed_logits, singleton=-1):
    """Scalar-loop reference; no production helper or returned cells are reused."""
    def normalize(x):
        shifted = x - max(x)
        logp = shifted - np.log(np.exp(shifted).sum())
        return logp, np.exp(logp)

    seed_logp = np.asarray([normalize(row)[0] for row in seed_logits])
    cells = [[] for _ in range(len(seed_logits) + 1)]
    for key in range(seed_logits.shape[-1]):
        owner = int(np.argmax(seed_logp[:, key]))
        cells[len(seed_logits) if key == singleton else owner].append(key)
    reference = []
    for row in target_logits:
        logp, p = normalize(row)
        records = []
        conditional_sum = 0.0
        for slot, cell in enumerate(cells):
            if not cell:
                continue
            source = seed_logits[slot, cell] if slot < len(seed_logits) else np.zeros(1)
            logw, w = normalize(source)
            logpc, pc = normalize(row[cell])
            mass = p[cell].sum()
            reverse_conditional = np.sum(w * (logw - logpc))
            forward_conditional = np.sum(pc * (logpc - logw))
            conditional_sum += mass * (reverse_conditional + forward_conditional)
            local_log_partition = np.sum(w * logp[cell]) - np.sum(w * logw)
            records.append((cell, logw, local_log_partition))
        raw_beta = np.array([record[2] for record in records])
        logbeta, beta = normalize(raw_beta)
        loghat = np.empty_like(logp)
        for index, (cell, logw, _) in enumerate(records):
            loghat[cell] = logbeta[index] + logw
        phat = np.exp(loghat)
        reverse = np.sum(phat * (loghat - logp))
        forward = np.sum(p * (logp - loghat))
        reference.append((reverse, forward, conditional_sum, phat, loghat))
    return reference


class ConditionalJeffreysTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(716)

    def test_independent_reconstruction_and_conditional_identity(self):
        target = self.rng.normal(size=(7, 11)) * 2
        seeds = self.rng.normal(size=(4, 11)) * 2
        for singleton in (-1, 0, 7):
            result = partition_metrics(target, seeds, singleton, return_distributions=True)
            reference = independent_projection(target, seeds, singleton)
            for i, (reverse, forward, symmetric, phat, loghat) in enumerate(reference):
                self.assertAlmostEqual(result["reverse_gap"][i], reverse, places=12)
                self.assertAlmostEqual(result["forward_kl"][i], forward, places=12)
                self.assertAlmostEqual(result["jeffreys"][i], symmetric, places=12)
                self.assertAlmostEqual(symmetric, reverse + forward, places=12)
                np.testing.assert_allclose(result["p_hat"][i], phat, atol=2e-14, rtol=2e-14)
                np.testing.assert_allclose(result["log_p_hat"][i], loghat, atol=2e-14, rtol=2e-14)
            self.assertTrue((result["reverse_gap"] > -1e-13).all())
            np.testing.assert_allclose(result["p_hat"].sum(-1), 1, atol=1e-14)

    def test_single_cell_has_correct_kl_direction(self):
        target = np.log([[.99, .01], [.2, .8]])
        seed = np.log([[.97, .03]])
        result = partition_metrics(target, seed)
        expected = (np.exp(seed) * (seed - target)).sum(-1)
        np.testing.assert_allclose(result["reverse_gap"], expected, atol=1e-14)
        np.testing.assert_allclose(
            result["jeffreys"], result["reverse_gap"] + result["forward_kl"], atol=1e-14,
        )

    def test_all_individual_cells_including_singleton_are_exact(self):
        target = self.rng.normal(size=(8, 3)) * 20
        seeds = np.array([[4., 0., -2.], [0., 4., -2.]])
        result = partition_metrics(target, seeds, 2, return_distributions=True)
        np.testing.assert_array_equal(result["owner"], [0, 1, 2])
        for key in ("reverse_gap", "forward_kl", "jeffreys"):
            np.testing.assert_allclose(result[key], 0, atol=3e-14)
        np.testing.assert_allclose(result["p_hat"], result["p"], atol=1e-14)

    def test_single_key_and_empty_bulk_cells(self):
        result = partition_metrics([[100.], [-80.]], [[1.], [2.], [3.]], 0,
                                   return_distributions=True)
        np.testing.assert_array_equal(result["active_cells"], [False, False, False, True])
        for key in ("reverse_gap", "forward_kl", "jeffreys"):
            np.testing.assert_array_equal(result[key], [0., 0.])

    def test_duplicate_seeds_make_empty_cells_and_low_seed_slot_wins(self):
        target = self.rng.normal(size=(5, 7))
        one = self.rng.normal(size=(1, 7))
        duplicate = np.repeat(one, 4, axis=0)
        result = partition_metrics(target, duplicate, return_distributions=True)
        baseline = partition_metrics(target, one)
        np.testing.assert_array_equal(result["owner"], np.zeros(7, dtype=int))
        np.testing.assert_array_equal(result["active_cells"], [True, False, False, False, False])
        self.assertTrue(np.isneginf(result["log_weights"][1:]).all())
        for key in baseline:
            np.testing.assert_allclose(result[key], baseline[key], atol=1e-14)

    def test_ownership_full_block_before_singleton_removal(self):
        # On the complement, seed 1 would win key 1. The full-block comparison
        # assigns it to seed 0 because seed 1's mass is mostly at singleton 0.
        seeds = np.log([[.1, .4, .5], [.98, .018, .002]])
        target = np.log([[.2, .3, .5]])
        result = partition_metrics(target, seeds, 0, return_distributions=True)
        np.testing.assert_array_equal(result["owner_before_singleton"], [1, 0, 0])
        np.testing.assert_array_equal(result["owner"], [2, 0, 0])
        np.testing.assert_allclose(result["weights"][0, 1:], [4 / 9, 5 / 9])

    def test_target_and_seed_row_shift_invariance(self):
        target = self.rng.normal(size=(2, 4, 9))
        seeds = self.rng.normal(size=(2, 3, 9))
        result = partition_metrics(target, seeds, [0, -1], return_distributions=True)
        shifted = partition_metrics(
            target + self.rng.normal(size=(2, 4, 1)) * 500,
            seeds + self.rng.normal(size=(2, 3, 1)) * 500,
            [0, -1], return_distributions=True,
        )
        for key in ("reverse_gap", "forward_kl", "jeffreys", "p_hat"):
            np.testing.assert_allclose(result[key], shifted[key], atol=4e-13, rtol=4e-13)
        np.testing.assert_array_equal(result["owner"], shifted["owner"])

    def test_batch_broadcasting(self):
        target = self.rng.normal(size=(2, 1, 4, 6))
        seeds = self.rng.normal(size=(3, 2, 6))
        singleton = np.array([[0], [-1]])
        result = partition_metrics(target, seeds, singleton)
        self.assertEqual(result["jeffreys"].shape, (2, 3, 4))
        for a in range(2):
            for b in range(3):
                single = partition_metrics(target[a, 0], seeds[b], int(singleton[a, 0]))
                for key in single:
                    np.testing.assert_allclose(result[key][a, b], single[key], atol=1e-14)

    def test_peaked_finite_logits_keep_finite_owned_log_weights(self):
        target = np.array([[0., -2000., -4000.], [-4000., -2000., 0.], [0., 0., 0.]])
        seeds = np.array([[0., -1500., -3000.]])
        result = partition_metrics(target, seeds, return_distributions=True)
        self.assertTrue(np.isfinite(result["log_weights"][0]).all())
        self.assertEqual(result["weights"][0, 1], 0.)
        for key in ("reverse_gap", "forward_kl", "jeffreys", "log_p_hat"):
            self.assertTrue(np.isfinite(result[key]).all())
        reference = independent_projection(target, seeds)
        for i, (reverse, forward, symmetric, _, _) in enumerate(reference):
            self.assertAlmostEqual(result["reverse_gap"][i], reverse, places=10)
            self.assertAlmostEqual(result["forward_kl"][i], forward, places=10)
            self.assertAlmostEqual(result["jeffreys"][i], symmetric, places=10)

    def test_greedy_anchor_distinctness_and_independent_step_scores(self):
        logits = self.rng.normal(size=(2, 8, 9)) * 3
        weights = self.rng.random(size=(2, 8))
        anchors = np.array([5, 1])
        chosen = select_jeffreys_seeds(logits, anchors, weights, 4, [0, -1])
        np.testing.assert_array_equal(chosen[:, 0], anchors)
        self.assertEqual(chosen.dtype, np.int64)
        for batch in range(2):
            self.assertEqual(len(set(chosen[batch].tolist())), 4)
            for step in range(1, 4):
                previous = chosen[batch, :step]
                reference = independent_projection(
                    logits[batch], logits[batch, previous], [0, -1][batch],
                )
                gamma = np.array([item[2] for item in reference])
                priority = weights[batch] * np.maximum(gamma, 0.)
                priority[previous] = -np.inf
                self.assertEqual(chosen[batch, step], np.argmax(priority))

    def test_exact_ties_zero_weights_and_num_seeds_one(self):
        logits = np.zeros((2, 6, 4))
        np.testing.assert_array_equal(
            select_jeffreys_seeds(logits, [3, 5]), [[3, 0, 1, 2], [5, 0, 1, 2]],
        )
        random_logits = self.rng.normal(size=(6, 4))
        np.testing.assert_array_equal(
            select_jeffreys_seeds(random_logits, 4, np.zeros(6)), [4, 0, 1, 2],
        )
        np.testing.assert_array_equal(select_jeffreys_seeds(random_logits, 2, num_seeds=1), [2])

    def test_weight_scale_and_shift_preserve_greedy_selection(self):
        logits = self.rng.normal(size=(3, 7, 8))
        weights = self.rng.random(size=(3, 7))
        chosen = select_jeffreys_seeds(logits, 1, weights)
        other = select_jeffreys_seeds(
            logits + self.rng.normal(size=(3, 7, 1)) * 20,
            1, weights * np.array([1e200, 1e-200, 7.])[:, None],
        )
        np.testing.assert_array_equal(chosen, other)

    def test_jeffreys_need_not_rank_reverse_gap_or_tv_the_same(self):
        p = np.array([[.99, .01]])
        a = np.array([[.999999, .000001]])
        b = np.array([[.97, .03]])
        result_a = partition_metrics(np.log(p), np.log(a))
        result_b = partition_metrics(np.log(p), np.log(b))
        self.assertLess(result_a["reverse_gap"][0], result_b["reverse_gap"][0])
        self.assertGreater(result_a["jeffreys"][0], result_b["jeffreys"][0])
        self.assertLess(np.abs(p - a).sum(), np.abs(p - b).sum())

    def test_reject_nonfinite_invalid_shapes_and_indices(self):
        valid = np.zeros((4, 5))
        for invalid in (np.nan, np.inf, -np.inf):
            changed = valid.copy()
            changed[0, 0] = invalid
            with self.assertRaises(ValueError):
                partition_metrics(changed, valid)
            with self.assertRaises(ValueError):
                partition_metrics(valid, changed)
            with self.assertRaises(ValueError):
                select_jeffreys_seeds(changed, 0)
        for bad in (-2, 5, 1.5, True):
            with self.assertRaises(ValueError):
                partition_metrics(valid, valid, bad)
        for bad in (-1, 4, 0.5, True):
            with self.assertRaises(ValueError):
                select_jeffreys_seeds(valid, bad)
        for bad in (0, 5, 1.5, True):
            with self.assertRaises(ValueError):
                select_jeffreys_seeds(valid, 0, num_seeds=bad)
        for bad in ([1, 2], [-1, 0, 0, 0], [1, np.inf, 1, 1], [1, np.nan, 1, 1]):
            with self.assertRaises(ValueError):
                select_jeffreys_seeds(valid, 0, bad)
        for target, seeds in (([1, 2], valid), (np.empty((0, 5)), valid),
                              (valid, np.empty((0, 5))), (valid, np.zeros((2, 6))),
                              (np.zeros((2, 4, 5)), np.zeros((3, 2, 5)))):
            with self.assertRaises(ValueError):
                partition_metrics(target, seeds)


if __name__ == "__main__":
    unittest.main()
