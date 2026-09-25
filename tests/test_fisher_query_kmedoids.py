import unittest

import numpy as np

from analysis.fisher_query_kmedoids import (
    coverage, direct_initialization, fisher_features, fisher_gs, lloyd_medoids,
)


class FisherQueryMedoidTests(unittest.TestCase):
    def test_metric_identity_and_logit_shift_invariance(self):
        z = np.random.default_rng(11).normal(size=(5, 9, 7))
        x = fisher_features(z)
        ref = z.mean(1)
        rho = np.exp(ref - np.logaddexp.reduce(ref, axis=-1)[:, None])
        for bank in range(len(z)):
            g = np.diag(rho[bank]) - np.outer(rho[bank], rho[bank])
            delta = z[bank, :, None] - z[bank, None, :]
            expected = np.einsum("ijp,pq,ijq->ij", delta, g, delta)
            actual = ((x[bank, :, None] - x[bank, None, :]) ** 2).sum(-1)
            np.testing.assert_allclose(actual, expected, atol=1e-12)
        shifted = fisher_features(z + np.arange(9)[None, :, None] * 7)
        np.testing.assert_allclose(x, shifted, atol=1e-12)

    def test_independent_scalar_fixed_point_and_monotonicity(self):
        x = np.random.default_rng(35).normal(size=(20, 12, 5))
        initial = fisher_gs(x, np.arange(20) % 12)
        for fixed in (False, True):
            result = lloyd_medoids(x, initial, fixed_first=fixed)
            self.assertTrue(result["converged"].all())
            self.assertTrue(np.all(np.diff(result["objective_history"], axis=0) <= 1e-10))
            for bank, seeds in enumerate(result["seeds"]):
                distance = np.array([[np.sum((a - x[bank, s]) ** 2) for s in seeds] for a in x[bank]])
                labels = distance.argmin(-1)
                labels[seeds] = np.arange(4)
                for slot in range(int(fixed), 4):
                    members = np.flatnonzero(labels == slot)
                    sums = {s: sum(np.sum((x[bank, i] - x[bank, s]) ** 2) for i in members)
                            for s in members}
                    self.assertLessEqual(sums[seeds[slot]], min(sums.values()) + 1e-9)
                self.assertEqual(len(set(seeds)), 4)
            if fixed:
                np.testing.assert_array_equal(result["seeds"][:, 0], initial[:, 0])

    def test_first_seed_can_move(self):
        x = np.array([[[0.], [1.], [2.], [3.], [30.], [60.], [90.]]])
        initial = np.array([[0, 4, 5, 6]])
        free = lloyd_medoids(x, initial)
        fixed = lloyd_medoids(x, initial, fixed_first=True)
        self.assertEqual(free["seeds"][0, 0], 1)
        self.assertEqual(fixed["seeds"][0, 0], 0)
        self.assertLess(coverage(x, free["seeds"])[0], coverage(x, fixed["seeds"])[0])

    def test_duplicates_degenerate_metric_and_full_count(self):
        for x in (np.zeros((3, 8, 5)), np.repeat(np.arange(8)[None, :, None] // 2, 3, axis=0)):
            for count in (1, 4, 8):
                initial = direct_initialization(x, count)
                result = lloyd_medoids(x, initial, count=count)
                self.assertTrue(np.all(np.diff(np.sort(result["seeds"], axis=-1), axis=-1) > 0))
                if count == 8:
                    np.testing.assert_array_equal(coverage(x, result["seeds"]), np.zeros(3))

    def test_bad_input_and_convergence_guard(self):
        with self.assertRaises(ValueError):
            fisher_features(np.full((1, 4, 3), np.nan))
        with self.assertRaises(ValueError):
            lloyd_medoids(np.zeros((1, 4, 3)), np.array([[0, 0, 1, 2]]))
        x = np.array([[[0.], [1.], [2.], [3.], [30.], [60.], [90.]]])
        with self.assertRaises(RuntimeError):
            lloyd_medoids(x, np.array([[0, 4, 5, 6]]), max_iterations=1)


if __name__ == "__main__":
    unittest.main()
