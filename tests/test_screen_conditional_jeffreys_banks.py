"""CPU tests for the fixed-budget objective/probe cross-audit."""
import unittest

import numpy as np

from analysis import screen_conditional_jeffreys_banks as audit


class ProbeScreenTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(13)
        self.logits = self.rng.normal(size=(2, 45, 80))
        self.rows = np.broadcast_to(np.arange(32), (2, 32)).copy()
        self.anchor = np.array([3, 7])
        self.z = audit.lse(self.logits) + 4
        self.g = self.z - 2
        self.cols = np.array([20, 21])

    def choose(self, method, logits=None, z=None, g=None):
        return audit.choose(method, self.logits if logits is None else logits,
                            self.rows, self.anchor, self.z if z is None else z,
                            self.g if g is None else g, self.cols)

    def test_four_distinct_real_rows_and_anchor(self):
        for method in audit.METHODS:
            chosen = self.choose(method)
            np.testing.assert_array_equal(chosen[:, 0], self.anchor)
            self.assertTrue(np.all(np.diff(np.sort(chosen, axis=-1), axis=-1) > 0))
            self.assertTrue(np.all((chosen >= 0) & (chosen < 32)))

    def test_mass_and_jeffreys_row_shift_invariance(self):
        shift = self.rng.normal(scale=20, size=(2, 45))
        for method in audit.METHODS:
            if method.startswith("hellinger"):
                continue
            np.testing.assert_array_equal(self.choose(method), self.choose(
                method, self.logits + shift[..., None], self.z + shift, self.g + shift))

    def test_gs_weight_scale_and_deterministic_ties(self):
        candidate = self.logits[:, :32, ::5]
        weights = self.rng.uniform(0.1, 2, size=(2, 32))
        np.testing.assert_array_equal(audit.select_span(candidate, self.anchor, weights),
            audit.select_span(candidate, self.anchor, 19 * weights))
        chosen = audit.select_span(candidate * 0, self.anchor, weights)
        np.testing.assert_array_equal(chosen, [[3, 0, 1, 2], [7, 0, 1, 2]])

    def test_hellinger_response_not_row_shift_invariant(self):
        local = np.zeros((1, 4, 16))
        before = audit.response_weights(local)
        after = audit.response_weights(local + np.arange(4)[None, :, None])
        self.assertFalse(np.allclose(before, after))

    def test_full80_evaluation_detects_unobserved_peak(self):
        logits = np.zeros((1, 45, 80))
        logits[0, 40, 79] = 12
        values = np.zeros((1, 80, 128))
        values[0, 79] = 1
        result = audit.evaluate(logits, values, np.array([[0, 1, 2, 3]]),
                                np.array([20]), audit.lse(logits) + 3)
        self.assertGreater(result["reverse_gap"][0, 40], 5)
        self.assertGreater(result["conditional_output_sse"][0, 40], 100)
        np.testing.assert_allclose(result["jeffreys"],
            result["reverse_gap"] + result["forward_kl"], atol=1e-12)

    def test_pool_uses_mass_sums_not_unweighted_prompt_means(self):
        metrics = {"rho": np.array([[0.1, 0.2], [0.8, 0.9]])}
        for name in ("reverse_gap", "forward_kl", "jeffreys", "deficit", "conditional_output_sse"):
            metrics[name] = np.array([[1., 2.], [9., 10.]])
        one = audit.aggregate(metrics, np.array([True, False]))
        two = audit.aggregate(metrics, np.array([False, True]))
        combined = audit.pool([one, two])
        expected = audit.aggregate(metrics)
        for key in expected:
            self.assertAlmostEqual(combined[key], expected[key])
        self.assertNotAlmostEqual(combined["weighted_mean_reverse_gap"],
            (one["weighted_mean_reverse_gap"] + two["weighted_mean_reverse_gap"]) / 2)

    def test_bad_method_and_insufficient_candidates_rejected(self):
        with self.assertRaises(ValueError):
            self.choose("unknown")
        with self.assertRaises(ValueError):
            audit.select_span(np.zeros((1, 3, 16)), np.array([0]), np.ones((1, 3)))

    def test_impossible_oracle_mass_rejected(self):
        with self.assertRaisesRegex(ValueError, "exceeds cached"):
            audit.evaluate(self.logits, np.zeros((2, 80, 128)),
                           np.tile(np.arange(4), (2, 1)), self.cols,
                           audit.lse(self.logits) - 0.1)


if __name__ == "__main__":
    unittest.main()
