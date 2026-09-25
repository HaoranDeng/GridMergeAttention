"""Independent raw-logit/reconstructed-KL oracles for the two-choice guard."""
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from analysis.anchor_gap_guard_seed import (
    anchor_gap_guard, anchor_gap_guard_numpy,
    quadrature_partition_gap, quadrature_partition_gap_numpy,
    uniform16_sq1_quadrature,
)


def raw_partition_oracle(z, slots, singleton, mu):
    """Use raw logits + measure entropy, then independently reconstruct KL.

Unlike the implementation, do not score normalized log-probability tangents.
Build the final partition, route over its corrected raw logits, and evaluate
KL of the reconstructed BIN distribution against the exact target bins.
"""
    z, mu = np.asarray(z, dtype=np.float64), np.asarray(mu, dtype=np.float64)
    weighted = z + np.log(mu)
    log_z = np.logaddexp.reduce(weighted, axis=-1)
    logp = weighted - log_z[:, None]
    p = np.exp(logp)
    owners = np.argmax(p[np.asarray(slots)], axis=0)
    owners[singleton] = len(slots)
    weights, bounds = [], []
    for cell, slot in enumerate(slots):
        cols = np.flatnonzero(owners == cell)
        if not len(cols):
            continue
        unnormalized = mu[cols] * np.exp(z[slot, cols] - z[slot, cols].max())
        weight = unnormalized / unnormalized.sum()
        effective_entropy = -np.sum(weight * np.log(weight / mu[cols]))
        bounds.append(z[:, cols] @ weight + effective_entropy)
        full_weight = np.zeros(z.shape[-1])
        full_weight[cols] = weight
        weights.append(full_weight)
    singleton_weight = np.zeros(z.shape[-1])
    singleton_weight[singleton] = 1.0
    weights.append(singleton_weight)
    bounds.append(z[:, singleton])
    bounds = np.stack(bounds, axis=-1)
    compressed_log_z = np.logaddexp.reduce(bounds, axis=-1)
    route = np.exp(bounds - compressed_log_z[:, None])
    reconstruction = route @ np.asarray(weights)
    terms = np.zeros_like(reconstruction)
    mask = reconstruction > 0
    terms[mask] = reconstruction[mask] * (np.log(reconstruction[mask]) - logp[mask])
    return {"gap": log_z - compressed_log_z, "direct_kl": terms.sum(axis=-1),
            "owner": owners, "reconstruction": reconstruction}


def guard_oracle(z, incumbent, singleton, mu):
    prefix_gap = raw_partition_oracle(z, incumbent[:3], singleton, mu)["gap"]
    score = prefix_gap.copy()
    score[incumbent[:3]] = -np.inf
    challenger = int(np.argmax(score))
    trial = list(incumbent[:3]) + [challenger]
    base_gap = raw_partition_oracle(z, incumbent, singleton, mu)["gap"]
    trial_gap = raw_partition_oracle(z, trial, singleton, mu)["gap"]
    accepted = trial_gap.mean() < base_gap.mean()
    return {"seeds": np.asarray(trial if accepted else incumbent),
            "challenger": challenger, "accepted": accepted,
            "prefix_gap": prefix_gap, "base_gap": base_gap, "trial_gap": trial_gap}


def fixtures():
    # Bank5 accepts slot7; bank6 proposes slot6 but must reject its worse gap.
    return np.random.default_rng(485).normal(size=(64, 8, 7)) * 2


class AnchorGapGuardNumpyTests(unittest.TestCase):
    def test_exact_gap_matches_raw_partition_and_direct_reverse_kl(self):
        rng = np.random.default_rng(1361)
        for column in (0, 5, 17, 79):
            spec = uniform16_sq1_quadrature(column)
            z = rng.normal(size=(32, len(spec["measures"]))) * 3
            chosen = [7, 3, 21, 11]
            kw = {"singleton_index": spec["singleton_index"], "measures": spec["measures"]}
            got = quadrature_partition_gap_numpy(z, chosen, **kw, return_details=True)
            expected = raw_partition_oracle(z, chosen, spec["singleton_index"], spec["measures"])
            np.testing.assert_allclose(got["gap"], expected["gap"], atol=5e-14, rtol=1e-13)
            np.testing.assert_allclose(got["gap"], expected["direct_kl"], atol=5e-14, rtol=1e-13)
            np.testing.assert_allclose(got["reconstruction"], expected["reconstruction"], atol=2e-14)
            np.testing.assert_array_equal(got["owner"], expected["owner"])
            np.testing.assert_allclose(got["reconstruction"].sum(-1), 1, atol=2e-14)

    def test_accept_reject_and_incumbent_challenger_cases_are_real(self):
        z = fixtures()
        got = anchor_gap_guard_numpy(z, [0, 1, 2, 3], 0, measures=np.ones(7))
        self.assertTrue(got["accepted"][5])
        self.assertEqual(got["challenger"][5], 7)
        self.assertFalse(got["accepted"][6])
        self.assertEqual(got["challenger"][6], 6)
        self.assertGreater(got["trial_mean"][6], got["base_mean"][6])
        np.testing.assert_array_equal(got["seeds"][6], [0, 1, 2, 3])
        self.assertEqual(got["challenger"][0], 3)  # The old fourth is eligible.
        self.assertFalse(got["accepted"][0])
        for bank, result in zip(z, got["seeds"]):
            expected = guard_oracle(bank, [0, 1, 2, 3], 0, np.ones(7))
            np.testing.assert_array_equal(result, expected["seeds"])
        self.assertTrue((np.where(got["accepted"], got["trial_mean"], got["base_mean"])
                         <= got["base_mean"]).all())

    def test_all_gap_vectors_match_independent_oracle(self):
        z = fixtures()[:20]
        got = anchor_gap_guard_numpy(z, [0, 1, 2, 3], 0, measures=np.ones(7))
        for i, bank in enumerate(z):
            expected = guard_oracle(bank, [0, 1, 2, 3], 0, np.ones(7))
            for key in ("prefix_gap", "base_gap", "trial_gap"):
                self.assertEqual(got[key].dtype, np.float64)
                np.testing.assert_allclose(got[key][i], expected[key], atol=1e-14, rtol=1e-13)
            self.assertEqual(got["challenger"][i], expected["challenger"])
            self.assertEqual(bool(got["accepted"][i]), expected["accepted"])

    def test_width80_quadrature_all_columns_no_duplicate_singleton(self):
        for column in range(80):
            spec = uniform16_sq1_quadrature(column)
            columns, mu, sq = spec["probe_columns"], spec["measures"], spec["singleton_index"]
            np.testing.assert_array_equal(columns[:16], np.arange(0, 80, 5))
            self.assertEqual(len(columns), 16 if column % 5 == 0 else 17)
            self.assertEqual(len(np.unique(columns)), len(columns))
            self.assertEqual(columns[sq], column)
            self.assertEqual(mu[sq], 1.0)
            self.assertAlmostEqual(float(mu.sum()), 80.0)
            np.testing.assert_array_equal(np.delete(mu, sq), 79.0 / (len(mu) - 1))

    def test_integer_measures_equal_explicit_repeated_original_keys(self):
        z = np.random.default_rng(83).normal(size=(7, 4)) * 2
        mu = np.array([1, 2, 3, 4])
        weighted = quadrature_partition_gap_numpy(z, [0, 2, 4, 6], 0, measures=mu)
        expanded = np.repeat(z, mu, axis=-1)
        original = quadrature_partition_gap_numpy(expanded, [0, 2, 4, 6], 0,
                                                 measures=np.ones(mu.sum()))
        np.testing.assert_allclose(weighted, original, atol=3e-15, rtol=2e-13)

    def test_owners_are_computed_before_singleton_peeling(self):
        p = np.array([[.95, .03, .02], [.01, .45, .54], [.2, .2, .6]])
        result = quadrature_partition_gap_numpy(np.log(p), [0, 1], 0,
                                               measures=np.ones(3), return_details=True)
        np.testing.assert_array_equal(result["owner"], [2, 1, 1])
        normalized_bulk = p[:2, 1:] / p[:2, 1:].sum(-1, keepdims=True)
        np.testing.assert_array_equal(normalized_bulk.argmax(0), [0, 1])

    def test_gauge_broadcasting_shapes_and_input_preservation(self):
        rng = np.random.default_rng(124)
        z = rng.normal(size=(2, 3, 8, 7))
        base = np.array([[[7, 2, 4, 0]], [[1, 5, 3, 6]]])
        sq = np.array([[0], [6]])
        mu = np.ones((2, 1, 7)) * 3
        mu[0, 0, 0], mu[1, 0, 6] = 1, 1
        copies = (z.copy(), base.copy(), sq.copy(), mu.copy())
        got = anchor_gap_guard_numpy(z, base, sq, measures=mu)
        shifted = anchor_gap_guard_numpy(z + rng.integers(-20, 20, size=(2, 3, 8, 1)),
                                        base, sq, measures=mu)
        self.assertEqual(got["seeds"].shape, (2, 3, 4))
        self.assertEqual(got["accepted"].dtype, np.bool_)
        self.assertEqual(got["seeds"].dtype, np.int64)
        np.testing.assert_array_equal(got["seeds"], shifted["seeds"])
        np.testing.assert_array_equal(got["seeds"][..., :3], np.broadcast_to(base, (2, 3, 4))[..., :3])
        self.assertTrue((np.diff(np.sort(got["seeds"], axis=-1)) > 0).all())
        for key in ("prefix_gap", "base_gap", "trial_gap"):
            np.testing.assert_allclose(got[key], shifted[key], atol=1e-14)
        for original, untouched in zip((z, base, sq, mu), copies):
            np.testing.assert_array_equal(original, untouched)

    def test_exact_ties_keep_incumbent_and_choose_lowest_unused_prefix_slot(self):
        for probes in (4, 16, 17):
            z = np.zeros((8, probes))
            got = anchor_gap_guard_numpy(z, [7, 5, 2, 6], 0, measures=np.ones(probes))
            self.assertEqual(int(got["challenger"]), 0)
            self.assertFalse(got["accepted"])
            np.testing.assert_array_equal(got["seeds"], [7, 5, 2, 6])
            np.testing.assert_array_equal(got["base_gap"], got["trial_gap"])

    def test_raw_negative_roundoff_ranking_and_strict_acceptance_contract(self):
        prefix = np.array([0., 0., 0., -5e-16, -1e-16, -3e-16])
        base = np.zeros(6)
        trial = np.full(6, -1e-16)
        with patch("analysis.anchor_gap_guard_seed._numpy_gap", side_effect=[prefix, base, trial]):
            got = anchor_gap_guard_numpy(np.zeros((6, 4)), [0, 1, 2, 5], 0,
                                        measures=np.ones(4))
        self.assertEqual(int(got["challenger"]), 4)  # Clipping would pick slot3.
        self.assertTrue(got["accepted"])  # No hidden acceptance tolerance.
        np.testing.assert_array_equal(got["prefix_gap"], prefix)

    def test_identical_tuple_rejects_one_ulp_false_gain_without_altering_raw_gaps(self):
        # BR diagnostic bank13079 had incumbent [23,22,28,20] and challenger20.
        # Inject a one-ULP evaluation difference, not a different objective.
        # The two other banks ensure different tuples still use strict <.
        incumbents = np.array([[23, 22, 28, 20], [23, 22, 28, 19], [23, 22, 28, 18]])
        prefix = np.zeros((3, 32))
        prefix[:, 20] = 2.0
        base = np.ones((3, 32))
        trial = np.stack((np.nextafter(base[0], 0.), np.nextafter(base[1], 0.),
                          np.nextafter(base[2], np.inf)))
        with patch("analysis.anchor_gap_guard_seed._numpy_gap", side_effect=[prefix, base, trial]):
            got = anchor_gap_guard_numpy(np.zeros((3, 32, 17)), incumbents, 16,
                                        measures=[79 / 16] * 16 + [1.])
        np.testing.assert_array_equal(got["challenger"], [20, 20, 20])
        np.testing.assert_array_equal(got["accepted"], [False, True, False])
        expected = incumbents.copy()
        expected[1, 3] = 20
        np.testing.assert_array_equal(got["seeds"], expected)
        self.assertLess(got["trial_mean"][0], got["base_mean"][0])
        for key, original in (("prefix_gap", prefix), ("base_gap", base), ("trial_gap", trial)):
            np.testing.assert_array_equal(got[key], original)

    def test_nonuniform_identical_profiles_and_extreme_finite_logits(self):
        profile = np.array([5., -2., 3., 0.])
        z = np.broadcast_to(profile, (8, 4)).copy()
        got = anchor_gap_guard_numpy(z, [0, 2, 4, 6], 0, measures=[1, 2, 3, 4])
        self.assertEqual(int(got["challenger"]), 1)
        self.assertFalse(got["accepted"])
        z[:, 0] = -1000
        z[:, 1] = 1000
        saturated = anchor_gap_guard_numpy(z, [0, 2, 4, 6], 0, measures=[1, 2, 3, 4])
        self.assertTrue(np.isfinite(saturated["base_gap"]).all())
        self.assertFalse(saturated["accepted"])

    def test_input_guards(self):
        z = np.zeros((6, 4))
        invalid = [
            (z, [0, 0, 2, 3], 0, np.ones(4)),
            (z, [0, 1, 2, 6], 0, np.ones(4)),
            (z, [0., 1., 2., 3.], 0, np.ones(4)),
            (z, [True, False, True, False], 0, np.ones(4)),
            (z, [0, 1, 2], 0, np.ones(4)),
            (z, [0, 1, 2, 3], -1, np.ones(4)),
            (z, [0, 1, 2, 3], 0., np.ones(4)),
            (z, [0, 1, 2, 3], True, np.ones(4)),
            (z, [0, 1, 2, 3], 0, [2, 1, 1, 1]),
            (z, [0, 1, 2, 3], 0, [1, 0, 1, 1]),
            (z, [0, 1, 2, 3], 0, [1, np.inf, 1, 1]),
            (z, [0, 1, 2, 3], 0, [1, 1]),
            (z + np.nan, [0, 1, 2, 3], 0, np.ones(4)),
            (np.zeros((0, 6, 4)), [0, 1, 2, 3], 0, np.ones(4)),
        ]
        for logits, slots, sq, mu in invalid:
            with self.subTest(slots=slots, sq=sq, mu=mu):
                with self.assertRaises(ValueError):
                    anchor_gap_guard_numpy(logits, slots, sq, measures=mu)
        with self.assertRaises(TypeError):
            anchor_gap_guard_numpy(z.astype(int), [0, 1, 2, 3], 0, measures=np.ones(4))
        for column in (-1, 80, True, 1.0, [1]):
            with self.assertRaises(ValueError): uniform16_sq1_quadrature(column)


@unittest.skipIf(torch is None, "Torch unavailable in local runtime; run actual CPU tests remotely")
class AnchorGapGuardTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_all_scores_indices_and_acceptance_match_independent_numpy(self):
        z = fixtures()
        got = anchor_gap_guard(torch.from_numpy(z), [0, 1, 2, 3], 0, measures=np.ones(7))
        expected = anchor_gap_guard_numpy(z, [0, 1, 2, 3], 0, measures=np.ones(7))
        for key in ("seeds", "challenger", "accepted"):
            np.testing.assert_array_equal(got[key].numpy(), expected[key])
        for key in ("prefix_gap", "base_gap", "trial_gap", "base_mean", "trial_mean"):
            self.assertEqual(got[key].dtype, torch.float64)
            np.testing.assert_allclose(got[key].numpy(), expected[key], atol=2e-14, rtol=2e-13)
        self.assertEqual(got["seeds"].dtype, torch.int64)
        self.assertEqual(got["accepted"].dtype, torch.bool)

    def test_quadrature_16_and_17_bins_kl_reconstruction(self):
        rng = np.random.default_rng(613)
        for column in (0, 5, 7, 79):
            spec = uniform16_sq1_quadrature(column)
            z = rng.normal(size=(4, 32, len(spec["measures"]))) * 2
            # FP32 input is promoted; compare against exactly those FP32 values.
            tz = torch.tensor(z, dtype=torch.float32)
            result = quadrature_partition_gap(tz, [0, 3, 7, 19], spec["singleton_index"],
                                              measures=spec["measures"], return_details=True)
            for i, bank in enumerate(tz.double().numpy()):
                oracle = raw_partition_oracle(bank, [0, 3, 7, 19], spec["singleton_index"], spec["measures"])
                np.testing.assert_allclose(result["gap"][i].numpy(), oracle["direct_kl"], atol=2e-14)
                np.testing.assert_allclose(result["reconstruction"][i].numpy(), oracle["reconstruction"], atol=2e-14)

    def test_fractional_python_measures_keep_original_double_precision(self):
        z = np.random.default_rng(244).normal(size=(6, 32, 16)) * 2
        measures = [1.0] + [79.0 / 15] * 15
        got = anchor_gap_guard(torch.from_numpy(z), [0, 3, 7, 19], 0, measures=measures)
        expected = anchor_gap_guard_numpy(z, [0, 3, 7, 19], 0, measures=measures)
        exact_tensor = anchor_gap_guard(torch.from_numpy(z), [0, 3, 7, 19], 0,
                                        measures=torch.tensor(measures, dtype=torch.float64))
        for key in ("prefix_gap", "base_gap", "trial_gap"):
            self.assertTrue(torch.equal(got[key], exact_tensor[key]))
            np.testing.assert_allclose(got[key].numpy(), expected[key], atol=2e-14, rtol=2e-13)
        for key in ("seeds", "challenger", "accepted"):
            np.testing.assert_array_equal(got[key].numpy(), expected[key])
        for invalid_measure in ([True] * 16, [1.0 + 0j] * 16):
            with self.assertRaises(ValueError):
                anchor_gap_guard(torch.from_numpy(z), [0, 3, 7, 19], 0, measures=invalid_measure)

    def test_broadcasting_empty_cells_ties_and_input_preservation(self):
        z = torch.zeros((2, 3, 8, 4), dtype=torch.float64)
        base = torch.tensor([[[7, 5, 2, 6]], [[6, 5, 4, 3]]])
        old_z, old_base = z.clone(), base.clone()
        got = anchor_gap_guard(z, base, 0, measures=[1, 2, 3, 4])
        self.assertEqual(tuple(got["seeds"].shape), (2, 3, 4))
        self.assertEqual(got["challenger"].tolist(), [[0, 0, 0], [0, 0, 0]])
        self.assertFalse(bool(got["accepted"].any()))
        self.assertTrue(torch.equal(got["seeds"], base.expand(2, 3, 4)))
        self.assertTrue(torch.equal(z, old_z))
        self.assertTrue(torch.equal(base, old_base))

    def test_gauge_shift_and_raw_negative_ranking_contract(self):
        z = torch.from_numpy(fixtures()[:12])
        old = anchor_gap_guard(z, [0, 1, 2, 3], 0, measures=np.ones(7))
        shifted = anchor_gap_guard(z + torch.arange(8)[None, :, None], [0, 1, 2, 3], 0,
                                   measures=np.ones(7))
        self.assertTrue(torch.equal(old["seeds"], shifted["seeds"]))
        for key in ("prefix_gap", "base_gap", "trial_gap"):
            torch.testing.assert_close(old[key], shifted[key], atol=2e-14, rtol=2e-13)
        prefix = torch.tensor([0., 0., 0., -5e-16, -1e-16, -3e-16], dtype=torch.float64)
        base, trial = torch.zeros(6, dtype=torch.float64), torch.full((6,), -1e-16, dtype=torch.float64)
        with patch("analysis.anchor_gap_guard_seed._torch_gap", side_effect=[prefix, base, trial]):
            got = anchor_gap_guard(torch.zeros((6, 4)), [0, 1, 2, 5], 0, measures=np.ones(4))
        self.assertEqual(int(got["challenger"]), 4)
        self.assertTrue(got["accepted"])

    def test_identical_tuple_rejects_one_ulp_false_gain_without_altering_raw_gaps(self):
        incumbents = torch.tensor([[23, 22, 28, 20], [23, 22, 28, 19], [23, 22, 28, 18]])
        prefix = torch.zeros((3, 32), dtype=torch.float64)
        prefix[:, 20] = 2.0
        base = torch.ones((3, 32), dtype=torch.float64)
        trial = torch.stack((torch.nextafter(base[0], torch.zeros_like(base[0])),
                             torch.nextafter(base[1], torch.zeros_like(base[1])),
                             torch.nextafter(base[2], torch.full_like(base[2], float("inf")))))
        with patch("analysis.anchor_gap_guard_seed._torch_gap", side_effect=[prefix, base, trial]):
            got = anchor_gap_guard(torch.zeros((3, 32, 17)), incumbents, 16,
                                   measures=[79 / 16] * 16 + [1.])
        self.assertEqual(got["challenger"].tolist(), [20, 20, 20])
        self.assertEqual(got["accepted"].tolist(), [False, True, False])
        expected = incumbents.clone()
        expected[1, 3] = 20
        self.assertTrue(torch.equal(got["seeds"], expected))
        self.assertLess(got["trial_mean"][0], got["base_mean"][0])
        for key, original in (("prefix_gap", prefix), ("base_gap", base), ("trial_gap", trial)):
            self.assertTrue(torch.equal(got[key], original))

    def test_validation_and_extreme_finite_inputs(self):
        z = torch.zeros((8, 4), dtype=torch.float64)
        for slots in ([0, 0, 2, 3], [0, 1, 2, 8], [0., 1., 2., 3.], [0, 1, 2]):
            with self.assertRaises(ValueError): anchor_gap_guard(z, slots, 0, measures=[1, 1, 1, 1])
        for sq in (-1, 4, True, 0.):
            with self.assertRaises(ValueError): anchor_gap_guard(z, [0, 1, 2, 3], sq, measures=[1, 1, 1, 1])
        for mu in ([2, 1, 1, 1], [1, 0, 1, 1], [1, float("nan"), 1, 1], [1, 1]):
            with self.assertRaises(ValueError): anchor_gap_guard(z, [0, 1, 2, 3], 0, measures=mu)
        with self.assertRaises(ValueError): anchor_gap_guard(z + float("inf"), [0, 1, 2, 3], 0, measures=[1, 1, 1, 1])
        with self.assertRaises(TypeError): anchor_gap_guard(z.long(), [0, 1, 2, 3], 0, measures=[1, 1, 1, 1])
        z[:, 0], z[:, 1] = -1000., 1000.
        got = anchor_gap_guard(z, [0, 1, 2, 3], 0, measures=[1, 1, 1, 1])
        self.assertTrue(bool(torch.isfinite(got["base_gap"]).all()))
        self.assertFalse(got["accepted"])


if __name__ == "__main__":
    unittest.main()
