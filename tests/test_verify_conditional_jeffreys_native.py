"""CPU-only parity-auditor tests; no Torch or Triton imports are necessary."""
import math
import unittest

import numpy as np

from analysis.conditional_jeffreys_seed import partition_metrics, select_jeffreys_seeds
from analysis.verify_conditional_jeffreys_native import (
    ABS_TOL, audit_all_rows, audit_recorded_intermediates, audit_score_arrays, bank_coordinates,
    reference_logits_from_vectors,
)


class NativeJeffreysAuditTest(unittest.TestCase):
    def fixture(self, *, uniform=False):
        rng = np.random.default_rng(701)
        logits = np.zeros((3, 6, 16)) if uniform else rng.normal(size=(3, 6, 16))
        norm = rng.normal(size=(3, 6))
        rows = np.tile(np.arange(0, 30, 5), (3, 1))
        anchor = np.array([0, 2, 4])
        column = np.array([0, 1, 75])
        local_z = np.logaddexp.reduce(logits, axis=-1) / math.log(2)
        mass = local_z - norm
        weight = np.exp2(mass - mass.max(axis=-1, keepdims=True))
        singleton = np.where(column % 5 == 0, column // 5, -1)
        slots = select_jeffreys_seeds(logits, anchor, weight, singleton_index=singleton)
        chosen = rows[np.arange(3)[:, None], slots]
        scores = self.scores(logits, weight, slots, singleton)
        return logits, norm, rows, chosen, scores, column, anchor

    def scores(self, logits, weight, slots, singleton):
        result = np.empty((len(logits), 3, logits.shape[1]))
        for step in range(3):
            prefix = logits[np.arange(len(logits))[:, None], slots[:, :step + 1]]
            gamma = partition_metrics(logits, prefix, singleton)["jeffreys"] / math.log(2)
            result[:, step] = weight * np.maximum(gamma, 0)
        return result

    def test_exact_reference_passes_all_scores(self):
        report, arrays = audit_score_arrays(*self.fixture())
        self.assertTrue(report["valid"])
        self.assertEqual(report["checked_candidate_scores"], 54)
        np.testing.assert_allclose(arrays["reference_rank_defect"], 0, atol=1e-14)

    def test_wrong_base_units_fail(self):
        values = list(self.fixture())
        values[4] = values[4] * math.log(2)
        report, _ = audit_score_arrays(*values)
        self.assertFalse(report["valid"])
        self.assertGreater(report["score_comparison"]["failed_values"], 0)

    def test_corruption_in_used_candidate_is_not_masked_away(self):
        values = list(self.fixture())
        values[4][0, 0, values[6][0]] += 1
        report, _ = audit_score_arrays(*values)
        self.assertFalse(report["valid"])
        self.assertEqual(report["score_comparison"]["failed_values"], 1)

    def test_actual_prefix_replay_detects_rank_failure(self):
        values = list(self.fixture())
        values[3][0, [1, 3]] = values[3][0, [3, 1]]
        slots = (values[3][:, :, None] == values[2][:, None, :]).argmax(axis=-1)
        local_z = np.logaddexp.reduce(values[0], axis=-1) / math.log(2)
        logmass = local_z - values[1]
        weight = np.exp2(logmass - logmass.max(axis=-1, keepdims=True))
        singleton = np.where(values[5] % 5 == 0, values[5] // 5, -1)
        values[4] = self.scores(values[0], weight, slots, singleton)
        report, _ = audit_score_arrays(*values)
        self.assertTrue(report["score_comparison"]["valid"])
        self.assertFalse(report["rank_comparison"]["valid"])

    def test_tied_different_choices_use_rank_defect(self):
        values = list(self.fixture(uniform=True))
        values[3][0, [1, 2]] = values[3][0, [2, 1]]
        report, _ = audit_score_arrays(*values)
        self.assertTrue(report["valid"])
        self.assertGreater(report["rank_comparison"]["nonidentical_reference_argmax_choices"], 0)

    def test_small_fixed_error_passes_larger_fails(self):
        values = list(self.fixture())
        values[4] = values[4] + ABS_TOL / 2
        self.assertTrue(audit_score_arrays(*values)[0]["valid"])
        values[4] = values[4] + 1
        self.assertFalse(audit_score_arrays(*values)[0]["valid"])

    def test_shift_logits_and_normalizer_preserves_weights(self):
        values = list(self.fixture())
        shift = np.array([[1], [2], [-3]]) + np.arange(6)[None, :] / 10
        values[0] = values[0] + shift[..., None]
        values[1] = values[1] + shift / math.log(2)
        self.assertTrue(audit_score_arrays(*values)[0]["valid"])

    def test_invalid_selected_rows_return_failed_report(self):
        for change in ("anchor", "duplicate", "outside"):
            values = list(self.fixture())
            if change == "anchor":
                values[3][0, [0, 1]] = values[3][0, [1, 0]]
            elif change == "duplicate":
                values[3][0, 1] = values[3][0, 0]
            else:
                values[3][0, 1] = 44
            report, arrays = audit_score_arrays(*values)
            self.assertFalse(report["valid"])
            self.assertEqual(arrays, {})

    def test_nonfinite_and_invalid_geometry_rejected(self):
        for index in (0, 1, 4):
            values = list(self.fixture())
            values[index].flat[0] = np.nan
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                audit_score_arrays(*values)
        values = list(self.fixture())
        values[2][0, 1] = values[2][0, 0]
        with self.assertRaisesRegex(ValueError, "geometry"):
            audit_score_arrays(*values)

    def test_fixed_bank_coordinates(self):
        coords = bank_coordinates()
        self.assertEqual(coords.shape, (600, 4))
        np.testing.assert_array_equal(coords[0], [0, 0, 0, 0])
        np.testing.assert_array_equal(coords[-1], [3, 76, 20, 44])
        self.assertEqual(len(np.unique(coords, axis=0)), 600)

    def test_reference_accumulates_original_values_in_fp64(self):
        # Every source value is exactly BF16-representable. FP32 reduction can
        # erase the unit between opposite large terms; FP64 retains it.
        q = np.asarray([[[2 ** 27, 1, -(2 ** 27)]]], dtype=np.float32)
        k = np.ones((1, 1, 3), dtype=np.float32)
        expected = reference_logits_from_vectors(q, k, 0.5)
        self.assertEqual(expected.dtype, np.float64)
        np.testing.assert_array_equal(expected, [[[0.5]]])
        self.assertEqual(float(np.sum(q * k, dtype=np.float32)), 0.0)

    def test_reference_matches_independent_scalar_products(self):
        rng = np.random.default_rng(74)
        q = rng.integers(-20, 21, size=(2, 4, 7)).astype(np.float32) / 8
        k = rng.integers(-20, 21, size=(2, 3, 7)).astype(np.float32) / 16
        expected = np.asarray([[[sum(float(x) * float(y) for x, y in zip(qb, kb)) / 3
                                for kb in k[batch]] for qb in q[batch]] for batch in range(2)])
        np.testing.assert_allclose(reference_logits_from_vectors(q, k, 1 / 3), expected, rtol=0, atol=1e-15)

    def test_reference_rejects_bad_shapes_and_nonfinite_values(self):
        with self.assertRaisesRegex(ValueError, "matched nonempty"):
            reference_logits_from_vectors(np.zeros((1, 2, 4)), np.zeros((1, 3, 5)), 1)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            reference_logits_from_vectors(np.full((1, 2, 4), np.nan), np.zeros((1, 3, 4)), 1)

    def test_full_group_validation_includes_unsampled_frame(self):
        table = np.asarray([sorted({r, *[x for x in range(45) if x != r][:31]})
                            for r in range(45)], dtype=np.int64)
        rows = np.empty((1, 1, 21, 45, 80, 4, 4), dtype=np.uint8)
        for row in range(45):
            choices = [row] + [x for x in table[row] if x != row][:3]
            rows[:, :, :, row, :, :, :] = choices
        good = audit_all_rows(rows, table)
        self.assertTrue(good["valid"])
        self.assertEqual(good["groups"], 21 * 45 * 80 * 4)
        rows[0, 0, 7, 12, 31, 2, 0] = 250
        bad = audit_all_rows(rows, table)
        self.assertFalse(bad["valid"])
        self.assertEqual(bad["out_of_range_rows"], 1)
        self.assertEqual(bad["anchor_failures"], 1)

    def intermediate_fixture(self):
        rng = np.random.default_rng(319)
        logits2 = rng.normal(size=(32, 16)) * 3 + np.arange(32)[:, None] * 40
        norm2 = rng.normal(size=32) + logits2.max(-1)
        local_z2 = np.logaddexp.reduce(logits2 * math.log(2), axis=-1) / math.log(2)
        logp2 = logits2 - local_z2[:, None]
        logmass = local_z2 - norm2
        weight = np.exp2(logmass - logmass.max())
        selected = select_jeffreys_seeds(logits2 * math.log(2), 7, weight, singleton_index=4)
        planes = np.zeros((19, 32, 16), dtype=np.float64)
        planes[0] = logits2 / (128 ** -0.5 / math.log(2))
        planes[1], planes[2] = logits2, logits2 - logits2.max(-1)[:, None]
        planes[3], planes[4] = logp2, np.exp2(logp2)
        planes[5], planes[6] = weight[:, None], local_z2[:, None]
        for step in range(3):
            owner = logp2[selected[:step + 1]].argmax(axis=0)
            owner_logp = logp2[selected[:step + 1]].max(axis=0)
            gamma = partition_metrics(logits2 * math.log(2), logits2[selected[:step + 1]] * math.log(2), 4)["jeffreys"] / math.log(2)
            planes[7 + step] = owner[None, :]
            planes[10 + step] = owner_logp[None, :]
            planes[13 + step] = gamma[:, None]
            planes[16 + step] = (weight * np.maximum(gamma, 0))[:, None]
        return planes, np.arange(32), selected, norm2, 20, logits2

    def test_recorded_native_profiles_replay_independently(self):
        report, arrays = audit_recorded_intermediates(*self.intermediate_fixture())
        self.assertTrue(report["actual_native_logits_replay"]["valid"])
        self.assertTrue(report["recorded_profile_fixed_owner_score_comparison"]["valid"])
        self.assertTrue(report["recorded_profile_fixed_owner_gamma_comparison"]["valid"])
        self.assertEqual(report["native_owner_vs_recorded_logp_mismatches_by_prefix"], [0, 0, 0])
        self.assertEqual(arrays["native_intermediate_planes"].shape, (19, 32, 16))

    def test_recorded_gamma_corruption_is_not_hidden_by_priority(self):
        values = list(self.intermediate_fixture())
        values[0][14, 3] += 1
        report, _ = audit_recorded_intermediates(*values)
        self.assertTrue(report["actual_native_logits_replay"]["valid"])
        self.assertTrue(report["recorded_profile_fixed_owner_score_comparison"]["valid"])
        self.assertFalse(report["recorded_profile_fixed_owner_gamma_comparison"]["valid"])

    def test_recorded_owner_corruption_has_separate_mismatch(self):
        values = list(self.intermediate_fixture())
        values[0][8, :, 0] = 1 - values[0][8, :, 0]
        report, _ = audit_recorded_intermediates(*values)
        self.assertEqual(report["native_owner_vs_recorded_logp_mismatches_by_prefix"], [0, 1, 0])

    def test_recorded_repeated_plane_shape_is_checked(self):
        values = list(self.intermediate_fixture())
        values[0][5, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "repeat across keys"):
            audit_recorded_intermediates(*values)


if __name__ == "__main__":
    unittest.main()
