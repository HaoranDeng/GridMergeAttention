"""CPU algebra and fail-closed contracts; not native GPU parity evidence."""
import ast
import copy
from pathlib import Path
import unittest

import numpy as np

from analysis import check_barycentric_native as parent
from analysis.check_phase_averaged_case import (
    DOTS_PER_BANK, DOT_DIAGNOSTIC_LIMIT, DOT_GAMMA, DOT_TERMS, FP32_UNIT_ROUNDOFF,
    dot_forward_error, explicit_phase_average, validate_phase_record,
)

ROOT = Path(__file__).resolve().parents[1]


def valid_record(frames):
    banks = frames * 7200
    record = dict(anchor_mismatches=0, nonunique_banks=0, invalid_candidate_banks=0,
        score_audit_mismatched_banks=0, transform_bit_mismatches=0, spatial_bit_mismatches=0,
        signed_zero_bit_differences=0, transform_nonzero_bit_mismatches=0,
        transform_equal_except_zero_sign=True, transform_zero_sign_policy="nonzero_bits_exact_zero_sign_ignored",
        q_input_bit_mismatches=0, k_input_bit_mismatches=0, fp32_dot_mismatches=0,
        nonfinite_score_values=0, score_audit_on_off_equal=True, transform_bitwise_equal=True,
        spatial_bitwise_equal=True, q_input_unchanged=True, k_input_unchanged=True,
        fp32_reference_dots_equal_fp64=True, transformed_q_contiguous=True,
        fp32_dot_bound_violations=0, fp32_dot_zero_bound_violations=0, nonfinite_dot_values=0,
        fp32_dot_bound_comparison_passed=True, max_fp32_dot_bound_ratio=0.0,
        fp32_unit_roundoff=FP32_UNIT_ROUNDOFF, fp32_dot_terms=DOT_TERMS, fp32_dot_gamma=DOT_GAMMA,
        temporal_frames=frames, temporal_width=44, spatial_width=84,
        frame_banks=[7200] * frames, frame_mismatched_banks=[0] * frames,
        q_dtype="torch.bfloat16", k_dtype="torch.bfloat16", transformed_q_dtype="torch.bfloat16",
        q_shape=[1, 1, 45, 80, 2, 128], transformed_q_shape=[1, 1, 45, 80, 2, 128],
        k_shape=[1, frames, 45, 80, 2, 128], fp32_dot_values_checked=banks * 512,
        max_fp32_dot_abs_error=0.0, score_audit_dtype="torch.float32",
        score_audit_values_observed=banks * 96, score_values_compared=0,
        score_comparison_performed=False, selector_block_j=4,
        oracle_implementation="analysis.anchor_fullgrid_seed.anchor_indices",
        fixture="independent_dense_integer_qk_-4_to_4")
    return record


def valid_result():
    hashes = {"source.py": "a" * 64}
    result = dict(complete=True, family="phase_averaged", shape=list(parent.Q_SHAPE),
                  source_sha256=hashes, source_sha256_after=hashes, records=[])
    for seed, frames, scale, method, banks in parent.expected_records("phase_averaged"):
        record = valid_record(frames)
        record.update(input_seed=seed, key_frames=frames, scale=scale, method=method, banks=banks,
                      group_agreement=1.0, mismatched_banks=0, native_sha256="b" * 64)
        result["records"].append(record)
    return result, hashes


class ExplicitRotationAlgebraTests(unittest.TestCase):
    def test_explicit_sum_matches_independent_complex_rotation_and_preserves_spatial(self):
        rng = np.random.default_rng(553)
        original = rng.integers(-4, 5, (2, 3, 128)).astype(np.float64)
        untouched = original.copy()
        frequency = np.power(10000.0, -np.arange(22) / 22.0)
        for frames in (1, 3, 21):
            actual = explicit_phase_average(original, frames)
            pairs = original[..., :44:2] + 1j * original[..., 1:44:2]
            rotations = np.exp(1j * np.arange(frames)[:, None] * frequency)
            expected = pairs * np.mean(rotations, axis=0)
            np.testing.assert_allclose(actual[..., :44:2], expected.real, rtol=2e-14, atol=2e-14)
            np.testing.assert_allclose(actual[..., 1:44:2], expected.imag, rtol=2e-14, atol=2e-14)
            np.testing.assert_array_equal(actual[..., 44:].view(np.uint64), original[..., 44:].view(np.uint64))
            self.assertEqual(actual.dtype, np.float64)
        np.testing.assert_array_equal(original, untouched)
        np.testing.assert_array_equal(explicit_phase_average(original, 1), original)

    def test_positive_rotation_interleaved_pairs_and_not_real_frame_mean(self):
        query = np.zeros((1, 128))
        query[0, 0] = 1
        query[0, 44:] = np.arange(84)
        actual = explicit_phase_average(query, 2)
        self.assertAlmostEqual(actual[0, 0], (1 + np.cos(1)) / 2)
        self.assertAlmostEqual(actual[0, 1], np.sin(1) / 2)
        self.assertGreater(actual[0, 1], 0)
        np.testing.assert_array_equal(actual[0, 2:44], 0)
        np.testing.assert_array_equal(actual[0, 44:], query[0, 44:])

    def test_reference_rejects_invalid_input(self):
        for frames in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                explicit_phase_average(np.ones(128), frames)
        for query in (np.ones(127), np.full(128, np.nan), np.full(128, np.inf), np.array(1)):
            with self.assertRaises(ValueError):
                explicit_phase_average(query, 21)

    def test_integer_keys_do_not_imply_exact_float32_accumulation(self):
        # All entries below are exactly BF16-representable; keys are dense
        # integers. Each product is exact but summing in FP32 loses information.
        query = np.full(128, 2.0 ** -24, dtype=np.float32)
        query[0] = 4.0
        keys = np.full(128, 4, dtype=np.float32)
        result32 = np.sum(query * keys, dtype=np.float32)
        result64 = np.sum(query.astype(np.float64) * keys, dtype=np.float64)
        self.assertNotEqual(float(result32), float(result64))
        _, bound, ratio, finite, violations, zero_bad = dot_forward_error(
            np.asarray(result32, dtype=np.float64), np.asarray(result64),
            np.sum(np.abs(query.astype(np.float64) * keys)), array=np)
        self.assertTrue(bool(finite))
        self.assertFalse(bool(violations))
        self.assertFalse(bool(zero_bad))
        self.assertGreater(float(ratio), 0)
        self.assertLessEqual(float(ratio), 1)
        self.assertGreater(float(bound), 0)

    def test_preset_forward_bound_rejects_excess_nonfinite_and_nonexact_zero_bound(self):
        self.assertEqual(FP32_UNIT_ROUNDOFF, 2 ** -24)
        self.assertEqual(DOT_TERMS, 128)
        self.assertEqual(DOT_GAMMA, 127 * 2**-24 / (1 - 127 * 2**-24))
        truth = np.array([0., 0., 0., 0.])
        abs_sum = np.array([1., 1., 0., 0.])
        observed = np.array([DOT_GAMMA * .5, DOT_GAMMA * 1.01, 0., 2**-24])
        _, _, ratios, finite, violations, zero_bad = dot_forward_error(observed, truth, abs_sum, array=np)
        np.testing.assert_array_equal(violations, [False, True, False, True])
        np.testing.assert_array_equal(zero_bad, [False, False, False, True])
        self.assertTrue(finite.all())
        self.assertEqual(ratios[2], 0)
        self.assertTrue(np.isinf(ratios[3]))
        for value in (np.nan, np.inf):
            _, _, _, finite, violations, _ = dot_forward_error(
                np.asarray(value), np.asarray(0.), np.asarray(1.), array=np)
            self.assertFalse(bool(finite))
            self.assertTrue(bool(violations))


class PhaseFamilyContractTests(unittest.TestCase):
    def test_exact_cases_banks_scales_and_old_families_unchanged(self):
        self.assertEqual(parent.family_cases("phase_averaged"), ((17, 1, .125), (53, 21, 128 ** -.5)))
        specs = parent.expected_records("phase_averaged")
        self.assertEqual([spec[3] for spec in specs], ["phase_averaged"] * 2)
        self.assertEqual([spec[-1] for spec in specs], [7200, 151200])
        self.assertEqual(sum(spec[-1] for spec in specs), 158400)
        self.assertEqual(DOTS_PER_BANK, 512)
        self.assertEqual(DOT_DIAGNOSTIC_LIMIT, 8)
        self.assertEqual(len(parent.expected_records("barycentric")), 6)
        for family in ("expected_fisher", "anchor_gap_guard"):
            self.assertEqual(parent.family_cases(family), ((17, 1, .125), (53, 2, 128 ** -.5)))

    def test_complete_21_frame_bank_decode_including_last_frame(self):
        ids = np.arange(151200)
        actual = parent.decode_bank_ids(ids)
        for observed, expected in zip(actual, np.unravel_index(ids, (21, 45, 80, 2))):
            np.testing.assert_array_equal(observed, expected)
        self.assertEqual(parent.decode_bank_ids(144000), (20, 0, 0, 0))
        self.assertEqual(parent.decode_bank_ids(151199), (20, 44, 79, 1))

    def test_every_phase_gate_is_fail_closed_including_missing_fields(self):
        good = valid_record(21)
        validate_phase_record(good, 151200, 21)
        # Every declared field is necessary; no missing bool defaults to pass.
        for field in good:
            bad = good.copy()
            del bad[field]
            with self.subTest(missing=field), self.assertRaises(RuntimeError):
                validate_phase_record(bad, 151200, 21)
        invalid = dict(transform_bit_mismatches=1, fp32_dot_mismatches=1,
            max_fp32_dot_abs_error=1e-20, score_audit_on_off_equal=False,
            score_comparison_performed=True, score_values_compared=96,
            q_input_unchanged=False, k_input_unchanged=False,
            temporal_frames=1, selector_block_j=1, score_audit_dtype="torch.float64",
            fixture="one_hot_keys", fp32_dot_values_checked=7200 * 512)
        invalid.update(fp32_dot_bound_violations=1, fp32_dot_zero_bound_violations=1,
            nonfinite_dot_values=1, fp32_dot_bound_comparison_passed=False,
            max_fp32_dot_bound_ratio=1.01, fp32_unit_roundoff=2**-23,
            fp32_dot_terms=127, fp32_dot_gamma=1e-4)
        for field, value in invalid.items():
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                validate_phase_record(good | {field: value}, 151200, 21)
        for value in (float("nan"), float("inf"), -1):
            with self.assertRaises(RuntimeError):
                validate_phase_record(good | {"max_fp32_dot_abs_error": value}, 151200, 21)
            with self.assertRaises(RuntimeError):
                validate_phase_record(good | {"max_fp32_dot_bound_ratio": value}, 151200, 21)

    def test_legal_roundoff_record_passes_but_native_rows_still_must_be_exact(self):
        result, hashes = valid_result()
        result["records"][1].update(fp32_dot_mismatches=12, fp32_reference_dots_equal_fp64=False,
            max_fp32_dot_abs_error=2**-18, max_fp32_dot_bound_ratio=.25)
        parent.validate_completed_result(result, "phase_averaged", hashes)
        result["records"][1]["mismatched_banks"] = 1
        with self.assertRaises(RuntimeError):
            parent.validate_completed_result(result, "phase_averaged", hashes)

    def test_parent_rejects_each_case_identity_and_phase_failure(self):
        result, hashes = valid_result()
        parent.validate_completed_result(result, "phase_averaged", hashes)
        for field, value in (("input_seed", 17), ("key_frames", 2), ("scale", .125),
                             ("method", "fisher_expected"), ("banks", 14400),
                             ("group_agreement", .999), ("mismatched_banks", 1),
                             ("native_sha256", ""), ("fp32_dot_mismatches", 1),
                             ("transform_bitwise_equal", False), ("q_input_unchanged", False)):
            bad = copy.deepcopy(result)
            bad["records"][1][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                parent.validate_completed_result(bad, "phase_averaged", hashes)
        for field, value in (("complete", False), ("source_sha256_after", {}),
                             ("records", result["records"][:1])):
            with self.assertRaises(RuntimeError):
                parent.validate_completed_result(result | {field: value}, "phase_averaged", hashes)

    def test_zero_sign_exception_is_explicit_and_does_not_allow_any_nonzero_difference(self):
        good = valid_record(21)
        sign_only = good | dict(transform_bit_mismatches=12, signed_zero_bit_differences=12,
                               transform_bitwise_equal=False)
        validate_phase_record(sign_only, 151200, 21)
        for changes in (dict(transform_nonzero_bit_mismatches=1),
                        dict(transform_bit_mismatches=13), dict(signed_zero_bit_differences=11),
                        dict(transform_bitwise_equal=True), dict(transform_equal_except_zero_sign=False),
                        dict(signed_zero_bit_differences=-1), dict(transform_zero_sign_policy="allclose"),
                        dict(q_input_bit_mismatches=1), dict(k_input_bit_mismatches=1),
                        dict(spatial_bit_mismatches=1)):
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                validate_phase_record(sign_only | changes, 151200, 21)

    def test_worker_uses_independent_transform_anchor_reference_and_original_native_inputs(self):
        tree = ast.parse((ROOT / "analysis/check_phase_averaged_case.py").read_text())
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "explicit_phase_average")
        helper_source = ast.unparse(helper)
        self.assertIn("for frame in range(temporal_frames)", helper_source)
        self.assertNotIn("phase_averaged_seed", helper_source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "check_phase_case")
        source = ast.unparse(function)
        for fragment in ("phase_averaged_query(q, frames)",
                         "explicit_phase_average(q_before.double().cpu().numpy(), frames)",
                         "phase_averaged_fisher_seed_rows(q, k, scale, **kwargs)",
                         "phase_averaged_fisher_seed_rows(q, k, scale, score_audit=scores, **kwargs)",
                         "dtype=torch.float32", "block_j=4", "frame, row, col, head = decode_bank_ids(ids)",
                         "reference_q[0, 0, real, col[:, None], head[:, None]].float()",
                         "k_before[0, frame[:, None], row[:, None]",
                         "torch.bmm(qq.double(), kk.double().transpose(1, 2))",
                         "torch.bmm(qq.double().abs(), kk.double().abs().transpose(1, 2))",
                         "dot_forward_error(dot32.double(), dot64, absolute_sum, array=torch)",
                         "z = dot64 * scale", "anchor_indices(z, anchors[row])",
                         "transformed.view(torch.int16)", "q.view(torch.int16)", "k.view(torch.int16)",
                         "transform_difference & (transformed == 0) & (reference_q == 0)",
                         "transform_difference & ~signed_zero_difference"):
            self.assertIn(fragment, source)
        self.assertEqual(source.count("torch.randint(-4, 5"), 2)
        for forbidden in ("torch.load", "repeat(", "repeat_interleave(", "expand(", "float8", "one_hot"):
            self.assertNotIn(forbidden, source)
        reference = ast.parse((ROOT / "analysis/anchor_fullgrid_seed.py").read_text())
        for name in ("anchor_features", "anchor_indices"):
            body = ast.unparse(next(node for node in reference.body if isinstance(node, ast.FunctionDef)
                                    and node.name == name))
            self.assertNotIn("triton", body)
            self.assertNotIn("native", body)
        self.assertIn("selected = [anchor]", body)

    def test_source_closure_and_phase_worker_does_not_claim_score_value_parity(self):
        for source in ("analysis/check_phase_averaged_case.py", "analysis/anchor_fullgrid_seed.py",
                       "attention/phase_averaged_seed.py"):
            self.assertIn(source, parent.SOURCES)
        tree = ast.parse((ROOT / "analysis/check_barycentric_native.py").read_text())
        worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "worker")
        branch = next(node for node in ast.walk(worker) if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "args.family == 'phase_averaged'"
                      and "validate_phase_record" in ast.unparse(node))
        text = ast.unparse(branch)
        self.assertIn("validate_phase_record(record, record['banks'], frames)", text)
        self.assertIn("no score-value comparison", text)
        self.assertIn("continue", text)
        self.assertNotIn("SCORE_ATOL", text)


if __name__ == "__main__":
    unittest.main()
