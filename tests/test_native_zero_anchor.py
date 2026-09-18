"""Actual zero-anchor native AST against NumPy; no Torch/Triton or GPU."""
import ast
import unittest
from unittest import mock

import numpy as np

from tests import test_affine_hellinger_seed as native_contract
from tests.test_affine_hellinger_seed import (
    KERNEL, SOURCE, MetadataTensor, actual_greedy, function,
)


def numpy_zero_gs(logits, dummy):
    """Independent scalar Uniform16 Hessian embedding, three free pivots."""
    values = np.asarray(logits, dtype=np.float64)
    choices, traces = [], []
    for bank in values:
        residual = np.array([(row - sum(row) / 16.0) / 4.0 for row in bank])
        picked, scores = [], []
        for _ in range(3):
            score = np.array([np.dot(row, row) for row in residual])
            scores.append(score.copy())
            score[picked] = -np.inf
            best = int(np.argmax(score))
            picked.append(best)
            basis = residual[best] / np.sqrt(np.dot(residual[best], residual[best]) + 1e-12)
            residual -= np.outer(residual @ basis, basis)
        choices.append([dummy, *picked])
        traces.append(scores)
    return np.asarray(choices), np.asarray(traces)


class NativeZeroMathTests(unittest.TestCase):
    def test_actual_feature_and_three_prefix_scores_match_numpy(self):
        rng = np.random.default_rng(93)
        for width in (1, 2, 4, 8):
            for dummy in (0, 13, 31):
                z = rng.normal(size=(width, 32, 16)) * 2
                chosen, score = actual_greedy(z, dummy, feature_mode="fisher_zero")
                expected, expected_score = numpy_zero_gs(z, dummy)
                np.testing.assert_array_equal(chosen, expected)
                np.testing.assert_allclose(score, expected_score, rtol=2e-13, atol=2e-13)
                self.assertTrue((np.diff(np.sort(chosen[:, 1:], axis=-1), axis=-1) > 0).all())

    def test_dummy_is_not_excluded_and_not_part_of_four_real_distinctness(self):
        z = np.zeros((2, 32, 16))
        chosen, score = actual_greedy(z, 0, feature_mode="fisher_zero")
        np.testing.assert_array_equal(chosen, [[0, 0, 1, 2]] * 2)
        np.testing.assert_array_equal(score, np.zeros((2, 3, 32)))
        z[:, 19, 0] = 10
        chosen, score = actual_greedy(z, 19, feature_mode="fisher_zero")
        np.testing.assert_array_equal(chosen[:, :2], [[19, 19]] * 2)
        self.assertTrue((score[:, 0, 19] > 0).all())
        # Ordinary modes retain the existing real anchor exclusion.
        for mode in ("hellinger", "fisher_geometric", "fisher_anchor", "fisher_antipodal"):
            original, _ = actual_greedy(np.zeros_like(z), 0, feature_mode=mode)
            np.testing.assert_array_equal(original, [[0, 1, 2, 3]] * 2)

    def test_dummy_position_cannot_change_the_three_pivots(self):
        z = np.random.default_rng(51).normal(size=(4, 32, 16))
        base, base_score = actual_greedy(z, 0, feature_mode="fisher_zero")
        moved, moved_score = actual_greedy(z, 22, feature_mode="fisher_zero")
        np.testing.assert_array_equal(base[:, 1:], moved[:, 1:])
        np.testing.assert_array_equal(base_score, moved_score)
        shifted, shifted_score = actual_greedy(z + np.arange(32)[None, :, None] * 16,
                                              0, feature_mode="fisher_zero")
        np.testing.assert_array_equal(base, shifted)
        np.testing.assert_allclose(base_score, shifted_score, rtol=2e-12, atol=2e-12)

    def test_fp64_feature_conversion_precedes_centering(self):
        zero_branch = next(node for node in ast.walk(function(KERNEL))
                           if isinstance(node, ast.If)
                           and ast.unparse(node.test) == "FEATURE_MODE == 'fisher_zero'"
                           and any(isinstance(item, ast.Assign) and isinstance(item.targets[0], ast.Name)
                                   and item.targets[0].id == "z" for item in node.body))
        source = ast.unparse(ast.Module(body=zero_branch.body, type_ignores=[]))
        self.assertIn("dot_logits.to(tl.float64) * SCALE", source)
        self.assertIn("tl.sum(z, axis=2)", source)
        self.assertIn("residual = centered * 0.25", source)
        self.assertNotIn("anchor_z", source)


class NativeZeroMetadataTests(unittest.TestCase):
    setUp = native_contract.NativeMetadataTests.setUp

    def test_zero_mode_keeps_c32_shape_and_supports_all_three_prefix_audits(self):
        audit = MetadataTensor((2, 1, 5, 45, 80, 3, 3, 32), "fp32")
        output = self.selector(self.q, self.k, feature_mode="fisher_zero", score_audit=audit)
        self.assertEqual(output.shape, (2, 1, 5, 45, 80, 3, 4))
        self.assertEqual(output.dtype, "u8")
        self.candidates.assert_called_once_with(45, 32, self.q.device, geometry_count=45)
        launch = self.launches[0]
        self.assertEqual(launch.kwargs["FEATURE_MODE"], "fisher_zero")
        self.assertEqual(launch.kwargs["CANDIDATES"], 32)
        self.assertEqual(launch.kwargs["BLOCK_ROWS"], 32)
        self.assertTrue(launch.kwargs["AUDIT_SCORES"])
        self.assertIs(launch.args[5], audit)
        self.assertIs(launch.args[6], self.q)  # unused normalizer, no extra computation
        self.assertIn("a real row may equal the dummy", self.selector.__doc__)

    def test_zero_rejects_full45_or_non_bf16_but_normal_modes_stay_valid(self):
        with self.assertRaises(ValueError):
            self.selector(self.q, self.k, feature_mode="fisher_zero", candidate_count=45)
        q = MetadataTensor(dtype="fp16")
        k = MetadataTensor((2, 5, 45, 80, 3, 128), dtype="fp16")
        with self.assertRaises(TypeError):
            self.selector(q, k, feature_mode="fisher_zero")
        self.selector(q, k, feature_mode="fisher_anchor", candidate_count=45)
        self.assertEqual(self.launches[0].kwargs["BLOCK_ROWS"], 64)

    def test_wrapper_labels_dummy_semantics_and_requires_zero_merge_override(self):
        target = mock.Mock(return_value="sentinel")
        scope = {"affine_hellinger_seed_rows_triton": target}
        tree = ast.fix_missing_locations(ast.Module(
            body=[function("fisher_zero_seed_rows_triton")], type_ignores=[]))
        exec(compile(tree, str(SOURCE), "exec"), scope)
        wrapper = scope["fisher_zero_seed_rows_triton"]
        self.assertEqual(wrapper(self.q, self.k, .5, global_probe_count=256,
                                 score_audit="audit"), "sentinel")
        target.assert_called_once_with(self.q, self.k, .5,
                                       feature_mode="fisher_zero", score_audit="audit")
        self.assertIn("zero-query merge override", wrapper.__doc__)
        self.assertIn("only slots 1:4 are distinct", wrapper.__doc__)
        with self.assertRaises(ValueError):
            wrapper(self.q, self.k, global_probe_count=128)


if __name__ == "__main__":
    unittest.main()
