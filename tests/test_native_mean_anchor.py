"""Native mean-query AST against independent NumPy and optional Torch CPU."""
import ast
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from tests import test_affine_hellinger_seed as contract
from tests.test_affine_hellinger_seed import KERNEL, SOURCE, MetadataTensor, NumpyCasts, function, target_name


def actual_mean_greedy(z, mean_z, anchor, mode):
    """Execute actual native feature/pivot AST with supplied FP32 dot logits."""
    z = np.asarray(z)
    width, count, probes = z.shape
    assert count == 32 and probes == 16 and mode in ("fisher_mean", "fisher_mean_anchor")
    loop = next(node for node in function(KERNEL).body
                if isinstance(node, ast.For) and node.target.id == "key_outer")
    start = next(i for i, node in enumerate(loop.body) if target_name(node) == "maximum")
    end = next(i for i, node in enumerate(loop.body) if target_name(node) == "output_offset")
    body = loop.body[start:end]
    pivots = next(node for node in body if isinstance(node, ast.For))
    at = next(i for i, node in enumerate(pivots.body) if target_name(node) == "score")
    pivots.body[at + 1:at + 1] = ast.parse("traces.append(score.copy())").body
    def maximum(values, axis, return_indices=False, return_indices_tie_break_left=False):
        if return_indices:
            assert return_indices_tie_break_left
            return np.max(values, axis=axis), np.argmax(values, axis=axis)
        return np.max(values, axis=axis)
    tl = SimpleNamespace(exp2=np.exp2, sqrt=np.sqrt, rsqrt=lambda x: 1 / np.sqrt(x),
        max=maximum, sum=np.sum, where=np.where, reshape=np.reshape, arange=np.arange,
        gather=lambda x, index, axis: np.take_along_axis(x, index, axis=axis),
        static_range=range, int32=np.int32, float64=np.float64, broadcast_to=np.broadcast_to,
        cast=lambda x, dtype: np.asarray(x).astype(dtype))
    scope = {"FEATURE_MODE": mode, "tl": tl, "libdevice": SimpleNamespace(exp=np.exp, log=np.log),
        "dot_logits": z.reshape(-1, 16), "logits": (z / np.log(2)).reshape(-1, 16),
        "mean_dot_logits": np.pad(mean_z, ((0, 16 - width), (0, 0))), "SCALE": 1.,
        "CANDIDATES": 32, "BLOCK_J": width, "BLOCK_ROWS": 32,
        "valid_2d": np.ones((width, 32), dtype=bool), "candidate_ids": np.arange(32)[None],
        "anchor_index": anchor, "candidate_rows_2d": np.broadcast_to(np.arange(32), (width, 32)),
        "seed_slots": np.arange(4)[None], "AUDIT_SCORES": False, "traces": []}
    tree = ast.fix_missing_locations(NumpyCasts().visit(ast.Module(body=body, type_ignores=[])))
    exec(compile(tree, str(SOURCE), "exec"), scope)
    return scope["seeds"], np.stack(scope["traces"], axis=1)


def numpy_oracle(z, mean_z, anchor, force_anchor):
    selected, all_scores = [], []
    for bank, center in zip(z.astype(np.float64), mean_z.astype(np.float64)):
        rho = np.exp(center - np.logaddexp.reduce(center))
        delta = bank - center
        residual = np.array([np.sqrt(rho) * (row - rho @ row) for row in delta])
        picks, scores = [], []
        for step in range(3):
            norm = np.array([row @ row for row in residual])
            scores.append(norm.copy())
            norm[picks] = -np.inf
            pick = anchor if force_anchor and step == 0 else int(norm.argmax())
            picks.append(pick)
            basis = residual[pick] / np.sqrt(residual[pick] @ residual[pick] + 1e-12)
            residual -= np.outer(residual @ basis, basis)
        selected.append([anchor, *picks])
        all_scores.append(scores)
    return np.asarray(selected), np.asarray(all_scores)


class NativeMeanMathTests(unittest.TestCase):
    def test_actual_native_features_and_all_prefix_scores_match_numpy(self):
        rng = np.random.default_rng(724)
        for width in (1, 2, 4, 8):
            z = rng.normal(size=(width, 32, 16)).astype(np.float32) * 3
            mean_z = rng.normal(size=(width, 16)).astype(np.float32) * 2
            for mode in ("fisher_mean", "fisher_mean_anchor"):
                actual, score = actual_mean_greedy(z, mean_z, 17, mode)
                expected, expected_score = numpy_oracle(z, mean_z, 17, mode.endswith("anchor"))
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_allclose(score, expected_score, rtol=5e-13, atol=5e-13)
                self.assertTrue((np.diff(np.sort(actual[:, 1:]), axis=-1) > 0).all())

    def test_zero_ties_forced_anchor_and_unforced_dummy_semantics(self):
        z, m = np.zeros((4, 32, 16)), np.zeros((4, 16))
        actual, _ = actual_mean_greedy(z, m, 7, "fisher_mean")
        np.testing.assert_array_equal(actual, [[7, 0, 1, 2]] * 4)
        forced, _ = actual_mean_greedy(z, m, 7, "fisher_mean_anchor")
        np.testing.assert_array_equal(forced, [[7, 7, 0, 1]] * 4)
        z[:, 7, 0] = 10
        actual, _ = actual_mean_greedy(z, m, 7, "fisher_mean")
        np.testing.assert_array_equal(actual[:, :2], [[7, 7]] * 4)

    def test_actual_mean_pointer_load_is_per_batch_column_head_with_padding(self):
        native = function(KERNEL)
        branch = next(node for node in native.body if isinstance(node, ast.If)
                      and any(target_name(item) == "mean_query" for item in node.body))
        # All values are exact integers: this checks offsets and masks, not QK rounding.
        means = np.arange(2 * 80 * 3 * 5).reshape(2, 80, 3, 5)
        def load(offset, mask, other):
            result = np.full(offset.shape, other, dtype=np.float32)
            result[mask] = means.reshape(-1)[offset[mask]]
            return result
        scope = {"mass_ptr": 0, "batch": 1, "head": 2, "group": 7, "COLS": 80,
            "HEADS": 3, "D_HEAD": 5, "BLOCK_J": 4, "dims": np.arange(32),
            "tl": SimpleNamespace(arange=np.arange, load=load)}
        exec(compile(ast.Module(body=branch.body, type_ignores=[]), str(SOURCE), "exec"), scope)
        expected = np.zeros((16, 32), dtype=np.float32)
        expected[:4, :5] = means[1, 28:32, 2]
        np.testing.assert_array_equal(scope["mean_query"], expected)

    def test_mean_helper_has_fp32_reduction_bf16_output_without_explicit_q_float(self):
        node = function("mean_query_bf16")
        result = next(item.value for item in node.body if isinstance(item, ast.Return))
        self.assertEqual(ast.unparse(result), "q.mean(dim=2, dtype=torch.float32).to(q.dtype).contiguous()")
        text = ast.unparse(function(KERNEL))
        self.assertIn("mean_dot_logits = tl.dot(mean_query, tl.trans(keys), input_precision='ieee')", text)
        self.assertIn("tl.gather(mean_dot_logits, mean_index, axis=0).to(tl.float64) * SCALE", text)


class NativeMeanMetadataTests(unittest.TestCase):
    setUp = contract.NativeMetadataTests.setUp

    def test_only_rf0_mean_is_computed_and_forwarded_in_the_unused_mass_slot(self):
        means = MetadataTensor((2, 1, 80, 3, 128))
        helper = mock.Mock(return_value=means)
        self.selector.__globals__["mean_query_bf16"] = helper
        audit = MetadataTensor((2, 1, 5, 45, 80, 3, 3, 32), "fp32")
        result = self.selector(self.q, self.k, feature_mode="fisher_mean", score_audit=audit)
        helper.assert_called_once_with(self.q)
        self.assertEqual(result.shape, (2, 1, 5, 45, 80, 3, 4))
        self.assertIs(self.launches[0].args[6], means)
        self.assertIs(self.launches[0].args[5], audit)
        self.assertEqual(self.launches[0].kwargs["CANDIDATES"], 32)
        self.assertEqual(self.launches[0].kwargs["BLOCK_ROWS"], 32)

    def test_both_wrappers_match_requested_dispatch_names_and_mode_flags(self):
        for name, mode in (("fisher_mean_seed_rows_triton", "fisher_mean"),
                           ("fisher_mean_anchor_seed_rows_triton", "fisher_mean_anchor")):
            called = mock.Mock(return_value="native-output")
            scope = {"affine_hellinger_seed_rows_triton": called}
            tree = ast.fix_missing_locations(ast.Module(body=[function(name)], type_ignores=[]))
            exec(compile(tree, str(SOURCE), "exec"), scope)
            self.assertEqual(scope[name](self.q, self.k, .5, candidate_count=32), "native-output")
            called.assert_called_once_with(self.q, self.k, .5, feature_mode=mode, candidate_count=32)
            with self.assertRaises(ValueError): scope[name](self.q, self.k, global_probe_count=128)
        for mode in ("fisher_mean", "fisher_mean_anchor"):
            with self.assertRaises(ValueError): self.selector(self.q, self.k, feature_mode=mode, candidate_count=45)


@unittest.skipIf(torch is None, "optional actual PyTorch CPU mean/prototype parity")
class MeanTorchParityTests(unittest.TestCase):
    def test_real_cpu_mean_matches_old_cast_reduce_cast_and_rf0_scope(self):
        node = function("mean_query_bf16")
        scope = {"torch": torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(SOURCE), "exec"), scope)
        q = torch.randn(1, 2, 45, 80, 2, 7, generator=torch.Generator().manual_seed(518)).to(torch.bfloat16)
        actual = scope["mean_query_bf16"](q)
        expected = q.float().mean(2).to(q.dtype).contiguous()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        q[:, 1] = float("nan")
        only_rf0 = scope["mean_query_bf16"](q[:, :1])
        torch.testing.assert_close(only_rf0, expected[:, :1], rtol=0, atol=0)
        self.assertEqual(actual.dtype, torch.bfloat16)

    def test_actual_native_ast_equals_existing_mean_seed_indices(self):
        from analysis.mean_seed_pytorch import mean_seed_indices
        rng = np.random.default_rng(215)
        z = rng.normal(size=(4, 32, 16)).astype(np.float32)
        m = rng.normal(size=(4, 16)).astype(np.float32)
        for mode, prototype in (("fisher_mean", "mean_fisher32"), ("fisher_mean_anchor", "mean_anchor_fisher32")):
            actual, _ = actual_mean_greedy(z, m, 13, mode)
            expected = mean_seed_indices(torch.from_numpy(z), torch.from_numpy(m), [13] * 4, prototype)
            np.testing.assert_array_equal(actual[:, 1:], expected.numpy())


if __name__ == "__main__":
    unittest.main()
