"""CPU algebra/AST dispatch tests; these do NOT compile or execute CUDA/Triton.

Execute the actual JIT helper Python bodies against a small NumPy language
facade, and compare to independent explicit stacked-feature GS. Separately
execute the real wrapper body with metadata-only doubles. This checks formula
and interface contracts, not GPU reductions, compiler layouts or performance.
"""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "attention" / "expected_fisher_seed_triton.py"


class NumpyLanguage:
    float64 = np.float64
    int32 = np.int32
    arange = staticmethod(np.arange)
    full = staticmethod(np.full)
    where = staticmethod(np.where)
    maximum = staticmethod(np.maximum)
    sum = staticmethod(np.sum)
    range = staticmethod(range)

    @staticmethod
    def max(value, axis, return_indices=False, return_indices_tie_break_left=False):
        maximum = np.max(value, axis=axis)
        if return_indices:
            if not return_indices_tie_break_left:
                raise AssertionError("tie contract omitted")
            return maximum, np.argmax(value, axis=axis)
        return maximum


def helper_namespace():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    wanted = {"_expected_geometry", "_expected_covariance_column", "_safeguarded_factor", "_expected_pivots"}
    definitions = [node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name in wanted]
    for node in definitions:
        node.decorator_list = []
    namespace = {"tl": NumpyLanguage, "libdevice": SimpleNamespace(exp=np.exp)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])),
                 str(SOURCE), "exec"), namespace)
    return namespace


def stacked_feature_oracle(z, anchor):
    """Average per-candidate Hessians via explicit R*P features, not Y/Gram."""
    exponential = np.exp(z - z.max(axis=-1, keepdims=True))
    probabilities = exponential / exponential.sum(axis=-1, keepdims=True)
    delta = z - z[anchor]
    delta -= delta[:, :1]
    feature = np.stack([
        np.concatenate([np.sqrt(p / len(z)) * (d - p @ d) for p in probabilities])
        for d in delta
    ])
    residual = feature.copy()
    selected = [int(anchor)]
    scores = []
    for step in range(3):
        score = np.einsum("ij,ij->i", residual, residual)
        scores.append(score.copy())
        score[selected] = -np.inf
        pivot = int(score.argmax())
        selected.append(pivot)
        if step < 2:
            basis = residual[pivot].copy()
            basis /= np.sqrt(basis @ basis + 1e-12)
            residual -= np.outer(residual @ basis, basis)
    return np.array(selected), feature, np.array(scores)


class ExpectedFisherAlgebraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = helper_namespace()

    def test_actual_helper_bodies_match_independent_stacked_feature_gs(self):
        rng = np.random.default_rng(57012)
        for bank in range(20):
            z = rng.normal(size=(32, 16)) * 2.5
            anchor = bank * 7 % 32
            got, *scores = self.functions["_expected_pivots"](z, anchor)
            expected, feature, reference_scores = stacked_feature_oracle(z, anchor)
            np.testing.assert_array_equal(got, expected)
            np.testing.assert_allclose(np.array(scores), reference_scores, atol=2e-11, rtol=2e-11)
            delta, rho, y, diagonal = self.functions["_expected_geometry"](z, anchor)
            gram = feature @ feature.T
            self.assertEqual(y.shape, (32, 32))
            self.assertEqual(y.dtype, np.float64)
            np.testing.assert_allclose(diagonal, np.diag(gram), atol=2e-11, rtol=2e-11)
            for pivot in expected[1:3]:
                column = self.functions["_expected_covariance_column"](delta, rho, y, pivot)
                np.testing.assert_allclose(column, gram[:, pivot], atol=2e-11, rtol=2e-11)

    def test_safeguard_near_epsilon_and_exact_factor(self):
        factor = self.functions["_safeguarded_factor"]
        for norm in (0., 1e-15, 1e-12, 2e-12, 0.5, 12.):
            self.assertEqual(float(factor(norm)), (norm + 2e-12) / (norm + 1e-12) ** 2)
        self.assertNotEqual(float(factor(1e-12)), 1.0 / (1e-12 + 1e-12))
        rng = np.random.default_rng(943)
        for anchor in (0, 4, 15, 31):
            z = rng.normal(size=(32, 16)) * 1e-6
            got, *scores = self.functions["_expected_pivots"](z, anchor)
            expected, _, expected_scores = stacked_feature_oracle(z, anchor)
            np.testing.assert_array_equal(got, expected)
            np.testing.assert_allclose(scores, expected_scores, atol=1e-24, rtol=3e-11)

    def test_exact_gauge_ties_choose_first_unused_and_keep_real_anchor(self):
        profile = np.arange(16, dtype=np.float64) / 4
        z = np.tile(profile, (32, 1)) + np.arange(32)[:, None] * 8
        for anchor in (0, 17, 31):
            got, *scores = self.functions["_expected_pivots"](z, anchor)
            expected = [anchor] + [i for i in range(32) if i != anchor][:3]
            np.testing.assert_array_equal(got, expected)
            np.testing.assert_array_equal(scores, np.zeros((3, 32)))
        z = np.full((32, 16), -1000., dtype=np.float64)
        z[:, 0] = 1000.
        got, *scores = self.functions["_expected_pivots"](z, 7)
        np.testing.assert_array_equal(got, [7, 0, 1, 2])
        self.assertTrue(np.isfinite(scores).all())

    def test_row_offsets_and_probability_covariance_correction(self):
        rng = np.random.default_rng(761)
        z = rng.normal(size=(32, 16)) * 3
        shifted = z + rng.integers(-20, 20, size=(32, 1))
        original = self.functions["_expected_pivots"](z, 19)
        changed = self.functions["_expected_pivots"](shifted, 19)
        np.testing.assert_array_equal(original[0], changed[0])
        np.testing.assert_allclose(original[1:], changed[1:], atol=2e-11, rtol=2e-11)
        delta, rho, y, diagonal = self.functions["_expected_geometry"](z, 19)
        fisher_at_mean = (delta * delta * rho).sum(1) - (delta @ rho) ** 2
        correction = y.var(axis=1)
        np.testing.assert_allclose(fisher_at_mean - diagonal, correction, atol=2e-12, rtol=2e-12)
        self.assertGreater(float(correction.max()), 0.01)

    def test_second_column_is_updated_before_second_rank_reduction(self):
        z = np.random.default_rng(443).normal(size=(32, 16))
        got, *scores = self.functions["_expected_pivots"](z, 11)
        _, feature, _ = stacked_feature_oracle(z, 11)
        gram = feature @ feature.T
        first = int(got[1])
        f1 = (gram[first, first] + 2e-12) / (gram[first, first] + 1e-12) ** 2
        residual = gram - np.outer(gram[:, first], gram[:, first]) * f1
        second = int(got[2])
        f2 = (residual[second, second] + 2e-12) / (residual[second, second] + 1e-12) ** 2
        expected = np.maximum(np.diag(residual) - residual[:, second] ** 2 * f2, 0)
        wrong = np.maximum(np.diag(residual) - gram[:, second] ** 2 * f2, 0)
        np.testing.assert_allclose(scores[2], expected, atol=1e-12, rtol=1e-11)
        self.assertGreater(float(np.max(np.abs(wrong - expected))), 1e-4)


class MetadataTensor:
    def __init__(self, shape, dtype="bf16", device="cuda:0", contiguous=True):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype, self.device = dtype, device
        self.is_cuda = device.startswith("cuda:")
        self._contiguous = contiguous

    def contiguous(self):
        return self

    def is_contiguous(self):
        return self._contiguous


def wrapper_namespace():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    definition = next(n for n in tree.body
                      if isinstance(n, ast.FunctionDef) and n.name == "fisher_expected_seed_rows_triton")
    # Only replace the import statement with the injected recording table stub.
    definition.body = [ast.Pass() if isinstance(n, ast.ImportFrom) else n for n in definition.body]
    calls = []
    allocations = []

    def empty(shape, *, device, dtype):
        allocations.append((shape, device, dtype))
        return MetadataTensor(shape, dtype, device)

    def table(*args, **kwargs):
        calls.append(("table", args, kwargs))
        return "candidate_table", "anchor_table"

    class KernelRecorder:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append(("kernel", grid, args, kwargs))
            return launch

    namespace = {"math": math,
        "torch": SimpleNamespace(bfloat16="bf16", uint8="uint8", int32="int32", float64="fp64", empty=empty),
        "triton": SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
        "multiscale_candidate_table": table, "_expected_fisher_seed_kernel": KernelRecorder()}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[])), str(SOURCE), "exec"), namespace)
    return namespace["fisher_expected_seed_rows_triton"], calls, allocations


class ExpectedFisherNativeContractTests(unittest.TestCase):
    def setUp(self):
        self.wrapper, self.calls, self.allocations = wrapper_namespace()
        self.q = MetadataTensor((1, 1, 45, 80, 40, 128))
        self.k = MetadataTensor((1, 21, 45, 80, 40, 128))

    def test_actual_j1_grid_standard_layout_and_no_hbm_geometry_workspace(self):
        output = self.wrapper(self.q, self.k)
        self.assertEqual(output.shape, (1, 1, 21, 45, 80, 40, 4))
        self.assertEqual(output.dtype, "uint8")
        self.assertEqual(len(self.allocations), 1)
        self.assertEqual(self.calls[0], ("table", (45, 32, "cuda:0"), {"geometry_count": 45}))
        _, grid, args, kwargs = self.calls[1]
        self.assertEqual(grid, (3600, 40))
        self.assertIs(args[0], self.q)
        self.assertIs(args[1], self.k)
        self.assertEqual(kwargs, {"K_OUTER": 21, "HEADS": 40, "D_HEAD": 128, "BLOCK_D": 128,
                                 "SCALE": 128 ** -0.5, "AUDIT_SCORES": False, "num_warps": 4, "num_stages": 1})

    def test_reject_j4_or_conflicting_oracle_feature_and_precision_options(self):
        for kwargs in ({"block_j": 4}, {"block_j": True}, {"candidate_count": 45},
                       {"candidate_geometry_count": 32}, {"probe_count": 80},
                       {"global_probe_count": 128}, {"gs_num_warps": 8},
                       {"scale": float("nan")}, {"scale": float("inf")}):
            with self.assertRaises(ValueError):
                self.wrapper(self.q, self.k, **kwargs)
        for kwargs in ({"global_normalizer": self.q}, {"feature_mode": "fisher_mass"},
                       {"native_low_precision_dot": True}, {"num_stages": 2}):
            with self.assertRaises(TypeError):
                self.wrapper(self.q, self.k, **kwargs)
        self.assertEqual(self.allocations, [])

    def test_rf0_bf16_shape_and_output_guards(self):
        for tensor in (MetadataTensor((1, 21, 45, 80, 40, 128)),
                       MetadataTensor((1, 1, 44, 80, 40, 128)),
                       MetadataTensor((1, 1, 45, 80, 40, 129)),
                       MetadataTensor((1, 1, 45, 80, 40, 128), device="cpu")):
            with self.assertRaises(ValueError):
                self.wrapper(tensor, self.k)
        for dtype in ("fp16", "fp32", "fp64"):
            with self.assertRaises(TypeError):
                self.wrapper(MetadataTensor(self.q.shape, dtype=dtype), self.k)
        with self.assertRaises(ValueError):
            self.wrapper(self.q, MetadataTensor((2, 21, 45, 80, 40, 128)))
        with self.assertRaises(TypeError):
            self.wrapper(self.q, self.k, output_dtype="fp32")

    def test_optional_fp64_score_audit_is_explicit_and_not_allocated(self):
        shape = (1, 1, 21, 45, 80, 40, 3, 32)
        audit = MetadataTensor(shape, dtype="fp64")
        self.wrapper(self.q, self.k, score_audit=audit, output_dtype="int32")
        self.assertEqual(len(self.allocations), 1)
        self.assertIs(self.calls[-1][2][-1], audit)
        self.assertTrue(self.calls[-1][3]["AUDIT_SCORES"])
        for wrong in (MetadataTensor(shape, dtype="fp32"), MetadataTensor(shape[:-1], dtype="fp64"),
                      MetadataTensor(shape, dtype="fp64", contiguous=False)):
            with self.assertRaises(ValueError):
                self.wrapper(self.q, self.k, score_audit=wrong)

    def test_kernel_ast_keeps_one_bf16_dot_then_fp64_geometry(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        kernel = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == "_expected_fisher_seed_kernel")
        dots = [n for n in ast.walk(kernel) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "dot"]
        self.assertEqual(len(dots), 1)
        self.assertEqual(ast.literal_eval(dots[0].keywords[0].value), "ieee")
        self.assertEqual(ast.unparse(dots[0].keywords[1].value), "tl.float32")
        z = next(n for n in ast.walk(kernel) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "z")
        self.assertEqual(ast.unparse(z.value), "dot_logits.to(tl.float64) * SCALE")
        self.assertIn("for key_frame in tl.range(0, K_OUTER)", ast.unparse(kernel))
        self.assertIn("column = bank % 80", ast.unparse(kernel))
        text = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("global_probe_log_norm_triton", text)
        self.assertNotIn("from analysis", text)
        helpers = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and n.name in ("_expected_geometry", "_expected_covariance_column", "_expected_pivots")]
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                             and n.func.attr == "dot" for h in helpers for n in ast.walk(h)))


if __name__ == "__main__":
    unittest.main()
