"""CPU algebra/AST tests only; do not compile or execute native GPU kernels."""
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "attention" / "anchor_gap_guard_seed_triton.py"
REFERENCE = ROOT / "analysis" / "sq1_partition_seed_reference.py"
spec = importlib.util.spec_from_file_location("sq1_reference_for_guard_native_test", REFERENCE)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class NumpyLanguage:
    float64 = np.float64
    int32 = np.int32
    constexpr = int
    arange = staticmethod(np.arange)
    full = staticmethod(np.full)
    where = staticmethod(np.where)
    sum = staticmethod(np.sum)
    static_range = staticmethod(range)

    @staticmethod
    def max(value, axis, return_indices=False, return_indices_tie_break_left=False):
        maximum = np.max(value, axis=axis)
        if return_indices:
            if not return_indices_tie_break_left:
                raise AssertionError("lowest-index tie contract omitted")
            return maximum, np.argmax(value, axis=axis)
        return maximum


def helper_namespace():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    wanted = {"_gap_guard_log_probabilities", "_gap_guard_reverse_gap", "_gap_guard_last_slot"}
    definitions = [node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name in wanted]
    for definition in definitions:
        definition.decorator_list = []
    namespace = {"tl": NumpyLanguage, "libdevice": SimpleNamespace(exp=np.exp, log=np.log)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])),
                 str(SOURCE), "exec"), namespace)
    return namespace


def support_case(column, logits):
    quadrature = reference.uniform16_sq1_quadrature(column)
    count = len(quadrature["probe_columns"])
    padded = np.full((32, 32), 1e6, dtype=np.float64)
    padded[:, :count] = logits[:, :count]
    valid = np.arange(32) < count
    return quadrature, padded, valid


def independent_guard(logits, incumbent, singleton, measures):
    metrics = reference.sq1_partition_metrics
    prefix_gap = metrics(logits, logits[incumbent[:3]], singleton, measures=measures)["reverse_gap"]
    score = prefix_gap.copy()
    score[incumbent[:3]] = -np.inf
    challenger = int(score.argmax())
    trial = np.concatenate((incumbent[:3], [challenger]))
    base_gap = metrics(logits, logits[incumbent], singleton, measures=measures)["reverse_gap"]
    trial_gap = metrics(logits, logits[trial], singleton, measures=measures)["reverse_gap"]
    accepted = challenger != incumbent[-1] and trial_gap.mean() < base_gap.mean()
    output = trial if accepted else incumbent.copy()
    return output, prefix_gap, base_gap, trial_gap, challenger, accepted


class GapGuardAlgebraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = helper_namespace()

    def test_all80_columns_have_positive_width80_measure_no_duplicate_sq(self):
        for column in range(80):
            quad, padded, valid = support_case(column, np.zeros((32, 32)))
            actual = self.helpers["_gap_guard_log_probabilities"](padded, valid, quad["singleton_index"])
            expected = np.log(quad["measures"] / 80.0)
            np.testing.assert_allclose(actual[:, valid], np.broadcast_to(expected, (32, len(expected))),
                                       atol=1e-14, rtol=1e-14)
            self.assertEqual(len(quad["probe_columns"]), 16 if column % 5 == 0 else 17)
            self.assertEqual(len(np.unique(quad["probe_columns"])), valid.sum())
            self.assertEqual(quad["measures"][quad["singleton_index"]], 1.0)
            np.testing.assert_array_equal(actual[:, ~valid], 0.0)

    def test_actual_helpers_match_independent_sq1_reference_and_guard(self):
        rng = np.random.default_rng(21106)
        acceptances = rejections = 0
        for case in range(48):
            column = case * 13 % 80
            quad, padded, valid = support_case(column, rng.normal(size=(32, 32)) * (0.4 + case % 4))
            incumbent = rng.choice(32, size=4, replace=False)
            actual = self.helpers["_gap_guard_last_slot"](padded, valid, quad["singleton_index"], incumbent)
            expected = independent_guard(padded[:, valid], incumbent, quad["singleton_index"], quad["measures"])
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[0][:3], incumbent[:3])
            self.assertEqual(len(np.unique(actual[0])), 4)
            self.assertEqual(int(actual[4]), expected[4])
            self.assertEqual(bool(actual[5]), bool(expected[5]))
            for got, wanted in zip(actual[1:4], expected[1:4]):
                np.testing.assert_allclose(got, wanted, atol=3e-12, rtol=3e-12)
            if actual[5]:
                acceptances += 1
                self.assertLess(actual[3].mean(), actual[2].mean())
            else:
                rejections += 1
                np.testing.assert_array_equal(actual[0], incumbent)
        self.assertGreater(acceptances, 0)
        self.assertGreater(rejections, 0)

    def test_singleton_in_normalizer_before_ownership_and_empty_cells(self):
        rng = np.random.default_rng(3701)
        for column in (0, 1, 75, 79):
            quad, padded, valid = support_case(column, rng.normal(size=(32, 32)))
            singleton = quad["singleton_index"]
            padded[0, singleton] = 30.0
            padded[1] = padded[2]  # Identical profiles force an empty later cell.
            selected = np.array([0, 1, 2, 3])
            logp = self.helpers["_gap_guard_log_probabilities"](padded, valid, singleton)
            for count in (3, 4):
                actual = self.helpers["_gap_guard_reverse_gap"](logp, valid, singleton, selected, count)
                result = reference.sq1_partition_metrics(padded[:, valid], padded[selected[:count]][:, valid],
                    singleton, measures=quad["measures"], return_details=True)
                np.testing.assert_allclose(logp[:, valid], result["log_p"], atol=1e-13, rtol=1e-13)
                np.testing.assert_allclose(actual, result["reverse_gap"], atol=2e-12, rtol=2e-12)
                self.assertFalse(result["active_cells"][2])
                self.assertTrue(np.isfinite(actual).all())

    def test_row_gauges_and_extreme_logits_remain_finite(self):
        rng = np.random.default_rng(16453)
        incumbent = np.array([3, 13, 27, 31])
        for scale in (1.0, 100.0):
            quad, padded, valid = support_case(23, rng.normal(size=(32, 32)) * scale)
            original = self.helpers["_gap_guard_last_slot"](padded, valid, quad["singleton_index"], incumbent)
            shifted = padded.copy()
            shifted[:, valid] += rng.normal(size=(32, 1)) * 1000
            changed = self.helpers["_gap_guard_last_slot"](shifted, valid, quad["singleton_index"], incumbent)
            np.testing.assert_array_equal(original[0], changed[0])
            for left, right in zip(original[1:4], changed[1:4]):
                self.assertTrue(np.isfinite(left).all())
                np.testing.assert_allclose(left, right, atol=2e-11, rtol=2e-11)

    def test_exact_ties_first_unused_and_equal_objective_keep_incumbent(self):
        incumbent = np.array([5, 0, 1, 17])
        for column in (0, 7):
            quad, padded, valid = support_case(column, np.zeros((32, 32)))
            result = self.helpers["_gap_guard_last_slot"](padded, valid, quad["singleton_index"], incumbent)
            self.assertEqual(int(result[4]), 2)
            self.assertFalse(result[5])
            np.testing.assert_array_equal(result[0], incumbent)
            np.testing.assert_array_equal(result[2], result[3])

    def test_raw_negative_ranking_and_strict_mean_rule_are_not_clipped(self):
        # Controlled negative roundoff-like scores test the actual decision
        # body independently of the gap helper's floating-point reductions.
        helpers = helper_namespace()
        incumbent = np.array([0, 1, 2, 3])
        for trial_value, accepted in ((-2e-15, True), (-1e-15, False), (-.5e-15, False)):
            def gaps(logp, valid, singleton, chosen, count):
                if count == 3:
                    score = np.full(32, -3e-15); score[9] = -1e-15
                    return score
                return np.full(32, trial_value if chosen[-1] == 9 else -1e-15)
            helpers["_gap_guard_reverse_gap"] = gaps
            result = helpers["_gap_guard_last_slot"](np.zeros((32, 32)), np.arange(32) < 17, 16, incumbent)
            self.assertEqual(int(result[4]), 9)  # Clipping would wrongly choose3.
            self.assertEqual(bool(result[5]), accepted)
            self.assertEqual(int(result[0][-1]), 9 if accepted else 3)

    def test_challenger_may_be_existing_last_and_does_not_force_a_change(self):
        helpers = helper_namespace()
        incumbent = np.array([0, 1, 2, 3])
        def gaps(logp, valid, singleton, selected, count):
            result = np.ones(32)
            if count == 3:
                result[3] = 2.0
            return result
        helpers["_gap_guard_reverse_gap"] = gaps
        result = helpers["_gap_guard_last_slot"](np.zeros((32, 32)), np.arange(32) < 16, 0, incumbent)
        self.assertEqual(int(result[4]), 3)
        self.assertFalse(result[5])
        np.testing.assert_array_equal(result[0], incumbent)

    def test_identical_tuple_is_rejected_even_if_raw_reductions_differ_one_ulp(self):
        helpers = helper_namespace()
        incumbent = np.array([0, 1, 2, 20])
        prefix_gap = np.ones(32)
        prefix_gap[20] = 2.0
        base_gap = np.ones(32)
        trial_gap = np.full(32, np.nextafter(1.0, -np.inf))
        calls = iter((prefix_gap, base_gap, trial_gap))
        helpers["_gap_guard_reverse_gap"] = lambda *args: next(calls)
        result = helpers["_gap_guard_last_slot"](
            np.zeros((32, 32)), np.arange(32) < 17, 16, incumbent)
        self.assertEqual(int(result[4]), 20)
        self.assertLess(result[3].mean(), result[2].mean())
        self.assertFalse(result[5])
        np.testing.assert_array_equal(result[0], incumbent)
        np.testing.assert_array_equal(result[2], base_gap)
        np.testing.assert_array_equal(result[3], trial_gap)


class MetadataTensor:
    def __init__(self, shape, dtype="bf16", device="cuda:0", contiguous=True):
        self.shape, self.ndim = tuple(shape), len(shape)
        self.dtype, self.device = dtype, device
        self.is_cuda = device.startswith("cuda:")
        self._contiguous = contiguous

    def contiguous(self):
        return self

    def is_contiguous(self):
        return self._contiguous


def wrapper_namespace():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "anchor_gap_guard_seed_rows_triton")
    definition.body = [ast.Pass() if isinstance(node, ast.ImportFrom) else node for node in definition.body]
    calls = []
    def incumbent(q, k, scale, **kwargs):
        output = MetadataTensor((q.shape[0], 1, k.shape[1], 45, 80, q.shape[4], 4), kwargs["output_dtype"])
        calls.append(("incumbent", (q, k, scale), kwargs, output))
        return output
    def table(*args, **kwargs):
        calls.append(("table", args, kwargs))
        return "candidate_table", "anchor_table"
    class KernelRecorder:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append(("correction", grid, args, kwargs))
            return launch
    namespace = {"math": math,
        "torch": SimpleNamespace(bfloat16="bf16", uint8="uint8", int32="int32", float64="fp64"),
        "triton": SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
        "multiscale_candidate_table": table, "fisher_anchor_seed_rows_triton": incumbent,
        "_anchor_gap_guard_correction_kernel": KernelRecorder()}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[])),
                 str(SOURCE), "exec"), namespace)
    return namespace["anchor_gap_guard_seed_rows_triton"], calls


class GapGuardNativeContractTests(unittest.TestCase):
    def setUp(self):
        self.wrapper, self.calls = wrapper_namespace()
        self.q = MetadataTensor((2, 1, 45, 80, 40, 128))
        self.k = MetadataTensor((2, 21, 45, 80, 40, 128))

    def test_true_unchanged_j4_incumbent_then_actual_j1_in_place_correction(self):
        output = self.wrapper(self.q, self.k)
        self.assertEqual([call[0] for call in self.calls], ["incumbent", "table", "correction"])
        name, args, kwargs, base = self.calls[0]
        self.assertEqual(args, (self.q, self.k, 128 ** -.5))
        self.assertEqual(kwargs, {"candidate_count": 32, "candidate_geometry_count": 45, "block_j": 4,
            "probe_count": 16, "global_probe_count": 256, "output_dtype": "uint8", "gs_num_warps": 4})
        self.assertIs(output, base)
        self.assertEqual(output.shape, (2, 1, 21, 45, 80, 40, 4))
        name, grid, arguments, options = self.calls[-1]
        self.assertEqual(grid, (3600, 80))
        self.assertEqual(arguments[:3], (self.q, self.k, "candidate_table"))
        self.assertIs(arguments[3], output)
        self.assertIs(arguments[4], output)
        self.assertIs(arguments[5], output)
        self.assertEqual(options, {"K_OUTER": 21, "HEADS": 40, "D_HEAD": 128, "BLOCK_D": 128,
            "SCALE": 128 ** -.5, "AUDIT_GAPS": False, "AUDIT_DECISION": False,
            "num_warps": 4, "num_stages": 1})

    def test_conflicting_layout_precision_geometry_and_oracle_fields_rejected(self):
        for kwargs in ({"block_j": 1}, {"block_j": 8}, {"block_j": True}, {"candidate_count": 45},
                       {"candidate_geometry_count": 32}, {"probe_count": 17}, {"global_probe_count": 0},
                       {"gs_num_warps": 8}, {"scale": float("nan")}, {"scale": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.wrapper(self.q, self.k, **kwargs)
        for kwargs in ({"global_normalizer": self.q}, {"feature_mode": "fisher_mass"},
                       {"dense_reference": self.q}, {"native_low_precision_dot": True},
                       {"incumbent": self.q}, {"num_stages": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                self.wrapper(self.q, self.k, **kwargs)
        self.assertEqual(self.calls, [])

    def test_original_bf16_rf0_standard_shape_and_output_guards(self):
        for shape in ((2, 21, 45, 80, 40, 128), (2, 1, 44, 80, 40, 128),
                      (2, 1, 45, 80, 40, 129), (2, 1, 45, 80, 0, 128)):
            with self.assertRaises(ValueError):
                self.wrapper(MetadataTensor(shape), self.k)
        for dtype in ("fp16", "fp32", "fp8"):
            with self.assertRaises(TypeError):
                self.wrapper(MetadataTensor(self.q.shape, dtype), self.k)
        for wrong in (MetadataTensor(self.q.shape, device="cpu"),
                      MetadataTensor(self.q.shape, device="cuda:1")):
            with self.assertRaises(ValueError):
                self.wrapper(wrong, self.k)
        with self.assertRaises(TypeError):
            self.wrapper(self.q, self.k, output_dtype="fp32")
        with self.assertRaises(ValueError):
            self.wrapper(self.q, MetadataTensor((1, 21, 45, 80, 40, 128)))
        self.assertEqual(self.calls, [])

    def test_optional_audits_are_independent_explicit_and_preserve_incumbent_call(self):
        gaps = MetadataTensor((2, 1, 21, 45, 80, 40, 3, 32), "fp64")
        decisions = MetadataTensor((2, 1, 21, 45, 80, 40, 3), "int32")
        for gap, decision in ((gaps, None), (None, decisions), (gaps, decisions)):
            self.calls.clear()
            output = self.wrapper(self.q, self.k, .125, gap_audit=gap, decision_audit=decision)
            self.assertNotIn("gap_audit", self.calls[0][2])
            self.assertNotIn("score_audit", self.calls[0][2])
            self.assertIs(self.calls[-1][2][4], gap if gap is not None else output)
            self.assertIs(self.calls[-1][2][5], decision if decision is not None else output)
            self.assertEqual(self.calls[-1][3]["AUDIT_GAPS"], gap is not None)
            self.assertEqual(self.calls[-1][3]["AUDIT_DECISION"], decision is not None)
        for field, example in (("gap_audit", gaps), ("decision_audit", decisions)):
            for bad in (MetadataTensor(example.shape, "fp32"), MetadataTensor(example.shape[:-1], example.dtype),
                        MetadataTensor(example.shape, example.dtype, contiguous=False),
                        MetadataTensor(example.shape, example.dtype, device="cuda:1")):
                with self.assertRaises(ValueError):
                    self.wrapper(self.q, self.k, **{field: bad})

    def test_kernel_ast_exact_support_dot_precision_frame_layout_and_only_last_store(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        kernel = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                      and node.name == "_anchor_gap_guard_correction_kernel")
        text = ast.unparse(kernel)
        for required in ("duplicate = column % 5 == 0", "singleton = tl.where(duplicate, column // 5, 16)",
                         "valid = (probes < 16) | (probes == 16) & ~duplicate",
                         "z = accumulated.to(tl.float64) * SCALE", "for key_frame in tl.range(0, K_OUTER)",
                         "batch = (batch_head // HEADS).to(tl.int64)",
                         "output_bank = (((batch * K_OUTER + key_frame) * 45 + key_row) * 80 + column) * HEADS + head"):
            self.assertIn(required, text)
        dots = [node for node in ast.walk(kernel) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == "dot"]
        self.assertEqual(len(dots), 1)
        self.assertEqual(ast.unparse(dots[0]),
            "tl.dot(queries, tl.trans(keys), input_precision='ieee', out_dtype=tl.float32)")
        seed_stores = [node for node in ast.walk(kernel) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Attribute) and node.func.attr == "store"
                       and "seed_ptr" in ast.unparse(node.args[0])]
        self.assertEqual(len(seed_stores), 1)
        self.assertEqual(ast.unparse(seed_stores[0]), "tl.store(seed_ptr + output_bank * 4 + 3, last_row)")
        for store in ("tl.store(decision_ptr + output_bank * 3 + 0, challenger)",
                      "tl.store(decision_ptr + output_bank * 3 + 1, accept.to(tl.int32))",
                      "tl.store(decision_ptr + output_bank * 3 + 2, old_last)"):
            self.assertIn(store, text)
        source = SOURCE.read_text(encoding="utf-8")
        for forbidden in ("global_probe_log_norm_triton", "from analysis", "torch.empty", "torch.zeros"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
