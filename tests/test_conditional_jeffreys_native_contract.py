"""CPU-only contracts for the isolated native Conditional Jeffreys candidate.

The actual wrapper AST runs with metadata tensors and recording launchers.
The actual greedy/kernel-expression AST runs with NumPy reductions; scalar
projection from the independent reference checks its mathematical result.
These tests do not import Torch/Triton or compile/execute GPU code. Native
FP32 reductions, near ties, register pressure and timing need GPU validation.
"""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from analysis.conditional_jeffreys_seed import partition_metrics, select_jeffreys_seeds


SOURCE = Path(__file__).resolve().parents[1] / "attention/conditional_jeffreys_seed_triton.py"
PUBLIC = "conditional_jeffreys_seed_rows_triton"
KERNEL = "_conditional_jeffreys_seed_kernel"


def function(name):
    return next(node for node in ast.walk(ast.parse(SOURCE.read_text(encoding="utf-8")))
                if isinstance(node, ast.FunctionDef) and node.name == name)


def target_name(node):
    return node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None


def log_normalize(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


class NumpyCasts(ast.NodeTransformer):
    """Translate only Tensor.to(dtype); preserve all mathematical expressions."""

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
            return ast.copy_location(ast.Call(
                func=ast.Attribute(value=ast.Name(id="tl", ctx=ast.Load()), attr="cast", ctx=ast.Load()),
                args=[node.func.value, *node.args], keywords=node.keywords,
            ), node)
        return node


def actual_greedy(logits, global_norm, anchor, columns, *, reuse=True, dtype=np.float64):
    """Execute the native greedy AST on [J,32,16], returning raw trace in bits."""
    logits = np.asarray(logits, dtype=dtype)
    groups, count, probes = logits.shape
    assert count == 32 and probes == 16
    loop = next(node for node in function(KERNEL).body
                if isinstance(node, ast.For) and node.target.id == "key_outer")
    # Include the actual local-normalization front end: previously starting at
    # probability silently supplied a more stable logp than the native kernel.
    start = next(i for i, node in enumerate(loop.body) if target_name(node) == "row_max")
    end = next(i for i, node in enumerate(loop.body) if target_name(node) == "output_offset")
    body = loop.body[start:end]
    pivots = next(node for node in body if isinstance(node, ast.For) and node.target.id == "step")
    priority_at = next(i for i, node in enumerate(pivots.body) if target_name(node) == "priority")
    pivots.body[priority_at + 1:priority_at + 1] = ast.parse(
        "traces.append((gamma.copy(), priority.copy(), owner.copy(), owner_logp.copy()))"
    ).body

    def maximum(values, axis, return_indices=False, return_indices_tie_break_left=False):
        if return_indices:
            assert return_indices_tie_break_left
            return np.max(values, axis=axis), np.argmax(values, axis=axis)
        return np.max(values, axis=axis)

    arithmetic_dtype = dtype

    def zeros(shape, dtype):
        return np.zeros(shape, dtype=dtype if np.issubdtype(dtype, np.integer) else arithmetic_dtype)

    scope = {
        "tl": SimpleNamespace(
            exp2=np.exp2, log2=np.log2, where=np.where, max=maximum, sum=np.sum,
            reshape=np.reshape, zeros=zeros,
            maximum=np.maximum, static_range=range, int32=np.int32, float32=np.float32,
            cast=lambda value, target: np.asarray(value).astype(target),
        ),
        "logits": np.asarray(logits / np.log(2), dtype=dtype).reshape(-1, 16),
        "global_norm": np.asarray(global_norm, dtype=dtype) / np.log(2),
        "BLOCK_J": groups, "BLOCK_ROWS": count, "COLS": 80,
        "valid_2d": np.ones((groups, count), dtype=bool),
        "output_columns": np.asarray(columns), "probe_columns": np.arange(0, 80, 5),
        "candidate_ids": np.arange(count)[None, :],
        "candidate_rows_2d": np.broadcast_to(np.arange(count), (groups, count)),
        "anchor_index": anchor, "seed_slots": np.arange(4),
        "REUSE_OWNER_LOGP": reuse, "AUDIT_SCORES": False, "AUDIT_INTERMEDIATES": False, "traces": [],
    }
    program = ast.fix_missing_locations(NumpyCasts().visit(ast.Module(body=body, type_ignores=[])))
    exec(compile(program, str(SOURCE), "exec"), scope)
    return scope["seeds"], scope["traces"]


def actual_log_softmax2(logits2):
    """Run exactly the native base2 normalization assignments, including offsets."""
    groups = logits2.shape[0]
    loop = next(node for node in function(KERNEL).body
                if isinstance(node, ast.For) and node.target.id == "key_outer")
    start = next(i for i, node in enumerate(loop.body) if target_name(node) == "row_max")
    end = next(i for i, node in enumerate(loop.body) if target_name(node) == "probability")
    scope = {"tl": SimpleNamespace(max=np.max, log2=np.log2, sum=np.sum, exp2=np.exp2, reshape=np.reshape),
             "logits": logits2.reshape(-1, 16), "BLOCK_J": groups, "BLOCK_ROWS": 32}
    exec(compile(ast.Module(body=loop.body[start:end], type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope["log_probability"]


class MetadataTensor:
    def __init__(self, shape=(2, 1, 45, 80, 3, 128), dtype="bf16", device="cuda:metadata-only", contiguous=True):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.ndim, self.is_cuda, self._contiguous = len(shape), device.startswith("cuda"), contiguous

    def size(self, axis):
        return self.shape[axis]

    def contiguous(self):
        return self

    def is_contiguous(self):
        return self._contiguous


class RecordingKernel:
    def __init__(self, launches):
        self.launches = launches

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append(SimpleNamespace(grid=grid, args=args, kwargs=kwargs))
        return launch


class NativeWrapperContracts(unittest.TestCase):
    def setUp(self):
        self.launches, self.allocations = [], []
        self.q, self.k = MetadataTensor(), MetadataTensor((2, 5, 45, 80, 3, 128))
        self.norm = MetadataTensor((2, 1, 45, 80, 3), "fp32")
        package_name = "_conditional_jeffreys_contract"
        package = ModuleType(package_name)
        package.__path__ = []
        candidates = ModuleType(package_name + ".adaptive_seed_pytorch")
        candidates.multiscale_candidate_table = self.candidates = mock.Mock(return_value=(object(), object()))
        normalizer = ModuleType(package_name + ".entropy_mass_seed_triton")
        normalizer.global_probe_log_norm_triton = self.normalizer = mock.Mock(return_value=self.norm)
        modules = mock.patch.dict(sys.modules, {
            package_name: package, candidates.__name__: candidates, normalizer.__name__: normalizer,
        })
        modules.start()
        self.addCleanup(modules.stop)

        def empty(shape, *, device, dtype):
            value = MetadataTensor(shape, dtype, device)
            self.allocations.append(value)
            return value

        self.scope = {
            "__package__": package_name, "math": math,
            "torch": SimpleNamespace(bfloat16="bf16", float16="fp16", float32="fp32", uint8="u8", int32="i32", empty=empty),
            "triton": SimpleNamespace(cdiv=lambda x, y: (x + y - 1) // y,
                                      next_power_of_2=lambda x: 1 << (x - 1).bit_length()),
            KERNEL: RecordingKernel(self.launches),
        }
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        program = ast.fix_missing_locations(ast.Module(body=[future, function(PUBLIC)], type_ignores=[]))
        exec(compile(program, str(SOURCE), "exec"), self.scope)
        self.selector = self.scope[PUBLIC]

    def test_default_launch_and_original_normalizer(self):
        result = self.selector(self.q, self.k)
        self.assertEqual(result.shape, (2, 1, 5, 45, 80, 3, 4))
        self.assertEqual(result.dtype, "u8")
        self.assertEqual(len(self.allocations), 1)
        self.normalizer.assert_called_once_with(self.q, self.k, scale=128 ** -0.5, global_probe_count=256)
        self.candidates.assert_called_once_with(45, 32, self.q.device, geometry_count=45)
        launch = self.launches[0]
        self.assertEqual(launch.grid, (45 * 20, 6))
        for name, value in {"K_OUTER": 5, "ROWS": 45, "COLS": 80, "CANDIDATES": 32,
                            "BLOCK_J": 4, "BLOCK_ROWS": 32, "num_warps": 4,
                            "num_stages": 1, "REUSE_OWNER_LOGP": True, "AUDIT_SCORES": False}.items():
            self.assertEqual(launch.kwargs[name], value)
        self.assertIs(launch.args[2], self.norm)
        self.assertIs(launch.args[6], result)  # Compile-time disabled audit uses a dummy pointer.
        self.assertIs(launch.args[7], result)
        self.assertFalse(launch.kwargs["AUDIT_INTERMEDIATES"])

    def test_shared_normalizer_audit_and_execution_alias(self):
        audit = MetadataTensor((2, 1, 5, 45, 80, 3, 3, 32), "fp32")
        self.selector(self.q, self.k, global_normalizer=self.norm, score_audit=audit,
                      num_warps=8, reuse_owner_logp=False, output_dtype="i32")
        self.normalizer.assert_not_called()
        launch = self.launches[0]
        self.assertIs(launch.args[6], audit)
        self.assertTrue(launch.kwargs["AUDIT_SCORES"])
        self.assertFalse(launch.kwargs["REUSE_OWNER_LOGP"])
        self.assertEqual(launch.kwargs["num_warps"], 8)
        self.assertEqual(len(self.allocations), 1)

    def test_api_defaults_and_no_fitted_or_oracle_parameter(self):
        defaults = {name: value.default for name, value in inspect.signature(self.selector).parameters.items()}
        self.assertEqual(defaults["candidate_count"], 32)
        self.assertEqual(defaults["candidate_geometry_count"], 45)
        self.assertEqual(defaults["probe_count"], 16)
        self.assertEqual(defaults["global_probe_count"], 256)
        self.assertEqual(defaults["gs_num_warps"], 4)
        self.assertTrue(defaults["reuse_owner_logp"])
        for excluded in ["alpha", "temperature", "exact_z", "no_sq_score", "reference_frame"]:
            self.assertNotIn(excluded, defaults)

    def test_reject_changes_to_prespecified_candidate(self):
        for option in [{"probe_count": 80}, {"global_probe_count": 128}, {"candidate_count": 16},
                       {"candidate_geometry_count": 32}, {"block_j": 3}, {"gs_num_warps": 2},
                       {"num_stages": 0}, {"scale": float("nan")}, {"scale": float("inf")}]:
            with self.subTest(option=option), self.assertRaises(ValueError):
                self.selector(self.q, self.k, **option)
        with self.assertRaises(TypeError):
            self.selector(self.q, self.k, reuse_owner_logp=1)
        for wrong in [MetadataTensor((2, 2, 45, 80, 3, 128)), MetadataTensor((2, 1, 44, 80, 3, 128)),
                      MetadataTensor((2, 1, 45, 79, 3, 128)), MetadataTensor(device="cpu")]:
            with self.assertRaises(ValueError):
                self.selector(wrong, self.k)
        self.assertEqual(self.launches, [])

    def test_reject_bad_optional_buffers(self):
        for wrong in [MetadataTensor((2, 1, 45, 80, 3), "bf16"),
                      MetadataTensor((2, 1, 45, 80, 4), "fp32"),
                      MetadataTensor((2, 1, 45, 80, 3), "fp32", contiguous=False)]:
            with self.assertRaises(ValueError):
                self.selector(self.q, self.k, global_normalizer=wrong)
        for wrong in [MetadataTensor((2, 1, 5, 45, 80, 3, 3, 32), "bf16"),
                      MetadataTensor((2, 1, 5, 45, 80, 3, 3, 31), "fp32"),
                      MetadataTensor((2, 1, 5, 45, 80, 3, 3, 32), "fp32", contiguous=False)]:
            with self.assertRaises(ValueError):
                self.selector(self.q, self.k, global_normalizer=self.norm, score_audit=wrong)
        self.assertEqual(self.launches, [])

    def test_targeted_intermediate_audit_layout_and_coordinates(self):
        intermediate = MetadataTensor((19, 32, 16), "fp32")
        coordinate = (1, 3, 22, 20, 2)
        self.selector(self.q, self.k, global_normalizer=self.norm,
                      intermediate_audit=intermediate, intermediate_audit_bank=coordinate)
        launch = self.launches[0]
        self.assertIs(launch.args[7], intermediate)
        self.assertTrue(launch.kwargs["AUDIT_INTERMEDIATES"])
        for name, expected in zip(["AUDIT_BATCH", "AUDIT_FRAME", "AUDIT_ROW", "AUDIT_COLUMN", "AUDIT_HEAD"], coordinate):
            self.assertEqual(launch.kwargs[name], expected)
        self.assertEqual(len(self.allocations), 1)

    def test_targeted_intermediate_audit_rejects_bad_pair_or_coordinates(self):
        valid = MetadataTensor((19, 32, 16), "fp32")
        for bank in [None, [0, 0, 0, 0, 0], (0, 0, 0, 0), (0, 5, 0, 0, 0),
                     (0, 0, 45, 0, 0), (0, 0, 0, 80, 0), (0, 0, 0, 0, 3), (True, 0, 0, 0, 0)]:
            with self.assertRaises(ValueError):
                self.selector(self.q, self.k, global_normalizer=self.norm,
                              intermediate_audit=valid, intermediate_audit_bank=bank)
        for wrong in [MetadataTensor((19, 32, 16), "bf16"), MetadataTensor((18, 32, 16), "fp32"),
                      MetadataTensor((19, 32, 16), "fp32", contiguous=False)]:
            with self.assertRaises(ValueError):
                self.selector(self.q, self.k, global_normalizer=self.norm,
                              intermediate_audit=wrong, intermediate_audit_bank=(0, 0, 0, 0, 0))
        with self.assertRaises(ValueError):
            self.selector(self.q, self.k, global_normalizer=self.norm, intermediate_audit_bank=(0, 0, 0, 0, 0))
        self.assertEqual(self.launches, [])


class ActualKernelMathContracts(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(604)

    def check_reference(self, logits, norm, anchor, columns, reuse=True, atol=3e-12):
        chosen, traces = actual_greedy(logits, norm, anchor, columns, reuse=reuse)
        local_norm = np.logaddexp.reduce(logits, axis=-1)
        mass = local_norm - norm
        importance = np.exp(mass - mass.max(axis=-1, keepdims=True))
        singleton = np.where(np.asarray(columns) % 5 == 0, np.asarray(columns) // 5, -1)
        expected = select_jeffreys_seeds(logits, anchor, importance, 4, singleton)
        np.testing.assert_array_equal(chosen, expected)
        for index, (gamma, priority, owner, owner_logp) in enumerate(traces, start=1):
            prefix = chosen[:, :index]
            selected = np.take_along_axis(logits, np.broadcast_to(prefix[..., None], (*prefix.shape, 16)), axis=1)
            reference = partition_metrics(logits, selected, singleton, return_distributions=True)
            np.testing.assert_allclose(gamma * np.log(2), reference["jeffreys"], atol=atol, rtol=atol)
            np.testing.assert_allclose(priority * np.log(2), importance * np.maximum(reference["jeffreys"], 0), atol=atol, rtol=atol)
            np.testing.assert_array_equal(owner, reference["owner_before_singleton"])
            winning = np.max(log_normalize(selected), axis=1) / np.log(2)
            np.testing.assert_allclose(owner_logp, winning, atol=atol, rtol=atol)
        return chosen, traces

    def test_random_scores_prefixes_and_owner_reuse_match_reference(self):
        logits = self.rng.normal(size=(4, 32, 16)) * 2.1
        norm = self.rng.normal(size=(4, 32)) * .7 + 10
        for reuse in [False, True]:
            self.check_reference(logits, norm, 17, [0, 1, 5, 79], reuse)

    def test_identical_rows_empty_cells_deterministic_low_ties(self):
        logits = np.zeros((4, 32, 16))
        for reuse in [False, True]:
            selected, traces = self.check_reference(logits, np.zeros((4, 32)), 7, [0, 1, 10, 79], reuse)
            np.testing.assert_array_equal(selected, np.tile([7, 0, 1, 2], (4, 1)))
            self.assertTrue(all(np.isfinite(t[0]).all() for t in traces))

    def test_peaked_finite_logits_keep_stable_owned_logs(self):
        logits = self.rng.normal(size=(4, 32, 16)) * 800
        self.check_reference(logits, np.zeros((4, 32)), 3, [0, 1, 10, 79], atol=4e-9)

    def test_row_shift_invariance_when_normalizer_shifts_together(self):
        logits = self.rng.normal(size=(4, 32, 16))
        norm = self.rng.normal(size=(4, 32))
        shifts = self.rng.normal(size=(4, 32)) * 100
        first, _ = actual_greedy(logits, norm, 4, [0, 1, 5, 79])
        second, _ = actual_greedy(logits + shifts[..., None], norm + shifts, 4, [0, 1, 5, 79])
        np.testing.assert_array_equal(first, second)

    def test_singleton_changes_scoring_but_never_owner_profiles(self):
        # Column0 is observed; column1 is absent from Uniform16. This differs
        # only in post-ownership cell support, not full16 p or mass weighting.
        one = self.rng.normal(size=(1, 32, 16)) * 3
        logits = np.repeat(one, 2, axis=0)
        norm = np.zeros((2, 32))
        _, traces = self.check_reference(logits, norm, 5, [0, 1])
        first = traces[0]
        np.testing.assert_array_equal(first[2][0], first[2][1])
        np.testing.assert_array_equal(first[3][0], first[3][1])
        self.assertGreater(np.max(np.abs(first[0][0] - first[0][1])), 1e-5)

    def test_fp32_tiny_score_audit_is_finite_not_claimed_bit_exact(self):
        logits = self.rng.normal(size=(4, 32, 16)).astype(np.float32) * 1e-4
        selected, traces = actual_greedy(logits, np.zeros((4, 32)), 4, [0, 1, 5, 79], dtype=np.float32)
        self.assertTrue(all(len(set(row)) == 4 and row[0] == 4 for row in selected))
        for gamma, priority, _, _ in traces:
            self.assertTrue(np.isfinite(gamma).all() and np.isfinite(priority).all())
            self.assertTrue((priority >= 0).all())
        # This is an audit of cancellation safety, not FP64 exact rank parity.

    def test_high_offset_frontend_keeps_normalized_small_log_probabilities(self):
        profile = np.array([0, -1, -3, -6, -7, -9, -12, -13, -14, -15, -16, -17, -18, -19, -20, -23], dtype=np.float32)
        logits = np.broadcast_to(profile, (4, 32, 16)).copy()
        shifted = logits + np.array([0, 512, 1024, 4096], dtype=np.float32)[:, None, None]
        actual = actual_log_softmax2(shifted)
        expected = log_normalize(shifted.astype(np.float64) * np.log(2)) / np.log(2)
        np.testing.assert_allclose(actual, expected, atol=1.2e-6, rtol=1e-7)
        np.testing.assert_allclose(np.exp2(actual).sum(-1), 1, atol=2e-7, rtol=0)
        np.testing.assert_array_equal(actual, np.broadcast_to(actual[0], actual.shape))
        # Demonstrate why the former absolute-LSE subtraction is unsafe here.
        peak = shifted.max(-1)
        old = shifted - (peak + np.log2(np.exp2(shifted - peak[..., None]).sum(-1)))[..., None]
        self.assertGreater(float(np.max(np.abs(old - expected))), 1e-5)

    def test_original_probe_and_audit_storage_expressions(self):
        kernel = function(KERNEL)
        assignments = {target_name(n): n for n in ast.walk(kernel) if target_name(n)}
        self.assertEqual(ast.unparse(assignments["probe_columns"].value), "probe_slots * (COLS // 16)")
        cell = ast.unparse(assignments["cell"].value)
        self.assertIn("owner == cell_slot", cell)
        self.assertIn("probe_columns[None, :] != output_columns[:, None]", cell)
        update = ast.unparse(assignments["new_owner"].value)
        self.assertEqual(update, "new_logp > owner_logp")
        priority = ast.unparse(assignments["priority"].value)
        self.assertEqual(priority, "importance * tl.maximum(gamma, 0.0)")
        audit = ast.unparse(assignments["audit_offset"].value)
        self.assertIn("3 * CANDIDATES", audit)
        self.assertIn("(step - 1) * CANDIDATES + candidate_ids", audit)

    def test_actual_intermediate_pointer_shape_and_single_writer_mask(self):
        assignments = {target_name(n): n for n in ast.walk(function(KERNEL)) if target_name(n)}
        statements = [assignments[name] for name in (
            "intermediate_offset", "intermediate_bank", "intermediate_column", "intermediate_mask")]
        for groups in [1, 2, 4, 8]:
            scope = {"tl": SimpleNamespace(zeros=np.zeros, int32=np.int32),
                     "BLOCK_J": groups, "candidate_ids": np.arange(32)[None, :],
                     "probe_slots": np.arange(16), "output_columns": np.arange(20, 20 + groups),
                     "valid_2d": np.ones((groups, 32), dtype=bool),
                     "batch": 0, "head": 3, "key_row": 22, "key_outer": 10,
                     "AUDIT_BATCH": 0, "AUDIT_HEAD": 3, "AUDIT_ROW": 22,
                     "AUDIT_FRAME": 10, "AUDIT_COLUMN": 20}
            exec(compile(ast.Module(body=statements, type_ignores=[]), str(SOURCE), "exec"), scope)
            pointer = scope["intermediate_offset"]
            self.assertEqual(pointer.shape, (groups, 32, 16))
            # Simulate tl.store broadcasting every actual payload to pointer.
            for value_shape in [(groups, 32, 16), (groups, 32, 1), (groups, 1, 16)]:
                self.assertEqual(np.broadcast_to(np.zeros(value_shape), pointer.shape).shape, pointer.shape)
            mask = np.broadcast_to(scope["intermediate_mask"], pointer.shape)
            addresses = pointer[mask]
            np.testing.assert_array_equal(np.sort(addresses), np.arange(32 * 16))
            self.assertEqual(len(np.unique(addresses)), len(addresses))
            # Other frame/programs must never write the single-bank buffer.
            scope["key_outer"] = 9
            exec(compile(ast.Module(body=[assignments["intermediate_mask"]], type_ignores=[]), str(SOURCE), "exec"), scope)
            self.assertFalse(scope["intermediate_mask"].any())


if __name__ == "__main__":
    unittest.main()
