"""CPU math and actual native-AST contracts; no Torch/Triton import or GPU."""

from __future__ import annotations

import ast
import inspect
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from analysis.affine_hellinger_seed import (
    audit_affine_scores, prefix_scores, reference_logits_from_vectors,
    root_profiles, select_affine_seeds,
)
from analysis.verify_affine_hellinger_native import rows_to_candidate_slots

SOURCE = Path(__file__).resolve().parents[1] / "attention/affine_hellinger_seed_triton.py"
PUBLIC = "affine_hellinger_seed_rows_triton"
KERNEL = "_affine_hellinger_seed_kernel"


def function(name):
    return next(node for node in ast.walk(ast.parse(SOURCE.read_text(encoding="utf-8")))
                if isinstance(node, ast.FunctionDef) and node.name == name)


def target_name(node):
    return node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None


class NumpyCasts(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
            return ast.copy_location(ast.Call(
                func=ast.Attribute(value=ast.Name(id="tl", ctx=ast.Load()), attr="cast", ctx=ast.Load()),
                args=[node.func.value, *node.args], keywords=node.keywords), node)
        return node


def actual_greedy(logits, anchor, *, dtype=np.float64, feature_mode="hellinger"):
    """Run actual front-end and greedy AST, not a rewritten native formula."""
    logits = np.asarray(logits, dtype=dtype)
    groups, count, probes = logits.shape
    assert count in (32,45) and probes == 16
    block_rows=1 << (count-1).bit_length()
    if block_rows!=count:
        logits=np.pad(logits,((0,0),(0,block_rows-count),(0,0)))
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

    scope = {
        "FEATURE_MODE": feature_mode,
        "tl": SimpleNamespace(exp2=np.exp2, sqrt=np.sqrt, rsqrt=lambda x: 1 / np.sqrt(x),
                              max=maximum, sum=np.sum, where=np.where, reshape=np.reshape,
                              static_range=range, int32=np.int32, float64=np.float64,
                              broadcast_to=np.broadcast_to,
                              cast=lambda x, target: np.asarray(x).astype(target)),
        "libdevice": SimpleNamespace(exp=np.exp, log=np.log),
        "dot_logits": logits.reshape(-1,16), "SCALE": 1.0, "CANDIDATES": count,
        "logits": (logits / np.log(2)).astype(dtype).reshape(-1, 16),
        "BLOCK_J": groups, "BLOCK_ROWS": block_rows,
        "valid_2d": np.broadcast_to(np.arange(block_rows)<count,(groups,block_rows)),
        "candidate_ids": np.arange(block_rows)[None, :], "anchor_index": anchor,
        "candidate_rows_2d": np.broadcast_to(np.where(np.arange(block_rows)<count,np.arange(block_rows),0), (groups, block_rows)),
        "seed_slots": np.arange(4)[None, :], "AUDIT_SCORES": False, "traces": [],
    }
    tree = ast.fix_missing_locations(NumpyCasts().visit(ast.Module(body=body, type_ignores=[])))
    exec(compile(tree, str(SOURCE), "exec"), scope)
    return scope["seeds"], np.stack(scope["traces"], axis=1)[:,:,:count]


class AffineMathTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(905)
        self.logits = self.rng.normal(size=(4, 32, 16)) * 2

    def test_first_score_is_full_root_distance_to_real_anchor(self):
        profiles = root_profiles(self.logits)
        expected = np.sum((profiles - profiles[:, 7:8]) ** 2, axis=-1)
        np.testing.assert_allclose(prefix_scores(self.logits, [7]), expected, atol=1e-15)
        seeds, scores = select_affine_seeds(self.logits, 7, return_scores=True)
        np.testing.assert_array_equal(seeds[:, 0], 7)
        np.testing.assert_allclose(scores[:, 0], expected, atol=1e-15)

    def test_actual_kernel_frontend_and_all_actual_prefix_scores(self):
        for groups in [1, 2, 4, 8]:
            logits = self.rng.normal(size=(groups, 32, 16)) * 2
            chosen, scores = actual_greedy(logits, 17)
            expected, expected_scores = select_affine_seeds(logits, 17, return_scores=True)
            np.testing.assert_array_equal(chosen, expected)
            np.testing.assert_allclose(scores, expected_scores, atol=2e-14, rtol=2e-14)
            for step in range(1, 4):
                np.testing.assert_allclose(scores[:, step - 1], prefix_scores(logits, chosen[:, :step]), atol=2e-14)

    def test_fixed_prefix_reference_matches_independent_affine_lstsq(self):
        profile = root_profiles(self.logits)
        prefix = [3, 11, 25]
        score = prefix_scores(self.logits, prefix)
        for bank in range(4):
            difference = profile[bank] - profile[bank, 3]
            basis = difference[prefix[1:]].T
            coefficients = np.linalg.lstsq(basis, difference.T, rcond=None)[0]
            residual = difference.T - basis @ coefficients
            # Reference GS uses the shared 1e-12 safeguard; independent exact
            # affine projection differs only below this numerical margin here.
            np.testing.assert_allclose(score[bank], np.sum(residual * residual, axis=0), atol=2e-11, rtol=2e-11)

    def test_uniform_candidate_is_nonzero_relative_to_nonuniform_anchor(self):
        p = np.full((32, 16), 1 / 16)
        p[0] = np.array([.85] + [.01] * 15)
        logits = np.log(p)
        roots = root_profiles(logits)
        centered = roots - roots.mean(axis=-1, keepdims=True)
        np.testing.assert_array_equal(centered[1], np.zeros(16))
        self.assertGreater(prefix_scores(logits, [0])[1], .5)
        self.assertGreater(np.log(16) + np.sum(p[0] * np.log(p[0])), 1.5)
        chosen = select_affine_seeds(logits, 0)
        np.testing.assert_array_equal(chosen, [0, 1, 2, 3])

    def test_identical_profiles_distinct_lowest_slot_ties(self):
        logits = np.zeros((4, 32, 16))
        for method in [actual_greedy, lambda z, a: select_affine_seeds(z, a, return_scores=True)]:
            chosen, scores = method(logits, 7)
            np.testing.assert_array_equal(chosen, np.tile([7, 0, 1, 2], (4, 1)))
            np.testing.assert_array_equal(scores, np.zeros((4, 3, 32)))

    def test_large_row_shifts_and_peaks_are_stable(self):
        logits = self.rng.integers(-30, 1, size=(4, 32, 16)).astype(np.float32)
        shifts = np.array([0, 512, 1024, 4096])[:, None, None]
        for z in [logits, logits + shifts, logits * 800]:
            chosen, score = actual_greedy(z, 4, dtype=np.float32)
            self.assertTrue(np.isfinite(score).all())
            report, _ = audit_affine_scores(z.astype(np.float64), 4, chosen, score)
            self.assertTrue(report["pass"], report)
        first, _ = actual_greedy(logits, 4)
        second, _ = actual_greedy(logits + shifts, 4)
        np.testing.assert_array_equal(first, second)

    def test_probability_and_additive_logit_gauge_invariance(self):
        profiles = root_profiles(self.logits)
        np.testing.assert_allclose(np.sum(profiles * profiles, axis=-1), 1)
        shift = self.rng.normal(size=(4, 32, 1)) * 100
        np.testing.assert_allclose(root_profiles(self.logits + shift), profiles, atol=1e-14)
        np.testing.assert_array_equal(select_affine_seeds(self.logits, 4), select_affine_seeds(self.logits + shift, 4))

    def test_fp64_dot_cast_occurs_before_accumulation(self):
        # BF16-representable powers produce an FP32 cancellation failure.
        q = np.array([[2**24, 1, -(2**24)]], dtype=np.float32)
        k = np.ones((1, 3), dtype=np.float32)
        actual = reference_logits_from_vectors(q, k, .5)
        self.assertEqual(actual.dtype, np.float64)
        self.assertEqual(actual[0, 0], .5)

    def test_audit_checks_all_scores_including_used_candidates(self):
        chosen, scores = select_affine_seeds(self.logits, 6, return_scores=True)
        report, reference = audit_affine_scores(self.logits, 6, chosen, scores)
        self.assertTrue(report["pass"])
        self.assertEqual(report["score_count"], 4 * 3 * 32)
        np.testing.assert_array_equal(reference, scores)
        scores[0, 0, 6] += .001  # used anchor is not exempt from raw score audit.
        report, _ = audit_affine_scores(self.logits, 6, chosen, scores)
        self.assertFalse(report["pass"])
        self.assertEqual(report["score_failures"], 1)
        json.dumps(report, allow_nan=False)

    def test_audit_actual_prefix_not_free_running_reference(self):
        chosen = select_affine_seeds(self.logits, 2)
        chosen[:, 1] = 31
        # Deliberately choose valid distinct, non-greedy rows.
        chosen[:] = [2, 31, 30, 29]
        scores = np.stack([prefix_scores(self.logits, chosen[:, :s]) for s in range(1, 4)], axis=1)
        report, _ = audit_affine_scores(self.logits, 2, chosen, scores)
        self.assertEqual(report["score_failures"], 0)
        self.assertGreater(report["rank_failures"], 0)
        self.assertFalse(report["pass"])

    def test_audit_structure_nonfinite_and_fixed_tolerance(self):
        chosen, scores = select_affine_seeds(self.logits, 2, return_scores=True)
        broken = chosen.copy()
        broken[0, 1] = 2
        report, _ = audit_affine_scores(self.logits, 2, broken, scores)
        self.assertFalse(report["structure_pass"])
        scores[0, 0, 0] = np.nan
        report, _ = audit_affine_scores(self.logits, 2, chosen, scores)
        self.assertFalse(report["pass"])
        self.assertEqual(report["nonfinite_scores"], 1)
        self.assertEqual(report["score_atol"], 2e-5)
        self.assertEqual(report["score_rtol"], 1e-4)
        json.dumps(report, allow_nan=False)

    def test_bad_reference_metadata_rejected(self):
        for prefix in [[0, 0], [32], [-1], [1.5], []]:
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                prefix_scores(self.logits, prefix)
        for anchor in [-1, 32, 1.5]:
            with self.assertRaises(ValueError):
                select_affine_seeds(self.logits, anchor)
        with self.assertRaises(ValueError):
            root_profiles([[np.nan]])

    def test_grid_rows_are_mapped_to_slots_and_missing_or_duplicate_rejected(self):
        table = np.broadcast_to(np.arange(10, 42), (2, 32))
        native = np.array([[15, 10, 41, 25], [14, 13, 12, 11]])
        np.testing.assert_array_equal(rows_to_candidate_slots(table, native), native - 10)
        for bad in [np.array([[0, 10, 41, 25], [14, 13, 12, 11]]), native.astype(float)]:
            with self.assertRaises(ValueError):
                rows_to_candidate_slots(table, bad)
        duplicated = table.copy()
        duplicated[:, 0] = 15
        with self.assertRaises(ValueError):
            rows_to_candidate_slots(duplicated, native)


class MetadataTensor:
    def __init__(self, shape=(2, 1, 45, 80, 3, 128), dtype="bf16", device="cuda:metadata", contiguous=True):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.ndim, self.is_cuda, self._contiguous = len(shape), device.startswith("cuda"), contiguous

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


class NativeMetadataTests(unittest.TestCase):
    def setUp(self):
        self.launches, self.allocations = [], []
        self.q, self.k = MetadataTensor(), MetadataTensor((2, 5, 45, 80, 3, 128))
        package_name = "_affine_hellinger_contract"
        package = ModuleType(package_name)
        package.__path__ = []
        candidates = ModuleType(package_name + ".adaptive_seed_pytorch")
        candidates.multiscale_candidate_table = self.candidates = mock.Mock(return_value=(object(), object()))
        modules = mock.patch.dict(sys.modules, {package_name: package, candidates.__name__: candidates})
        modules.start()
        self.addCleanup(modules.stop)

        def empty(shape, *, device, dtype):
            result = MetadataTensor(shape, dtype, device)
            self.allocations.append(result)
            return result

        scope = {
            "__package__": package_name, "math": math,
            "torch": SimpleNamespace(bfloat16="bf16", float16="fp16", float32="fp32", uint8="u8", int32="i32", empty=empty),
            "triton": SimpleNamespace(cdiv=lambda x, y: (x + y - 1) // y,
                                      next_power_of_2=lambda x: 1 << (x - 1).bit_length()),
            KERNEL: RecordingKernel(self.launches),
        }
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        tree = ast.fix_missing_locations(ast.Module(body=[future, function(PUBLIC)], type_ignores=[]))
        exec(compile(tree, str(SOURCE), "exec"), scope)
        self.selector = scope[PUBLIC]

    def test_default_c32_u16_rf0_launch_no_mass_or_global_normalizer(self):
        output = self.selector(self.q, self.k)
        self.assertEqual(output.shape, (2, 1, 5, 45, 80, 3, 4))
        self.assertEqual(output.dtype, "u8")
        self.assertEqual(len(self.allocations), 1)
        self.candidates.assert_called_once_with(45, 32, self.q.device, geometry_count=45)
        launch = self.launches[0]
        self.assertEqual(launch.grid, (45 * 20, 6))
        self.assertIs(launch.args[5], output)
        self.assertFalse(launch.kwargs["AUDIT_SCORES"])
        self.assertEqual(launch.kwargs["BLOCK_D"], 128)
        self.assertEqual(launch.kwargs["SCALE"], 128 ** -.5)
        self.assertEqual(launch.kwargs["num_warps"], 4)
        parameters = inspect.signature(self.selector).parameters
        for excluded in ["alpha", "weight_mode", "global_normalizer", "global_probe_count", "singleton_complement", "reference_frame"]:
            self.assertNotIn(excluded, parameters)

    def test_audit_layout_warps_alias_output_dtype(self):
        audit = MetadataTensor((2, 1, 5, 45, 80, 3, 3, 32), "fp32")
        out = self.selector(self.q, self.k, score_audit=audit, num_warps=8, output_dtype="i32")
        self.assertEqual(out.dtype, "i32")
        self.assertIs(self.launches[0].args[5], audit)
        self.assertTrue(self.launches[0].kwargs["AUDIT_SCORES"])
        self.assertEqual(self.launches[0].kwargs["num_warps"], 8)

    def test_anchor_full45_preserves_output_and_pads_candidate_tile(self):
        out=self.selector(self.q,self.k,candidate_count=45,feature_mode='fisher_anchor')
        self.assertEqual(out.shape,(2,1,5,45,80,3,4))
        self.candidates.assert_called_once_with(45,45,self.q.device,geometry_count=45)
        self.assertEqual(self.launches[0].kwargs['CANDIDATES'],45)
        self.assertEqual(self.launches[0].kwargs['BLOCK_ROWS'],64)

    def test_reject_protocol_input_and_buffer_changes(self):
        for kwargs in [{"probe_count": 80}, {"candidate_count": 16}, {"candidate_geometry_count": 32},
                       {"block_j": 3}, {"gs_num_warps": 2}, {"num_stages": 0}, {"scale": np.nan}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.selector(self.q, self.k, **kwargs)
        for bad in [MetadataTensor((2, 2, 45, 80, 3, 128)), MetadataTensor((2, 1, 44, 80, 3, 128)),
                    MetadataTensor((2, 1, 45, 80, 3, 256)), MetadataTensor(device="cpu")]:
            with self.assertRaises(ValueError):
                self.selector(bad, self.k)
        with self.assertRaises(TypeError):
            self.selector(MetadataTensor(dtype="fp32"), MetadataTensor((2, 5, 45, 80, 3, 128), "fp32"))
        for wrong in [MetadataTensor((2, 1, 5, 45, 80, 3, 3, 32), "bf16"),
                      MetadataTensor((2, 1, 5, 45, 80, 3, 3, 31), "fp32"),
                      MetadataTensor((2, 1, 5, 45, 80, 3, 3, 32), "fp32", contiguous=False)]:
            with self.assertRaises(ValueError):
                self.selector(self.q, self.k, score_audit=wrong)
        self.assertEqual(self.launches, [])

    def test_actual_audit_pointer_unique_full_layout_coverage(self):
        assignments = {target_name(n): n for n in ast.walk(function(KERNEL)) if target_name(n)}
        body = [assignments["audit_offset"]]
        for block_j in [1, 2, 4, 8]:
            scope = {"batch": 1, "K_OUTER": 5, "key_outer": 3, "ROWS": 45, "key_row": 22,
                     "COLS": 80, "output_columns": np.arange(20, 20 + block_j),
                     "HEADS": 3, "head": 2, "CANDIDATES": 32,
                     "candidate_ids": np.arange(32)[None, :], "step": 2}
            exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), scope)
            expected = np.empty((block_j, 32), dtype=np.int64)
            for j in range(block_j):
                for c in range(32):
                    expected[j, c] = np.ravel_multi_index((1, 0, 3, 22, 20 + j, 2, 1, c), (2, 1, 5, 45, 80, 3, 3, 32))
            np.testing.assert_array_equal(scope["audit_offset"], expected)
            self.assertEqual(len(np.unique(expected)), block_j * 32)

    def test_native_gs_has_no_anchor_projection_or_per_profile_mean(self):
        kernel = function(KERNEL)
        assignments = {target_name(n): n for n in ast.walk(kernel) if target_name(n)}
        self.assertEqual(ast.unparse(assignments["residual"].value), "profile - anchor_profile[:, None, :]")
        score = next(n for n in ast.walk(kernel) if target_name(n) == "score")
        self.assertEqual(ast.unparse(score.value), "tl.sum(residual * residual, axis=2)")
        self.assertEqual(ast.unparse(assignments["probe_columns"].value), "tl.arange(0, 16) * (COLS // 16)")
        for node in ast.walk(kernel):
            if isinstance(node, ast.If) and ast.unparse(node.test) == "step < 3":
                self.assertTrue(any(target_name(x) == "projection" for x in node.body))
                break
        else:
            self.fail("GS projection must only follow pivots 1 and 2")


if __name__ == "__main__":
    unittest.main()
