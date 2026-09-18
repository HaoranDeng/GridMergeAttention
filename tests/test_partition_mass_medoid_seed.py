"""CPU math/static contracts for the isolated first-medoid experiment.

These tests do not compile Triton, launch CUDA, or claim native numerical
parity. NumPy executes the actual kernel's small medoid expression; explicit
pairwise KL is the independent oracle. The three copied greedy steps must
remain AST-identical to the unchanged production kernel.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "attention/partition_mass_medoid_seed_triton.py"
BASE = ROOT / "attention/partition_mass_seed_triton.py"


def function(path, name):
    return next(node for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                if isinstance(node, ast.FunctionDef) and node.name == name)


def assignment_name(node):
    return node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None


def key_loop(path, name):
    return next(node for node in function(path, name).body
                if isinstance(node, ast.For) and node.target.id == "key_outer")


def medoid(logp, importance, valid=None):
    """Evaluate actual source expressions only; no separate medoid formula."""
    loop = key_loop(SOURCE, "_partition_mass_medoid_seed_kernel")
    start = next(i for i, node in enumerate(loop.body) if assignment_name(node) == "importance_sum")
    end = next(i for i, node in enumerate(loop.body) if assignment_name(node) == "used")

    def maximum(value, axis, return_indices, return_indices_tie_break_left):
        assert return_indices and return_indices_tie_break_left
        return np.max(value, axis=axis), np.argmax(value, axis=axis)

    scope = {"tl": SimpleNamespace(sum=np.sum, exp2=np.exp2, where=np.where, max=maximum),
             "log_probability": np.asarray(logp), "importance": np.asarray(importance),
             "valid_2d": np.ones(logp.shape[:2], dtype=bool) if valid is None else valid}
    exec(compile(ast.Module(body=loop.body[start:end], type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope["first_index"], scope["medoid_cost"]


def log_softmax2(logits):
    shifted = np.asarray(logits, dtype=np.float64) - np.max(logits, axis=-1, keepdims=True)
    return shifted - np.log2(np.exp2(shifted).sum(axis=-1, keepdims=True))


def original_bank():
    """Run the actual Python bank builder with list-valued tensor construction."""
    source = ROOT / "attention/adaptive_seed_pytorch.py"
    node = function(source, "multiscale_candidate_table")
    scope = {"torch": SimpleNamespace(device=lambda value: value, int32="int32",
                                      tensor=lambda value, **kwargs: value), "_MULTISCALE_TABLE_CACHE": {}}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    return scope["multiscale_candidate_table"](45, 16, "cpu", geometry_count=16)


def greedy_reference(logp, importance, first):
    """Small independent CPU partition-deficit oracle, returning positions."""
    selected = [int(first)]
    for _ in range(3):
        seed_logp = logp[selected]
        owner = seed_logp.argmax(axis=0)  # first owner wins ties
        represented = np.zeros(logp.shape[0])
        for slot in range(len(selected)):
            cell = owner == slot
            if not cell.any():
                continue
            conditional_logp = seed_logp[slot, cell]
            conditional_logp = log_softmax2(conditional_logp)
            weight = np.exp2(conditional_logp)
            entropy = -(weight * conditional_logp).sum()
            represented += np.exp2(logp[:, cell] @ weight + entropy)
        score = importance * np.maximum(1.0 - represented, 0.0)
        score[selected] = -1.0
        selected.append(int(score.argmax()))
    return selected


class MedoidMathTests(unittest.TestCase):
    def test_four_columns_float32_scores_match_independent_pairwise_oracle(self):
        rng = np.random.default_rng(20260904)
        logp = log_softmax2(rng.normal(size=(4, 16, 16)) * 3).astype(np.float32)
        importance = np.exp2(rng.normal(size=(4, 16)) * 2).astype(np.float32)
        first, costs = medoid(logp, importance)
        for column in range(4):
            lp = logp[column].astype(np.float64)
            pair = (np.exp2(lp[:, None, :]) * (lp[:, None, :] - lp[None, :, :])).sum(axis=2)
            weights = importance[column].astype(np.float64)
            expected = pair @ (weights / weights.sum())
            np.testing.assert_allclose(costs[column], expected, rtol=2e-6, atol=2e-6)
            self.assertEqual(first[column], expected.argmin())

    def test_actual_expression_matches_pairwise_reverse_kl_and_leaves_anchor(self):
        probability = np.repeat(np.array([[.01, .99], [.1, .9], [.3, .7], [.6, .4]]), 8, axis=1) / 8
        logp = np.log2(probability)
        importance = np.array([.7, .1, .1, .1])
        selected, cost = medoid(logp[None], importance[None])
        pair_kl = (probability[:, None, :] * (logp[:, None, :] - logp[None, :, :])).sum(axis=2)
        np.testing.assert_allclose(cost[0], pair_kl @ importance, rtol=1e-12, atol=1e-12)
        self.assertEqual(selected.tolist(), [0])  # deliberately not anchor position 3
        self.assertEqual(int((importance @ pair_kl).argmin()), 1)  # wrong KL direction differs
        self.assertLess(cost[0, 0], cost[0, 3])
        np.testing.assert_allclose(cost[0] * np.log(2),
                                   [.12367121617023244, .16783384365864804,
                                    .5781504406063502, 1.5601499640687087], atol=2e-14)

    def test_zero_logits_ties_and_four_distinct_real_rows(self):
        logp = np.full((1, 16, 16), -4.0, dtype=np.float32)
        first, cost = medoid(logp, np.ones((1, 16), dtype=np.float32))
        np.testing.assert_array_equal(cost, 0.0)
        self.assertEqual(first.tolist(), [0])
        selected = greedy_reference(logp[0], np.ones(16), first[0])
        self.assertEqual(selected, [0, 1, 2, 3])
        bank, anchors = original_bank()
        for key_row, rows in enumerate(bank):
            self.assertEqual(len(rows), 16)
            self.assertEqual(len(set(rows)), 16)
            self.assertEqual(rows, sorted(rows))
            self.assertEqual(rows[anchors[key_row]], key_row)
            actual_rows = [rows[index] for index in selected]
            self.assertEqual(len(set(actual_rows)), 4)
            self.assertTrue(all(0 <= row < 45 for row in actual_rows))
        self.assertNotEqual(bank[44][selected[0]], 44)  # no implicit mandatory matched row

    def test_single_weight_masked_candidates_and_fully_padded_column(self):
        rng = np.random.default_rng(71)
        logp = log_softmax2(rng.normal(size=(2, 16, 16)))
        importance = np.zeros((2, 16))
        importance[0, 1] = 1
        valid = np.zeros((2, 16), dtype=bool)
        valid[0, :4] = True
        logp[0, 4:] = log_softmax2(rng.normal(size=(12, 16)) * 1000)
        first, cost = medoid(logp, importance, valid)
        self.assertEqual(first[0], 1)
        self.assertTrue(np.isfinite(cost).all())
        self.assertAlmostEqual(cost[0, 1], 0.0, places=13)
        # Column 1 has no valid candidate/output store; its safe reduction is finite.
        self.assertEqual(first[1], 0)

    def test_underflow_and_common_weight_scaling(self):
        logits = np.full((1, 16, 16), -2000.0)
        for candidate in range(16):
            logits[0, candidate, candidate] = 0.0
        logp = log_softmax2(logits)
        importance = np.arange(1.0, 17.0)[None]
        first, cost = medoid(logp, importance)
        scaled_first, scaled_cost = medoid(logp, importance * 27)
        self.assertTrue(np.isfinite(cost).all())
        self.assertEqual(first.tolist(), [15])
        np.testing.assert_array_equal(first, scaled_first)
        np.testing.assert_allclose(cost, scaled_cost, rtol=1e-14)
        for initial in first:
            selected = greedy_reference(logp[0], importance[0], initial)
            self.assertEqual(len(set(selected)), 4)


class MedoidSourceContractTests(unittest.TestCase):
    def test_original_qk_importance_three_steps_and_output_are_unchanged(self):
        base = key_loop(BASE, "_partition_mass_seed_kernel").body
        new = key_loop(SOURCE, "_partition_mass_medoid_seed_kernel").body
        def prefix(nodes, name):
            return nodes[:next(i for i, node in enumerate(nodes) if assignment_name(node) == name)]
        def suffix(nodes):
            return nodes[next(i for i, node in enumerate(nodes)
                              if isinstance(node, ast.For) and node.target.id == "step"):]
        dump = lambda nodes: ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)
        self.assertEqual(dump(prefix(base, "used")), dump(prefix(new, "importance_sum")))
        self.assertEqual(dump(suffix(base)), dump(suffix(new)))

    def test_one_normalizer_original_bank_and_fixed_launch_contract(self):
        wrapper = function(SOURCE, "partition_mass_medoid_seed_rows_triton")
        defaults = dict(zip((arg.arg for arg in wrapper.args.kwonlyargs), wrapper.args.kw_defaults))
        expected = {"candidate_count": 16, "candidate_geometry_count": 16, "block_j": 4,
                    "probe_count": 16, "global_probe_count": 256, "gs_num_warps": 4,
                    "num_stages": 1, "reuse_owner_logp": True}
        for name, value in expected.items():
            self.assertEqual(ast.literal_eval(defaults[name]), value)
        calls = [node for node in ast.walk(wrapper) if isinstance(node, ast.Call)]
        self.assertEqual(sum(isinstance(node.func, ast.Name) and node.func.id == "global_probe_log_norm_triton"
                             for node in calls), 1)
        self.assertEqual(sum(isinstance(node.func, ast.Name) and node.func.id == "multiscale_candidate_table"
                             for node in calls), 1)
        kernel_launch = next(node for node in calls if isinstance(node.func, ast.Subscript))
        launch = {kw.arg: kw.value for kw in kernel_launch.keywords}
        self.assertEqual(ast.literal_eval(launch["num_warps"]), 4)
        self.assertIs(ast.literal_eval(launch["REUSE_OWNER_LOGP"]), True)
        self.assertEqual(ast.literal_eval(launch["num_stages"]), 1)
        original_wrapper = function(BASE, "partition_mass_seed_rows_triton")
        allocation = lambda node: next(item for item in node.body if assignment_name(item) == "output")
        self.assertEqual(ast.dump(allocation(wrapper)), ast.dump(allocation(original_wrapper)))
        # Execute the actual leading scope guards, without importing Torch/Triton.
        guards = wrapper.body[1:4]
        self.assertTrue(all(isinstance(node, ast.If) for node in guards))
        code = compile(ast.Module(body=guards, type_ignores=[]), str(SOURCE), "exec")
        scope = dict(expected, num_warps=None)
        exec(code, scope.copy())
        for key, value in {"candidate_count": 32, "candidate_geometry_count": 45,
                           "probe_count": 32, "global_probe_count": 128, "block_j": 2,
                           "gs_num_warps": 8, "num_warps": 8, "num_stages": 2,
                           "reuse_owner_logp": False}.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                exec(code, dict(scope, **{key: value}))

    def test_frozen_provenance_no_production_dispatch_or_anchor_reinsertion(self):
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn("93dbfd0b5c25171fef696af8f7d9afdae7e95064212711be17999997dac95775", text)
        kernel = function(SOURCE, "_partition_mass_medoid_seed_kernel")
        names = {node.id for node in ast.walk(kernel) if isinstance(node, ast.Name)}
        self.assertNotIn("anchor_ptr", names)
        self.assertNotIn("anchor_index", names)
        dispatcher = (ROOT / "attention/token_merging.py").read_text(encoding="utf-8")
        self.assertNotIn("partition_mass_medoid_seed_rows_triton", dispatcher)


@unittest.skipIf(torch is None, "Torch CPU required for the diagnostic reference")
class IndependentTorchReferenceTests(unittest.TestCase):
    def setUp(self):
        node = function(SOURCE, "partition_mass_medoid_seeds_reference")
        scope = {"torch": torch}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), scope)
        self.reference = scope[node.name]

    def test_full_four_seeds_match_independent_numpy_greedy(self):
        rng = np.random.default_rng(51)
        natural_logits = rng.normal(size=(4, 16, 16)) * 2
        importance = np.exp(rng.normal(size=(4, 16)))
        logp = log_softmax2(natural_logits / np.log(2))
        first, _ = medoid(logp, importance)
        expected = [greedy_reference(logp[i], importance[i], first[i]) for i in range(4)]
        result = self.reference(torch.tensor(natural_logits), torch.tensor(importance))
        self.assertEqual(result.tolist(), expected)
        checked, diagnostics = self.reference(torch.tensor(natural_logits), torch.tensor(importance),
                                               return_diagnostics=True)
        self.assertEqual(checked.tolist(), expected)
        self.assertEqual(diagnostics["first_medoid_index"].tolist(), first.tolist())
        self.assertEqual(tuple(diagnostics["first_medoid_cost"].shape), (4, 16))

    def test_uniform_ties_and_bad_inputs(self):
        raw = torch.zeros(2, 16, 16, dtype=torch.float64)
        weight = torch.ones(2, 16, dtype=torch.float64)
        self.assertEqual(self.reference(raw, weight).tolist(), [[0, 1, 2, 3]] * 2)
        for bad in (torch.zeros_like(weight), -weight, torch.full_like(weight, float("nan"))):
            with self.assertRaises(ValueError):
                self.reference(raw, bad)


if __name__ == "__main__":
    unittest.main()
