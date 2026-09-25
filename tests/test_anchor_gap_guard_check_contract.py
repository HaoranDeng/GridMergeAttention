"""CPU contracts for the guarded synthetic native test; no GPU evidence."""
import ast
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import unittest

from analysis import check_barycentric_native as parent
from analysis.check_anchor_gap_guard_case import (
    GAP_ATOL, GAP_RTOL, MISMATCH_DIAGNOSTIC_LIMIT, append_decision_diagnostic, validate_guard_record,
)


def valid_record(frames=2):
    banks = frames * 7200
    return dict(anchor_mismatches=0, nonunique_banks=0, invalid_candidate_banks=0,
        prefix_mismatched_banks=0, decision_mismatched_banks=0, self_accepted_banks=0,
        row_change_decision_mismatched_banks=0,
        score_audit_mismatched_banks=0, score_audit_on_off_equal=True,
        score_comparison_passed=True, score_atol=GAP_ATOL, score_rtol=GAP_RTOL,
        score_values_checked=banks * 96, max_score_tolerance_ratio=0.01,
        frame_banks=[7200] * frames, frame_mismatched_banks=[0] * frames,
        q_dtype="torch.bfloat16", k_dtype="torch.bfloat16", k_shape=[1, frames, 45, 80, 2, 128],
        singleton_already_probed_banks=banks // 5, singleton_added_banks=banks * 4 // 5,
        accepted_banks=100, incumbent_implementation="fisher_anchor_seed_rows_triton")


class AnchorGuardNativeCheckTests(unittest.TestCase):
    def test_cases_and_old_defaults(self):
        self.assertEqual(parent.family_cases("anchor_gap_guard"), ((17, 1, .125), (53, 2, 128**-.5)))
        specs = parent.expected_records("anchor_gap_guard")
        self.assertEqual([s[3] for s in specs], ["anchor_gap_guard"] * 2)
        self.assertEqual([s[-1] for s in specs], [7200, 14400])
        self.assertEqual(len(parent.expected_records("barycentric")), 6)
        self.assertEqual(len(parent.expected_records("expected_fisher")), 2)

    def test_each_structure_and_decision_failure_is_rejected(self):
        good = valid_record()
        validate_guard_record(good, 14400, 2)
        invalid = dict(anchor_mismatches=1, nonunique_banks=1, invalid_candidate_banks=1,
            prefix_mismatched_banks=1, decision_mismatched_banks=1, self_accepted_banks=1,
            row_change_decision_mismatched_banks=1,
            score_audit_mismatched_banks=1, score_audit_on_off_equal=False,
            score_comparison_passed=False, score_atol=1e-5, score_rtol=1e-5,
            score_values_checked=1, max_score_tolerance_ratio=1.01,
            frame_banks=[14400], frame_mismatched_banks=[0, 1], q_dtype="torch.float32",
            k_dtype="torch.float16", k_shape=[1, 1, 45, 80, 2, 128],
            singleton_already_probed_banks=0, singleton_added_banks=0,
            accepted_banks=0, incumbent_implementation="geometric_fisher")
        for name, value in invalid.items():
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                validate_guard_record(good | {name: value}, 14400, 2)
        for ratio in (float("nan"), float("inf"), -1):
            with self.assertRaises(RuntimeError):
                validate_guard_record(good | {"max_score_tolerance_ratio": ratio}, 14400, 2)

    def test_parent_also_applies_guard_validation(self):
        hashes = {"test": "a" * 64}
        result = dict(complete=True, family="anchor_gap_guard", source_sha256=hashes,
            source_sha256_after=hashes, shape=list(parent.Q_SHAPE), records=[])
        for seed, frames, scale, method, banks in parent.expected_records("anchor_gap_guard"):
            record = valid_record(frames)
            record.update(input_seed=seed, key_frames=frames, scale=scale, method=method, banks=banks,
                          group_agreement=1.0, mismatched_banks=0, native_sha256="b" * 64)
            result["records"].append(record)
        parent.validate_completed_result(result, "anchor_gap_guard", hashes)
        bad = copy.deepcopy(result)
        bad["records"][1]["decision_mismatched_banks"] = 1
        with self.assertRaises(RuntimeError):
            parent.validate_completed_result(bad, "anchor_gap_guard", hashes)

    def test_actual_worker_uses_distinct_sq_supports_and_actual_incumbent(self):
        path = Path(__file__).resolve().parents[1] / "analysis/check_anchor_gap_guard_case.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "check_guard_case")
        source = ast.unparse(function)
        for fragment in ("fisher_anchor_seed_rows_triton(q, k, scale, **kwargs)",
                         "block_j=4", "gap_audit=gaps", "decision_audit=decisions",
                         "for already_probed in (True, False)", "79 / 15", "79 / 16",
                         "torch.bmm(qq, kk.transpose(1, 2)).double() * scale",
                         "anchor_gap_guard(logits, slots, singleton, measures=measures)",
                         "oracle['challenger']", "oracle['accepted']", "slots[:, -1]",
                         "native[:, :3] != incumbent[:, :3]",
                         "('prefix_gap', 'base_gap', 'trial_gap')"):
            self.assertIn(fragment, source)
        self.assertNotIn("torch.load", source)
        self.assertEqual(source.count("torch.randint(-4, 5"), 2)

    def test_fixed_scores_and_explicit_reference_source_are_frozen(self):
        self.assertEqual(GAP_ATOL, 2e-10)
        self.assertEqual(GAP_RTOL, 2e-11)
        self.assertIn("analysis/check_anchor_gap_guard_case.py", parent.SOURCES)
        self.assertIn("analysis/anchor_gap_guard_seed.py", parent.SOURCES)

    def test_failure_diagnostics_are_bounded_and_never_relax_the_decision_gate(self):
        self.assertEqual(MISMATCH_DIAGNOSTIC_LIMIT, 8)
        diagnostics = []
        output = io.StringIO()
        with redirect_stdout(output):
            for bank in range(20):
                appended = append_decision_diagnostic(diagnostics, {
                    "bank_id": bank, "frame": 0, "row": 0, "col": bank // 2, "head": bank % 2,
                    "actual_decision": [7, 0, 3], "expected_decision": [8, 0, 3],
                })
                self.assertEqual(appended, bank < 8)
        self.assertEqual(len(diagnostics), 8)
        self.assertEqual([item["bank_id"] for item in diagnostics], list(range(8)))
        self.assertEqual(len(output.getvalue().splitlines()), 8)
        self.assertIn("actual=[7, 0, 3] expected=[8, 0, 3]", output.getvalue())
        # Even when final seed rows could coincide, any decision mismatch is
        # still a failure. Diagnostics cannot grant an exception or new tol.
        bad = valid_record() | {"decision_mismatched_banks": 1,
                               "decision_mismatch_diagnostics": diagnostics}
        with self.assertRaises(RuntimeError):
            validate_guard_record(bad, 14400, 2)
        validate_guard_record(valid_record() | {"decision_mismatch_diagnostics": []}, 14400, 2)

    def test_diagnostic_payload_preserves_fp64_json_and_replay_inputs(self):
        measure = 79 / 15
        raw_gap = -1.1102230246251565e-16
        logits = [[(row - col) * (128 ** -.5) for col in range(16)] for row in range(32)]
        payload = {"bank_id": 7200, "frame": 1, "row": 0, "col": 0, "head": 0,
            "actual_decision": [7, 0, 3], "expected_decision": [8, 0, 3],
            "incumbent_slots": [0, 1, 2, 3], "actual_prefix_gap": [raw_gap] * 32,
            "oracle_prefix_gap": [raw_gap / 2] * 32,
            "actual_base_mean": raw_gap, "actual_trial_mean": raw_gap,
            "oracle_base_mean": raw_gap, "oracle_trial_mean": 0.0,
            "logits": logits, "measures": [1.0] + [measure] * 15, "singleton_index": 0}
        diagnostics = []
        with redirect_stdout(io.StringIO()):
            append_decision_diagnostic(diagnostics, payload)
        reloaded = json.loads(json.dumps(diagnostics, allow_nan=False))[0]
        self.assertEqual(reloaded, payload)
        self.assertEqual(reloaded["measures"][1].hex(), measure.hex())
        self.assertEqual(reloaded["actual_prefix_gap"][0].hex(), raw_gap.hex())
        self.assertEqual(len(reloaded["logits"]), 32)
        self.assertTrue(all(len(row) == 16 for row in reloaded["logits"]))
        # Independent index check: second key frame begins at bank7200.
        self.assertEqual(parent.decode_bank_ids(reloaded["bank_id"]), (1, 0, 0, 0))

    def test_actual_diagnostic_path_copies_only_bounded_original_fp64_arrays(self):
        path = Path(__file__).resolve().parents[1] / "analysis/check_anchor_gap_guard_case.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "check_guard_case")
        source = ast.unparse(function)
        for fragment in ("decision_errors += int(decision_disagreement.sum())",
                         "remaining = MISMATCH_DIAGNOSTIC_LIMIT - len(decision_diagnostics)",
                         "decision_disagreement.nonzero().flatten()[:remaining].cpu().tolist()",
                         "logits[position].cpu().tolist()", "measures[position].cpu().tolist()",
                         "slots[position].cpu().tolist()", "int(singleton[position])",
                         "actual_gap_vectors[0]", "oracle_gap_vectors[0]",
                         "float(oracle['base_mean'][position])", "float(oracle['trial_mean'][position])",
                         "'decision_mismatch_diagnostics': decision_diagnostics"):
            self.assertIn(fragment, source)
        diagnostic_block = next(node for node in ast.walk(function) if isinstance(node, ast.If)
                                and ast.unparse(node.test) == "remaining > 0")
        diagnostic_source = ast.unparse(diagnostic_block)
        self.assertNotIn(".float()", diagnostic_source)
        self.assertNotIn("float32", diagnostic_source)
        self.assertNotIn("clamp", diagnostic_source)
        self.assertNotIn("maximum", diagnostic_source)
        self.assertNotIn("GAP_ATOL =", diagnostic_source)


if __name__ == "__main__":
    unittest.main()
