"""CPU-only checker contracts; no CUDA imports, compilation or native claims."""
import ast
from contextlib import redirect_stderr
import copy
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis" / "check_barycentric_native.py"
SPEC = importlib.util.spec_from_file_location("expected_fisher_checker_contract", SOURCE)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def valid_result(family):
    hashes = {"synthetic_source.py": "a" * 64}
    result = {"family": family, "complete": True, "shape": list(checker.Q_SHAPE),
              "source_sha256": hashes.copy(), "source_sha256_after": hashes.copy(), "records": []}
    for seed, frames, scale, method, banks in checker.expected_records(family):
        record = {"input_seed": seed, "key_frames": frames, "scale": scale, "method": method,
                  "banks": banks, "group_agreement": 1.0, "mismatched_banks": 0,
                  "native_sha256": "b" * 64}
        if family == "expected_fisher":
            record.update(anchor_mismatches=0, nonunique_banks=0, invalid_candidate_banks=0,
                score_audit_mismatched_banks=0, score_audit_on_off_equal=True,
                score_atol=checker.SCORE_ATOL, score_rtol=checker.SCORE_RTOL,
                score_values_checked=banks * 96, score_comparison_passed=True,
                max_score_tolerance_ratio=0.01, frame_banks=[7200] * frames,
                frame_mismatched_banks=[0] * frames, q_dtype="torch.bfloat16", k_dtype="torch.bfloat16",
                k_shape=[1, frames, 45, 80, 2, 128])
        result["records"].append(record)
    return result, hashes


class FamilyAndLayoutTests(unittest.TestCase):
    def test_default_cli_preserves_barycentric_and_explicit_expected_family(self):
        for suffix, family in (([], "barycentric"), (["--family", "expected_fisher"], "expected_fisher")):
            argv = [str(SOURCE), "--worker", "--gpu", "4", "--output", "unused", *suffix]
            with patch.object(checker.sys, "argv", argv), patch.object(checker, "worker", side_effect=lambda args: args):
                args = checker.main()
            self.assertEqual(args.family, family)

    def test_unknown_family_fails_before_worker_or_guard(self):
        with self.assertRaises(ValueError):
            checker.family_cases("automatic")
        with self.assertRaises(ValueError):
            checker.expected_records("automatic")
        argv = [str(SOURCE), "--worker", "--gpu", "4", "--output", "unused", "--family", "automatic"]
        with patch.object(checker.sys, "argv", argv), patch.object(checker, "worker") as worker:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                checker.main()
            worker.assert_not_called()

    def test_default_six_records_remain_original_inputs_methods_and_scales(self):
        self.assertEqual(checker.family_cases("barycentric"), ((17, 1, .125), (53, 1, .125)))
        specs = checker.expected_records("barycentric")
        self.assertEqual(len(specs), 6)
        self.assertEqual([s[3] for s in specs], list(checker.BARYCENTRIC_METHODS) * 2)
        self.assertEqual([s[4] for s in specs], [7200] * 6)

    def test_expected_two_original_bf16_shapes_and_real_second_scale(self):
        self.assertEqual(checker.Q_SHAPE, (1, 1, 45, 80, 2, 128))
        self.assertEqual(checker.family_cases("expected_fisher"), ((17, 1, .125), (53, 2, 128**-.5)))
        specs = checker.expected_records("expected_fisher")
        self.assertEqual(len(specs), 2)
        self.assertEqual([s[-1] for s in specs], [7200, 14400])
        self.assertEqual(sum(s[-1] for s in specs), 21600)
        self.assertEqual([s[3] for s in specs], ["fisher_expected"] * 2)
        self.assertNotEqual(float(np.float32(specs[1][2])), specs[1][2])

    def test_bank_decode_round_trips_every_bank_and_frame_boundary(self):
        for frames in (1, 2):
            ids = np.arange(frames * 7200)
            frame, row, column, head = checker.decode_bank_ids(ids)
            np.testing.assert_array_equal((((frame * 45 + row) * 80 + column) * 2 + head), ids)
            expected = np.unravel_index(ids, (frames, 45, 80, 2))
            for actual, wanted in zip((frame, row, column, head), expected):
                np.testing.assert_array_equal(actual, wanted)
        self.assertEqual(checker.decode_bank_ids(7199), (0, 44, 79, 1))
        self.assertEqual(checker.decode_bank_ids(7200), (1, 0, 0, 0))
        self.assertEqual(checker.decode_bank_ids(14399), (1, 44, 79, 1))

    def test_actual_worker_uses_rf0_and_decoded_k_frame_original_bf16(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "check_expected_case")
        text = ast.unparse(function)
        self.assertIn("frame, row, col, head = decode_bank_ids(ids)", text)
        self.assertIn("q[0, 0, real, col[:, None], head[:, None]].float()", text)
        self.assertIn("k[0, frame[:, None], row[:, None]", text)
        self.assertIn("torch.bmm(qq, kk.transpose(1, 2)).double() * scale", text)
        self.assertIn("k_shape = (1, frames, 45, 80, 2, 128)", text)
        self.assertEqual(text.count("torch.randint(-4, 5"), 2)
        self.assertEqual(text.count(".to(torch.bfloat16)"), 2)
        for forbidden in ("repeat(", "repeat_interleave(", "expand(", "torch.load(", "float8", "mean_seed"):
            self.assertNotIn(forbidden, text)
        self.assertIn("block_j=1", text)
        self.assertIn("score_audit=score_audit", text)
        self.assertIn("score_audit = torch.full", text)
        self.assertIn("expected_fisher_indices(z, anchors[row])", text)
        self.assertIn("gram_scores_at_native_prefix(expected_fisher_gram(z, anchors[row]), actual_slots)", text)


class IndependentPrefixScoreTests(unittest.TestCase):
    def test_actual_gram_helper_against_stacked_features_at_non_greedy_prefixes(self):
        # Prefixes deliberately differ from argmax pivots. This detects checking
        # the oracle's own prefix instead of the ACTUAL native selected prefix.
        rng = np.random.default_rng(19503)
        for magnitude in (1.0, 1e-6):
            all_features = []
            for bank in range(4):
                z = rng.normal(size=(32, 16))
                p = np.exp(z - z.max(-1, keepdims=True))
                p /= p.sum(-1, keepdims=True)
                delta = (z - z[bank]) * magnitude
                delta -= delta[:, :1]
                feature = np.stack([np.concatenate([
                    np.sqrt(profile / 32) * (row - row @ profile) for profile in p
                ]) for row in delta])
                all_features.append(feature)
            features = np.array(all_features)
            gram = features @ features.transpose(0, 2, 1)
            untouched = gram.copy()
            slots = np.array([[bank, 7, 11, 23] for bank in range(4)])
            actual = np.stack(checker.gram_scores_at_native_prefix(gram, slots), axis=1)
            reference = []
            residual = features.copy()
            batch = np.arange(4)
            for step in range(3):
                reference.append(np.sum(residual * residual, axis=-1))
                if step < 2:
                    pivot = residual[batch, slots[:, step + 1]].copy()
                    pivot /= np.sqrt(np.sum(pivot * pivot, axis=-1, keepdims=True) + 1e-12)
                    residual -= (residual @ pivot[:, :, None]) * pivot[:, None, :]
            np.testing.assert_allclose(actual, np.stack(reference, axis=1), atol=2e-14, rtol=2e-12)
            np.testing.assert_array_equal(gram, untouched)

    def test_fixed_tolerance_is_not_a_cli_parameter(self):
        self.assertEqual(checker.SCORE_ATOL, 2e-10)
        self.assertEqual(checker.SCORE_RTOL, 2e-11)
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        flags = [node.args[0].value for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"]
        self.assertFalse(any("tol" in flag for flag in flags))


class FailClosedResultTests(unittest.TestCase):
    def test_valid_complete_metadata_both_families(self):
        for family in ("barycentric", "expected_fisher"):
            result, hashes = valid_result(family)
            checker.validate_completed_result(result, family, hashes)

    def test_partial_wrong_family_wrong_counts_or_changed_sources_rejected(self):
        result, hashes = valid_result("expected_fisher")
        variants = []
        for field, value in (("complete", False), ("family", "barycentric"), ("records", result["records"][:1]),
                             ("source_sha256", {}), ("source_sha256_after", {}), ("shape", [1, 2, 45, 80, 2, 128])):
            bad = copy.deepcopy(result); bad[field] = value; variants.append(bad)
        bad = copy.deepcopy(result); bad["records"].reverse(); variants.append(bad)
        for bad in variants:
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                checker.validate_completed_result(bad, "expected_fisher", hashes)

    def test_every_expected_row_structure_score_and_frame_gate_rejects_failure(self):
        result, hashes = valid_result("expected_fisher")
        invalid_fields = {
            "input_seed": 17, "key_frames": 1, "scale": .125, "method": "mean_fisher32", "banks": 7200,
            "group_agreement": .99999, "mismatched_banks": 1, "native_sha256": "",
            "anchor_mismatches": 1, "nonunique_banks": 1, "invalid_candidate_banks": 1,
            "score_audit_mismatched_banks": 1, "score_audit_on_off_equal": False,
            "score_atol": 1e-5, "score_rtol": 1e-5, "score_values_checked": 14400 * 95,
            "score_comparison_passed": False, "max_score_tolerance_ratio": 1.0000001,
            "frame_banks": [14400], "frame_mismatched_banks": [0, 1], "q_dtype": "torch.float32",
            "k_dtype": "torch.float8_e4m3fn", "k_shape": [1, 1, 45, 80, 2, 128],
        }
        for field, value in invalid_fields.items():
            bad = copy.deepcopy(result); bad["records"][1][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                checker.validate_completed_result(bad, "expected_fisher", hashes)
        for value in (float("nan"), float("inf"), -1):
            bad = copy.deepcopy(result); bad["records"][1]["max_score_tolerance_ratio"] = value
            with self.subTest(ratio=value), self.assertRaises(RuntimeError):
                checker.validate_completed_result(bad, "expected_fisher", hashes)

    def test_worker_refuses_without_own_monitor_parent_before_torch_import(self):
        with patch.dict(checker.os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "monitoring parent"):
            checker.worker(None)

    def test_source_closure_and_byte_identical_freeze_detect_both_changes(self):
        hashes = checker.source_hashes()
        for name in ("analysis/expected_fisher_seed.py", "attention/expected_fisher_seed_triton.py",
                     "attention/__init__.py", "attention/hard_route_pytorch.py",
                     "analysis/zero_anchor_video_adapter.py", "analysis/quality_first_mode_contract.py"):
            self.assertIn(name, hashes)
        # Temporary fixture only; no repository sources or run artifacts edited.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            small = {"analysis/check_barycentric_native.py": hashes["analysis/check_barycentric_native.py"]}
            with patch.object(checker, "source_hashes", return_value=small):
                checker.freeze_sources(output, small)
                checker.verify_source_hashes(output, small)
                copied = output / "sources" / "analysis" / "check_barycentric_native.py"
                self.assertEqual(copied.read_bytes(), SOURCE.read_bytes())
                with patch.object(checker, "source_hashes", return_value={}):
                    with self.assertRaisesRegex(RuntimeError, "live source"):
                        checker.verify_source_hashes(output, small)
                copied.write_bytes(b"fixture corruption")
                with self.assertRaisesRegex(RuntimeError, "frozen source"):
                    checker.verify_source_hashes(output, small)

    def test_parent_retains_double_clean_five_second_monitor_and_no_timing_claim(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        parent = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        text = ast.unparse(parent)
        self.assertEqual(text.count("guard.require_clean("), 2)
        for fragment in ("guard.child_environment(", "guard.gpu_processes(", "time.sleep(5)",
                         "guard.stop_own_child(process)", "validate_completed_result(result, args.family, hashes)",
                         "'mean_attention_ms': None", "'speedup_vs_sdpa': None"):
            self.assertIn(fragment, text)
        self.assertGreaterEqual(text.count("verify_source_hashes(args.output, hashes)"), 2)


if __name__ == "__main__":
    unittest.main()
