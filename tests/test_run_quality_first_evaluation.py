"""CPU-only evaluator provenance, protocol, import-order and guard tests."""

import builtins
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from analysis import run_quality_first_evaluation as evaluation
from analysis import evaluate_video_quality as quality
from utils.prompt_contract import build_prompt_manifest, load_nonempty_prompts


class EvaluationFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.reference = self.root / "reference"
        self.mass = self.root / "mass/videos"
        self.hf = self.root / "hf/videos"
        for name, value in (("REFERENCE", self.reference), ("MASS", self.mass), ("HELLINGER", self.hf)):
            context = patch.object(evaluation, name, value)
            context.start()
            self.addCleanup(context.stop)
        prompts_path = evaluation.ROOT / "prompts/MovieGenVideoBench_extended.txt"
        prompts = load_nonempty_prompts(prompts_path)
        manifest = build_prompt_manifest(prompts_path, prompts, selected_count=20, seed=0,
                                         num_samples=1, vbench_filenames=False, allow_noncanonical=False)
        for directory in (self.reference, self.mass, self.hf):
            directory.mkdir(parents=True)
            self.write(directory / "prompt_manifest.json", manifest)
            for index in range(20):
                (directory / f"{index:03d}-00.mp4").write_bytes(f"encoded fixture {index}".encode())
        source = self.mass.parent / "source"
        names = ["generate.py", "attention/token_merging.py", "attention/entropy_mass_seed_triton.py",
                 "analysis/run_quality_first_videos.py", "configs/test.json"]
        source_records = []
        for name in names:
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("CPU fixture source", encoding="utf-8")
            source_records.append({"path": name, **{k: v for k, v in evaluation.fingerprint(target).items() if k != "path"}})
        self.provenance = {
            "method": evaluation.METHODS[0], "declaration": evaluation.method_contract("mass_rf0"),
            "planned": {"router_frame": 0, "num_videos": 20, "num_samples": 1, "seed": 0,
                        "frames": 81, "denoise_steps": 4, "attention_calls_per_video": 160},
            "config": names[-1], "files": source_records,
        }
        self.write(self.mass.parent / "source_provenance.json", self.provenance)
        context = patch.object(evaluation, "MASS_PROVENANCE_SHA",
                               evaluation.fingerprint(self.mass.parent / "source_provenance.json")["sha256"])
        context.start()
        self.addCleanup(context.stop)
        self.dispatch = {"complete": True, "error": None, "videos_observed": 20,
            "self_attention_calls": 3200, "core_calls": 3200, "selector_calls": 3200,
            "merge_route_head_chunk_calls": 16000, "per_layer_calls": [80] * 40,
            "declaration": evaluation.method_contract("mass_rf0")}
        self.write(self.mass.parent / "dispatch_audit.json", self.dispatch)
        self.runtime = {"checkpoint_weights": "raw_state_dict", "records": [
            {"prompt_index": i, "attention_calls": 160, "full_attention_calls": 160} for i in range(20)]}
        self.write(self.mass.parent / "runtime.json", self.runtime)
        self.write(self.mass.parent / "run_manifest.json", {"method": evaluation.METHODS[0]})
        (self.mass.parent / "generation.exitcode").write_text("0\n")
        (self.mass.parent / "purity.log").touch()
        self.completion = {"valid": True, "timing_valid": True, "worker_exit_code": 0,
            "method": evaluation.METHODS[0], "foreign": [], "failure": None, "monitor_polls": 10,
            "verified_artifacts": {}}
        self.refresh_completion()
        historical = {"root": str(self.hf.parent), "recorded_git_commit": "audited-fixture",
                      "policy": {"router_frame": 10, "probe_policy": "Uniform16, not historical A8U8",
                                 "pure_all_layer_config": True}}
        for name in ("config", "quality", "runtime", "generation_log"):
            target = self.hf.parent / (name + ".json")
            self.write(target, {"read_only_fixture": name})
            historical[name] = evaluation.fingerprint(target)
        historical.update(self.inventory_section(self.hf))
        self.audit = {"valid": True, "references": {"canonical_20260826": {
            "root": str(self.reference), **self.inventory_section(self.reference)}}, "hellinger_rf10": historical}
        self.audit_path = self.root / "reference_audit.json"
        self.write(self.audit_path, self.audit)
        context = patch.object(evaluation, "AUDIT_SHA", evaluation.fingerprint(self.audit_path)["sha256"])
        context.start()
        self.addCleanup(context.stop)
        self.args = SimpleNamespace(gpu=0, reference_dir=self.reference, mass_dir=self.mass,
                                    hellinger_dir=self.hf, reference_audit=self.audit_path,
                                    output=self.root / "evaluation")

    def write(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def inventory_section(self, path):
        return {"manifest": evaluation.fingerprint(path / "prompt_manifest.json"),
                "files": [{"name": p.name, **{k: v for k, v in evaluation.fingerprint(p).items() if k != "path"}}
                          for p in sorted(path.glob("*.mp4"))]}

    def refresh_completion(self):
        for name, key in (("dispatch_audit.json", "dispatch_audit_sha256"), ("runtime.json", "runtime_sha256"),
                          ("generation.exitcode", "generation_exitcode_sha256"), ("source_provenance.json", "source_provenance_sha256")):
            self.completion["verified_artifacts"][key] = evaluation.fingerprint(self.mass.parent / name)["sha256"]
        self.write(self.mass.parent / "completion.json", self.completion)

    def test_all_three_manifests_and_pinned_encoded_sources_pass_before_metric_imports(self):
        original = builtins.__import__
        def guard(name, *args, **kwargs):
            if name in ("torch", "lpips"):
                raise AssertionError("GPU/model import occurred during preflight")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", guard):
            receipt, comparisons = evaluation.preflight(self.args, self.audit_path)
        self.assertTrue(receipt["all_three_prompt_manifests_verified"])
        self.assertTrue(receipt["mass_generation"]["valid"])
        self.assertEqual([name for name, _, _ in comparisons], list(evaluation.METHODS))
        self.assertTrue(all(len(pairs) == 20 for _, pairs, _ in comparisons))
        self.assertFalse(receipt["historical_identity"]["historical_rf10_timing_established"])

    def test_corrupt_mass_prompt_rejected_before_lpips_import(self):
        manifest = evaluation.read_json(self.mass / "prompt_manifest.json")
        manifest["prompts"][19]["sha256"] = "0" * 64
        self.write(self.mass / "prompt_manifest.json", manifest)
        original = builtins.__import__
        def guard(name, *args, **kwargs):
            if name in ("torch", "lpips"):
                self.fail("metric model imported before all manifests were checked")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", guard), self.assertRaisesRegex(ValueError, "SHA256"):
            evaluation.preflight(self.args, self.audit_path)

    def test_failed_nonzero_or_contended_generation_parent_rejected(self):
        for change in ({"valid": False}, {"timing_valid": False}, {"worker_exit_code": 1},
                       {"worker_exit_code": False}, {"foreign": [{"pid": 1}]},
                       {"failure": "interrupted"}, {"monitor_polls": 0}, {"method": "mass_rf10"}):
            with self.subTest(change=change):
                self.write(self.mass.parent / "completion.json", self.completion | change)
                with self.assertRaises(ValueError):
                    evaluation.validate_mass_generation(self.mass)

    def test_mass_source_provenance_and_source_bytes_are_pinned(self):
        changed = self.provenance | {"method": "another_run"}
        self.write(self.mass.parent / "source_provenance.json", changed)
        with self.assertRaisesRegex(ValueError, "source identity"):
            evaluation.validate_mass_generation(self.mass)
        self.write(self.mass.parent / "source_provenance.json", self.provenance)
        (self.mass.parent / "source/attention/token_merging.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "receipt mismatch"):
            evaluation.validate_mass_generation(self.mass)

    def test_dispatch_deviations_rejected_even_with_updated_parent_digest(self):
        for change in ({"complete": False}, {"error": "bad"}, {"selector_calls": 3199},
                       {"merge_route_head_chunk_calls": 15999}, {"per_layer_calls": [79] + [80] * 39}):
            with self.subTest(change=change):
                self.write(self.mass.parent / "dispatch_audit.json", self.dispatch | change)
                self.refresh_completion()
                with self.assertRaisesRegex(ValueError, "dispatch"):
                    evaluation.validate_mass_generation(self.mass)

    def test_generation_exit_and_runtime_source_receipts_must_agree(self):
        (self.mass.parent / "generation.exitcode").write_text("1\n")
        with self.assertRaises(ValueError):
            evaluation.validate_mass_generation(self.mass)
        (self.mass.parent / "generation.exitcode").write_text("0\n")
        bad_runtime = copy.deepcopy(self.runtime)
        bad_runtime["records"][19]["full_attention_calls"] = 159
        self.write(self.mass.parent / "runtime.json", bad_runtime)
        self.refresh_completion()
        with self.assertRaisesRegex(ValueError, "omitted"):
            evaluation.validate_mass_generation(self.mass)

    def test_no_directory_substitution_or_reencoded_historical_video(self):
        with self.assertRaisesRegex(ValueError, "exact pinned"):
            evaluation.pinned_directories(SimpleNamespace(**(vars(self.args) | {"reference_dir": self.hf})))
        (self.hf / "019-00.mp4").write_bytes(b"reencoded")
        with self.assertRaisesRegex(ValueError, "encoded video changed"):
            evaluation.preflight(self.args, self.audit_path)

    def test_missing_extra_and_empty_videos_fail(self):
        extra = self.mass / "020-00.mp4"
        extra.write_bytes(b"extra")
        with self.assertRaises(ValueError):
            evaluation.verify_inventory(self.mass)
        extra.unlink()
        (self.mass / "019-00.mp4").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "empty"):
            evaluation.verify_inventory(self.mass)

    def test_snapshot_preserves_exact_evaluator_and_protocol_then_detects_change(self):
        evaluation.prepare(self.args)
        self.assertEqual(evaluation.verify_snapshot(self.args.output)["metric_source_shas"], evaluation.METRIC_SOURCE_SHAS)
        for name, expected in evaluation.METRIC_SOURCE_SHAS.items():
            self.assertEqual(evaluation.fingerprint(self.args.output / "source" / name)["sha256"], expected)
        (self.args.output / "source/analysis/evaluate_video_quality.py").write_text("changed metric")
        with self.assertRaisesRegex(ValueError, "receipt mismatch"):
            evaluation.verify_snapshot(self.args.output)

    def test_busy_parent_does_not_prepare_or_launch(self):
        with patch.object(evaluation.guard, "require_clean", side_effect=ValueError("busy")), \
             patch.object(evaluation, "prepare") as prepare, patch.object(evaluation.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(ValueError, "busy"):
                evaluation.launch(self.args)
            prepare.assert_not_called()
            spawn.assert_not_called()

    def test_contamination_stops_only_its_worker_and_records_failed_completion(self):
        class Child:
            pid = 999
            returncode = None
            terminated = False
            def poll(self):
                return self.returncode
            def terminate(self):
                self.terminated = True
                self.returncode = -15
            def wait(self, timeout):
                return self.returncode
        child = Child()
        clean = {"gpu": {"uuid": "GPU-fixture"}, "compute_processes": []}
        with patch.object(evaluation.guard, "require_clean", return_value=clean), \
             patch.object(evaluation.guard, "gpu_processes", return_value=[{"pid": 777, "process_name": "foreign"}]), \
             patch.object(evaluation.subprocess, "Popen", return_value=child):
            with self.assertRaisesRegex(ValueError, "foreign GPU"):
                evaluation.launch(self.args)
        self.assertTrue(child.terminated)
        completion = evaluation.read_json(self.args.output / "completion.json")
        self.assertFalse(completion["valid"])
        self.assertEqual(completion["worker_exit_code"], -15)
        self.assertEqual(completion["foreign"][0]["pid"], 777)
        self.assertEqual((self.args.output / "worker.exitcode").read_text().strip(), "-15")

    def test_resume_or_wrong_worker_path_never_overwrites_existing_results(self):
        self.args.output.mkdir()
        path = self.args.output / "results.json"
        self.write(path, {"preserve": "existing evidence"})
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "frozen snapshot"):
            evaluation.worker(self.args)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse((self.args.output / "evaluation.exitcode").exists())
        startup = {"parent_pid": os.getppid(), "immediately_before_spawn": {"gpu": {"uuid": "GPU-fixture"}}}
        self.write(self.args.output / "startup.json", startup)
        with patch.object(evaluation, "__file__", str(self.args.output / "source" / evaluation.SOURCE)), \
             patch.dict(os.environ, {"GCA_QUALITY_EVALUATION_PARENT_PID": str(os.getppid()), "CUDA_VISIBLE_DEVICES": "GPU-fixture"}):
            with self.assertRaisesRegex(ValueError, "refuse worker resume"):
                evaluation.worker(self.args)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse((self.args.output / "evaluation.exitcode").exists())


class QualityProtocolTests(unittest.TestCase):
    def test_decoder_checks_full_resolution_without_transforming_objects(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        original = lambda path: iter([frame])
        module = SimpleNamespace(frames=original)
        with evaluation.full_resolution_decoder(module):
            self.assertIs(next(module.frames("video.mp4")), frame)
        self.assertIs(module.frames, original)
        for wrong in (np.zeros((360, 640, 3), dtype=np.uint8), np.zeros((720, 1280, 3), dtype=np.float32)):
            module.frames = lambda path: iter([wrong])
            with evaluation.full_resolution_decoder(module), self.assertRaises(ValueError):
                next(module.frames("video.mp4"))

    def test_video_results_require_exact81_finite_metrics_and_filename(self):
        valid = dict(video="019-00.mp4", frames=81, psnr=12.1, ssim=.4, lpips=.3)
        evaluation.validate_video_metric(valid, 19)
        for change in ({"frames": 80}, {"frames": 82}, {"video": "018-00.mp4"},
                       {"psnr": float("inf")}, {"lpips": float("nan")}):
            with self.assertRaises(ValueError):
                evaluation.validate_video_metric(valid | change, 19)

    def test_terminal_recomputes_summaries_and_does_not_borrow_timing(self):
        methods = {}
        for name in evaluation.METHODS:
            videos = [dict(video=f"{i:03d}-00.mp4", frames=81, psnr=12.1, ssim=.4, lpips=.3) for i in range(20)]
            methods[name] = dict(videos=videos, summary=quality.aggregate(videos), mean_attention_ms=None, speedup_vs_sdpa=None)
        result = dict(valid=True, complete=True, protocol=evaluation.PROTOCOL, methods=methods,
            mean_attention_ms=None, speedup_vs_sdpa=None, historical_rf10_timing_established=False,
            inputs_reverified_at_end=True, sources_reverified_at_end=True, psnr_above_12=True)
        self.assertTrue(evaluation.result_complete(result))
        for kind in ("summary", "coverage", "timing", "source", "target"):
            changed = copy.deepcopy(result)
            if kind == "summary":
                changed["methods"][evaluation.METHODS[0]]["summary"]["psnr"] = 13
            elif kind == "coverage":
                changed["methods"][evaluation.METHODS[1]]["videos"].pop()
            elif kind == "timing":
                changed["methods"][evaluation.METHODS[1]]["mean_attention_ms"] = 142
            elif kind == "source":
                changed["sources_reverified_at_end"] = False
            else:
                changed["psnr_above_12"] = False
            self.assertFalse(evaluation.result_complete(changed))


class HellingerCandidateTests(unittest.TestCase):
    """Exercise g without replacing, relabeling, or relaxing the Mass e tests."""
    write = EvaluationFixture.write
    inventory_section = EvaluationFixture.inventory_section
    refresh_completion = EvaluationFixture.refresh_completion

    def setUp(self):
        EvaluationFixture.setUp(self)
        self.g = self.root / "hellinger_rf0/videos"
        shutil.copytree(self.mass.parent, self.g.parent)
        context = patch.object(evaluation, "HELLINGER_RF0", self.g)
        context.start()
        self.addCleanup(context.stop)
        self.g_provenance = copy.deepcopy(self.provenance)
        self.g_provenance.update(method="hellinger_rf0_bf16score",
            declaration=evaluation.method_contract("hellinger_rf0"),
            remote_base_commit=evaluation.HELLINGER_RF0_BASE_COMMIT)
        selector = self.g.parent / "source/attention/adaptive_seed_triton.py"
        selector.write_text("CPU-only Hellinger fixture", encoding="utf-8")
        self.g_provenance["files"].append({"path": "attention/adaptive_seed_triton.py",
            **{k: v for k, v in evaluation.fingerprint(selector).items() if k != "path"}})
        self.write(self.g.parent / "source_provenance.json", self.g_provenance)
        context = patch.object(evaluation, "HELLINGER_RF0_PROVENANCE_SHA",
            evaluation.fingerprint(self.g.parent / "source_provenance.json")["sha256"])
        context.start()
        self.addCleanup(context.stop)
        self.g_dispatch = self.dispatch | {"declaration": evaluation.method_contract("hellinger_rf0")}
        self.write(self.g.parent / "dispatch_audit.json", self.g_dispatch)
        self.write(self.g.parent / "run_manifest.json", {"method": "hellinger_rf0_bf16score"})
        self.g_completion = copy.deepcopy(self.completion)
        self.g_completion["method"] = "hellinger_rf0_bf16score"
        self.refresh_g_completion()
        self.g_args = SimpleNamespace(**(vars(self.args) | {"candidate_method": "hellinger_rf0",
            "candidate_dir": self.g, "mass_dir": None, "output": self.root / "evaluation_g"}))

    def refresh_g_completion(self):
        for name, key in (("dispatch_audit.json", "dispatch_audit_sha256"), ("runtime.json", "runtime_sha256"),
                          ("generation.exitcode", "generation_exitcode_sha256"), ("source_provenance.json", "source_provenance_sha256")):
            self.g_completion["verified_artifacts"][key] = evaluation.fingerprint(self.g.parent / name)["sha256"]
        self.write(self.g.parent / "completion.json", self.g_completion)

    def test_g_admission_labels_and_historical_control_equal_default_f(self):
        original = builtins.__import__
        def guard(name, *args, **kwargs):
            if name in ("torch", "lpips"):
                self.fail("metric/model import during CPU preflight")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", guard):
            g, comparisons = evaluation.preflight(self.g_args, self.audit_path)
            f, _ = evaluation.preflight(self.args, self.audit_path)
        self.assertTrue(g["hellinger_generation"]["valid"])
        self.assertNotIn("mass_generation", g)
        expected = ["hellinger_rf0_bf16score", evaluation.METHODS[1]]
        self.assertEqual([name for name, _, _ in comparisons], expected)
        self.assertEqual(set(g["inputs"]), {"reference", *expected})
        self.assertEqual(g["inputs"]["reference"], f["inputs"]["reference"])
        self.assertEqual(g["inputs"][evaluation.METHODS[1]], f["inputs"][evaluation.METHODS[1]])
        self.assertEqual(g["historical_source_receipts"], f["historical_source_receipts"])
        self.assertEqual(evaluation.protocol_for(), evaluation.PROTOCOL)

    def test_g_rejects_mass_path_legacy_label_and_implicit_selection(self):
        for args in (
            vars(self.g_args) | {"candidate_dir": self.mass},
            vars(self.g_args) | {"candidate_method": "mass_rf0"},
            vars(self.g_args) | {"candidate_dir": None, "mass_dir": self.g},
            vars(self.g_args) | {"mass_dir": self.mass},
            vars(self.g_args) | {"candidate_method": "jeffreys_rf0"},
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                evaluation.pinned_directories(SimpleNamespace(**args))
        with self.assertRaisesRegex(ValueError, "source identity"):
            evaluation.validate_mass_generation(self.g)

    def test_g_incomplete_generation_cannot_prepare_and_imports_no_model(self):
        (self.g.parent / "completion.json").unlink()
        with patch("builtins.__import__", side_effect=AssertionError("unexpected import")), \
             self.assertRaises(FileNotFoundError):
            evaluation.prepare(self.g_args)
        self.assertFalse(self.g_args.output.exists())

    def test_g_completion_and_dispatch_must_match_hellinger_identity(self):
        for change in ({"valid": False}, {"timing_valid": False}, {"worker_exit_code": 1},
                       {"method": evaluation.METHODS[0]}, {"foreign": [{"pid": 123}]}):
            with self.subTest(change=change):
                self.write(self.g.parent / "completion.json", self.g_completion | change)
                with self.assertRaises(ValueError):
                    evaluation.validate_candidate_generation(self.g, "hellinger_rf0")
        for change in ({"declaration": evaluation.method_contract("mass_rf0")},
                       {"per_layer_calls": [79] + [80] * 39}, {"selector_calls": 3199}):
            with self.subTest(change=change):
                self.write(self.g.parent / "dispatch_audit.json", self.g_dispatch | change)
                self.refresh_g_completion()
                with self.assertRaisesRegex(ValueError, "dispatch"):
                    evaluation.validate_candidate_generation(self.g, "hellinger_rf0")

    def test_g_source_digest_identity_base_and_adaptive_selector_are_required(self):
        wrong = self.g_provenance | {"method": evaluation.METHODS[0]}
        self.write(self.g.parent / "source_provenance.json", wrong)
        with self.assertRaisesRegex(ValueError, "source identity"):
            evaluation.validate_candidate_generation(self.g, "hellinger_rf0")
        # Repinned fixtures isolate semantic checks after the real fixed-SHA gate.
        for field, value, message in (
            ("method", evaluation.METHODS[0], "not the BF16"),
            ("declaration", evaluation.method_contract("mass_rf0"), "not the BF16"),
            ("remote_base_commit", "other-commit", "base commit"),
            ("files", [v for v in self.g_provenance["files"] if v["path"] != "attention/adaptive_seed_triton.py"], "incomplete"),
        ):
            with self.subTest(field=field):
                self.write(self.g.parent / "source_provenance.json", self.g_provenance | {field: value})
                with patch.object(evaluation, "HELLINGER_RF0_PROVENANCE_SHA",
                        evaluation.fingerprint(self.g.parent / "source_provenance.json")["sha256"]), \
                     self.assertRaisesRegex(ValueError, message):
                    evaluation.validate_candidate_generation(self.g, "hellinger_rf0")

    def test_g_manifest_and_historical_encoded_hash_gates_remain_strict(self):
        manifest = evaluation.read_json(self.g / "prompt_manifest.json")
        manifest["prompts"][19]["sha256"] = "0" * 64
        self.write(self.g / "prompt_manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            evaluation.preflight(self.g_args, self.audit_path)
        self.write(self.g / "prompt_manifest.json", evaluation.read_json(self.mass / "prompt_manifest.json"))
        (self.hf / "019-00.mp4").write_bytes(b"changed historical control")
        with self.assertRaisesRegex(ValueError, "encoded video changed"):
            evaluation.preflight(self.g_args, self.audit_path)

    def test_g_snapshot_target_cannot_be_verified_as_mass(self):
        evaluation.prepare(self.g_args)
        evaluation.verify_snapshot(self.g_args.output, "hellinger_rf0")
        protocol = evaluation.read_json(self.g_args.output / "protocol.json")
        self.assertEqual(protocol["target"]["method"], "hellinger_rf0_bf16score")
        self.assertEqual(protocol | {"target": evaluation.PROTOCOL["target"]}, evaluation.PROTOCOL)
        with self.assertRaisesRegex(ValueError, "protocol changed"):
            evaluation.verify_snapshot(self.g_args.output)

    def test_g_terminal_requires_its_own_label_target_and_original_full_protocol(self):
        rows = [dict(video=f"{i:03d}-00.mp4", frames=81, psnr=12.1, ssim=.4, lpips=.3) for i in range(20)]
        methods = {name: dict(videos=copy.deepcopy(rows), summary=quality.aggregate(rows),
                   mean_attention_ms=None, speedup_vs_sdpa=None) for name in evaluation.methods_for("hellinger_rf0")}
        result = dict(valid=True, complete=True, protocol=evaluation.protocol_for("hellinger_rf0"), methods=methods,
            mean_attention_ms=None, speedup_vs_sdpa=None, historical_rf10_timing_established=False,
            inputs_reverified_at_end=True, sources_reverified_at_end=True, psnr_above_12=True)
        self.assertTrue(evaluation.result_complete(result, "hellinger_rf0"))
        self.assertFalse(evaluation.result_complete(result))
        wrong = copy.deepcopy(result)
        wrong["protocol"]["target"]["method"] = evaluation.METHODS[0]
        self.assertFalse(evaluation.result_complete(wrong, "hellinger_rf0"))
        wrong = copy.deepcopy(result)
        wrong["methods"][evaluation.METHODS[0]] = wrong["methods"].pop("hellinger_rf0_bf16score")
        self.assertFalse(evaluation.result_complete(wrong, "hellinger_rf0"))
        wrong = copy.deepcopy(result)
        wrong["protocol"]["resize"] = True
        self.assertFalse(evaluation.result_complete(wrong, "hellinger_rf0"))

    def test_cli_preserves_legacy_mass_and_requires_explicit_g_selection(self):
        shared = ["--reference-dir", str(self.reference), "--hellinger-dir", str(self.hf),
                  "--output", str(self.args.output), "--gpu", "0"]
        legacy = evaluation.parse_args(shared + ["--mass-dir", str(self.mass)])
        self.assertEqual(evaluation.selected_candidate(legacy), "mass_rf0")
        self.assertEqual(evaluation.candidate_directory(legacy), self.mass)
        new = evaluation.parse_args(shared + ["--candidate-method", "hellinger_rf0", "--candidate-dir", str(self.g)])
        self.assertEqual(evaluation.candidate_directory(new), self.g)
        with self.assertRaisesRegex(ValueError, "only valid"):
            evaluation.parse_args(shared + ["--candidate-method", "hellinger_rf0", "--mass-dir", str(self.g)])

    def test_g_parent_propagates_explicit_method_to_frozen_worker_and_completion(self):
        child = Mock(pid=991, returncode=0)
        child.poll.return_value = child.wait.return_value = 0
        clean = {"gpu": {"uuid": "GPU-fixture"}, "compute_processes": []}
        with patch.object(evaluation.guard, "require_clean", return_value=clean), \
             patch.object(evaluation.guard, "gpu_processes", return_value=[]), \
             patch.object(evaluation.subprocess, "Popen", return_value=child) as spawn, \
             patch.object(evaluation, "verify_completion", return_value={"fixture": True}) as complete:
            result = evaluation.launch(self.g_args)
        self.assertTrue(result["valid"])
        command = spawn.call_args.args[0]
        self.assertIn("--worker", command)
        self.assertEqual(command[command.index("--candidate-method") + 1], "hellinger_rf0")
        self.assertEqual(command[command.index("--candidate-dir") + 1], str(self.g))
        self.assertNotIn("--mass-dir", command)
        complete.assert_called_once_with(self.g_args.output.resolve(), "hellinger_rf0")
        startup = evaluation.read_json(self.g_args.output / "startup.json")
        self.assertEqual(startup["protocol"], evaluation.protocol_for("hellinger_rf0"))


if __name__ == "__main__":
    unittest.main()
