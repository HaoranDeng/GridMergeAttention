"""CPU contract tests; never open a GPU, model, or real QKV capture."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from analysis import benchmark_barycentric_full40 as bench


class CaptureFixture:
    def __init__(self, root):
        self.run = Path(root) / bench.CAPTURE_NAME
        self.root = self.run / "captures"
        self.root.mkdir(parents=True)
        self.captures = []
        for row in bench.expected_records():
            if row["capture_file"] is None: continue
            file = self.root / row["capture_file"]
            file.write_bytes(b"x")  # Synthetic receipts, never mistaken for actual QKV.
            receipt = {"path": str(file.resolve()), "bytes": 1,
                       "mtime_ns": file.stat().st_mtime_ns, "sha256": bench.sha(file)}
            self.captures.append(row | {"file": receipt, "bytes": 1, "sha256": receipt["sha256"],
                "shape": list(bench.SHAPE), "heads": list(range(40)), "dtype": "torch.bfloat16",
                "all_tensors_finite": True})
        self.manifest = {"complete": True, "total_self_attention_calls": 480, "total_captures": 24,
                         "records": bench.expected_records(), "captures": self.captures,
                         "checkpoint_weights": "raw_state_dict", "actual_forced_sdpa_backend": "flash",
                         "actual_use_torch_rmsnorm": False, "heads": list(range(40))}
        (self.root / "exit_code").write_text("0\n")
        self.save()

    def save(self):
        path = self.root / "capture_manifest.json"
        bench.write(path, self.manifest)
        self.completion = {"valid": True, "failure": None, "foreign": [], "worker_exit_code": 0,
                           "verified_artifacts": {"manifest_sha256": bench.sha(path), "captures": [
                               {k: row[k] for k in ("capture_file", "bytes", "sha256")} for row in self.captures]}}
        bench.write(self.run / "completion.json", self.completion)


class BenchmarkCaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = CaptureFixture(self.tmp.name)

    def validate(self):
        # Avoid creating 74GB of fake sparse files in CPU unit tests. The
        # separate test below verifies real launch cannot accept this fixture.
        with patch.object(bench, "MIN_CAPTURE_BYTES", 1):
            return bench.validate_capture_run(self.fixture.run)

    def test_complete_original_receipts_and_input_coverage(self):
        result = self.validate()
        self.assertEqual(len(result["captures"]), 24)
        self.assertEqual({row["layer_index"] for row in result["captures"]}, {0, 33})
        self.assertEqual({row["step_index"] for row in result["captures"]}, set(range(4)))
        self.assertEqual(len(bench.expected_records()), 480)

    def test_real_validator_rejects_undersized_fake_files(self):
        with self.assertRaisesRegex(ValueError, "undersized"):
            bench.validate_capture_run(self.fixture.run)

    def test_incomplete_parent_refused(self):
        self.fixture.completion["valid"] = False
        bench.write(self.fixture.run / "completion.json", self.fixture.completion)
        with self.assertRaisesRegex(ValueError, "not completed"):
            self.validate()

    def test_foreign_process_and_failed_exit_refused(self):
        for field, value in (("foreign", [{"pid": 123}]), ("worker_exit_code", 1)):
            self.fixture.save()
            self.fixture.completion[field] = value
            bench.write(self.fixture.run / "completion.json", self.fixture.completion)
            with self.assertRaises(ValueError): self.validate()

    def test_manifest_sha_and_receipt_sha_changes_refused(self):
        original = (self.fixture.root / "capture_manifest.json").read_text()
        (self.fixture.root / "capture_manifest.json").write_text(original + " ")
        with self.assertRaisesRegex(ValueError, "manifest SHA"):
            self.validate()
        self.fixture.save()
        self.fixture.completion["verified_artifacts"]["captures"][0]["sha256"] = "a" * 64
        bench.write(self.fixture.run / "completion.json", self.fixture.completion)
        with self.assertRaisesRegex(ValueError, "parent SHA"):
            self.validate()

    def test_noncanonical_call_order_and_repeated_heads_refused(self):
        self.fixture.manifest["records"][1]["layer_index"] = 0
        self.fixture.save()
        with self.assertRaises(ValueError): self.validate()
        self.fixture.manifest["records"] = bench.expected_records()
        self.fixture.captures[0]["heads"] = [0, 10, 20, 30] * 10
        self.fixture.save()
        with self.assertRaisesRegex(ValueError, "full40"):
            self.validate()

    def test_extra_missing_file_and_modified_size_refused(self):
        extra = self.fixture.root / "extra.pt"
        extra.write_bytes(b"x")
        with self.assertRaises(ValueError): self.validate()
        extra.unlink()
        Path(self.fixture.captures[0]["file"]["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed after"):
            self.validate()

    def test_payload_shape_dtype_metadata_guards_without_torch(self):
        receipt = self.fixture.captures[0]
        metadata = {k: v for k, v in receipt.items() if k not in ("file", "bytes", "sha256")}
        tensor = SimpleNamespace(shape=bench.SHAPE, dtype="BF16", device=SimpleNamespace(type="cpu"), is_contiguous=lambda: True)
        payload = {name: tensor for name in ("q", "k", "v", "dense_output")} | {"metadata": metadata}
        with patch.dict("sys.modules", {"torch": SimpleNamespace(bfloat16="BF16")}):
            bench.validate_payload(payload, receipt)
            payload["q"] = SimpleNamespace(**(vars(tensor) | {"dtype": "FP16"}))
            with self.assertRaises(ValueError): bench.validate_payload(payload, receipt)
            payload["q"] = tensor; payload["metadata"] = metadata | {"heads": [0] * 40}
            with self.assertRaises(ValueError): bench.validate_payload(payload, receipt)


class BenchmarkMethodTests(unittest.TestCase):
    def test_stable_suite_is_explicit_and_old_default_is_unchanged(self):
        self.assertIs(bench.suite_methods(), bench.METHODS)
        self.assertEqual(bench.suite_methods(True), ("sdpa_flash", "historical_hf_rf10_fp8score", "hellinger_rf0",
                         "fisher_anchor_rf0", "hellinger_stable_sq1_rf0", "fisher_anchor_stable_sq1_rf0"))
        self.assertEqual(bench.repeat_order(1, bench.STABLE_SQ1_SUITE), tuple(reversed(bench.STABLE_SQ1_SUITE)))
        with self.assertRaises(ValueError): bench.suite_methods("true")

    def test_stable_source_prediction_and_artifact_revalidation_cpu_only(self):
        from analysis.stable_sq1_bulk_kernel import generate_stable_sq1_source
        with tempfile.TemporaryDirectory() as root:
            expected = bench.expected_stable_receipt(root)
            path = Path(expected["path"])
            self.assertFalse(path.exists())
            source = generate_stable_sq1_source((bench.ROOT / "attention/hard_c4_wan21_14b_720p_triton.py").read_text())
            path.write_bytes(source.encode("utf-8"))
            self.assertEqual(bench.verify_stable_receipt(expected, expected), expected)
            with self.assertRaises(ValueError): bench.verify_stable_receipt(expected | {"sha256": "0" * 64}, expected)
            path.write_bytes(path.read_bytes() + b"# changed\n")
            with self.assertRaisesRegex(ValueError, "source changed"): bench.verify_stable_receipt(expected, expected)
        with self.assertRaisesRegex(ValueError, "outside immutable source"):
            bench.expected_stable_receipt(bench.ROOT / "do_not_create")

    def test_stable_helpers_are_required_source_dependencies(self):
        with patch.object(bench, "sha", return_value="read-only-test"):
            ordinary, stable = bench.source_hashes(), bench.source_hashes(True)
        for name in ("stable_sq1_bulk_kernel.py", "stable_sq1_video_adapter.py"):
            path = str((bench.ROOT / "analysis" / name).resolve())
            self.assertNotIn(path, ordinary)
            self.assertIn(path, stable)

    def test_stable_binding_uses_original_selector_and_never_mean_lifecycle(self):
        from analysis import stable_sq1_video_adapter as adapter_module
        from analysis.quality_first_mode_contract import method_contract
        attention, token, core, native = (ModuleType(name) for name in ("attention", "attention.token_merging", "attention.core", "attention.affine_hellinger_seed_triton"))
        attention.__path__ = []
        attention.token_merging, attention.core = token, core
        def function(identity):
            def result(*args, **kwargs): return output
            result.source_identity = identity
            return result
        output = SimpleNamespace(contiguous=lambda: output, reshape=lambda shape: output)
        token.hellinger_span_seed_rows_triton = function("attention.adaptive_seed_triton.hellinger_span_seed_rows_triton")
        token.entropy_mass_seed_rows_triton = function("unused.mass.selector")
        native.fisher_anchor_seed_rows_triton = function("attention.affine_hellinger_seed_triton.fisher_anchor_seed_rows_triton")
        setattr(token, bench.MERGE, function("original.merge"))
        core.flash_attention = function("original.dense")
        def complete(*args, **kwargs):
            selected = getattr(token, method_contract(current[0])["selector_symbol"])("original RF0 query", "original K", 128 ** -.5)
            for start in range(0, 40, 8):
                getattr(token, bench.MERGE)(*([output] * 3), selected, 128 ** -.5, head_start=start, head_count=8)
            return output
        complete.source_identity = "original.complete"
        token.token_merging_attn = complete
        class Adapter:
            # Deliberately no begin_attention/end_attention methods.
            def __init__(self, original, module, kernel): self.original, self.calls, self.kernel_calls = original, 0, 0
            def __call__(self, *args, **kwargs):
                self.calls += 1; self.kernel_calls += 1
                return self.original(*args, **kwargs)
        Adapter.__call__.source_identity = "stable.adapter"
        modules = {"attention": attention, "attention.token_merging": token, "attention.core": core,
                   "attention.affine_hellinger_seed_triton": native,
                   "attention.hard_c4_wan21_14b_720p_triton": ModuleType("hard")}
        current = [None]
        with patch.dict("sys.modules", modules), patch.object(bench, "identity", side_effect=lambda f: {"callable": f.source_identity}), \
                patch.object(adapter_module, "StableSQ1MergeAdapter", Adapter):
            for name in bench.STABLE_METHODS:
                current[0] = name
                binding = bench.Binding(name, (output,) * 3, stable_kernel=object(), stable_receipt={"sha256": "test-only"})
                self.assertEqual(binding.symbol, method_contract(name)["selector_symbol"])
                with binding.context() as (invoke, state):
                    invoke(); invoke()
                self.assertEqual(state["stable_counts"], {"complete_calls": 2, "merge_calls": 10, "stable_merge_kernel_calls": 10})
                self.assertNotIn("mean_counts", state)

    def test_stable_summary_checks_all_actual_kernel_counts(self):
        records = []
        for index in range(24):
            methods = {}
            for name in bench.STABLE_SQ1_SUITE:
                row = {"repeat_ms": [1.] * 4, "error": {"numel": 400, "squared_error_sum": 0., "per_head_squared_error_sum": [0.] * 40}}
                if name in bench.STABLE_METHODS:
                    row["warmup_observation"] = {"stable_counts": {"complete_calls": 2, "merge_calls": 10, "stable_merge_kernel_calls": 10}}
                    row["timed_stable_counts"] = [{"complete_calls": 5, "merge_calls": 25, "stable_merge_kernel_calls": 25} for _ in range(4)]
                methods[name] = row
            records.append({"capture_file": f"capture{index}", "methods": methods})
        summary = bench.summarize(records, bench.STABLE_SQ1_SUITE)
        for name in bench.STABLE_METHODS:
            self.assertEqual(summary[name]["timed_calls"], 480)
            self.assertEqual(summary[name]["stable_merge_kernel_calls"], 2400)
            self.assertEqual(summary[name]["warmup_stable_merge_kernel_calls"], 240)
        records[0]["methods"][bench.STABLE_METHODS[0]]["timed_stable_counts"][0]["stable_merge_kernel_calls"] = 24
        with self.assertRaisesRegex(ValueError, "timed kernels"): bench.summarize(records, bench.STABLE_SQ1_SUITE)

    def test_account_cap_checked_before_preparation_spawn_and_each_monitor_poll(self):
        from analysis import launch_quality_first_videos as guard
        for stages in ((["a", "b"],), ([], ["a", "b"]), ([], [], ["a", "b", "c"])):
            with self.subTest(stages=stages), tempfile.TemporaryDirectory() as root:
                args = SimpleNamespace(stable_sq1_suite=False, capture_run=Path(root), historical_audit=Path(root),
                                       gpu=0, output=Path(root) / "fresh")
                process = SimpleNamespace(pid=123, returncode=0, poll=lambda: 0)
                with patch.object(bench, "validate_capture_run", return_value={}), patch.object(bench, "historical_mode"), \
                        patch.object(bench, "source_hashes", return_value={}), patch.object(bench, "sha", return_value="test"), \
                        patch.object(guard, "require_clean", return_value={"gpu": {"uuid": "GPU-test"}}), \
                        patch.object(guard, "account_limit", return_value=2), patch.object(guard, "account_devices", side_effect=stages), \
                        patch.object(guard, "gpu_processes", return_value=[]), patch.object(guard, "child_environment", return_value={}), \
                        patch.object(guard, "stop_own_child") as stop, patch.object(bench.subprocess, "Popen", return_value=process) as spawn:
                    with self.assertRaisesRegex(ValueError, "account GPU cap"): bench.launch(args)
                    self.assertEqual(spawn.call_count, int(len(stages) == 3))
                    if len(stages) == 1:
                        self.assertFalse(args.output.exists())
                    else:
                        receipt = json.loads((args.output / "completion.json").read_text())
                        self.assertFalse(receipt["timing_valid"])
                        self.assertEqual(receipt["last_account_devices"], stages[-1])
                        stop.assert_called_once_with(process if len(stages) == 3 else None)

    def test_historical_exact_mode_and_audited_rf10(self):
        config = "token_merging_args:\n  seed_mode: " + bench.HF10_MODE + "\n"
        payload = {"hellinger_rf10": {"policy": {"router_frame": 10, "candidate_count": 32, "pure_all_layer_config": True},
                   "config_text": config, "config": {"sha256": hashlib.sha256(config.encode()).hexdigest()}}}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "audit.json"; bench.write(path, payload)
            with patch.object(bench, "AUDIT_SHA", bench.sha(path)):
                self.assertEqual(bench.historical_mode(path), bench.HF10_MODE)
            with self.assertRaises(ValueError): bench.historical_mode(path)
        self.assertIn("_t1_u16_", bench.HF10_MODE)
        self.assertIn("_fp8score_", bench.HF10_MODE)

    def test_method_set_order_and_no_dense_rescue_options(self):
        self.assertEqual(len(bench.METHODS), 6)
        self.assertIn("mean_fisher_rf0", bench.METHODS)
        self.assertIn("barycentric_medoid_rf0", bench.METHODS)
        self.assertEqual(bench.repeat_order(0), tuple(reversed(bench.repeat_order(1))))
        self.assertEqual(bench.repeat_order(0), bench.repeat_order(2))
        options = bench.operator_options("test")
        self.assertEqual((options["query_outer_chunk"], options["head_chunk"], options["num_merge_tokens"]), (21, 8, 4))
        for key in ("seed_query", "seed_table_cache", "cache_timestep", "q_init"):
            self.assertIsNone(options[key])
        self.assertEqual(options["dense_heads"], ())
        self.assertFalse(options["first_frame_full_attention"])

    def test_pooled_timing_mse_speedups_and_complete_scope(self):
        records = []
        for index in range(24):
            methods = {}
            for number, name in enumerate(bench.METHODS, 1):
                methods[name] = {"repeat_ms": [float(number)] * 4, "error": {
                    "numel": 400, "squared_error_sum": number * 400,
                    "per_head_squared_error_sum": [number * 10] * 40}}
            records.append({"capture_file": f"sample{index}.pt", "methods": methods})
        result = bench.summarize(records)
        self.assertEqual(result["sdpa_flash"]["timed_calls"], 480)
        self.assertEqual(result["sdpa_flash"]["capture_inputs"], 24)
        self.assertEqual(result[bench.METHODS[-1]]["mean_attention_ms"], 6)
        self.assertEqual(result[bench.METHODS[-1]]["speedup_vs_sdpa"], 1 / 6)
        self.assertEqual(result[bench.METHODS[-1]]["speedup_vs_historical_hf_rf10"], 2 / 6)
        self.assertEqual(result[bench.METHODS[-1]]["per_head_mse"], [6] * 40)
        with self.assertRaises(ValueError): bench.summarize(records[:-1])
        records[0]["methods"]["sdpa_flash"]["repeat_ms"][0] = float("nan")
        with self.assertRaises(ValueError): bench.summarize(records)

    def test_live_parent_guard(self):
        with patch.object(bench.os, "getppid", return_value=123), patch.dict(bench.os.environ, {}, clear=True):
            with self.assertRaises(ValueError): bench.require_parent()
            bench.os.environ["BARYCENTRIC_BENCH_PARENT"] = "123"
            bench.require_parent()

    def test_actual_timing_and_mean_lifecycle_ast(self):
        tree = ast.parse(Path(bench.__file__).read_text())
        invoke = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "invoke")
        calls = [node for node in ast.walk(invoke) if isinstance(node, ast.Call)]
        names = [node.func.attr for node in calls if isinstance(node.func, ast.Attribute)]
        self.assertIn("begin_attention", names)
        self.assertIn("end_attention", names)
        self.assertIn("contiguous", names)
        self.assertFalse(any(isinstance(n, ast.Name) and n.id in ("reference", "dense_output", "payload") for n in ast.walk(invoke)))
        worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "worker")
        timing = next(node for node in ast.walk(worker) if isinstance(node, ast.With)
                      and any(isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Attribute)
                              and item.context_expr.func.attr == "context" and not item.context_expr.keywords for item in node.items))
        statements = [ast.unparse(node) for node in timing.body]
        start = next(i for i, text in enumerate(statements) if text == "start.record()")
        loop = next(i for i, text in enumerate(statements) if "range(CALLS_PER_REPEAT)" in text)
        end = next(i for i, text in enumerate(statements) if text == "end.record()")
        self.assertLess(start, loop); self.assertLess(loop, end)
        # GPU transfers preserve dtype and happen outside timing; no head repeat.
        source = ast.unparse(worker)
        self.assertNotIn("repeat_interleave", source)
        self.assertNotIn(".repeat(", source)
        self.assertNotIn("to(dtype=", source)


if __name__ == "__main__": unittest.main()
