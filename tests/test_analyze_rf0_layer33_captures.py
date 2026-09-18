"""CPU contracts for audited L33 analysis; no native imports or GPU execution."""
import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis/analyze_rf0_layer33_captures.py"
spec = importlib.util.spec_from_file_location("analyze_layer33", SOURCE)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def make_bundle(parent):
    root = Path(parent) / "captures"
    root.mkdir()
    for directory, name, text in ((root, "exit_code", "0\n"), (root.parent, "exit_code", "0\n"),
                                  (root.parent, "worker_exit_code", "0\n"), (root.parent, "purity.log", "")):
        (directory / name).write_text(text)
    prompt_path = ROOT / "prompts/MovieGenVideoBench_extended.txt"
    prompts = analysis.build_prompt_manifest(prompt_path, analysis.load_nonempty_prompts(prompt_path), selected_count=3,
        seed=0, num_samples=1, vbench_filenames=False, allow_noncanonical=False)
    records = [analysis.producer.call_record(p, c, c % 40, c // 40, analysis.producer.ACTUAL_TIMESTEPS[c // 40])
               for p in range(3) for c in range(160)]
    sources = {name: analysis.fingerprint(ROOT / name)["sha256"] for name in analysis.producer.SOURCES}
    manifest = {"complete": True, "total_self_attention_calls": 480, "expected_total_calls": 480,
        "total_captures": 12, "expected_captures": 12, "planned_raw_tensor_bytes": analysis.producer.RAW_CAPTURE_BYTES,
        "heads": list(analysis.producer.HEADS), "capture_calls_per_prompt": list(analysis.producer.TARGET_CALLS),
        "actual_warped_timesteps": list(analysis.producer.ACTUAL_TIMESTEPS), "checkpoint_weights": "raw_state_dict",
        "actual_forced_sdpa_backend": "flash", "actual_use_torch_rmsnorm": False,
        "checkpoint_identity": {"path": "/audited/checkpoint.pt", "bytes": 123, "mtime_ns": 456},
        "source_sha256": sources, "config_sha256": analysis.fingerprint(ROOT / analysis.CONFIG)["sha256"],
        "settings": {"config": "/reviewed/" + analysis.CONFIG}, "prompt_manifest": prompts,
        "records": records, "captures": []}
    for row in records:
        if row["capture_file"] is None:
            continue
        # Small bytes deliberately test the identity gate only, not Tensor decoding.
        content = row["capture_file"].encode()
        (root / row["capture_file"]).write_bytes(content)
        manifest["captures"].append(row | {"heads": list(analysis.producer.HEADS),
            "shape": list(analysis.producer.SAVED_SHAPE), "dtype": "torch.bfloat16", "sample_seed": row["prompt_index"],
            "checkpoint_weights": "raw_state_dict", "prompt": prompts["prompts"][row["prompt_index"]],
            "checkpoint_identity": manifest["checkpoint_identity"], "config_sha256": manifest["config_sha256"],
            "capture_source_sha256": sources["analysis/capture_rf0_layer33_diagnostic.py"],
            "frozen_stream_sha256": analysis.producer.FROZEN_STREAM_SHA256, "scale": 128 ** -0.5,
            "qk_position": "after Q/K RMSNorm and RoPE; V after projection",
            "reference": "Same call's original 40-head forced Flash SDPA output, sliced to saved heads", "bytes": len(content)})
    for name, value in (("capture_manifest.json", manifest), ("prompt_manifest.json", prompts),
                         ("prompt_preflight_manifest.json", prompts)):
        (root / name).write_text(json.dumps(value), encoding="utf-8")
    return root, manifest


def metric(count, sse, energy, marginal):
    return {"numel": count, "squared_error_sum": sse, "reference_squared_sum": energy,
            "head_squared_error_sum": marginal, "frame_squared_error_sum": marginal}


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root, self.manifest = make_bundle(self.temp.name)

    def write_manifest(self, manifest):
        (self.root / "capture_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_complete_small_identity_bundle_and_nested_wrapper_evidence(self):
        _, audit = analysis.validate_bundle(self.root)
        self.assertEqual(len(audit["capture_files"]), 12)
        self.assertEqual(audit["tensor_payloads_validated"], 0)
        self.assertIn("wrapper_worker_exit_code", audit["inputs"])
        self.assertEqual(Path(audit["inputs"]["wrapper_purity.log"]["path"]).parent, self.root.parent)
        self.assertIn("no per-file SHA256 receipt", audit["hash_scope"])

    def test_incomplete_wrong_order_or_provenance_rejected(self):
        mutations = [("complete", False), ("total_self_attention_calls", 479), ("checkpoint_weights", "ema"),
                     ("heads", [0, 10, 30, 20]), ("actual_warped_timesteps", [1000, 750, 500, 250]),
                     ("actual_use_torch_rmsnorm", True), ("actual_forced_sdpa_backend", "auto"),
                     ("records", self.manifest["records"][::-1]), ("config_sha256", "0" * 64)]
        for key, value in mutations:
            with self.subTest(key=key):
                self.write_manifest(self.manifest | {key: value})
                with self.assertRaises(ValueError):
                    analysis.validate_bundle(self.root)
        self.write_manifest(self.manifest)

    def test_all_exit_statuses_and_purity_are_required(self):
        for directory, filename in ((self.root, "exit_code"), (self.root.parent, "exit_code"),
                                    (self.root.parent, "worker_exit_code"), (self.root.parent, "purity.log")):
            original = (directory / filename).read_text()
            (directory / filename).write_text("1\n")
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                analysis.validate_bundle(self.root)
            (directory / filename).write_text(original)

    def test_source_hash_and_later_prompt_metadata_are_strict(self):
        changes = [("sample_seed", 0), ("prompt", self.manifest["prompt_manifest"]["prompts"][0]),
                   ("bytes", 1), ("dtype", "torch.float16"), ("scale", 1), ("call_index", 152)]
        for key, value in changes:
            changed = copy.deepcopy(self.manifest)
            changed["captures"][-1][key] = value
            self.write_manifest(changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                analysis.validate_bundle(self.root)
        changed = copy.deepcopy(self.manifest)
        changed["source_sha256"]["generate.py"] = "a" * 64
        self.write_manifest(changed)
        with self.assertRaisesRegex(ValueError, "source hash"):
            analysis.validate_bundle(self.root)

    def test_prompt_preflight_and_exact_file_inventory(self):
        preflight = self.root / "prompt_preflight_manifest.json"
        bad = copy.deepcopy(self.manifest["prompt_manifest"])
        bad["prompts"][2]["text"] += " changed"
        preflight.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "preflight"):
            analysis.validate_bundle(self.root)
        preflight.write_text(json.dumps(self.manifest["prompt_manifest"]))
        (self.root / "extra.pt").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "filename set"):
            analysis.validate_bundle(self.root)

    def test_payload_metadata_and_bf16_shape_contract_without_large_tensor(self):
        class FakeTensor:
            shape = analysis.producer.SAVED_SHAPE
            dtype = "bf16"
            device = SimpleNamespace(type="cpu")
            def is_contiguous(self):
                return True
        item = self.manifest["captures"][0]
        path = self.root / item["capture_file"]
        payload = {name: FakeTensor() for name in analysis.TENSORS}
        payload["metadata"] = {k: v for k, v in item.items() if k != "bytes"}
        fake = SimpleNamespace(Tensor=FakeTensor, bfloat16="bf16", load=lambda *a, **k: payload,
                               isfinite=lambda value: SimpleNamespace(all=lambda: True))
        with patch.dict(sys.modules, {"torch": fake}):
            loaded = analysis.load_capture(path, item, analysis.fingerprint(path))
            self.assertIs(loaded, payload)
            payload["metadata"]["bytes"] = item["bytes"]
            with self.assertRaisesRegex(ValueError, "PT metadata"):
                analysis.load_capture(path, item, analysis.fingerprint(path))
            del payload["metadata"]["bytes"]
            payload["q"].dtype = "fp16"
            with self.assertRaisesRegex(ValueError, "BF16"):
                analysis.load_capture(path, item, analysis.fingerprint(path))


class MetricsTests(unittest.TestCase):
    def test_pooled_sse_numel_energy_and_axis_marginals(self):
        values = [metric(8, 4., 16., [1., 3.]), metric(16, 16., 64., [6., 10.])]
        merged = analysis.aggregate_metrics(values)
        self.assertEqual(merged["numel"], 24)
        self.assertAlmostEqual(merged["mse"], 20 / 24)
        self.assertNotAlmostEqual(merged["mse"], (4 / 8 + 16 / 16) / 2)
        self.assertAlmostEqual(merged["relative_l2"], math.sqrt(20 / 80))
        np.testing.assert_allclose(merged["head_mse"], [7 / 12, 13 / 12])

    def test_invalid_aggregation_rejected(self):
        for values in ([], [metric(0, 0., 1., [0., 0.])], [metric(8, 4., 16., [2., 3.])],
                       [metric(8, float("nan"), 16., [1., 3.])]):
            with self.assertRaises(ValueError):
                analysis.aggregate_metrics(values)

    def test_native_seed_rows_distinct_anchor_and_order_comparison(self):
        rows = np.arange(45).reshape(1, 1, 1, 45, 1, 1, 1)
        table = np.broadcast_to((rows + np.arange(4)) % 45, analysis.SEED_SHAPE).astype(np.uint8)
        other = table[..., [0, 2, 1, 3]]
        result = analysis.seed_comparison(other, table)
        self.assertEqual(result["groups"], 302400)
        self.assertEqual(result["mean_shared_rows_vs_hellinger"], 4)
        self.assertEqual(result["same_set_fraction"], 1)
        self.assertEqual(result["exact_ordered_tuple_fraction"], 0)
        for slot, value in ((0, 44), (1, 0), (2, 45)):
            bad = table.copy()
            bad[0, 0, 0, 0, 0, 0, slot] = value
            with self.assertRaises(ValueError):
                analysis.seed_comparison(bad)

    def test_uniform_and_concentrated_probe_statistics(self):
        positions = {"partition_c16": np.tile(np.arange(4), (2, 1))}
        uniform = analysis.probe_statistics(np.zeros((2, 16, 16)), np.zeros((2, 16)), positions)
        np.testing.assert_allclose(uniform["max_normalized_candidate_weight"], 1 / 16)
        np.testing.assert_allclose(uniform["candidate_weight_effective_count"], 16)
        np.testing.assert_allclose(uniform["normalized_candidate_weight_entropy"], 1)
        np.testing.assert_allclose(uniform["mean_normalized_probe_entropy"], 1)
        np.testing.assert_allclose(uniform["partition_c16_selected_candidate_weight_share"], 1 / 4)
        logits, mass = np.full((2, 16, 16), -2000.), np.full((2, 16), -2000.)
        logits[..., 0], mass[..., 0] = 0, 0
        concentrated = analysis.probe_statistics(logits, mass, positions)
        self.assertTrue(all(np.isfinite(value).all() for value in concentrated.values()))
        np.testing.assert_allclose(concentrated["candidate_weight_effective_count"], 1)
        np.testing.assert_allclose(concentrated["mean_normalized_probe_entropy"], 0)
        np.testing.assert_allclose(concentrated["partition_c16_selected_candidate_weight_share"], 1)

    def test_pure_existing_operator_no_fake_table_and_no_recomputed_reference(self):
        analysis.benchmark.validate_modes(analysis.MODES)
        source, tree = SOURCE.read_text(encoding="utf-8"), ast.parse(SOURCE.read_text(encoding="utf-8"))
        observed = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "observed_attention")
        text = ast.get_source_segment(source, observed)
        self.assertIn("require_production_options(options)", text)
        self.assertIn("table = real(*inputs, **options)", text)
        self.assertIn("return table", text)
        self.assertNotIn("options.update", text)
        self.assertNotIn("PARTITION_OVERRIDE", source)
        self.assertNotIn("scaled_dot_product_attention(", source)
        self.assertNotIn("partition_mass_medoid_seed_triton", source)


if __name__ == "__main__":
    unittest.main()
