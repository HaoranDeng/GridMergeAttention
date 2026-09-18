"""Synthetic artifact tests using real canonical prompts; never decode MP4s."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "analysis" / "validate_rf0_video_run.py"
spec = importlib.util.spec_from_file_location("rf0_video_validator", SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
from utils.prompt_contract import build_prompt_manifest, load_nonempty_prompts


class RF0VideoRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run_dir = Path(self.temporary.name) / "rf0_fixture"
        self.run_dir.mkdir()
        (self.run_dir / "videos").mkdir()
        for name in audit.NAMES:
            (self.run_dir / "videos" / name).write_bytes(b"fake-not-decoded")
        (self.run_dir / "generation.exitcode").write_text("0\n")
        (self.run_dir / "purity.log").write_text("")
        prompt_path = audit.ROOT / "prompts" / "MovieGenVideoBench_extended.txt"
        self.prompts = build_prompt_manifest(prompt_path, load_nonempty_prompts(prompt_path), selected_count=20,
                                             seed=0, num_samples=1, vbench_filenames=False, allow_noncanonical=False)
        self.write("videos/prompt_manifest.json", self.prompts)
        self.write("prompt_preflight_manifest.json", self.prompts)
        self.config_name = "configs/fixture_rf0.yaml"
        config = "token_merging_args:\n  enable: true\n  seed_mode: " + audit.MODES["hellinger_rf0"] + "\n"
        sources = {"generate.py": "# generation\n", "wan/modules/model.py": "# model\n",
                   "attention/token_merging.py": "# dispatcher\n", "attention/adaptive_seed_triton.py": "# selector\n",
                   self.config_name: config}
        self.source = {"method": "Hellinger C4 RF0 control", "remote_base_commit": "a" * 40,
                       "files": [], "planned": {"router_frame": 0, "seed": 0, "num_samples": 1,
                                                 "quality_reference": "/remote/dense/videos"}}
        for name, text in sources.items():
            target = self.run_dir / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            self.source["files"].append({"path": name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        self.manifest = {"method": "GCA_Hellinger_RF0_Uniform16", "git_commit": "a" * 40,
                         "config": "/remote/work/" + self.config_name,
                         "config_sha256": self.source["files"][-1]["sha256"], "checkpoint": "/remote/raw.pt",
                         "checkpoint_weights": None, "num_videos": 20, "frames": 81, "seed": 0,
                         "denoise_steps": 4, "attention_calls_per_video": 160}
        records = []
        for i, prompt in enumerate(self.prompts["prompts"]):
            core, full, inference = (20., 30., 8.) if i == 0 else (10., 20., 5.)
            records.append({"output": audit.NAMES[i], "prompt_index": i, "sample_index": 0,
                            "prompt_text": prompt["text"], "prompt_sha256": prompt["sha256"],
                            "attention_calls": 160, "full_attention_calls": 160,
                            "attention_total_ms": core * 160, "full_attention_total_ms": full * 160,
                            "mean_attention_ms": core, "mean_full_attention_ms": full,
                            "inference_seconds": inference, "export_seconds": .1})
        summary = {}
        for prefix, chosen in (("", records), ("steady_state_", records[1:])):
            summary.update({prefix + "videos": len(chosen), prefix + "attention_calls_per_video": 160.,
                            prefix + "mean_attention_ms": sum(row["mean_attention_ms"] for row in chosen) / len(chosen),
                            prefix + "mean_full_attention_ms": sum(row["mean_full_attention_ms"] for row in chosen) / len(chosen),
                            prefix + "mean_inference_seconds": sum(row["inference_seconds"] for row in chosen) / len(chosen)})
        self.runtime = {"config": self.manifest["config"], "checkpoint": self.manifest["checkpoint"],
                        "checkpoint_weights": "raw_state_dict", "prompt_manifest": self.prompts,
                        "records": records, "summary": summary}
        self.write("run_manifest.json", self.manifest)
        self.write("source_provenance.json", self.source)
        self.write("runtime.json", self.runtime)

    def write(self, relative, document):
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def set_mass_c32_run(self):
        """Freeze the production RF0 config, with only the Mass-GS selector present."""
        self.source["files"] = [entry for entry in self.source["files"]
                                if entry["path"] not in (self.config_name, "attention/adaptive_seed_triton.py")]
        self.config_name = "configs/token_merging_wan_720p_entropy_mass_span_rf0_fewstep.yaml"
        for name, content in ((self.config_name, (audit.ROOT / self.config_name).read_bytes()),
                              ("attention/entropy_mass_seed_triton.py", b"# Mass-GS selector fixture\n")):
            target = self.run_dir / "source" / name
            target.write_bytes(content)
            self.source["files"].append({"path": name, "sha256": hashlib.sha256(content).hexdigest()})
        self.source.update(method="Mass-GS C32 RF0 GCA", seed_mode=audit.MODES["mass_c32_rf0"])
        self.manifest.update(method="GCA_MassGS_C32_RF0_Uniform16", seed_mode=audit.MODES["mass_c32_rf0"],
                             config="/remote/work/" + self.config_name,
                             config_sha256=self.source["files"][-2]["sha256"])
        self.runtime["config"] = self.manifest["config"]
        self.write("source_provenance.json", self.source)
        self.write("run_manifest.json", self.manifest)
        self.write("runtime.json", self.runtime)

    def mass_quality(self):
        item = {"directory": "/remote/" + self.run_dir.name + "/videos",
                "prompt_contract": audit.CONTRACT | {"verified_prompt_indices": list(range(20))},
                "summary": {"videos": 20, "frames": 1620, "psnr": 12., "ssim": .4, "lpips": .5},
                "videos": [{"video": name, "frames": 81, "psnr": 12., "ssim": .4, "lpips": .5}
                           for name in audit.NAMES]}
        return {"reference": "/remote/dense/videos", "protocol": audit.PROTOCOL,
                "methods": {"mass_c32_rf0": item}}

    def test_valid_completed_run_has_all20_steady19_and_unavailable_quality_speed(self):
        report = audit.validate(self.run_dir)
        self.assertTrue(report["valid"])
        self.assertEqual(report["method_key"], "hellinger_rf0")
        self.assertIsNone(report["quality"])
        self.assertIsNone(report["stages"]["quality"]["valid"])
        self.assertTrue(report["stages"]["generation"]["valid"])
        all20, steady = report["generation_runtime"]["all20"], report["generation_runtime"]["steady19"]
        self.assertEqual((all20["attention_calls"], steady["attention_calls"]), (3200, 3040))
        self.assertEqual((all20["mean_attention_ms"], steady["mean_attention_ms"]), (10.5, 10.))
        self.assertEqual((all20["mean_full_attention_ms"], steady["mean_full_attention_ms"]), (20.5, 20.))
        self.assertIsNone(all20["speedup_vs_sdpa"])
        self.assertFalse(report["provenance"]["manifest_checkpoint_backfilled"])
        self.assertTrue({"runtime", "run_manifest", "source_provenance", "prompt_manifest"} <= report["inputs"].keys())

    def test_unfinished_or_contaminated_generation_is_rejected(self):
        for filename, content in (("generation.exitcode", "1"), ("purity.log", "foreign process")):
            original = (self.run_dir / filename).read_text()
            (self.run_dir / filename).write_text(content)
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                audit.validate(self.run_dir)
            (self.run_dir / filename).write_text(original)

    def test_partition_run_uses_its_own_exact_mode_and_source_files(self):
        config = self.run_dir / "source" / self.config_name
        config.write_text("token_merging_args:\n  seed_mode: " + audit.MODES["partition_rf0"] + "\n")
        self.source["method"] = "Partition C16 RF0 GCA"
        self.manifest["method"] = "GCA_PartitionMass_RF0_Uniform16"
        self.manifest["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
        for entry in self.source["files"]:
            if entry["path"] == self.config_name:
                entry["sha256"] = self.manifest["config_sha256"]
        for name in ("attention/partition_mass_seed_triton.py", "attention/entropy_mass_seed_triton.py"):
            target = self.run_dir / "source" / name
            target.write_text("# partition source fixture\n")
            self.source["files"].append({"path": name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        self.write("source_provenance.json", self.source)
        self.write("run_manifest.json", self.manifest)
        report = audit.validate(self.run_dir)
        self.assertEqual(report["method_key"], "partition_rf0")
        self.assertEqual(report["provenance"]["seed_mode"], audit.MODES["partition_rf0"])

    def test_mass_c32_run_uses_production_config_and_its_own_selector(self):
        self.set_mass_c32_run()
        report = audit.validate(self.run_dir)
        self.assertEqual(report["method_key"], "mass_c32_rf0")
        self.assertEqual(report["provenance"]["seed_mode"], audit.MODES["mass_c32_rf0"])
        self.assertIn("attention/entropy_mass_seed_triton.py", report["provenance"]["source_files"])
        self.assertNotIn("attention/partition_mass_seed_triton.py", report["provenance"]["source_files"])
        self.assertEqual(report["generation_runtime"]["all20"]["attention_calls"], 3200)
        self.assertIsNone(report["generation_runtime"]["all20"]["speedup_vs_sdpa"])

    def test_mass_c32_missing_selector_or_wrong_method_labels_rejected(self):
        self.set_mass_c32_run()
        source = copy.deepcopy(self.source)
        source["files"] = [entry for entry in source["files"]
                           if entry["path"] != "attention/entropy_mass_seed_triton.py"]
        self.write("source_provenance.json", source)
        with self.assertRaisesRegex(ValueError, "missing required generation/selector"):
            audit.validate(self.run_dir)
        self.write("source_provenance.json", self.source)
        for label in ("GCA_MassGS_C32_RF10", "GCA_PartitionMass_C32_RF0", "GCA_MassGS_C16_RF0"):
            self.write("run_manifest.json", dict(self.manifest, method=label))
            with self.subTest(label=label), self.assertRaises(ValueError):
                audit.validate(self.run_dir)

    def test_mass_c32_rf10_mode_rejected_even_with_matching_frozen_hashes(self):
        self.set_mass_c32_run()
        config = self.run_dir / "source" / self.config_name
        config.write_text(config.read_text().replace("_rf0_", "_rf10_"))
        digest = hashlib.sha256(config.read_bytes()).hexdigest()
        self.manifest["config_sha256"] = digest
        for entry in self.source["files"]:
            if entry["path"] == self.config_name:
                entry["sha256"] = digest
        self.write("source_provenance.json", self.source)
        self.write("run_manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "supported exact RF0"):
            audit.validate(self.run_dir)

    def test_mass_c32_renamed_config_rejected(self):
        self.set_mass_c32_run()
        name = "configs/unrelated_rf0.yaml"
        (self.run_dir / "source" / self.config_name).rename(self.run_dir / "source" / name)
        for entry in self.source["files"]:
            if entry["path"] == self.config_name:
                entry["path"] = name
        self.manifest["config"] = self.runtime["config"] = "/remote/work/" + name
        self.write("source_provenance.json", self.source)
        self.write("run_manifest.json", self.manifest)
        self.write("runtime.json", self.runtime)
        with self.assertRaisesRegex(ValueError, "requires its declared few-step config"):
            audit.validate(self.run_dir)

    def test_mass_c32_quality_is_accepted_only_for_matching_method_and_run(self):
        self.set_mass_c32_run()
        quality = self.mass_quality()
        quality_path = self.write("quality.json", quality)
        stage = self.run_dir / "quality_evaluation"
        stage.mkdir()
        (stage / "exit_code").write_text("0")
        (stage / "purity.log").write_text("")
        report = audit.validate(self.run_dir, quality_path=quality_path)
        self.assertTrue(report["quality"]["valid"])
        self.assertEqual(report["quality"]["method"], "mass_c32_rf0")
        self.assertEqual(report["quality"]["summary"]["psnr"], 12.)
        self.assertTrue(report["quality"]["method_directory_recorded"])
        for key in ("mass_span", "mass_c32_rf10", "partition_c16_rf0", "hellinger_uniform16_rf0"):
            altered = copy.deepcopy(quality)
            altered["methods"][key] = altered["methods"].pop("mass_c32_rf0")
            self.write("quality.json", altered)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "quality method key"):
                audit.validate(self.run_dir, quality_path=quality_path)
        quality["methods"]["mass_c32_rf0"]["directory"] = "/remote/different/videos"
        self.write("quality.json", quality)
        with self.assertRaisesRegex(ValueError, "directory does not identify this run"):
            audit.validate(self.run_dir, quality_path=quality_path)

    def test_missing_video_and_incomplete_runtime_are_rejected(self):
        path = self.run_dir / "videos" / audit.NAMES[-1]
        path.unlink()
        with self.assertRaises(ValueError):
            audit.validate(self.run_dir)
        path.write_bytes(b"not-decoded")
        runtime = copy.deepcopy(self.runtime)
        runtime["records"].pop()
        self.write("runtime.json", runtime)
        with self.assertRaises(ValueError):
            audit.validate(self.run_dir)

    def test_runtime_count_mean_prompt_and_weights_corruption_rejected(self):
        for field, value in (("attention_calls", 159), ("mean_full_attention_ms", 100.),
                             ("prompt_index", 2), ("inference_seconds", float("nan"))):
            runtime = copy.deepcopy(self.runtime)
            runtime["records"][0][field] = value
            self.write("runtime.json", runtime)
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.validate(self.run_dir)
        runtime = copy.deepcopy(self.runtime)
        runtime["checkpoint_weights"] = "ema"
        self.write("runtime.json", runtime)
        with self.assertRaises(ValueError):
            audit.validate(self.run_dir)

    def test_self_consistent_noncanonical_later_prompt_is_rejected(self):
        prompt = copy.deepcopy(self.prompts)
        prompt["prompts"][8]["text"] = "some other prompt"
        prompt["prompts"][8]["sha256"] = audit.prompt_sha256("some other prompt")
        self.write("videos/prompt_manifest.json", prompt)
        with self.assertRaisesRegex(ValueError, "canonical prompt"):
            audit.validate(self.run_dir)

    def test_source_hash_and_wrong_method_labels_rejected(self):
        target = self.run_dir / "source" / "attention/token_merging.py"
        original = target.read_bytes()
        target.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "source hash"):
            audit.validate(self.run_dir)
        target.write_bytes(original)
        manifest = dict(self.manifest, method="GCA_Hellinger_RF10_Uniform16")
        self.write("run_manifest.json", manifest)
        with self.assertRaises(ValueError):
            audit.validate(self.run_dir)

    def test_optional_quality_requires_complete_20_by_81_and_same_reference(self):
        quality = {"reference": "/remote/dense/videos", "protocol": audit.PROTOCOL,
                   "methods": {"hellinger_uniform16_rf0": {"prompt_contract": audit.CONTRACT | {"verified_prompt_indices": list(range(20))},
                     "summary": {"videos": 20, "frames": 1620, "psnr": 12., "ssim": .4, "lpips": .5},
                     "videos": [{"video": name, "frames": 81, "psnr": 12., "ssim": .4, "lpips": .5} for name in audit.NAMES]}}}
        quality_path = self.write("quality.json", quality)
        stage = self.run_dir / "quality_evaluation"
        stage.mkdir()
        (stage / "exit_code").write_text("0")
        (stage / "purity.log").write_text("")
        report = audit.validate(self.run_dir, quality_path=quality_path)
        self.assertTrue(report["quality"]["valid"])
        self.assertEqual(report["quality"]["summary"]["psnr"], 12.)
        self.assertFalse(report["quality"]["method_directory_recorded"])
        quality["methods"]["hellinger_uniform16_rf0"]["videos"][0]["frames"] = 80
        self.write("quality.json", quality)
        with self.assertRaises(ValueError):
            audit.validate(self.run_dir, quality_path=quality_path)
        quality["methods"]["hellinger_uniform16_rf0"]["videos"][0]["frames"] = 81
        quality["reference"] = "/remote/different/videos"
        self.write("quality.json", quality)
        with self.assertRaisesRegex(ValueError, "reference differs"):
            audit.validate(self.run_dir, quality_path=quality_path)
        quality["reference"] = "/remote/dense/videos"
        quality["methods"]["rf10_reference_matched"] = quality["methods"].pop("hellinger_uniform16_rf0")
        self.write("quality.json", quality)
        with self.assertRaisesRegex(ValueError, "quality method key"):
            audit.validate(self.run_dir, quality_path=quality_path)


if __name__ == "__main__":
    unittest.main()
