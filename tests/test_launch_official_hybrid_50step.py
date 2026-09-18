"""CPU-only checks for the new released-setting launch/report contract."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis import launch_official_hybrid_50step as run


class OfficialHybridContractTests(unittest.TestCase):
    def test_flashinfer_cache_is_private_to_run(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "ear"
            second = Path(directory) / "svg2"
            self.assertEqual(Path(run.flashinfer_workspace(first)).parent, first.resolve())
            self.assertNotEqual(run.flashinfer_workspace(first), run.flashinfer_workspace(second))

    def test_canonical_prompt_contract(self):
        manifest = run.prompt_manifest()
        self.assertEqual(manifest["selected_prompt_count"], 20)
        self.assertEqual(manifest["seed"], 0)
        self.assertEqual(manifest["prompts"][0]["sha256"],
                         "d0bf2d0fa5fd5f0a08da801d40232bb9e960c84d5d4889c175a0da4e4cc0aa05")

    def test_hybrid_call_arithmetic(self):
        self.assertEqual(run.CALLS, 50 * 40 * 2)
        self.assertEqual(run.DENSE_CALLS, 2 * (10 * 40 + 40))
        self.assertEqual(run.SPARSE_CALLS + run.DENSE_CALLS, run.CALLS)
        self.assertEqual(run.METHODS, ("svg", "svg2", "ear", "gca_fisher_anchor_stable_rf0_all",
                                     "gca_fisher_anchor_stable_rf0_hw_swap_all", "gma_c4_mix"))
        self.assertEqual(run.GCA_METHODS, run.METHODS[3:5])

    def test_density_includes_dense_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "density.jsonl"
            path.write_text("\n".join(json.dumps({"avg_density": value})
                                      for value in (0.1, 0.3, 0.2, 0.4)))
            with patch.multiple(run, VIDEOS=2, SPARSE_CALLS=2, DENSE_CALLS=1, CALLS=3):
                result = run.summarize_density(path, {}, "ear")
            self.assertAlmostEqual(result["sparse_call_density"], 0.25)
            self.assertAlmostEqual(result["all_call_density"], 0.5)
            self.assertAlmostEqual(result["per_video"][0]["all_call_density"], 1.4 / 3)
            self.assertEqual(result["dense_calls"], 2)

    def test_density_rejects_partial_or_invalid_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "density.jsonl"
            for value in (0.3, -0.1, 1.1):
                path.write_text(json.dumps({"avg_density": value}))
                with self.assertRaises(ValueError):
                    run.summarize_density(path, {}, "svg2")

    def test_worker_cannot_bypass_live_parent(self):
        with patch.dict(run.os.environ, {"GCA_OFFICIAL_50STEP_PARENT": "invalid"}):
            with self.assertRaisesRegex(ValueError, "guarded parent"):
                run.worker(None)

    def test_both_gca_modes_enforce_full_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "videos").mkdir()
            for index in range(20):
                (output / "videos" / f"{index:03d}-00.mp4").touch()
            run.write_json(output / "native_validation.json", {"valid": True})
            runtime = {
                "protocol": {"num_steps": 50, "num_frames": 81, "canonical_names": True,
                    "original_script_seeding": True, "gca_stable": {"layout": {
                        "key_axis": "H", "query_axis": "W", "key_group_size": 45,
                        "query_group_size": 80, "key_groups_per_frame": 80,
                        "query_groups_per_frame": 45, "compressed_slots_per_group": 5, "full_route": True}}},
                "summary": {"attention_calls": 80000},
                "videos": [{"index": index, "attention_calls": 4000, "full_attention_calls": 4000,
                    "attention_dispatch": {"dense": 0, "sparse": 0, "gca": 4000,
                        "per_layer": [{"gca": 100, "dense": 0, "sparse": 0} for _ in range(40)],
                        "gca_dispatch": {"core_calls": 4000, "selector_calls": 4000,
                            "merge_wrapper_calls": 20000, "stable_kernel_calls": 20000}}}
                    for index in range(20)]}
            for method in run.GCA_METHODS:
                runtime["backend"] = method
                run.write_json(output / "videos/runtime.json", runtime)
                self.assertEqual(run.validate_runtime(output, method)["backend"], method)
                runtime["videos"][0]["attention_dispatch"]["gca_dispatch"]["stable_kernel_calls"] = 19999
                run.write_json(output / "videos/runtime.json", runtime)
                with self.assertRaisesRegex(ValueError, "merge coverage"):
                    run.validate_runtime(output, method)
                runtime["videos"][0]["attention_dispatch"]["gca_dispatch"]["stable_kernel_calls"] = 20000
            run.write_json(output / "videos/runtime.json", runtime)
            run.write_json(output / "native_validation.json", {"valid": False})
            with self.assertRaisesRegex(ValueError, "native validation"):
                run.validate_runtime(output, run.GCA_METHODS[1])

    def test_dirty_or_wrong_upstream_rejected_before_snapshot(self):
        with patch.object(run, "git", return_value="wrong"):
            with self.assertRaisesRegex(ValueError, "revision"):
                run.source_inventory()

    def test_staged_edits_are_also_rejected(self):
        def fake_git(root, *arguments):
            if arguments == ("rev-parse", "HEAD"):
                return run.SVG_COMMIT
            if arguments == ("diff", "HEAD", "--name-only"):
                return "staged_change.py"
            return ""
        with patch.object(run, "git", side_effect=fake_git):
            with self.assertRaisesRegex(ValueError, "tracked source edits"):
                run.source_inventory()


if __name__ == "__main__":
    unittest.main()
