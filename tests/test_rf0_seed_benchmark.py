"""CPU-only arithmetic and source-contract checks; no Torch/CUDA import."""
import ast
import importlib.util
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "analysis" / "benchmark_rf0_seed_attention.py"
spec = importlib.util.spec_from_file_location("rf0_seed_benchmark", SOURCE)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class RF0SeedBenchmarkTests(unittest.TestCase):
    def test_fixed_protocol_matches_stream(self):
        benchmark.validate_modes(benchmark.MODES)
        self.assertEqual(set(benchmark.MODES), {"rf0", "mass_c32", "mass_c16", "partition_c16"})
        self.assertEqual((benchmark.ITERATIONS, benchmark.REPEATS, benchmark.WARMUP), (5, 4, 2))
        self.assertEqual(benchmark.PARTITION_OVERRIDE, {"gs_num_warps": 4, "reuse_owner_logp": True})
        self.assertEqual(len(benchmark.DEFAULT_CAPTURES), 2)

    def test_implicit_rf10_or_extra_method_is_rejected(self):
        for replacement in ("_t1_u16_", "_t1_rf10_u16_", "_t3_rf0_u16_"):
            modes = dict(benchmark.MODES)
            modes["rf0"] = modes["rf0"].replace("_t1_rf0_u16_", replacement)
            with self.assertRaises(ValueError):
                benchmark.validate_modes(modes)
        with self.assertRaises(ValueError):
            benchmark.validate_modes(benchmark.MODES | {"extra": benchmark.MODES["rf0"]})

    def test_summary_and_speed_ratios_use_same_run_means(self):
        timing = {"rf0": benchmark.latency_stats([10, 12]),
                  "mass_c16": benchmark.latency_stats([8, 10]),
                  "sdpa": benchmark.latency_stats([20, 24])}
        result = benchmark.relative_timings(timing)
        self.assertEqual(result["rf0"]["mean_attention_ms"], 11)
        self.assertEqual(result["rf0"]["std_ms"], 1)
        self.assertEqual(result["rf0"]["speedup_vs_sdpa"], 2)
        self.assertAlmostEqual(result["mass_c16"]["ratio_vs_rf0"], 9 / 11)
        self.assertAlmostEqual(result["mass_c16"]["speedup_vs_sdpa"], 22 / 9)
        self.assertNotIn("speedup_vs_sdpa", timing["rf0"])

    def test_bad_timing_samples_are_not_reported(self):
        for samples in ([], [0], [-1], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                benchmark.latency_stats(samples)

    def test_source_hashes_cover_configs_selector_and_merge(self):
        identity = benchmark.provenance()
        self.assertEqual(set(identity["configs"]), set(benchmark.MODES))
        self.assertIn("attention/token_merging.py", identity["source_sha256"])
        self.assertIn("attention/partition_mass_seed_triton.py", identity["source_sha256"])
        self.assertIn("attention/singleton_c4_wan21_14b_720p_triton.py", identity["source_sha256"])
        self.assertTrue(all(len(item) == 64 for item in identity["source_sha256"].values()))

    def test_actual_full_operator_and_forced_flash_are_used_without_rf10_helper(self):
        source = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertNotIn("benchmark_partition_c16_seeds", source)
        self.assertNotIn("benchmark_joint_frame_mass_seeds", source)
        self.assertNotIn("10:11", source)
        self.assertIn("SDPBackend.FLASH_ATTENTION", source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        operator = next(node for node in calls if isinstance(node.func, ast.Attribute)
                        and node.func.attr == "_hard_c4_attention")
        options = {keyword.arg: keyword.value for keyword in operator.keywords}
        self.assertEqual(ast.literal_eval(options["head_chunk"]), 8)
        self.assertEqual(ast.literal_eval(options["query_outer_chunk"]), 21)
        self.assertIs(ast.literal_eval(options["first_frame_full_attention"]), False)
        self.assertEqual(ast.literal_eval(options["full_attention_anchor_time_count"]), 0)
        self.assertIn("samples[name].append(start.elapsed_time(end) / ITERATIONS)", source)
        self.assertIn("error.mean((0, 1, 3)).tolist()", source)


if __name__ == "__main__":
    unittest.main()
