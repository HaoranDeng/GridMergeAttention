"""CPU-only checks for the isolated native experiment and timing contract."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("jeff_native_experiment", ROOT / "analysis/experiment_conditional_jeffreys_native.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class NativeExperimentTests(unittest.TestCase):
    def test_fixed_modes_keep_gs_geometry_and_real_rf10_control(self):
        self.assertEqual(experiment.MODES["mass_rf0"], experiment.MODES["jeffreys_rf0"])
        self.assertIn("_rf0_u16_cg45_", experiment.MODES["mass_rf0"])
        self.assertEqual(experiment.MODES["hellinger_rf0"].replace("_rf0_", "_rf10_"),
                         experiment.MODES["hellinger_rf10"])
        for mode in experiment.MODES.values():
            self.assertIn("c32", mode)
            self.assertTrue(mode.endswith("_sq1_qm_fp8score_skv_nl_ml2"))

    def test_no_hybrid_options(self):
        for key in ("exact_intra_frame_attention", "exact_neighbor_window_columns", "exact_query_rows",
                    "first_frame_full_attention", "full_attention_anchor_time_count", "full_attention_anchor_row_count"):
            self.assertFalse(experiment.ATTENTION_OPTIONS[key])
        self.assertEqual(experiment.ATTENTION_OPTIONS["query_outer_chunk"], 21)
        self.assertEqual(experiment.ATTENTION_OPTIONS["head_chunk"], 8)

    def test_actual_seed_layout_unique_and_anchor(self):
        shape = (1, 1, 21, 45, 80, 1, 4)
        anchors = np.arange(45, dtype=np.uint8).reshape(1, 1, 1, 45, 1, 1, 1)
        table = np.broadcast_to((anchors + np.arange(4, dtype=np.uint8)) % 45, shape).copy()
        self.assertTrue(experiment.seed_check(table, 1)["distinct_real_rows_and_anchor"])
        table[..., 1] = table[..., 0]
        with self.assertRaises(ValueError):
            experiment.seed_check(table, 1)

    def test_pool_weighted_by_scalar_count(self):
        rows = [{"numel": 100, "squared_error_sum": 20, "rf0_mse": .1, "other20_frames_mse": .205},
                {"numel": 200, "squared_error_sum": 70, "rf0_mse": .2, "other20_frames_mse": .3575}]
        result = experiment.pool_metrics(rows)
        self.assertEqual(result["mse"], .3)
        self.assertAlmostEqual(result["rf0_mse"], 1/6)
        self.assertEqual(result["numel"], 300)

    def test_gpu_state_and_foreign_process_parsing(self):
        with patch.object(experiment.subprocess, "check_output", return_value="NVIDIA H200, GPU-test, 4, 0\n"):
            self.assertEqual(experiment.gpu_state(3)["memory_mib"], 4)
        with patch.object(experiment.subprocess, "check_output", return_value="234, 123, python\n"):
            self.assertEqual(experiment.gpu_processes("GPU-test")[0]["pid"], 234)
        with patch.object(experiment.subprocess, "check_output", return_value=""):
            self.assertEqual(experiment.gpu_processes("GPU-test"), [])

    def test_timing_covers_complete_call_and_excludes_observer(self):
        text = (ROOT / "analysis/experiment_conditional_jeffreys_native.py").read_text()
        self.assertIn("range(ITERATIONS)", text)
        self.assertIn("output = functions[name]()", text)
        self.assertIn('row["timings"]["sdpa"]["mean_attention_ms"]', text)
        body = text[text.index("for repeat in range(REPEATS)"):text.index("def worker")]
        self.assertNotIn("observed_attention(", body)
        self.assertNotIn("global_normalizer=", body)
        self.assertEqual((experiment.ITERATIONS, experiment.REPEATS, experiment.WARMUP), (5, 6, 3))


if __name__ == "__main__":
    unittest.main()
