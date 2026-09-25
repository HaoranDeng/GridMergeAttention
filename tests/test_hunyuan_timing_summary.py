"""CPU-only tests for aggregating Hunyuan multi-GPU timing reports."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER = (
    REPO_ROOT / "scripts" / "generate" / "summarize_hunyuan_timings.py"
)


def worker_report(
    *,
    generated,
    generation_total,
    export_total,
    process_start,
    process_finish,
    work_start,
    work_finish,
    model_load,
):
    """Construct the minimum valid worker report used by the CLI test."""

    return {
        "schema_version": 1,
        "generated_videos": generated,
        "pipeline_generation_seconds_total": generation_total,
        "video_export_seconds_total": export_total,
        "model_load_seconds": model_load,
        "process_started_unix": process_start,
        "process_finished_unix": process_finish,
        "work_started_unix": work_start,
        "work_finished_unix": work_finish,
        "samples": [{} for _ in range(generated)],
    }


class HunyuanTimingSummaryTests(unittest.TestCase):
    def test_weighted_latency_and_parallel_throughput_are_distinct(self):
        """Average call latency and actual multi-GPU throughput need both metrics."""

        with tempfile.TemporaryDirectory(prefix="hunyuan_timing_test_") as root:
            root = Path(root)
            report_paths = [root / "worker0.json", root / "worker1.json"]
            report_paths[0].write_text(
                json.dumps(
                    worker_report(
                        generated=2,
                        generation_total=20.0,
                        export_total=2.0,
                        process_start=100.0,
                        process_finish=140.0,
                        work_start=110.0,
                        work_finish=132.0,
                        model_load=10.0,
                    )
                ),
                encoding="utf-8",
            )
            report_paths[1].write_text(
                json.dumps(
                    worker_report(
                        generated=1,
                        generation_total=8.0,
                        export_total=1.0,
                        process_start=101.0,
                        process_finish=130.0,
                        work_start=111.0,
                        work_finish=120.0,
                        model_load=9.0,
                    )
                ),
                encoding="utf-8",
            )
            summary_path = root / "summary.json"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SUMMARIZER),
                    "--label",
                    "hard_c2",
                    "--output",
                    str(summary_path),
                    *(str(path) for path in report_paths),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Average pipeline generation latency", result.stdout)
            self.assertIn("Effective multi-GPU throughput", result.stdout)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["generated_videos"], 3)
            self.assertEqual(summary["worker_count"], 2)
            self.assertAlmostEqual(
                summary["average_pipeline_generation_seconds_per_video"],
                28.0 / 3.0,
            )
            self.assertAlmostEqual(
                summary["effective_multi_gpu_work_wall_seconds"], 22.0
            )
            self.assertAlmostEqual(
                summary["effective_multi_gpu_seconds_per_video"], 22.0 / 3.0
            )
            self.assertAlmostEqual(
                summary["worker_process_wall_seconds_including_model_load"],
                40.0,
            )
            self.assertFalse((root / ".summary.json.partial").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
