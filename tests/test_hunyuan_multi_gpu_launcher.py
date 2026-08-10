"""Static and configuration tests for the Hunyuan hard-route launcher."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    REPO_ROOT
    / "scripts"
    / "generate"
    / "launch_hunyuan_hard_configs_multi_gpu.sh"
)
CONFIGS = [
    REPO_ROOT / f"configs/token_merging_hunyuan_hard_route_c{count}_480p.yaml"
    for count in (2, 4, 6)
]


class HunyuanMultiGpuLauncherTests(unittest.TestCase):
    def test_launcher_contains_only_hard_c2_c4_c6(self):
        """The requested launcher must never start an unsupported select run."""

        source = LAUNCHER.read_text(encoding="utf-8")
        for count in (2, 4, 6):
            self.assertIn(
                f"token_merging_hunyuan_hard_route_c{count}_480p.yaml",
                source,
            )
            self.assertIn(f'"hard_c{count}"', source)
        self.assertNotIn("select_route", source)
        self.assertNotIn('"select_c', source)

        self.assertIn("generate_hunyuan.py", source)
        self.assertIn("--prompt_shard_id", source)
        self.assertIn("--num_prompt_shards", source)
        self.assertIn("--vbench_filenames", source)
        self.assertIn("--skip_existing", source)
        self.assertNotIn("--checkpoint", source)
        self.assertNotIn("--use_ema", source)

    def test_all_three_configs_are_matching_hard_route_experiments(self):
        """Only the merge-token count should distinguish C2, C4, and C6."""

        loaded = []
        for path in CONFIGS:
            self.assertTrue(path.is_file(), f"missing config: {path}")
            loaded.append(yaml.safe_load(path.read_text(encoding="utf-8")))

        for expected_count, config in zip((2, 4, 6), loaded):
            token = config["token_merging_args"]
            self.assertTrue(token["enable"])
            self.assertEqual(token["layout"], "hard_route")
            self.assertEqual(token["num_merge_tokens"], expected_count)
            self.assertEqual(token["seed_mode"], "random_with_ith")
            self.assertEqual(token["query_outer_chunk"], 1)
            self.assertEqual(
                config["generation"], loaded[0]["generation"]
            )
            self.assertEqual(config["model"], loaded[0]["model"])

    def test_launcher_requires_explicit_reserved_gpu_ids(self):
        """It must fail safely instead of assuming GPUs are unoccupied."""

        environment = os.environ.copy()
        environment.pop("GPU_IDS", None)
        result = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Set GPU_IDS", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
