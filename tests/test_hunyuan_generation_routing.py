"""CPU-only test of the Hunyuan generation entry point and file routing."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
import yaml

import generate_hunyuan


class FakeHunyuanPipeline:
    def __init__(self):
        self.calls = []
        self._hunyuan_token_merging_config = {
            "grid_f": 2,
            "grid_h": 2,
            "grid_w": 3,
            "installed_processors": [{}] * 4,
        }

    def set_progress_bar_config(self, **unused):
        return None

    def __call__(self, **kwargs):
        self.calls.append(
            {
                **kwargs,
                "generator_seed": kwargs["generator"].initial_seed(),
            }
        )
        frames = np.zeros((2, 4, 6, 3), dtype=np.uint8)
        return SimpleNamespace(frames=[frames])


class HunyuanGenerationRoutingTests(unittest.TestCase):
    def test_main_passes_geometry_and_resumes_individual_samples(self):
        """The CLI must install for its run geometry and atomically save missing outputs."""

        with tempfile.TemporaryDirectory(prefix="hunyuan_generation_test_") as root:
            root = Path(root)
            config_path = root / "config.yaml"
            prompt_path = root / "prompts.txt"
            output_dir = root / "videos"
            timing_path = root / "timing.json"
            output_dir.mkdir()
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "model": {
                            "id": "unused/model",
                            "dtype": "bf16",
                            "revision": "test-revision",
                            "scheduler_shift": 7.0,
                            "model_cpu_offload": True,
                        },
                        "generation": {
                            "height": 32,
                            "width": 48,
                            "num_frames": 5,
                            "num_inference_steps": 2,
                            "guidance_scale": 6.0,
                            "true_cfg_scale": 1.0,
                            "fps": 12,
                        },
                        "token_merging_args": {
                            "enable": True,
                            "f_tied": 1,
                            "h_reduce": 1,
                            "w_reduce": 1,
                            "layout": "hard_route",
                            "num_merge_tokens": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            prompt_path.write_text("first prompt\nsecond prompt\n", encoding="utf-8")
            # This one completed sample should be preserved and skipped.
            (output_dir / "000-00.mp4").write_bytes(b"existing")

            fake_pipe = FakeHunyuanPipeline()
            loader_calls = []

            def fake_loader(model_id, **kwargs):
                loader_calls.append((model_id, kwargs))
                return fake_pipe

            exported = []

            def fake_export(frames, path, fps):
                exported.append((tuple(frames.shape), Path(path).name, fps))
                Path(path).write_bytes(b"video")

            argv = [
                "generate_hunyuan.py",
                "--config",
                str(config_path),
                "--prompt_path",
                str(prompt_path),
                "--output_dir",
                str(output_dir),
                "--num_videos",
                "2",
                "--num_samples",
                "2",
                "--seed",
                "10",
                "--skip_existing",
                "--token_merging_w_reduce",
                "3",
                "--timing_output",
                str(timing_path),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(torch.cuda, "is_available", return_value=True),
                mock.patch.object(
                    generate_hunyuan,
                    "load_hunyuan_token_merging_pipeline",
                    side_effect=fake_loader,
                ),
                mock.patch("diffusers.utils.export_to_video", side_effect=fake_export),
            ):
                generate_hunyuan.main()

            self.assertEqual(len(loader_calls), 1)
            model_id, loader_options = loader_calls[0]
            self.assertEqual(model_id, "unused/model")
            token_options = loader_options["token_merging_args"]
            self.assertEqual(
                (token_options["height"], token_options["width"], token_options["num_frames"]),
                (32, 48, 5),
            )
            self.assertEqual(token_options["w_reduce"], 3)
            self.assertTrue(loader_options["model_cpu_offload"])

            self.assertEqual(len(fake_pipe.calls), 3)
            self.assertEqual(
                [call["generator_seed"] for call in fake_pipe.calls],
                [11, 12, 13],
            )
            self.assertTrue(all(call["output_type"] == "np" for call in fake_pipe.calls))
            self.assertEqual(len(exported), 3)
            self.assertTrue(all(item[2] == 12 for item in exported))
            self.assertEqual((output_dir / "000-00.mp4").read_bytes(), b"existing")
            for filename in ("000-01.mp4", "001-00.mp4", "001-01.mp4"):
                self.assertEqual((output_dir / filename).read_bytes(), b"video")
            self.assertEqual(list(output_dir.glob(".*.partial.mp4")), [])

            timing = json.loads(timing_path.read_text(encoding="utf-8"))
            self.assertEqual(timing["schema_version"], 1)
            self.assertEqual(timing["generated_videos"], 3)
            self.assertEqual(timing["skipped_existing_videos"], 1)
            self.assertEqual(len(timing["samples"]), 3)
            self.assertGreaterEqual(
                timing["pipeline_generation_seconds_total"], 0.0
            )
            self.assertAlmostEqual(
                timing["pipeline_generation_seconds_mean"],
                timing["pipeline_generation_seconds_total"] / 3,
            )
            self.assertGreaterEqual(timing["video_export_seconds_total"], 0.0)
            self.assertGreaterEqual(timing["model_load_seconds"], 0.0)
            self.assertFalse((root / ".timing.json.partial").exists())

    def test_vbench_prompts_are_deduplicated_and_sharded(self):
        """Each prompt stays on one worker and uses VBench-compatible filenames."""

        with tempfile.TemporaryDirectory(prefix="hunyuan_sharding_test_") as root:
            root = Path(root)
            config_path = root / "config.yaml"
            prompt_path = root / "prompts.txt"
            output_dir = root / "videos"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "model": {"dtype": "bf16"},
                        "generation": {
                            "height": 32,
                            "width": 48,
                            "num_frames": 5,
                            "num_inference_steps": 2,
                        },
                        "token_merging_args": {
                            "enable": True,
                            "layout": "hard_route",
                            "num_merge_tokens": 2,
                            "seed_mode": "random_with_ith",
                        },
                    }
                ),
                encoding="utf-8",
            )
            prompt_path.write_text("alpha\nbeta\nalpha\n", encoding="utf-8")
            fake_pipe = FakeHunyuanPipeline()

            def fake_export(unused_frames, path, fps=None):
                del unused_frames, fps
                Path(path).write_bytes(b"video")

            argv = [
                "generate_hunyuan.py",
                "--config",
                str(config_path),
                "--prompt_path",
                str(prompt_path),
                "--output_dir",
                str(output_dir),
                "--num_videos",
                "-1",
                "--num_samples",
                "1",
                "--num_prompt_shards",
                "2",
                "--prompt_shard_id",
                "1",
                "--vbench_filenames",
                "--seed",
                "20",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(torch.cuda, "is_available", return_value=True),
                mock.patch.object(
                    generate_hunyuan,
                    "load_hunyuan_token_merging_pipeline",
                    return_value=fake_pipe,
                ),
                mock.patch("diffusers.utils.export_to_video", side_effect=fake_export),
            ):
                generate_hunyuan.main()

            self.assertEqual(len(fake_pipe.calls), 1)
            self.assertEqual(fake_pipe.calls[0]["prompt"], "beta")
            self.assertEqual(fake_pipe.calls[0]["generator_seed"], 21)
            self.assertEqual((output_dir / "beta-0.mp4").read_bytes(), b"video")
            self.assertFalse((output_dir / "alpha-0.mp4").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
