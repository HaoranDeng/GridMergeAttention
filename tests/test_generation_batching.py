"""Tests for the merged prompt batching and resume-safe generation entry point."""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from omegaconf import OmegaConf

from tests._support import import_module_cpu_safely


class MovableComponent:
    def to(self, *args, **kwargs):
        del args, kwargs
        return self


class RecordingPipeline:
    def __init__(self) -> None:
        self.text_encoder = MovableComponent()
        self.generator = MovableComponent()
        self.vae = MovableComponent()
        self.calls = []

    def to(self, *args, **kwargs):
        del args, kwargs
        return self

    def eval(self):
        return self

    def inference(self, noise, text_prompts):
        self.calls.append(
            {
                "noise_shape": tuple(noise.shape),
                "text_prompts": list(text_prompts),
            }
        )
        # Production returns [batch, frames, channels, height, width] in [0, 1].
        return torch.zeros(noise.size(0), 1, 3, 2, 2)


class GenerationBatchingTests(unittest.TestCase):
    def test_main_batches_only_missing_samples_and_preserves_output_mapping(self):
        """Combine prompt batching, skip-existing, and fast atomic video saving."""

        generate = import_module_cpu_safely("generate")
        config = OmegaConf.create(
            {
                "causal": True,
                "image_or_video_shape": [1, 1, 3, 2, 2],
                "token_merging_args": {"enable": True},
            }
        )
        pipeline = RecordingPipeline()

        with tempfile.TemporaryDirectory(prefix="generation_batching_") as tmp:
            output_dir = Path(tmp)
            # Prompt zero/sample one is already complete and must neither enter
            # the model batch nor be saved again.
            (output_dir / "000-01.mp4").write_bytes(b"already generated")
            args = SimpleNamespace(
                config="unused.yaml",
                checkpoint="unused.pt",
                prompt_path="unused.txt",
                output_dir=str(output_dir),
                num_videos=2,
                num_samples=2,
                batch_size=2,
                sample_batch_size=2,
                skip_existing=True,
                prompt_shard_id=0,
                num_prompt_shards=1,
                seed=7,
                use_ema=True,
                fps=16,
                dtype="bf16",
                video_preset="ultrafast",
                video_crf=23,
                video_quality=None,
                vae_batch_decode=False,
                vbench_filenames=False,
                token_merging_w_reduce=None,
                token_merging_query_outer_chunk=None,
            )

            def fake_randn(shape, device, dtype):
                del device
                return torch.zeros(shape, dtype=dtype)

            with mock.patch.object(generate, "parse_args", return_value=args), \
                mock.patch.object(generate.torch.cuda, "is_available", return_value=True), \
                mock.patch.object(generate.OmegaConf, "load", return_value=config), \
                mock.patch.object(
                    generate,
                    "load_prompts",
                    return_value=["first prompt", "second prompt"],
                ), \
                mock.patch.object(generate, "build_pipeline", return_value=pipeline) as build, \
                mock.patch.object(generate, "load_generator_checkpoint") as load_checkpoint, \
                mock.patch.object(generate, "set_seed"), \
                mock.patch.object(generate.torch, "randn", side_effect=fake_randn), \
                mock.patch.object(generate, "save_video_atomic") as save_video:
                generate.main()

            build.assert_called_once_with(config, device=torch.device("cuda"))
            load_checkpoint.assert_called_once_with(pipeline, "unused.pt", True)
            self.assertEqual(
                pipeline.calls,
                [
                    {
                        "noise_shape": (3, 1, 3, 2, 2),
                        "text_prompts": [
                            "first prompt",
                            "second prompt",
                            "second prompt",
                        ],
                    }
                ],
            )

            saved_names = [call.args[0].name for call in save_video.call_args_list]
            self.assertEqual(saved_names, ["000-00.mp4", "001-00.mp4", "001-01.mp4"])
            for call in save_video.call_args_list:
                self.assertEqual(call.kwargs["fps"], 16)
                self.assertEqual(call.kwargs["preset"], "ultrafast")
                self.assertEqual(call.kwargs["crf"], 23)
                self.assertIsNone(call.kwargs["quality"])
                self.assertEqual(call.args[1].dtype, torch.uint8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
