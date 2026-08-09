"""Real-checkpoint differential test for the future Hunyuan pipeline.

The test is written before the production integration. Once that code exists,
enable this test to compare its dense-probe output against untouched Diffusers
Hunyuan at every denoising step. The parent can run in ``etm``; both model
subprocesses use a Hunyuan-capable environment such as the existing ``SVG``
Conda environment, which has Diffusers 0.34.0.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch


RUN_PARITY = os.environ.get("RUN_HUNYUAN_PIPELINE_PARITY") == "1"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "tests" / "hunyuan_pipeline_parity_runner.py"
DEFAULT_HUNYUAN_PYTHON = Path("/home/zrlu/miniforge3/envs/SVG/bin/python")
DEFAULT_TEMP_ROOT = Path("/xuanwu-tank/east/zrl")


@unittest.skipUnless(
    RUN_PARITY,
    "set RUN_HUNYUAN_PIPELINE_PARITY=1 after implementing the Hunyuan pipeline",
)
class HunyuanPipelineReferenceParityTests(unittest.TestCase):
    def test_real_checkpoint_dense_probe_matches_diffusers(self) -> None:
        """Compare per-step/final latents using the same real model on CUDA."""

        if not torch.cuda.is_available():
            self.fail("RUN_HUNYUAN_PIPELINE_PARITY=1 requires a CUDA GPU")

        hunyuan_python = Path(
            os.environ.get("HUNYUAN_PYTHON", DEFAULT_HUNYUAN_PYTHON)
        ).resolve()
        self.assertTrue(
            hunyuan_python.is_file(),
            f"missing Hunyuan Python interpreter: {hunyuan_python}",
        )
        self.assertTrue(RUNNER.is_file(), f"missing parity runner: {RUNNER}")
        for required_module in (
            REPO_ROOT / "pipeline" / "token_merging_hunyuan.py",
            REPO_ROOT / "attention" / "hunyuan_token_merging.py",
        ):
            self.assertTrue(
                required_module.is_file(),
                f"implement the Hunyuan production contract before running "
                f"the real checkpoint: missing {required_module}",
            )

        model_source = os.environ.get("HUNYUAN_MODEL")
        self.assertTrue(
            model_source,
            "set HUNYUAN_MODEL to a local checkpoint directory or Hugging Face model ID",
        )
        revision = os.environ.get("HUNYUAN_REVISION", "refs/pr/18")
        transformer_subfolder = os.environ.get(
            "HUNYUAN_TRANSFORMER_SUBFOLDER", "transformer"
        )
        local_files_only = os.environ.get("HUNYUAN_LOCAL_FILES_ONLY") == "1"

        # These tensors are generated once in the parent and replayed in both
        # subprocesses. The dimensions match the original HunyuanVideo model.
        # The spatial/temporal request is intentionally small to keep activation
        # memory low; the full real checkpoint is nevertheless loaded and every
        # transformer layer executes twice.
        fixture_generator = torch.Generator(device="cpu").manual_seed(20260809)
        fixture = {
            "height": 64,
            "width": 64,
            "num_frames": 5,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "latents": torch.randn(
                (1, 16, 2, 8, 8), generator=fixture_generator
            ),
            "prompt_embeds": torch.randn(
                (1, 8, 4096), generator=fixture_generator
            ),
            "pooled_prompt_embeds": torch.randn(
                (1, 768), generator=fixture_generator
            ),
            # Seven valid text tokens plus one padded token exercise the joint
            # attention mask rather than only the all-valid special case.
            "prompt_attention_mask": torch.tensor(
                [[1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.long
            ),
        }

        configured_artifact_dir = os.environ.get(
            "HUNYUAN_PARITY_ARTIFACT_DIR"
        )
        temporary_directory = None
        if configured_artifact_dir:
            artifact_dir = Path(configured_artifact_dir).resolve()
            artifact_dir.mkdir(parents=True, exist_ok=True)
        else:
            temporary_directory = tempfile.TemporaryDirectory(
                prefix="hunyuan_pipeline_parity_",
                dir=DEFAULT_TEMP_ROOT,
            )
            artifact_dir = Path(temporary_directory.name)

        try:
            fixture_path = artifact_dir / "fixture.pt"
            reference_path = artifact_dir / "diffusers_reference.pt"
            current_path = artifact_dir / "current_dense_probe.pt"
            torch.save(fixture, fixture_path)

            environment = os.environ.copy()
            environment_bin = hunyuan_python.parent
            environment["PATH"] = (
                f"{environment_bin}{os.pathsep}{environment.get('PATH', '')}"
            )
            environment["CONDA_PREFIX"] = str(environment_bin.parent)
            environment["CONDA_DEFAULT_ENV"] = environment_bin.parent.name
            environment["PYTHONPATH"] = (
                f"{REPO_ROOT}{os.pathsep}{environment.get('PYTHONPATH', '')}"
            )
            environment.setdefault(
                "HF_HOME", "/xuanwu-tank/east/zrl/huggingface_cache"
            )

            for implementation, output_path in (
                ("reference", reference_path),
                ("current", current_path),
            ):
                command = [
                    str(hunyuan_python),
                    str(RUNNER),
                    "--implementation",
                    implementation,
                    "--fixture",
                    str(fixture_path),
                    "--output",
                    str(output_path),
                    "--model",
                    model_source,
                    "--revision",
                    revision,
                    "--transformer-subfolder",
                    transformer_subfolder,
                ]
                if local_files_only:
                    command.append("--local-files-only")
                print(
                    f"[hunyuan-parity] launching {implementation} with "
                    f"{hunyuan_python} on visible CUDA device(s)",
                    flush=True,
                )
                subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    check=True,
                )

            reference = torch.load(
                reference_path, map_location="cpu", weights_only=False
            )
            current = torch.load(
                current_path, map_location="cpu", weights_only=False
            )

            # Tight parity is meaningful only if model and software are equal.
            self.assertEqual(current["torch_version"], reference["torch_version"])
            self.assertEqual(
                current["diffusers_version"], reference["diffusers_version"]
            )
            self.assertEqual(current["cuda_device"], reference["cuda_device"])
            self.assertEqual(current["imported_svg_modules"], [])
            self.assertEqual(reference["imported_svg_modules"], [])
            self.assertEqual(current["model_source"], reference["model_source"])
            self.assertEqual(current["revision"], reference["revision"])
            self.assertEqual(current["model_config"], reference["model_config"])
            self.assertEqual(
                current["model_fingerprint"]["parameter_count"],
                reference["model_fingerprint"]["parameter_count"],
            )
            for key, current_sample in current["model_fingerprint"]["samples"].items():
                torch.testing.assert_close(
                    current_sample,
                    reference["model_fingerprint"]["samples"][key],
                    rtol=0,
                    atol=0,
                )

            torch.testing.assert_close(
                current["timesteps"], reference["timesteps"], rtol=0, atol=0
            )
            self.assertEqual(tuple(current["latents"].shape), (1, 16, 2, 8, 8))
            self.assertEqual(current["step_latents"].shape[0], 2)
            self.assertEqual(
                current["step_latents"].shape,
                reference["step_latents"].shape,
            )

            config = current["model_config"]
            latent_frames, latent_height, latent_width = fixture["latents"].shape[2:]
            video_tokens = (
                (latent_frames // config["patch_size_t"])
                * (latent_height // config["patch_size"])
                * (latent_width // config["patch_size"])
            )
            text_tokens = fixture["prompt_embeds"].shape[1]
            valid_keys = video_tokens + int(
                fixture["prompt_attention_mask"].sum().item()
            )
            expected_calls = fixture["num_inference_steps"] * (
                config["num_layers"] + config["num_single_layers"]
            )

            calls = current["attention_calls"]
            self.assertEqual(len(calls), expected_calls)
            for call in calls:
                self.assertEqual(call["video_tokens"], video_tokens)
                self.assertEqual(call["text_tokens"], text_tokens)
                self.assertEqual(
                    call["heads"], config["num_attention_heads"]
                )
                self.assertEqual(
                    call["head_dim"], config["attention_head_dim"]
                )
                self.assertEqual(call["valid_key_count"], [valid_keys])
                self.assertEqual(
                    call["mask_shape"],
                    (1, 1, 1, video_tokens + text_tokens),
                )

            processors = current["processors"]
            self.assertEqual(
                len(processors),
                config["num_layers"] + config["num_single_layers"],
            )
            self.assertEqual(
                {item["block_kind"] for item in processors},
                {"dual", "single"},
            )
            for item in processors:
                self.assertNotEqual(item["module"].split(".")[0], "diffusers")

            difference = current["latents"].float() - reference["latents"].float()
            mae = difference.abs().mean().item()
            max_error = difference.abs().max().item()
            rmse = difference.square().mean().sqrt().item()
            cosine = torch.nn.functional.cosine_similarity(
                current["latents"].float().flatten(),
                reference["latents"].float().flatten(),
                dim=0,
            ).item()
            print(
                "[hunyuan-parity] "
                f"shape={tuple(current['latents'].shape)} "
                f"mae={mae:.8g} max={max_error:.8g} "
                f"rmse={rmse:.8g} cosine={cosine:.8g}",
                flush=True,
            )

            max_mae = float(os.environ.get("HUNYUAN_PARITY_MAX_MAE", "1e-3"))
            max_rmse = float(
                os.environ.get("HUNYUAN_PARITY_MAX_RMSE", "2e-3")
            )
            max_absolute_error = float(
                os.environ.get("HUNYUAN_PARITY_MAX_ABS", "2e-2")
            )
            min_cosine = float(
                os.environ.get("HUNYUAN_PARITY_MIN_COSINE", "0.99999")
            )
            self.assertLessEqual(mae, max_mae)
            self.assertLessEqual(rmse, max_rmse)
            self.assertLessEqual(max_error, max_absolute_error)
            self.assertGreaterEqual(cosine, min_cosine)

            rtol = float(os.environ.get("HUNYUAN_PARITY_RTOL", "2e-3"))
            atol = float(os.environ.get("HUNYUAN_PARITY_ATOL", "1e-2"))
            torch.testing.assert_close(
                current["step_latents"].float(),
                reference["step_latents"].float(),
                rtol=rtol,
                atol=atol,
            )
            torch.testing.assert_close(
                current["latents"].float(),
                reference["latents"].float(),
                rtol=rtol,
                atol=atol,
            )
        finally:
            if temporary_directory is not None:
                temporary_directory.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
