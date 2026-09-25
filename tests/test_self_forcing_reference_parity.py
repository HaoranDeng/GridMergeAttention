"""Opt-in differential test against MonarchRT's original Self-Forcing path.

Unlike the Token-Merging integration test, this test disables both accelerated
attention algorithms and asks a narrower question: does this repository's
causal pipeline produce the same latents as MonarchRT when checkpoint, prompt
context, initial noise, re-noising tensors, schedule, and attention primitive
are controlled?

The two implementations run in separate subprocesses to prevent their
top-level ``wan``, ``pipeline``, and ``utils`` packages from colliding.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch


RUN_PARITY = os.environ.get("RUN_SELF_FORCING_PARITY") == "1"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "tests" / "reference_parity_runner.py"
DEFAULT_MONARCH_ROOT = Path("/xuanwu-tank/east/zrl/MonarchRT")
DEFAULT_MONARCH_PYTHON = Path(
    "/home/zrlu/miniforge3/envs/monarch_rt/bin/python"
)


@unittest.skipUnless(
    RUN_PARITY,
    "set RUN_SELF_FORCING_PARITY=1 to compare against MonarchRT",
)
class SelfForcingReferenceParityTests(unittest.TestCase):
    def test_dense_causal_latents_match_monarchrt(self) -> None:
        """Compare two causal blocks with all stochastic inputs replayed exactly."""

        if not torch.cuda.is_available():
            self.fail("RUN_SELF_FORCING_PARITY=1 requires a CUDA GPU")

        monarch_root = Path(
            os.environ.get("MONARCHRT_ROOT", DEFAULT_MONARCH_ROOT)
        ).resolve()
        checkpoint = Path(
            os.environ.get(
                "SELF_FORCING_CHECKPOINT",
                REPO_ROOT / "checkpoints" / "self_forcing_dmd.pt",
            )
        ).resolve()
        monarch_python = Path(
            os.environ.get("MONARCHRT_PYTHON", DEFAULT_MONARCH_PYTHON)
        ).resolve()
        self.assertTrue(monarch_root.is_dir(), f"missing MonarchRT: {monarch_root}")
        self.assertTrue(checkpoint.is_file(), f"missing checkpoint: {checkpoint}")
        self.assertTrue(
            monarch_python.is_file(),
            f"missing MonarchRT Python interpreter: {monarch_python}",
        )

        num_frames = int(os.environ.get("SELF_FORCING_PARITY_FRAMES", "6"))
        block_frames = 3
        self.assertGreater(num_frames, 0)
        self.assertEqual(num_frames % block_frames, 0)

        # Generate all stochastic inputs once on CPU. Replaying tensors is
        # stronger than resetting a seed because CUDA RNG implementations can
        # differ across PyTorch/CUDA versions.
        fixture_generator = torch.Generator(device="cpu").manual_seed(20260806)
        initial_noise = torch.randn(
            (1, num_frames, 16, 60, 104),
            generator=fixture_generator,
            dtype=torch.float32,
        ).to(torch.bfloat16)
        prompt_embeds = torch.randn(
            (1, 512, 4096),
            generator=fixture_generator,
            dtype=torch.float32,
        ).to(torch.bfloat16)
        num_transitions = (num_frames // block_frames) * 3
        transition_noises = torch.randn(
            (num_transitions, block_frames, 16, 60, 104),
            generator=fixture_generator,
            dtype=torch.float32,
        ).to(torch.bfloat16)

        configured_artifact_dir = os.environ.get(
            "SELF_FORCING_PARITY_ARTIFACT_DIR"
        )
        temporary_directory = None
        if configured_artifact_dir:
            artifact_dir = Path(configured_artifact_dir).resolve()
            artifact_dir.mkdir(parents=True, exist_ok=True)
        else:
            temporary_directory = tempfile.TemporaryDirectory(
                prefix="self_forcing_parity_"
            )
            artifact_dir = Path(temporary_directory.name)

        try:
            fixture_path = artifact_dir / "fixture.pt"
            current_path = artifact_dir / "current_latents.pt"
            reference_path = artifact_dir / "monarchrt_latents.pt"
            torch.save(
                {
                    "initial_noise": initial_noise,
                    "prompt_embeds": prompt_embeds,
                    "transition_noises": transition_noises,
                    "num_frames": num_frames,
                    "num_frame_per_block": block_frames,
                },
                fixture_path,
            )

            environment = os.environ.copy()
            environment.pop("TOKEN_MERGING_FORCE_TORCH_RMSNORM", None)
            commands = [
                (
                    "current",
                    Path(sys.executable),
                    REPO_ROOT,
                    current_path,
                ),
                (
                    "monarchrt",
                    monarch_python,
                    monarch_root,
                    reference_path,
                ),
            ]
            for (
                implementation,
                python_executable,
                working_directory,
                output_path,
            ) in commands:
                command = [
                    str(python_executable),
                    str(RUNNER),
                    "--implementation",
                    implementation,
                    "--fixture",
                    str(fixture_path),
                    "--output",
                    str(output_path),
                    "--checkpoint",
                    str(checkpoint),
                    "--monarch-root",
                    str(monarch_root),
                ]
                print(
                    f"[parity] launching {implementation} with "
                    f"{python_executable} in {working_directory}",
                    flush=True,
                )
                process_environment = environment.copy()
                environment_bin = python_executable.parent
                process_environment["PATH"] = (
                    f"{environment_bin}{os.pathsep}"
                    f"{process_environment.get('PATH', '')}"
                )
                process_environment["CONDA_PREFIX"] = str(
                    environment_bin.parent
                )
                process_environment["CONDA_DEFAULT_ENV"] = (
                    environment_bin.parent.name
                )
                subprocess.run(
                    command,
                    cwd=working_directory,
                    env=process_environment,
                    check=True,
                )

            current = torch.load(
                current_path, map_location="cpu", weights_only=False
            )
            reference = torch.load(
                reference_path, map_location="cpu", weights_only=False
            )
            current_latents = current["latents"].float()
            reference_latents = reference["latents"].float()
            self.assertEqual(current_latents.shape, reference_latents.shape)

            difference = current_latents - reference_latents
            mae = difference.abs().mean().item()
            max_error = difference.abs().max().item()
            rmse = difference.square().mean().sqrt().item()
            cosine = torch.nn.functional.cosine_similarity(
                current_latents.flatten(),
                reference_latents.flatten(),
                dim=0,
            ).item()
            print(
                "[parity] "
                f"current_torch={current['torch_version']} "
                f"reference_torch={reference['torch_version']} "
                f"shape={tuple(current_latents.shape)} "
                f"mae={mae:.8g} max={max_error:.8g} "
                f"rmse={rmse:.8g} cosine={cosine:.8g}",
                flush=True,
            )

            # The implementations run in their real Conda environments:
            # current uses torch/CUDA 2.13/13.0 and MonarchRT uses
            # 2.8/12.8. BF16 kernel rounding is therefore expected even after
            # controlling every stochastic tensor and the attention API. Use
            # aggregate bounds to catch schedule/model divergence and a
            # separate maximum bound to catch localized corruption.
            max_mae = float(
                os.environ.get("SELF_FORCING_PARITY_MAX_MAE", "1e-2")
            )
            max_rmse = float(
                os.environ.get("SELF_FORCING_PARITY_MAX_RMSE", "1.25e-2")
            )
            max_absolute_error = float(
                os.environ.get("SELF_FORCING_PARITY_MAX_ABS", "6.25e-2")
            )
            min_cosine = float(
                os.environ.get("SELF_FORCING_PARITY_MIN_COSINE", "0.9995")
            )
            self.assertLessEqual(mae, max_mae)
            self.assertLessEqual(rmse, max_rmse)
            self.assertLessEqual(max_error, max_absolute_error)
            self.assertGreaterEqual(cosine, min_cosine)

            expected_end = num_frames * 30 * 52
            for payload in (current, reference):
                self.assertEqual(
                    payload["replay_noise_calls"], num_transitions
                )
                self.assertEqual(
                    payload["global_end_indices"], [expected_end] * 30
                )
                self.assertEqual(
                    payload["local_end_indices"], [expected_end] * 30
                )
            torch.testing.assert_close(
                current["denoising_steps"].float(),
                reference["denoising_steps"].float(),
                rtol=0,
                atol=0,
            )

            rtol = float(os.environ.get("SELF_FORCING_PARITY_RTOL", "1e-2"))
            atol = float(os.environ.get("SELF_FORCING_PARITY_ATOL", "5e-2"))
            torch.testing.assert_close(
                current_latents,
                reference_latents,
                rtol=rtol,
                atol=atol,
            )
        finally:
            if temporary_directory is not None:
                temporary_directory.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
