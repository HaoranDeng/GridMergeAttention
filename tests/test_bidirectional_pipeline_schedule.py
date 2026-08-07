"""Regression tests for the merged ordinary-Wan denoising loop.

Remote main added denoising-step-aware hard-route policies. Our branch fixed
the loop so every configured model timestep is evaluated. These tests require
both properties simultaneously: policy state is updated before each model
call, the final timestep is evaluated, and re-noising happens only between
model evaluations.
"""

from __future__ import annotations

import unittest

import torch

from tests._support import require_symbol


class RecordingModel:
    def __init__(self) -> None:
        self.policy_calls = []

    def set_token_merging_denoise_step(
        self, denoise_idx: int, num_denoise_steps: int
    ) -> None:
        self.policy_calls.append((denoise_idx, num_denoise_steps))


class RecordingGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = RecordingModel()
        self.timesteps = []

    def forward(self, noisy_image_or_video, conditional_dict, timestep):
        del conditional_dict
        self.timesteps.append(timestep.detach().cpu().clone())
        prediction = torch.full_like(
            noisy_image_or_video,
            float(timestep.flatten()[0]),
        )
        return torch.zeros_like(prediction), prediction


class RecordingScheduler:
    def __init__(self) -> None:
        self.next_timesteps = []

    def add_noise(self, original_samples, noise, timestep):
        del noise
        self.next_timesteps.append(timestep.detach().cpu().clone())
        return original_samples


class FakeTextEncoder(torch.nn.Module):
    def forward(self, text_prompts):
        return {"prompt_embeds": torch.zeros(len(text_prompts), 1, 1)}


class CloneDecoder:
    def __call__(self, latent):
        # The production VAE returns a distinct decoded tensor. Cloning keeps
        # the pipeline's in-place video normalization from changing latents.
        return latent.clone()


class FakeVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = CloneDecoder()


class BidirectionalPipelineScheduleTests(unittest.TestCase):
    def test_all_steps_update_policy_and_only_transitions_add_noise(self) -> None:
        pipeline_class = require_symbol(
            "pipeline.token_merging_wan", "TokenMergingWanPipeline"
        )

        # Construct only the lightweight inference state. The constructor
        # normally loads Wan, UMT5, and the VAE, which are irrelevant to this
        # control-flow contract.
        pipeline = pipeline_class.__new__(pipeline_class)
        torch.nn.Module.__init__(pipeline)
        pipeline.generator = RecordingGenerator()
        pipeline.scheduler = RecordingScheduler()
        pipeline.text_encoder = FakeTextEncoder()
        pipeline.vae = FakeVAE()
        pipeline.denoising_step_list = torch.tensor([1000, 750, 500, 250])

        noise = torch.zeros(1, 2, 1, 1, 1)
        _, latents = pipeline.inference(
            noise=noise,
            text_prompts=["synthetic"],
            return_latents=True,
        )

        self.assertEqual(
            pipeline.generator.model.policy_calls,
            [(0, 4), (1, 4), (2, 4), (3, 4)],
        )
        self.assertEqual(
            [int(step.unique().item()) for step in pipeline.generator.timesteps],
            [1000, 750, 500, 250],
        )
        self.assertEqual(
            [int(step.unique().item()) for step in pipeline.scheduler.next_timesteps],
            [750, 500, 250],
        )
        torch.testing.assert_close(latents, torch.full_like(noise, 250.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
