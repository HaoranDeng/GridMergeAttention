"""Behavioral tests for the Self-Forcing temporal generation loop.

The fake components make this a fast CPU test and record every model and
scheduler call. The test catches full-video generation, skipped final
timesteps, an out-of-range ``index + 1``, missing clean-cache commits, and
incorrect ``current_start_frame`` updates without loading Wan weights.
"""

from __future__ import annotations

import unittest

import torch
from omegaconf import OmegaConf

from tests._support import require_symbol


class RecordingScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.arange(1000, -1, -1, dtype=torch.float32)
        self.add_noise_calls = []

    def add_noise(self, original_samples, noise, timestep):
        del noise
        self.add_noise_calls.append(timestep.detach().cpu().clone())
        # Returning x0 keeps this mock deterministic; the test is concerned
        # with control flow, not the scheduler's already-tested mathematics.
        return original_samples


class FakeGeneratorModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_layers = 2
        self.num_heads = 2
        self.dim = 8
        self.text_len = 4
        self.patch_size = (1, 1, 1)
        self.local_attn_size = -1
        self.num_frame_per_block = 1
        self.token_merging_args = {}


class RecordingGenerator(torch.nn.Module):
    def __init__(self, scheduler) -> None:
        super().__init__()
        self.model = FakeGeneratorModel()
        self.scheduler = scheduler
        self.calls = []

    def get_scheduler(self):
        return self.scheduler

    def forward(
        self,
        noisy_image_or_video,
        conditional_dict,
        timestep,
        kv_cache=None,
        crossattn_cache=None,
        current_start_frame=None,
    ):
        del conditional_dict
        prediction = torch.full_like(
            noisy_image_or_video,
            float(current_start_frame + 1),
        )
        self.calls.append(
            {
                "input": noisy_image_or_video.detach().clone(),
                "prediction": prediction.detach().clone(),
                "timestep": timestep.detach().cpu().clone(),
                "start_frame": current_start_frame,
                "frames": noisy_image_or_video.size(1),
                "kv_cache": kv_cache,
                "crossattn_cache": crossattn_cache,
            }
        )
        return torch.zeros_like(noisy_image_or_video), prediction


class FakeTextEncoder(torch.nn.Module):
    def forward(self, text_prompts):
        return {
            "prompt_embeds": torch.zeros(len(text_prompts), 4, 8),
        }


class IdentityDecoder:
    def __call__(self, latent, **kwargs):
        del kwargs
        return latent


class FakeVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = IdentityDecoder()


def make_config(batch_size: int):
    return OmegaConf.create(
        {
            "causal": True,
            "image_or_video_shape": [batch_size, 21, 1, 1, 1],
            "denoising_step_list": [1000, 750, 500, 250],
            "warp_denoising_step": False,
            "num_frame_per_block": 3,
            "independent_first_frame": False,
            "context_noise": 0,
            "model_kwargs": {},
            "token_merging_args": {
                "enable": True,
                "f_tied": 1,
                "h_reduce": 1,
                "w_reduce": 1,
                "layout": "hard_route",
                "q_init": "ith",
                "num_merge_tokens": 1,
            },
        }
    )


def make_pipeline(batch_size: int = 1):
    pipeline_class = require_symbol(
        "pipeline.causal_token_merging_wan",
        "CausalTokenMergingWanPipeline",
    )
    scheduler = RecordingScheduler()
    generator = RecordingGenerator(scheduler)
    try:
        pipeline = pipeline_class(
            make_config(batch_size),
            device=torch.device("cpu"),
            generator=generator,
            text_encoder=FakeTextEncoder(),
            vae=FakeVAE(),
        )
    except TypeError as exc:
        raise AssertionError(
            "CausalTokenMergingWanPipeline should accept optional generator, "
            "text_encoder, and vae dependencies so its scheduling logic can "
            "be tested without loading multi-gigabyte weights."
        ) from exc
    return pipeline, generator, scheduler


class CausalPipelineScheduleTests(unittest.TestCase):
    def test_seven_blocks_run_four_steps_and_one_clean_commit_each(self) -> None:
        """A 21-frame/3-frame-block run must make exactly 35 generator calls."""

        pipeline, generator, scheduler = make_pipeline()
        noise = torch.zeros(1, 21, 1, 1, 1)
        _, latents = pipeline.inference(
            noise=noise,
            text_prompts=["synthetic prompt"],
            return_latents=True,
        )

        self.assertEqual(len(generator.calls), 7 * 5)
        self.assertEqual(len(scheduler.add_noise_calls), 7 * 3)
        expected_steps = [1000, 750, 500, 250]

        for block_index in range(7):
            calls = generator.calls[block_index * 5 : (block_index + 1) * 5]
            expected_start = block_index * 3
            self.assertEqual([call["start_frame"] for call in calls], [expected_start] * 5)
            self.assertEqual([call["frames"] for call in calls], [3] * 5)
            self.assertEqual(
                [int(call["timestep"].unique().item()) for call in calls[:4]],
                expected_steps,
            )
            self.assertEqual(int(calls[4]["timestep"].unique().item()), 0)

            # The fifth call commits the clean prediction produced by the
            # fourth denoising call into the same cache positions.
            torch.testing.assert_close(calls[4]["input"], calls[3]["prediction"])
            self.assertIsNotNone(calls[4]["kv_cache"])
            self.assertIsNotNone(calls[4]["crossattn_cache"])

        self.assertEqual(latents.shape, noise.shape)
        for block_index in range(7):
            expected_value = float(block_index * 3 + 1)
            torch.testing.assert_close(
                latents[:, block_index * 3 : (block_index + 1) * 3],
                torch.full((1, 3, 1, 1, 1), expected_value),
            )

    def test_cache_is_reallocated_when_sample_batch_size_changes(self) -> None:
        """A smaller last sample batch must not reuse caches with the old batch dimension."""

        pipeline, _, _ = make_pipeline(batch_size=2)
        pipeline.inference(
            noise=torch.zeros(2, 21, 1, 1, 1),
            text_prompts=["a", "b"],
        )
        pipeline.inference(
            noise=torch.zeros(1, 21, 1, 1, 1),
            text_prompts=["c"],
        )

        kv_cache = getattr(pipeline, "kv_cache", None)
        if kv_cache is None:
            kv_cache = getattr(pipeline, "kv_cache1", None)
        self.assertIsNotNone(kv_cache, "The causal pipeline must own a per-layer KV cache.")
        for layer_cache in kv_cache:
            self.assertEqual(layer_cache["k"].size(0), 1)
            self.assertEqual(layer_cache["v"].size(0), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

