"""Tests that the user-facing generation command selects the causal path.

Correct causal model and attention classes are not sufficient if ``generate.py``
still constructs the bidirectional pipeline. These tests cover that final
wiring and the dedicated Self-Forcing configuration used by the shell script.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import unittest
from unittest import mock

import torch
from omegaconf import OmegaConf

from tests._support import import_module_cpu_safely, require_symbol


REPO_ROOT = Path(__file__).resolve().parents[1]


class GenerationRoutingTests(unittest.TestCase):
    def test_build_pipeline_routes_causal_and_bidirectional_configs(self) -> None:
        """``causal: true`` must select the causal Token-Merging pipeline."""

        generate = import_module_cpu_safely("generate")
        build_pipeline = require_symbol("generate", "build_pipeline")
        causal_pipeline = require_symbol(
            "generate", "CausalTokenMergingWanPipeline"
        )
        bidirectional_pipeline = require_symbol(
            "generate", "TokenMergingWanPipeline"
        )
        causal_sentinel = object()
        bidirectional_sentinel = object()

        with mock.patch.object(
            generate,
            causal_pipeline.__name__,
            return_value=causal_sentinel,
        ) as causal_constructor, mock.patch.object(
            generate,
            bidirectional_pipeline.__name__,
            return_value=bidirectional_sentinel,
        ) as bidirectional_constructor:
            selected = build_pipeline(
                OmegaConf.create({"causal": True}),
                device=torch.device("cpu"),
            )
            self.assertIs(selected, causal_sentinel)
            causal_constructor.assert_called_once()
            bidirectional_constructor.assert_not_called()

            selected = build_pipeline(
                OmegaConf.create({"causal": False}),
                device=torch.device("cpu"),
            )
            self.assertIs(selected, bidirectional_sentinel)
            bidirectional_constructor.assert_called_once()

    def test_generation_main_uses_the_routing_function(self) -> None:
        """Defining ``build_pipeline`` is insufficient if main bypasses it."""

        generate = import_module_cpu_safely("generate")
        main_source = inspect.getsource(generate.main)
        self.assertIn("pipeline = build_pipeline(config, device=device)", main_source)
        self.assertNotIn(
            "pipeline = TokenMergingWanPipeline(config, device=device)",
            main_source,
        )

    def test_sample_chunks_preserve_resumed_sample_indices(self) -> None:
        """Prompt batching must not recreate samples already skipped on resume."""

        generate = import_module_cpu_safely("generate")
        chunks = list(
            generate.iter_sample_chunks(
                [
                    (4, "first", [1, 3, 4]),
                    (9, "second", [0]),
                ],
                sample_batch_size=2,
            )
        )
        self.assertEqual(
            chunks,
            [
                (4, "first", [1, 3]),
                (4, "first", [4]),
                (9, "second", [0]),
            ],
        )

    def test_self_forcing_config_declares_all_causal_requirements(self) -> None:
        """The dedicated config must not rely on absent MonarchRT defaults."""

        config_path = (
            REPO_ROOT / "configs" / "token_merging_self_forcing_causal.yaml"
        )
        self.assertTrue(config_path.is_file(), f"missing causal config: {config_path}")
        config = OmegaConf.load(config_path)
        self.assertTrue(config.causal)
        self.assertEqual(config.num_frame_per_block, 3)
        self.assertFalse(config.independent_first_frame)
        self.assertEqual(config.context_noise, 0)
        self.assertEqual(list(config.denoising_step_list), [1000, 750, 500, 250])
        self.assertTrue(config.warp_denoising_step)
        self.assertTrue(config.token_merging_args.enable)
        self.assertEqual(config.token_merging_args.layout, "hard_route")
        self.assertEqual(config.token_merging_args.num_merge_tokens, 1)

    def test_self_forcing_script_uses_causal_config_and_batch_one(self) -> None:
        """Batch-one is the safe default because the full KV cache is large."""

        script_path = (
            REPO_ROOT
            / "scripts"
            / "generate"
            / "generate_token_merging_self_forcing.sh"
        )
        script = script_path.read_text(encoding="utf-8")
        self.assertIn("token_merging_self_forcing_causal.yaml", script)
        self.assertIn("--sample_batch_size", script)
        self.assertRegex(script, r"SAMPLE_BATCH_SIZE[^\n]*:-1")

    def test_wan_14b_50step_configs_share_the_same_schedule(self) -> None:
        """Dense and hard C4 must use the same 50 shifted schedule points."""

        dense = OmegaConf.load(
            REPO_ROOT / "configs" / "dense_wan_14b_720p_50step.yaml"
        )
        hard = OmegaConf.load(
            REPO_ROOT
            / "configs"
            / "token_merging_wan_14b_hard_c4_triton_720p_50step.yaml"
        )
        expected_steps = list(range(1000, 0, -20))

        self.assertEqual(list(dense.denoising_step_list), expected_steps)
        self.assertEqual(list(hard.denoising_step_list), expected_steps)
        self.assertTrue(dense.warp_denoising_step)
        self.assertTrue(hard.warp_denoising_step)
        self.assertEqual(list(dense.image_or_video_shape), [1, 21, 16, 90, 160])
        self.assertEqual(list(hard.image_or_video_shape), [1, 21, 16, 90, 160])
        self.assertFalse(dense.token_merging_args.enable)
        self.assertTrue(hard.token_merging_args.enable)
        self.assertEqual(hard.token_merging_args.hard_route_backend, "hard_c4")


if __name__ == "__main__":
    unittest.main(verbosity=2)
