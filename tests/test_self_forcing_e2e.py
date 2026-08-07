"""Opt-in end-to-end test using the real Wan and Self-Forcing checkpoint.

This test is intentionally disabled by default because it needs a GPU and
loads the full Wan text encoder, generator, VAE, and 5.3 GB checkpoint. Enable
it with ``RUN_SELF_FORCING_E2E=1`` after the fast CPU contract tests pass.
"""

from __future__ import annotations

import collections
import importlib
import os
from pathlib import Path
import unittest
from unittest import mock

import torch
from omegaconf import OmegaConf

from tests._support import require_symbol


RUN_E2E = os.environ.get("RUN_SELF_FORCING_E2E") == "1"


@unittest.skipUnless(RUN_E2E, "set RUN_SELF_FORCING_E2E=1 to run the real GPU test")
class RealSelfForcingE2ETests(unittest.TestCase):
    def test_real_generation_uses_current_block_q_and_growing_kv_prefix(self) -> None:
        """Prove the real run is both causal Self-Forcing and Token Merging."""

        if not torch.cuda.is_available():
            self.fail("RUN_SELF_FORCING_E2E=1 requires a CUDA/ROCm GPU.")

        repo_root = Path(__file__).resolve().parents[1]
        config_path = Path(
            os.environ.get(
                "SELF_FORCING_CONFIG",
                repo_root / "configs" / "token_merging_self_forcing_causal.yaml",
            )
        )
        checkpoint_path = Path(
            os.environ.get(
                "SELF_FORCING_CHECKPOINT",
                repo_root / "checkpoints" / "self_forcing_dmd.pt",
            )
        )
        self.assertTrue(config_path.is_file(), f"missing causal config: {config_path}")
        self.assertTrue(checkpoint_path.is_file(), f"missing checkpoint: {checkpoint_path}")

        pipeline_class = require_symbol(
            "pipeline.causal_token_merging_wan",
            "CausalTokenMergingWanPipeline",
        )
        causal_model_class = require_symbol(
            "wan.modules.causal_model", "CausalWanModel"
        )
        causal_module = importlib.import_module("wan.modules.causal_model")
        original_cache_attention = require_symbol(
            "wan.modules.causal_model", "token_merging_attn_with_kv_cache"
        )

        config = OmegaConf.load(config_path)
        self.assertTrue(config.causal)
        self.assertEqual(config.num_frame_per_block, 3)
        self.assertTrue(config.token_merging_args.enable)
        self.assertEqual(config.token_merging_args.num_merge_tokens, 1)

        print("[e2e] constructing the real causal pipeline", flush=True)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        pipeline = pipeline_class(config, device=device)
        print("[e2e] loading generator_ema strictly", flush=True)
        state = torch.load(checkpoint_path, map_location="cpu")
        state = state["generator_ema"]
        incompatible = pipeline.generator.load_state_dict(state, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertIsInstance(pipeline.generator.model, causal_model_class)

        print("[e2e] moving Wan, UMT5, and VAE to the GPU", flush=True)
        pipeline = pipeline.to(dtype=dtype)
        pipeline.text_encoder.to(device=device)
        pipeline.generator.to(device=device)
        pipeline.vae.to(device=device)
        pipeline.eval()

        frame_tokens = 30 * 52
        trace = []

        def traced_cache_attention(
            q,
            k_cache,
            v_cache,
            new_k,
            new_v,
            start_idx,
            end_idx,
            *args,
            **kwargs,
        ):
            pair = (q.size(1) // frame_tokens, end_idx // frame_tokens)
            if not trace or trace[-1] != pair:
                print(
                    f"[e2e] cached Token Merging: Q={pair[0]} frames, "
                    f"visible K/V={pair[1]} frames",
                    flush=True,
                )
            trace.append(pair)
            return original_cache_attention(
                q,
                k_cache,
                v_cache,
                new_k,
                new_v,
                start_idx,
                end_idx,
                *args,
                **kwargs,
            )

        torch.manual_seed(0)
        noise = torch.randn(
            list(config.image_or_video_shape),
            device=device,
            dtype=dtype,
        )
        print("[e2e] starting seven-block Self-Forcing generation", flush=True)
        # Pass the tracing function as ``new`` rather than as a Mock
        # ``side_effect``. MagicMock records every call and would retain the
        # large Q/K/V tensor arguments for all 1,050 calls, eventually causing
        # an instrumentation-induced GPU OOM.
        with mock.patch.object(
            causal_module,
            "token_merging_attn_with_kv_cache",
            new=traced_cache_attention,
        ):
            video, latents = pipeline.inference(
                noise=noise,
                text_prompts=["A red car driving along a mountain road."],
                return_latents=True,
            )

        self.assertEqual(tuple(latents.shape), (1, 21, 16, 60, 104))
        self.assertEqual(video.size(0), 1)
        self.assertEqual(video.size(1), 81)
        self.assertTrue(torch.isfinite(latents).all())
        self.assertTrue(torch.isfinite(video).all())

        num_layers = pipeline.generator.model.num_layers
        counts = collections.Counter(trace)
        self.assertEqual(len(trace), 7 * 5 * num_layers)
        for visible_frames in (3, 6, 9, 12, 15, 18, 21):
            self.assertEqual(
                counts[(3, visible_frames)],
                5 * num_layers,
                msg=f"wrong causal Token-Merging count at prefix {visible_frames}",
            )

        kv_cache = getattr(pipeline, "kv_cache", None)
        if kv_cache is None:
            kv_cache = getattr(pipeline, "kv_cache1", None)
        self.assertIsNotNone(kv_cache)
        expected_end = 21 * frame_tokens
        for layer_cache in kv_cache:
            self.assertEqual(layer_cache["global_end_index"], expected_end)
            self.assertEqual(layer_cache["local_end_index"], expected_end)


if __name__ == "__main__":
    unittest.main(verbosity=2)
