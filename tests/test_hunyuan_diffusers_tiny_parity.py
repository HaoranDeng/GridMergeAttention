"""Optional whole-transformer parity test using Diffusers' tiny Hunyuan model.

Run this in an environment with Diffusers 0.34.0. It needs no checkpoint and
no GPU, but it exercises real dual-stream blocks, real single-stream blocks,
the text refiner, QK normalization, RoPE, and the final transformer output.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
import unittest

import torch


RUN_TINY_PARITY = os.environ.get("RUN_HUNYUAN_TINY_PARITY") == "1"


@unittest.skipUnless(
    RUN_TINY_PARITY,
    "set RUN_HUNYUAN_TINY_PARITY=1 in a Diffusers 0.34 environment",
)
class HunyuanDiffusersTinyParityTests(unittest.TestCase):
    def test_exact_partition_matches_untouched_diffusers_transformer(self):
        """Our processor must preserve the full model when no key is merged."""

        try:
            from diffusers import HunyuanVideoTransformer3DModel
        except ImportError as error:
            self.fail(
                "RUN_HUNYUAN_TINY_PARITY=1 requires Diffusers 0.34.0: "
                f"{error}"
            )

        from pipeline.token_merging_hunyuan import (
            install_hunyuan_token_merging_attention,
        )

        torch.manual_seed(20260809)
        model_options = {
            "in_channels": 4,
            "out_channels": 4,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "num_layers": 2,
            "num_single_layers": 2,
            "num_refiner_layers": 1,
            "mlp_ratio": 2.0,
            "patch_size": 2,
            "patch_size_t": 1,
            "qk_norm": "rms_norm",
            "guidance_embeds": True,
            "text_embed_dim": 16,
            "pooled_projection_dim": 8,
            "rope_axes_dim": (2, 2, 4),
        }
        reference = HunyuanVideoTransformer3DModel(**model_options).eval()
        current = HunyuanVideoTransformer3DModel(**model_options).eval()
        current.load_state_dict(reference.state_dict(), strict=True)

        pipe = SimpleNamespace(
            transformer=current,
            vae_scale_factor_spatial=1,
            vae_scale_factor_temporal=1,
        )
        install_hunyuan_token_merging_attention(
            pipe,
            {
                "enable": True,
                "height": 4,
                "width": 6,
                "num_frames": 2,
                "f_tied": 1,
                # The post-patch grid is 2x3. Making every GridMerge key group
                # one token removes approximation while retaining our complete
                # production processor and core execution path.
                "h_reduce": 2,
                "w_reduce": 3,
                "layout": "hard_route",
                "num_merge_tokens": 1,
                "q_init": "ith",
                "query_outer_chunk": 2,
            },
        )

        inputs = {
            "hidden_states": torch.randn(1, 4, 2, 4, 6),
            "timestep": torch.tensor([500.0]),
            "encoder_hidden_states": torch.randn(1, 4, 16),
            "encoder_attention_mask": torch.tensor([[1, 1, 1, 0]]),
            "pooled_projections": torch.randn(1, 8),
            "guidance": torch.tensor([6000.0]),
            "return_dict": False,
        }
        with torch.inference_mode():
            expected = reference(**inputs)[0]
            actual = current(**inputs)[0]

        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)

    def test_hard_c2_c4_c6_execute_real_diffusers_blocks(self):
        """Every requested hard-route C value must execute the real model."""

        try:
            from diffusers import HunyuanVideoTransformer3DModel
        except ImportError as error:
            self.fail(
                "RUN_HUNYUAN_TINY_PARITY=1 requires Diffusers 0.34.0: "
                f"{error}"
            )

        from pipeline.token_merging_hunyuan import (
            install_hunyuan_token_merging_attention,
        )

        torch.manual_seed(20260810)
        model = HunyuanVideoTransformer3DModel(
            in_channels=4,
            out_channels=4,
            num_attention_heads=1,
            attention_head_dim=8,
            num_layers=1,
            num_single_layers=1,
            num_refiner_layers=1,
            mlp_ratio=2.0,
            patch_size=2,
            patch_size_t=1,
            qk_norm="rms_norm",
            guidance_embeds=True,
            text_embed_dim=16,
            pooled_projection_dim=8,
            rope_axes_dim=(2, 2, 4),
        ).eval()
        pipe = SimpleNamespace(
            transformer=model,
            vae_scale_factor_spatial=1,
            vae_scale_factor_temporal=1,
        )
        inputs = {
            # A 12x12 input and 2x2 patches produce a 6x6 video grid, so both
            # the query-row and key-column dimensions can hold all six seeds.
            "hidden_states": torch.randn(1, 4, 1, 12, 12),
            "timestep": torch.tensor([500.0]),
            "encoder_hidden_states": torch.randn(1, 4, 16),
            "encoder_attention_mask": torch.tensor([[1, 1, 1, 0]]),
            "pooled_projections": torch.randn(1, 8),
            "guidance": torch.tensor([6000.0]),
            "return_dict": False,
        }

        for merge_tokens in (2, 4, 6):
            with self.subTest(num_merge_tokens=merge_tokens):
                install_hunyuan_token_merging_attention(
                    pipe,
                    {
                        "enable": True,
                        "height": 12,
                        "width": 12,
                        "num_frames": 1,
                        "f_tied": 1,
                        "h_reduce": 1,
                        "w_reduce": 1,
                        "layout": "hard_route",
                        "num_merge_tokens": merge_tokens,
                        "seed_mode": "random_with_ith",
                        "seed": 0,
                        "q_init": "ith",
                        "query_outer_chunk": 1,
                    },
                )
                with torch.inference_mode():
                    output = model(**inputs)[0]

                self.assertEqual(output.shape, inputs["hidden_states"].shape)
                self.assertTrue(torch.isfinite(output).all())
                processors = [
                    block.attn.processor
                    for block in (
                        list(model.transformer_blocks)
                        + list(model.single_transformer_blocks)
                    )
                ]
                self.assertTrue(processors)
                self.assertTrue(
                    all(
                        processor.num_merge_tokens == merge_tokens
                        for processor in processors
                    )
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
