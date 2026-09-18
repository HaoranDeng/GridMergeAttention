"""Contract tests for the causal Wan model and diffusion wrapper.

The causal and bidirectional models execute attention differently but must
retain identical learned parameter names and shapes. That compatibility is
what lets the same Self-Forcing state_dict load strictly into the causal model.
"""

from __future__ import annotations

import inspect
import unittest
from unittest import mock

import torch

from tests._support import assert_state_dict_schemas_match, require_symbol


class CausalModelStateDictTests(unittest.TestCase):
    def test_tiny_causal_and_bidirectional_models_have_identical_state_schema(self) -> None:
        """Changing forward-time causality must not introduce checkpoint parameters."""

        causal_model_class = require_symbol(
            "wan.modules.causal_model", "CausalWanModel"
        )
        bidirectional_model_class = require_symbol(
            "wan.modules.model", "WanModel"
        )
        common = dict(
            model_type="t2v",
            patch_size=(1, 1, 1),
            text_len=4,
            in_dim=2,
            dim=24,
            ffn_dim=48,
            freq_dim=8,
            text_dim=16,
            out_dim=2,
            num_heads=2,
            num_layers=2,
            qk_norm=True,
            cross_attn_norm=True,
            eps=1e-6,
        )
        bidirectional = bidirectional_model_class(window_size=(-1, -1), **common)
        causal = causal_model_class(
            local_attn_size=-1,
            sink_size=0,
            **common,
        )
        assert_state_dict_schemas_match(self, bidirectional, causal)

        # This strict load is the small, fast analogue of loading the real
        # 5.3 GB Self-Forcing checkpoint.
        incompatible = causal.load_state_dict(bidirectional.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_token_merging_configuration_reaches_every_causal_block(self) -> None:
        """Enabling Token Merging on the model must configure all self-attention layers."""

        causal_model_class = require_symbol(
            "wan.modules.causal_model", "CausalWanModel"
        )
        model = causal_model_class(
            model_type="t2v",
            patch_size=(1, 1, 1),
            text_len=4,
            in_dim=2,
            dim=24,
            ffn_dim=48,
            freq_dim=8,
            text_dim=16,
            out_dim=2,
            num_heads=2,
            num_layers=3,
            local_attn_size=-1,
            sink_size=0,
        )
        model.num_frame_per_block = 2
        model.token_merging_args = {
            "enable": True,
            "f_tied": 1,
            "h_reduce": 1,
            "w_reduce": 1,
            "layout": "hard_route",
            "q_init": "ith",
            "num_merge_tokens": 1,
        }

        for block in model.blocks:
            attention = block.self_attn
            self.assertTrue(attention.enable_token_merging)
            self.assertEqual(attention.num_frame_per_block, 2)
            self.assertEqual(attention.token_merging_num_merge_tokens, 1)


class CausalWrapperTests(unittest.TestCase):
    def test_wrapper_selects_causal_model_and_preserves_frame_timesteps(self) -> None:
        """The wrapper must not collapse a causal [B,F] timestep tensor to [B]."""

        wrapper_class = require_symbol(
            "utils.wan_wrapper", "WanDiffusionWrapper"
        )
        wrapper_module = inspect.getmodule(wrapper_class)
        self.assertIn("is_causal", inspect.signature(wrapper_class).parameters)
        causal_model_class = require_symbol(
            "utils.wan_wrapper", "CausalWanModel"
        )

        class RecordingModel(torch.nn.Module):
            patch_size = (1, 1, 1)

            def __init__(self) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.seen_timestep = None

            def forward(self, x, *, t, **kwargs):
                del kwargs
                self.seen_timestep = t.detach().clone()
                return torch.zeros_like(x)

        recording_model = RecordingModel()
        with mock.patch.object(
            causal_model_class,
            "from_pretrained",
            return_value=recording_model,
        ) as from_pretrained:
            wrapper = wrapper_class(
                model_name="synthetic-wan",
                model_root="/does/not/need/to/exist",
                timestep_shift=5.0,
                is_causal=True,
                local_attn_size=-1,
                sink_size=0,
                image_or_video_shape=(2, 3, 2, 1, 1),
            )

        from_pretrained.assert_called_once()
        self.assertFalse(wrapper.uniform_timestep)

        noisy = torch.randn(2, 3, 2, 1, 1)
        timestep = torch.tensor([[1000, 750, 500], [750, 500, 250]])
        wrapper(
            noisy_image_or_video=noisy,
            conditional_dict={"prompt_embeds": torch.zeros(2, 1, 4)},
            timestep=timestep,
            kv_cache=[],
            crossattn_cache=[],
            current_start_frame=0,
        )
        torch.testing.assert_close(recording_model.seen_timestep, timestep)


if __name__ == "__main__":
    unittest.main(verbosity=2)
