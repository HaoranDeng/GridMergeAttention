"""Fast structural tests for installing Hunyuan Token-Merging processors."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from attention.hunyuan_attention_processor import (
    HunyuanTokenMergingAttnProcessor2_0,
)
from pipeline.token_merging_hunyuan import (
    derive_hunyuan_video_grid,
    install_hunyuan_token_merging_attention,
)


class FakeAttention:
    def __init__(self):
        self.processor = object()

    def set_processor(self, processor):
        self.processor = processor


class FakeBlock:
    def __init__(self):
        self.attn = FakeAttention()


def fake_pipeline():
    transformer = SimpleNamespace(
        config=SimpleNamespace(patch_size=2, patch_size_t=1),
        transformer_blocks=[FakeBlock(), FakeBlock()],
        single_transformer_blocks=[FakeBlock(), FakeBlock(), FakeBlock()],
        # This sentinel represents the separate text refiner. The installer
        # must traverse only the two main block lists above.
        context_embedder=SimpleNamespace(attn=FakeAttention()),
    )
    return SimpleNamespace(
        transformer=transformer,
        vae_scale_factor_spatial=8,
        vae_scale_factor_temporal=4,
    )


class HunyuanPipelineInstallationTests(unittest.TestCase):
    def test_runtime_pixel_request_derives_post_patch_grid(self):
        """480x720x129 maps to the actual 33x30x45 attention-token grid."""

        self.assertEqual(
            derive_hunyuan_video_grid(
                fake_pipeline(), height=480, width=720, num_frames=129
            ),
            (33, 30, 45),
        )

    def test_installer_replaces_both_main_block_families_only(self):
        """Dual and single blocks change; the text-only refiner stays dense."""

        pipe = fake_pipeline()
        text_refiner_processor = pipe.transformer.context_embedder.attn.processor
        install_hunyuan_token_merging_attention(
            pipe,
            {
                "enable": True,
                "height": 480,
                "width": 720,
                "num_frames": 129,
                "f_tied": 1,
                "h_reduce": 1,
                "w_reduce": 1,
                "layout": "hard_route",
                "num_merge_tokens": 1,
                "q_init": "ith",
                "query_outer_chunk": 1,
            },
        )

        dual = pipe.transformer.transformer_blocks
        single = pipe.transformer.single_transformer_blocks
        self.assertTrue(
            all(
                isinstance(block.attn.processor, HunyuanTokenMergingAttnProcessor2_0)
                for block in dual + single
            )
        )
        self.assertEqual(
            [block.attn.processor.block_kind for block in dual],
            ["dual", "dual"],
        )
        self.assertEqual(
            [block.attn.processor.layer_index for block in single],
            [0, 1, 2],
        )
        self.assertIs(
            pipe.transformer.context_embedder.attn.processor,
            text_refiner_processor,
        )
        metadata = pipe._hunyuan_token_merging_config
        self.assertEqual(
            (metadata["grid_f"], metadata["grid_h"], metadata["grid_w"]),
            (33, 30, 45),
        )
        self.assertEqual(len(metadata["installed_processors"]), 5)

    def test_invalid_geometry_and_options_fail_during_installation(self):
        """Configuration mistakes fail before a 13B checkpoint starts denoising."""

        with self.assertRaisesRegex(ValueError, "must be divisible"):
            install_hunyuan_token_merging_attention(
                fake_pipeline(),
                {
                    "enable": True,
                    "height": 481,
                    "width": 720,
                    "num_frames": 129,
                },
            )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            install_hunyuan_token_merging_attention(
                fake_pipeline(),
                {
                    "enable": True,
                    "height": 480,
                    "width": 720,
                    "num_frames": 129,
                    "seed_mode": "random",
                },
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
