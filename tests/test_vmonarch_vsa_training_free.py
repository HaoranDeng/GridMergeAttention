import unittest

import torch
import torch.nn.functional as F

from attention.vmonarch_wan import vmonarch_attention, vmonarch_attention_reference
from attention.vsa_training_free_wan import tile_video_qkv, untile_video_output


class VMonarchTrainingFreeTests(unittest.TestCase):
    def test_single_frame_matches_dense_attention(self):
        torch.manual_seed(0)
        query = torch.randn(1, 3 * 4, 2, 8)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        actual = vmonarch_attention_reference(
            query,
            key,
            value,
            (1, 3, 4),
            iterations=2,
            first_frame_full_attention=False,
        )
        expected = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
        ).transpose(1, 2)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    def test_first_frame_recomputation_overwrites_only_first_frame(self):
        torch.manual_seed(1)
        query = torch.randn(1, 2 * 3 * 4, 1, 8)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        baseline = vmonarch_attention_reference(
            query,
            key,
            value,
            (2, 3, 4),
            iterations=2,
            first_frame_full_attention=False,
        )
        recomputed = vmonarch_attention_reference(
            query,
            key,
            value,
            (2, 3, 4),
            iterations=2,
            first_frame_full_attention=True,
        )
        dense_first = F.scaled_dot_product_attention(
            query[:, :12].transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
        ).transpose(1, 2)
        torch.testing.assert_close(recomputed[:, :12], dense_first)
        torch.testing.assert_close(recomputed[:, 12:], baseline[:, 12:])

    @unittest.skipUnless(torch.cuda.is_available(), "requires a GPU")
    def test_triton_matches_reference(self):
        torch.manual_seed(2)
        query = torch.randn(
            1, 2 * 2 * 3, 1, 32, device="cuda", dtype=torch.bfloat16
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        expected = vmonarch_attention_reference(
            query,
            key,
            value,
            (2, 2, 3),
            iterations=2,
            first_frame_full_attention=True,
        )
        actual = vmonarch_attention(
            query,
            key,
            value,
            (2, 2, 3),
            iterations=2,
            head_chunk=1,
            first_frame_full_attention=True,
        )
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


class VSATrainingFreeTests(unittest.TestCase):
    def test_tile_roundtrip_preserves_nondivisible_grid(self):
        values = torch.arange(3 * 5 * 6 * 2 * 3, dtype=torch.float32).reshape(
            1, 3 * 5 * 6, 2, 3
        )
        tiled = tile_video_qkv(values, (3, 5, 6))
        self.assertEqual(tuple(tiled.shape), (1, 2, 4 * 8 * 8, 3))
        restored = untile_video_output(tiled, (3, 5, 6))
        torch.testing.assert_close(restored, values)


if __name__ == "__main__":
    unittest.main(verbosity=2)
