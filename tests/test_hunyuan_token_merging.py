"""Fast CPU tests for Hunyuan's joint video/text Token-Merging equations."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from attention.hunyuan_token_merging import hunyuan_token_merging_attn
from attention.token_merging import token_merging_attn


def dense_joint_attention(video_q, video_k, video_v, text_q, text_k, text_v, mask):
    """Independent dense SDPA reference with the same public tensor layout."""

    video_length = video_q.shape[1]
    query = torch.cat((video_q, text_q), dim=1).transpose(1, 2)
    key = torch.cat((video_k, text_k), dim=1).transpose(1, 2)
    value = torch.cat((video_v, text_v), dim=1).transpose(1, 2)
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask[:, None, None, :],
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2)
    return output[:, :video_length], output[:, video_length:]


class HunyuanJointTokenMergingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260809)
        self.batch = 2
        self.frames = 2
        self.height = 2
        self.width = 3
        self.video_length = self.frames * self.height * self.width
        self.text_length = 4
        self.heads = 2
        self.head_dim = 4

        video_shape = (
            self.batch,
            self.video_length,
            self.heads,
            self.head_dim,
        )
        text_shape = (
            self.batch,
            self.text_length,
            self.heads,
            self.head_dim,
        )
        self.video_q = torch.randn(video_shape)
        self.video_k = torch.randn(video_shape)
        self.video_v = torch.randn(video_shape)
        self.text_q = torch.randn(text_shape)
        self.text_k = torch.randn(text_shape)
        self.text_v = torch.randn(text_shape)
        self.mask = torch.ones(
            self.batch,
            self.video_length + self.text_length,
            dtype=torch.bool,
        )
        # Exercise a different number of valid text keys in each batch item.
        self.mask[0, -1] = False
        self.mask[1, -2:] = False

    def call(self, **overrides):
        options = {
            "f_tied": 1,
            "h_reduce": self.height,
            # One key column per block makes GridMerge an exact partition of
            # video keys. This is the non-approximate limiting case.
            "w_reduce": self.width,
            "h": self.height,
            "w": self.width,
            "attention_mask": self.mask[:, None, None, :],
            "q_init": "ith",
            "query_outer_chunk": 2,
            "layout": "hard_route",
            "num_merge_tokens": 1,
        }
        options.update(overrides)
        return hunyuan_token_merging_attn(
            self.video_q,
            self.video_k,
            self.video_v,
            self.text_q,
            self.text_k,
            self.text_v,
            **options,
        )

    def test_one_key_groups_match_dense_joint_attention(self):
        """Exact-limit GridMerge must preserve shared video/text normalization."""

        actual_video, actual_text = self.call()
        expected_video, expected_text = dense_joint_attention(
            self.video_q,
            self.video_k,
            self.video_v,
            self.text_q,
            self.text_k,
            self.text_v,
            self.mask,
        )
        torch.testing.assert_close(
            actual_video, expected_video, rtol=1e-5, atol=2e-6
        )
        torch.testing.assert_close(
            actual_text, expected_text, rtol=1e-5, atol=2e-6
        )

    def test_masked_text_keys_cannot_affect_either_output_stream(self):
        """Padded prompt tokens must be excluded for video and text queries."""

        first_video, first_text = self.call(w_reduce=1, query_outer_chunk=1)

        modified_k = self.text_k.clone()
        modified_v = self.text_v.clone()
        for batch_index in range(self.batch):
            invalid = ~self.mask[batch_index, self.video_length :]
            modified_k[batch_index, invalid] = 1.0e6
            modified_v[batch_index, invalid] = -1.0e6

        second_video, second_text = hunyuan_token_merging_attn(
            self.video_q,
            self.video_k,
            self.video_v,
            self.text_q,
            modified_k,
            modified_v,
            f_tied=1,
            h_reduce=self.height,
            w_reduce=1,
            h=self.height,
            w=self.width,
            attention_mask=self.mask,
            q_init="ith",
            query_outer_chunk=1,
            layout="hard_route",
            num_merge_tokens=1,
        )
        torch.testing.assert_close(first_video, second_video, rtol=0, atol=0)
        torch.testing.assert_close(first_text, second_text, rtol=0, atol=0)

    def test_spatial_padding_is_masked_and_cropped(self):
        """Reduction-factor padding must not become visible attention keys."""

        torch.manual_seed(81)
        height, width = 3, 5
        video_shape = (1, height * width, 2, 4)
        text_shape = (1, 3, 2, 4)
        streams = [torch.randn(video_shape) for _ in range(3)] + [
            torch.randn(text_shape) for _ in range(3)
        ]
        mask = torch.ones(1, height * width + 3, dtype=torch.bool)
        actual_video, actual_text = hunyuan_token_merging_attn(
            *streams,
            f_tied=1,
            # Height is padded from three to four. Width groups still contain
            # one key, so valid tokens retain exact dense semantics.
            h_reduce=2,
            w_reduce=width,
            h=height,
            w=width,
            attention_mask=mask,
            layout="hard_route",
            num_merge_tokens=1,
        )
        expected_video, expected_text = dense_joint_attention(*streams, mask)
        self.assertEqual(actual_video.shape, video_shape)
        torch.testing.assert_close(
            actual_video, expected_video, rtol=1e-5, atol=2e-6
        )
        torch.testing.assert_close(
            actual_text, expected_text, rtol=1e-5, atol=2e-6
        )

    def test_hard_route_c2_c4_c6_exact_partition_matches_dense_joint_attention(self):
        """Hard C2/C4/C6 must share one exact video/text softmax."""

        for num_merge_tokens in (2, 4, 6):
            with self.subTest(num_merge_tokens=num_merge_tokens):
                size = num_merge_tokens
                video_q = torch.zeros(1, size, size, 1, size)
                video_k = torch.zeros_like(video_q)
                for row in range(size):
                    video_q[:, row, :, 0, row] = 1.0
                for column in range(size):
                    video_k[:, :, column, 0, column] = 1.0
                video_q = video_q.reshape(1, size * size, 1, size)
                video_k = video_k.reshape_as(video_q)

                generator = torch.Generator().manual_seed(900 + size)
                video_v = torch.randn(
                    video_q.shape, generator=generator
                )
                text_q = torch.randn(
                    1, 3, 1, size, generator=generator
                )
                text_k = torch.randn(
                    1, 3, 1, size, generator=generator
                )
                text_v = torch.randn(
                    1, 3, 1, size, generator=generator
                )
                mask = torch.ones(1, size * size + 3, dtype=torch.bool)
                mask[:, -1] = False

                actual_video, actual_text = hunyuan_token_merging_attn(
                    video_q,
                    video_k,
                    video_v,
                    text_q,
                    text_k,
                    text_v,
                    f_tied=1,
                    h_reduce=1,
                    w_reduce=1,
                    h=size,
                    w=size,
                    attention_mask=mask,
                    layout="hard_route",
                    num_merge_tokens=num_merge_tokens,
                    seed_mode="random_with_ith",
                    seed=17,
                )
                expected_video, expected_text = dense_joint_attention(
                    video_q,
                    video_k,
                    video_v,
                    text_q,
                    text_k,
                    text_v,
                    mask,
                )
                torch.testing.assert_close(
                    actual_video, expected_video, rtol=1e-5, atol=2e-6
                )
                torch.testing.assert_close(
                    actual_text, expected_text, rtol=1e-5, atol=2e-6
                )

    def test_hard_route_c2_c4_c6_match_wan_core_when_text_is_masked(self):
        """The approximate video-key path must retain the established hard-route math."""

        torch.manual_seed(707)
        height, width = 6, 7
        video_shape = (1, height * width, 2, 4)
        text_shape = (1, 2, 2, 4)
        video_q = torch.randn(video_shape)
        video_k = torch.randn(video_shape)
        video_v = torch.randn(video_shape)
        text_q = torch.randn(text_shape)
        text_k = torch.randn(text_shape)
        text_v = torch.randn(text_shape)
        mask = torch.ones(1, height * width + 2, dtype=torch.bool)
        mask[:, height * width :] = False

        for num_merge_tokens in (2, 4, 6):
            with self.subTest(num_merge_tokens=num_merge_tokens):
                actual, _ = hunyuan_token_merging_attn(
                    video_q,
                    video_k,
                    video_v,
                    text_q,
                    text_k,
                    text_v,
                    f_tied=1,
                    h_reduce=1,
                    w_reduce=1,
                    h=height,
                    w=width,
                    attention_mask=mask,
                    layout="hard_route",
                    num_merge_tokens=num_merge_tokens,
                    seed_mode="random_with_ith",
                    seed=31,
                )
                expected = token_merging_attn(
                    video_q,
                    video_k,
                    video_v,
                    f_tied=1,
                    h_reduce=1,
                    w_reduce=1,
                    h=height,
                    w=width,
                    layout="hard_route",
                    num_merge_tokens=num_merge_tokens,
                    seed_mode="random_with_ith",
                    seed=31,
                )
                torch.testing.assert_close(
                    actual, expected, rtol=1e-5, atol=2e-6
                )

    def test_unsupported_wan_layouts_are_rejected_explicitly(self):
        """Unvalidated Wan-only layouts must not silently run as Hunyuan."""

        with self.assertRaisesRegex(ValueError, "only layout='hard_route'"):
            self.call(layout="hard_route_transpose")
        with self.assertRaisesRegex(ValueError, "only layout='hard_route'"):
            self.call(layout="select_route", num_merge_tokens=2)
        with self.assertRaisesRegex(ValueError, "num_merge_tokens >= 1"):
            self.call(num_merge_tokens=0)

    def test_geometry_and_mask_contracts_fail_early(self):
        """Wrong grids and additive masks would otherwise corrupt token boundaries."""

        with self.assertRaisesRegex(ValueError, "not divisible"):
            self.call(h=4, w=4)
        with self.assertRaisesRegex(TypeError, "boolean key mask"):
            self.call(attention_mask=self.mask.float())


if __name__ == "__main__":
    unittest.main(verbosity=2)
