"""Training-free Video Sparse Attention using FastVideo's official kernel."""

from __future__ import annotations

import math
from functools import lru_cache

import torch


VSA_TILE_SIZE = (4, 4, 4)
VSA_SPARSITY = 0.9
WAN_720P_VIDEO_SHAPE = (21, 45, 80)


@lru_cache(maxsize=8)
def _metadata(video_shape, device_index):
    from fastvideo_kernel import build_vsa_metadata

    return build_vsa_metadata(
        video_shape,
        VSA_TILE_SIZE,
        device=torch.device("cuda", device_index),
    )


def vsa_training_free_attention(
    query,
    key,
    value,
    video_shape=WAN_720P_VIDEO_SHAPE,
    sparsity=VSA_SPARSITY,
):
    """Apply gate-free VSA to head-major ``[B, H, N, D]`` Q/K/V tensors."""
    from fastvideo_kernel import video_sparse_attn

    metadata = _metadata(tuple(video_shape), query.device.index)
    valid_tokens = math.prod(video_shape)
    padded_tokens = (
        metadata["variable_block_sizes"].numel() * metadata["max_block_size"]
    )

    def tile(tensor):
        padded = tensor.new_zeros(
            tensor.size(0), tensor.size(1), padded_tokens, tensor.size(3)
        )
        padded[:, :, metadata["non_pad_index"], :] = tensor[
            :, :, metadata["tile_partition_indices"], :
        ]
        return padded

    blocks = metadata["variable_block_sizes"].numel()
    topk = max(1, min(math.ceil((1.0 - sparsity) * blocks), blocks))
    output = video_sparse_attn(
        tile(query),
        tile(key),
        tile(value),
        metadata["variable_block_sizes"],
        metadata["variable_block_sizes"],
        topk,
        block_size=VSA_TILE_SIZE,
        compress_attn_weight=None,
    )
    untile_index = metadata["non_pad_index"][
        metadata["reverse_tile_partition_indices"]
    ]
    return output[:, :, untile_index[:valid_tokens], :]
