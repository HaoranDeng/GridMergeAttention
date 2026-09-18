"""Training-free Video Sparse Attention baseline for ordinary Wan.

The sparse kernel and block selection come from FastVideo's official VSA
implementation. A dense-checkpoint model has no learned VSA gate, so this
baseline uses the sparse fine output only (coarse gate 0, fine gate 1).
"""

from __future__ import annotations

import math

import torch


VSA_TRAINING_FREE_BACKEND = "vsa_training_free_all"
VSA_TILE_SIZE = (4, 4, 4)


def _padded_shape(grid_size, tile_size=VSA_TILE_SIZE):
    return tuple(
        math.ceil(int(size) / int(tile)) * int(tile)
        for size, tile in zip(grid_size, tile_size)
    )


def _variable_block_sizes(grid_size, device, tile_size=VSA_TILE_SIZE):
    frames, height, width = (int(x) for x in grid_size)
    tile_t, tile_h, tile_w = (int(x) for x in tile_size)
    padded_t, padded_h, padded_w = _padded_shape(grid_size, tile_size)
    num_t, num_h, num_w = (
        padded_t // tile_t,
        padded_h // tile_h,
        padded_w // tile_w,
    )

    def sizes(length, tile, count):
        result = torch.full((count,), tile, dtype=torch.int32, device=device)
        result[-1] = length - (count - 1) * tile
        return result

    t_sizes = sizes(frames, tile_t, num_t)
    h_sizes = sizes(height, tile_h, num_h)
    w_sizes = sizes(width, tile_w, num_w)
    return (
        t_sizes[:, None, None]
        * h_sizes[None, :, None]
        * w_sizes[None, None, :]
    ).reshape(-1)


def _tile_partition_indices(grid_size, device, tile_size=VSA_TILE_SIZE):
    frames, height, width = (int(x) for x in grid_size)
    tile_t, tile_h, tile_w = (int(x) for x in tile_size)
    indices = torch.arange(
        frames * height * width, dtype=torch.long, device=device
    ).reshape(frames, height, width)
    tiles = []
    for frame in range(0, frames, tile_t):
        for row in range(0, height, tile_h):
            for column in range(0, width, tile_w):
                tiles.append(
                    indices[
                        frame : frame + tile_t,
                        row : row + tile_h,
                        column : column + tile_w,
                    ].flatten()
                )
    return torch.cat(tiles)


def _non_pad_indices(block_sizes, block_elements):
    starts = torch.arange(
        block_sizes.numel(), dtype=torch.long, device=block_sizes.device
    ) * int(block_elements)
    offsets = torch.arange(
        int(block_elements), dtype=torch.long, device=block_sizes.device
    )
    valid = offsets[None, :] < block_sizes[:, None]
    return (starts[:, None] + offsets[None, :])[valid]


def tile_video_qkv(x, grid_size, tile_size=VSA_TILE_SIZE):
    """Convert ``[B,S,H,D]`` into VSA's contiguous ``[B,H,S_pad,D]``."""

    frames, height, width = (int(v) for v in grid_size)
    tile_t, tile_h, tile_w = (int(v) for v in tile_size)
    batch, sequence, heads, dim = x.shape
    if sequence != frames * height * width:
        raise ValueError("grid_size does not match the QKV sequence length.")
    block_sizes = _variable_block_sizes(grid_size, x.device, tile_size)
    block_elements = tile_t * tile_h * tile_w
    padded_sequence = int(block_sizes.numel()) * block_elements
    padded = torch.zeros(
        (batch, padded_sequence, heads, dim),
        dtype=x.dtype,
        device=x.device,
    )
    tile_indices = _tile_partition_indices(grid_size, x.device, tile_size)
    non_pad_indices = _non_pad_indices(block_sizes, block_elements)
    padded[:, non_pad_indices] = x[:, tile_indices]
    return padded.transpose(1, 2).contiguous()


def untile_video_output(x, grid_size, tile_size=VSA_TILE_SIZE):
    """Invert :func:`tile_video_qkv` and crop VSA padding."""

    frames, height, width = (int(v) for v in grid_size)
    tile_t, tile_h, tile_w = (int(v) for v in tile_size)
    batch, heads, sequence, dim = x.shape
    block_sizes = _variable_block_sizes(grid_size, x.device, tile_size)
    block_elements = tile_t * tile_h * tile_w
    if sequence != int(block_sizes.numel()) * block_elements:
        raise ValueError("VSA output length does not match its padded grid.")
    non_pad_indices = _non_pad_indices(block_sizes, block_elements)
    tile_indices = _tile_partition_indices(grid_size, x.device, tile_size)
    reverse_tile_indices = torch.argsort(tile_indices)
    tiled = x.transpose(1, 2).contiguous()[:, non_pad_indices]
    return tiled[:, reverse_tile_indices]


def vsa_training_free_attention(
    query,
    key,
    value,
    grid_size,
    *,
    sparsity=0.9,
    head_chunk=1,
):
    """Run official VSA block selection/kernel without trained gate weights."""

    try:
        from fastvideo_kernel import video_sparse_attn
    except ImportError as error:
        raise RuntimeError(
            "VSA needs FastVideo's official fastvideo-kernel package on PYTHONPATH."
        ) from error

    sparsity = float(sparsity)
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("VSA sparsity must be in [0, 1).")
    batch, _, heads, dim = query.shape
    output = torch.empty_like(query)
    block_sizes = _variable_block_sizes(grid_size, query.device)
    num_blocks = int(block_sizes.numel())
    topk = max(1, min(math.ceil((1.0 - sparsity) * num_blocks), num_blocks))
    head_chunk = max(1, int(head_chunk))

    for head_start in range(0, heads, head_chunk):
        head_end = min(head_start + head_chunk, heads)
        q_tiled = tile_video_qkv(query[:, :, head_start:head_end], grid_size)
        k_tiled = tile_video_qkv(key[:, :, head_start:head_end], grid_size)
        v_tiled = tile_video_qkv(value[:, :, head_start:head_end], grid_size)
        # Dense checkpoints have no learned coarse gate. Passing zero exactly
        # selects VSA's fine sparse branch while retaining official routing.
        coarse_gate = torch.zeros_like(q_tiled)
        sparse = video_sparse_attn(
            q_tiled,
            k_tiled,
            v_tiled,
            block_sizes,
            block_sizes,
            topk,
            block_size=VSA_TILE_SIZE,
            compress_attn_weight=coarse_gate,
        )
        output[:, :, head_start:head_end] = untile_video_output(
            sparse, grid_size
        )
    return output


__all__ = [
    "VSA_TRAINING_FREE_BACKEND",
    "tile_video_qkv",
    "untile_video_output",
    "vsa_training_free_attention",
]
