"""Temporal entropy-barycenter routers in a common RoPE coordinate frame.

Wan's ``wan/modules/model.py`` uses interleaved rotary pairs and concatenates
temporal, height, and width frequencies.  Its temporal rotary width is
``D - 4 * (D // 6)``: 44 channels / 22 pairs for D=128.  Pair m rotates by
``a * 10000**(-2*m/temporal_width)`` at frame a.

Captured Q is already rotated.  Applying the relative temporal rotation
``R(reference-a)`` puts each frame's Q into the reference frame's positional
coordinates before averaging.  Spatial rotary channels are unchanged by this
transport.  The resulting mean is the reverse-KL barycenter query of these
*transported* distributions; it is not the barycenter of the original,
untransported distributions.

This module only builds router Q.  Selected indices must still load the real
current-frame Q in the existing GCA merge, as in ordinary RF10 routing.
"""

from __future__ import annotations

import json

try:
    import torch
except ModuleNotFoundError:
    torch = None


def _temporal_width(head_dim: int) -> int:
    width = head_dim - 4 * (head_dim // 6)
    if head_dim < 2 or head_dim % 2 or width % 2:
        raise ValueError("Wan rotary head dimension and temporal width must be even")
    return width


def align_temporal_queries(
    query: torch.Tensor,
    reference_frame: int | None = None,
) -> torch.Tensor:
    """Transport post-RoPE ``query[head,time,row,column,dim]`` to one time.

    The returned tensor has the same shape and uses float32 accumulation
    (float64 for float64 input).  No temporal averaging is done here.  Only
    temporal rotary pairs change; the remaining spatial channels are copied.
    Relative rotation also handles a nonzero video ``start_frame`` because
    the common absolute start cancels from ``reference_frame - frame``.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for temporal query transport")
    if query.ndim != 5:
        raise ValueError("query must have shape [head,time,row,column,dim]")
    frames = query.shape[1]
    if frames < 1 or not query.is_floating_point():
        raise ValueError("query must be floating-point with at least one frame")
    reference_frame = frames // 2 if reference_frame is None else int(reference_frame)
    if not 0 <= reference_frame < frames:
        raise ValueError("reference_frame is outside the query sequence")
    width = _temporal_width(query.shape[-1])
    dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    work = query.to(dtype=dtype)
    offsets = reference_frame - torch.arange(frames, device=query.device, dtype=dtype)
    exponents = torch.arange(0, width, 2, device=query.device, dtype=dtype) / width
    frequency = torch.pow(10000.0, -exponents)
    angle = (offsets[:, None] * frequency[None, :])[None, :, None, None, :]
    cosine, sine = angle.cos(), angle.sin()
    even = work[..., :width:2]
    odd = work[..., 1:width:2]
    rotated = torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine),
        dim=-1,
    ).flatten(-2)
    return torch.cat((rotated, work[..., width:]), dim=-1)


@torch.no_grad() if torch is not None else (lambda function: function)
def temporal_transport_router(
    query: torch.Tensor,
    mode: str = "mean21",
    reference_frame: int | None = None,
) -> torch.Tensor:
    """Return ``[head,1,row,column,dim]`` router Q in the input dtype.

    ``mean21`` averages every available frame (21 in the canonical capture).
    ``simpson3`` uses fixed endpoint/midpoint weights 1:4:1.  For Simpson's
    midpoint rule the frame count must be odd.  Neither mode adds probe QK
    products or changes the downstream seed-row table dimensions.
    """
    if mode not in {"mean21", "simpson3"}:
        raise ValueError("mode must be 'mean21' or 'simpson3'")
    if query.ndim != 5:
        raise ValueError("query must have shape [head,time,row,column,dim]")
    frames = query.shape[1]
    if mode == "simpson3" and (frames < 3 or frames % 2 == 0):
        raise ValueError("simpson3 requires an odd frame count of at least three")
    aligned = align_temporal_queries(query, reference_frame)
    if mode == "mean21":
        router = aligned.mean(dim=1, keepdim=True)
    else:
        middle = frames // 2
        router = (
            aligned[:, :1]
            + 4.0 * aligned[:, middle:middle + 1]
            + aligned[:, -1:]
        ) / 6.0
    return router.to(dtype=query.dtype)


def _numpy_self_check() -> dict[str, float | int]:
    """Check transport sign, pair order, width, and quadrature on the CPU."""
    import numpy as np

    generator = np.random.default_rng(4410)
    heads, frames, rows, columns, head_dim = 2, 21, 3, 4, 128
    width = _temporal_width(head_dim)
    assert width == 44
    base = generator.normal(size=(heads, 1, rows, columns, head_dim))
    frequency = 10000.0 ** (-np.arange(0, width, 2) / width)
    # Include a nonzero absolute start to verify cancellation of that offset.
    absolute_frames = 7 + np.arange(frames)
    angle = (absolute_frames[:, None] * frequency)[None, :, None, None, :]
    cosine, sine = np.cos(angle), np.sin(angle)
    even, odd = base[..., :width:2], base[..., 1:width:2]
    temporal = np.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), axis=-1
    ).reshape(heads, frames, rows, columns, width)
    spatial = np.broadcast_to(base[..., width:], (heads, frames, rows, columns, head_dim-width))
    captured = np.concatenate((temporal, spatial), axis=-1)
    reference = captured[:, 10:11]
    relative_angle = ((10 - np.arange(frames))[:, None] * frequency)[None, :, None, None, :]
    cosine, sine = np.cos(relative_angle), np.sin(relative_angle)
    even, odd = captured[..., :width:2], captured[..., 1:width:2]
    transported = np.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), axis=-1
    ).reshape(heads, frames, rows, columns, width)
    aligned = np.concatenate((transported, captured[..., width:]), axis=-1)
    assert np.array_equal(aligned[..., width:], captured[..., width:])
    mean = aligned.mean(axis=1, keepdims=True)
    simpson = (aligned[:, :1] + 4.0 * aligned[:, 10:11] + aligned[:, -1:]) / 6.0
    errors = {
        "temporal_width": width,
        "transport_max_abs_error": float(np.abs(aligned - reference).max()),
        "mean21_max_abs_error": float(np.abs(mean - reference).max()),
        "simpson3_max_abs_error": float(np.abs(simpson - reference).max()),
    }
    assert max(value for key, value in errors.items() if key != "temporal_width") < 1.0e-12
    if torch is not None:
        tensor = torch.from_numpy(captured)
        target = torch.from_numpy(reference)
        transported_tensor = align_temporal_queries(tensor, 10)
        torch.testing.assert_close(transported_tensor, torch.from_numpy(aligned), rtol=0.0, atol=1.0e-12)
        for mode in ("mean21", "simpson3"):
            result = temporal_transport_router(tensor, mode, 10)
            torch.testing.assert_close(result, target, rtol=0.0, atol=1.0e-12)
            errors[f"torch_{mode}_max_abs_error"] = float((result - target).abs().max())
    return errors


if __name__ == "__main__":
    print(json.dumps(_numpy_self_check(), indent=2))
