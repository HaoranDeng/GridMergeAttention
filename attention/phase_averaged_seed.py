"""RF0-only, analytic temporal-RoPE averaging for real-row seed selection.

Only the first 44 channels (22 interleaved temporal pairs) are transformed.
For each pair, average the *forward relative* rotations a=0,...,T-1, with
omega_m=10000**(-2*m/44). The remaining 84 spatial channels are copied bitwise.
An input already rotated at absolute time b therefore represents virtual
times b,...,b+T-1; b is not applied a second time. No other query frame is read.

This is a synthetic selection surrogate, not the actual queries at other
times. The wrapper returns the original selector's four real row indices;
the original current-frame Q/K/V must still be used by the ordinary merge.
It neither supplies a synthetic merge seed nor modifies ownership or route.

In exact arithmetic, softmax(K @ mean_a(R_a q)) minimizes the mean reverse
KL, mean_a KL(p || softmax(K @ R_a q)), over p on this fixed key support.
That virtual-distribution identity is before the final BF16 cast; it is not
a guarantee for real temporal content, hard C4 partitioning, or video error.

The initial implementation uses FP64 coefficients and temporal products,
then one BF16 cast. It allocates an RF0 output and FP64 temporal temporaries,
not T virtual query frames. Only the tiny analytic coefficient tensors are
cached. This Torch reference has no speed claim and no new Triton kernel.
"""

from functools import lru_cache
from numbers import Integral


TEMPORAL_WIDTH = 44
HEAD_DIM = 128
ROPE_THETA = 10000.0


def _frame_count(temporal_frames):
    if isinstance(temporal_frames, bool) or not isinstance(temporal_frames, Integral):
        raise TypeError("temporal_frames must be a positive integer")
    if temporal_frames < 1:
        raise ValueError("temporal_frames must be positive")
    return int(temporal_frames)


@lru_cache(maxsize=32)
def _torch_phase_coefficients(temporal_frames, device_type, device_index):
    """Private read-only [22] coefficients; no Q, K, or result enters cache."""
    import torch

    frames = _frame_count(temporal_frames)
    device = torch.device(device_type, device_index)
    exponent = -torch.arange(0, TEMPORAL_WIDTH, 2, dtype=torch.float64,
                             device=device) / TEMPORAL_WIDTH
    omega = torch.pow(ROPE_THETA, exponent)
    phase = torch.arange(frames, dtype=torch.float64, device=device)[:, None] * omega
    return phase.cos().mean(dim=0), phase.sin().mean(dim=0)


def phase_averaged_query(q, temporal_frames):
    """Return contiguous BF16 q_eff for original BF16 [B,1,45,80,H,128] q.

    The input is never modified or cached. The time-pair products and sums
    use FP64, while spatial channels are copied without arithmetic. T=1 is
    a bitwise identity copy, including signed zeros. The transform accepts
    CPU tensors for testing; the downstream native selector requires CUDA.
    """
    import torch

    frames = _frame_count(temporal_frames)
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")
    if q.ndim != 6 or tuple(q.shape[1:4]) != (1, 45, 80) or q.shape[-1] != HEAD_DIM:
        raise ValueError("q must be [B,1,45,80,H,128], containing RF0 only")
    if q.shape[0] < 1 or q.shape[-2] < 1:
        raise ValueError("q must have nonempty batch and head dimensions")
    if q.dtype != torch.bfloat16:
        raise TypeError("q must contain original BF16 values")
    with torch.no_grad():
        output = q.clone(memory_format=torch.contiguous_format)
        if frames == 1:
            return output
        cosine, sine = _torch_phase_coefficients(frames, q.device.type, q.device.index)
        pairs = q[..., :TEMPORAL_WIDTH].to(torch.float64).reshape(*q.shape[:-1], 22, 2)
        even, odd = pairs[..., 0], pairs[..., 1]
        output[..., :TEMPORAL_WIDTH:2] = (even * cosine - odd * sine).to(torch.bfloat16)
        output[..., 1:TEMPORAL_WIDTH:2] = (odd * cosine + even * sine).to(torch.bfloat16)
        return output


def phase_averaged_query_numpy(q, temporal_frames):
    """Independent explicit-frame FP64 reference for real arrays [...,128].

    NumPy has no native BF16 contract here: pass exact BF16 values expanded
    to FP32/FP64 and cast the returned FP64 temporal result to BF16 explicitly
    when comparing with Torch. Unlike the production transform, this loops
    over rotations instead of first averaging analytic coefficients. The
    FP64 reduction order can differ at an eventual BF16 rounding boundary.
    """
    import numpy as np

    frames = _frame_count(temporal_frames)
    array = np.asarray(q)
    if array.ndim < 1 or array.shape[-1] != HEAD_DIM:
        raise ValueError("q must end in 128 channels")
    if array.dtype.kind != "f":
        raise TypeError("q must be a real floating-point array")
    output = np.array(array, dtype=np.float64, copy=True, order="C")
    if frames == 1:
        return output
    omega = ROPE_THETA ** (-np.arange(0, TEMPORAL_WIDTH, 2, dtype=np.float64)
                          / TEMPORAL_WIDTH)
    pairs = array[..., :TEMPORAL_WIDTH].astype(np.float64).reshape(*array.shape[:-1], 22, 2)
    total_even = np.zeros_like(pairs[..., 0])
    total_odd = np.zeros_like(pairs[..., 1])
    for relative_frame in range(frames):
        cosine = np.cos(relative_frame * omega)
        sine = np.sin(relative_frame * omega)
        total_even += pairs[..., 0] * cosine - pairs[..., 1] * sine
        total_odd += pairs[..., 1] * cosine + pairs[..., 0] * sine
    output[..., :TEMPORAL_WIDTH:2] = total_even / frames
    output[..., 1:TEMPORAL_WIDTH:2] = total_odd / frames
    return output


def _anchor_selector():
    # Lazy import keeps the mathematical reference usable without CUDA/Triton.
    from .affine_hellinger_seed_triton import fisher_anchor_seed_rows_triton

    return fisher_anchor_seed_rows_triton


def phase_averaged_fisher_seed_rows(q, k, scale=None, **kwargs):
    """Anchor-Fisher on phase-averaged RF0; return its real-row table unchanged.

    T is exactly k.shape[1], not an independent hyperparameter. All options,
    including score_audit, are forwarded unchanged to the existing selector.
    Its matched-row first seed and three distinct real pivots are preserved.
    There is no access to other Q frames, V, merge weights, or dense outputs.
    """
    if q.ndim != 6 or k.ndim != 6:
        raise ValueError("q and k must be [B,F,45,80,H,128]")
    if k.shape[0] != q.shape[0] or tuple(k.shape[2:]) != tuple(q.shape[2:]):
        raise ValueError("q and k must share batch/grid/head/channel shapes")
    frames = _frame_count(k.shape[1])
    q_effective = phase_averaged_query(q, frames)
    return _anchor_selector()(q_effective, k, scale, **kwargs)
