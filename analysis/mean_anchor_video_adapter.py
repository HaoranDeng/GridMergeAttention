"""Explicit current-frame mean-query merge binding, isolated from defaults.

The parent calls begin_attention/end_attention around one complete GCA call.
All-frame/all-head means are computed once, then reused only for that call's
five head chunks. No data-pointer-based cache survives the explicit boundary.
"""
from __future__ import annotations

from unittest.mock import patch

from analysis.zero_anchor_video_adapter import (
    GRID_SHAPE, KERNEL_NAME, SEED_SHAPE, ZeroSeedMergeAdapter,
    _TrackedZeroKernel, _reject_overrides,
)


def current_frame_means(q):
    """Use the same FP32-reduction/BF16-output rule as the RF0 selector."""
    from attention.affine_hellinger_seed_triton import mean_query_bf16
    return mean_query_bf16(q)


def _signature(tensor):
    """Storage metadata only; no tensor value reads or CUDA synchronization."""
    return (tensor.data_ptr(), tensor.storage_offset(), tuple(tensor.stride()),
            tuple(tensor.shape), str(tensor.dtype), str(tensor.device))


class _TrackedMeanKernel:
    def __init__(self, owner, means):
        self.owner, self.means = owner, means
        self.attempts = 0
        self.forwarded = 0

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.attempts += 1
            if self.attempts != 1:
                raise RuntimeError("expected exactly one temporal shared merge kernel launch")
            _reject_overrides(kwargs)
            self.forwarded += 1
            self.owner.kernel_calls += 1
            return self.owner._kernel[grid](*args, **kwargs,
                MEAN_SEED_ZERO=True, mean_seed_ptr=self.means)
        return launch


class MeanSeedMergeAdapter:
    """Current-frame mean seed plus three real rows; original merge and route.

    begin_attention() resets the means even if Q reuses a previous data_ptr.
    The five successful calls must have head_start 0,8,16,24,32 and count8.
    end_attention() verifies completion; it does not compute extra attention.
    means is a read-only pointer for metadata, not a reusable cross-call cache.
    total_calls/kernel_calls count accepted merge/forwarded native attempts;
    mean_computations counts actual per-attention mean constructions.

    The dummy seed row in slot0 may repeat a real row. The adapter never
    changes Q/K/V, the row table, entropy/SQ1 options, or route computation.
    """
    def __init__(self, original_merge, merge_module):
        if not callable(original_merge) or isinstance(original_merge, (MeanSeedMergeAdapter, ZeroSeedMergeAdapter)):
            raise TypeError("require the original merge callable, not a second synthetic adapter")
        kernel = getattr(merge_module, KERNEL_NAME)
        if isinstance(kernel, (_TrackedMeanKernel, _TrackedZeroKernel)) or hasattr(kernel, "mean_queries"):
            raise ValueError("merge kernel already has a synthetic seed override")
        if not hasattr(kernel, "__getitem__"):
            raise TypeError("require the original launchable temporal shared merge kernel")
        self.original_merge, self.merge_module, self._kernel = original_merge, merge_module, kernel
        self._attention_active = False
        self._in_merge = False
        self._chunk = 0
        self._means = None
        self._inputs = None
        self.total_calls = 0
        self.kernel_calls = 0
        self.mean_computations = 0
        self.completed_attention_calls = 0

    @property
    def means(self):
        return self._means

    def begin_attention(self):
        if self._attention_active or self._in_merge:
            raise RuntimeError("previous mean-query attention call is not complete")
        self._attention_active = True
        self._chunk = 0
        self._means = None
        self._inputs = None

    def end_attention(self):
        if not self._attention_active or self._in_merge or self._chunk != 5 or self._means is None:
            raise RuntimeError("mean-query attention requires exactly five ordered head chunks")
        self.completed_attention_calls += 1
        self._attention_active = False

    def __call__(self, q, k, v, seedtable, scale, **kwargs):
        if not self._attention_active:
            raise RuntimeError("begin_attention must precede mean-query merge")
        if self._in_merge or getattr(self.merge_module, KERNEL_NAME) is not self._kernel:
            raise RuntimeError("mean-query merge forbids nested or changed kernel overrides")
        _reject_overrides(kwargs)
        if self._chunk >= 5 or kwargs.get("head_start") != self._chunk * 8 or kwargs.get("head_count") != 8:
            raise ValueError("require ordered head_start 0,8,16,24,32 with head_count8")
        if scale != 128 ** -0.5:
            raise ValueError("mean-query video merge requires the original attention scale")
        for tensor in (q, k, v):
            if tuple(tensor.shape) != GRID_SHAPE or str(tensor.dtype) != "torch.bfloat16":
                raise ValueError("require original BF16 Q/K/V [1,21,45,80,40,128]")
            if tensor.device != q.device or not tensor.is_cuda:
                raise ValueError("Q/K/V must share the original CUDA device")
        if (tuple(seedtable.shape) != SEED_SHAPE or str(seedtable.dtype) != "torch.uint8"
                or seedtable.device != q.device or not seedtable.is_cuda):
            raise ValueError("require native uint8 RF0 seed table on the Q/K/V device")
        received = tuple(_signature(tensor) for tensor in (q, k, v, seedtable))
        if self._inputs is None:
            self._inputs = received
        elif received != self._inputs:
            raise ValueError("Q/K/V or seed-table storage changed within one attention call")
        if self._means is None:
            means = current_frame_means(q)
            if (tuple(means.shape) != (1, 21, 80, 40, 128) or means.dtype != q.dtype
                    or means.device != q.device or not means.is_contiguous()):
                raise ValueError("row mean must preserve the BF16/device/current-frame contract")
            self._means = means
            self.mean_computations += 1

        tracked = _TrackedMeanKernel(self, self._means)
        self.total_calls += 1
        self._in_merge = True
        try:
            with patch.object(self.merge_module, KERNEL_NAME, tracked):
                result = self.original_merge(q, k, v, seedtable, scale, **kwargs)
                if tracked.attempts != 1 or tracked.forwarded != 1:
                    raise RuntimeError("expected exactly one temporal shared merge kernel launch")
                self._chunk += 1
                return result
        finally:
            self._in_merge = False


__all__ = ["MeanSeedMergeAdapter", "current_frame_means"]
