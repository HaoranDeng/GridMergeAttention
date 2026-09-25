"""Isolated zero-query merge binding for the standard BF16 video experiment.

Only slot-zero seed Q changes. The original merge receives the identical
Q/K/V tensors, native row table, scale and options, and still executes its
original full compressed route. No production dispatch is enabled here.
"""
from __future__ import annotations

from unittest.mock import patch


KERNEL_NAME = "_hard_c4_720p_temporal_shared_merge_kernel"
GRID_SHAPE = (1, 21, 45, 80, 40, 128)
SEED_SHAPE = (1, 1, 21, 45, 80, 40, 4)


def _reject_overrides(options):
    for name, value in options.items():
        lower = name.lower()
        if lower in ("mean_seed_zero", "mean_seed_ptr"):
            raise ValueError("zero-query merge forbids a second seed-zero override")
        if "fp8" in lower and value is not False and value is not None:
            raise ValueError("zero-query merge requires the original BF16 path, not FP8")
        if ("row_map" in lower or "rowmap" in lower) and value is not False and value is not None:
            raise ValueError("zero-query merge does not support temporal row maps")


class _TrackedZeroKernel:
    """One-call interception; never replaces a different kernel or route."""
    def __init__(self, owner, zeros):
        self.owner, self.zeros = owner, zeros
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
            return self.owner._kernel[grid](
                *args, **kwargs, MEAN_SEED_ZERO=True, mean_seed_ptr=self.zeros)
        return launch


class ZeroSeedMergeAdapter:
    """Bind an immutable cached zero tensor to one original merge invocation.

    total_calls counts accepted token-level merge attempts; kernel_calls
    counts forwarded native launch attempts, including one that raises.
    A successful call must forward exactly one launch. Exceptions restore
    the original module symbol before propagating. Counters are audit data,
    not runtime or quality evidence.

    Zero buffers are created only through Q.new_zeros and never mutated by
    this adapter. They are keyed by shape/dtype/device and retained across
    head chunks and attention calls. Q values are never read or averaged.
    Slots 1:4 of the row table are the real seeds; slot zero is only a dummy
    and may repeat one of their row indices. No four-distinct-row check runs.
    """
    def __init__(self, original_merge, merge_module):
        if not callable(original_merge) or isinstance(original_merge, ZeroSeedMergeAdapter):
            raise TypeError("require an original merge callable, not a second zero adapter")
        kernel = getattr(merge_module, KERNEL_NAME)
        if isinstance(kernel, _TrackedZeroKernel) or hasattr(kernel, "mean_queries"):
            raise ValueError("merge kernel already has a synthetic seed override")
        if not hasattr(kernel, "__getitem__"):
            raise TypeError("require the original launchable temporal shared merge kernel")
        self.original_merge = original_merge
        self.merge_module = merge_module
        self._kernel = kernel
        self._zero_cache = {}
        self._zeros = None
        self._active = False
        self.total_calls = 0
        self.kernel_calls = 0

    @property
    def zeros(self):
        """Most recently used cached buffer, for read-only provenance metadata."""
        return self._zeros

    def __call__(self, q, k, v, seedtable, scale, **kwargs):
        if self._active or getattr(self.merge_module, KERNEL_NAME) is not self._kernel:
            raise RuntimeError("zero-query merge forbids nested or changed kernel overrides")
        _reject_overrides(kwargs)
        if scale != 128 ** -0.5:
            raise ValueError("zero-query video merge requires the original attention scale")
        for tensor in (q, k, v):
            if tuple(tensor.shape) != GRID_SHAPE or str(tensor.dtype) != "torch.bfloat16":
                raise ValueError("require original BF16 Q/K/V [1,21,45,80,40,128]")
            if tensor.device != q.device or not tensor.is_cuda:
                raise ValueError("Q/K/V must share the original CUDA device")
        if (tuple(seedtable.shape) != SEED_SHAPE or str(seedtable.dtype) != "torch.uint8"
                or seedtable.device != q.device or not seedtable.is_cuda):
            raise ValueError("require native uint8 RF0 seed table on the Q/K/V device")

        zero_shape = (1, 21, 80, q.shape[4], 128)
        key = (zero_shape, str(q.dtype), str(q.device))
        zeros = self._zero_cache.get(key)
        if zeros is None:
            zeros = q.new_zeros(zero_shape)
            if (tuple(zeros.shape) != zero_shape or zeros.dtype != q.dtype
                    or zeros.device != q.device or not zeros.is_contiguous()):
                raise ValueError("new_zeros must return the contiguous original-dtype zero buffer")
            self._zero_cache[key] = zeros
        self._zeros = zeros

        tracked = _TrackedZeroKernel(self, zeros)
        self.total_calls += 1
        self._active = True
        try:
            with patch.object(self.merge_module, KERNEL_NAME, tracked):
                result = self.original_merge(q, k, v, seedtable, scale, **kwargs)
                if tracked.attempts != 1 or tracked.forwarded != 1:
                    raise RuntimeError("expected exactly one temporal shared merge kernel launch")
                return result
        finally:
            self._active = False


__all__ = ["ZeroSeedMergeAdapter"]
