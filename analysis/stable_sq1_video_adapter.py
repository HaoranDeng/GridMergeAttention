"""Explicit stable bulk normalization for the isolated all-layer video path.

This is never installed on import and does not change selector/defaults.
The original C4+SQ1 wrapper and full compressed route remain in use.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
from unittest.mock import patch

from analysis.stable_sq1_bulk_kernel import (
    GENERATED_KERNEL, SOURCE_KERNEL, generate_stable_sq1_source, validate_launch_contract,
)


def load_generated_kernel(original_kernel, output):
    """Archive one generated module outside the immutable source snapshot."""
    output = Path(output)
    source = generate_stable_sq1_source(inspect.getsource(original_kernel.fn))
    with output.open("x", encoding="utf-8") as stream:
        stream.write(source)
    module_name = "_gca_stable_sq1_video_" + hashlib.sha256(source.encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(module_name, output)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, GENERATED_KERNEL), {
        "path": str(output.resolve()), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "semantic_change": "direct bulk softmax after full80 ownership and SQ1 removal; structural nonempty mask",
        "weights": "FP32 normalized bulk before BF16 weighted dot; no subtractive residual division",
        "selector_and_route": "unchanged real seed table, original full C4+SQ1 route",
    }


class _TrackedStableKernel:
    def __init__(self, owner):
        self.owner, self.calls = owner, 0

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            if self.calls:
                raise RuntimeError("expected one stable merge launch per wrapper call")
            meta = {name: kwargs[name] for name in ("num_warps", "num_stages") if name in kwargs}
            if meta != {"num_warps": 4, "num_stages": 1}:
                raise ValueError("stable video merge requires original launch geometry")
            bound = self.owner.signature.bind(*args, **{name: value for name, value in kwargs.items() if name not in meta})
            bound.apply_defaults()
            options = bound.arguments
            validate_launch_contract(options)
            for name, expected in {"SCALE": 128 ** -.5, "LOOP_STAGES": 2,
                                   "DISABLE_LICM": False, "INPUT_PRECISION": "ieee"}.items():
                if options.get(name) != expected:
                    raise ValueError("stable video merge changed " + name)
            self.calls += 1
            result = self.owner.kernel[grid](*args, **kwargs)
            self.owner.kernel_calls += 1
            return result
        return launch


class StableSQ1MergeAdapter:
    """Swap only the merge kernel during each original merge/route call."""
    def __init__(self, original_merge, merge_module, kernel):
        self.original_merge, self.merge_module, self.kernel = original_merge, merge_module, kernel
        self.original_kernel = getattr(merge_module, SOURCE_KERNEL)
        self.signature = inspect.signature(self.original_kernel.fn)
        self.kernel_calls = 0
        self.calls = 0
        self._active = False

    def __call__(self, q, k, v, seeds, scale, **kwargs):
        if self._active or getattr(self.merge_module, SOURCE_KERNEL) is not self.original_kernel:
            raise RuntimeError("stable merge forbids nested/changed kernel bindings")
        if scale != 128 ** -.5 or kwargs.get("head_count") != 8 or kwargs.get("head_start") != (self.calls % 5) * 8:
            raise ValueError("stable video merge requires five ordered original eight-head chunks")
        for tensor in (q, k, v):
            if (tuple(tensor.shape) != (1, 21, 45, 80, 40, 128) or str(tensor.dtype) != "torch.bfloat16"
                    or not tensor.is_cuda or tensor.device != q.device):
                raise ValueError("stable video merge requires original CUDA BF16 QKV")
        if (tuple(seeds.shape) != (1, 1, 21, 45, 80, 40, 4) or str(seeds.dtype) != "torch.uint8"
                or not seeds.is_cuda or seeds.device != q.device):
            raise ValueError("stable video merge requires original RF0 real-row seed table")
        tracked = _TrackedStableKernel(self)
        self._active = True
        try:
            with patch.object(self.merge_module, SOURCE_KERNEL, tracked):
                result = self.original_merge(q, k, v, seeds, scale, **kwargs)
                if tracked.calls != 1:
                    raise RuntimeError("original wrapper did not invoke exactly one stable merge")
            self.calls += 1
            return result
        finally:
            self._active = False
