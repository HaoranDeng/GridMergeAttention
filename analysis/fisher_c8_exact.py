"""Experimental exact-capacity geometric-Fisher C8 attention.

This operator changes only the selector in the retained C8-r2 stack.  The
selected eight real rows feed the retained exact-C8 pack, merge, and route
stages.  There is no ninth/raw/residual K/V route.

The production selector fuses BF16 probe dots with FP32 geometric-Fisher
Gram--Schmidt and Hellinger-style recursive-halving reductions. The separate
native and FP64 Triton paths remain comparison controls. Capacity is still
exact: eight merged representatives, with no singleton/raw/residual route.
"""

from pathlib import Path

import torch

from attention.capacity_contract import require_exact_merged_capacity
from kernels.best.c8_r2.gma_c8_r2 import MERGE_TILES


ROOT = Path(__file__).resolve().parents[1]
C8_REPRESENTATIVES = require_exact_merged_capacity(8, 0)
_SEED_SHAPE = (1, 1, 21, 45, 80, 40, C8_REPRESENTATIVES)


class ExactFisherC8Attention:
    """Geometric-Fisher selector plus the retained exact-C8 back half."""

    def __init__(self, device="cuda", *, selector_backend="fused", backend=None,
                 selector_source=None):
        if selector_backend not in ("fused", "native", "triton"):
            raise ValueError("selector_backend must be 'fused', 'native', or 'triton'")
        if backend is None:
            from kernels.best import create_c8_r2
            backend = create_c8_r2(device=device)
        self.backend = backend
        self.device = backend.device
        self.selector_backend = selector_backend
        self.selector = None
        self.selector_source = None
        if selector_backend == "fused":
            from native_fisher.fused_c8_selector import NativeFusedFisherC8Selector
            if selector_source is None:
                selector_source = ROOT / "native_fisher/fisher_c8_selector_fused_v2_dbuf.cu"
            self.selector_source = Path(selector_source).resolve()
            self.selector = NativeFusedFisherC8Selector(
                C8_REPRESENTATIVES,
                device=self.device,
                source_path=self.selector_source,
            )
        elif selector_backend == "native":
            from native_fisher.selector import NativeFisherSelector
            if selector_source is None:
                selector_source = ROOT / "native_fisher/fisher_c8_gs_register_halving.cu"
            self.selector_source = Path(selector_source).resolve()
            self.selector = NativeFisherSelector(
                C8_REPRESENTATIVES,
                device=self.device,
                layout="probe4",
                source_path=self.selector_source,
            )
        self.last_seeds = None

    def _validated_seeds(self, seeds):
        if (tuple(seeds.shape) != _SEED_SHAPE or seeds.dtype != torch.uint8
                or seeds.device != self.device or not seeds.is_contiguous()):
            raise ValueError(
                "Fisher C8 must return contiguous CUDA uint8 seeds with exactly "
                f"{C8_REPRESENTATIVES} representatives"
            )
        return seeds

    @torch.no_grad()
    def select(self, q, k, scale=128 ** -0.5):
        q = self.backend._check(q)
        k = self.backend._check(k)
        if self.selector_backend in ("fused", "native"):
            seeds = self.selector(q[:, :1], k, scale)
        else:
            from attention.affine_hellinger_seed_triton import (
                fisher_geometric_c8_seed_rows_triton,
            )
            seeds = fisher_geometric_c8_seed_rows_triton(
                q[:, :1], k, scale,
                candidate_count=32,
                candidate_geometry_count=45,
                block_j=4,
                probe_count=16,
                output_dtype=torch.uint8,
                gs_num_warps=4,
            )
        self.last_seeds = self._validated_seeds(seeds)
        return self.last_seeds

    @torch.no_grad()
    def run_with_seeds(self, q, k, v, seeds, scale=128 ** -0.5, output=None):
        """Run only the retained exact-C8 pack/merge/route stages."""
        original_shape = q.shape
        q, k, v = [self.backend._check(x) for x in (q, k, v)]
        seeds = self._validated_seeds(seeds)
        if output is None:
            output = torch.empty_like(q)
        else:
            output = self.backend._check(output)
            if output.data_ptr() in (q.data_ptr(), k.data_ptr(), v.data_ptr()):
                raise ValueError("Output must not alias an input")

        self.backend.merge.pack(q, self.backend.packed_q)
        for head in range(0, 40, 5):
            self.backend.merge.run(
                self.backend.packed_q, k, v, seeds,
                self.backend.mk, self.backend.mv, self.backend.bias,
                head, 5, MERGE_TILES, scale, self.backend.empty, 0, 1,
            )
            self.backend.route.run(
                q, self.backend.mk, self.backend.mv, self.backend.bias,
                output, head, 5, scale,
            )
        return output.view(original_shape)

    @torch.no_grad()
    def __call__(self, q, k, v, scale=128 ** -0.5, output=None):
        seeds = self.select(q, k, scale)
        return self.run_with_seeds(q, k, v, seeds, scale, output)

    def metadata(self):
        return {
            "method": "fisher_geometric_c8_rf0_fp32",
            "capacity": C8_REPRESENTATIVES,
            "merged_representatives": C8_REPRESENTATIVES,
            "singleton_count": 0,
            "selector_backend": self.selector_backend,
            "selector_arithmetic": (
                self.selector.arithmetic
                if self.selector_backend in ("fused", "native") else "fp64"
            ),
            "selector_source": None if self.selector_source is None else str(self.selector_source),
            "back_half": "kernels.best.c8_r2",
        }

    def resources(self):
        result = dict(self.backend.resources())
        if self.selector_backend == "fused":
            result["selector_resources"] = self.selector.resources()
        result.update(self.metadata())
        return result


__all__ = ["C8_REPRESENTATIVES", "ExactFisherC8Attention"]
