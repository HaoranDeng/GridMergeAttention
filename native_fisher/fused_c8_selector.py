"""Fused BF16 probe-dot plus FP32 Fisher exact-capacity C8 selector."""

import hashlib
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def extension(source_path=None):
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    os.environ.setdefault("MAX_JOBS", "2")
    source = (
        Path(source_path).resolve()
        if source_path is not None
        else Path(__file__).with_name("fisher_c8_selector_fused_v2_dbuf.cu")
    )
    header = source.with_name("warp_mma.cuh")
    digest = hashlib.sha256(source.read_bytes() + header.read_bytes()).hexdigest()
    module = load(
        name=f"fisher_c8_fused_selector_{digest[:16]}",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=False,
    )
    if hashlib.sha256(source.read_bytes() + header.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Fused C8 selector source changed during compilation")
    module.source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    module.source_path = str(source)
    return module


class NativeFusedFisherC8Selector:
    def __init__(self, capacity=8, device="cuda", source_path=None):
        if capacity != 8:
            raise ValueError("Fused Fisher C8 selector requires exactly eight seeds")
        from attention.capacity_contract import require_exact_merged_capacity
        from attention.adaptive_seed_pytorch import multiscale_candidate_table

        require_exact_merged_capacity(capacity, 0)
        self.extension = extension(source_path)
        self.source_path = Path(self.extension.source_path)
        self.arithmetic = "fp32"
        candidates, anchors = multiscale_candidate_table(
            45, 32, torch.device(device), geometry_count=45
        )
        self.candidates = candidates.int().contiguous()
        self.anchors = anchors.int().contiguous()
        self.seeds = torch.empty(
            (1, 1, 21, 45, 80, 40, 8), device=device, dtype=torch.uint8
        )
        self.empty = torch.empty(0, device=device, dtype=torch.float32)

    def __call__(self, q, k, scale=128**-0.5):
        self.extension.run(
            q,
            k,
            self.candidates,
            self.anchors,
            self.seeds,
            scale,
            self.empty,
        )
        return self.seeds

    def resources(self):
        registers, local_bytes, smem_bytes, ctas_per_sm = self.extension.resources()
        return {
            "registers": registers,
            "local_bytes": local_bytes,
            "smem_bytes": smem_bytes,
            "ctas_per_sm": ctas_per_sm,
        }


__all__ = ["NativeFusedFisherC8Selector"]
