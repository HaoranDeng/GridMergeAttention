"""Digest-keyed loader for exact-capacity Fisher-C8 merge candidates."""

import hashlib
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def extension(source_path):
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    os.environ.setdefault("MAX_JOBS", "2")
    source = Path(source_path).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    includes = Path(torch.__file__).resolve().parent.parent / "flashinfer/data/cutlass/include"
    module = load(
        name=f"fisher_c8_merge_{digest[:16]}",
        sources=[str(source)],
        extra_include_paths=[str(includes)],
        extra_ldflags=["-lcuda"],
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "-lineinfo",
            "--ptxas-options=-v",
        ],
        verbose=False,
    )
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Fisher-C8 merge source changed during compilation")
    module.source_sha256 = digest
    module.source_path = str(source)
    return module


class NativeFusedFisherC8Merge:
    def __init__(self, source_path):
        from attention.capacity_contract import require_exact_merged_capacity

        require_exact_merged_capacity(8, 0)
        self.extension = extension(source_path)
        self.source_path = Path(self.extension.source_path)

    def run(self, *args):
        return self.extension.run(*args)

    def resources(self):
        registers, local_bytes, smem_bytes, ctas_per_sm = self.extension.resources()
        return {
            "registers": registers,
            "local_bytes": local_bytes,
            "smem_bytes": smem_bytes,
            "ctas_per_sm": ctas_per_sm,
        }


__all__ = ["NativeFusedFisherC8Merge"]
