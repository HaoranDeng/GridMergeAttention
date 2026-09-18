# Production GMA kernels

This directory contains the supported BF16 kernel packages. Historical
candidates, builders, diagnostics, receipts, and unused controls are not
part of the production tree.

| Variant | Entry point | Implementation | Complete call |
| --- | --- | --- | ---: |
| GMA-C4 | kernels.best.create_c4 | CUDA selector and merge; Triton route | Rebenchmark required after exact-capacity correction |
| GMA-C8 reference | kernels.best.create_c8 | Native CUDA | 174.598 ms |
| GMA-C8 r2 | kernels.best.create_c8_r2 | Native CUDA | 159.796 ms |
| GMA-C10 | kernels.best.create_c10 | CUDA GS and merge; Triton probe and route | 267.166 ms |

C4 and C8 are exact capacities: they emit exactly four and eight merged K/V
representatives, respectively, with no singleton, SQ1, raw, or residual
route. C8 r2 is the preferred C8 implementation; C8 is retained as its
production reference.

Construct one instance with create_c4, create_c8, create_c8_r2, or
create_c10, then reuse it serially on one CUDA stream. Inputs are contiguous
BF16 tensors with shape [1, 21, 45, 80, 40, 128]. C8 also accepts the
flattened shape [1, 75600, 40, 128].

The native CUDA packages target Hopper SM90a. Runtime compilation requires
NVCC, PyTorch, and the CUTLASS headers supplied by the existing FlashInfer
installation.

Run python -m kernels.best._integrity before use to verify every retained
source and shared dependency.
