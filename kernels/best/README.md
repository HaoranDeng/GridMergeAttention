# Production GMA kernels

This directory contains the supported BF16 kernel packages. Historical
candidates, builders, diagnostics, receipts, and unused controls are not
part of the production tree.

| Variant | Entry point | Implementation | Complete call |
| --- | --- | --- | ---: |
| GMA-C4 | kernels.best.create_c4 | CUDA selector and merge; Triton route | Rebenchmark required after exact-capacity correction |
| GMA-C8 (default, r2) | kernels.best.create_c8 | Native CUDA | 159.796 ms |
| GMA-C8 slow_reference | kernels.best.create_c8_slow_reference | Native CUDA | 174.598 ms |
| GMA-C10 | kernels.best.create_c10 | CUDA GS and merge; Triton probe and route | 267.166 ms |

C4 and C8 are exact capacities: they emit exactly four and eight merged K/V
representatives, respectively, with no singleton, SQ1, raw, or residual
route. C8 means the r2 implementation by default. `create_c8_r2` remains
an explicit alias for the same implementation. The pre-r2 implementation
is available only through the explicit `slow_reference` factory; its
retained source files remain under `c8/` for provenance.

Both `kernels.best.create_c8()` and `kernels.best.c8.create()` construct r2.
Use `kernels.best.create_c8_slow_reference()` or
`kernels.best.c8_slow_reference.create()` for the previous C8 implementation.

Construct one instance with create_c4, create_c8, create_c8_slow_reference,
or create_c10, then reuse it serially on one CUDA stream. Inputs are contiguous
BF16 tensors with shape [1, 21, 45, 80, 40, 128]. C8 also accepts the
flattened shape [1, 75600, 40, 128].

Numerical acceptance preserves the mathematical GMA operator and requires
numerically close results; bitwise identity of seeds, assignments,
intermediates, or outputs is not a default acceptance condition. R2 is the
user-approved default, including its finite-precision reductions, fast
normalization, approximate log, and packed winner comparison. Near-tied
assignments can differ. Record error metrics and the tolerances chosen for
each comparison against `slow_reference`, and investigate material numerical
or video-quality regressions. The user has not specified one universal
numerical tolerance. Historical timings above are retained measurements,
not new measurements of this dispatch change.

The native CUDA packages target Hopper SM90a. Runtime compilation requires
NVCC, PyTorch, and the CUTLASS headers supplied by the existing FlashInfer
installation.

Run python -m kernels.best._integrity before use to verify every retained
source and shared dependency.
