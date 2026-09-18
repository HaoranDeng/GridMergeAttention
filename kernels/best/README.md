# Retained best kernels (2026-09-10)

One fixed, callable snapshot per BF16 capacity; FP8 is saved separately.
These are the fastest retained complete H40 implementations in the recorded
experiments, not a claim of a theoretical optimum. Historical experiments stay
in their existing locations. These entries do not change generation defaults.

## BF16

| Variant | Directory | Complete call | Same-run SDPA | Speedup |
| --- | --- | ---: | ---: | ---: |
| GMA-C4 | [c4](c4) | unmeasured after exact-capacity correction | n/a | n/a |
| GMA-C8 | [c8_r2](c8_r2) | 159.796 ms | 327.310 ms | 2.048x |
| GMA-C10 | [c10](c10) | 267.166 ms | 324.771 ms | 1.216x |

**Capacity invariant:** C4 is exactly four merged K/V representatives and C8
is exactly eight. Neither may append or stream SQ1/singleton/raw K/V. The C4
timings below are historical C4+SQ1 measurements and are retired; they do not
describe the current exact-capacity C4 operator.

Historically, GMA-C4's selector took a third exact step on 2026-09-15:
`fisher_c4_selector_v9c_halving_mxredux.cu` replaces the Gram-Schmidt butterfly
reductions with recursive-halving reductions (each probe lane ends with one
candidate's score, so the local argmax collapses onto the REDUX cross-lane
argmax) and takes the center max by REDUX on order-preserving integer keys.
Every added add pairs the same operands as the butterfly it replaces; seeds and
complete output are bitwise equal to the previous selector on layers 00/19/39
and the tie edge cases. Registers 89 -> 80, 5 CTAs per SM. Selector stage
5.26 -> 4.88 ms (-0.37 ms per call, 90 of 90); complete call 98.486 -> 98.175 ms
in the same window (-0.29 ms, 55 of 90, at the complete-call instrument's
resolution floor). The 98.175 ms row is that window's candidate figure.

GMA-C4's selector was replaced again later on 2026-09-15 by
`fisher_c4_selector_v8_ldsm_packed.cu`, an exact rewrite of the FP32 GS
register-pipelined selector (padded shared rows with ldmatrix fragment loads,
branchless and REDUX argmax, hoisted anchor and candidate row, halved transpose,
one packed 32-bit seed store per frame). Seeds and complete output are bitwise
equal to the previous selector on layers 00/19/39 and the tie edge cases;
registers 124 -> 89 and 4 -> 5 CTAs per SM. Selector stage 7.25 -> 5.23 ms
(-2.02 ms per call, 90 of 90 paired wins); complete call 100.288 -> 98.954 ms in
the same window (-1.27 ms, 80 of 90 paired wins). The 98.954 ms row is that
window's candidate figure; compare paired deltas, not absolutes across windows.

GMA-C4 was re-measured and re-promoted on 2026-09-15. Its merge is now
`fisher_c4_desc_select.cu` (WGMMA descriptors derived from three per-CTA bases,
hoisted self-column tests; output bitwise equal to the previous merge) and its
selector `fisher_c4_selector_fp32gs_regpipe.cu` (FP32 Fisher Gram-Schmidt with a
register-pipelined frame loop). The selector is a declared numerics change:
seeds differ from the FP64 selector on 12 / 2 / 5191 of 12,096,000 positions on
layers 00 / 19 / 39, output relative L2 under 5e-5, accepted on a 20-video PSNR
screen against the dense reference (+0.032 dB paired, same host, no difference).
The paired complete-call measurement is 100.704 ms against 104.359 ms for the
previous pair in the same process and window (-3.656 ms, 87 of 90 paired wins,
layers 00/19/39). No same-run SDPA figure was taken in that window, so the SDPA
column is not spliced from the 2026-09-10 run. The previous sources stay in the
directory as the paired control; receipts are in `analysis/receipts_c4_r1/`.

## FP8 intermediate storage

| Variant | Directory | Complete call | Same-run SDPA | Speedup |
| --- | --- | ---: | ---: | ---: |
| GMA-C10 FP8 | [fp8/c10](fp8/c10) | 245.754 ms | 327.203 ms | 1.331x |

FP8 here means compressed K/V stored as E4M3, directly rounded once. Original
Q/K/V and the returned output remain BF16. No C4 or C8 FP8 implementation is
promoted in this snapshot set.

### GMA-C8 r2 numerics implementation

The C8 row above is `c8_r2`, the r2 optimisation campaign result. Its sole
runtime entry is the screened numerics implementation selected on 2026-09-12.
The retained C8 snapshot stays in [c8](c8) unchanged as the reference every
gate below is defined against. Historical campaign arms remain in
`c8_r2/evidence.json`, but they are not packaged or callable.

| Entry | Complete call | Speedup vs same-run SDPA | Paired saving vs v9 |
| --- | ---: | ---: | ---: |
| retained `c8` (v9) | 174.598 ms | 1.875x | reference |
| **`create_c8_r2`** — numerics, serial | **159.796 ms** | **2.048x** | **14.801 ms**, t=17.37, 59/60 |

Same-run SDPA is 327.310 ms in this D13A window. The window ran on one
exclusively claimed H200 with a foreign-process monitor.

This implementation is NOT bitwise-equal to `c8`. What does not change: the
selector, the probe, the route, the C=8 hard clustering, the C32/CG45 geometry,
the routing, and the seeds, which are bitwise identical on all three captures —
so the selected sparsity pattern is unchanged. What changes is the arithmetic
inside the merge: measured against `c8` on three real full-40-head captures,
rel-L2 8.5e-05 / 1.2e-04 / 2.2e-04 and max-abs 0.035 / 0.037 / 0.113, and
5.5237e-06 of key-ownership decisions — about 5.5 per million near-ties —
resolve to the other seed.

Three independent 20-video same-node paired generation screens cover this
arithmetic, each against a v9 control generated on the same node, and all three
pass with every harmful-direction 95% CI straddling zero:

| Screen | Candidate | ΔPSNR | ΔSSIM | ΔLPIPS |
| --- | --- | ---: | ---: | ---: |
| D4T (scai6, 2026-09-11) | guarded-softmax stack | -0.045 (t=-1.41) | -0.00094 | +0.0026 |
| D9B (scai7, 2026-09-12) | the v18 merge source | +0.0091 (t=0.42) | +0.00063 | +0.00051 |
| D10A (scai7, 2026-09-12) | **the shipping bytes** | -0.0177 (t=-0.71) | -0.00014 | +0.00137 |

A direct-PSNR gate is not used because the cross-node reproduction floor for an
unmodified operator is about 15 dB, so such a gate fires on the null change; the
control must be generated on the same node. Full records, including the
provenance of the rel-L2 figure quoted at the decision, are in
`c8_r2/evidence.json` (`default_tier`, `numerics_envelope_note`,
`video_screens`).

The 2.048x ratio replicated on 2026-09-13 in a fresh 60-block anchor window
(D19 S4): v9 177.924 ms, `create_c8_r2` 162.279 ms, same-window SDPA
332.748 ms -> 2.050x, paired +15.645 ms, t=14.01, 60 of 60. Compare the
ratio and the paired delta across windows, never the millisecond count.

## Use

Run from the repository root in the existing `monarch_rt` environment:

```python
from kernels.best import create_c4, create_c8, create_c10

attention = create_c4(device="cuda:0")  # choose one capacity
output = attention(q, k, v)
```

C8 r2 exposes one fixed numerics entry:

```python
from kernels.best import create_c8_r2

attention = create_c8_r2(device="cuda:0")
output = attention(q, k, v)
```

`create_c8_r2` is the screened numerics implementation. It is faster than
`create_c8`, is not bitwise-equal to it, and holds about 33 GiB of reusable
device buffers.

The r2 merge kernel is tuned for a 105-tile span and the wrapper fixes it.
`merge.run`'s `tiles` argument only sets `grid.z = ceil(210/tiles)`, so every
legal value in {21, 42, 105, 210} returns identical bytes and a wrapper that
passes the wrong one is silently slower rather than wrong. The retained `c8`
wrapper hardcodes 21; `NativeC8R2` forwards `MERGE_TILES`. The span was
re-swept over {21, 42, 105,
210} on the shipping kernel after all of this campaign's changes: 105 wins, and
21, 42 and 210 cost 1.681, 0.335 and 0.583 ms respectively on a 900-sample
in-call instrument, with every value bit-identical to 105.

FP8 has an explicit, separate entry:

```python
from kernels.best.fp8.c10 import create

attention = create(device="cuda:0")
output = attention(q, k, v)
```

Construct once and reuse. Each instance owns large reusable GPU buffers and
must be called serially on one CUDA stream. Inputs must be contiguous BF16
`[1, 21, 45, 80, 40, 128]` tensors on the instance device. C8 also accepts
`[1, 75600, 40, 128]`. The normal call processes all 40 heads and includes
selection, packing, merging and full compressed attention. These are inference
implementations specialized for Hopper SM90a; runtime compilation needs NVCC,
PyTorch and CUTLASS headers in the existing FlashInfer installation.

C8 is native CUDA throughout. C4 uses a CUDA selector and merge with a Triton
route. C10 BF16/FP8 use Triton probe logits and route with CUDA GS and merge.
Saving the fastest implementations does not make C4 or C10 Triton-free.

Only the fixed `create(device=...)` entry is supported. Internal helper modules
retain historical experimental branches for source fidelity; their alternative
settings and optional methods are not packaged public interfaces.

## Measurement and validation scope

All table entries are previously measured complete attention calls on an
exclusively idle H200, excluding compilation. They are not four-step video
latencies or new measurements from this repository cleanup. SDPA is paired
within each experiment. Layer coverage differs, so this is not a fresh
controlled capacity sweep:

- C4: desc_select merge plus FP32 GS register-pipelined selector; 90 paired
  complete calls over real layers 00/19/39 with an A/A null pair. The merge is
  bitwise equal to the previous guarded merge; the selector carries a declared
  5e-4 seed-flip ceiling (package proof on layers 00/19/39 and tie cases) and a
  20-video same-host PSNR screen against the dense reference (+0.03 dB, no
  difference). The screened arm is bitwise equal to the promoted binary
  (`r8_equiv_promoted_vs_screened.json`).
- C8: native v9 packed-owner implementation; actual package constructor tested
  with 20 alternating pairs. Outputs and seeds match the preceding native
  control exactly. This is historical Hellinger C32/CG45, J4/RF0, adaptive-8 plus
  uniform-8 probes; C8 has exactly eight merged representatives and no SQ1.
  C10 uses Fisher Uniform-16 RF0 with its separate historical route. No v9 generation
  result is claimed.
- C8 r2 (`c8_r2`): same operator, faster merge and selector. Gates are a
  ptxas/occupancy check (120 registers, zero spill, 4 CTAs/SM), full-call
  output and seed equality against packaged v9 on real layers 00/19/39, a
  900-sample in-call stage instrument, 60-pair ABBA full calls and interleaved
  Latin-square anchor windows with a paired SDPA control, all on one
  exclusively claimed H200 with a foreign-process monitor. Replication counts
  were declared before timing. The retained numerics implementation additionally
  completed three 20-video same-node paired generation screens, one of them on
  its exact shipping bytes. Compilation is excluded from every timing. The
  historical acceptance script predates the numerics-only packaging; current
  package integrity is checked by `python -m kernels.best._integrity`.
- C10 BF16: long-tiles owner tree, 84 tiles, register-distributed GS and route
  block 64 / four warps / three loop stages; 48 paired calls over real layers
  00/19/39. Full outputs/seeds match the preceding native control exactly.
  The exact scheduling refinement was not rerun through video generation.
- C10 FP8: local-score merge and FP8-via-FP32 route, block 64 / four warps /
  one loop stage; 48 paired calls over real layers 00/19/39. Replay outputs
  match the preceding FP8 implementation exactly. The prior FP8 shared-store
  version completed 20 quality videos; no separate generation claim is made
  for this local-score refinement.

Exactness above is relative to the stated native control, not dense SDPA or
the original Triton implementation. Each directory includes `evidence.json`
with timing samples, numerical gates and provenance, plus `manifest.json`
with source hashes. Detailed original logs remain in the experiment archive:

- `gma_c4_cuda_80ms_20260909/scai7/register_basis_*`
- `gma_c8_cuda_150ms_20260908/native_v9_package_validation_scai7.json`
- `gma_c10_150ms_20260910/full40_long_tiles_tree_layer{00,19,39}.json`
- `gma_c10_fp8_150ms_20260910/scai7/local_scores_paired.json`

The copied CUDA/Triton arithmetic is unchanged. Route imports point at a
frozen local module; C10 BF16 GS line endings match its benchmark hash.
Candidate-table helpers remain repository dependencies and are also hashed.
Git attributes preserve frozen source bytes across Windows/Linux clones.
Each factory verifies integrity before compilation; checks can also run
without PyTorch or a GPU:

```sh
python -m kernels.best._integrity
```

Packaging verification passed locally (all manifests and Python syntax) and
on scai7 (all four public entries, 14 implementation modules and shared helper
imports; PyTorch 2.8.0+cu128, Triton 3.7.1). This CPU-only check created no CUDA
context and did not rerun kernel compilation, timing or generation.
