# MI300 Fisher C4 selector and attention

## 180 ms target: complete attention

`Mi300C4Attention` is an opt-in entry for the full Wan14B 720p BF16 C4/SQ1
operator. It includes the native HIP selector and ROCm Triton merge/route.
It retains full compressed attention and the analytic seed algorithm.

```python
from native_hip.attention import Mi300C4Attention

attention = Mi300C4Attention()
# Contiguous BF16 Q/K/V: [1,21,45,80,40,128] on one MI300X.
output = attention(q, k, v)
```

The instance owns selector scratch and is intended for sequential calls on
one device/stream. Each output is newly allocated unless `output=` is supplied.
The stock module's kernels, constants and generation defaults are unchanged.
Compressed K/V/bias workspace remains about 65.54 GB, plus selector logits,
QKV/output and allocator caching, as in the previous H20 version.

Job **418792**, exclusive amdhpc `k002-004-v1`, gfx942, ROCm 7.2,
PyTorch 2.12.0+rocm7.2, Triton 3.7.0. Three real step0 captures, all40 heads:

| Layer | Previous HIP H20/J4 | New complete call | Same-run Flash SDPA |
|---|---:|---:|---:|
| 00 | 301.878 ms | 177.129 ms | 364.153 ms |
| 19 | 305.048 ms | 178.151 ms | 368.859 ms |
| 39 | 302.561 ms | 177.931 ms | 362.915 ms |
| Mean | 303.162 ms | **177.737 ms** | 365.309 ms |

Latency falls **41.37%** versus the previous HIP H20 operator, with
**2.055x versus same-run Flash SDPA**. The original Triton-selector H5/J4
control measured 323.474 ms. Every one of 36 candidate event samples is below
180 ms (maximum 179.712 ms). Each method has 12 samples per layer after warmup;
GMA order alternates and SDPA uses the same inputs and timing method afterward.
No graph capture. Initial GPU utilization was zero with no KFD PIDs. All22
continuous process-monitor samples were clean. These are replay measurements,
not a guarantee for every layer/step or GPU clock state.

All36,288,000 seed indices match the original selector. Outputs are finite
but **not bitwise equal**: relative L2 versus stock is 2.75e-5 / 2.14e-5 /
3.20e-5, with max absolute difference 0.03125 / 0.0078125 / 0.0078125.
The cross-layer gate was relative L2 < 1e-4. Zero Q/K with real V also passes
a full-output bitwise comparison. No video PSNR/SSIM/LPIPS/VBench was measured.
Use the earlier H20/J4 path if bitwise output preservation is required.

The speedup comes from J16 with explicit 16x16 BF16 matrix instructions,
two merge warps, four route warps and `waves_per_eu=2` in the route. Merge
normalization reuses one reciprocal per row, then corrects the quotient with
two FMAs. The original guarded residual denominator is retained. A four-warp
merge with stock routing was bitwise equal on layer00 but slower (~205 ms).
The uncompensated reciprocal screened at ~172 ms but is not the default here.

Reproduce final validation with `analysis/validate_mi300_180ms.py` and the
existing three capture files. On amdhpc, use:

```bash
sbatch --export=ALL,BENCHMARK=analysis/validate_mi300_180ms.py \
  scripts/amdhpc/mi300_180ms.sbatch
```

Do not exceed three active/pending jobs. Jobs418789,418791,418792 ran serially.
`analysis/mi300_180ms_worker.py` supports file-fed screening queues.
`analysis/build_mi300_tiled_merge.py` regenerates the merge copy from the stock
kernel, including optional scheduling/division experiments; rerun validation
after regenerating against changed source. Canonical evidence is under
`C:/GMA/experiments/gma_mi300_180ms_20260914`: summary, raw receipts, queues,
logs and GPU monitoring. The result is recorded in `C:/GMA/tmp.xlsx` row1286.

## Native selector

Experimental native HIP FP64 Gram-Schmidt selector for MI300X (gfx942).
The probe-logit matrix uses the existing BF16-input / FP32-accumulation Triton
kernel. Current-Q seed selection, Uniform16 probes, Fisher geometry, ties,
and full compressed attention support are unchanged. Generation defaults and
the retained H200 snapshots are unchanged.

The standalone HIP kernel compiles with ROCm 7.2 `hipcc`, without PyTorch C++
headers or CUDA inline assembly. Two independent 32-lane logical groups per
CDNA wave are faster than one 64-lane group on the tested capture. All
shuffles specify their logical width. FP64 reductions retain balanced order;
only Gram-Schmidt residual updates use explicit FMA. No fast-math flag or
reduced-precision seed scores are used.

## Use

```python
from native_hip.selector import HipFisherSelector

selector = HipFisherSelector()  # C4, logical width32, P4, 256 threads
seeds = selector(q, k)
```

Requires ROCm PyTorch and contiguous BF16 Q/K on one gfx942 device. K is
`[1,21,45,80,40,128]`; Q has that shape or just its first-frame slice.
Output is uint8 `[1,1,21,45,80,40,4]`. The instance reuses its output and
6.19 GB FP32 logit buffer: use serially on one stream and clone seeds when
retaining them across calls. This is an inference primitive, not a complete
native HIP attention operator. Merge and route remain Triton on ROCm.
C10 compilation is experimental and has not been GPU-validated in this run.

For the retained complete C4 call, reuse the selector above and call the
existing merge/route twice with `head_start=0,20`, `head_count=20`,
`top_r=1`, `query_column_selector=True`, `seed_rows_native_layout=True`,
`query_column_major=True`, `split_kv_accum=True`, `block_j=4`,
`merge_loop_stages=1`, `route_loop_stages=1`, `merge_disable_licm=False`,
and one shared output tensor. Keep the stock four-warp merge/route and
64-token route tile. The exact invocation is in
`analysis/tune_mi300_hip_pipeline.py`.

Set `ROCM_PATH` for the compiler (default `/opt/rocm-7.2.0`) and
`GMA_HIP_CACHE` for compiled libraries. Compilation is excluded from timing.

## Retained three-layer result

Job 418746 on exclusive `k002-004-v6` completed the launch sweep and validated
real step0 layers 00/19/39. The retained configuration combines the same HIP
selector with H20 head chunks. J4, four merge/route warps, route64, and one
merge/route loop stage remain fixed. It passes all 36,288,000 seed indices
and all three full40 outputs bitwise against the original Triton C4 path.

| Layer | Triton control H5 | HIP + H20 | Same-run Flash SDPA |
|---|---:|---:|---:|
| 00 | 323.709 ms | 305.875 ms | 367.196 ms |
| 19 | 326.362 ms | 307.781 ms | 372.508 ms |
| 39 | 324.220 ms | 306.619 ms | 363.735 ms |
| Mean | 324.764 ms | 306.758 ms | 367.813 ms |

Complete latency falls **5.544%** (1.0587x versus the H5 Triton control).
Speedup versus paired Flash SDPA is **1.1990x**. Each layer has eight
alternating reference/candidate pairs and eight SDPA event samples after
warmups. The continuous monitor recorded 43 samples, only the benchmark's
GPU process, and no errors. Compare within this job: its timings differ
from the first job's GPU/clock state.

The H20 merge needs approximately 65.54 GB for compressed K/V and bias,
versus 16.38 GB for H5, in addition to the 6.19 GB selector logits and
original QKV/output. Allocator caching can raise observed VRAM use further.
This tradeoff fits the tested MI300X; it is not a memory reduction.

J8/J16 reached 267.8/255.5 ms in screening but changed BF16 output rounding
(relative L2 0.000224 on layer00). They were not retained or promoted as
quality-preserving optimizations. Route128 also changed rounding and was
slower. Eight merge/route warps and two merge loop stages were slower.

Evidence is in canonical `C:/GMA/experiments/gma_mi300_hip_20260914`:
`summary.json`, `result.418746.json`, process telemetry, job logs, and verified
capture hashes. The HIP source byte SHA256 is
`55bc837db00267618a9b887e11ebfe2770be9580c01d009897087ca4e72396b2`.
These three step0 captures do not establish all-denoising-step or generated
video quality. No PSNR, SSIM, LPIPS, or VBench measurement was made.

## Initial selector-only change, before head-chunk tuning

amdhpc job 418743, exclusive `k002-004-v2`, MI300X 192 GiB,
PyTorch 2.12.0+rocm7.2, Triton 3.7.0. Real Wan14B 720p BF16 QKV,
step0 layer00, all40 heads. Capture SHA256:
`3c5c727c40a248e917b79dcedabd7bc61465aea3e4b07077c77f07f9ece07585`.

| Metric | Original Triton selector | HIP selector |
|---|---:|---:|
| Selector including probe logits | 16.087 ms | 7.942 ms |
| Complete attention call | 315.692 ms | 307.616 ms |
| Same-run Flash SDPA | 354.730 ms | 354.730 ms |
| Speedup versus SDPA | 1.124x | 1.153x |

Complete attention includes selection and eight H5 merge/route chunks;
BF16 inputs/output, C4/SQ1, full compressed route. Eight alternating pairs,
CUDA/ROCm events after warmups, no graph capture. Latency falls 2.558%.
The 2.026x selector-only improvement is not a complete-attention speedup.

All 14 logical-width/probe/thread configurations produced exactly the same
12,096,000 seed indices as the ROCm Triton reference. All seven distinct
logical-width/probe layouts also passed the full zero-logit tie case. The
retained C4 produced a bitwise-equal full40 attention output (relative L2=0).
The original selector imported NVIDIA libdevice unconditionally; it now
selects HIP libdevice under ROCm so the reference can compile.

These are single-layer replay results, not generated-video quality results.
Initial GPU use was 0%, with no KFD processes and only 297,758,720 bytes VRAM
allocated. The first job saved before/after snapshots, not a continuous
process trace. Subsequent tuning uses a continuous process monitor. No PSNR,
SSIM, LPIPS, VBench, or cross-layer validation is claimed by this first result.

Raw evidence: `experiments/gma_mi300_hip_20260914/result.418743.json` in the
canonical C:/GMA workspace. The JSON field `hip_source_sha256` in this first
receipt is the build-cache digest (source plus compiler path and flags).
It is not a hash of source bytes alone.

`analysis/benchmark_mi300_hip_fisher.py` reproduces the selector sweep,
seed/tie validation and complete paired benchmark. The Slurm runner uses one
exclusive MI300 node and preserves Slurm's device visibility settings.
Submit at most three active/pending jobs, as requested for this experiment.
`analysis/tune_mi300_hip_pipeline.py` is a separate exploratory launch sweep;
its choices do not change defaults or establish improvements until measured.
