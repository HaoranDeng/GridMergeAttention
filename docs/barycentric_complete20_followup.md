# Barycentric seed candidates: complete20 follow-up

Updated 2026-09-06 UTC. Exploratory results only; no paper or production
default change. The goal remains canonical four-step/20-video PSNR above
12 and complete attention no slower than Hellinger C4. Neither criterion
is established for these candidates.

## What changed

All three candidates use RF0-only selection, the same C32/g45 candidate
table and Uniform16 probes, original BF16 Q/K/V, FP32 QK accumulation,
FP64 selection features, C4+SQ1, and full compressed core attention in
all40 layers and all4 steps. There is no training, calibration, dense
exception, FP8 candidate score path, or reference-frame expansion.

- `mean_fisher_rf0` (bd): one current-frame mean over all45 query rows
  and three real RF0 Fisher-GS pivots.
- `barycentric_medoid_rf0` (be): the first real row minimizes
  `KL(p_s || softmax(mean_i z_i))`; three ordinary anchored Fisher-GS
  pivots follow. No synthetic seed is used.
- `mean_anchor_fisher_rf0` (bg): one current-frame mean, the original
  matched real row, and two RF0 Fisher-GS pivots.

Means use FP32 reduction rounded to BF16, without norm restoration. They
are constructed once per complete attention call. Real row indices, not
RF0 query vectors, are reused across query frames.

The one-cell mean-query identity is exact in ideal arithmetic, but it
does not optimize the final seed-dependent partition. In particular,
fixed-cell multi-component optimization needs query-dependent component
masses, not generally a uniform query average. See
[the existing mathematical limits](seed_barycenter_limits.md).

## Complete20 PSNR: all three candidates rejected

The immutable generation source is `bf49da4` in
`GridMergeAttention_worktrees/barycentric_native_20260905bc` on scai6.
Each run has20 canonical videos,3200 compressed attention calls and80
calls per layer, verified `raw_state_dict` loading, successful worker
exit and no observed foreign GPU process.

| Candidate | Generation run suffix | Valid completion UTC | Full-quality run suffix |
| --- | --- | --- | --- |
| Mean-Fisher | `20260905bd` | 2026-09-05 23:56:32 | `20260905bh` |
| Real barycentric medoid | `20260905be` | 2026-09-05 23:57:13 | `20260905bi` |
| Mean + matched anchor | `20260905bg` | 2026-09-06 00:01:31 | `20260905bj` |

Quality evaluation uses the same canonical dense reference, all 81 frames
at 720x1280 without resizing or cropping, framewise RGB PSNR, full-resolution
SSIM and AlexNet LPIPS. All three CPU PSNR streams have completed the exact
20-video manifest. No prefix result is used to rank candidates.

| Candidate | Full20 PSNR | Delta vs existing Anchor32 |
| --- | ---: | ---: |
| Mean-Fisher | 11.030490212430504 | -0.605918215436279 |
| Real barycentric medoid | 11.485429304894780 | -0.150979122972003 |
| Mean + matched anchor | 11.288276912442422 | -0.348131515424361 |

All three fail the PSNR target and are below Anchor32's 11.636408427866783.
They are not promoted, and no engineering optimization is justified for
these particular candidates. This is empirical evidence for the specified
methods, not a refutation of the fixed-cell mean-query identity.

Full-resolution metric run bh completed validly at 00:09:37 UTC on
2026-09-06, with 20 videos / 1,620 frames: Mean-Fisher PSNR
11.030490212430506, SSIM 0.33774328250803387 and LPIPS
0.5669000752178239. The independent PSNR aggregation differs from the
CPU result only by floating-point summation rounding. bj completed validly
at 00:31:17 UTC with 20 videos / 1,620 frames: PSNR 11.288276912442422,
SSIM 0.34581627983075214 and LPIPS 0.5324094534472182. bi was stopped
at the user's reduced GPU-resource limit, after eight metric records.
Its partial SSIM/LPIPS are not full20 results and are not published as
method averages. The already completed be generation and full20 CPU PSNR
remain valid; stopping the separate metric worker is not a quality failure.

## Matched complete-attention timing: speed criterion failed

`barycentric_full40_bf16_benchmark_20260905bf`, frozen `4a6331d`, completed
validly on one exclusively monitored H200. Its24 distinct original-BF16
inputs contain all40 heads: three prompts, all4 steps, layers0/33.
Each method has480 timed invocations (20 repetitions per input), not480
distinct captured inputs. The complete operator includes seed selection,
optional once-per-call means, KV merging, SQ1, full compressed core
attention and contiguous output. Loading, compilation/warmup and external
audit bookkeeping are excluded.

| Candidate | Complete attention ms | Speedup vs SDPA | Latency increase vs Hellinger RF10 |
| --- | ---: | ---: | ---: |
| Existing Anchor32 | 137.185875 | 2.397989x | +6.0763% |
| Mean-Fisher | 149.406114 | 2.201852x | +15.5253% |
| Real barycentric medoid | 160.069936 | 2.055165x | +23.7709% |

Matched denominators are328.970197ms for SDPA and129.327572ms for the
historical Hellinger RF10 FP8-score mode, executed on the same pinned
current kernels. Mean + matched anchor was not included; do not infer
its matched latency from Mean-Fisher.

All24 same-call SDPA checks have exactly zero MSE against the captured
reference. All40-head and once-per-call mean counters passed. GPU
compute-process exclusivity was sampled every5 seconds; other GPUs ran
generation jobs, so host-wide CPU contention was not isolated. The
captured-output MSEs are diagnostics, not video-quality evidence.

Verified local artifacts are under
`C:/GMA/experiments/barycentric_full40_bf16_benchmark_20260905bf`.
`results.json` SHA256 is
`3ff6241440593aedf8b22797d480ea849ff785c69a025ee43b32b8482b3d8e66`.
The exploratory workbook keeps these paired timings separate from
instrumented generation timings.

## Completed direct partition-error selection tests

Fresh runs `20260906bk` (`sq1_partition_rf0`) and `20260906bl`
(`jeffreys_rf0`) were launched from the same immutable `bf49da4` source
on separately clean scai6 H200s at 00:11:35 UTC. The bl guard subsequently
detected a foreign GPU process and terminated only its own worker at
00:12:27 UTC, before any video was produced. That run is invalid and is
preserved as a contaminated launch, not an algorithm-quality failure.
The fresh Jeffreys run `20260906bm` passed clean-GPU admission on GPU0 and
spawned at 00:15:05 UTC; it does not resume or overwrite bl. They retain the matched
real anchor and three other real rows, RF0, C32/g45, Uniform16, original
G256 importance, BF16 merge scores and all-layer full compressed attention.
They are different selectors from the already completed Mass-GS run.

SQ1 partition selects an underrepresented query using the variational
represented-mass deficit after the singleton has been extracted. Conditional
Jeffreys selects using the current partition's conditional symmetrized KL.
Neither uses GS, synthetic means or fitted parameters. Both optimize a
surrogate for the current partition, not the population objective after
inserting and repartitioning around a candidate.

The old SQ1 video run j was interrupted by SIGTERM after six videos and
never supplied a valid complete20 result. Its interruption does not prove
algorithm failure. No completed Jeffreys BF16 video run was found in the
bounded history audit. These new full runs fill that evidence gap; their
quality is now known from all20 canonical videos, without a prefix gate:

| Candidate | Run | Full20 PSNR | Generation completion UTC |
| --- | --- | ---: | --- |
| SQ1 represented-mass deficit | `20260906bk` | 11.507225480428119 | 2026-09-06 00:36:08 |
| Conditional Jeffreys | `20260906bm` | 11.506616749918376 | 2026-09-06 00:39:08 |

Both generation workers exited successfully with all3,200 compressed
attention calls, 80 calls per layer and no observed foreign GPU process.
Both CPU evaluators completed20 videos with81 full-resolution frames each.
Neither exceeds the historical Hellinger RF10 PSNR of11.529624325204969,
the existing Anchor32 result, or the target of12. They are not promoted.
Full20 SSIM/LPIPS are not available for these two methods.

Instrumented generation reports166.04715013027192ms per complete GCA call
for bk and155.51306447982788ms for bm. Their complete-wrapper times,
including metadata checks, are202.88528812408447ms and194.65222311019897ms.
These are not paired benchmark results: no matched SDPA speedup is claimed.
Their complete-attention speed on the new original-full40 BF16 stream has
not been measured, and no speed optimization is justified by these failed
quality results.

## Next candidate

The [Expected-Fisher candidate](expected_fisher_seed_selection.md) changes
the shared local curvature to the average of individual candidate Fishers,
while retaining four real seeds. Its guarded synthetic native parity
passed before a fresh complete20 run `20260906bo` was started from
immutable code `cb3bd87` at 00:42:39 UTC. It uses RF0, C32/g45, Uniform16,
BF16 QK operands and FP64 geometry, with actual J1 selection. No video
quality or matched speed claim is available yet. Active experiments obey
the user's limit of at most two scai6 H200s and one scai7 H200.
