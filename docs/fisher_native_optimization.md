# Fisher C4/C10 native optimization, 2026-09-09

Active targets: C4 <=80 ms and C10 <=180 ms for a complete native720p
40-head attention call, including selection, Q packing, all SQ1 merges,
full compressed routing, and intermediate I/O. Neither target is met.

The user requires at least 30% of account usage to remain. Check the live
account limit at the start of each goal turn and between substantial batches.
Use 65% consumed as the conservative wrap-up point, leaving 35% before
accounting lag. Latest live reading was 53% consumed, 47% remaining. Do not
mark the performance goal complete when stopping for the usage reserve.

These are experimental modules; production dispatch and generated videos
are unchanged. The user excluded VBench. New kernel output quality has not
been evaluated on generated videos.

## Algorithm and source scope

`native_fisher/operator.py` keeps geometric-Fisher FP64 GS mathematics,
fixed Uniform16 probes, C32/CG45 candidates, RF0, and query-column
SQ1. The merge derives from the private, validated native C8 owner-pack CUDA
source (SHA256 b16821e91a710526856092449e6f615b83ed2818a4655553f744d9164fc41f5a).
That historical C8 benchmark used a different selector and no SQ1; its
end-to-end operator is not substituted for current Fisher experiments.

C4 places 16 columns with four seeds in a 64-row WGMMA tile. C10 places six
columns with ten seeds in 60 live rows; four unused matrix rows do not produce
route tokens. C10 writes ten physical seeds instead of the reference's 16
padded slots. Its final two-column tail uses guarded stores. Original full
Triton SQ1 routing is retained for both prototypes.

The retained selector splits unchanged BF16/FP32 probe logits from FP64 GS.
Its GS warp uses four probe lanes and eight candidate lanes, with four
candidates and four probes in each thread. Candidate and probe reductions
keep the original balanced FP64 arithmetic tree, and argmax retains left
ties. All seed indices match the original J4 selector on tested captures.
This adds a reusable 6.19 GB FP32 logit workspace. The previous J2 Triton
selector remains an explicit comparison option. C10 uses
the explicit FP32 SFU logarithm for positive normal inputs. Its numerical
differences are included in the full-output checks below. C4 retains
libdevice log2 because the SFU variant's small timing change was inconclusive.

The singleton is removed after owner assignment. Critically, the original
operator rounds unpeeled normalized weights to BF16, multiplies K/V, and
then divides the FP32 accumulation by residual mass. Moving that division
before BF16 rounding caused a 0.00953 relative output error and was rejected.
The retained implementation computes the reciprocal once and scales each
FP32 accumulator; it does not change the mathematical grouping or support.

## Verified real-QKV results

One freshly clean scai7 H200 GPU1, source capture
`gma_c8_cuda_150ms_20260908/real_full40_step0_layer00.pt`, BF16
`[1,21,45,80,40,128]`. Five CUDA-event samples after two warmups, with a
continuous foreign-process monitor. All reported complete timings include
the current selector and all eight H5 native merge/route chunks. Reference
full calls use the original C4 H8 or C10 H4 chunking.

| Retained prototype | Complete ms | Original ms | Same-run Flash SDPA ms | vs SDPA | Relative L2 | Max abs |
|---|---:|---:|---:|---:|---:|---:|
| C4 distributed FP64 GS | 109.150 | 236.279 | 326.042 | 2.987x | 0.000387865 | 0.196289 |
| C10 SFU + distributed FP64 GS | 294.126 | 525.055 | 325.496 | 1.107x | 0.000660994 | 0.039063 |

Outputs are finite but **not bitwise equal**. The initial screening threshold
of relative L2 <0.002 is a diagnostic gate, not evidence of unchanged video
quality or a promotion threshold. Larger validation coverage remains required.

Separate component diagnostics (means; these are not substitutes for complete
call timing): C4 selection 14.273 ms, packing 0.392 ms, H5 merge 7.070 ms,
H5 route 4.141 ms. C10 selection 30.664 ms, packing 0.387 ms, H5 merge
21.293 ms, H5 route 9.790 ms. Both have eight H5 chunks.

C10 also passed full40 checks on real step0 layers 19 and 39: 293.006 and
290.209 ms, respectively (same-run SDPA 325.552 and 322.776 ms, 1.111x and
1.112x). Relative L2 errors are 0.000354959 and 0.000313349; maximum absolute
error is 0.015625 in both. Seed mismatch counts are zero for all three layers.
This expands layer coverage, but not denoising-step or generated-video coverage.

C4 passed the same layer19/39 checks: 108.492 /107.344 ms, same-run SDPA
326.178 /323.526 ms (3.006x /3.014x). Relative L2 is 0.000120150
/0.000108578; maximum absolute error 0.021484 /0.031250. All seed indices
match. The previous J2-selector full-call measurements were 113.208 ms for
C4 and 309.045 ms for C10. The current comparison is approximately 3.6%
and 4.8% faster on layer0; it is not a formal paired estimate.

Artifacts: `/local2/denghaoran/GridMergeAttention/experiments/gma_c4_c10_native_20260909`
on scai7 and `C:/GMA/experiments/gma_c4_c10_native_20260909` locally.
Primary result files are `c4_probe4_layer00.json` and `c10_probe4_layer00.json`;
exploratory workbook entries are in existing `C:/GMA/tmp.xlsx`.

## Rejected variants and useful evidence

- C4 per-element correctly-rounded division: 170.366 ms. Reciprocal reuse
  reduced this to 118.640 ms; no algorithmic shortcut was introduced.
- Lane-shared logarithms, singleton reduction over its single nonzero term,
  and reciprocal reuse in entropy correction produced the retained
  C4 113.064 ms and C10 335.649 ms results. These independent-run differences
  are screening evidence, not a paired estimate of the small incremental gain.
- C10 per-element division: 735.371 ms. Reciprocal reuse reduced it to
  341.621 ms. The latter includes the necessary TMA-read completion fence
  before reusing output staging as shared score storage.
- C10 warp-local partial summaries restored four resident CTAs but increased
  complete latency to 441.713 ms. Rejected and archived, not the retained
  `fisher_c10_merge.cu`.
- Exact ordered-integer REDUX on four interleaved lane masks made that variant
  much worse, 960.698 ms complete. Rejected; no score quantization was used.
- `native_fisher/selector.py` and `fisher_gs.cu` split FP32 probe-logit
  construction from warp FP64 Fisher GS. On real layer0, all 12,096,000 C4
  and 30,240,000 C10 seed indices match the production selector exactly;
  all-zero-query tie cases also match. It is slower (31.119 vs18.292 ms C4;
  87.050 vs61.501 ms C10) and is **not used by the retained operator**.
- A reciprocal-square-root basis variant also matches all seed indices and
  zero-input ties but did not improve latency (31.312 /87.555 ms). Rejected.
  Disassembly of the first native selector shows FP64 helper calls and local
  stack traffic; a source-level rsqrt change alone did not remove the cost.
- Temporal merge tiles 1/3/7/21 produce bitwise-identical native outputs;
  21 remains fastest for both methods. Smaller tiles are rejected. See
  `c4_tiles.json` and `c10_tiles.json`.
- The selector profile separates probe logits (2.42 ms) from native FP64 GS
  (28.69 ms C4, 84.66 ms C10). Eight production launch configurations all
  preserve real layer0 seeds and zero-query ties. J2/4-warps is fastest:
  17.63 ms C4 and 52.13 ms C10, versus J4 18.26 and 61.51 ms. See
  `selector_profile_sweep.json`. The original candidate-lane native GS remains unused.
- C10 explicit SFU log2 removes the observed local spill and reduces full
  latency to 319.83 ms with J4 selection, then 309.05 ms with J2. C4's SFU
  measurement was 112.37 ms, too close to timing variation to justify the
  additional numerical change; it was not retained.
- Route tiles 64/128/256, loop stages 1/3, and selected 8-warp variants did
  not improve either method. Default 64/4-warps/3-loop-stages remains best;
  256-token/3-stage variants exceed shared memory. See `c4_j2_layer00.json`
  and `c10_lg2_route.json` for timings and numerical comparisons.
- Distributed FP64 GS replaces the slow candidate-only layout. Four-probe
  and eight-probe lane variants match all real layer0 seeds and zero-query
  ties. Selector times are approximately 14.54 /30.87 ms for C4/C10,
  including the 2.42 ms probe-logit pass. See `selector_probe4_layer00.json`
  and the archived initial `selector_probe8_layer00.json`.
- Sharing FP64 exp/sqrt/division results across candidate lanes added shuffle
  overhead and regressed selection to 15.27 /31.93 ms despite exact seeds.
  Rejected and archived in `selector_probe4_lane_math.json`.
- Balanced merge sums produced 113.09 /310.40 ms and small rounding changes;
  there was no speed improvement. Both kernels retain their prior sums.
- Nsight Compute hardware-counter profiling failed with ERR_NVGPUCTRPERM.
  No profiler timing is promoted. The profiler-only mode intentionally
  records neither a full-call timing nor a correctness claim.

## Intermediate-buffer bandwidth controls

The `--io-diagnostic` mode measures contiguous writes and copies separately,
after complete-call correctness and timing. It is not a DRAM counter or an
alternative attention implementation. For one H5 chunk:

| Capacity | K/V buffer bytes (GB) | Write ms | Copy read+write ms | Copy TB/s |
|---|---:|---:|---:|---:|
| C4 | 16.257 | 4.789 | 3.807 | 4.270 |
| C10 | 40.643 | 12.001 | 9.459 | 4.297 |

One copy reads one K-sized buffer and writes another, so its total traffic
equals the two K/V buffers. Complete calls logically write then read
260.112 /650.281 GB across eight chunks. Cache effects and actual merge
DRAM traffic remain unmeasured. These controls support focusing on reducing
materialization, rather than treating the existing route as mostly arithmetic.

## Reproduction and next work

### Shared-ring routing passed; current fused kernels rejected on latency

`warp_route.cuh` implements 64-token online softmax with BF16 probability
rounding, FP32 accumulation, and explicit empty-block handling. Against an
independent FP32 tiled oracle, it passes lengths1/63/64/65/129/640, a13-row
query tail, empty first/middle/all blocks, alternating invalid keys, and
zero-query ties. Three real query groups also pass complete materialized
compressed streams: C4 length3780 has relative L2 9.78e-6; C10 length9450
has relative L2 1.014e-4. These are route-component checks, not full40
attention or video-quality results. See `warp_route_gate.json`.

`warp_fused.cu` connects the exact SQ1 merge with that ring, reusing raw-K
shared storage for overflow after a producer barrier. Selected partitions
0/7/14 at query frames0/10/20, columns0/1,36/37,78/79, heads0/2/4, and all45
query rows match materialized-native merge plus warp route bit for bit.
The following kernel-only timings cover all H5 compressed partitions and
their FP32 partial stores. They **exclude** selection, Q packing, the SQ1
attention contribution, partial combination, and the other35 heads; there
is no complete-attention latency or SDPA speedup for these fused versions.

| Version | Capacity | Registers/thread | Local bytes/thread | CTAs/SM | H5 kernel-only ms |
|---|---:|---:|---:|---:|---:|
| Four warps, two consumer states | C10 | 255 | 768 | 2 | 352.455 |
| Eight warps, separate producer/consumer loops | C10 | 255 | 760 | 1 | 530.251 |
| Consumer function not inlined | C10 | 255 | 1072 | 1 | 532.325 |
| Weight function not inlined | C4 | 255 | 704 | 1 | 474.251 |
| Weight function not inlined | C10 | 254 | 160 | 1 | 574.328 |

All selected-partition checks pass and GPU monitors are clean. Even these
incomplete H5 timings are far too slow; none is promoted. The isolated
merge diagnostic uses128 registers and zero stack, while fused SASS has
many local-memory instructions. The split-role C10 kernel contains189
static LDL and187 STL instructions, with no device calls. These are static
counts, not executed traffic measurements. Reducing C10 local storage to
160 bytes did not improve speed, so spills alone do not explain the gap;
instruction count, occupancy, and synchronization still need separation.

Raw evidence: `warp_fused_c10_gate_profile.json`,
`warp_fused_c10_split_roles.json`, `warp_fused_c10_bounded_consumer.json`,
and `warp_fused_bounded_weights.json`. Exact header/kernel variants are
archived under `source/warp_fusion_round/` with hashes checked against those
results. Current tracked fusion is a rejected diagnostic implementation.
The existing native operator remains unchanged at109.150/294.126 ms.

The first H5 native workspace has2.188% empty C4 slots and13.268% empty C10
slots. This counts positive-infinity bias only; no compaction or skipping
was introduced, and these fractions must not be generalized to all heads
or denoising steps. They do not by themselves account for the target gap.

The generic seed-query loader refactor also re-passed both original strict
320-real/320-zero-query merge gates after all changes. See
`warp_merge_c{4,10}_refactored_default.json`. Native production dispatch and
the retained standalone operator were not changed by this work.

Three C10 warp layouts were subsequently implemented and rejected. All keep
ten physical output/route slots, exact Fisher seeds, and bitwise-identical
full40 output relative to the retained native merge on step0/layer00.

| C10 layout | Complete attention ms | Paired native control ms | Same-run SDPA ms | Speedup vs SDPA |
|---|---:|---:|---:|---:|
| Four columns, 16 matrix rows each | 329.145 | 291.393 | 325.964 | 0.990x |
| Same, masked unused Q loads and parallel TMA stores | 326.600 | 292.144 | 325.907 | 0.998x |
| Six columns, eight main seeds plus two shared tail seeds | 317.999 | 292.357 | 326.611 | 1.027x |

The first two trade 60 live matrix rows per tile for 40; six masked matrix
rows per column are never written into global K/V storage. The third retains
60 live rows and combines warp-local eight-seed winners with two tail seeds
through shared memory. All reduce shared allocation from 65,280 to 57,344
bytes, allow four CTAs/SM instead of three, and have zero local bytes/thread.
Nevertheless their H5 merge stages take 25.060, 25.059, and 24.687 ms,
respectively, versus approximately 21 ms for the retained kernel. Greater
occupancy and less owner scratch did not translate into lower latency.

`c10_layout_review.json` records full timings and checks; exact sources are
under `source/c10_layout_round/`, hash-verified against the three raw JSONs.
Every GPU monitor is clean. These are rejected diagnostics, not replacements
for either native defaults or the earlier video-generation implementation.

### C4 column-block workspace experiment

`fisher_c4_column_workspace.cu` stores merged K/V and bias as
`[query_frame, column_block16, head, key_block, column16, seed4, D]`.
Its TMA store writes each 64-row matrix result contiguously instead of
scattering across sixteen query-column regions. `column_workspace_route.py`
copies the retained route arithmetic and changes only compressed-buffer
addresses. Total workspace size, route token order, and SQ1 support remain
unchanged. It is enabled only by an explicit experimental source and
`--workspace-columns 16`; the default operator keeps its original layout.

On real step0/layer00 full40 input, seeds and native output match exactly.
Complete attention is 114.518 ms, paired native control 114.125 ms, and
same-run SDPA 328.136 ms (2.865x vs SDPA). The control samples vary from
107.676 to 127.866 ms; this does not establish equivalence or a small
performance difference. The candidate fails to improve the retained
109.150 ms result or approach the 80 ms target. Its H5 merge/route stages
are 7.629/4.209 ms; resources are 128 registers, no local bytes, 57,344
shared bytes, and four CTAs/SM. The monitor is clean. Do not promote this
layout or repeat it unchanged. Evidence: `c4_column_workspace16.json`.

### C10 shared-score bank-layout experiment

`fisher_c10_score_swizzle.cu` preserves the retained 60-live-row matrix
mapping and all arithmetic. It pads only the shared normalized-score row
from 80 to 96 floats and XOR-permutes its key index so row bits occupy bank
bits 0/3/4 and lane-key bits occupy 1/2. Address enumeration confirms 32
distinct bank indices for each scalar score-store group; no hardware-counter
claim is made. Physical global output and route storage still have ten seeds.

Step0/layer00 full40 output is bitwise identical to native control, but
complete latency is 299.828 ms versus control 292.548 ms and same-run SDPA
326.671 ms (1.090x vs SDPA). H5 merge is 22.136 ms. The kernel retains
128 registers, zero local bytes, and three CTAs/SM while shared allocation
grows to 69,376 bytes. GPU monitoring is clean. Reject this layout:
removing these address-level conflicts alone did not improve performance.
Evidence: `c10_score_swizzle96.json`; this does not prove every shared-memory
access is conflict-free or that shared-memory conflicts never matter.

Precomputing `exp2(best)` once per column/key and sharing the result across
the ten seeds also fails to improve latency. This preserves every score,
tie, BF16 rounding point, and native output bitwise. It adds 480 shared
floats without an extra barrier: 67,200 bytes total, 128 registers, no local
bytes, three CTAs/SM. Complete attention is 298.492 ms versus paired control
292.453 ms and SDPA 326.630 ms (1.094x vs SDPA); H5 merge is 22.101 ms.
The clean-monitor result is `c10_shared_probability.json`. Eliminating this
source-level exponential repetition did not improve the full operator.

The standalone retained C10 matrix kernel was also recompiled with
`__launch_bounds__(128,3)`, matching its shared-memory occupancy limit.
Registers increase from 128 to 156, with zero local bytes and the same
65,280 shared bytes / three CTAs per SM. Full40 output stays bitwise equal,
but complete latency is 294.253 ms versus paired control 292.027 ms and
SDPA 326.672 ms (1.110x vs SDPA). H5 merge is 21.393 ms. This is the
ordinary separate-K/V kernel, distinct from the earlier rejected joint-K/V
CTA2/3 variants. The clean result `c10_cta3.json` supplies no evidence of
an improvement; the default launch bound remains unchanged.

The four diagnostics in this batch are summarized in
`native_layout_review.json`, with exact source archives and hashes.
Together with the three warp-layout failures, they argue against another
unmotivated tile/register/bank sweep.

### Cache-sliced merge/route: complete tests reject immediate small slices

`CacheSlicedFisherAttention` alternates merge and full SQ1 route for one
head, one native column tile (16 columns C4 / six C10, with the original
two-column C10 tail), and 1/3/7/21 query frames. Kernel slice offsets are
runtime parameters; arithmetic, physical storage, compressed-token order,
and support are unchanged. The original workspace remains allocated, but
the maximum logical K/V region touched by a one-frame pair is only
30.966 MB for C4 /29.030 MB for C10. These are tensor-size calculations,
not proof of L2 residence or measured DRAM traffic.

All eight complete native40 step0/layer00 variants match retained native
output bitwise. Each also passes graph replay for the original Q, Q scaled
by0.875, and original Q again. Every replay reruns selection and includes
three input copies and an independent output clone; capture/compilation
are excluded from timing. Original-selector seed mismatch counts are zero.
All GPUs were freshly clean and continuous monitors report no contamination.

| Capacity | Frames/slice | Merge/route pairs per full call | Uncaptured ms | Graph complete ms | Same-run SDPA ms | Graph vs SDPA |
|---|---:|---:|---:|---:|---:|---:|
| C4 | 21 | 200 | 130.137 | 131.439 | 328.484 | 2.499x |
| C4 | 7 | 600 | 137.543 | 137.578 | 328.484 | 2.388x |
| C4 | 3 | 1400 | 214.468 | 210.360 | 328.484 | 1.562x |
| C4 | 1 | 4200 | 495.872 | 481.916 | 328.484 | 0.682x |
| C10 | 21 | 560 | 335.701 | 335.332 | 328.345 | 0.979x |
| C10 | 7 | 1680 | 544.855 | 540.170 | 328.345 | 0.608x |
| C10 | 3 | 3920 | 976.022 | 964.604 | 328.345 | 0.340x |
| C10 | 1 | 11760 | 2494.847 | 2477.130 | 328.345 | 0.133x |

Same-run captured native controls take110.528 ms C4 /296.769 ms C10.
Uncaptured controls following each candidate remain109.06--109.27 ms C4
and294.93--295.18 ms C10. None of the sliced variants improves the retained
operator. Graph replay does not remove the regression. Slice-capable merge
resources remain128 registers and zero local bytes, with57,344 shared bytes
/four CTAs per SM for C4 and65,280 /three for C10.

Separate H5 graph activity traces for the one-frame slice attribute the
regression to kernel execution, not large gaps between submissions:

| Capacity | Merge launches | Merge total ms | Route total ms | Kernel-gap ms |
|---|---:|---:|---:|---:|
| C4 | 525 | 16.158 | 41.906 | 0.354 |
| C10 | 1470 | 51.030 | 253.812 | 1.005 |

These are instrumented H5 intervals, not full40 timings or hardware
counters. A one-frame route grid contains only16 C4 CTAs or six C10 CTAs
(two in its tail), compared with8400 in the ordinary H5 route. Reduced
parallelism and lost temporal K/V reuse are consistent with the timings;
the traces do not quantify either contribution separately. Do not repeat
these slices unchanged or assume adding a graph makes small cache-sized
batches efficient.

Evidence: `c4_cache_slices.json`, `c10_cache_slices.json`, the two
`c*_cache_slice1_activity.json` traces, and `cache_slice_review.json`.
Exact measured sources are hash-verified in `source/cache_slice_round/`.
The diagnostics are opt-in; the default operator and video dispatch remain
unchanged. Both80/180ms targets, broader denoising-step validation,
integration, and generated-video quality checks remain outstanding.

The older persistent merge's2866ms H5 regression remains unexplained;
do not revive its schedule solely because it avoids host launches.

### Sampled merge clocks and exact reciprocal gate

The isolated `fisher_c{4,10}_clock.cu` diagnostics timestamp CTA thread0
at stage boundaries for query frames0/10/20. C4 records23,625 CTAs per
sample; C10 records61,425 full six-column CTAs, excluding its two-column
global-store tail. H5 event times still include that tail. Raw timestamps
are saved in `c*_merge_clocks_frame*.npz` and summarized by
`diagnose_fisher_merge_clocks.py`.

These are within-CTA issue intervals, including dependency and scheduling
stalls. They are not isolated operation latencies, DRAM counters, or pieces
that can be summed across CTAs to obtain full-call time. In particular,
small explicit store-wait fractions do not rule out memory backpressure
elsewhere. The first sampled query frame includes different cold/steady
behavior from later frames. All three sampled runs preserve H5 native
output exactly. GPU monitors are clean.

| Capacity | Control H5 merge before/after ms | Sampled frames0/10/20 H5 ms | Registers control/diagnostic | Local bytes |
|---|---:|---:|---:|---:|
| C4 | 6.860 /6.950 | 7.202 /6.921 /6.927 | 126 /127 | 0 |
| C10 | 21.631 /21.458 | 21.758 /21.928 /21.927 | 128 /128 | 0 |

Diagnostic kernels with timestamp storage disabled take7.265 ms C4
and22.095 ms C10, so added code and sampling can perturb scheduling even
without increasing spills or changing occupancy. Fractions across the
three sampled frames are:

| Sampled region | C4 cycle fraction | C10 cycle fraction |
|---|---:|---:|
| Q load and score matrix work | 22.9--25.2% | 20.5--23.5% |
| Row normalization | 12.4--13.5% | 10.0--10.5% |
| Shared owner selection | Included below | 16.2--16.8% |
| Owner/SQ1 weights (C4), SQ1 weights (C10) | 32.2--39.7% | 22.7--23.8% |
| Explicit K/V staging waits combined | 2.5--3.2% | 2.5% plus0.9--1.0% prior-V wait |

This motivates examining normalization/ownership/SQ1 instructions rather
than another output-layout-only sweep. Retained C10 SASS contains four
MUFU.RCP sites plus input-range handling for its `div.full.f32` reciprocals;
static disassembly is archived as `c10_retained_clock_review.sass`.

`reciprocal_gate.cu` exhaustively compares the original division with
`rcp.approx.ftz.f32` for all260,046,848 positive FP32 bit patterns in
`[2^-24,2^7)`. On this H200/compiler build there are zero bit mismatches
and zero ULP difference. This is a scalar equivalence gate, not a speed or
video-quality result. `reciprocal_normal_gate.json` preserves its source hash
and clean monitor. The guarded candidate retains original division outside
that range, including tiny, large, negative, and nonfinite values.

The SQ1 residual alone is already bounded: valid cells have
`1e-6 < residual <=1`; invalid cells use1. A second candidate therefore
replaces only that reciprocal with no new range branch, while arbitrary
cluster sums keep the original division. All tested complete outputs match
native control bitwise; seed indices also match the original selector.

| Candidate | Complete ms | Native control ms | Same-run SDPA ms | vs SDPA |
|---|---:|---:|---:|---:|
| C10 guarded normal reciprocal | 299.049 | 291.931 | 326.930 | 1.093x |
| C10 residual-only reciprocal, interleaved | 294.565 | 293.790 | 325.563 | 1.105x |
| C4 residual-only reciprocal, interleaved | 111.592 | 112.441 | 327.788 | 2.937x |
| C4 warp-shared winner exponentials, interleaved | 113.542 | 112.509 | 328.340 | 2.892x |

The C4 exponential candidate computes four independent winning-score
exponentials in different seed-lane groups and broadcasts them using warp
shuffles. It adds no shared buffer or barrier, preserves exact native
output, and has128 registers/no local bytes, but shows no speed benefit.

`--interleaved-control` alternates eight ABBA/BAAB blocks, measuring16 full
calls per method. It was added after separate five-sample runs exaggerated
the apparent C4 residual benefit (109.203 vs114.611 ms; later108.571
vs114.426 ms). In the interleaved run, its0.850 ms mean advantage wins only
five of eight blocks and does not establish a robust improvement. No
candidate is promoted. The generated C4 guarded-normal counterpart was
not benchmarked and is archived as unmeasured, not a result.

Evidence and exact measured source snapshots are summarized in
`merge_clock_numerical_review.json` and `source/merge_clock_round/`.
Native defaults remain109.150 /294.126 ms; the80/180ms goal is still open.

Next distinct owner-reduction hypothesis: compare an order-preserving
64-bit score/owner encoding using shared atomic maximum, avoiding the
full normalized-score scratch array while retaining the six-column matrix
tile. The experimental `fisher_c10_atomic_owner.cu` now implements this
hypothesis with `owner_key.h`. It preserves the first owner on equal scores,
canonicalizes signed zero only for comparison, restores the winning zero's
sign, ignores NaNs, and initializes to negative infinity with owner zero.
A CPU C++ gate using the exact shared header passes257,225 adversarial and
seeded random ten-score cases, including shuffled update order. This is not
a GPU synchronization or full-output check.

The CUDA extension compiles for sm90a with127 registers and zero local bytes
for full40. Requested dynamic shared memory drops from65,280 to57,344 bytes;
actual occupancy has not been queried on a clean GPU. Disassembly shows
`ATOMS.CAST.SPIN.64` compare-and-swap loops, so smaller scratch does not
establish a speed benefit. Source/header hashes, compiler receipt and static
resource output are archived in `atomic_owner_review.json` and
`source/atomic_owner_round/`.

At this continuation's preflight, every scai7 H200 had a foreign compute
process, including the previously clean GPU1. No GPU experiment was launched
or queued. Compilation explicitly hid CUDA devices and left the CUDA runtime
uninitialized. The next required gate is full40 native output comparison and
interleaved timing on a freshly exclusive H200, including tie/empty GPU
checks before any promotion. Default C4/C10 sources remain unchanged.

The follow-up `analysis/validate_fisher_owner_cuda.py` now supplies a bounded
GPU gate for the same128-thread/64-row score layout. It compares512 tiles
against an independent CPU strict-comparison loop for each of four issue
orders (245,760 winner groups per order). Cases include random FP32 bits,
ties, all-NaN/all-negative-infinity groups, signed zeros and subnormals;
the four padded rows contain positive infinity to detect accidental inclusion.
Source and header hashes jointly identify the gate extension. This diagnostic
does not exercise QKV score generation, SQ1 arithmetic, TMA reuse, or complete
attention, so it cannot replace the full40 native comparison.

`--compile-only` hides all CUDA devices and does not initialize CUDA. GPU
execution requires an explicit freshly clean GPU and continuously checks for
foreign processes. With a clean single H200,
run the gate and then the full benchmark, using new receipt paths:

```sh
python analysis/validate_fisher_owner_cuda.py --gpu 1 --output /local2/.../atomic_owner_gpu_gate.json
python analysis/benchmark_native_fisher.py --gpu 1 --capacity 10 --merge-source native_fisher/fisher_c10_atomic_owner.cu --native-control-source native_fisher/fisher_c10_merge.cu --interleaved-control --output /local2/.../c10_atomic_owner_interleaved.json
```

Do not launch the benchmark if the gate fails, or either launch if its clean
GPU preflight fails. Neither command is queued automatically.

The first GPU attempt (`atomic_owner_gpu_gate.json`) started on freshly clean
GPU4 at22:54:24 UTC. At22:54:56 the monitor detected a foreign VLLM process
(PID740322). The run exited with `valid=false` and zero completed rotations;
it supplies no GPU correctness or performance evidence. No full benchmark
was launched. The driver now also checks monitor errors after module loading
and immediately before input allocation, so a detected occupancy change
during preparation prevents that allocation. The receipt and original source
archive remain intact; subsequent attempts must use new receipt paths.

### scai6 validation: 64-bit atomic owner rejected

The user subsequently authorized scai6. Its shared worktree/captures were
available at the same paths. A freshly clean scai6 H200 GPU4 (UUID
`GPU-923ad71f-2808-6bf7-701e-dc79b7aaa38f`) completed the gate with zero score
bit or owner differences across983,040 group comparisons, including poisoned
padding rows. All monitors stayed clean. The complete real step0/layer00
40-head candidate then matched retained native output bitwise, with zero
selector mismatch. It retains the native-versus-original relative L2 error
0.000660994; generated-video quality is still unverified.

| scai6 full40 method | Complete ms | Same-run SDPA ms | vs SDPA |
|---|---:|---:|---:|
| 64-bit atomic owner candidate | 509.409 | 331.508 | 0.651x |
| Retained native control | 290.619 | 331.508 | 1.141x |

Eight alternating ABBA/BAAB blocks provide16 calls per method; the candidate
loses all eight blocks. It requests57,344 dynamic shared bytes, has127
registers/no local bytes and reaches four resident CTAs, but its H5 merge
takes49.954 ms. Smaller scratch and higher occupancy do not compensate for
the added work. Reject it; retain prior C4/C10 implementations. Raw evidence:
`atomic_owner_gpu_gate_scai6.json`, `c10_atomic_owner_scai6_interleaved.json`,
and the updated `atomic_owner_review.json`.

A bounded CPU-only opcode diagnostic confirms unsigned32-bit shared maxima
compile to native `ATOMS.MAX`, unlike the64-bit CAS spin sequence. The next
candidate splits exact ordered-score selection and owner tie resolution into
two32-bit atomic passes, preserving the original -infinity/owner0 initializer
and winning signed-zero bits. It adds a synchronization and is not assumed
faster. Its separate GPU gate and full40 control remain mandatory. The gate
driver accepts explicit `--gate-source` and `--merge-source` and hashes both
the selected gate source and the common encoding header for extension identity.

The two-pass32-bit candidate subsequently passed the same983,040 GPU owner
comparisons and full40 native bitwise output check, with clean monitors and
zero selector mismatch. It also regresses in all eight interleaved blocks:

| scai6 full40 method | Complete ms | Same-run SDPA ms | vs SDPA |
|---|---:|---:|---:|
| Two-pass32-bit atomic owner | 381.179 | 331.488 | 0.870x |
| Retained native control | 292.475 | 331.488 | 1.133x |

It uses128 registers/no local bytes,57,344 dynamic shared bytes and four
resident CTAs. Its H5 merge takes33.023 ms and route9.868 ms. Native32-bit
atomics reduce the64-bit variant's overhead, but the extra atomic passes and
synchronization still fail to beat the original shared score reduction.
This does not measure each cost separately. Reject both variants; no native
default or current-best workbook row changes. Evidence is in
`atomic32_owner_review.json`, `atomic32_owner_gpu_gate_scai6.json`,
`c10_atomic32_owner_scai6_interleaved.json` and `source/atomic32_owner_round/`.
The80/180ms goal remains unmet; do not repeat these owner variants unchanged.

The subsequent streamed-fragment fusion experiment is also rejected. It
uses4,608 extra shared bytes to avoid retaining a64-float K tile across a
barrier, and selected C4/C10 normalized partial outputs match the materialized
reference bitwise. Local storage drops to64 bytes, but all H5 compressed
partitions still take408.921 /467.359 ms on clean scai6 GPU4. These timings
exclude SQ1, final combination, selection and the other35 heads; they are
not full-call results. See `docs/fisher_warp_fusion_design.md` and
`streamed_merge_review.json`. Defaults remain unchanged.

The isolated matrix-load follow-up passes tiny GEMM and native K/V/bias gates
for C4/C10 and selected fused-output comparisons. Its local storage is zero,
but242 registers still permit only one256-thread CTA per SM. H5 partial-fusion
timings are283.623 /320.076 ms on clean scai6 GPU4, excluding SQ1, combination,
selection and35 heads. These are still too slow to promote; no complete
attention result is implied. See `ldmatrix_review.json` and the fusion design
document for exact scope, the recovered C10 driver-initialization failure and
the next warp-layout investigation. Retained full40 results remain unchanged.

The one-column matrix-load follow-up reaches two128-thread resident CTAs,
but still only eight resident warps. H5 partial C4/C10 times are299.139 /
299.662 ms. Separating complete producer/consumer loops into noinline functions
leaves243 registers and gives269.679 /314.326 ms. Both layouts pass nine
selected normalized partial outputs bitwise per capacity with clean monitors,
but remain too slow. These exclude selection, packing, SQ1 contribution,
combination and35 heads. No candidate is promoted and no full-call improvement
is claimed. See `one_column_review.json`, `loop_calls_review.json` and the
fusion design for resource caveats and the distinct next investigation.

### Fixed seed reuse and shared query loading: rejected

`fisher_c{4,10}_seed_reuse.cu` fixes each CTA to one column tile and all21
query frames. RF0 seed indices and query base addresses move outside the
frame loop; it explicitly rejects tile counts other than21. C10 SASS confirms
two seed LDG.U8 instructions inside the retained frame loop are outside the
candidate loop. Static instruction count nevertheless increases2088 to2128;
these static figures do not identify the cause of measured runtime changes.

`fisher_c{4,10}_shared_query.cu` instead derives directly from the retained
native sources. It copies original row-major Q with cp.async into shared
output staging after prior TMA reads complete, then uses shared/shared score
WGMMA with unchanged K-dimension accumulation order. Its pack entry point
copies Q unchanged, so its pack and run must be used together. Native controls
independently perform their original packing permutation on every full call.
This preserves shared allocation but moves the prior-output read wait ahead
of score construction and adds synchronization around the shared Q operand.

All four tests use real full40 step0/layer00 input on freshly exclusive scai6
GPU4 and eight alternating ABBA/BAAB blocks,16 full calls per method. Exact
Fisher seeds, finite original-reference checks and bitwise native full-output
checks pass. Every GPU monitor is clean.

| Variant | Capacity | Complete ms | Paired native ms | Same-run SDPA ms | vs SDPA | Winning blocks/8 |
|---|---:|---:|---:|---:|---:|---:|
| Fixed seed reuse | C4 | 114.226 | 113.267 | 331.345 | 2.901x | 2 |
| Shared query | C4 | 117.676 | 113.192 | 331.087 | 2.814x | 0 |
| Fixed seed reuse | C10 | 301.606 | 296.185 | 329.210 | 1.092x | 1 |
| Shared query | C10 | 308.450 | 295.961 | 329.891 | 1.070x | 0 |

Seed reuse keeps126/128 registers for C4/C10; shared query uses128 for both.
All report zero local bytes,57,344 /65,280 dynamic shared bytes and four/three
resident CTAs for C4/C10. Neither approach improves full attention, so all are
rejected and retained native modules remain unchanged. In particular the C4
unpaired means suggested a benefit that the alternating comparison did not
confirm. These tests do not extend later-step or video-quality coverage.

`query_loading_review.json` contains full metrics and block counts;
`seed_reuse_sass_review.json` records the main backedge and seed-load addresses.
Exact candidate/control sources and the benchmark are archived under
`source/query_loading_round/`. Do not repeat these query-loading variants
unchanged or infer performance from moving source expressions outside a loop.

The feature-major consumer QK loop subsequently reduces standalone route
registers166 to128 while preserving all existing route gates and selected
fused normalized partial outputs bitwise. Combined fusion still uses247
registers and one256-thread CTA; H5 partial times293.445 /343.977 ms do not
justify promotion. See `dmajor_route_review.json` and the fusion design for
scope and the next native-seed producer resource investigation. Full40
retained results and broader validation requirements remain unchanged.

Native-seed producer isolation now validates bitwise K/V/bias ring snapshots
for ten CTAs per capacity, using165/166 registers for C4/C10. A feature-major
producer reduces only C4 to158; C10 stays166. A separate shared consumer-state
fusion reaches252 registers and H5 partial times296.779 /355.055 ms despite
passing selected exact outputs; it is rejected. See `producer_roles_review.json`
and the fusion design. The next distinct implementation is a native WGMMA
shared-ring producer, with capacity-specific resource sizing, rather than
another unchanged warp-fusion layout. Retained full40 results remain unchanged.

### CUDA fusion primitives: matrix and SQ1-merge gates passed

The next implementation uses explicit warp matrix fragments so register
lifetimes can be controlled before joining merge and route in shared
memory. `native_fisher/warp_mma.cuh` implements score/merge operands;
`warp_merge.cuh` implements the unchanged normalization, left-tie owner
selection, BF16 weight rounding, SQ1 removal and FP32 residual correction.
These helpers currently run only through diagnostic wrappers.

Integer/zero GEMMs match FP32 references exactly; random and real-capture
GEMMs have relative L2 <=3.73e-7. Native Q packing yields identical score
output to ordinary gathered Q. Both C4 and C10 then match retained native
K, V, and bias **bit for bit**, including signed-zero storage checks, on
320 real tiles and the same 320 tiles with zero queries. Coverage includes
five head lanes, first/middle/last query frames, columns0/1/37/78/79, and
key-block ranges0–63,430–493,881–944. Zero queries exercise equal seed
scores, empty clusters, +inf bias, and singleton removal. Both selected
GPU monitors remained clean. This is a merge gate, not complete attention
correctness, runtime, or video-quality evidence.

Evidence: `warp_mma_fragment_gate.json`, `warp_merge_c4_storage_ties.json`,
and `warp_merge_c10_storage_ties.json`. Reproduce with
`analysis/validate_fisher_warp_mma.py --gpu 1 --merge-capacity 4` (or10)
and a fresh `--output` path. See `fisher_warp_fusion_design.md` for remaining
ring routing, overflow handling, SQ1 partial reduction, and full-call gates.

### Rejected joint K/V matrix instruction

The raw K/V shared panels are consecutive, so one N256 WGMMA instruction
can replace two N128 instructions without changing merged output. Four
complete-call experiments produced bitwise-identical retained-native
outputs and exact seeds, but none improved latency:

| Capacity | Minimum CTAs requested | Registers | Local bytes/thread | Actual CTAs/SM | Complete ms | Native control ms | SDPA ms | vs SDPA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C4 | 2 | 254 | 0 | 2 | 120.453 | 113.905 | 327.062 | 2.715x |
| C4 | 3 | 166 | 32 | 3 | 117.753 | 113.895 | 327.840 | 2.784x |
| C10 | 2 | 179 | 0 | 2 | 320.128 | 291.932 | 327.688 | 1.024x |
| C10 | 3 | 167 | 32 | 3 | 303.594 | 291.615 | 326.089 | 1.074x |

C4 controls contain high outliers (the CTA3 control ranges108.223–127.830
ms); these five-sample means are not a precise estimate of a small speed
difference. The joint candidate is consistently slower than the normal
108–109 ms samples, and its isolated H5 merge is slower as well. All GPU
monitors remained clean. The retained CUDA kernels are restored unchanged.
Sources are archived as `source/fisher_c{4,10}_joint_kv_cta{2,3}.cu`;
results are `c{4,10}_joint_kv.json` and
`c{4,10}_joint_kv_cta3_hashed.json` in the experiment directory.

An earlier interrupted build left a PyTorch extension lock. A subsequent
process was verified waiting at extension construction with no compiler
child. That waiter was terminated; it provided no timing result. Extension
names now include the source SHA256 prefix, so different source variants
use separate build directories. A before/after source hash check also
rejects changes during compilation. This does not automatically remove a
stale lock for an interrupted build of identical source.

### Rejected fused materialization prototypes

A Triton prototype computes seed scores, assignment, SQ1 peeling, merged
BF16 K/V, and full compressed routing inside each CTA. It writes FP32
partial accumulations/maxima/sums and combines them afterward. Fisher
Uniform16 selection and all compressed keys remain; SQ1 is consumed with
each key block instead of appended after all compressed keys. This changes
floating-point online-softmax order and is not bitwise equivalent.

For C10, 64 key blocks per partition reduce the full40 logical intermediate
write-plus-read volume from 650.281 GB to 67.092 GB. This excludes raw Q/K/V
reads, selector workspace traffic, and register spills; it is not a measured
DRAM-traffic reduction. Nevertheless, full attention becomes much slower:

| Columns per CTA | Query rows per column | Warps | Complete ms | Same-run SDPA ms | vs SDPA | Relative L2 vs original |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 8 | 2119.003 | 328.237 | 0.155x | 0.00148037 |
| 4 | 16 | 8 | 3209.499 | 329.453 | 0.103x | 0.00146299 |
| 4 | 32 | 16 | 10721.513 | 327.421 | 0.031x | 0.00146299 |

All three have exact native seed indices, finite full40 outputs, and clean
GPU monitors. The single-column kernel uses 176 registers and no local
stack but lowers the seed/merge work to small matrix instructions. Grouping
four columns raises register pressure: the 8-warp version reports 255
registers and 124 spills; the 16-warp/32-query-row version reports 128
registers and 352 spills. Merely increasing the warp count did not resolve
this. No C4 result or video-quality preservation is claimed for these fused
prototypes. None is retained or exposed in production dispatch.

Exact tested sources are archived as `source/fused_single_column_rejected.py`
and `source/rejected_materialization_fused/native_fisher/fused.py` alongside
the benchmark. Reproduce against the pre-joint native operator at `fa52897`.
These results reject this particular Triton implementation; they do not
establish that reducing K/V materialization is intrinsically slower.

### Head scheduling and activity diagnostics

The H1/2/4/5/8/10 chunk sweep did not improve either method. Every chunk
produced bitwise-identical native output. C10 retained H5: 293.804 ms in
the sweep and 295.308 ms in its confirmation. Other chunks took 301.65
to 305.13 ms. C4's initial fully dynamic head-stride kernel regressed to
113--116 ms; restoring the H5 compile-time stride recovered a 109.161 ms
sequential control. None of these supersedes the retained table above.

Two independent K/V workspaces and a route stream preserve native output
exactly on three repeated full40 calls, but do not improve complete latency:

| Capacity | Sequential control ms | Double-buffer pipeline ms | Same-run SDPA ms | Pipeline vs SDPA |
|---|---:|---:|---:|---:|
| C4 | 109.161 | 113.920 | 327.337 | 2.873x |
| C10 | 294.098 | 294.138 | 325.179 | 1.106x |

CUDA activity tracing is available without privileged hardware counters.
The C10 pipeline trace confirms distinct merge/route streams but only
0.850 ms of actual overlap. Its kernel durations total 182.709 ms for
merging, 76.404 ms for routing, 30.861 ms for selection, and 0.382 ms for
packing. C4 overlaps only 0.882 ms. These instrumented observations are
not the CUDA-event headline measurements. `summarize_native_activity.py`
computes interval unions and overlap from the exported trace.

A persistent merge grid of two CTAs per SM was also rejected. An isolated
C10 H5 merge produces bitwise-identical K/V and bias, but takes about
2866 ms rather than 21 ms; three repeated calls confirm the regression.
Waiting for complete TMA writes instead of shared-memory reads at task
boundaries does not improve it (2865.91--2866.25 ms). The initial full-call
test was manually terminated after excessive runtime; the isolated runs
show severe slowdown, not evidence of deadlock. Its partial JSON is not
a valid full-attention timing. No persistent C4 result is claimed.

All scheduling candidates are rejected and the original H5 sequential
operator is restored. Experimental source is archived under
`source/rejected_scheduling/` in the artifact directory; the archived C4
persistent edits were not run. Reproduction should use the accompanying
archived benchmark and operator together. Do not repeat these scheduling
variants without a concrete change addressing the measured limitations.

CUDA Graph replay was measured with three input copies and an independent
output clone inside each timed call. Capture and warmup are excluded, just
as compilation and warmup are excluded for other variants. Selection runs
again on every replay. Both methods match uncaptured output bitwise for
the original query, a query scaled by 0.875, and the original query again.

| Capacity | Uncaptured control ms | Graph with copies ms | Same-run SDPA ms | Graph vs SDPA |
|---|---:|---:|---:|---:|
| C4 | 108.978 | 115.444 | 327.206 | 2.834x |
| C10 | 293.542 | 293.680 | 326.205 | 1.111x |

Neither graph result improves the retained operator. C4 includes a 129.26 ms
sample; even its fastest sample is slower than the uncaptured mean. The
graph wrapper remains an explicit diagnostic option, disabled by default,
and is not integrated into video inference. All ordinary head-sweep,
pipeline, and graph results above have clean GPU monitors. Their small
differences from the original reference remain the previously reported
native merge rounding differences; no new video-quality claim is made.

Use the monarch_rt environment, CUDA13 SM90a compiler, and FlashInfer's
CUTLASS headers. The scripts require a freshly idle H200 and monitor the
selected GPU; they never wait or queue for a busy device.

```bash
python analysis/benchmark_native_fisher.py --gpu 1 --capacity 4 --output /local2/.../new_c4.json
python analysis/benchmark_native_fisher.py --gpu 1 --capacity 10 --output /local2/.../new_c10.json
python analysis/benchmark_native_fisher.py --gpu 1 --capacity 10 --cuda-graph --activity-trace --output /local2/.../graph.json
python analysis/summarize_native_activity.py /local2/.../graph.trace.json --output /local2/.../graph.activity.json
python analysis/benchmark_fisher_selector.py --gpu 1 --output /local2/.../new_selector.json
python analysis/benchmark_fisher_selector.py --gpu 1 --layout probe4 --output /local2/.../distributed_selector.json
```

Next: explore avoiding full-buffer materialization while preserving the
operator. Head chunk sweeps and naive double-buffer overlap did not help.
Use activity tracing for launch gaps; hardware counters need an external
permission change.
Logical merged K/V writes per
full call are 130.06 GB for C4 and 325.14 GB for C10, with the route reading
the same amount. These are tensor-size calculations, not measured DRAM bytes.
Also restructure native FP64 GS only with exact seed checks; dot-logit tuning
cannot address its dominant cost. Do not repeat the rejected tile, route,
masked REDUX or rsqrt variants without a new concrete change.
Then validate complete calls on real layers 0/19/39 across denoising steps,
empty clusters and extreme inputs, integrate only a validated retained
variant, and check video PSNR/SSIM/LPIPS. Do not declare the 80/180 ms goal
achieved from one-layer or stage-only timings.
