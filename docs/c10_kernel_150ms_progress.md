# C10 kernel-only optimization, 2026-09-10

Target: less than 150 ms per complete native40 attention call. Algorithm stays
geometric Fisher Uniform16 RF0, ten real seeds, SQ1, and full compressed
attention. One exclusively clean scai7 H200 at a time. Account usage must leave
at least 30% available; latest observed usage was 65%.

The latest retained experimental kernel combines a native-layout owner tree
with 84 query tiles per CTA, reusing resident raw K/V across four six-column
groups. Complete calls average **267.166 ms** over real step0 layers 00/19/39.
The same-run previous retained kernel averages **271.490 ms** and matched SDPA
**324.771 ms**: **1.59%** incremental improvement and **1.216x vs SDPA**.
All three full outputs and seed indices are bitwise equal to the retained
native control. Sixteen interleaved calls per method/layer, exclusively clean
GPU4. This is a kernel replay result; generation was not rerun for this change.
The 150 ms objective remains unmet and production dispatch is unchanged.

| Captured layer | Complete ms/call | Previous retained control | Matched SDPA |
|---|---:|---:|---:|
| 00 | 268.245 | 272.453 | 325.612 |
| 19 | 268.277 | 272.491 | 325.930 |
| 39 | 264.974 | 269.524 | 322.772 |

Use `native_fisher/fisher_c10_long_tiles_tree.cu`, merge `tiles=84`, and
`native_fisher/fisher_gs_register_distribute.cu`. Sources are experimental
overrides to `NativeFisherAttention`; default dispatch has not changed.
Receipts: `full40_long_tiles_tree_layer{00,19,39}.json`. Stage-only scheduling
screen `long_tiles.json` found 42/84 tiles slightly faster and 294 no better;
all K/V/bias were bitwise exact. The combined source retains 116 registers,
zero local bytes, 57,344 shared bytes and four resident CTAs.

## Earlier exact retained baseline

The earlier experimental complete operator measured **270.149 ms**
averaged over captured real step0 layers 00, 19, and 39 (twelve calls per layer).
The paired previous native operator averages **292.752 ms**; matched Flash
SDPA averages **325.391 ms**, giving **1.204x**.
The 150 ms target is not met. These are kernel replay measurements, not video
generation measurements or an all-layer/step performance claim.

| Captured layer | Complete ms/call | Paired native control | Matched SDPA | Speedup vs SDPA |
|---|---:|---:|---:|---:|
| 00 | 272.015 | 294.697 | 327.217 | 1.203x |
| 19 | 270.323 | 293.088 | 325.661 | 1.205x |
| 39 | 268.110 | 290.471 | 323.296 | 1.206x |

All three complete native outputs are bitwise equal to the paired original
native implementation, with zero seed mismatches. The average reduction is
22.603 ms (7.72%).

## Exact GS optimization

`native_fisher/fisher_gs_register_distribute.cu` preserves FP64 arithmetic,
balanced reduction trees, all nine pivots, and leftmost tie selection. Opaque
PTX register selects prevent compiler array-index recognition from storing
the feature tables in thread-local memory. Probe-wise exponentials, square
roots and correctly rounded divisions are computed once per coordinate in
each half-warp and broadcast. No approximate reciprocal, FP32 GS, changed
probe set, or changed seed rule is used.

Three real layer captures (00, 19, 39) have zero differing seed entries. The
all-zero-logit tie case is also exact. GS-only paired means decrease from
29.384/29.391/29.416 ms to 13.136/13.130/13.260 ms. The complete outputs
are bitwise equal to the previous native operator. Layer00's pre-existing difference
from the padded Triton reference remains relative L2 0.000661; the GS change
adds no error to the native output.

The retained merge candidate `fisher_c10_half_owner.cu` processes owner scores
in two 40-key halves and reuses the output staging space. It retains the exact
FP32 mass and entropy accumulation sequence, including the original 64/16-key
tail split. Shared storage falls from 65,280 to 57,344 bytes, giving four
resident CTAs instead of three; 116 registers and zero local bytes. All H5
K/V/bias entries are exact. Paired H5 merge means are 20.217 versus 21.665 ms.
The combined GS/merge full40 result above is also bitwise equal to the original
native operator. The earlier GS-only complete result is 279.681 ms (control
294.716 ms). Both measurements include all stages and heads.

The complete benchmark includes selection, Q packing, all eight H5 merges,
and all eight full routes. Six alternating ABBA/BAAB blocks provide twelve
calls per implementation. Layer00 used GPU3 UUID
`GPU-60bf0dcb-1a54-e638-748c-5d799b96c077` was clean before launch and monitored
throughout, with no foreign process recorded. Layers19/39 used exclusively
clean GPU4 UUID `GPU-bf8f3ffe-1da0-7e2c-a98f-ccd994c0fee1` after GPU3 became
occupied by another task. Only one GPU was used by this task at any instant.
Selector source SHA256:
`49089d5e099dc2260c32d4148cffdac203f0a31b184fd2073693aebd40b937f9`.
Merge source SHA256:
`e6727391caea653a20822f7f3290e92636711ec00a42dfd1162a5d3a39ae5b64`.

Reproduce using an explicitly freshly idle GPU index:

```sh
python analysis/benchmark_native_fisher.py --gpu 3 --capacity 10 \
  --selector-source native_fisher/fisher_gs_register_distribute.cu \
  --merge-source native_fisher/fisher_c10_half_owner.cu \
  --native-control-source native_fisher/fisher_c10_merge.cu \
  --selector-control --joint-control \
  --output /local2/denghaoran/GridMergeAttention/experiments/gma_c10_150ms_20260910/new_full40.json
```

Default production dispatch has not been changed. The optimized selector is
an explicit source override of the experimental native operator.

## Rejected or diagnostic work

- Register launch bounds 3/2/1 retain exact H5 K/V/bias but take
  21.534/21.525/21.522 ms versus approximately 21.45 ms controls.
- Shared exponential reuse is exact but takes 22.249 ms versus 21.452 ms.
- Two/three/four WGMMA groups per CTA are exact but take
  23.161/33.098/33.062 ms versus approximately 21.46 ms controls.
- GS loop bounding and explicit scalar reductions do not materially improve
  their paired controls. Arithmetic distribution alone is insufficient while
  feature-table local memory remains.
- With register selects and distributed math, P8 and P2 GS layouts take
  approximately 19.4 and 14.2 ms; P1 takes approximately 88.1 ms. P4 is retained.
- Full-support split-K route tests pass relative-L2 < 0.002 in small slices.
  They reduce routing under-utilization, but slice merge costs remain too high.
  No full-call result is inferred by multiplying small-slice times. This
  alternative routing schedule is not adopted.
- Sixteen-key owner tiles and N32 stores remain exact but take 24.119 ms.
  Byte owner storage reaches five resident CTAs but still takes 22.385 ms;
  both lose to the retained half-owner kernel.
- Compact 64-key partitions produce exact K/V/bias. Resumable routing passes
  relative-L2 0.000878, but one-head/six-column group time is 0.820 ms and is
  not competitive. No complete-call estimate is promoted.
- A per-key-block fused Triton producer/route fails the output gate (relative
  L2 approximately 0.00205). Its FP32 probability-product diagnostic remains
  above the gate; the eight-warp configuration encounters illegal memory
  accesses. These runs are rejected; no fused result is adopted.
- Nsight performance counters are unavailable (ERR_NVGPUCTRPERM); no server
  permission or configuration was changed. Direct measurement finds 86.73%
  of C10 slots have finite bias in the layer00 H5 capture.

Primary receipts are under
`C:/GMA/experiments/gma_c10_150ms_20260910` and the corresponding scai7
`/local2/denghaoran/GridMergeAttention/experiments/` directory. Provisional
results are recorded in the existing `C:/GMA/tmp.xlsx` sheet. Pending work:
a larger improvement to merge/route execution. No video-quality claim
or 150 ms completion claim is made.

## Same-call phase measurements and ordinary rounding follow-up

The user now allows normal floating-point rounding differences with unchanged
algorithms, and requires numerical and generated-video validation. Bitwise
equality remains reported but is no longer a hard rejection criterion. No
quantization, FP8, reduced-precision GS, changed seed/probe rules, or partial
compressed attention is authorized by this relaxation.

CUDA events inside complete H40 calls give the following three-layer means:
selection 17.422 ms, Q packing 0.385 ms, merge 178.222 ms, full SQ1 route
76.558 ms. The instrumented total is 272.587 ms, versus 268.848 ms without
intermediate events in the same diagnostic. This is a profiling receipt, not
a new matched-SDPA headline. Multiplying isolated H5 times by eight understates
the merge contribution in the complete operator. Receipt:
`full40_stage_breakdown.json`.

Further exact experiments have not displaced the retained implementation:
two-stream producer/consumer execution takes 271.197 ms versus 270.867 ms
serial; workspace row padding and smaller frame tiles are slower. TMA cache
hints, exponential reuse, fewer barriers, and vector owner reads yield only
small isolated-stage changes. The alternative warp-owner layout takes
32.749 ms/H5 and is rejected. Source clock instrumentation attributes 35.27%
of producer CTA cycles to ownership processing, but perturbs execution and
cannot replace complete-call measurements.

The FP32 balanced-sum candidate keeps the FP64 selector and original C10/SQ1
formulas. It changes softmax/mass reduction order, splits entropy into two
accumulators, and uses an approximate hardware FP32 reciprocal. Three clean
GPU4 paired tests, each with sixteen complete calls per method, give:

| Real layer | Candidate ms | Same-run retained CUDA ms | Same-run SDPA ms | Relative L2 vs CUDA | Maximum absolute error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 00 | 270.720 | 272.636 | 325.948 | 6.584e-5 | 0.046875 |
| 19 | 269.990 | 272.208 | 325.975 | 3.014e-5 | 0.015625 |
| 39 | 267.918 | 269.447 | 322.895 | 4.857e-5 | 0.013672 |

All seed indices are unchanged and all final outputs are finite. Candidate
mean is 269.543 ms, paired retained control 271.431 ms, and matched SDPA
324.939 ms (1.206x). The incremental improvement is only 0.70%; this remains
an experimental candidate, subsequently rejected after preliminary video
validation below. Receipts are
`full40_balanced_sums_layer{00,19,39}.json`; CUDA SHA256 is
`0885da229577d1d218922ad312648dce7d53d36a32827b3d07ef05da64063dc9`.
The reciprocal-only candidate is bitwise exact and has no speed gain.
Probability-domain ownership has relative L2 1.869e-4 and takes 272.217 ms
versus 272.852 ms control; it offers less benefit than balanced sums.

`analysis/validate_c10_native_videos.py` prepares frozen-source paired videos
using original Wan2.1-T2V-14B-Diffusers, 720p/81 frames, 50 UniPC steps,
CFG 5, seed 0, and the first twenty canonical prompts. Control and candidate
execute sequentially for each prompt with one workspace on one monitored
GPU. The wrapper replaces all forty self-attention cores and checks 4,000
actual calls per video with 100 calls in every layer. The control uses the
original native CUDA merge and the bitwise-validated optimized selector.
No generation result is claimed before completion and paired quality metrics.

The first two pairs have completed 4,000 native calls each, with all forty
layers checked. CPU evaluation of all 162 paired frames gives PSNR 22.668 dB,
SSIM 0.80668 and Alex-LPIPS 0.13047 versus the original native CUDA merge.
These are two-prompt preliminary fidelity measurements, not twenty-video
quality validation. VBench is still pending. The first video's sampled frames
retain the same subject and scene but differ in position and fine detail;
small one-call errors have amplified during fifty denoising steps. The
candidate is rejected: its incremental complete-call benefit is below one
percent while the generated videos show measurable drift. The trial was
intentionally stopped after two complete pairs and the third control video;
the unmatched control is retained but excluded from paired quality. Only this
trial's CPU follow-up and GPU worker were terminated, and the parent recorded
incomplete generation. The final purity receipt contains no foreign-process
errors. Twenty-pair validation and VBench were not completed and are not
claimed. The first two candidate video attention means are 278.629 and
277.100 ms/call; matched SDPA was not measured during video generation, so
no video-run SDPA speedup is asserted. Initial compile overhead affects the
first control video. CPU metrics ran concurrently with later video generation.

## Batched fusion prototype: compiled, not GPU-validated

The current separate producer writes 325,140,480,000 bytes of BF16 K/V per
complete call, before the route reads those tensors. The experimental
`native_fisher/batched_fused_route.py` instead fuses merge and partial routing
for multiple query columns, then combines sixteen FP32 online-softmax states.
It retains all 945 key blocks, all ten seeds, and every SQ1 contribution.
This changes floating-point evaluation order and has no accuracy or runtime
claim yet.

CPU-only sm90 compilation rejects the initial four-column/64-query-row
configuration on resource grounds: eight warps require 255 registers and a
1,728-byte per-thread stack; sixteen warps require 128 registers and a
2,176-byte stack. Smaller query tiles still spill heavily. Two-column
configurations with 32 query rows/eight warps or sixteen query rows/four warps
compile with 255 registers and an eight-byte stack. The latter needs 81,920
bytes dynamic shared storage plus 1,024 bytes static shared storage. Its
partition-combination kernel has 86 registers and zero stack bytes.
These are compiler resource reports, not measured performance or occupancy.

Receipts: `batched_fusion_compile{,_v2,_v3,_ready}.json`. Reproduce the
compiler gate without a GPU using `analysis/compile_c10_batched_fusion.py`.
The future guarded real-QKV gate is `analysis/screen_c10_batched_fusion.py`;
run H1 first, then `--heads 40` only if the finite/relative-L2 gate passes.
After the rejected video trial released GPU4, the two-column/BM16/four-warp
H1 test produced finite output with relative L2 0.00209024 and maximum
absolute error 0.0258789. It slightly misses the 0.002 screening gate.
A separately labeled diagnostic-only timing run takes 71.8275 ms for just
one head's merge/full-support route, excluding selection. This is too slow
to justify a full40 trial; the batched fusion family is rejected. The GPU
monitor was clean. Receipts are `batched_fusion_h1_m16_w4.json` and
`batched_fusion_h1_m16_w4_diagnostic.json`. No complete-call speedup is
inferred from the H1 result. The 150 ms goal remains unachieved.

## Bounded bandwidth diagnostic after user IO hint

`analysis/diagnose_c10_bandwidth.py`, receipt `bandwidth_diagnostic.json`,
compares one H5 merge on real layer00 with deliberately invalid diagnostic
ablations. No production kernel changed. Eight alternating-order measurements
on exclusively clean GPU4; each H5 writes 40.64256 GB of useful merged K/V.

| Diagnostic | ms | Useful KV TB/s |
|---|---:|---:|
| Retained full merge | 20.0520 | 2.0269 |
| Synthetic writes, same TMA layout and 56 KiB shared allocation | 11.0636 | 3.6735 |
| Same writes, 16 KiB shared / 8 resident CTAs | 11.0413 | 3.6810 |
| Contiguous vector writes | 9.5052 | 4.2758 |
| Contiguous copy, counting reads and writes | 9.4622 | 4.2952 |
| Original arithmetic, opaque consumption of accumulators, no KV staging/store | 11.7704 | N/A |

The existing write layout reaches about 86% of the contiguous-write control.
Increasing residency from four to eight CTAs barely changes its throughput.
The real merge's additional arithmetic and synchronization are therefore a
major contributor to its lower effective output rate. Its time is much closer
to the sum of compute-only and write-only times than their maximum, suggesting
limited overlap; these ablation times are not additive phase measurements.
The earlier instrumented CTA profile also puts about 35% in ownership and
21% in normalization/weights/bias. Prioritize shortening those dependencies
and overlapping useful work with output traffic, rather than occupancy alone.
These are useful-byte rates, not privileged hardware DRAM-counter readings;
no full40 latency is inferred from H5. Per user request, stop this diagnostic
here and conserve tokens for substantive kernel optimization.

## Cooperative owner comparison rejected

`analysis/build_c10_cooperative_owner.py` distributes the ten score comparisons
over two or four lanes per key without changing the MMA tile or tie rule.
The clean-GPU4 H5 layer00 screen (`cooperative_owner.json`) is bitwise exact
for all merged K/V and bias. Eight ABBA samples give 21.1831 ms (two lanes)
and 24.4404 ms (four lanes), versus 20.0781 ms for retained half-owner measured
in the same run. Original unoptimized controls are 21.40–21.46 ms.
Both candidates retain four resident CTAs and no local-memory allocation;
extra communication outweighs the shorter comparison chain. Reject both
without full40 or video expansion. The retained complete-call result remains
270.149 ms and the 150 ms target remains unmet.

## Separate TMA output buffers rejected

`analysis/build_c10_output_overlap.py` keeps ownership scratch separate from
one or two TMA staging buffers. The one-buffer form removes the owner-stage
wait; the two-buffer form lets the newest opposite-kind transfer continue
while the older staging buffer is reused. Arithmetic and reduction order stay
unchanged. Clean-GPU4 H5 layer00 K/V/bias are bitwise exact for both variants.
Eight ABBA samples (`output_overlap.json`) give 21.7973 ms for one buffer
(69,632 shared bytes, three resident CTAs) and 25.9930 ms for two buffers
(86,016 bytes, two CTAs), versus retained 20.0493 ms (57,344 bytes, four CTAs).
Both use 116 registers and zero local bytes. The extra shared allocation costs
more than the overlap opportunity saves; reject without full40/video expansion.
Any future overlap design needs a smaller live working set, not more buffers
on top of the current allocation. No complete-call improvement is claimed.

## N32 merge and streamed V rejected

`analysis/build_c10_n32_stream_v.py` tiles the output feature dimension in
32-wide MMAs, keeping only a 5 KiB V tile in shared memory. It preserves the
five K16 accumulation steps and post-MMA SQ1 division. The corrected CuTe
SW64 layout is bitwise exact for all H5 merged K/V and bias on real layer00.
Resources drop to 41,984 shared bytes and 96 registers (16 local bytes),
allowing five resident CTAs. Nevertheless, eight clean-GPU4 ABBA measurements
take 31.6341 ms, versus retained 20.1071 ms in the same screen. Additional V
loads, synchronizations and smaller MMA instructions outweigh the residency
gain. Reject without expanding to full40 or videos.

Receipts: `n32_stream_v_corrected.json`; `n32_stream_v.json` preserves the
initial failed V-layout check and has no candidate timing. The builder now
uses CuTe Swizzle<2,4,3> source bits correctly. Increasing occupancy through
extra work is not sufficient; the complete-call best remains 270.149 ms.

## Native-layout owner tree: small unconfirmed gain

`analysis/build_c10_owner_tree_native_layout.py` balances the ten-way maximum
comparison while retaining the original shared layout and leftmost tie rule.
Unlike the earlier vector-layout tree and cooperative-lane variants, it adds
neither transposition nor shuffle communication. Real layer00 H5 K/V/bias are
bitwise equal; resources remain 116 registers, zero local bytes and four CTAs.
The H5 screen gives 19.9389 ms versus retained 20.1075 ms.

The follow-up complete H40 test has zero new output error and identical seeds.
Sixteen interleaved calls give 271.1541 ms versus retained 272.3274 ms, with
same-run SDPA 325.6893 ms (1.2011x). The observed 0.43% gain is smaller than
individual-call variability; do not promote it as a reliable new best.
No further layers or videos were scheduled for this marginal result.
Receipts: `owner_tree_native_layout.json`, `full40_owner_tree_layer00.json`.
The retained three-layer result remains 270.149 ms; 150 ms is still unmet.

## Compact route row tiling and warp count: rejected

The earlier compact experiment used BM16, reading the same K/V for three
query-row blocks. BM64 covers all45 queries once. In the one-head/six-column
group gate it takes 0.7361 ms with one query frame per producer CTA, or
0.7146 ms with three. This improves the historical BM16/F3 0.8204 ms group
measurement but remains too slow to justify a complete operator trial.

At BM64/F3, eight and sixteen route warps regress to 0.7838 and 1.0199 ms.
All four new configurations produce finite group outputs with relative L2
0.000877611 and maxAbs 0.015625; nine selected producer slices have bitwise
exact K/V/bias. Fifteen partitions still cover every one of the945 key blocks
and SQ1. Seven warmed graph replays on clean GPU4. These are group timings,
not full40 latency or SDPA speedups; no full-call extrapolation is promoted.

Receipts: `compact_partition_m64.json`, `compact_partition_m64_warps.json`.
Reproduce using `analysis/build_c10_compact_partition.py`, then
`analysis/screen_c10_compact_partition.py --block-m 64 --frames 3` with
`--route-warps 4`, `8`, or `16`. This family remains rejected. Current retained
complete-call mean remains 267.166 ms; the 150 ms goal is unmet.

## CTA traversal with long query tiles: rejected

The new 84-tile/tree retained kernel was tested with contiguous key-block
traversal and with key/head interleaving, separately from the historical
short-tile experiment. Both are bitwise exact on all H5 K/V/bias. Eight ABBA
samples on clean GPU4 give 19.6570 versus paired retained 19.6534 ms for key
order, and 20.1568 versus 19.6932 ms for key/head order. No improvement, so
neither was expanded to full40. Receipt: `long_tile_launch_order.json`.

The retained CUDA file's raw SHA changes from `22008a819e5a9dac...` (Windows
CRLF used for the original three-layer run) to `5a3f55b85baf540d...` after Git
checkout with LF. Normalizing CRLF to LF gives the latter SHA for both files;
this is a newline-only difference, not a changed kernel. Default traversal
and the 267.166 ms retained complete-call result remain unchanged.

## Native WGMMA shared-ring producer: validated component

`analysis/build_c10_wgmma_ring.py` extracts the retained WGMMA arithmetic into
`native_fisher/c10_wgmma_ring.cuh`. Five columns each receive a16-entry shared
ring: ten merged clusters, the original SQ1 K/V with zero bias, then five zero
K/V entries with positive-infinity bias. Owner scratch aliases the output ring
before weights are packed; padding is restored before consumers may read it.
The production function writes no materialized global K/V workspace. The
separate gate copies the ring to observable global outputs only for validation.

`analysis/validate_c10_wgmma_ring.py` compares81 slices on each real layer
00/19/39: frames0/10/20, starting columns0/35/75, key blocks0/448/944 and
heads0/2/4. All243 selected slices have bitwise equal merged K/V/bias, original
SQ1 K/V and valid padding. Clean GPU4; receipt `wgmma_ring_producer.json`.
The gate reports128 registers,zero local bytes,82,240 shared bytes and two
resident128-thread CTAs. It is a component gate, not fused-attention latency.
Both the gate source and included generated header are hashed; the header
hash is part of the extension module identity to prevent stale cached builds.

The next concrete step is a16-token consumer with five columns x three
16-query warps, paired with the four-warp native producer. Two shared rings
can separate the producer's next key block from the consumer's previous block;
SQ1 is already in the ring so the consumer does not depend on raw K/V storage.
Validate register allocation and barrier sequencing before timing. Consumer,
partition combination, complete output error and full40 performance remain
unimplemented/unverified at this component checkpoint. The150 ms objective remains unmet.

## Overlapped WGMMA producer and sixteen-token consumer: rejected

The subsequent `c10_ring_pipeline.cu` implements two alternating shared rings,
four producer warps and sixteen consumer warps (one idle), with explicit
640-thread barriers. Each live consumer warp covers sixteen query rows. All
945 key blocks and original SQ1 entries are included in sixteen partitions;
`c10_ring_combine.py` combines their online softmax states. The default operator
does not dispatch to this experimental implementation.

CPU compilation succeeds. Runtime resources are96 initial registers,
123,520 dynamic shared bytes and one resident640-thread CTA. Dynamic register
allocation assigns128 to producer threads and88 to consumer threads, within
the initial61,440-register pool. CUDA reports88 local bytes for dynamic mode
versus312 for static mode; static mode was not timed. These resource facts
suggest pressure but are not a measured attribution of the runtime bottleneck.

On real layer00, the selected49-key-block smoke output has relative L2
0.000969735 against a materialized FP32 attention reference. Whole-head output
against retained CUDA is finite, with relative L2 0.002081492 and maxAbs0.015625.
It fails the existing0.002 screening threshold; bitwise equality is false.
A separately labeled diagnostic run measures18.4477 ms mean over six warmed
calls, including all query frames/columns and partition combination for H1,
excluding selection and Q packing. GPU4 monitoring is clean. Numerical gate
failure remains recorded; diagnostic timing does not promote the candidate.

This design removes the large materialized K/V global workspace but is too
slow at H1 to justify expansion. No H40 extrapolation, generation validation,
or speedup is claimed. Receipts: `wgmma_ring_pipeline_compile.json`,
`wgmma_ring_pipeline_h1.json`, `wgmma_ring_pipeline_h1_diagnostic.json`.
Reproduce the diagnostic with `analysis/screen_c10_ring_pipeline.py --gpu 4
--heads 1 --diagnostic-timing --output <fresh-path>` under a180-second watchdog.
Retained complete H40 mean stays267.16555 ms;150 ms remains unmet.

The bounded follow-up `analysis/build_c10_ring_noinline.py` changes only the
producer's force-inline qualifier to no-inline in a generated source copy.
Dynamic-mode local stack drops from88 to32 bytes (static grows312 to344),
with96 initial registers and unchanged shared memory/occupancy. Whole-H1
relative L2 and maxAbs remain identical to the prior screen. Six warmed calls
measure18.72919 ms, providing no useful improvement over the preceding18.44772
ms diagnostic (these are separate runs, not an interleaved speedup estimate).
The numerical gate still fails. Clean GPU4; no larger test or generation run.
Reducing this stack allocation alone does not rescue the fusion design.
Receipts: `wgmma_ring_noinline_compile.json`, `wgmma_ring_noinline_h1.json`.
Build the generated directory first, then pass it as the optional second
argument of the compile script and as `--source-directory` to the screen.

## TMA compressed K/V reads in the retained route: rejected

`analysis/build_c10_tma_route.py` replaces only the compressed K/V loads of
the existing full-support route with TensorDescriptor TMA loads, preserving
the64-token loop, arithmetic and original SQ1 path. Descriptors are cached
by workspace addresses, shape and token block size. No production dispatch
change. The generated `screen_c10_tma_route.py` compares real layers00/19/39
using retained long-tiles84 merge and the exact GS selector.

All three H5 outputs are bitwise equal to the retained route, with zero error.
Eight interleaved samples per method/layer give10.8480/10.8950/10.6657 ms for
TMA versus9.6289/9.6407/9.5697 ms for the controls. This is route-only H5,
not a complete attention call; the candidate is slower in every layer and
is rejected without expansion or generation. GPU4 guard and monitor clean.
Receipt `tma_route.json`. Complete retained H40 remains267.16555 ms and the
150 ms goal remains unmet.

The bounded larger-TMA follow-up also fails: block128/stages3 gives
10.1821/10.3719/10.0500 ms versus9.6390/9.6882/9.5897; block256/stages1
gives14.4927/14.6092/14.6350 versus9.4806/9.4818/9.5145 on layers00/19/39.
Both retain four warps. All outputs are finite; relative L2 ranges
0.00018844–0.00046235 and0.00024333–0.00055130 respectively, with maximum
absolute error0.03125. This is ordinary changed softmax reduction rounding,
below the0.002 screening line, but every performance comparison is slower.
Eight interleaved calls per method/layer, same full H5 route scope and clean
GPU4. No generation or complete-call trial is justified. Receipts
`tma_route_b128s3.json`, `tma_route_b256s1.json`; reproduce with
`--route-block 128 --route-stages 3` or `--route-block 256 --route-stages 1`.
The TMA route family is closed without replacing the retained kernel.

## Algebraic merge/attention reassociation: implementation rejected

The prototype computes raw QK once, gathers the selected seed rows from those
scores, forms the same ten cluster weights and SQ1 entry, then reassociates
the matrix products instead of storing merged K/V. It moves BF16 rounding
boundaries and therefore cannot inherit the native implementation's numerical
or generation claims. Exact supplied GS seeds, ten clusters, all945 blocks
and original SQ1 are retained; production dispatch is untouched.

The first BF16 version passes a finite one-block smoke check (242 registers,
zero spills,81,920 shared bytes) but complete H1 layer00 output has relative
L2 0.005009336 and maxAbs0.056640625, failing the0.002 screening line. It is
not timed. Its exact source is archived as `reassociated_route_bf16_v0.py`,
SHA256 `76bc5fceb0cb8754e2ecfd976225ce35e3cfdfec050b5aef520be6a6f3c98907`.
Receipt `reassociated_h1.json`.

The subsequent optional FP32 reassociation path retains more intermediate
precision using tf32x3 products, but its first one-block launch reports an
illegal memory access. No output error, performance, or quality result is
claimed for that path. Receipt `reassociated_fp32_h1.json`; source
`reassociated_route.py`, harness `screen_c10_reassociated.py --fp32`.
Both processes are terminal, exclusive GPU4 monitoring stayed clean, and
GPU4 subsequently returned to4 MiB/0% utilization. This rejects the current
implementation, not a proof that algebraic reassociation is impossible.
No larger test or generation run. Retained267.16555 ms;150 ms remains unmet.

A minimal one-column/one-key-block Compute Sanitizer check localizes the
FP32 failure to an out-of-bounds2,048-byte shared read at the wide tf32x3
probability/weight product (source line53 in commit1d2ad32). It reports10
errors in total; resources255 registers,30 spills,131,072 shared bytes.
No full operator runs under the sanitizer. `--minimal-memcheck` uses small
valid buffers and ten in-range seed rows solely to check address safety.

An optional `--split-bf16` path replaces the two wide tf32x3 products with
high/residual BF16 components. Its minimal sanitizer run completes with zero
errors and finite output,255 registers,zero spills,81,920 shared bytes.
The following full H1 real-layer00 gate is finite but relative L2 remains
0.004914889, maxAbs0.03125. It still fails0.002, so no performance or generation
test follows. Address-safety progress does not establish numerical adequacy.
This reassociation implementation remains rejected.

Receipts: `reassociated_fp32_memcheck.{json,log}`,
`reassociated_split_bf16_memcheck.{json,log}` and
`reassociated_split_bf16_h1.json`. Every run uses only clean GPU4 with monitoring.

## Vectorized shared winner/exponential reuse: rejected

The older scalar shared-exponential experiment added separate reads. The
bounded follow-up `analysis/build_c10_vector_exp.py` starts from retained
long-tiles84/tree arithmetic and packs score, exact exp2 and owner into a
16-byte shared vector. Each consumer reads two vectors, without adding a
barrier; column strides40 and42 test the two proposed shared layouts. Both
fit the existing16 KiB staging region and preserve the mass/entropy order.

Both complete H5 K/V/bias outputs are bitwise equal on real layer00. Eight
interleaved samples per method give21.7112 versus19.6458 ms for stride40,
and21.7916 versus19.6924 for stride42. Both variants require128 registers
and24 local bytes (57,344 shared bytes,four resident CTAs), versus retained
116 registers andzero local bytes. They are slower and are not expanded.
Receipt `vector_exp.json`; only H5 merge is measured, no complete-call or
generation claim. This vectorized reuse path is closed. Full40 best stays
267.16555 ms and150 ms remains unmet.
