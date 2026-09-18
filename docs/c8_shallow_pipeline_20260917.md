# Exact C8 shallow pipeline exploration

The user requests a complete H40 attention call at or below **140 ms** on
scai7. This supersedes the original 150 ms target. No tested candidate in
this report has met the target or replaced the retained implementation.

The fixed control is commit `57f269f`, fused FP32 Fisher selection plus
`kernels.best.c8_r2`. The merge source SHA256 is
`e842a82cb6d3e95672749f598fd7aca9781cc1f840b2d14058d6b8f9104cd486`.
All candidates preserve exactly eight merged representatives, the selector,
the full compressed second stage, and training-free/all-layer eligibility.
No singleton, dense exception, video generation, or quality claim is added.

## Completed rounds

Mean complete-call latency over real step-0 layers 00/19/39, H40 D128:

| Candidate | Candidate ms | Paired control ms | Same-run SDPA / candidate | Registers | Local bytes | CTAs/SM |
|---|---:|---:|---:|---:|---:|---:|
| Two query tiles, serial, LB3 | 173.667 | 164.174 | 1.919 | 167 | 0 | 3 |
| Two query tiles, overlap, LB3 | 180.067 | 163.728 | 1.851 | 168 | 16 | 3 |
| Two query tiles, overlap, LB4 | 208.190 | 166.488 | 1.601 | 128 | 320 | 4 |
| Two independent warpgroups | 166.349 | 163.151 | 1.972 | 128 | 0 | 2 |
| Two warpgroups, alternating QK issue | 168.440 | 163.131 | 1.948 | 128 | 0 | 2 |
| Two warpgroups, alternating all three products | 181.415 | 161.119 | 1.797 | 128 | 64 | 2 |
| Two independent warpgroups, span 210 | 165.896 | 162.298 | 1.965 | 128 | 0 | 2 |
| Two warpgroups, all products, span 210 | 182.263 | 161.953 | 1.789 | 128 | 64 | 2 |
| N64 fragments, serial | 162.952 | 162.895 | 2.005 | 106 | 0 | 4 |
| N64 fragments, epilogue overlap | 162.900 | 162.856 | 2.006 | 120 | 0 | 4 |
| N32 fragments, corrected compact staging | 178.548 | 164.376 | 1.847 | 110 | 0 | 4 |
| Two warpgroups, handoff after weights | 165.423 | 163.050 | 1.981 | 128 | 0 | 2 |
| Two warpgroups, handoff after normalization | 166.108 | 163.101 | 1.973 | 128 | 0 | 2 |
| Static WG paths with uniform branch vote | 163.677 | 163.262 | 2.007 | 126 | 0 | 2 |
| Uniform WG index, independent | 164.163 | 163.306 | 1.999 | 124 | 0 | 2 |
| Uniform WG index, after-weights handoff | 163.888 | 163.478 | 2.002 | 122 | 0 | 2 |
| Balanced column groups, independent | 162.486 | 163.908 | 2.016 | 122 | 0 | 2 |
| Balanced column groups, after-weights handoff | 163.061 | 163.421 | 2.009 | 120 | 0 | 2 |
| Balanced columns, independent confirmation | 162.553 | 163.778 | 2.020 | 122 | 0 | 2 |
| Balanced columns, V-product query prefetch | 162.260 | 162.439 | 2.011 | 127 | 0 | 2 |
| Separate roles, one shared slot | 172.106 | 161.934 | 1.891 | 117 | 0 | 2 |
| Separate roles, two shared slots | 173.464 | 161.378 | 1.876 | 117 | 0 | 2 |
| Shared-query roles, LB2 | 188.398 | 161.123 | 1.730 | 121 | 0 | 2 |
| Shared-query roles, LB3 | 195.887 | 161.786 | 1.664 | 80 | 64 | 3 |
| Packed output, serial | 161.950 | 162.530 | 2.015 | 125 | 0 | 4 |
| Packed output, K/V overlap | 165.208 | 162.614 | 1.975 | 128 | 16 | 4 |
| Rotated V high-half staging | 160.359 | 162.107 | 2.036 | 125 | 0 | 4 |
| Rotated V high-half confirmation | 160.803 | 162.939 | 2.030 | 125 | 0 | 4 |
| Rotated full V staging | 163.538 | 162.530 | 1.996 | 128 | 0 | 4 |
| Rotated V high half, balanced columns | 163.739 | 162.430 | 1.994 | 124 | 0 | 2 |
| Rotated V high half, stage-diagnostic run | 160.500 | 162.144 | 2.032 | 125 | 0 | 4 |
| K/V overlap, enforced late prefetch | 161.622 | 162.325 | 2.019 | 128 | 0 | 4 |
| Rotated V high half, span 21 | 162.915 | 163.309 | 2.000 | 125 | 0 | 4 |
| Rotated V high half, span 42 | 162.167 | 163.258 | 2.010 | 125 | 0 | 4 |
| Rotated V high half, span 210 | 162.666 | 163.117 | 2.004 | 125 | 0 | 4 |
| Rotated V high half, empty-slot diagnostic | 160.781 | 162.280 | 2.023 | 125 | 0 | 4 |

The first sixteen do not improve their paired controls. Balanced independent
columns and rotated V high-half staging retain small savings in confirmation; other candidates have either
unresolved small differences or clear regressions. All pass raw bitwise checks
for the entire H5 merged K/V/bias workspace and complete H40 outputs on all
three captures; seeds match and outputs are finite. The unpaired 21st query
frame is included. Screening uses 16 ABBA blocks, 32 candidate and 32 control
samples per layer/candidate; v12/v23 confirmation uses 64 blocks, 128 samples
per arm/layer. Every timed call recomputes selector, pack, all merge
chunks, and the full route. SDPA Flash is measured in the same run. Equal
capture weighting is valid because sample counts are identical.

Completed receipts in the experiment artifact directory:

- `pair_v0_attempt001_gpu0.json`: clean scai7 GPU0.
- `two_wg_v1_attempt003_gpu3.json`: clean scai7 GPU3.
- `two_wg_v2_attempt005_gpu4.json`: clean scai7 GPU4.
- `fragment_v3_attempt006_gpu4.json`: clean scai7 GPU4.
- `fragment_v7_attempt008_gpu2.json`: clean scai7 GPU2.
- `handoff_v8_attempt009_gpu3.json`: clean scai7 GPU3.
- `vote_v10_attempt010_gpu3.json`: clean scai7 GPU3.
- `uniform_v11_attempt011_gpu3.json`: clean scai7 GPU3.
- `columns_v12_attempt012_gpu3.json`: clean scai7 GPU3.
- `columns_v12_confirm_attempt013_gpu3.json`: clean scai7 GPU3.
- `earlyq_v13_attempt014_gpu4.json`: clean scai7 GPU4.
- `roles_v15_attempt015_gpu4.json`: clean scai7 GPU4.
- `roles_v17_attempt016_gpu4.json`: clean scai7 GPU4.
- `packed_v18_attempt017_gpu4.json`: clean scai7 GPU4.

Other attempts that encountered foreign compute processes are invalid and
are excluded. Absolute times vary with device/run conditions; decisions
use the interleaved same-run controls, not an older 157 ms measurement.

## What the machine code establishes

The paired-tile overlap version increases register lifetimes and ptxas
inserts a wait after each matrix instruction: 36 HGMMA / 36 DEPBAR, versus
36 / 6 for its serial control. It does not implement the intended overlap.
The LB4 version also spills heavily. Repeating this schedule with only a
launch-bound change is not justified.

The two-warpgroup version shares immutable K/V while giving each warpgroup
its own score registers and 16 KiB output staging. It uses 72 KiB dynamic
shared memory, 128 registers, no local accesses, and 18 HGMMA / 3 DEPBAR.
It resolves the compiler serialization but does not improve complete calls.

The v2 all-stage named-barrier schedule and a tile-span-210 control were
tested and rejected. All-stage scheduling again compiles into per-matrix
waits and local accesses, consistent with its substantial regression.

## N64 fragment epilogue pipeline (v3, no measured gain)

Split the N128 K and V products into K0, K1, V0, V1, each with N64. Two
32-float accumulator banks fit within the original 64-float accumulator
budget. After K0 completes, issue K1 while converting and staging K0; then
issue V0 while staging K1 and issuing the K TMA store; finally issue V1 while
staging V0. Every WGMMA operand/accumulator and TMA scratch reuse retains
its required wait. Both output halves are present before their TMA store.

There is an otherwise identical serial N64 control. Both compile without
local accesses: serial 106 registers, overlap 120 registers, 28 HGMMA and
5 DEPBAR each. SASS confirms four STSM instructions between the pending
matrix operation and its wait in each of the three overlap stages, including
four UTMASTG instructions in the middle stage. The serial control has no
such intervening stores. This establishes the intended instruction schedule.
Both variants subsequently pass the complete three-capture bitwise gates,
but neither improves full-call time. The approximate paired 95% interval
for each N64 candidate on each individual capture includes zero saving.
Correct asynchronous instruction overlap alone is insufficient here.

## Smaller staging workspace (v4, in progress)

An N32 follow-up stages and stores each quarter separately, using 4 KiB
instead of 16 KiB of output shared memory. Its five-dimensional tensor map
is `[128, 8, 945, AH, Q*80]`, with a `[32, 8, 1, 1, 2]` box, a 64-byte
shared-memory swizzle, and dimension-0 offsets 0/32/64/96. The logical output
layout and arithmetic remain fixed. The per-warp staging buffer is 1 KiB.
The descriptor follows the official [tensor-map API requirements](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html).

Reducing dynamic shared memory to 45,056 bytes makes five resident CTAs
possible in shared memory, but the register limit is also binding. LB5
serial/overlap variants compile with 32/64-byte stacks and per-matrix waits;
the LB4 overlap version uses 110 registers with no spills and preserves the
intended overlap. Delaying all query prefetch until the last V fragment
still leaves a 32-byte stack and per-matrix waits in LB5. These three LB5
compile failures are not submitted for GPU timing. The spill-free LB4
version fails the full three-capture raw K/V and H40 output gates while
bias remains exact; no candidate timings were collected. Receipt:
`fragment_v4_attempt007_gpu4.json`, clean GPU4.

A v5 follow-up packs the twenty integer owner labels into two 32-bit words,
three bits per label, using exact BFI/BFE operations. Winner values,
probabilities, arithmetic order and BF16 conversions are unchanged. This
targets integer register lifetimes that could prevent spill-free five-CTA
residency. However, the prefetch and late-prefetch versions still have
64/32-byte stacks and per-matrix waits; neither is queued for GPU timing.
Packing labels alone does not remove the limiting live register set.

The next v6 compile probe divides the ordered eight score instructions into
two batches of four, loading only 16 query words at a time. It waits before
reusing those operand registers, then accumulates the second K64 half into
the same scores. This trades one additional group wait for a smaller query
register footprint and keeps the score accumulation order. It uses the
N32 compact epilogue and late query prefetch. Both ordinary and explicitly
pinned input-operand versions compile without local accesses, but still
serialize all 48 matrix instructions; they are not submitted for timing.
An independent recompile records ptxas C7512 explicitly attributing this
serialization to insufficient register resources in H1 and H40. Thus a
zero-spill resource report does not prove that asynchronous execution is
preserved. Receipt: `compiler_diagnostic_v6.log`; the standalone diagnostic
command and retained PTX are in remote `compiler_diagnostic_v6/`.
No speedup is inferred from nominal occupancy or resource counts.

The v4 correctness failure is traced to SW64 staging. The original expression
XORed BF16 element bits 3:4 with 5:6. The correct `Swizzle<2,3,3>` uses bits
6:7 as the source, consistent with the official [CUTLASS layout definitions](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits_sm90_gmma.hpp).
The v7 source names preserve this fix separately from the failed receipts.
The corrected LB4 candidate passes all raw intermediate and H40 output
checks on three captures, but regresses to 178.548 ms versus 164.376 ms.
It is rejected. Reducing the staging allocation alone does not improve this
operator; it quadruples the static TMA issue sites while occupancy stays four.

## Scalar-to-tensor phase handoff (v8, rejected)

The original two-warpgroup QK handoff releases the peer immediately after
QK issue, leaving both warpgroups free to enter scalar normalization at
nearly the same time. v8 instead releases the peer after normalization or
after complete weight formation. Its purpose is to overlap one warpgroup's
K/V tensor products and stores with its peer's scalar work. The original
wait-before-QK handshake and the final odd-frame tail are preserved.

Both compile without local accesses or per-matrix serialization (18 HGMMA,
3 DEPBAR). The after-normalization variant has 125 static MUFU instructions,
versus 103 for the after-weights variant, so machine-code scheduling differs
beyond barrier placement even though mathematical operations are unchanged.
Both pass all three bitwise gates, but are slower than paired controls.

## Warp-uniform TMA coordinates (v9-v11)

The dynamic two-WG implementations contain 25 R2UR and eight BRA.U sites,
compared with two R2UR and no BRA.U in the single-WG N64 controls. Duplicating
the full loop with each WG index a literal (v9) does not fix this: both
compiled branches retain the broadcast loops. No GPU timing is requested
for that unsuccessful compile transformation.

In v10, the outer branch instead uses a warp vote over `tid < 128`. This
predicate is unchanged for every valid warp, while giving the compiler
explicit warp-uniform information. It removes all 16 broadcast-loop sites
from the two static paths and lowers R2UR count from 52 to three. H40 uses
126 registers, no spills, and the same four resident warpgroups. All three
raw intermediate and H40 output gates pass. Complete latency is 163.677 ms
versus 163.262 ms control, so removing the instruction overhead still does
not establish a net operator gain.

v11 computes the WG index directly from the equivalent warp vote and keeps
one copy of the computation. Both independent and after-weights handoff
versions have 18 HGMMA, three DEPBAR, two R2UR and no BRA.U or local accesses.
This avoids doubling the static instruction body. Both pass complete GPU
validation, but neither improves full-call latency. Every generated source was checked against the archived compile
hashes after extending the builder, preserving prior receipts.

## Balanced column-group distribution (v12, small confirmed saving)

The frame-split implementations assign 11 frames to WG0 and 10 to WG1 for
every column group. v12 instead assigns five of the ten column groups to
each WG and retains all 21 frames in each group. A 210-tile span covers the
whole column dimension, so both WGs perform exactly 105 query tiles and
every one of the 210 logical query tiles is covered once. K/V loading is
still shared. This tests removal of the odd-frame load imbalance, rather
than increasing the number of live score tensors.

The optional after-weights handoff has 105 forward exchanges and 104
backward exchanges; WG0 skips the initial wait and WG1 skips its final
unused signal. The independent control has no inter-WG handoff. Both
compile with 18 HGMMA, three DEPBAR, two R2UR, zero BRA.U and no local
accesses. Both pass all three captures. The independent version saves
1.556, 1.452, and 1.259 ms on layers 00, 19, and 39, respectively; the
approximate paired-block 95% intervals exclude zero in this screening run.
Its equal-capture mean is 162.486 ms versus 163.908 ms. The handoff version
saves 0.360 ms on average, with every layer's interval including zero.
A prespecified 64-block-per-layer confirmation saves 1.278, 1.224, and
1.173 ms, with approximate paired 95% intervals [0.827, 1.729],
[0.854, 1.594], and [0.920, 1.427] ms. The candidate wins 157 of 192
blocks. Its 162.553 ms mean versus 163.778 ms control is a 0.75% saving
on these three captures. This limited improvement does not reach 140 ms
and has not replaced the retained kernel.

## Reproduction and monitoring

The v13 compile probes move the next query's register loads between merge
WGMMA commit and wait, starting from v12's independent balanced schedule.
The three arms move loads for K, V, or both products. Their addresses and
arithmetic are unchanged. K-only and both-product prefetch compile with
64/88-byte stacks and 18 waits for 18 matrices; both are rejected without
GPU timing. V-only uses 127 registers, no local accesses and three waits.
Its V-product interval contains five global vector loads, versus four in
v12: ptxas already overlaps some loads in the original schedule. Only the
V-only arm passes all three gates. Its 162.260 ms mean versus 162.439 ms
control is a 0.180 ms difference; two of three paired intervals include
zero. It does not provide convincing further improvement. No speedup is
inferred from instruction counts. Historical cross-stream merge/route
overlap remains rejected.

v14 is a separate cross-query compile probe based on the N64 fragment
schedule. It primes the first QK product before the query loop. After the
current tile's final V product completes, it loads the next query's remaining
register fragment and issues its QK product before staging and enqueuing the
current V output. The next loop iteration waits before reading those scores.
The last query skips the unused QK issue; column spans retain independent
prologues. Compilation rejects this candidate: H40 reaches 128 registers,
256 bytes of stack, 64 LDL and 45 STL sites, and 36 DEPBAR waits for 36
HGMMA instructions. The intended cross-query overlap is not retained. No
GPU correctness or timing run is queued for this spilling/serialized source.
Receipt: `compile_v14.json` and `sass_v14.json`.

## Separate score and merge roles (v15, rejected)

v15 gives the first warpgroup QK, normalization, owner selection, bias and
weight formation; the second warpgroup receives the same twenty uint32 RS
operand words per thread and performs K/V tensor products and output stores.
This moves the overlap across two register files instead of holding next-QK
scores and the previous V accumulators in one warpgroup. The full CTA shares
the immutable K/V tile. Arithmetic, reduction order and BF16 rounding remain
the retained operations; the transfer adds no numeric conversion.

One-slot and two-slot candidates allocate 67,584 and 77,824 dynamic shared
bytes, respectively, and use 256 threads. Filled and free named barriers
have separate ids and explicit 256-thread arrival counts, following the
[PTX producer/consumer synchronization pattern](https://docs.nvidia.com/cuda/archive/12.1.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier).
The producer waits for reuse only after a slot's initial fill. The consumer
resolves all twenty shared loads before releasing that slot, then uses
independent register operands. Slot and iteration counters continue across
column spans. Unconsumed final free notifications have no waiting producer.
Both H40 variants compile at 117 registers with zero stack and no local
accesses. They preserve 18 matrix instructions and three waits, avoiding
v14's resource failure. Both pass all three full intermediate/output gates,
but regress to 172.106 and 173.464 ms versus paired controls of 161.934 and
161.378 ms. Both have two resident CTAs. Decoupling register files avoids
serialization but does not compensate for the role schedule's costs here.
Receipts: `compile_v15.json`, `sass_v15.json`, and
`roles_v15_attempt015_gpu4.json`.

v16 uses one slot, exact three-bit owner packing in the producer, and serial
N64 fragments in the consumer to reduce live accumulator storage. LB2 and
LB3 compile probes test whether this can admit three resident CTAs without
local accesses or matrix serialization. LB2 uses 113 registers and five waits
with no spills; LB3 falls to 80 registers but adds a 152-byte stack and 28
waits for 28 matrices. Thus it does not achieve spill-free three-CTA residency.
Neither is queued while the specific resource bottleneck is investigated.

v17 removes the producer's 32-word Q register operand. The one shared slot
is enlarged from 10 KiB to 16 KiB and first holds losslessly unpacked Q for
an SS score product. After QK completes, the same slot holds the unchanged
twenty RS weight words for the consumer. The producer waits for slot reuse
before loading Q, uses a separate 128-thread producer barrier and proxy
fence, and waits for QK before overwriting Q. The consumer still releases the
slot only after copying every weight word. The total dynamic allocation is
73,728 bytes. CPU symbolic checks confirm the pack transpose is its own
inverse and all 8,192 query elements occupy distinct in-bounds locations.

It also rematerializes the final owner-normalization `mul.rn.f32`, preventing
reuse of an earlier scaled-score temporary without changing arithmetic.
This preserves the original fused subtract in the stable fallback path.
LB2/LB3 are compile probes; full raw gates are required because the QK operand
layout has changed. LB2 compiles at 121 registers with no spills. LB3 uses
80 registers and a 64-byte stack (20 LDL/19 STL sites), but both retain five
waits for 28 matrices. Unlike v16 LB3, this preserves grouped asynchronous
execution. Both were tested to measure whether the reduced register limit's
additional residency outweighs its local traffic. This bounded empirical
test relaxes the earlier spill-free screening heuristic; spills without
matrix serialization need not be rejected solely from instruction counts.
Both pass every raw intermediate, output, seed and finite gate. LB2 measures
188.398 ms versus 161.123 ms paired control; LB3 measures 195.887 ms versus
161.786 ms. All three layers show large regressions. GPU resource queries
confirm two and three resident CTAs, respectively, so increased residency is
real but does not improve this operator. Both are rejected. This result does
not isolate the cost of each bundled change (SS query, owner packing,
rematerialization and split consumer); it does close their tested combination.
The confirmed balanced-column candidate remains the only repeatable small
gain in this campaign, and the 140 ms goal is still unmet.

## Pack completed outputs before overlap (v18-v20)

v18 retains full N128 K/V products. It converts each completed 64-float
output to the same final 32 uint32 BF16 pairs, stages the first half of K,
and issues V before staging/enqueuing the remaining K half. Only sixteen
packed K words must coexist with V accumulators. A serial packed-output
control isolates the added scheduling. A third arm also issues the next
query's QK before the final V half is staged, with a loop-carried score.

The serial control uses 125 registers with no spills. K/V overlap retains
18 HGMMA and three waits but has a 16-byte stack. Its V interval includes
four STSM, four TMA stores and four query loads after the final matrix
instruction and before the wait, demonstrating instruction overlap. The
cross-query arm serializes all 26 matrices and has a 248-byte stack; it is
not queued. v19 explicitly resets loop-carried scores before the next
ScaleOut::Zero product. This does not change either v14 or v18 cross-query
resource/scheduling failure, so the old-score-value hypothesis is rejected.

v20 checks two narrower lifetime changes: retain only a quarter K output
while issuing V, or defer Q prefetch until V completes while retaining
half-output staging overlap. Both still have 128 registers and 16-byte stacks.
The late-prefetch arm has exactly the same H40 instruction hash as v18 K/V
overlap, so it is not timed again. The quarter-output arm overlaps two STSM
sites instead of four and retains the same spill sites; it is not queued.

v18 serial and K/V-overlap both pass all three raw gates. Serial takes
161.950 ms versus 162.530 ms control, with two layer intervals including
zero. K/V overlap takes 165.208 ms versus 162.614 ms, with every layer's
interval showing a regression. Demonstrating asynchronous issue and retaining
four resident CTAs is therefore insufficient for a net gain in this schedule.

v21 compiles two queries within one loop body. Each score accumulator is
scoped locally; no outstanding score crosses the back edge. The first
query's V output is packed, partially staged, and completed while the second
query's QK is issued. The second query is then completed conventionally.
The final odd query uses a single-query tail. This diagnoses whether the
loop-carried asynchronous score, rather than its old value, triggers the
compiler resource failure. Both retain six waits for 36 matrices, but have
56/104-byte stacks and twenty TMA uniform-broadcast loops. v22 uses a fixed
pair counter and warp-uniform tail condition. This removes every broadcast
loop (R2UR 42 to 3) but retains a 48-byte stack. None is GPU-timed yet.

v23 rotates the loop instead: only sixteen completed uint32 BF16 V words
cross the back edge. A fresh, local current-query QK is issued, the prior
query's V high half is staged/enqueued, and then QK is waited on. The bottom
of each iteration stages the current V low half and saves its high half.
Each column span explicitly drains its final query before starting a new
span. K store reuse waits and the final kernel TMA drain are preserved.

This compiles with 125 registers and no local accesses or stack, 18 HGMMA,
three DEPBAR and no BRA.U. The QK interval has four STSM and four TMA stores
after its final matrix instruction and before the wait. Thus continuous
cross-query overlap is finally compiled without increasing dynamic matrix
count or nominal occupancy cost. The extra static four TMA issue sites are
the per-span final-query drain, not extra output copies.

The screen passes every raw gate at 160.359 ms versus 162.107 ms control.
Independent confirmation uses 64 ABBA blocks per capture and passes every
gate again. It measures 160.803 ms versus 162.939 ms: a 2.136 ms saving,
about 1.31%. Layer 00/19/39 savings are 2.361/2.197/1.851 ms, with approximate
95% intervals [2.014,2.708], [1.907,2.486], [1.598,2.104] ms. The candidate
wins 185 of 192 paired blocks. Receipts are
`rotated_v23_attempt018_gpu4.json` and
`rotated_v23_confirm_attempt019_gpu4.json`. This remains above both the
original 150 ms threshold and the subsequent 140 ms goal.

v24 tests carrying all 32 packed V words, and combining the partial rotation
with v12 balanced columns. The first has 128 registers and eight STSM inside
the QK-to-wait interval; the second has 124 registers and four STSM. Both
retain three waits and no spills. However, all-gates-passing full calls are
163.538/163.739 ms versus paired controls 162.530/162.430 ms. These schedules
are not promoted. The full-V/balanced combination has an eight-byte stack
and is rejected at compilation screening. The two timed variants are in
`rotated_v24_attempt020_gpu4.json`.

v25 diagnoses the earlier loop-carried-score serialization by moving the
initial QK wait into the prologue and each next-QK wait to the loop tail,
after V staging. Only completed scores cross the back edge. Two packed
epilogue variants both retain 26 waits for 26 matrices and 224/248-byte
stacks. They are rejected before GPU execution: completing the scores
before the back edge alone does not solve this compiler resource failure.

v26 tries to delay Q0 prefetch until V has completed, using an accumulator
dependent bitwise-AND-zero offset. ptxas folds that dependency away. The
K/V version is instruction-identical to v18 (SHA256
`85da871016f7c0fcb2059ae953ebf45ba4bc04bc6680e2a5badfb26eb2e2245c`),
and still has a sixteen-byte stack. Combining it with partial/full rotated
V staging yields 48/32-byte stacks. All retain four Q loads in the V
matrix-to-wait interval. None is GPU-timed.

v27 replaces the foldable zero with two XOR-1 warp shuffles followed by a
bitwise XOR with the original accumulator word. The lane permutation is
an involution, so the offset is exactly zero for every bit pattern. This
tests whether a surviving dependency can move the loads and reduce their
overlapping lifetime. The K/V-only version has 128 registers, no stack or
local accesses, three waits, four late K STSM/TMA stores inside V, and no
Q loads in that interval. Its rotated partial/full combinations have
40-byte stacks and are rejected before GPU timing.

`prefetch_v27_attempt022_gpu4.json` passes all raw gates. It takes 161.622 ms
versus 162.325 ms control, but all three layer intervals include zero. This
does not establish an improvement beyond the v23 confirmation. Removing
the spill fixes the resource issue without yielding a large latency gain.

v28 applies exact three-bit owner packing, rematerialized `mul.rn` owner
scaling, and both to the combined partial-rotation/KV-overlap schedule.
These are compile-screening experiments to reduce the remaining live
register pressure; no arithmetic reduction order or capacity is changed.
Owner packing and the combined change have 64-byte stacks and serialize
all eighteen matrices. Rematerializing scaling alone retains a forty-byte
stack. All three are rejected before GPU timing.

v29 retunes the CTA span for the confirmed v23 kernel: fixed launch spans
21, 42 and 210 are compared with the retained span-105 control. Only the
host launch value changes; the kernel body and its arithmetic are unchanged.
The purpose is to check work scheduling and reuse after the pipeline change,
not to infer latency from identical resource counts. All three H40 kernel
instruction hashes match v23, with 125 registers, no local accesses and
three waits. `spans_v29_attempt023_gpu4.json` passes every raw gate. Spans
21/42/210 measure 162.915/162.167/162.666 ms versus their same-run controls
163.309/163.258/163.117 ms. Every interval includes zero for spans 21/210;
two of three do for span 42. None establishes a gain over the independently
confirmed span-105 v23 schedule. Span 105 remains the experimental choice.

Subsequent benchmark receipts add six diagnostic serial-stage samples per
arm using CUDA event boundaries, with one synchronization after the full
call. Selector, pack, each H5 merge and each H5 route retain their original
order. Instrumented stage times are separate from the uninstrumented ABBA
latency and are not replacements for it. Hardware counters on this host
are administrator-restricted (`RmProfilingAdminOnly: 1`), so no Nsight stall
measurement is claimed.

The first diagnostic receipt, `rotated_v23_stages_attempt021_gpu4.json`,
passes all gates and records separate ABBA full-call means of 160.500 ms
candidate and 162.144 ms control. Instrumented candidate means across
captures are approximately 8.19 ms selection, 0.386 ms packing, 92.08 ms
merge and 59.62 ms route. These establish stage costs, not stall causes.
Reaching 150 ms therefore requires roughly another 11 ms complete-call
saving beyond the independently confirmed v23 result.

## Empty slots and bounded-roundoff experiments

`rotated_v23_empty_attempt024_gpu4.json` measures all H40 heads and every
eight-slot block on the three real captures, outside latency regions. The
empty-slot fractions are 3.6374%, 1.8102% and 3.3107% for layers 00/19/39.
All bias entries are finite or the expected positive-infinity sentinel.
This is a small potential reduction in K/V stores, so no sparse-store
implementation is promoted or timed from this diagnostic alone.

v30 reuses the normalization's exponentials, replacing `exp2(1-u)` by a
cached exponential divided by the corresponding normalization sum. The
normalization, owner-key arithmetic, selector, representative count and
full route stay fixed, but rounding changes. Serial and rotated variants
have 128 registers, sixteen-byte stacks, three waits and 84 static MUFU
instructions instead of 103. Neither passes the complete numerical gate.

Named arithmetic candidates use a separate, prespecified comparison
contract in `analysis/c8_numerical_gate.py`; all other sources still require
bitwise equality. Full H5 merged K/V and complete H40 output require
relative L2 error <= 1e-3 and maximum absolute error divided by reference
peak <= 1e-2. Finite bias values use limits 1e-5 and 1e-4 respectively.
Special-value patterns must match, NaNs fail, and selected seeds must be
identical. Seven independent CPU cases test the comparison, including
known nonzero errors, small perturbations, infinities, NaNs and zero
references. These are operator comparisons, not video-quality metrics.

In `exponential_v30_attempt025_gpu4.json`, both variants fail layer 00:
output relative L2 error is 1.826e-4 but peak error/reference peak is
0.016423, above the fixed 0.01 limit. Raw K/V/bias error limits pass.
Layers 19/39 pass but do not supply a valid three-layer aggregate latency.
No threshold was relaxed after observing the failure.

v31 retains the original `1-u` input and replaces only its final exponential
with FP32 FMA polynomials of degree five or six after integer range
reduction. FTZ underflow and exceptional fallback are retained. A CPU scan
of 1,000,001 inputs in [-126,0] gives maximum relative function errors
3.291e-6/2.216e-7 versus FP64 exp2. Both compile with 123 registers, no
stack/local accesses and three waits; static MUFU sites remain for fallback.
`polynomial_v31_attempt026_gpu4.json` rejects both on layer 00 with the same
0.016423 output peak ratio. Passing layers 19/39 take roughly 181-183 ms,
well above their 159-161 ms controls. Low scalar function error is therefore
insufficient to validate the complete attention call or establish speedup.

v32 returns to the original arithmetic and combines K and V into one
N256 WGMMA product. The shared N64 panels are already adjacent in the order
K-low/K-high/V-low/V-high, preserving the per-output K16 accumulation
sequence. The existing rotated V epilogue consumes the appropriate output
half. The two-CTA build uses 244 registers, no local memory and two waits.
The three-CTA build spills and is rejected at the CPU screen. The two-CTA
build passes every original bitwise gate in
`fused_v32_attempt027_gpu4.json`, but takes 181.288 ms per complete H40 call
versus its paired 159.490 ms control (1.799x same-run SDPA). Saving one wait
does not compensate for the reduced resident work.

v33/v34 explore a dynamically shared register pool between two warpgroups:
96/160 or 104/152 registers for scalar/QK and fused shared/shared K/V phases.
The packed weights are transposed into shared memory before expansion. A
symbolic layout audit verifies all 5,120 valid weights and 1,024 zero padding
elements without collisions. v34 adds explicit release/acquire handoffs
between groups; v33 is a compile-only sketch and must never be queued.
Both versions currently fail compilation with ptxas C7602: the static
128-register target is insufficient (the first reported instruction needs
at least 154). Neither version has run on a GPU. Failed receipts are
`compile_v33.json` and `compile_v34.json`.

v35 gives each warpgroup a separate compile-time control-flow region, while
retaining v34's handoffs. It also fails with C7602 at the 128-register target
(minimum reported requirement 154), so no dynamic-register source is queued.

v36 instead retains the confirmed v23 shallow pipeline and original
arithmetic. One variant interleaves the two independent normalization
left-folds; another accumulates the two sums and two first moments as each
winner is computed; a third combines them. Each per-row add, FMA and shuffle
sequence is preserved, including the first moment's original initial FMA.
All three compile without local accesses and preserve eighteen matrices /
three waits. `interleaved_v36_attempt028_gpu4.json` passes all raw and H40
bitwise gates on the three captures. Normalization-only measures
160.613/162.924 ms candidate/control (2.034x same-run SDPA), moments-only
162.531/163.174 ms (2.010x), and combined 162.664/163.223 ms (2.008x).
The latter two have all three paired confidence intervals crossing zero.
Normalization-only has a small paired gain versus the frozen control, but
does not establish improvement over independently confirmed v23.

v37 is a diagnostic-only clock sampler for six intervals inside one steady
qo=10 iteration of every H5 CTA/column span. Its conditional sampling branches
force eighteen WGMMA waits and introduce six local loads / three stores, so
it is rejected before GPU execution: those intervals would not describe the
original asynchronous schedule. v38 changes sampling to predicated inline
PTX with no intervening control-flow branch. It must preserve the original
three waits and avoid local accesses before being used for diagnosis.
Clock samples include scheduling delay and instrumentation overhead; they
are not privileged performance-counter attribution or isolated stage costs.

v38 still spills and serializes all matrices. v39 relaxes the compile target
to three CTAs: 166 registers, zero local memory, eighteen matrices / three
waits. `clock_v39_attempt029_gpu4.json` collects 47,250 complete samples at
layer 19 and passes all raw/output bitwise gates. Mean sampled phase shares
are QK plus prior-V store 16.69%, scalar reductions plus K issue 37.70%, K
wait 2.60%, K store/prefetch plus V issue 28.06%, V wait 2.63%, and V
conversion/store/prefetch 12.30%. This suggests looking beyond the two
explicit K/V waits, but changed residency and instrumentation prohibit
using these fractions as an exact limit for v23. Its instrumented 172.051 ms
versus 161.496 ms control is a single-capture diagnostic, not a candidate.

v40 preserves v23's pipeline and applies a fractional L2 eviction hint to
TMA output stores (0.5 or 1.0). The hypothesis is that streamed merged
outputs needlessly displace repeatedly used Q fragments. The
[PTX TMA-store cache-policy operand](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
changes a cache hint, not synchronization or output arithmetic. Both builds
have no local accesses and preserve eighteen matrices / three waits.
`cache_v40_attempt030_gpu4.json` passes all three captures' bitwise gates.
The half/all fractions take 161.249/161.685 ms against paired controls
163.109/163.292 ms, with same-run SDPA speedups 2.025x/2.019x. These do not
establish improvement over confirmed v23; no cache policy is promoted.

v41 assigns separate 16-KiB shared store buffers to K and V. Per-warp TMA
groups alternate K,V; before reusing a buffer, `wait_group.read 1` retires
all but the most recent group (which reads the opposite buffer). The initial
iteration, column-span boundaries and final odd query retain this ordering;
kernel exit drains every group. Shared storage rises from 56 to 72 KiB, so
at most three CTAs fit. No arithmetic is changed.
`stores_v41_attempt031_gpu4.json` passes every raw and complete H40 bitwise
gate, but measures 169.264 ms versus 162.298 ms control (1.925x same-run
SDPA). Every layer's paired interval shows regression. It uses 127 registers,
no local memory, eighteen matrices and three waits. The independent output
buffers are rejected: in this implementation reduced store waiting does not
compensate for the lost residency.

v42 tests five independent WGs per CTA (640 threads), sharing the raw K/V
and keeping five private 16-KiB output tiles. With the usual 105-query span,
each WG owns one column group and all 21 frames. Moving Q register loads
inside the consuming iteration still leaves spills: both N64 variants have
7 LDL / 4 STL and serialize all 28 matrices; packed-owner N128 has 46/43
local accesses and serializes all eighteen matrices. They are not GPU-timed.

v43 instead stages Q into each WG's output tile for SS QK. A full-WG barrier
after every issuing warp's TMA-read wait is required, because the Q swizzle
crosses the previous per-warp output slices. Another barrier publishes Q
before the async-proxy fence and score issue. Five-WG N64 regains five waits
for 28 matrices, but has 14 LDL / 11 STL. Five-WG N128 serializes; the balanced
six-WG N64 variant keeps five waits but has 43/36 local accesses. Six WGs
split two column lanes and three frame lanes, each handling 35 query tiles.

v44 combines the six-WG layout with the previously validated, exactly ordered
streamed moments, with or without interleaved normalization. Both preserve
five waits; local-access counts remain 29/25 and 64/50 respectively.

The new CPU-only `audit_c8_spill_sites.py` maps local instructions back to
source. In five-WG N64, several spill slots hold loop-invariant shared-Q
addresses. v45 recomputes these from a volatile thread-ID read at use. This
reduces five-WG local accesses to 6 LDL / 3 STL (96 registers, 16-byte stack).
The six-WG combination still has 25/18 accesses and is not queued. A bounded
exception admits the five-WG version to test the new concurrency/cost tradeoff:
`resident_v45_attempt032_gpu4.json` passes every raw and H40 bitwise gate but
takes 187.359 ms versus 160.561 ms control (1.740x same-run SDPA). Five resident
WGs do not offset Q staging and this schedule's overhead. It is rejected.

v46 retains RS Q and replaces each N80 score product by ordered N40 halves.
The first half is saved as exact FP32 bits in a 2,560-byte private slice per
warp after that warp's previous TMA read completes. The second half reuses
the score bank, then reloads the first half; Q operands need not be staged or
transposed. Each output retains the same eight K16 accumulation steps.
`audit_c8_multigroup_layout.py` checks all 210 query/column pairs, all 5,120
score coordinates, disjoint scratch offsets, and 10,240 descriptor-offset
relations. These integer checks do not replace GPU bitwise validation.

Five-WG N64 compiles with 95 registers, zero stack/local accesses, 36 matrices
and six waits. N128 spills and serializes, so it is rejected before GPU work.
Six-WG N64 retains six waits with 80 registers and a 32-byte stack (15 LDL /
9 STL). `staged_v46_attempt033_gpu4.json` passes all raw and H40 bitwise
gates. Five/six WGs take 169.498/179.066 ms versus paired controls
162.727/161.639 ms (1.927x/1.824x same-run SDPA). Every capture's paired
interval shows regression. Higher residency does not compensate for this
schedule's score staging and, for six WGs, remaining local accesses.

v47 preserves v23's four-CTA loop and adds one-time startup spacing of
`(blockIdx.x % 5) * gap` SM clocks, with gaps 1,024 and 2,048. The ID is only
a heuristic, not an assumption about CTA placement. A clock loop with
`nanosleep.u32 64` in thread zero and one CTA barrier precedes the query loop.
Both variants retain 125 registers, no local accesses, eighteen matrices and
three waits. `staggered_v47_attempt034_gpu4.json` passes every bitwise gate
and measures 161.206/161.330 ms versus paired controls 162.607/163.188 ms
(2.025x/2.023x same-run SDPA).
Neither establishes improvement over confirmed v23. No phase delay is
promoted.

v48 advances the V matrix issue to immediately after K shared staging. K
register outputs are already dead, and V only reads resident `sv`, so K TMA
issue can proceed while V is in flight. Two variants place the first half
of next-Q prefetch before or after the V wait. They retain v23's single
shared output buffer and its previous-V epilogue overlap with QK. CPU
compilation and SASS audit precede any GPU timing.

Both v48 source variants compile to the same H40 instruction hash
`e7f9324ea4670a5668cb2a9131cbecf33b42d8e466ec42a3277ccf7dde6b3edd`:
ptxas still schedules four Q loads before the V wait in both. There are
128 registers, no local accesses, four resident CTAs and eighteen matrices
with three waits. The K TMA issue now overlaps V in the emitted instruction
stream. `value_early_v48_attempt035_gpu4.json` passes all three captures'
bitwise gates. The two sequential measurements are 160.423 and 161.284 ms
against controls 162.854 and 162.853 ms (2.032x/2.021x same-run SDPA).
Since the machine code is identical,
these are repeated observations of one schedule, not two independent
improvements. The lower single mean does not establish a win over v23.

v49 tests a distinct residency point: four WGs per 512-thread CTA, two
CTAs per SM, targeting eight resident WGs with a 64-register limit. Shared
storage is 104 KiB per CTA. Two column lanes and two frame lanes cover all
210 query/column pairs once; odd/even frame lanes handle 55/50 queries.
The independent layout audit checks the complete coverage. N40 score
staging follows v46; three scalar schedules test plain, streamed moments,
and streamed moments plus interleaved normalization. Compilation must first
determine whether this smaller register budget is viable. All three builds
serialize all 36 matrices; plain/moments/both emit 139/138/202 local loads
and 107/106/163 local stores respectively. The eight-WG target exceeds this
implementation's practical register budget. All three are rejected before
GPU execution.

v50 changes only CTA traversal in the retained v23 pipeline. The flattened
grid interleaves heads, query-column spans, or both, testing whether adjacent
CTAs can improve K/V reuse or output traffic. All combinations still cover
the original Cartesian product of key blocks, active heads and spans. The
three builds retain four CTAs, no local accesses, eighteen matrices and
three waits, using 126/125/126 registers. `block_order_v50_attempt036_gpu4.json`
passes all raw and H40 bitwise gates. Head/span/head-plus-span variants take
165.511/162.836/168.712 ms versus controls 162.745/162.976/162.982 ms
(1.967x/1.999x/1.929x same-run SDPA). No traversal is promoted.

v51 extracts the unchanged stable-normalization fallback into a no-inline
device helper. Direct-path summation, fallback max/exp ordering, owner
selection and merge operations are unchanged. Its H40 build retains 125
registers, zero stack/local accesses and eighteen matrices / three waits.
The static MUFU count decreases from 103 to 83 because duplicated fallback
code is shared; this is a code-footprint change, not proof of fewer dynamic
exponentials. `outlined_v51_attempt037_gpu4.json` passes every raw/output
bitwise gate but measures 162.309 ms against 163.407 ms control (2.008x
same-run SDPA), worse than the confirmed v23 mean. It is not promoted.
These three real captures alone do not independently validate forced
fallback edge cases; further stress testing is deferred because this
candidate does not improve performance.

v52 follows a source-mapped spill audit of v49, which finds substantial
local traffic around query loads and matrix inputs in addition to scalar
reductions. Each N40 score half now executes ordered K64 steps, keeping
only sixteen Q words live. Arithmetic still accumulates the same eight K16
products in order. Eight-WG moments/both variants nevertheless serialize
all 36 matrices and retain 132/90 and 195/153 LDL/STL sites. Six-WG moments
keeps eight waits but has 23/17 local accesses. These builds are rejected
before GPU timing.

v53 moves exactly rounded BF16 weights into a dense, unswizzled K-major
8x8-atom layout. A WG uses 10 KiB for weights and a separate 8 KiB compact
N64 TMA output buffer. Four WGs require 112 KiB dynamic shared memory per
CTA, permitting a target of two CTAs/eight WGs by shared capacity; six WGs
require 148 KiB and target one CTA. Every warp waits before reusing its
output source; a WG barrier and proxy fence publish weights before SS MMA.
The weight A descriptor uses 128-byte leading and 1,280-byte stride offsets,
matching CuTe's `Layout_K_INTER_Atom` representation. Weights are stored as
the same packed BF16 bits; capacity and reduction order are unchanged.

`audit_c8_compact_shared_weights.py` checks all 5,120 weight coordinates,
packed-pair coverage, descriptor offsets, scratch/output isolation, TMA
output swizzle coverage and all 210 query/column pairs. It does not prove
GPU instruction semantics. Four-WG moments/both still serialize all
36 matrices, with 57/51 and 115/110 local accesses. Six WGs keep eight waits
but 14/14 local accesses remain. These builds are not GPU-timed.

v54 also stages the Q matrix in the same 18-KiB WG workspace, then reuses
its first 10 KiB for weights after the completed QK product. Q temporarily
overlaps 6 KiB of the output source: every issuing warp first waits for its
previous TMA read, and a WG barrier precedes any Q write. Another WG
barrier and proxy fence publish Q. This restores one full N80/K128 score
product and removes all Q operand registers. Shared capacity is unchanged.
The two four-WG builds fail C7602 in the H1 diagnostic specialization: the
64-register limit is below the compiler's reported minimum of 66. Six WGs
retain five waits for 28 matrices but have 13 LDL/13 STL sites. A bounded
exception admits only that six-WG source to a complete three-capture GPU
test, both to validate this new SS layout and to measure the tradeoff.
`shared_operands_v54_attempt038_gpu4.json` passes every raw H5 K/V/bias and
complete H40 bitwise gate on all three real captures. It uses 80 registers,
48 local bytes and one CTA/six WGs per SM, but takes 203.864 ms versus
161.361 ms control (1.602x same-run SDPA). The layout is correct on these
inputs; additional staging/synchronization and remaining local traffic do
not yield a performance benefit. It is rejected.

v55 assigns three independent frame lanes to each CTA and all ten column
groups to each lane, so each WG handles 70 queries. Two CTAs target six
resident WGs with the same 80-register budget as the six-WG CTA, but fewer
group-index calculations. Dynamic shared storage is 94 KiB per CTA. The
independent layout audit now includes its exact coverage. Shared Q/weight
ordering follows v54; plain streamed moments and interleaved normalization
are compiled separately.
Both retain five waits for 28 matrices, but moments has 14 LDL/14 STL and
the interleaved variant 49/38. They are not GPU-timed given unresolved
local traffic and v54's measured regression.

v56 removes only the optional H1 instantiation from the four-WG moments
source and explicitly requires H40/AH5. It still fails C7602 in H40 itself:
ptxas reports a minimum target of 66 registers versus the 64-register budget.
The native H40 failure, rather than the earlier H1 failure alone, rules out
this particular eight-WG SS schedule under the two-CTA register limit.

v57 splits that four-WG H40 score product into separate N64 and N16 banks,
retaining the original K16 accumulation order within each output element.
Both variants compile at 64 registers, but all 36 matrices are serialized
by 36 waits. Streamed moments has 48 LDL/41 STL sites and a 136-byte stack;
combined normalization/moments has 120/97 and a 240-byte stack. Neither is
GPU-timed. Merely reducing the score bank width does not make this schedule
fit the eight-WG register budget without spills.

v58 splits v23's existing 16-KiB TMA staging region into two 8-KiB halves
(2 KiB per warp per half), preserving its 56-KiB dynamic shared allocation.
K-low, K-high, V-low and V-high alternate between the two halves. V-high
remains packed in registers until the next QK issue, with a span footer
draining the last frame. The wait0 control drains all earlier TMA read
groups; the wait1 candidate permits the latest group, reading the opposite
half, to remain pending. No additional arithmetic or extra representatives
are introduced. The tradeoff is twice as many dynamic TMA store operations.
An independent integer audit checks all 8192 staging words, descriptor
addresses, row permutation and worst-case group reuse across 1/2/21 frames
and 1/2/10 spans. This does not replace GPU correctness gates.

Both sources compile with 118 registers, zero stack/local traffic, 18 HGMMA
and three matrix waits. Four STSM and four TMA stores remain under QK.
The shared-memory budget still permits four CTAs per SM. The guarded
`half_tma_v58_attempt039_gpu4.json` passes every raw H5 K/V/bias and full
H40 output/seed bitwise gate on all three captures, with no foreign GPU
process. Measured resources confirm four CTAs per SM. With 16 ABBA blocks
per capture, wait0 takes 164.390 ms versus 163.238 ms control; wait1 takes
164.974 ms versus 163.296 ms (1.983x and 1.976x same-run SDPA respectively).
Both regress. Additional overlap did not offset the half-store schedule's
cost in this experiment; the result does not isolate a hardware stall
cause. The initial
compile failed because the server's system disk was full. That receipt is
preserved as `compile_v58_diskfull.json`; compilation succeeded after using
the experiment filesystem for temporary files. The benchmark now places
its nvcc/Triton temporary files there as well.

v59 addresses the increased number of TMA stores with two- and four-warp
issuer groups, separately on v23's full buffer and v58's wait1 half buffers.
Each group stores adjacent columns in one combined box. The issuer waits
for the required prior read groups before a group barrier releases shared
storage to its writers. Every writer performs the async proxy fence, then
a second group barrier precedes the issuer's copy. This follows the
[CUDA TMA shared-to-global ordering example](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#asynchronous-data-copies).
The final kernel drain still waits on the actual issuing threads.

Full buffers retain their contiguous per-warp layout. Half buffers place
the low halves of adjacent warps together and then their high halves,
within the same 16-KiB allocation. `audit_c8_grouped_tma.py` independently
enumerates all source and tensor-box coordinates, verifies the SW128
layout and group participants, and includes v58's worst-case FIFO check.
Dynamic store counts per query become 4/2 for full buffers and 8/4 for half
buffers (two/four warps), versus 8/16 for v23/v58 respectively. The cost is
64- or 128-thread synchronization instead of warp-local synchronization.
All four compile with 18 HGMMA, three matrix waits and zero stack/local
traffic. Full-buffer variants use 125 registers, half-buffer variants 124;
the shared allocation and four-CTA resource budget remain unchanged.
QK still overlaps four STSM sites and two/one TMA issue sites for two/four
warps. Static TMA sites are 6/3 for full buffers and 10/5 for half buffers.
`grouped_tma_v59_attempt040_gpu0.json` completes 16 ABBA blocks per real
capture for all four variants. All raw H5 K/V/bias and full H40 output/seed
bitwise gates pass, timing is valid and there are no foreign processes.
Runtime resource queries confirm four CTAs and zero local bytes throughout.

| Issuer group / buffer | Candidate ms | Same-run control ms | Same-run SDPA ratio |
| --- | ---: | ---: | ---: |
| 2 warps / full | 166.188 | 166.794 | 2.021x |
| 2 warps / half | 170.300 | 166.456 | 1.972x |
| 4 warps / full | 168.069 | 166.563 | 1.998x |
| 4 warps / half | 171.262 | 166.599 | 1.961x |

Only two-warps/full has a lower paired mean (0.606 ms, 0.36%); each capture's
approximate 95% paired interval includes zero. This is a
small comparison against the frozen control, not evidence of superiority
over v23: v23 was not measured in the same GPU0 run. Absolute GPU0 timings
must not be treated as direct comparisons with earlier GPU4 timings.
The other three schedules regress; reduced TMA issue count alone does not
justify their synchronization/layout costs. No candidate reaches the target
or is promoted. The retained confirmed v23 result remains 160.803 ms on
GPU4. Workbook rows 1367-1372 contain the v58/v59 results.

The last CPU compilation was temporarily delayed by server I/O
(the compiler was observed in uninterruptible sleep) but completed.

v60 overlaps bias work with the independent K matrix product. The first
source computes the original two corrections before packing weights and
defers only their stores. Two further sources assign one output row to
each of the first two lanes in a four-lane group: an early control and a
late correction/store variant. A shuffle explicitly retains the original
storing lane's high-half dot; each new storing lane already owns the same
reciprocal/log operands that the original broadcast used. The operation
order and rounded weights remain unchanged. `audit_c8_bias_lane_mapping.py`
enumerates all 64 output addresses and their dot/reciprocal/log source
lanes; numerical and complete-output correctness still require GPU gates.

All three compile with 18 matrices, three waits, zero stack/local accesses,
and unchanged shared storage. Stores/early/late use 125/122/126 registers.
The stores variant has a normal global store after K's last matrix issue
and before its wait. The late variant also has two FMUL and one FADD in
that interval; the early control has none. SASS diagnostics now distinguish
ordinary STG from UTMASTG and count these arithmetic instructions. The
`bias_v60` job includes v23 as a separately labeled same-run reference,
plus all three candidates, with 16 ABBA blocks per real capture. The frozen
control remains unchanged. Runtime results are pending.

`analysis/build_c8_shallow_pipeline.py` deterministically generates sources
and prints a SHA256 manifest. `analysis/benchmark_c8_shallow_pipeline.py`
supports CPU-only compilation followed by guarded GPU correctness and ABBA
timing. `analysis/audit_c8_pipeline_sass.py` records instruction counts and
the matrix-to-wait intervals. These are experiment-only tools; production
dispatch stays unchanged.

Remote directory:
`/local2/denghaoran/GridMergeAttention/experiments/c8_shallow_pipeline_20260917`.
Its `source/` snapshot and dedicated extension/Triton caches isolate the run
from the shared working checkout. Local receipts live under
`C:/GMA/experiments/c8_shallow_pipeline_20260917`; provisional rows are in
the existing `C:/GMA/tmp.xlsx`.

The user explicitly authorized idle-GPU waiting and automatic launch.
`analysis/watch_c8_pipeline.py` reads `queue.json`, checks all scai7 H200s,
prefers GPU4 if idle, double-checks cleanliness, and starts one child on one
GPU. It never terminates foreign processes. Contention invalidates an
attempt and permits retry only on a freshly idle GPU. Other failures stop
for inspection. A PID lock prevents duplicate watchers, each job has a
15-minute default timeout, and `watcher.stop` ends the queue between jobs.
The target in queue metadata is 140 ms. `status_c8_pipeline.py` prints a
bounded read-only summary. `restart_c8_pipeline_watcher.py` only replaces
the verified owned watcher when it has no child, preserving completed jobs.

Do not promote a queued, contaminated, numerically failing, or stage-only
measurement. A promising full-call result still needs repeat confirmation
and wider input checks before changing the retained kernel.
