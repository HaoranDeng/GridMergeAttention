# Experimental CUDA fusion with a 64-token shared ring

This is a partial implementation and a plan for the remaining fusion. Warp
matrix primitives, isolated SQ1 merge, shared-ring routing, and selected
fused compressed partitions now have real-input checks. The separate SQ1
attention contribution, final partial combination, and complete fused
operator are NOT implemented. Current H5 fusion timings are far too slow
to justify promotion; first resolve the compiler/register bottleneck.
The Triton fused variants are rejected (2.1/3.2/10.7 seconds for C10).
Preserve the retained fa52897 sequential operator until the complete
candidate passes performance, correctness, and video-quality checks.

Implemented sources: `native_fisher/warp_mma.cuh`, `warp_merge.cuh` and their
diagnostic CUDA wrappers. `analysis/validate_fisher_warp_mma.py` validates
GEMM fragments and optionally calls `validate_fisher_warp_merge.py` to
compare against retained native H5 K/V/bias. These are diagnostic entry
points only; production dispatch is unchanged. Raw evidence is in
`experiments/gma_c4_c10_native_20260909/warp_*_gate.json` and the stricter
`warp_merge_c{4,10}_storage_ties.json` follow-up results. Additional sources
`warp_route.cuh` and `warp_fused.cu` implement online softmax and ring
overflow. Run `analysis/validate_fisher_warp_route.py` with optional
`--fused --fused-profile`; the latter times only all H5 compressed partitions,
not complete attention. Raw results use `warp_route_gate.json` and
`warp_fused_c10_*.json` in the same artifact directory.

## Scope and layout

- Keep native BF16 [1,21,45,80,40,128], exact Fisher Uniform16/probe4 selector,
  RF0, all compressed attention, and existing query-column SQ1. No VBench.
- One CTA covers two adjacent query columns,
  one query frame, one head, and up to 64 key blocks (a key block has 80 keys).
- The original four-warp version held two query tiles per even warp and
  spilled heavily. The eight-warp experiment uses one producer and three
  consumers per column; each consumer owns rows0–15,16–31, or32–44. Producer
  and consumer loops rendezvous at matched named barriers with all256
  threads. Extracting either the consumer or weight function did not
  improve speed; all currently measured fused variants are rejected.
- Use mma.sync m16n8k16 BF16 with manual fragments, rather than asking Triton
  to retain whole multidimensional tensors across five differently shaped
  dots. Reuse native FP32 normalization, winner rules, BF16 rounding before
  SQ1 residual division, and entropy correction.
- Per CTA shared memory: raw K/V 80x128x2 BF16 =40960 bytes; two columns of
  ring K/V, 64x128x2 BF16 each =65536 bytes; ring bias=512 bytes; overflow
  biases <=72 bytes. About107080 dynamic bytes (align128 =>107136), allowing
  two CTAs per H200 SM by shared capacity alone. The eight-warp layout
  would also need <=128 registers per thread, subject to allocation units;
  observed register counts restrict it to one CTA. Use measured occupancy.

## Ring overflow without extra shared K/V allocation

For C10, adding ten seeds to a partially full ring can cross token64.
Do not allocate an expanded 73-row K/V ring: that exceeds the two-CTA shared
budget once raw K/V is included. Reuse the raw-K shared buffer for overflow
after every producer has finished reading raw K. At most
2 columns *9 overflow tokens *128 dims *2 kinds *2 bytes =9216 bytes, within
the 20480-byte raw-K buffer.

1. All threads cooperatively load current raw K/V; CTA barrier.
2. Producer warps compute scores/ownership/weights/entropy and K merge. Store
   non-overflow K to ring; retain overflow K in the same accumulator registers.
3. All warps meet a CTA barrier before any producer overwrites raw K.
4. Producer warps store overflow K in raw-K scratch, then compute V merge from
   raw V and store ring V / overflow V into disjoint scratch regions.
5. CTA barrier before route reads. If pending+C >=64, all warps consume the
   64-token ring, including bias, then CTA barrier. Cooperatively copy
   overflow scratch back to ring start; barrier before the next raw-K load.
6. Final partial ring: zero invalid K/V rows and use +inf bias before routing
   so masked zero probabilities cannot multiply uninitialized NaNs.

The merge stage must retain even-warp local state across CTA barriers while
odd warps wait. Never put a CTA barrier only inside the producer branch.

## Matrix fragment maps to verify independently

For mma.sync m16n8k16, lane group g=lane>>2 and t=lane&3:

- FP32 C accumulator elements i=0..3 correspond to row g+8*((i&2)>>1),
  column 2*t+(i&1). Across N panels, add8*n_panel.
- A registers can reuse the losslessly packed native Q order: qreg[0..3]
  for each 16-dimensional slice, as native RS WGMMA uses the warp's M16
  fragment. This equivalence MUST be checked, not assumed from this plan.
- Score B (transposed K): b0 packs K[key_panel+g, d+2*t: d+2*t+2],
  b1 uses d+2*t+8. Load from shared raw K with native boff swizzle.
- Merge/PV B (ordinary K/V): b0 packs B[key_base+2*t, n_panel+g] and
  B[key_base+2*t+1, n_panel+g]; b1 uses key rows +8/+9. They are gathered
  along K, not contiguous feature elements. Verify against a tiny BF16 GEMM.
- Packed merge weights use native packed_weights[20] for K=80, four A
  registers per K16 slice. Route probabilities use the analogous16 registers
  for K=64.
- Swizzle ring row storage as row*128 + (d ^ ((row&7)*8)); this avoids
  obvious shared-bank conflicts for the score B load. Validate accesses.

For each 80-key seed-score row, reuse native reduce80 and dot80 order. Each
producer warp has16 seed rows (only C are valid). Its two row halves occupy
40 score floats per thread. Normalize them, then choose owner per key:
first compare row g with g+8 (ignore >=C), then reduce across g using XOR
shuffle distances4/8/16, retaining lower seed index on ties. The current
helper retains20 winner floats and20 owner indices before constructing both
halves' probabilities. Pack BF16 weights after entropy and singleton mass
are calculated; shortening these live ranges remains a possible compiler
experiment, not a demonstrated performance improvement.

## Online routing / SQ1 / reduction

- Consume compressed keys in64-token blocks. KEY_BLOCKS=64 makes each full
  partition C*64 tokens, aligned to the original64-token ordering for C4/C10.
- Scores are (QK*scale-bias)*log2(e). Each M16 query tile has two row maxima
  and sums per four-lane group. Use the original BF16 softmax-probability
  conversion before PV, accumulating FP32 output.
- Handle all-invalid chunks explicitly: zero probabilities, zero old scale
  when prior max=-inf, and preserve max=-inf/denom0 for an empty partial.
- Keep SQ1 separate from the compressed ring. A small Triton partial kernel
  attends all945 raw keys at the query column, in64-key blocks with zero
  bias, and writes one extra FP32 partial. This is the existing SQ1 support,
  not an additional dense/hybrid path.
- Combine the15 compressed partitions plus one SQ1 partition using FP32
  log-max rescaling and sums; convert to BF16 only at final output. The
  partition reduction changes floating-point order and requires direct
  original/native error checks even if merge K/V are exact.
- Partial workspace uses [21*80*H5,16,48,128] FP32 plus maxima/sums; include
  all partial writes/reads, selection, packing, every head and final reduction
  in full40 timings. Never report a small-tile timing as goal completion.

## Staged implementation gates

1. PASSED: tiny warp BF16 GEMMs establish A/B fragment maps and row/D stores.
2. PASSED: merge-only diagnostic for two columns, one frame/head and64 key blocks:
   compare K/V/bias against retained native H5 workspace slices. Require
   bitwise equality before adding routing when arithmetic is intended exact.
3. PASSED numerical gates: route-only ring and selected fused partitions.
   FAILED performance screen: all measured H5 fusion kernels are too slow.
4. Add SQ1 partial and final combine; test complete40 against retained native
   AND original reference, exact seeds, finite output, initial relL2<0.002.
   This threshold is only screening, not evidence of preserved video quality.
5. Only if materially faster: cover step0 layers0/19/39, later denoising
   steps/edge cases, all-layer production integration and PSNR/SSIM/LPIPS.

Use one freshly exclusive scai7 H200, monitor foreign PIDs, bounded diagnostic
timeouts, and >=30% account reserve (conservative wrap-up at65% consumed).

## Streamed merge fragments on scai6: rejected

After the user explicitly authorized scai6, `warp_fused_streamed_merge.cu`
tested a distinct register-lifetime change. Each eight-feature panel is
merged and stored immediately with four FP32 accumulators. An additional
4,608-byte shared K-overflow buffer replaces holding64 K accumulator floats
across a CTA barrier; V overflow still uses raw-K storage after all K readers
finish. Score/weight construction is inlined, and the feature-panel loop is
kept rolled. Dynamic shared storage increases from107,136 to111,744 bytes.

On one freshly exclusive scai6 H200 GPU4, both capacities passed nine selected
fused-output comparisons against native materialized K/V and the validated
warp route, with bitwise-equal normalized partial outputs. The tests include
the first/middle/final partitions and the13-row query tail. They do not cover
every partial output, later denoising steps, complete attention or videos.

| Capacity | Registers | Local bytes | Resident CTAs | All H5 compressed partitions ms |
|---|---:|---:|---:|---:|
| C4 | 255 | 64 | 1 | 408.921 |
| C10 | 254 | 64 | 1 | 467.359 |

Three event samples follow warmup and all foreign-process monitors are clean.
These are **partial-fusion diagnostic timings**, excluding selection, packing,
SQ1 contribution, partial combination and the other35 heads. They are not
complete-attention latency or speedup versus SDPA. The candidate remains too
slow and is rejected; the retained native operators are unchanged. Source
hashes, raw results and selected comparisons are in `streamed_merge_review.json`,
`warp_fused_streamed_merge_scai6.json` and `source/streamed_merge_round/`.

The next distinct hypothesis is operand loading: current `score_b` and
`merge_b` use scalar shared BF16 loads and packing. Candidate matrix loads
(`ldmatrix.x2` for score B and `ldmatrix.x2.trans` for merge B) must first pass
the existing tiny-GEMM fragment and native-storage gates. This is only a
proposed experiment, with no measured benefit yet. Do not rerun the streamed
merge unchanged or treat reduced local storage as proof of fast fusion.

## Matrix-load operand family: gates pass, fusion remains too slow

`native_fisher/ldmatrix/` is an isolated copy of the streamed-fragment family.
Score B loads use `ldmatrix.m8n8.x2`; merge/PV B loads use its transposed form,
replacing scalar BF16 loads and packing for both raw K/V and shared-ring K/V.
Each pointer addresses an aligned contiguous eight-BF16 row inside the
existing swizzle. The diagnostic buffers explicitly retain16-byte alignment.
The four validation entry points accept `--source-dir` and hash the selected
headers and CUDA source, leaving their default source family unchanged.

On freshly exclusive scai6 GPU4, integer/zero GEMMs match exactly and random/
real GEMMs pass the existing FP32-oracle error threshold; packed Q is exact.
C4 and C10 each pass320 real merge tiles plus320 zero-query tie tiles, with
bitwise-identical K/V/bias versus native. The first C10 attempt failed CUDA
driver initialization before computation; a separate cuInit check returned
CUDA_SUCCESS and unchanged code passed with a new receipt. Do not treat the
failed initialization attempt as a numerical or performance result.

Both capacities pass the route gates and nine selected fused normalized
partial-output checks bitwise against native materialization plus warp route.
All successful-run monitors are clean. The H5 diagnostics are:

| Capacity | Registers | Local bytes | Resident CTAs | H5 compressed partitions ms |
|---|---:|---:|---:|---:|
| C4 | 242 | 0 | 1 | 283.623 |
| C10 | 242 | 0 | 1 | 320.076 |

Dynamic shared storage remains111,744 bytes. These values are approximately
31% below the preceding scalar-load streamed-fragment run; this is a
comparison of separate exclusive runs, not an interleaved estimate. They
still exclude selection, packing, SQ1, final combination and35 heads and
cannot be presented as full-attention latency or SDPA speedup. The matrix-load
primitives are validated for further development, but this partial fusion
does not justify promotion or completing integration of the same slow layout.

`ldmatrix_review.json` records all gates, failed-attempt provenance, resource
counts and timing scopes. Exact sources are under `source/ldmatrix_round/`.
With spills eliminated but only one256-thread CTA resident, the next distinct
investigation is warp-role/CTA layout and its phase utilization using the
validated matrix loads. Do not repeat the old scalar-load variants unchanged.

## One-column CTA and whole-loop function boundaries: rejected

Two additional matrix-load layouts were tested on freshly exclusive scai6
GPU4. `warp_fused_one_column.cu` uses one producer warp and three consumer
warps in a128-thread CTA, with one query column per CTA. The public pair-range
API is preserved by launching two distinct column CTAs per pair. The64-token
shared ring and overflow buffers are reduced to one column; dynamic shared
storage is76,416 bytes. This duplicates raw K/V reads across the two columns.
`warp_fused_loop_calls.cu` additionally places the complete producer and
consumer loops in separate noinline functions with compile-time role selection.
Both roles retain the same128-thread named barrier sequence.

| Layout | Capacity | Registers | API local bytes | Resident CTAs | H5 partial ms |
|---|---:|---:|---:|---:|---:|
| One column | C4 | 242 | 64 | 2 | 299.139 |
| One column | C10 | 242 | 64 | 2 | 299.662 |
| Separate whole loops | C4 | 243 | 64 | 2 | 269.679 |
| Separate whole loops | C10 | 243 | 64 | 2 | 314.326 |

All route gates pass and each layout/capacity passes nine selected normalized
partial-output comparisons bitwise against native materialization plus the
validated warp route. All GPU monitors are clean. These are three-sample
warmup-following H5 timings, excluding selection, packing, SQ1 contribution,
final combination and35 heads. Comparisons between layouts are separate runs,
not interleaved estimates; no complete attention latency or SDPA speedup is
inferred. Neither candidate is promoted.

Two128-thread CTAs still mean eight resident warps, exactly as one256-thread
CTA did. The increase in CTA count therefore does not increase the resident
thread count. Standalone merge kernels use127/128 registers with no stack or
local allocation, while the standalone route reports166 registers and64 API
local bytes. Those modules differ in query loading and loop structure and do
not prove that simply joining producer and consumer causes the extra registers.
The whole-loop experiment directly fails to reduce the fusion count.
Its cuobjdump resource report shows243 registers,64 stack bytes, zero LOCAL
bytes and1,024 static shared bytes in addition to the dynamic allocation;
these allocation figures are not executed memory-traffic measurements.

Evidence: `one_column_review.json`, `loop_calls_review.json`, the corresponding
`ldmatrix_*_scai6.json` receipts, `ldmatrix_merge_standalone_resources.txt`,
`ldmatrix_loop_calls_resources.txt`, and exact sources in
`source/one_column_round/` and `source/loop_calls_round/`.
Do not repeat these layouts unchanged. Inspect producer/consumer register live
ranges and repeated raw-load/barrier work before selecting another decomposition;
standalone register counts and resident CTA count alone proved insufficient.

## Feature-major consumer QK loop: component reduction, fusion rejected

The isolated `native_fisher/ldmatrix_dmajor/` family changes only the route
QK loop. It iterates over16-feature panels before eight-key panels, retaining
four Q operand words instead of32. All32 score accumulators remain in
registers; every individual dot product retains the original feature-panel
MMA order, and bias/softmax/PV arithmetic is unchanged. Matrix and merge
headers and the two-column256-thread fused kernel are unchanged copies of
the validated matrix-load family.

On a freshly exclusive scai6 GPU4, the standalone route uses128 registers
instead of166 (64 API local bytes and33,024 dynamic shared bytes unchanged).
All existing route gates pass, including random ring tails, empty rings,
zero-query ties and real full compressed streams. Both C4 and C10 also pass
nine selected fused normalized partial outputs bitwise against materialized
native merge plus the validated route.

The resource gain does not carry into fusion: both capacities use247
registers,64 API local bytes,111,744 dynamic shared bytes and one resident
256-thread CTA. Three warmed H5 samples average293.445 ms C4 and343.977 ms
C10. These exclude selection, packing, SQ1 contribution, final combination
and35 heads; they are not complete attention or SDPA speedups. Comparisons
with earlier fusion layouts are separate runs. The candidate is not promoted.

Evidence is in `dmajor_route_review.json`, `ldmatrix_dmajor_fused_scai6.json`
and exact measured sources/drivers in `source/dmajor_route_round/`. The next
resource question is the native seed-loader producer, whose pointer layout
and loop differ from the gathered-Q standalone merge. Isolate that path before
assuming that consumer-register reduction or further source-level loop changes
will reduce the combined kernel count.

The fused validator now rejects a CUDA source outside the selected source
directory: quoted header includes resolve beside that source, and otherwise
the recorded header hashes could describe different code. Its negative CPU
test passes before importing Torch. All measured receipts above already used
matching source/header directories; archived drivers preserve the pre-guard
version used for the feature-major run.

## Native-seed producer isolation and shared consumer state

`diagnose_fisher_native_producer.py` runs `producer_diagnostic.cu` with the
same native seed-loader pointers, two-column producer loop, raw K/V staging,
ring overflow and256-thread barriers as the feature-major consumer fusion.
Consumer warps copy every live K/V/bias ring element to observable global
outputs instead of computing attention; zero/positive-infinity initialization
makes trailing padding verifiable. These output writes prevent the merge work
from becoming unobservable. They also change consumer code, so the resource
counts are not an exact register-liveness measurement of the complete fusion.

Both capacities pass bitwise K/V/bias comparisons against retained native
materialization for ten CTAs (twenty column streams), including consecutive
x/y grid positions, first/middle/final key partitions and trailing padding.
All runs use freshly exclusive scai6 GPU4 with clean monitors.

| Producer source | C4 registers | C10 registers | API local bytes |
|---|---:|---:|---:|
| Native seed pointers, original key-major score loop | 165 | 166 | 0 |
| Native seed pointers, feature-major score loop | 158 | 166 | 0 |

The second isolated family, `ldmatrix_all_dmajor`, changes only the producer
score-loop order to retain four query operand words per feature panel while
preserving each score's MMA accumulation order. Dynamic shared storage is
111,744 bytes and occupancy is one256-thread CTA for both families. No producer
latency or complete-attention speedup is claimed. Its copied full fused and
route diagnostic CUDA entry points were not executed in this round.

`warp_fused_shared_state.cu` separately tests whether storing the consumer's
loop-carried state between rings reduces combined register pressure. It adds
52,224 shared bytes using a bank-contiguous [state element,lane] layout for six
consumer warps. Volatile shared stores/loads prevent inter-ring state from
being promoted back into a persistent register array. On each ring, a local
RouteState is loaded or initialized, consumed, and stored with no arithmetic
change. Original producer and feature-major route code remain unchanged.

Both capacities pass all route gates and nine selected normalized partial
outputs bitwise. Registers nevertheless rise to252 with64 API local bytes,
163,968 dynamic shared bytes and one resident256-thread CTA. Three warmed H5
samples average296.779 ms C4 and355.055 ms C10. These exclude selection,
packing, SQ1 contribution, final combination and35 heads; separate-run
comparisons are not paired estimates. This variant is rejected.

`producer_roles_review.json` and `source/producer_role_round/` preserve raw
receipts, source hashes and scope. These results do not support another
unmotivated warp-loop/register-placement sweep. The next implementation should
use native WGMMA producer arithmetic, first checked as a shared-ring producer
against native K/V/bias for complete selected key partitions. For C10, the
archived four-column native layout supplies validated arithmetic; its rejected
materializing full operator must not be rerun unchanged. C4 needs its own
column/ring sizing: a layout that fits C10's consumer states need not fit C4's
larger number of query columns. Preserve all45 queries, all compressed keys,
exact current seed selection and SQ1 semantics throughout this work.
