# C10 FP8 intermediate KV: 150 ms goal

User objective: complete GMA-C10 FP8 intermediate-KV attention in 150 ms,
with at least 20% of account token allowance left. Goal remains active and
unmet. Latest quota check: 79% used, 21% remaining. The search stop threshold
has been reached; finish recording existing results and start no new search
to preserve the user's at-least20% reserve. Do not consume a usage reset.

This isolated branch starts at 509d283. Canonical artifacts:
`C:/GMA/experiments/gma_c10_fp8_150ms_20260910/scai7` and the matching
`/local2/denghaoran/GridMergeAttention/experiments` directory on scai7.
Remote worktree has the same basename as this local worktree. Only use one
clean H200; historical work used scai7 GPU3, UUID ending 5d799b96c077.
Recent trials use user-authorized scai6 GPU5, UUID ending 7fc32ed5ba3c,
with receipts under the sibling `scai6` artifact directory. Recheck exclusive
idleness immediately before each run; reject timing if monitoring finds a
foreign process. Other worktrees and the paper repository are read-only.

## Scope and validation

Preserve Fisher Uniform16, C10, RF0, SQ1, full compressed attention, native
BF16 Q/K/V and BF16 weight rounding. FP8 applies only to compressed KV
storage. User allows ordinary roundoff subject to errors and generation
quality validation. The frozen 20-pair native BF16/FP8 four-step comparison
has completed generation, full-frame PSNR/SSIM/LPIPS and six VBench dimensions
for the older shared-store FP8 source. The newer local-score source still
needs its own generation validation; no candidate is promoted into defaults.

Benchmark uses real QKV layers 00/19/39, shape (1,21,45,80,40,128), complete
H40 calls, 16 interleaved ABBA/BAAB samples per method per layer, with a
same-run SDPA control. Sampled quantization checks, exact seeds/bias and
finite outputs are storage correctness screens, not generation acceptance.

## Completed measurements

All rows below are clean-GPU complete H40, layer00 only, milliseconds.

| Candidate | FP8 | Matched BF16 | Status |
|---|---:|---:|---|
| Direct E4M3, fused selector | 296.888 | 263.130 | slower selector |
| Padded 16-seed-row layout | 495.401 | 261.061 | rejected |
| Per-token scaled E4M3 | 292.759 | 266.646 | scaled route sweep no improvement |
| Vector global stores | 311.241 | 263.604 | rejected, 48 local bytes |
| Single FP32-to-E4M3 rounding | 275.082 | 265.748 | retained as intermediate step |
| Shared halfword FP8 stores | 270.659 | 266.111 | reduces registers to 111 |
| Double-buffered shared stores | 269.650 | 266.654 | no convincing overlap gain |
| FP32 conversion using unit scale | 258.028 | 268.162 | exact vs direct FP8 |
| Explicit FP32 conversion, no scales | 254.878 | 267.664 | exact vs direct FP8 |
| FP32 conversion, one route stage | 251.246 | 268.095 | best layer00 configuration |
| N256 joint K/V WGMMA | 291.801 | 262.696 | rejected, 174 registers / 2 CTAs |
| N256 joint K/V, capped registers | 270.252 | 266.962 | rejected, 166 registers / 32 local bytes |

The direct single-round candidate has relative output L2 0.069643 vs BF16; sample storage
errors satisfy the combined BF16/FP8 rounding envelope. It preserves weight
rounding and removes only the final redundant BF16 storage rounding.
Merge resources: 128 registers, zero local bytes, 53120 shared bytes,
4 CTAs/SM. Layer00 diagnostic H5 merge 19.566 ms, route 12.823 ms,
selector 15.652 ms, packing 0.390 ms. Workbook rows 1242 through 1246
were added incrementally to C:/GMA/tmp.xlsx and verified.

## Previous best and closed experiments

Previous best was shared halfword FP8 stores plus explicit FP32 conversion
before BF16 narrowing, route block64/stages1/warps4. Full H40 layer00/19/39
times are 251.246 / 252.246 / 250.361 ms, mean **251.284 ms**, 48 total
calls. Matched SDPA mean 327.514 ms, **1.30336x**. Output L2 vs BF16 is
6.9643% / 1.4073% / 1.0551%; these are quantization errors, not ordinary
roundoff acceptance. Seeds and bias remain exact. Source hashes are in
the artifact `goal_checkpoint.json`. Original receipt bytes are preserved in
`scai7/best_source_snapshot`; the experimental harness has subsequently evolved.

Explicit experimental entry point: `native_fisher.c10_fp8_candidate.make_candidate`.
The CUDA merge uses 111 registers, zero local bytes, 53120 shared bytes,
4 CTAs. The Triton route uses 159 registers, zero spills, 49152 shared
bytes. The ten-config route sweep found stage1 best for this conversion;
do not repeat the unchanged sweep. The previous implementation has completed the20-pair generation evaluation
below. Target150ms is not achieved.

- FlashInfer 0.6.6 customized fa2 compressed attention uses BF16 Q, E4M3
  KV, FP16 output and FP32 log2 LSE. A separate unchanged BF16 SQ1 state
  is combined before BF16 output. No raw QKV quantization.
- Initial large-output mismatch was caused by 32-bit address arithmetic
  for 10.16 GB H5 KV tensors. Int64 indptr fixes this: complete H5 state
  comparison relative L2 0.001326 vs the original FP8 route. Split Triton
  states alone differ by 0.000612. Artifacts flashinfer_state_debug*.json.
  Full H40 then passed numerical checks on all three layers but measured
  395.478 / 394.269 / 394.372 ms. Rejected for speed; do not repeat fa2.
- Shared halfword stores were retained. Relaxing the original epilogue's
  register cap alone produced unchanged 128-register allocation, not timed.
- The two N256 joint-KV variants passed storage checks but were slower;
  this family is closed. Its GPU and CPU handles have completed.

New results through workbook row1264 are persisted in the existing tmp.xlsx.
Local artifacts and sources are synchronized to the isolated scai7 worktree.

## Further exact-FP8 trials and live work

Each new trial requires complete H40 output equality to the retained FP8
operator before the usual BF16 quantization checks and interleaved timing.
Three-layer entries have 48 calls; the swizzle entry has 16 layer00 calls.

| Candidate | Complete H40 ms | Matched BF16 ms | Assessment |
|---|---:|---:|---|
| Cache winning score/exp as float2 | 255.374 | 267.661 | slower, closed |
| Swizzle normalized-score shared addresses | 256.138 | 267.481 | slower, closed |
| Separate K/V TMA buffers and remove waits | 251.156 | 267.792 | indistinguishable from retained; not promoted |
| One-head intermediate workspace, all 40 heads | 257.867 | 267.384 | slower, closed |

Raw receipts: winner_pairs.json, score_swizzle.json, double_store.json,
head1.json. The double-store variant has not undergone racecheck; output
equality alone is not proof of synchronization safety. Retain 251.284 ms.
Nsight merge counter collection failed with ERR_NVGPUCTRPERM; no hardware
stall or DRAM-utilization measurements were obtained. Do not infer counters
from that unsuccessful profiler run or change system permissions.

## Completed generation and fidelity

Generation in `scai7/fourstep20_pairs` completed 20 native BF16 and 20 native
FP8 videos. The original parent3378648 and worker3381172 are terminal.
`completion.json` reports generation_valid=true, worker_exit_code=0 and no
errors. Both phases have 3200 audited native calls, 160 per video, four per
layer/video, across all40 layers. All40 videos and paired receipts are
archived locally. Prompt identity is the canonical MovieGen extended set,
indices0..19, seed0, 81frames, four denoising steps.

`control_generation.json`: mean core273.784706ms, full325.073109ms,
inference plus export80.738855s/video including first-video setup.
`candidate_generation.json`: core256.653686ms, full289.000823ms,
inference plus export61.234943s/video. These instrumented values do not
replace the251.284ms clean kernel benchmark. No matched SDPA generation
exists; its speedup is unavailable.

`quality_cpu.json` is valid for all20 pairs /1620 full-resolution frames,
without resize or subsampling: PSNR16.737064dB, SSIM0.606591,
Alex-LPIPS0.220973 against native BF16. This is fidelity evidence, not
final generation-quality acceptance. First-two-video contact sheets retain
coherent subjects/scenes with different framing and detail, but only cover
four sampled frames/video and cannot establish full temporal quality.

The six-dimension custom-input VBench comparison completed with clean GPU
monitoring and all20 videos per dimension/variant. Parent3742741 and both
children are terminal. Results: `fourstep20_pairs/vbench6/result.json`.

| VBench dimension | BF16 | FP8 | FP8 minus BF16 (percentage points) |
|---|---:|---:|---:|
| Subject consistency |0.971035|0.971382|+0.034708|
| Background consistency |0.969496|0.970578|+0.108200|
| Motion smoothness |0.987106|0.986849|-0.025740|
| Dynamic degree |0.350000|0.350000|0.000000|
| Aesthetic quality |0.680753|0.684440|+0.368728|
| Imaging quality |0.723786|0.730004|+0.621747|

This is six custom-input dimensions, not the official16-dimension total.
The20-pair means show no substantial aggregate regression on these checks;
this is not a statistical guarantee beyond this sample. All20 thumbnail
pairs at frames0/27/54/80 were reviewed. Subjects/scenes persist, without
blank output or pervasive corruption, but framing, identities/details and
motion timing differ. Full-playback temporal review was not performed.
`full20_thumbnail_review.md/json` records the scope and observations.
The generation/quality receipts belong to the previous251.284ms FP8
implementation. New kernel changes must preserve its results or undergo
further generation validation. Workbook rows1265/1266 now contain the six
individual values in notes; official-total columns remain empty.

## Closed subsequent kernel experiments

All complete-call results include selection, packing, H40 merges, full SQ1
route and intermediate IO. Each uses16 interleaved samples per method/layer
on an exclusively clean H200; output equality is against the retained FP8.

| Candidate | Layers | Full H40 mean ms | Matched BF16 ms | Decision |
|---|---|---:|---:|---|
| Two-slot pipeline with explicit FP32 conversion |00/19/39|252.374|267.843|no improvement|
| Packed shared FP8 word stores |00/19/39|265.788|267.395|slower than retained FP8|
| Aligned producer/consumer merge pipeline |00/19/39|276.198|264.825|slower|
| Integer E4M3-to-BF16 decode |00|360.804|260.886|slower, close this implementation|

Receipts: pipeline_via_float.json, packed_shared.json,
role_pipeline_aligned.json and integer_route.json. The first three each
have48 calls; integer decode has16. Seeds/bias and complete FP8 outputs
are exact; BF16 comparison includes the unchanged FP8 quantization error.

Role pipeline v0 produced incorrect KV while bias stayed exact. The
zero-base128-byte swizzle descriptors require1024-byte period alignment;
v0 shared arrays were only128-byte aligned. The corrected arena and
weight/output bases are1024-byte aligned, with95360 shared bytes,128regs,
zero local bytes, two CTAs. Corrected SHA256:
0b0b620b59e0687761e01ceeca871e82988543ce94b2a8c8ddbdbd2813d0601a.
Every H5 KV byte (20,321,280,000) and all79,380,000 bias floats pass the
separate gate; all three complete H40 outputs also match. No sanitizer run
was performed because the candidate was rejected for speed. Exact v0 bytes
and compilation receipt are archived in role_pipeline_v0_source; do not
repeat the known misaligned source.

Integer decode uses byte-permutation tables for subnormals and direct bit
mapping for normals. All256 E4M3 encodings were checked, including signed
zeros and NaN masks. Real layer00 output is bitwise equal to retained FP8.
Route resources167regs/0spills/49152shared versus retained159regs; the
360.804ms complete timing rejects the implementation. This experiment does
not establish which individual instruction caused the slowdown.

Workbook rows1265/1266 contain audited generation results, with full20
fidelity now filled in1266. Rows1267..1269 contain the three closed trials;
the integer-decode row is appended separately. Preserve all prior rows.

The150ms target remains unmet. Retain the251.284ms source snapshot and
production defaults. Current goal turn made progress; no blocker audit is
active. Account75% used /25% remaining, stop new searches by79% used.

## Current best: local normalized-score reuse

`analysis/build_c10_fp8_local_scores.py` generates a merge that stores only
owner indices in shared memory, then reuses each seed's normalized register
scores for probability/entropy. The contributing winner has exactly the
same score as the former shared reload. Original owner/tie selection,
BF16 weights, residual scaling, FP8 output and full SQ1 route are preserved.

Source SHA256:6dabac09d497f89bda35e41a43a2893fd05c99862da506f74c19478189ed1730.
Resources117registers,0local bytes,52160shared bytes,4CTAs. The complete H5
storage gate checks20,321,280,000 KV bytes and79,380,000 bias values exactly.
Three complete H40 outputs and seeds are bitwise equal to retained FP8.

First complete three-layer measurement:247.563060ms, matched BF16267.612056ms,
SDPA327.657998ms (1.32353x). Receipt local_scores.json; workbook1271.
Because the gain was small, a separate matched FP8 comparison was required:

| Layer | New ms | Retained FP8 ms |
|---|---:|---:|
|00|245.252286|249.166722|
|19|246.922106|251.278395|
|39|245.086317|249.160983|
|Mean|245.753570|249.868700|

The matched gain is4.115130ms (1.65%). Each method has16 complete calls per
layer /48 total in interleaved ABBA/BAAB blocks. Individual block differences
have noise; all three layer means favor the new source. Matched SDPA is
327.203265ms,1.33143x. Receipt local_scores_paired.json; workbook1272.
Both measurements used cleanGPU3, with all intermediate IO included.

`native_fisher.c10_fp8_candidate.make_candidate` now selects this explicit
experimental implementation. Production dispatch remains unchanged. Raw
new-source bytes are preserved separately in local_scores_source_snapshot;
the original best_source_snapshot and generation source remain intact.
The new merge has not been used for a new20-video generation run; its
complete outputs match the evaluated priorFP8 on the three real captures.
The150ms goal is still unmet. No own CPU/GPU workers remain after completion
of these tests and VBench. Quota75%used/25%remaining; reserve20%.

## Closed owner64 layout experiment

`analysis/build_c10_fp8_owner64.py` changes the two ownership tiles from
40+40 to64+16 keys, uses a bank-distributed64x64 score layout, and places
six64-entry uint16 owner rows in the four unused matrix rows60..63. Each
pair of owners is read as one32-bit word. Owner selection reads only the
60 live rows; barriers protect both transfers of scratch ownership. The
first64/last16 probability and entropy reduction order remains unchanged.

Source8d273c816e70ff07e42ad10862a626b7d7b616589ef6857bae833b8582a81f2c.
CPU compile and complete H5 gate passed:20,321,280,000 KV bytes and
79,380,000 bias values exactly match the prior FP8. Resources127regs,
0local bytes,57344shared bytes,4CTAs. Complete H40 outputs and seeds also
match current local-score FP8 on all three real captures.

Matched full-H40 means00/19/39:254.384778/255.036933/253.760844ms,
versus current local-score245.567060/246.883084/245.331291ms. Each method
has48 total interleaved calls. The candidate is slower on every layer and
is rejected. No hardware bank-conflict counters or sanitizer run were
collected; the result does not establish a particular cause for the loss.
Do not repeat this layout or the already-closed40-key score swizzle.
`compare_c10_fp8_retained.py` now accepts an explicit reference source to
keep future timing controls tied to the current retained implementation.
Current best remains245.753570ms; all workers from this trial are terminal.

## Closed native FP8 route experiments

Two explicit CUDA routes consume the unchanged local-score FP8 merge.
Both retain9450 compressed tokens and all945 SQ1 tokens, BF16 original
Q/K/V and BF16 matrix arithmetic. One8KiB TMA staging tile alternates K
and V. Conversion expands only SRAM data; no global BF16 KV array is
introduced. The256-encoding E4M3 decoder gate checks finite raw BF16 bits,
signed zeros and NaN masks against the existing conversion.

The SS version retains a16KiB query tile. The RS follow-up loads the
existing packed BF16 Q through explicit volatile L1 reads for each QK,
then releases query fragments before PV. It requires no additional pack
in a complete call. CPU-only constrained RS builds allocated local memory;
the final tested three-CTA build has168registers and0local bytes.

| Route | H40 ms | Matched retained FP8 ms | Same-run SDPA speedup |
|---|---:|---:|---:|
|Native SS / FP8 staging|307.376060|242.263803|1.063634x|
|Native RS / packed Q|312.956230|241.994057|1.044097x|

Each receipt contains16 calls per method per real layer00/19/39,
48 calls per method total in ABBA/BAAB blocks. Both candidates are slower
on all three layers. GPU3 was exclusively clean throughout both runs.
Their worst per-layer output relative L2 against retained FP8 is
0.00005223818; maximum absolute difference0.03125. These are kernel
rounding screens, not new generation-quality validation. Seeds and bias
are unchanged. Neither a sanitizer run nor hardware counter profiling
was performed; do not infer the exact bottleneck from these timings.

SS route resources:128registers,32local bytes,57616dynamic shared bytes,
3active CTAs. RS:168registers,0local bytes,41232dynamic shared bytes,
3active CTAs. These are route resources; merge remains117/0/52160/4.
The complete times include selector, original Q packing, all eight H5
merge/route chunks, intermediate IO and output allocation.

Sources: analysis/build_c10_fp8_native_route.py and the RS builder;
native_fisher/fisher_c10_fp8_native_route{,_rs}.cu. The copied C4 BF16
reference is provenance only. The original worktree was read without
modification. Reproduce with compare_c10_fp8_retained.py using the current
local-score merge for both --source and --reference-source, and the
chosen --native-route-source. Production defaults are unchanged.

Receipts:native_route_paired.json and native_route_rs_paired.json.
Exact receipt-addressed source snapshots are archived separately; older
driver versions were restored only after matching their recorded SHA256.
Workbook1274/1275 contain the two rejected results, with rounding errors
and route resources distinguished from merge resources. Owner64 remains
rejected in1273. Current best remains245.753570ms, and150ms is unmet.

## Empty-cluster IO opportunity: small

`analysis/probe_c10_fp8_empty_clusters.py` measures the retained merge's
positive-infinity bias slots on every head of real step0 layers00/19/39.
All eight H5 chunks are included per layer. Only finite and positive-infinity
bias values occur; none of the aligned64-token route tiles is wholly empty.

| Layer | Empty compressed tokens | Ideal avoided reads at3.36TB/s |
|---|---:|---:|
|00|6.813442%|3.296616ms|
|19|3.767403%|1.822820ms|
|39|5.268903%|2.549306ms|

These times divide avoidable useful bytes by assumed bandwidth; they are
not measured speedups. Skipping both writes and reads would double the
byte-count benefit before mask/compaction overhead, still a small fraction
of the gap to150ms. No empty-cluster skipping candidate was implemented.
Receipt:empty_clusters.json, with exact source hashes and a separate snapshot.

## Five-CTA resident-KV merge: rejected

`build_c10_fp8_n64_owner16.py` replaces128-feature outputs with two64-feature
MMAs and processes ownership in five16-key strips. Both full original BF16
K/V tiles remain resident. This differs from the older rejected N32 streamed-V
kernel, which repeatedly reloaded V. The five strips preserve the original
first64/last16 mass/entropy addition order. A64-byte-swizzled4KiB FP8 staging
tile is shared with ownership scratch. Global layout remains9450 C10 tokens.

Resources fall from117registers/52160shared bytes/four CTAs to96registers,
16local bytes,45440dynamic shared bytes and five CTAs. Higher residency
does not improve the complete operator:

| Variant | H40 ms | Matched retained FP8 ms | Same-run SDPA speedup |
|---|---:|---:|---:|
|N64 / owner16|265.392931|245.768087|1.233557x|
|N64 / owner16, overlap strip work|265.134171|245.584634|1.236371x|

The second builder removes only the five end-of-strip barriers. The next
normalized-score writes and previous owner-index reads use disjoint scratch;
the next score-write barrier precedes all owner overwrites. The existing
post-weight barrier protects final owner reads before FP8 staging reuse.
All other synchronization remains. It is still slower on every layer.

Both complete H5 gates compare20,321,280,000 KV bytes and79,380,000 bias
values exactly against prior FP8. Both three-layer H40 comparisons have
byte-exact outputs, bias and seeds against current local-score FP8. Each
method has16 samples per layer/48 total in ABBA/BAAB blocks. GPU3 checks
and monitoring are clean. No sanitizer, hardware counters or new generation
was run; the measurements do not isolate the source of the slowdown.

Receipts:n64_owner16{,_overlap}_paired.json and their separate source
snapshots. Workbook1276/1277 contain these rejected candidates. Keep the
245.753570ms implementation; neither candidate enters production defaults.
The150ms target remains unmet. Do not repeat this N64/owner16 family unchanged.

## Q access and invariant seed reads: no gain

Three more candidates retain current local-score arithmetic, BF16 weights,
FP8 output format, Fisher seeds, SQ1 and the full compressed route:

- `build_c10_fp8_head_column_q.py` changes only the physical packed-Q order
  to[head,column,frame,row,D], keeping the exact register-fragment permutation.
  The existing pack kernel performs both transformations. Original QKV and
  selector inputs keep their native layout. The gate compares all774,144,000
  packed bytes with a lossless reference permutation before comparing KV.
- `build_c10_fp8_seed_hoist.py` makes the column/frame loops explicit and
  loads frame-invariant seed rows once per column group. Every tile retains
  its original order, including partial column groups for small tile counts.
- `build_c10_fp8_query_prefetch.py` issues L1 requests for selected Q of the
  next frame after current QK completes. One lane per four-lane row group
  requests both128-byte lines of each row. Frame20 is excluded, all requested
  lines stay within native Q, and no new shared buffer is allocated.

| Candidate | H40 ms | Matched retained FP8 ms | Same-run SDPA speedup |
|---|---:|---:|---:|
|Head/column packed Q|246.029483|245.854511|1.331352x|
|Hoisted seed rows|246.963873|245.927179|1.326754x|
|Next-frame Q prefetch|248.165058|245.802914|1.320086x|

The physical layout has two slower layers and one slightly faster layer;
its0.07% aggregate difference is noise-sized, so there is no confirmed gain.
Hoisting and prefetching are slower on all three layers. All candidates have
48 interleaved complete calls per method,16 each on real00/19/39, on clean
GPU3. Every H5 KV/bias byte and all three H40 outputs, seeds and bias are
exactly equal to the retained FP8. No extra rounding or quantization is added.

Registers/local bytes/shared bytes/active CTAs are117/0/52160/4 for physical
packing,113/0/52160/4 for hoisting, and115/0/52160/4 for prefetching. These
timings include the original or replacement packing kernel. No new generation,
sanitizer or hardware-counter run was performed. The negative results do not
prove that query-memory latency is absent; only these implementations failed
to improve complete-call latency.

Receipts:head_column_q_paired.json,seed_hoist_paired.json,
query_prefetch_paired.json; exact source snapshots are separate. Workbook
rows1278..1280 contain the three measurements. Retain245.753570ms and leave
production defaults unchanged. The150ms goal is unmet; these routes are closed.

## Paired FP8 K/V byte storage: rejected

`build_c10_fp8_paired_kv.py` stores K in the low byte and V in the high byte
of an int16 container. It retains sequential BF16 K/V matrix operations,
FP32 residual correction and the original E4M3 rounding. The two-byte word
is storage only, with no FP16 arithmetic. One 16KiB shared output tile feeds
a single TMA store, with vector stores for the last two columns. The route
loads each pair once and converts both bytes to BF16 before the original
attention operations. Total compressed bytes are unchanged.

On user-authorized scai6 GPU5, complete H40 latency is271.703345ms versus
matched retained local-score FP8 246.549045ms. All three real step0 layers
regress:271.628719/273.069424/270.411892 versus
246.389466/247.967803/245.289865ms. There are16 interleaved ABBA/BAAB samples
per method/layer,48 total. Same-run SDPA is330.655190ms (1.216971x for the
candidate). Do not subtract these scai6 means from historical scai7 timings.

Complete H5 merge validation checks20,321,280,000 K/V bytes and79,380,000
bias values exactly. The route decoder passes all65,536 byte combinations
for finite raw BF16 bits and NaN masks. All three complete H40 outputs,
seeds and bias match retained FP8 exactly. Initial idle checks and continuous
foreign-process monitoring are clean. No generation, sanitizer or hardware
counter validation was performed for this candidate.

Merge resources are128 registers,56 local bytes,57,344 dynamic shared bytes
and4 active CTAs, versus retained117/0/52,160/4. Route registers rise from159
to166; both routes report0 spills and49,152 shared bytes. These measurements
show higher resource pressure but do not isolate its share of the slowdown.
Both workers completed and GPU5 returned to0 MiB/0% with no compute process.

Receipts are `scai6/paired_kv_gate.json` and `scai6/paired_kv_paired.json`
under `gma_c10_fp8_150ms_20260910`, with an immutable source snapshot. CUDA
source SHA256 is71da4781c4f60770d5053cf4b4d55819360a40f871fa69f85ee29b4293269ea4.
The experimental workbook records the complete-call aggregate. Retain the
local-score implementation and its historical245.753570ms result;150ms is
unmet. This paired-byte implementation is closed without changing defaults.

## Exact probability/owner tuples with private scores: rejected

The local-score receipt's isolated H5 diagnostics measure19.2446ms for merge
and9.9425ms for route, plus15.6788ms full selector and0.3908ms packing.
These stage probes motivate work on merge; they are not a new full-call
measurement or hardware-counter evidence.

`build_c10_fp8_owner_probability.py` derives from retained local-score reuse.
The owner-selection threads evaluate each winning exponential once and store
an exact uint2 probability/owner tuple. Consumers retain their private log
scores for entropy and read that tuple instead of evaluating another exp2.
Unlike the earlier twelve-byte score/probability/owner trial, the new state
uses eight bytes and does not reload the log score. No float bits, tie rules,
weight rounding, accumulation order or FP8 storage boundary change.

Complete H5 validation checks all20,321,280,000 KV bytes and79,380,000 bias
values exactly. All three real-layer H40 outputs, seeds and bias are also
bitwise equal to retained FP8. Nevertheless,48 interleaved full calls per
method give251.295141ms versus247.011018ms matched control on clean scai6
GPU5. Layers00/19/39 are251.137562/252.909749/249.838112 versus
246.979652/248.358427/245.694976ms: all slower. Same-run SDPA speedup is
1.313648x. No generation, sanitizer or hardware counters were run.

Merge resources become110 registers,zero local bytes,53,120 dynamic shared
bytes and4 CTAs, compared with117/0/52,160/4. Lower registers and fewer
exponential evaluations do not yield a complete-call gain in this layout;
the result alone does not quantify which added memory instructions caused
the slowdown. Route code and configuration remain unchanged.

Receipts:`scai6/owner_probability_gate.json` and
`scai6/owner_probability_paired.json`, with exact sources frozen separately
and the aggregate in the experimental workbook. Source SHA256:
8e02eebead14b372d6f79e6b24ef98e6e21c7c09676126adf73a5e38645c618c.
Both workers completed; GPU5 returned to0 MiB/0% with no compute processes.
This implementation is closed. Retain245.753570ms;150ms remains unmet.

## FP8 route with eight warps: rejected

The earlier ten-configuration FP8 route sweep used four warps throughout.
The new `--candidate-route-warps 8` option changes only that launch parameter,
keeping block64,one stage,unchanged local-score merge and BF16 arithmetic.
Its numerical gate permits ordinary roundoff within the pre-existing screen,
but all three measured complete H40 outputs are in fact byte-exact; seeds
and bias match exactly as well. No new H5 merge test is needed for the
identical merge source, and none is claimed here.

On clean scai6 GPU5,48 interleaved full calls give285.422230ms versus matched
warps4 246.455698ms. Layers00/19/39 measure285.229065/286.959757/284.077868
versus246.355065/248.020785/244.991244ms. Same-run SDPA is330.419653ms.
Route registers fall from159 to122 per thread, but threads per CTA double;
both compiled routes havezero spills and49,152 shared bytes. The complete
timing regresses on every layer despite lower per-thread register pressure.
There are no new generation, sanitizer or hardware-counter claims.

Receipt:`scai6/route_warps8_paired.json`, with an immutable source snapshot
and an aggregate in the experimental workbook. The worker completed and
GPU5 returned to0 MiB/0% with no compute process. Retain the four-warp route
and historical245.753570ms best. At79% account usage, do not start another
search;150ms is unmet and the goal is not completed.
