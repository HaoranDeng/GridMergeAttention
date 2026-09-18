# Negative Results and Guardrails

This file records experiments that should not be repeated unless there is a
new reason to revisit them. The goal is to keep hard-route / GridMerge C4 work
focused on variants that preserve quality against dense attention.

When an experiment fails, add it here with: result, change tested, evidence,
why not repeat, and the rule it implies. Treat this file as the first place to
check before launching another speed/quality sweep.

## Hard Constraints

- Keep grid-local compression. Do not replace it with global compression or
  share-all compression.
- After token merging, the second stage must run full attention over all
  compressed KV tokens. Do not replace it with sparse routing, sliding-window
  routing, or local-only routing when evaluating the accuracy-preserving path.

## Failed or Rejected Directions

### Contiguous Four-Row Seed Window for Hard C4

- Result: simpler seed addressing did not improve speed and slightly worsened
  dense-output error in the 12-call smoke. This note records that measurement;
  the code may still be switched to this form temporarily for inspection or
  further manual kernel work.
- Change tested: replace the maintained local-with-ith seed order with a
  contiguous clamped window. For key row `k`, use
  `start = clamp(k - 1, 0, rows - 4)` and seed rows
  `[start, start + 1, start + 2, start + 3]`. The intent was to make the four
  seed-query rows contiguous in memory and reduce fragmented address logic.
- Evidence: scai7/H200 12-call real Wan QKV smoke
  `hard_c4_contiguous_seed_12call_20260814` measured hard C4 at `16.05`
  ms/call, stage1 `11.25` ms/call, rel-L2 `0.11291`, and MSE `0.001950`.
  The immediately comparable old local-with-ith 12-call smoke measured about
  `15.68` ms/call, stage1 `11.26` ms/call, rel-L2 `0.11228`, and MSE
  `0.001929`.
- Why not repeat directly: the four seed rows are already a small part of the
  first-stage cost, and making them contiguous did not reduce the dominant
  merged K/V/bias intermediate IO. The seed set also changes from the previous
  local-with-ith choice, so quality can move.
- Rule: use this as a readability/debugging variant. Treat it as a speed or
  quality improvement only after a new full benchmark shows otherwise.

### Temporal-Key Grouped qf1/kf3/C3 Triton Kernel

- Result: quality improved over C1 at the same compressed-token density, but
  the tuned Triton backend was still slower and less accurate than hard C3, so
  it was deleted after measurement.
- Change tested: keep the original one-frame, one-column query block
  (`30` query tokens), but group the same spatial key row across three adjacent
  frames (`3 x 52 = 156` keys) and compress it into C=3 merged K/V tokens.
  This keeps compressed-token density at `3/(3*52)=1/52`, the same as C1.
- Evidence: PyTorch screening in `temporal_grouped_promising_full3_20260813`
  recorded all 360 real Wan self-attention calls over three prompts.
  `qf1_kf3_c3` measured rel-L2 `0.10417` and MSE `0.003159`, much better than
  C1/MonarchRT's rel-L2 around `0.174`. A Triton prototype was then tuned from
  `BLOCK_J=16, warps=4` to `BLOCK_J=32, warps=8`; the H200 tile sweep improved
  random-QKV full time from about `23.68` ms to `15.00` ms. The final real Wan
  QKV benchmark `hard_kf3_c3_tuned_realqkv_full3_20260813` measured
  `15.093` ms/call, rel-L2 `0.10410`, and MSE `0.003155`. In the same run,
  hard C3 measured `11.647` ms/call, rel-L2 `0.08771`, and MSE `0.002239`.
- Why not repeat directly: grouping key frames gives a useful density/quality
  tradeoff, but the merge kernel must softmax/route over 156 keys in one block.
  Even after tile tuning, the larger key tile makes the backend slower than
  hard C3 while also less accurate.
- Rule: keep the temporal grouping screening script for future idea checks, but
  do not keep or rerun the `qf1/kf3/C3` Triton backend unless the merge is
  redesigned to avoid the large 156-key tile cost.

### Temporal-Key kf7 Straight Two-Stage Kernelization

- Result: strong quality/density signal, but a straightforward two-stage
  Triton backend is not justified without a new merge design.
- Change tested: keep the original one-frame, one-column query block
  (`30` query tokens), but group the same spatial key row across seven adjacent
  frames (`7 x 52 = 364` keys). The full PyTorch screen tested C4/C5/C7/C10
  compression while preserving the full compressed second-stage attention.
- Evidence: scai7/H200 experiment `temporal_kf7_full3_20260813` recorded all
  360 real Wan self-attention calls over three prompts. `qf1_kf7_c7` kept C1
  compressed-token density `0.01923` and reached rel-L2 `0.07408` with MSE
  `0.001597`, essentially hard-C4 quality. `qf1_kf7_c5` had density
  `0.01374`, rel-L2 `0.08704`, and MSE `0.002205`, close to hard-C3 quality.
  However, the cost model estimates `qf1_kf7_c7` at `1.75x` C4 merge MAC and
  only `0.25x` C4 route MAC; the larger 364-key softmax/merge block would
  dominate a direct two-stage implementation.
- Why not repeat directly: the quality gain comes from a much larger key block,
  but the current kernel architecture already struggles with 104-key and
  156-key merge blocks. Simply increasing the merge tile to cover 364 keys is
  likely to spend the saved second-stage work in first-stage merge IO and
  softmax cost.
- Rule: keep `qf1/kf7/C5-C7` as a quality signal. Do not write a naive
  `hard_kf7_C` kernel that just enlarges the key tile and writes merged
  K/V/bias intermediates. Revisit only with a merge design that reduces the
  364-key compression cost.

### Temporal-Query Grouped qf3/qf7 with kf7

- Result: not a promising speed-quality route. The static cost model looks
  attractive because grouping query frames amortizes merge work, but the
  dense-output error worsens before it reaches the needed quality range.
- Change tested: group three or seven query frames into one query block while
  keeping the seven-frame key block (`364` keys) and full compressed
  second-stage attention. Tested `qf3_kf7_c5/c7/c10` and
  `qf7_kf7_c5/c7/c10`.
- Evidence: scai7/H200 screen `temporal_qf_kf7_screen_20260813` measured the
  first eight real Wan self-attention calls. At C1 density `0.01923`,
  `qf3_kf7_c7` measured rel-L2 `0.11772` and MSE `0.001787`, much worse than
  `qf1_kf7_c7` from the full screen at rel-L2 `0.07408`. `qf7_kf7_c7` was
  worse again, rel-L2 `0.15221` and MSE `0.002987`. Even increasing to
  `qf7_kf7_c10` left rel-L2 at `0.13290`.
- Why not repeat directly: sharing one compressed K/V set across multiple query
  frames removes the query-frame-specific alignment that made temporal-key
  grouping useful. The cheaper merge/route cost is not worth the quality loss.
- Rule: do not tune kernels for `qf3/kf7` or `qf7/kf7` unless a new seed or
  routing design first recovers quality in a real-QKV screen.

### Transpose Query-Row / Key-Column Layout

- Result: not a useful speed-quality candidate. Transpose C1 has worse error
  than non-transpose C1/MonarchRT despite higher compressed-token density, and
  transpose C4 is much denser than hard C2 while only reaching hard-C2-level
  error.
- Change tested: swap the original row/column roles so each block uses a
  52-query-column by 30-key-row tile. The one-token transpose variant compresses
  each key-column group to C1; the transpose C4 variant uses four seed query
  columns with full compressed second-stage attention.
- Evidence: scai7/H200 experiment `transpose_monarchrt_full3_h200_20260813`
  recorded 270 real Wan self-attention calls over three prompts. Transpose C1
  measured rel-L2 `0.17649`, MSE `0.009139`, density `0.03333`, and
  `25.492` ms/call. Transpose C4 measured rel-L2 `0.11976`, MSE `0.004208`,
  density `0.13333`, and `248.181` ms/call in the PyTorch analysis path. In
  existing H200 full3 runs, hard C2 has similar or better error at lower
  density: rel-L2 `0.11824`, MSE `0.004069`, density `0.03846`, and about
  `7.47` ms/call with the tuned kernel.
- Why not repeat directly: making query rows the inner query block weakens the
  useful row-local key/query alignment. The minimum transpose C1 density is
  also `1/30`, while the non-transpose one-token density is `1/52`, so it
  cannot match the lower-density MonarchRT setting without changing the
  algorithm.
- Rule: do not write a tuned transpose hard-route kernel unless a new seed or
  merge objective first shows substantially better dense error than hard C2 at
  comparable compressed-token density.

### Broad Mixed-Axis T/H/W Layouts

- Result: temporal-axis mixed blocks did not produce a better frontier point.
  The only temporal-query variant that matched the original row/column quality
  used much higher compressed-token density.
- Change tested: choose different inner axes from `T,H,W` for query and key
  blocks while preserving grid-local compression and full compressed
  second-stage attention. The screen included `qT-kW`, `qT-kH`, `qH-kT`, and
  `qW-kT` C1/C2/C3/C4 variants, plus the original `qH-kW` family as reference.
- Evidence: scai7/H200 screen `mixed_axis_broad_screen_20260813` measured the
  first eight real Wan self-attention calls. The best temporal-query row was
  `qt_kh_c4` with rel-L2 `0.08643`, MSE `0.000963`, and density `0.13333`.
  The original-axis `qh_kw_c4` was essentially the same quality, rel-L2
  `0.08769` and MSE `0.000992`, at density `0.07692`. Lower-density temporal
  variants were worse: for example `qt_kw_c3` had rel-L2 `0.11945` at density
  `0.05769`, while original-axis `qh_kw_c3` had rel-L2 `0.10021` at the same
  density.
- Why not repeat directly: using time as the query-inner axis does not preserve
  the spatial row/column alignment that makes the maintained layout useful. It
  can recover quality only by spending more compressed tokens, which moves away
  from the 3x speed target.
- Rule: do not write tuned kernels for temporal-query mixed-axis layouts unless
  a later screen beats original `qH-kW` at the same compressed-token density.

### Query-Column Grouped qc2/C4

- Result: dominated by hard C3. It was slower and had worse dense-output
  error, so the prototype Triton backend was deleted after measurement.
- Change tested: group two adjacent query columns together and compress each
  52-key row into C=4 tokens shared by the two columns. This keeps grid-local
  compression and preserves the second-stage full compressed attention.
- Evidence: scai7/H200 experiment `hard_qc2_c4_candidate_full3_20260813`
  recorded all 360 self-attention calls over three prompts. `hard_qc2_c4`
  measured `12.317` ms/call, rel-L2 `0.09603`, and MSE `0.002684` versus
  dense. In the same run, `hard_c3` measured `11.742` ms/call, rel-L2
  `0.08771`, and MSE `0.002239`; FlashAttention measured `14.714` ms/call.
- Why not repeat directly: sharing one C4 compression across two query columns
  cuts some merge work, but the quality loss is still enough that hard C3 gives
  both better accuracy and better speed.
- Rule: do not keep or rerun `qc2/kr1/C4` as a speed-quality candidate unless
  the query-column grouping has a new seed or routing design.

### Query-Column Grouped qc4/C4

- Result: dominated by hard C2. It was slower and had worse dense-output
  error, so the prototype Triton backend was deleted after measurement.
- Change tested: group four adjacent query columns together and compress each
  52-key row into C=4 tokens shared by the four columns. This preserves the
  second-stage full compressed attention, but reduces query-column-specific
  compression compared with current hard C4.
- Evidence: scai7/H200 experiment `qc4_c4_realqkv_full3_20260813` recorded all
  360 self-attention calls over three prompts. `hard_qc4_c4` measured
  `8.904` ms/call, rel-L2 `0.12512`, and MSE `0.004557` versus dense. In the
  same run, `hard_c2` measured `7.534` ms/call, rel-L2 `0.11824`, and MSE
  `0.004069`; FlashAttention measured `14.833` ms/call.
- Why not repeat directly: sharing one C4 compression across four query
  columns does reduce some merge work, but it loses too much query-column-local
  alignment and still keeps enough routing work that it does not beat hard C2.
- Rule: do not keep or rerun `qc4/kr1/C4` as a speed candidate. If revisiting
  query-column grouping, require either a different seed design or a clearly
  lower route cost before writing a new Triton backend.

### Wide Query-Column Grouping qc4/qc13

- Result: not a viable quality path despite attractive static IO/MAC counts.
- Change tested: group four or thirteen adjacent query columns into one query
  block, then combine with wider key-row groups and larger C. Tested
  `qc4_kr2_c8`, `qc4_kr2_c12`, `qc4_kr3_c12`, `qc4_kr5_c16`,
  `qc13_kr2_c8`, and `qc13_kr5_c20`.
- Evidence: scai7/H200 screen `rowcol_wide_query_screen_20260813` measured the
  first eight real Wan self-attention calls. At the same compressed-token
  density as hard C4 (`0.07692`), `qc4_kr2_c8` had rel-L2 `0.25104` and
  `qc13_kr2_c8` had rel-L2 `0.55868`; hard C4 in the same screen was rel-L2
  `0.08765`. Increasing C helped only partially: `qc4_kr2_c12` still had
  rel-L2 `0.14143` at density `0.11538`.
- Why not repeat directly: the lower static cost comes from sharing one
  compressed K/V set across many query columns. That removes the
  query-column-specific alignment that the current hard-route method relies on.
- Rule: do not tune kernels for `qc4_*` or `qc13_*` wide-query grouping unless
  a new merge objective first recovers quality in PyTorch screening.

### Moderate Query-Column Grouping qc2 With Wider Key Rows

- Result: not a better speed-quality frontier point. Some variants reduce
  compressed-token density, but the useful quality points are still
  merge-heavy and not attractive for direct two-stage Triton tuning.
- Change tested: group two adjacent query columns and combine them with
  two-, three-, or five-row key groups. Tested `qc2_kr2_c5/c6/c7`,
  `qc2_kr3_c8/c9`, and `qc2_kr5_c10/c12`.
- Evidence: scai7/H200 screen `rowcol_qc2_more_screen_20260813` measured the
  first eight real Wan self-attention calls. At hard-C3 density `0.05769`,
  `qc2_kr2_c6` had rel-L2 `0.16483` and `qc2_kr3_c9` had rel-L2 `0.11884`,
  both too weak. The best low-density row was `qc2_kr5_c12` with density
  `0.04615`, rel-L2 `0.09830`, and MSE `0.001246`, but the static model
  estimates `1.5x` hard-C4 merge MAC and `0.60x` route MAC, so the first-stage
  merge is likely to dominate.
- Why not repeat directly: grouping two query columns weakens
  query-column-specific compression, while wider key rows increase the merge
  working set. The combination does not offer a clear path to 3x versus
  FlashAttention at C3/C4-like error.
- Rule: do not write tuned kernels for `qc2_kr2_c5/c6/c7`,
  `qc2_kr3_c8/c9`, or `qc2_kr5_c10/c12` unless a new objective changes the
  quality result before kernel work starts.

### Direct qc1/kr5/C8 Two-Stage Triton

- Result: the layout remains a strong quality signal, but a direct two-stage
  Triton kernel is not a speed path.
- Change tested: keep one query column (`30` query tokens), group five adjacent
  key rows (`260` key/value tokens), route them to `C=8` compressed tokens, and
  preserve the second-stage full compressed attention.
- Evidence: the full3 PyTorch analysis run `rowcol_kr5_c8_c9_full3_20260813`
  measured `qc1_kr5_c8` at density `0.03077`, rel-L2 `0.07041`, and MSE
  `0.001443`, better than hard C4 error at lower compressed-token density.
  The direct H200 Triton smoke `hard_kr5_c8_smoke_20260813` then compiled only
  after replacing the padded `512`-key merge tile with a `256 + 16` split-tail
  merge. On one real Wan QKV call it measured `178.30` ms/call. Reusing the
  seed-key logits between assignment and merge, then changing the merge kernel
  from `num_warps=4` to `num_warps=8`, improved the first-three-call H200 run
  `hard_kr5_c8_reuse_warps8_first3_20260813b` to `84.38` ms/call. The same
  run measured FlashAttention `14.80` ms/call and hard C4 `17.26` ms/call.
- Why not repeat directly: the first-stage merge dominates. It performs eight
  seed-key passes over a `260 x D` key block, keeps large `[query_column,
  key]` score/route state, and pays extra work to avoid the invalid `512`-key
  padding. Removing the duplicate seed-key pass gave a real speedup, but the
  optimized direct kernel is still about `5.7x` slower than FlashAttention.
  This validates the cost-model warning that wider key-row blocks need a new
  merge design, not a copied `kr2/kr3` two-stage kernel.
- Rule: do not tune direct two-stage `qc1/kr5/C8` by only adjusting tile sizes.
  Revisit `kr5` only with a merge redesign that reduces seed-key recomputation
  and intermediate score/route state.

### Interleaved Key-Row Grouping

- Result: worse than contiguous key-row grouping at the same compressed-token
  density and same C.
- Change tested: instead of grouping adjacent key rows, group rows by stride.
  For example, `kr2_ginterleave` uses `(0,15),(1,16),...` rather than
  `(0,1),(2,3),...`; `kr3_ginterleave` similarly groups rows separated by ten.
- Evidence: scai7/H200 screen `rowcol_interleaved_keyrow_screen_20260813`
  measured the first eight real Wan self-attention calls. `qc1_kr2_c4`
  worsened from rel-L2 `0.10032` contiguous to `0.12114` interleaved.
  `qc1_kr2_c6` worsened from `0.08339` to `0.08906`, `qc1_kr3_c5` from
  `0.09819` to `0.12406`, and `qc1_kr3_c6` from `0.08992` to `0.11115`.
- Why not repeat directly: distant rows have different spatial/RoPE alignment;
  putting them in one key block makes one seed set explain unrelated key rows.
- Rule: when widening key blocks, keep key rows spatially contiguous unless a
  new position-aware seed or routing objective is introduced first.

### C2/C3 Seed-Logit Reuse in Merge

- Result: correctness matched the PyTorch reference, but real Wan QKV timing
  was slower, so the change was reverted.
- Change tested: in `hard_c2_rt_triton.py` and `hard_c3_rt_triton.py`, reuse
  the already computed seed-key logits `s0/s1(/s2)` inside the per-seed merge
  loop instead of recomputing the same seed-query/key dot products.
- Evidence: scai7/H200 experiment
  `hard_c2_c3_reuse_logits_realqkv_20260813` recorded all 360 self-attention
  calls over three prompts. `hard_c2` measured `7.641` ms/call with rel-L2
  `0.11824`; the existing frontier value was about `7.468` ms/call with the
  same error. `hard_c3` measured `12.596` ms/call with rel-L2 `0.08771`; the
  existing frontier value was about `11.766` ms/call with the same error.
- Why not repeat directly: eliminating the duplicate dot products increases
  the live score tensors in the merge kernel. On H200 this extra register/live
  state cost outweighed the saved tensor-core work for C2/C3.
- Rule: keep the original recompute-inside-merge-loop form for hard C2/C3
  unless the whole merge accumulator layout is redesigned.

### Spread C4 Seed Rows

- Result: worse dense-vs-hard output error with no useful speed gain.
- Change tested: replace local-with-ith C4 seed rows with spread rows
  `k, k+7, k+14, k+21 (mod 30)` for each key row `k`, while preserving hard
  assignment, entropy routing bias, and full compressed attention.
- Evidence: Slurm job `367149` on real Wan QKV measured spread-seed hard C4 at
  about 23.79 ms/call with rel-L2 `0.13442` and MSE `0.005189` versus dense.
  The preceding local-seed cleanup benchmark `367140` measured about
  23.92 ms/call with rel-L2 `0.09724` and MSE `0.002716` versus dense.
- Why not repeat directly: the spread seeds make the four representatives more
  globally distributed across the 30 query rows, but they lose the row-local
  alignment that kept each key row close to its matched query neighborhood.
- Rule: keep local-with-ith seed rows for the main hard C4 path. Treat spread
  seed rows only as an explicit low-quality ablation.

### Random-With-Ith C4 Seed Rows

- Result: worse dense-vs-hard output error than local-with-ith.
- Change tested: keep the index-matched seed row `k`, but choose the other
  three C4 seed rows by deterministic random sampling instead of local rows
  near `k`.
- Evidence: Slurm jobs `368147`, `368154`, and `368155` measured all 90
  self-attention calls for one Wan denoise trajectory on MI300X. Averaged over
  random seeds 0, 1, and 2, local-with-ith had rel-L2 `0.09011` and MSE
  `0.002514` versus dense, while random-with-ith had rel-L2 `0.11982` and MSE
  `0.004446`. Fully random was much worse, with rel-L2 `0.22024` and MSE
  `0.01502`.
- Why not repeat directly: preserving the matched row is not enough. The other
  three randomly sampled rows often lose the row-local positional alignment
  that makes the merged K/V useful for the 30-query by 52-key tile.
- Rule: keep local-with-ith seed rows for the main hard C4 path. Use
  random-with-ith or fully random only as explicit ablations.

### Fixed Geometry Four-Seed Variants

- Result: no fixed geometry variant beat local-with-ith.
- Change tested: keep the same hard C4 assignment, entropy routing bias, KV
  merge, and full compressed attention, but change only the four seed query
  rows. Tested nearest-right, nearest alternating, ring stride-5, ring
  stride-7, ring quarter, and ring symmetric-7 choices.
- Evidence: Slurm job `368295` measured the first 12 real Wan self-attention
  calls on MI300X. Local-with-ith had rel-L2 `0.13721` and MSE `0.002850`.
  The closest fixed alternatives were nearest-right with rel-L2 `0.13767` and
  nearest-alternating with rel-L2 `0.13772`; all ring/stride variants were
  worse, with rel-L2 around `0.147-0.151`.
- Why not repeat directly: spreading the non-matched seeds or changing the
  local tie direction does not improve the block-local query/key alignment.
- Rule: do not retry purely fixed geometric four-row seed patterns unless a
  new pattern is motivated by a different analysis. For higher precision, use
  content-adaptive seed selection instead of another fixed offset sweep.

### Adaptive Seed Objectives After Greedy Cover

- Result: no tested adaptive objective beat greedy-cover-with-ith.
- Change tested: keep the same four-seed hard C4 framework and force the first
  seed to be the original ith/index-matched row, but choose the other three
  seeds with alternative content-adaptive objectives: logsumexp coverage,
  query-row softmax-probability coverage, attention-mass coverage, top
  logsumexp, top attention-mass, and top mean.
- Evidence: Slurm job `368826` screened the new objectives over two prompts
  and the first eight self-attention calls per prompt. Greedy-cover-with-ith
  had the lowest rel-L2, `0.09248`; the closest new mode,
  greedy-prob-cover-with-ith, had rel-L2 `0.09284`. Slurm job `368827` then
  ran a fuller three-prompt / 270-call benchmark comparing the strongest
  candidates. Greedy-cover-with-ith remained best with rel-L2 `0.07967` and
  MSE `0.001872`, while greedy-prob-cover-with-ith had rel-L2 `0.08065` and
  MSE `0.001918`.
- Why not repeat directly: these objectives are reasonable alternatives, but
  their extra normalization or mass weighting did not reduce dense-output
  error beyond the simpler max-logit greedy coverage objective.
- Rule: treat greedy-cover-with-ith as the best precision-first four-seed
  selector found so far. Revisit adaptive seed selection only with a genuinely
  different objective, such as a local dense-output proxy or a more global
  multi-block objective.

### No-Ith Adaptive Greedy Seeds

- Result: removing the mandatory ith/index-matched seed did not improve the
  full benchmark.
- Change tested: choose all four seed rows adaptively from the 30 query rows,
  instead of forcing the first seed to be the key-row-matched ith query. Tested
  greedy-cover-no-ith and greedy-prob-cover-no-ith while preserving hard C4
  assignment, routing bias, KV merge, and full compressed attention.
- Evidence: Slurm job `368831` showed a small pilot advantage for
  greedy-cover-no-ith over greedy-cover-with-ith on two prompts / first eight
  calls per prompt: rel-L2 `0.09186` versus `0.09248`. The larger three-prompt
  / 270-call Slurm job `368835` reversed this: greedy-cover-with-ith was best
  with rel-L2 `0.07967` and MSE `0.001872`, while greedy-cover-no-ith had
  rel-L2 `0.08454` and MSE `0.002108`.
- Why not repeat directly: the no-ith pilot signal was not stable. The matched
  ith seed appears to be an important anchor once all layers and denoise steps
  are included across more prompts.
- Rule: keep the ith seed in precision-first adaptive C4 seed selection unless
  a new objective is validated on a full multi-prompt benchmark.

### Approximate Greedy Seed Selectors

- Result: prefilter-then-greedy is a useful lower-complexity approximation,
  but it does not beat exact greedy-cover in error.
- Change tested: keep the same four-seed hard C4 path and force the first seed
  to be the original ith/index-matched row, but reduce seed-selection cost by
  either sampling key columns, prefiltering query-row candidates with
  `q_i dot mean(K)`, or combining both. The main variants were
  `sample16_greedy_with_ith`, `prefilter12_greedy_with_ith`, and
  `prefilter12_sample16_greedy_with_ith`.
- Evidence: Slurm job `369096` ran the three-prompt / 270-call benchmark on
  MI300X. Exact greedy-cover-with-ith remained the lowest-error method with
  rel-L2 `0.07967` and MSE `0.001872`. The best approximation,
  prefilter12-greedy-with-ith, was very close with rel-L2 `0.08034` and MSE
  `0.001903`, while lowering measured PyTorch analysis time from about
  `578.3` ms/call to `462.9` ms/call. Prefilter12+sample16 was faster at
  `406.6` ms/call but had higher rel-L2 `0.08126`.
- Why not repeat directly: the ranking is now clear for this approximation
  family. Prefilter12-greedy is the best accuracy/speed tradeoff; key sampling
  adds more error for the tested sample size.
- Rule: if a lower-cost adaptive seed selector is needed, use
  prefilter12-greedy-with-ith before trying more sampled-key variants. Keep
  exact greedy-cover-with-ith as the precision-first reference.

### Naive Per-Block Triton Prefilter-Greedy Seed Selector

- Result: correct in FP32, but slower than the PyTorch analysis path and not
  worth integrating into the main attention backend.
- Change tested: write one Triton program per
  `(query_outer, key_outer, key_row, query_col)` block for C3
  `prefilter8_greedy`. The kernel keeps the ith seed, computes
  `q_i dot mean(K)` over all 30 query rows to choose seven extra candidates,
  computes candidate logits against the 52 key columns, then greedily selects
  the two extra seed rows.
- Evidence: scai7/H200 synthetic Wan-shaped chunk on 2026-08-13
  (`A=1,F=21,30x52,D=128,B=H=1`) measured FP32 exact agreement with PyTorch
  but `4.06` ms for Triton versus `0.92` ms for PyTorch. BF16 matched only
  `99.07%` of seed indices because small numeric ranking differences change
  top-k/greedy choices, and it was still slower (`3.42` ms versus `0.89` ms).
- Why not repeat directly: this kernel exposes too little reusable work per
  program and does scalar/vector reductions instead of efficient tensor-core
  work. It also duplicates the key-row load independently for every query
  column.
- Rule: do not integrate the naive one-block-one-program adaptive seed selector.
  If adaptive C3 seed selection is revisited, tile multiple query columns per
  program and reuse each loaded key row across those columns, or use a simpler
  fixed/local seed rule.

### Tiled Triton Prefilter-Greedy Seed Selector

- Result: reusing a loaded key row across multiple query columns did not make
  C3 prefilter8-greedy seed selection fast enough.
- Change tested: extend the prototype seed selector so one Triton program
  handles multiple query columns (`BLOCK_J=2,4,8,16`) for the same
  `(query_outer, key_outer, key_row)`, reusing the loaded `52 x D` key row and
  its mean across columns.
- Evidence: scai7/H200 synthetic Wan-shaped chunk on 2026-08-13 measured BF16
  scalar Triton at `3.42` ms versus tiled `J=2/4/8` at
  `3.95`/`4.26`/`4.64` ms, while PyTorch remained `0.89` ms. FP32 matched the
  PyTorch reference exactly, but scalar Triton was `4.07` ms and tiled
  `J=2/4/8/16` was `3.95`/`4.14`/`4.46`/`5.25` ms.
- Why not repeat directly: K reuse reduces repeated K loads, but the unrolled
  per-column top-k and greedy coverage logic increases live state/register
  pressure and scalar control work. Larger query-column tiles get worse, not
  better.
- Rule: do not pursue prefilter-greedy as a standalone per-block Triton seed
  selector. Adaptive seed selection needs a different formulation, likely one
  that uses dense tensor-core matmul/top-k primitives outside the merge kernel
  or a much simpler fixed seed rule.

### No-Bias Routing

- Result: bad video quality; rejected by visual inspection.
- Change tested: remove the routing-bias term after KV compression.
- Evidence: no-bias C4 configs such as
  `configs/token_merging_wan_hard_route_c4_fastseed_nobias_fewstep.yaml` and
  `configs/token_merging_wan_hard_route_c4_fastseed_outer_local_r4_nobias_fewstep.yaml`
  generated worse videos than the bias-preserving hard C4 path.
- Why not repeat: the routing bias carries the entropy/log-normalizer
  correction from the first-stage key assignment and merge. Dropping it changes
  the approximation too much, even if it can reduce memory traffic.
- Rule: do not use `*_nobias` merge/routing backends for the main hard C4 path.
  Keep them only as explicit ablations if needed.

### Global or Shared Compression

- Result: bad quality compared with grid-local compression.
- Change tested: compress KV once globally, share one compression across too
  many query groups, or otherwise remove the per-grid compression.
- Why not repeat: different query/key grids need different compressed KV
  summaries. Sharing compression destroys this local adaptation.
- Rule: keep the non-shared grid-local hard-route path for quality-preserving
  experiments.

### Sparse, Sliding-Window, or Local Second-Stage Attention

- Result: rejected for the main method because it changes the requested core
  attention semantics.
- Change tested: route each query only to a local/exact R-window of compressed
  key outers after token merging.
- Why not repeat: this can be faster, but it is no longer full compressed
  attention after merging.
- Rule: use only full compressed attention for the second stage in the main
  method. R-window variants are ablations only.

### Split-Full Fused Kernel

- Result: too slow in the smoke benchmark.
- Change tested: compute per-key-outer partial full-attention stats and reduce
  them afterward to avoid writing all compressed K/V.
- Evidence: `analysis/hard_c4_split_full_smoke_20260807/metrics.json` on AMDHPC
  reported about 1814 ms/call, far slower than the direct-route path.
- Why not repeat directly: it preserves full attention, but the multi-kernel
  partial/reduce implementation has too much overhead.
- Rule: only revisit if the fusion design avoids large partial buffers and
  avoids launching a separate reduction over every group.

### Key-Row-Block Merge Variants

- Result: no speed improvement.
- Change tested: process multiple key rows per Triton program (`kr2`, `kr3`,
  `kr5`) to reduce repeated launch/work overhead.
- Evidence: `analysis/hard_c4_kr_smoke_20260807/metrics.json` showed all tested
  variants slower than the current head-major direct-route path while matching
  its output.
- Rule: do not retry the same key-row blocking unless the memory layout or
  kernel structure changes substantially.

### Token-Major Routing `n8` Tile

- Result: slower than the token-major `n16` routing tile, and not competitive
  with the later `n32` default.
- Change tested: use `triton_direct_m32_n8_w1_s1_exp2_tokenmajor` after the
  token-major C4 merge.
- Evidence: `analysis/hard_c4_tokenmajor_n8_n16_20260807/metrics.json` on
  AMDHPC measured about 26.29 ms/call for `n8` versus about 23.57 ms/call for
  `n16`, with the same output error.
- Rule: do not use `n8` for the main token-major routing path.

### Token-Major Routing Larger-M Tiles

- Result: slower than the token-major `m32` routing family.
- Change tested: keep the same C4 merge output and full compressed attention,
  but use token-major direct routing tiles `m64_n16`, `m64_n32`, `m128_n16`,
  `m128_n32`, and `m128_n32_w4`.
- Evidence: Slurm job `366571` measured about 22.90 ms/call for the current
  `m32_n16` full hard C4 path, versus about 24.01 ms/call for `m64_n32`,
  25.22 ms/call for `m64_n16`, 25.51 ms/call for `m128_n32_w4`, 48.62
  ms/call for `m128_n16`, and 50.95 ms/call for `m128_n32`; output error was
  unchanged.
- Why not repeat directly: Q_ROWS is only 30, so larger `BLOCK_M` shapes waste
  lanes/registers in this direct-route layout.
- Rule: keep `BLOCK_M=32` for token-major direct routing unless the routing
  layout changes substantially.

### Grouped `g2` Token-Major Routing

- Result: slower than the ungrouped token-major routing path with identical
  output error.
- Change tested: process two adjacent route groups per Triton routing program
  using `triton_direct_g2_*_tokenmajor`, while keeping the same C4 merge output
  and full compressed attention.
- Evidence: Slurm job `366953` measured dense at about 23.41 ms/call,
  the old `hard_c4_bestlogit_tokenmajor` routing tile `m32_n16` at about
  24.17 ms/call, `m32_n32` at about 23.36 ms/call, `g2_m32_n16` at about
  25.06 ms/call, and `g2_m16_n16` at about 28.28 ms/call. All hard-route
  variants had the same dense-vs-output rel-L2, about 0.099699.
- Why not repeat directly: grouping two route groups reduces program count,
  but doubles the live K/V/Q/accumulator state and loses more to register
  pressure than it gains from launch reduction.
- Rule: do not use `triton_direct_g2_*_tokenmajor` for the main hard C4 route.
  The current routing default is `triton_direct_m32_n32_w1_s1_exp2_tokenmajor`.

### Token-Major Routing `n16` Versus `n32`

- Result: `n32` is the current default because it was slightly faster than
  `n16` in the latest multiprompt real-QKV benchmark with unchanged error.
- Change tested: compare `triton_direct_m32_n16_w1_s1_exp2_tokenmajor` against
  `triton_direct_m32_n32_w1_s1_exp2_tokenmajor` after the same bestlogit C4
  merge.
- Evidence: Slurm job `366953` measured `m32_n16` at about 24.17 ms/call and
  `m32_n32` at about 23.36 ms/call over 180 attention calls, both with
  rel-L2 about 0.099699 versus dense.
- Rule: use `m32_n32_w1_s1_exp2` as the default token-major routing tile until
  a more substantial routing or merge-layout change is validated.

### Generic SDPA Routing Fallback

- Result: slower than direct Triton routing with essentially the same output
  error.
- Change tested: keep the direct C4 merge output, but transpose the compressed
  KV/bias tensors to group-major layout and run the second-stage full
  compressed attention through PyTorch SDPA instead of the direct-route Triton
  kernel.
- Evidence: `analysis/hard_c4_sdpa_routing_20260807/metrics.json` measured
  about 29.83 ms/call for the SDPA routing fallback, versus about 24.50 ms/call
  for direct-route `m32_n32` and about 22.55 ms/call for dense attention. The
  dense-vs-output rel-L2 was effectively unchanged.
- Why not repeat directly: SDPA must consume an additive routing bias and a
  transposed group-major view, so it does not recover enough optimized Flash
  Attention behavior to compensate for the layout/mask overhead.
- Rule: keep the direct-route Triton routing kernel for the main hard C4 path.
  Only revisit SDPA routing if the compressed KV/bias layout changes enough to
  remove the transpose and mask overhead.

### Augmented-Bias SDPA Routing

- Result: compiles and preserves the same dense-vs-hard error, but is much
  slower than the current direct-route Triton path.
- Change tested: represent the routing bias as an extra Q/K channel
  (`q_aug=[q,1]`, `k_aug=[k,-bias/scale]`) and call PyTorch SDPA without an
  additive mask.
- Evidence: corrected real Wan benchmark Slurm job `366946` measured dense at
  about 22.60 ms/call, current `hard_c4_bestlogit_tokenmajor` at about
  23.34 ms/call, and augmented-bias SDPA at about 71.48 ms/call. The rel-L2
  versus dense stayed about 0.155913. A synthetic smoke job `366945` similarly
  measured about 0.204 ms/call for current direct routing versus about
  0.635 ms/call for augmented-bias SDPA.
- Why not repeat directly: avoiding the explicit SDPA mask is not enough. This
  path still materializes augmented K, changes layout, and loses the custom
  token-major direct-routing kernel.
- Rule: do not absorb routing bias into augmented Q/K for the main PyTorch SDPA
  routing path. Keep the bias-preserving direct-route Triton path unless a new
  design avoids both mask overhead and extra augmented-K materialization.

### Seed-Key Approximate Merged K

- Result: faster, but the output error is too high for the accuracy-preserving
  path.
- Change tested: replace the softmax-merged K with seed-query-derived K
  approximations such as `seedkey`, `seedkeymatch`, and `seedkeyproj`, while
  still running full compressed attention after token merging.
- Evidence: corrected Wan benchmark Slurm job `366947` measured dense at about
  22.77 ms/call and current hard C4 at about 23.60 ms/call with rel-L2
  0.155913 versus dense. `seedkey_n32_s1` was faster at about 14.06 ms/call,
  but rel-L2 rose to about 0.604. `seedkeymatch` had rel-L2 about 0.480, and
  `seedkeyproj` had rel-L2 about 0.359.
- Why not repeat directly: the speedup comes from changing the compressed K
  too aggressively, not from a safe memory-traffic reduction.
- Rule: do not replace merged K with seed-query, bias-matched seed-key, or
  scalar-projected seed-key approximations in the main hard C4 path.

### Compact/Merge-Weight Reconstruction Variants

- Result: no speedup in synthetic smoke, with tiny numerical differences for
  K/V reconstruction variants.
- Change tested: avoid materializing some compressed K/V tensors by storing
  merge weights or compact assignments, then reconstructing V or K/V in the
  full compressed-attention routing kernel.
- Evidence: corrected smoke job `366948` measured current bestlogit at about
  0.199 ms/call, versus about 0.236-0.241 ms/call for value-only
  `compactweights`/`mergeweights`. Corrected smoke job `366951` measured
  current bestlogit at about 0.255 ms/call, versus about 0.286 ms/call for
  `compactweightskv` and about 0.423 ms/call for `mergeweightskv`.
- Why not repeat directly: the variants reduce materialized tensors but add
  reconstruction work and less favorable routing memory access, which erased
  the traffic savings in smoke tests.
- Rule: do not rerun the same compact/merge-weight reconstruction variants as
  main candidates unless the design changes enough to avoid source K/V
  reconstruction overhead.

### Token-Major Routing `m16`, `n64`, and Stage Sweeps

- Result: no useful speedup with unchanged error.
- Change tested: keep the same C4 token-major merge output and full compressed
  attention, but use routing variants `m16_n16`, `m16_n32`, `m16_n64`,
  `n64_w1`, `n64_w2`, and extra routing stages `s2`/`s3`.
- Evidence: AMDHPC sweeps
  `hard_c4_n64_routing_sweep_20260808`,
  `hard_c4_m16_routing_sweep_20260808`, and
  `hard_c4_stage_sweep_20260808` measured the best variants around
  22.98-23.42 ms/call, roughly tied with or slower than dense and not better
  than the later `m32_n32_w1_s1` default.
- Why not repeat directly: these settings only retile the second-stage full
  compressed attention and do not reduce the dominant intermediate K/V traffic.
- Rule: do not retry these routing tile/stage settings unless the compressed
  K/V layout or attention kernel structure changes.

### Current Fast Path with `j64` Merge Tile

- Result: slower than the current `j32` merge tile.
- Change tested: replace the current non-shared C4 merge tile with
  `triton_c4_direct_route_j64_w2_exp2merge_forcekeys_fp16kvbias_s1_fastdot`
  while keeping token-major output, local-with-ith seed initialization, routing
  bias, and full compressed attention.
- Evidence: `analysis/hard_c4_j64_nohead_20260808/metrics.json` on AMDHPC
  measured about 26.32 ms/call for j64 head-major and 25.84 ms/call for j64
  no-headmajor, versus 24.84 ms/call for current j32 head-major and 24.46
  ms/call for current j32 no-headmajor, with unchanged output error.
- Why not repeat directly: the wider tile covers all 52 key columns in one
  program, but the lower occupancy / higher register pressure outweighs the
  reduced tiling overhead.
- Rule: keep `j32_w2` for the current token-major hard C4 merge path.

### Same-Precision Merge Tile/Warp Sweep

- Result: slower than the current `j32_w2` merge tile.
- Change tested: same-precision hard C4 merge variants `j16_w2`, `j32_w1`,
  and `j32_w4` while keeping local-with-ith seeds, token-major output, full
  compressed attention, and the same output error.
- Evidence: Slurm job `366543` measured about 24.59 ms/call for the current
  `j32_w2` path, versus about 26.58 ms/call for `j16_w2`, 26.60 ms/call for
  `j32_w1`, and 33.27 ms/call for `j32_w4`.
- Why not repeat directly: these variants change launch geometry but do not
  reduce the dominant memory traffic enough to offset poorer occupancy or extra
  scheduling overhead.
- Rule: do not keep or rerun these same tile/warp variants unless the merge
  kernel structure changes substantially.

### Key-Outer Merge Blocking F1-F7

- Result: slower than the non-keyouter token-major merge path with unchanged
  dense-vs-hard error.
- Change tested: process multiple key-outer groups per Triton merge program
  (`f1` through `f7`) while keeping grid-local C4 compression, local-with-ith
  seeds, routing bias, token-major output, and full compressed attention.
- Evidence: AMDHPC sweep
  `hard_c4_f2to7_bf16kv_noplaceholder_smoke_20260808` measured about
  22.79 ms/call for `hard_c4_bestlogit_tokenmajor`, 23.22 ms/call for
  `hard_c4_full_core_tokenmajor`, and 31-34 ms/call for tested keyouter
  variants. `hard_c4_f1_keyouter_smoke_20260808` measured about 22.96 ms/call
  for `f1`, still not a clear improvement over the current path.
- Why not repeat directly: batching key outers increases register pressure and
  store complexity more than it reduces launch or key/value load overhead.
- Rule: do not use `_f1`...`_f7` key-outer blocking for the main hard C4 path
  unless the merge kernel is redesigned.

### Hard C4 RT Key-Outer Blocking

- Result: slower in the current simplified RT-style hard C4 kernel.
- Change tested: make one merge program process two key outer groups so the
  four seed-query tiles are loaded once and reused across two K/V groups.
- Evidence: scai7 H200 smoke
  `hard_c4_fblock2_smoke_20260812` matched the PyTorch reference but slowed
  the hard C4 Triton call from about `0.33` ms to about `0.67` ms for
  `query_outer=2`, `key_outer=21`, `heads=2`, `dim=128`.
- Why not repeat directly: batching key outers reduces repeated seed-query
  loads, but the larger Triton program loses more to register pressure and
  occupancy than it gains from reuse.
- Rule: keep one key outer per merge program in the RT-style hard C4 kernel
  unless the merge kernel is redesigned around lower live state.

### Hard C4 RT Q-Reuse Merge Program Order

- Result: slower than the maintained K/V-reuse program order.
- Change tested: decode the merge-kernel `block_id` with `key_outer` varying
  fastest inside each `(query_outer, key_row)`, so nearby programs reuse the
  four seed-query rows more directly. This preserves the exact hard C4 math,
  key-softmax assignment, routing bias, and full compressed attention.
- Evidence: scai7 H200 GPU0 smoke on 2026-08-13 matched the PyTorch reference,
  but near-real random-QKV timing for `query_outer=10`, `key_outer=21`,
  `heads=12`, `dim=128` measured about `7.99` ms/call, with merge at about
  `5.97` ms. The maintained order, where `query_outer` varies fastest inside
  each `(key_outer, key_row)`, measured about `7.73` ms/call, with merge at
  about `5.72` ms under the same GPU0 timing script.
- Why not repeat directly: seed Q traffic is larger than K/V traffic on paper,
  but the Q-reuse order harms K/V cache locality and scheduler behavior enough
  to dominate any seed-Q reuse benefit.
- Rule: keep the maintained merge order:
  `query_outer -> key_outer -> key_row` inside `block_id`.

### Hard C4 RT Route Program Axis Swap

- Result: no speed improvement over the maintained route program order.
- Change tested: swap the route-kernel launch/decode from
  `(row_tile, batch_head, query_group)` to `(row_tile, query_group, batch_head)`.
  This preserves the exact hard C4 output and only changes scheduling order.
- Evidence: scai7 H200 GPU0 smoke on 2026-08-13 matched the PyTorch reference.
  Near-real random-QKV profile measured the swapped order at about `7.75`
  ms/call full, with route `1.99` ms. The maintained order measured about
  `7.73` ms/call full, with route `1.96` ms under the same script.
- Why not repeat directly: batching heads next to each other is already at
  least as good for the current compressed-K/V layout and H200 scheduler.
- Rule: keep the route grid order `(row_tile, batch_head, query_group)`.

### Hard C4 RT Inline Route-Mask Simplification

- Result: slower, despite shorter-looking code.
- Change tested: remove the precomputed `route0`/`route1`/`route2`/`route3`
  masks and build `mask & (route == seed_id)` inside the seed loop.
- Evidence: scai7 H200 stage profile
  `hard_c4_mask_inline_stage_profile_20260812` increased the near-real
  `query_outer=10`, `key_outer=21`, `heads=12`, `dim=128` merge stage from
  about `6.15` ms to about `6.66` ms.
- Why not repeat directly: the extra per-seed mask expression increases the
  generated work/register pressure more than it reduces source-code clutter.
- Rule: keep the four explicit route masks in the hard C4 RT merge kernel.

### Hard C4 RT Warp Count Sweep on H200

- Result: `4` warps remains best for the maintained RT-style hard C4 kernel.
- Change tested: set the shared `HARD_RT_NUM_WARPS` constant to `2` and `8`,
  preserving the same C4 merge, routing bias, and full compressed attention.
- Evidence: scai7 H200 GPU0 random-QKV stage profile on 2026-08-13 measured
  the default `4` warps at about `7.73` ms/call full, with merge at `5.72` ms
  and route at `1.96` ms. `2` warps made the merge kernel pathological at
  about `268.09` ms and full `270.72` ms. `8` warps measured full `11.84` ms,
  with merge `9.70` ms and route `2.03` ms.
- Route-only recheck: temporarily keeping merge at 4 warps while setting only
  the route kernel to 1, 2, or 8 warps also failed to improve speed. Route
  timings were about `4.43` ms, `2.60` ms, and `2.03` ms respectively, versus
  `1.95` ms for the maintained 4-warp route kernel.
- Why not repeat directly: `2` warps underfeeds the 64x64 tensor-core merge
  tile, while `8` warps increases scheduling/register overhead.
- Rule: keep `HARD_RT_NUM_WARPS=4` unless the tile shape changes.

### Hard C4 RT Merge Stage Sweep on H200

- Result: no clear win over the maintained `HARD_RT_MERGE_NUM_STAGES=1`.
- Change tested: set only the merge kernel `num_stages` to `2` and `3`,
  preserving the same tile shape, warps, key-softmax C4 merge, routing bias,
  and full compressed attention.
- Evidence: scai7 H200 GPU0 random-QKV stage profile on 2026-08-13 measured
  the default stages-1 full call at about `7.73` ms with merge `5.72` ms.
  Stages-2 measured about `7.74` ms full with merge `5.71` ms. Stages-3
  measured about `7.73` ms full with merge `5.73` ms.
- Why not repeat directly: the differences are within timing noise and do not
  justify changing the simpler maintained config.
- Rule: keep `HARD_RT_MERGE_NUM_STAGES=1` unless the merge tile is redesigned.

### Hard C4 RT Seed-Q `evict_first`

- Result: slower than the maintained seed-Q `evict_last` cache hint.
- Change tested: set the four seed-query `tl.load` calls in the merge kernel
  to `eviction_policy="evict_first"`, while leaving K/V loads as
  `evict_last` and preserving the exact hard C4 math.
- Evidence: scai7 H200 GPU0 random-QKV stage profile on 2026-08-13 measured
  seed-Q `evict_first` at about `8.02` ms/call full, with merge `6.05` ms and
  route `1.94` ms. The maintained all-merge-input `evict_last` profile was
  about `7.73` ms/call full, with merge `5.72` ms and route `1.96` ms.
- Why not repeat directly: although seed-Q reuse looks weaker than K/V reuse
  from the launch order, the cache hint change made the merge kernel slower in
  the actual H200 profile.
- Rule: keep `evict_last` on the merge kernel's seed-Q loads.

### Hard C4 RT Intermediate Store Cache Modifiers

- Result: `evict_first` on the intermediate merged-K/V and routing-bias stores
  is the best tested store hint so far; `.cg` and `.cs` are not better.
- Change tested: vary only the three merge-kernel `tl.store` calls that write
  intermediate merged K, merged V, and routing bias. The hard C4 routing,
  key-softmax assignment, compressed K/V values, and second-stage full
  attention are unchanged.
- Evidence: scai7 H200 GPU0 random-QKV stage profile on 2026-08-13 measured
  plain stores at about `7.73` ms/call full, store `evict_first` at about
  `7.39` ms/call full, `.cs`-only at about `7.40` ms/call full, and
  `.cg`-only at about `7.67` ms/call full. Combining `.cs` with
  `evict_first` does not compile in ptxas. Smoke tests matched the PyTorch
  reference with unchanged error.
- Why not repeat directly: this is an implementation hint only; it does not
  change the algorithm. Keep the fastest measured hint unless the intermediate
  layout changes again.
- Rule: use `eviction_policy="evict_first"` for intermediate merged-K/V and
  routing-bias stores, with no `cache_modifier`.

### Hard C4 RT Direct Pairwise Route Masks

- Result: much slower on realistic H200 shape, despite matching the PyTorch
  reference.
- Change tested: replace the explicit `route` id matrix and update chain with
  direct pairwise masks over normalized seed scores:
  `route0/route1/route2/route3`. Tie-breaking was kept identical to the
  maintained `>` update order, so this changed only the Triton expression.
- Evidence: scai7 H200 GPU0 random-QKV smoke on 2026-08-13 matched the
  PyTorch reference at the usual BF16 tolerance. The A=10 random stage profile
  measured full hard-C4 at about `10.82` ms/call with merge `8.88` ms, versus
  the maintained store-hinted path at about `7.39` ms/call with merge `5.42`
  ms.
- Why not repeat directly: keeping all four normalized score matrices live and
  doing pairwise mask comparisons increases register/scheduling pressure much
  more than removing the integer route matrix helps.
- Rule: keep the explicit `route` id update chain and derive four route masks
  from it.

### Hard C4 RT Route-Kernel Delayed V Load and Output Store Hint

- Result: no speed win over the maintained route kernel.
- Changes tested independently:
  delay the route-kernel merged-V load until after compressed logits and
  softmax probabilities are computed, and add `eviction_policy="evict_first"`
  to the final output store.
- Evidence: scai7 H200 GPU0 random-QKV smoke on 2026-08-13 matched the
  PyTorch reference. The delayed-V route profile measured about `7.41` ms/call
  full, and the output-store-hint profile also measured about `7.42` ms/call
  full. The maintained store-hinted path measured about `7.39` ms/call full in
  the same profile setup.
- Why not repeat directly: the route stage is already small compared with
  merge, and these changes slightly worsen scheduling/cache behavior.
- Rule: keep the route-kernel merged-V load before softmax arithmetic and keep
  the final output store without an explicit cache/eviction hint.

### Hard C4 RT Drop Head-Dim Mask

- Result: no speed win on the realistic H200 shape.
- Change tested: remove the `dims < D_HEAD` predicate from hard-C4 merge and
  route loads/stores, relying on the Wan fast path where `D_HEAD=128` and the
  Triton block dimension is also 128. This preserves output for the maintained
  Wan configuration and only removes redundant-looking mask expressions.
- Evidence: scai7 H200 GPU0 random-QKV smoke on 2026-08-13 matched the
  PyTorch reference. The A=10 random stage profile measured about `7.41`
  ms/call full with merge `5.41` ms and route `1.98` ms, not better than the
  maintained store-hinted path at about `7.39` ms/call full.
- Why not repeat directly: Triton already appears to optimize the all-true dim
  mask well enough; removing it reduces generic safety without improving the
  real Wan timing.
- Rule: keep the explicit dim mask in the maintained kernel.

### Hard C4 RT Packed Intermediate K/V

- Result: slower overall, although the second-stage route kernel alone became
  slightly faster.
- Change tested: replace separate intermediate `merged_k` and `merged_v`
  tensors with one packed workspace laid out as
  `[B, A, F, row, col, C, head, kv, dim]`, so each compressed token's K and V
  vectors are adjacent in memory. The hard C4 merge, routing bias, and full
  compressed attention math were unchanged.
- Evidence: scai7 H200 GPU0 small random smoke on 2026-08-13 matched the
  PyTorch reference. The realistic A=10 stage profile initially hit illegal
  memory access because the packed tensor's element offset exceeded int32
  range; fixing the packed offsets with int64 made it run, but measured about
  `7.89` ms/call full with merge `5.93` ms and route `1.93` ms. The
  maintained separate-K/V store-hinted path measured about `7.39` ms/call full
  with merge `5.42` ms and route about `1.98` ms.
- Why not repeat directly: packing improves K/V locality in route slightly,
  but the larger packed address space forces int64 offset arithmetic and hurts
  the dominant merge stage more than route improves.
- Rule: keep separate intermediate `merged_k` and `merged_v` tensors unless a
  later layout also avoids int64 offsets or removes the intermediate stores.

### Hard C4 RT Input/Route Load Default Cache Hints

- Result: no speed win over the maintained cache hints.
- Changes tested independently:
  remove `evict_last` from the merge-kernel input K/Q/V loads, and remove
  `evict_first` from the route-kernel merged-K/V/bias loads.
- Evidence: scai7 H200 GPU0 random-QKV smoke on 2026-08-13 matched the
  PyTorch reference in both cases. Removing merge-input hints measured about
  `7.44` ms/call full with merge `5.45` ms. Removing route-load hints measured
  about `7.41` ms/call full with route `1.98` ms. The maintained store-hinted
  path measured about `7.39` ms/call full.
- Why not repeat directly: the current cache hints are already close to the
  best measured behavior for this two-stage layout.
- Rule: keep `evict_last` on merge input K/Q/V loads and `evict_first` on
  route intermediate K/V/bias loads.

### Hard C4 RT Pair-Seed Merge Dots

- Result: much slower on realistic H200 shape.
- Change tested: compute seed `0/1` merged K/V with one larger
  `[2*BLOCK_J, BLOCK_L] @ [BLOCK_L, D]` dot and seed `2/3` with another,
  instead of four separate per-seed K dots and four separate per-seed V dots.
  The route assignment, cluster softmax weights, routing bias, and stored
  compressed tokens were mathematically unchanged.
- Evidence: scai7 H200 GPU0 random-QKV smoke on 2026-08-13 matched the
  PyTorch reference. The A=10 stage profile measured full hard-C4 at about
  `10.49` ms/call with merge `8.55` ms, versus the maintained store-hinted
  path at about `7.39` ms/call with merge `5.42` ms.
- Why not repeat directly: reducing dot call count is outweighed by the larger
  128x128 accumulator and extra live state from concatenated weights/biases.
- Rule: keep four separate per-seed K dots and four separate per-seed V dots
  unless the whole merge accumulator layout is redesigned.

### Hard C4 RT Route Tile/Stage Recheck After Store-Hint Optimization

- Result: the maintained route parameters are still best after adding
  intermediate-store `evict_first`.
- Changes tested: recheck route `BLOCK_K=16`, route `num_stages=1`,
  route `num_stages=3`, `BLOCK_I=16`, and `BLOCK_I=64` on the current
  store-hinted H200 path. The hard C4 math and first-stage merge were
  unchanged.
- Evidence: scai7 H200 GPU0 random-QKV smoke on 2026-08-13 matched the
  PyTorch reference in all cases. A=10 stage profiles measured
  `BLOCK_K=16` at about `7.46` ms/call full, route `num_stages=1` at about
  `7.71` ms, route `num_stages=3` at about `7.47` ms, `BLOCK_I=16` at about
  `7.86` ms, and `BLOCK_I=64` at about `7.95` ms. The maintained
  `BLOCK_I=32`, `BLOCK_K=32`, route `num_stages=2` profile remains about
  `7.39` ms/call full.
- Why not repeat directly: the latest store hint changed intermediate I/O, but
  not enough to alter the route-kernel tile optimum.
- Rule: keep `HARD_RT_BLOCK_I=32`, `HARD_RT_BLOCK_K=32`, and
  `HARD_RT_ROUTE_NUM_STAGES=2`.

### Hard C4 Streaming No-Intermediate Prototype

- Result: much slower even on a small H200 shape, so it was not promoted into
  the maintained code.
- Change tested: prototype a single streaming Triton kernel that does not
  materialize intermediate merged K/V/bias. For each query column, it loops
  over key rows, recomputes the same hard-C4 compression in registers, and
  immediately applies the resulting compressed tokens to the second-stage
  online softmax. This preserves the high-level hard-C4 formula and full
  compressed attention, but changes the implementation from two tensor-core
  friendly stages into many small per-row reductions.
- Evidence: scai7 H200 GPU0 small random profile on 2026-08-13 with
  `B=1, A=1, K_OUTER=2, ROWS=30, COLS=52, HEADS=2, D=128` measured the
  maintained two-stage kernel at about `0.107` ms/call and the streaming
  prototype at about `0.494` ms/call. The streaming output differed from the
  current Triton path at rel-L2 about `0.0303`, mainly because the prototype
  used scalar reductions rather than the same tensor-core dot/store/load path.
- Why not repeat directly: avoiding intermediate HBM traffic is not enough if
  the kernel loses the large `64x64` and `32x128` tensor-core-friendly tiles.
  A viable fused design would need to keep those tile shapes while eliminating
  the materialized workspace.
- Rule: do not use the per-query-column streaming design. Future fusion work
  must preserve tensor-core-friendly merge/route tiles.

### Hard C4 RT Seed-Max Cluster Softmax

- Result: no speedup over the maintained cluster-local max.
- Change tested: in the seed-cluster merge softmax, use the seed's global
  key-softmax max `m_c` as the exponent baseline instead of recomputing a
  masked max over only the keys assigned to that seed. This is mathematically
  equivalent in exact arithmetic and preserves key-softmax assignment, routing
  bias, and full compressed attention.
- Evidence: scai7 H200 GPU0 smoke on 2026-08-13 matched the PyTorch reference
  at the same error level, but the random-QKV stage profile measured about
  `7.74` ms/call full with merge `5.74` ms. The maintained cluster-local max
  baseline measured about `7.73` ms/call full with merge `5.72` ms under the
  same script.
- Why not repeat directly: removing the masked max does not reduce the real
  merge time; any saved reduction work is offset by generated-code or
  numerical-scaling overhead.
- Historical rule: changing only the exponent baseline did not help this older
  kernel. The later winner-probability reuse result in "Restored Hard-C4 2D
  Score Tiles" supersedes this rule because it also eliminates three full
  score-tile exponentials and shortens the score live ranges.

### Hard C4 RT `BLOCK_J=32`

- Result: slower than the maintained `BLOCK_J=64`.
- Change tested: split the 52 query columns in the merge kernel into two
  `BLOCK_J=32` programs instead of one padded `BLOCK_J=64` program, while
  preserving `BLOCK_L=64`, local C4 seeds, key-softmax assignment, routing
  bias, and full compressed attention.
- Evidence: scai7 H200 GPU0 random-QKV stage profile on 2026-08-13 measured
  `BLOCK_J=32` at about `10.37` ms/call full, with merge `8.30` ms and route
  `1.95` ms. The maintained `BLOCK_J=64` measured about `7.73` ms/call full,
  with merge `5.72` ms and route `1.96` ms under the same script.
- Why not repeat directly: the smaller score tile reduces per-program live
  state, but doubles the number of merge programs and loses more scheduling
  efficiency than it gains.
- Rule: keep `HARD_RT_BLOCK_J=64` for the current RT-style hard C4 merge.

### Hard C4 RT Prob-Dot-Then-Scale Merge

- Result: small-shape smoke looked faster, but the realistic chunk was slower.
- Change tested: in each seed cluster, compute `probs @ K` and `probs @ V`
  first, then multiply the merged outputs by `1 / sum(probs)`, instead of
  forming normalized `weights = probs / sum(probs)` before the two dot
  products. The formula is equivalent up to rounding and preserves key-softmax
  assignment, routing bias, and full compressed attention.
- Evidence: scai7 H200 smoke
  `hard_c4_probscale_smoke_20260812` improved the tiny
  `query_outer=2`, `heads=2` call from about `0.317` ms to about `0.302` ms.
  However, realistic A=10/H=12 stage profile
  `hard_c4_probscale_stage_profile_20260812` slowed the merge stage to about
  `6.25` ms, compared with about `6.17` ms for the maintained kernel.
- Why not repeat directly: moving normalization after the dot saves one
  elementwise divide over the score tile, but adds output scaling and live
  state in the tensor-core merge path; the real Wan chunk is the deciding
  benchmark.
- Rule: keep normalized merge weights before `weights @ K/V` in the hard C4 RT
  merge kernel.

### Invalid Hard C4 RT `BLOCK_L=32` Timing

- Result: fast but invalid.
- Change tested: launch the merge kernel with `BLOCK_L=32`.
- Evidence: scai7 H200 synthetic timing showed about `5.26` ms for the merge
  stage, but this configuration only covered the first 32 of the 52 key
  columns and skipped the remaining 20.
- Why not repeat directly: it changes the algorithm by dropping keys.
- Rule: do not compare `BLOCK_L < 52` timings unless the kernel is rewritten
  to correctly cover all key columns with an exact online/tiled reduction.

### Group-Major Inline-Seed Layout

- Result: slower than the current token-major layout with the same output
  error.
- Change tested: keep the same non-shared C4 full-core method and inline
  local-with-ith seed selection, but write/read compressed K/V in group-major
  layout instead of token-major layout.
- Evidence: Slurm job `366560` measured about 24.75 ms/call for
  `hard_c4_full_core_groupmajor_inline_seed`, versus about 22.98 ms/call for
  `hard_c4_full_core_tokenmajor`, with the same rel-L2 vs dense
  (`0.096872`).
- Why not repeat directly: routing reads are more contiguous in group-major
  layout, but the merge-side stores/load scheduling lose more than routing
  gains for the current kernel structure.
- Rule: keep token-major compressed K/V layout for the current hard C4 fast
  path unless the merge/routing fusion strategy changes.

### A2 Query-Outer Pair Kernel

- Result: slower than the current token-major direct-route kernel.
- Change tested: process two query outer groups in one Triton program to reuse
  key/value loads across neighboring query groups.
- Evidence: Slurm job `366561` measured about 31.95 ms/call for
  `hard_c4_full_core_a2_tokenmajor` in a smoke run, versus about 22.98 ms/call
  for the current token-major path in Slurm job `366560`, with similar output
  error.
- Why not repeat directly: the intended key/value reuse is outweighed by
  higher register pressure and lower occupancy in the fused two-outer program.
- Rule: do not keep an A2/query-outer-pair merge kernel unless the kernel is
  redesigned to reduce register pressure and revalidated against the current
  token-major path.

### Fully Fused Streaming Merge + Full Attention

- Result: much slower than the current two-kernel token-major path.
- Change tested: avoid writing compressed K/V/bias to global memory by
  recomputing each C4 merged token inside a single full-attention kernel and
  applying FlashAttention-style online softmax over all compressed tokens.
- Evidence: Slurm job `366569` measured about 526.72 ms/call for
  `hard_c4_fused_full_m32` and about 1008.94 ms/call for
  `hard_c4_fused_full_m16`, versus about 22.73 ms/call for
  `hard_c4_full_core_tokenmajor` in the same smoke benchmark. The error was
  similar, but the runtime was far worse.
- Recheck: Slurm jobs `367834` and `367835` reproduced the same direction with
  a simpler one-block streaming fused kernel. The vector-reduction version was
  1.59 ms vs 0.13 ms for `key_outer=2`, and 16.44 ms vs 0.32 ms for
  `key_outer=21`. Slurm jobs `367837`, `367838`, and `367866` then replaced
  the inner reductions with tiled matmuls; this improved accuracy and speed but
  was still slower than the current two-kernel path: 0.42 ms vs 0.12 ms for
  `key_outer=2`, 4.15 ms vs 0.32 ms for `key_outer=21`, and 87.95 ms vs
  10.57 ms for a near-real chunk with `query_outer=10`, `heads=12`.
- H200 recheck: scai7 smoke tests on 2026-08-13 tried a one-kernel streaming
  version that owns all 30 query rows and a query-column tile, then streams all
  key rows without storing merged K/V/bias. The `Tj=4` version matched the
  PyTorch reference error but measured about 0.486 ms versus 0.035 ms for the
  maintained two-stage `hard_c4` kernel on `query_outer=1`, `key_outer=1`,
  `heads=1`. Increasing to `Tj=8` also matched reference error but slowed to
  about 2.918 ms. The larger tile reduces theoretical IO but greatly increases
  accumulator/register pressure and scalar gather/update work.
- Why not repeat directly: the fused kernel saves global writes/reads of
  compressed K/V/bias, but loses the current merge kernel's reuse across query
  columns and the routing kernel's tensor-core tile over compressed tokens.
- Rule: do not fuse merge and full compressed attention by streaming one
  merged token at a time. Only revisit this direction if the design preserves
  both query-column reuse during merge and route-token tiling during full
  attention.

### Whole-Outer `query_outer_chunk=21`

- Result: unsafe for the current no-headmajor token-major hard C4 path.
- Change tested: process all 21 query outer groups in one chunk while keeping
  the same C4 merge and full compressed attention.
- Evidence: Slurm job `366526` hit a Triton illegal memory access while running
  the `chunk=21` case in
  `analysis/hard_c4_nohead_chunk_sweep_20260808`.
- Why not repeat directly: the current direct-route Triton path is not stable
  with the whole outer dimension in one chunk.
- Rule: do not set `query_outer_chunk=21` for the current hard C4 fast path
  unless the routing/merge kernel indexing is changed and revalidated.
  The later six-head dispatch does exactly that and supersedes this historical
  rule; whole-outer scheduling remains unsupported without `head_chunk=6`.

### Hard C4 RT `query_outer_chunk=21` With Int64 Offsets

- Result: the illegal memory access can be fixed, but the larger chunk is
  slower than the maintained `query_outer_chunk=10`.
- Change tested: cast compressed K/V store/load offsets to int64 so the
  `merged_k`/`merged_v` element offset can exceed the signed 32-bit range when
  all 21 query outers are processed in one hard-C4 RT chunk.
- Evidence: scai7 H200 synthetic
  `hard_c4_chunk21_int64_stage_profile_20260812` ran successfully with
  `query_outer=21`, but measured about `20.33` ms for the full two-stage hard
  C4 call. The chunk sweep
  `hard_c4_query_outer_chunk_sweep_20260812` estimated `chunk=10` around
  `18.30` ms for all 21 query outers, while `chunk=21` measured about
  `20.14` ms.
- Why not repeat directly: fixing the index width removes the crash, but the
  much larger intermediate K/V/bias workspace and int64 address arithmetic
  erase the launch-count savings.
- Rule: keep `query_outer_chunk=10` for the maintained hard-C4 RT path. Do not
  add int64 merged-K/V offsets unless a later design intentionally supports
  larger chunks and proves a speed win.

### Simplemerge Query-Outer Chunk Variants

- Result: no improvement over the default `query_outer_chunk=10`.
- Change tested: rerun the chunk sweep after switching the default merge kernel
  to corrected `bestlogit_simplemerge`.
- Evidence: Slurm job `366962` failed during warmup for
  `hard_c4_bestlogit_simplemerge_tokenmajor_chunk11` with a HIP illegal memory
  access, so larger two-chunk settings such as chunk11/chunk14 are unsafe in
  the current routing path. Slurm job `366963` measured chunk10 at about
  23.14 ms/call and chunk7 at about 23.58 ms/call on real Wan QKV, both with
  rel-L2 about 0.09417 versus dense.
- Why not repeat directly: larger chunks hit the same stability problem as the
  old whole-outer run, while smaller safe chunks increase scheduling overhead
  without reducing total K/V/bias traffic.
- Rule: keep `query_outer_chunk=10` for the simplemerge hard C4 default.

### Simplemerge BF16 Intermediate Format

- Result: slower in smoke and introduced small output differences versus the
  fp16-intermediate simplemerge default.
- Change tested: keep the same corrected bestlogit/simplemerge algorithm but
  write compressed K, V, or routing bias as bf16 (`bf16kv`, `bf16k`, `bf16v`,
  `bf16bias`) to match the Wan bf16 input dtype.
- Evidence: Slurm job `366965` smoke-tested these variants on MI300X. Default
  simplemerge measured about 0.1459 ms/call with exact agreement to corrected
  bestlogit. The bf16 variants measured about 0.158-0.160 ms/call and had
  small rel-L2 differences versus the default, about 0.0016-0.0049.
- Why not repeat directly: changing intermediate format does not reduce bytes
  versus fp16, but it did add conversion/scheduling cost and numerical drift.
- Rule: keep fp16 compressed K/V/bias for the default simplemerge path.

### Delayed Value-Block Load in Simplemerge

- Result: no speed improvement.
- Change tested: move the `value_block` load later in the simplemerge merge
  kernel to shorten V's live range during seed-key assignment.
- Evidence: Slurm job `366966` showed exact output agreement in smoke but no
  clear speed win. Real-QKV Slurm job `366967` measured about 23.18 ms/call,
  slower than the accepted simplemerge result from job `366959` at about
  23.04 ms/call, with identical dense rel-L2 about 0.09417.
- Why not repeat directly: Triton/ROCm did not translate the shorter apparent
  live range into a faster kernel on this shape.
- Rule: keep the original simplemerge V-load placement unless a broader kernel
  rewrite changes register allocation substantially.

### Quantized Intermediate K/V

- Result: rejected for the current optimization direction.
- Change tested: store the compressed hard C4 intermediate K/V in fp8 before
  the full compressed attention stage.
- Evidence: `analysis/hard_c4_fp8_smoke_20260808/metrics.json` showed a speed
  improvement in a smoke run, but it is a quantization method and it increased
  dense-vs-hard output error.
- Why not repeat: the current goal is same-precision memory-bound optimization,
  not quantization. Reducing memory traffic should be attempted without lowering
  K/V precision.
- Rule: do not use fp8/quantized intermediate K/V for the main hard C4 path
  unless quantization is explicitly reopened as a separate experiment.

### MI210 Smoke for MI300 Fastdot Backend

- Result: invalid hardware check for the current fast path.
- Change tested: run the MI300-oriented hard C4 direct-route backend
  `triton_c4_direct_route_j32_w2_exp2merge_forcekeys_fp16kvbias_s1_fastdot`
  on `mi2101x` as a provisional smoke test.
- Evidence: Slurm job `366607` failed during Triton compilation because MI210
  allowed dot input precisions were `ieee`, `bf16x3`, and `bf16x6`, while the
  current fastdot path requests `input_precision="tf32"`.
- Why not repeat directly: this is a hardware/compiler capability mismatch, not
  an attention-quality result. It cannot validate speed or correctness for the
  MI300 fastdot backend.
- Rule: benchmark the current fastdot hard C4 path on `mi3001x` for final
  decisions. Use MI210 only with a separate `ieee`/non-fastdot backend.

### Non-Power-of-Two Triton Merge Tiles

- Result: compile-time failure.
- Change tested: exact key/query tile sizes such as `j26`, `j52`, and `l52`
  for the current hard C4 direct-route merge kernel.
- Evidence: Slurm job `366687` wrote
  `analysis/hard_c4_kernel_variant_smoke_20260808/smoke.json`; every `j26`,
  `j52`, or `l52` variant failed with Triton reporting that `tl.arange` ranges
  must be powers of two.
- Why not repeat directly: the current kernel uses `tl.arange(0, BLOCK_J)` and
  `tl.arange(0, BLOCK_L)`, so non-power-of-two block sizes are not legal for
  this implementation.
- Rule: do not submit `j26`, `j52`, or `l52` direct-route backend strings.
  Exact-52 tiling would require a different kernel design that uses
  power-of-two blocks plus masks.

### Packed Token-Major K/V/Bias Layouts

- Result: unsafe or slower on the real Wan hard C4 shape.
- Change tested: packed token-major variants such as `packkv`, `packkvbias`,
  `packkvbiascompact`, `packkbias`, and `packkbiascompact`.
- Evidence: Slurm job `366724` ran real-shape smoke tests with
  `B=1, A=10, F=21, rows=30, columns=52, heads=12, dim=128`.
  `packkv`, `packkvbias`, `packkvbiascompact`, and `packkbias` triggered HIP
  illegal memory accesses. `packkbiascompact` completed, but measured about
  14.94 ms versus about 10.95 ms for the current bestlogit token-major
  baseline in the same smoke.
- Why not repeat directly: toy-shape smoke can pass, but the real Wan layout
  exposes indexing/backend issues. The only safe packed-key-bias compact path
  was still substantially slower and did not reduce total intermediate bytes.
- Rule: do not use packed K/V/bias hard C4 backends for the main path. Focus
  on designs that reduce materialized intermediate data rather than just
  packing the same K/V/bias tensors.

### Precomputed-Logit Materialization

- Result: much slower than both dense attention and the current hard C4
  bestlogit path.
- Change tested: during the hard C4 merge, write precomputed
  `q @ merged_key - routing_bias` logits and then run the second-stage softmax
  over those logits instead of reading materialized merged K.
- Evidence: Slurm jobs `366738` and `366739`. On real Wan QKV, dense attention
  measured 23.34 ms/call, current bestlogit hard C4 measured 24.46 ms/call,
  and precomputed-logit hard C4 measured 161.69 ms/call for fp16 logits and
  165.36 ms/call for fp32 logits. The dense-vs-hard error was also slightly
  worse than the current bestlogit path.
- Why not repeat directly: it reduces intermediate bytes on paper, but the
  precompute step loses the tensor-core-friendly full compressed-attention dot
  product and replaces it with many scalar reductions during merge.
- Rule: do not use precomputed-logit materialization for the main hard C4 path.
  Only revisit it if the logits can be produced with a tensor-core-friendly
  tiling that preserves the second-stage full attention semantics.

### Early Representative-Key K Approximations

- Result: no acceptable speed/error tradeoff.
- Change tested: replace the softmax-merged K with simpler representative-key
  choices such as `repkey`, `top2key`, or `tail4`, while still running full
  compressed attention after token merging.
- Evidence: AMDHPC smoke runs
  `analysis/hard_c4_repkey_smoke_20260807/metrics.json`,
  `analysis/hard_c4_top2key_smoke_20260807/metrics.json`, and
  `analysis/hard_c4_tail4_smoke_20260807/metrics.json`. `repkey` was only
  roughly tied with dense speed and increased dense rel-L2 from about 0.155 to
  0.181 in that smoke; `top2key` and `tail4` were slower than the direct-route
  baseline in the same setup.
- Why not repeat directly: these variants reduce or alter the K merge, but the
  quality loss or extra scheduling cost erased the benefit.
- Rule: do not rerun the exact `repkey`, `top2key`, or `tail4` representative-K
  variants. More constrained K approximations must be separately justified and
  measured against dense error.

### Merge-Side Max-Bias and Stripe Shortcuts

- Result: not acceptable for the main quality-preserving hard C4 path.
- Change tested: replace the entropy/routing-bias calculation with a max-logit
  shortcut (`maxbias`) or replace hard assignment with fixed key stripes
  (`stripe`) while still running full compressed attention in the second stage.
- Evidence: Slurm job `366956` on real Wan QKV measured dense at about
  23.39 ms/call. `hard_c4_bestlogit_maxbias_tokenmajor` was only slightly
  faster at about 23.05 ms/call but worsened rel-L2 from the corrected
  bestlogit value of about 0.09417 to about 0.11392. `stripe` was faster at
  about 17.92 ms/call but worsened rel-L2 to about 0.22431.
- Why not repeat directly: these shortcuts reduce some merge-side work, but
  they change the compression distribution or routing correction enough to
  break the dense-error target.
- Rule: keep the entropy-preserving bestlogit/simplemerge merge for the main
  path. Use `maxbias` or `stripe` only as explicit low-quality ablations.

### Simplemerge `j16` and `j64` Merge Tiles

- Result: no speed improvement over the default `j32` simplemerge tile.
- Change tested: keep corrected bestlogit/simplemerge semantics and only change
  the merge tile from `j32` to `j16` or `j64`.
- Evidence: Slurm job `366961` smoke-tested random hard C4 tensors on MI300X.
  Default `hard_c4_bestlogit_simplemerge_tokenmajor` measured about
  0.1438 ms/call. `j16` measured about 0.1453 ms/call and introduced a tiny
  numerical difference versus the default; `j64` matched output but measured
  about 0.1464 ms/call. Older real-QKV tests had already shown non-simplemerge
  `j16`/`j64` slower than `j32`.
- Why not repeat directly: the simplified kernel does not change the earlier
  tile-size conclusion; `j32` is still the best balance for 52 query columns.
- Rule: keep `j32_w2` for the main simplemerge path.

### Simplemerge Merge Warp Count

- Result: no real-QKV speed improvement over the default two-warp merge kernel.
- Change tested: keep corrected bestlogit/simplemerge semantics and only change
  the direct C4 merge kernel launch from `w2` to `w1` or `w4`.
- Evidence: Slurm job `366987` smoke-tested random hard C4 tensors and showed
  `w1` slightly faster than `w2`, but real Wan QKV Slurm job `366988` reversed
  that result. On real QKV, default `w2` measured about 22.85 ms/call,
  `w1` about 23.48 ms/call, and `w4` about 31.69 ms/call; all had identical
  dense rel-L2 about 0.09417.
- Why not repeat directly: the toy-shape occupancy signal did not hold on the
  real Wan tensor schedule, and `w4` increases resource pressure substantially.
- Rule: keep merge `w2` for the default simplemerge path.

### Simplemerge Merge Pipeline Stages

- Result: no smoke speed improvement over the default one-stage merge kernel.
- Change tested: keep corrected bestlogit/simplemerge semantics and only change
  the direct C4 merge kernel launch from `s1` to `s2` or `s3`.
- Evidence: Slurm job `366989` smoke-tested random hard C4 tensors on MI300X.
  Default `s1` measured about 0.1484 ms/call, `s2` about 0.1512 ms/call, and
  `s3` about 0.1488 ms/call; all matched output exactly.
- Why not repeat directly: extra pipeline stages did not improve this
  simplemerge tile and would add more scheduling/resource risk on real Wan QKV.
- Rule: keep merge `s1` for the default simplemerge path.

### H200 Hard-C4 RT Intermediate Layout Tweaks

- Result: no speed or accuracy improvement over the original C4 RT intermediate
  layout.
- Changes tested on scai7/H200 after the RT-style hard-C4 cleanup:
  - moving the intermediate compressed-KV physical layout from
    `[..., C, heads, dim]` to `[..., heads, C, dim]`;
  - changing the merge column tile from `BLOCK_J=64` to `BLOCK_J=32`;
  - storing the entropy routing bias in BF16 instead of FP32.
- Evidence: the one-prompt real-Wan-QKV benchmark with all 120 self-attention
  calls measured the original layout at about `18.54` ms/call with rel-L2
  `0.073345` and MSE `0.0016324`. The `[..., heads, C, dim]` layout measured
  about `19.35` ms/call with the same error. BF16 bias measured about `19.36`
  ms/call and slightly worsened error. The `BLOCK_J=32` random H200 smoke was
  slower than `BLOCK_J=64`.
- Rule: keep the original intermediate layout, `BLOCK_J=64`, and FP32 routing
  bias for the maintained `hard_c4` path.

### H200 Hard-C4 RT Merge-Scheduling Tweaks

- Result: no speed improvement from trying to reduce small intermediate values
  or reuse K/V across multiple query outer blocks inside one merge program.
- Changes tested on scai7/H200:
  - remove the explicit int32 `route` matrix and compute the four seed masks
    directly from the four key-softmax log-score matrices;
  - process `BLOCK_A=2` query outer blocks inside one merge program so the same
    key/value tile is loaded once for two query outer blocks.
- Evidence: random H200 smoke with `query_outer=4`, `key_outer=21`, 12 heads,
  and dim 128 measured the maintained baseline at about `3.43` ms/call. Direct
  seed-mask comparison measured about `4.67` ms/call. `BLOCK_A=2` measured
  about `7.12` ms/call. Both matched the PyTorch reference at the usual BF16
  tolerance, so the issue was performance, not correctness.
- Why not repeat directly: the extra compare pressure and larger per-program
  live state outweighed the saved intermediate or K/V load traffic.
- Rule: keep one query outer per merge program and keep the explicit `route`
  assignment for the maintained `hard_c4` path.

### H200 Hard-C4 RT Route Tile and V-Only Layout Tweaks

- Result: no speed improvement over the maintained delayed-V hard-C4 path.
- Changes tested on scai7/H200:
  - route tile sweep over `BLOCK_I in {16, 32, 64}` and
    `BLOCK_K in {8, 16, 32, 64}`;
  - keep compressed K in the original `[..., C, heads, dim]` intermediate
    layout but store compressed V as `[..., heads, C, dim]`, so route-kernel V
    reads are seed-contiguous for the fixed head.
- Evidence: random full-shape H200 sweep found the maintained
  `BLOCK_I=32, BLOCK_K=32` setting fastest at about `17.44` ms/full attention
  call. `BLOCK_K=16` was close at `17.52` ms, smaller/larger route tiles were
  slower, and `BLOCK_I=64, BLOCK_K=64` exceeded shared memory. The V-only
  route-contiguous layout matched the PyTorch reference but measured about
  `3.40` ms in smoke, slower than the delayed-V baseline at about `3.34` ms.
- Rule: keep route `BLOCK_I=32`, route `BLOCK_K=32`, and the original
  compressed-K/V intermediate layout.

### H200 Hard-C4 RT Route Loop Collapse

- Result: no speed improvement.
- Change tested: remove the inner static `key_start` loop in the route kernel
  because Wan has `ROWS=30` and the maintained `BLOCK_K=32`, so the loop only
  executes once for the current 480p shape.
- Evidence: random full-shape H200 smoke measured the collapsed-loop variant at
  about `3.2865` ms/call versus about `3.2811` ms/call for the maintained
  baseline, with the same BF16-level error against the PyTorch reference.
- Why not repeat directly: the compiler already handles the single iteration
  well enough; removing the loop did not reduce register pressure, memory
  traffic, or scheduling overhead in a measurable way.
- Rule: keep the explicit `key_start` static loop unless the route kernel is
  restructured for a genuinely different key-tile layout.

### Hard-C4 Chunk Workspace Reuse

- Result: no measurable speed improvement.
- Change tested: allocate the two-stage hard-C4 intermediate workspace
  (`merged_k`, `merged_v`, and routing bias) once per attention call and reuse
  it across same-sized query chunks, instead of letting each chunk call allocate
  its own workspace.
- Evidence: scai7/H200 synthetic full-shape chunk timing with
  `B=1, A=10, K_OUTER=21, ROWS=30, COLS=52, HEADS=12, D=128` measured
  `8.131` ms/call with internal allocation and `8.122` ms/call with explicit
  workspace reuse, only `1.001x` faster.
- Why not repeat directly: PyTorch's allocator/cache overhead is not the
  bottleneck. The dominant cost remains writing compressed K/V/bias to HBM and
  reading them back in the route kernel.
- Rule: do not add persistent workspace plumbing to the maintained hard-C4 path
  unless a later design also reduces intermediate HBM traffic.

### Hard-C4 Inline Route Masks

- Result: slower than the maintained explicit four-mask merge kernel.
- Change tested: remove the stored `route0`/`route1`/`route2`/`route3`
  boolean matrices in the merge kernel and compute `mask & (route == seed_id)`
  inside the static seed loop instead.
- Evidence: scai7/H200 smoke with
  `B=1, A=4, K_OUTER=21, ROWS=30, COLS=52, HEADS=12, D=128` measured
  `3.43` ms/call for the inline route-mask variant. The comparable maintained
  H200 smoke baseline was around `3.28` ms/call with the same BF16-level error
  against the PyTorch reference.
- Why not repeat directly: recomputing the per-seed route comparisons inside
  the loop did not reduce enough live state to offset the extra compare work.
- Rule: keep the explicit `route0`/`route1`/`route2`/`route3` masks in the
  maintained merge kernel unless the merge loop is restructured more deeply.

### Hard-C4 Int8 Route Matrix

- Result: slower than the maintained int32 route matrix.
- Change tested: store the merge-kernel `route` matrix as `tl.int8` instead of
  `tl.int32`, since the route id only needs values `0..3`.
- Evidence: scai7/H200 smoke compiled and matched the PyTorch reference at the
  usual BF16 tolerance. On the same A=10 synthetic full-chunk timing, the int8
  route variant measured `19.40` ms/call, while the maintained int32 baseline
  measured `18.07` ms/call.
- Why not repeat directly: the narrower logical dtype did not translate into a
  faster Triton implementation; conversion or packing overhead outweighed any
  register/storage benefit.
- Rule: keep `route` as `tl.int32` unless the assignment representation is
  redesigned more substantially.

### Hard-C4 Skip Empty-Seed K/V Stores and Loads

- Result: slower than the maintained unconditional compressed-K/V store/load
  path.
- Change tested: for empty no-force seed clusters, skip writing merged K/V in
  the merge kernel, load routing bias before merged K/V in the route kernel,
  and predicate route-kernel merged K/V loads on finite bias.
- Evidence: scai7/H200 smoke with
  `B=1, A=4, K_OUTER=21, ROWS=30, COLS=52, HEADS=12, D=128` matched the
  PyTorch reference at the usual BF16 tolerance. In a same-session comparison
  on the busy H200, skip-empty measured `8.69` ms/call while the maintained
  baseline measured `7.48` ms/call.
- Why not repeat directly: the extra dynamic bias-first load path and active
  mask logic in the route kernel outweighed the saved K/V traffic for empty
  clusters.
- Rule: keep unconditional compressed K/V stores and route loads unless the
  empty-token representation is redesigned at a coarser granularity.

### Hard-C4 Sqrt-Scale Q/K Storage

- Result: slower and numerically less aligned with the PyTorch reference.
- Change tested: mirror the MonarchRT Triton style by multiplying seed queries,
  keys, and second-stage queries by `sqrt(scale)`, storing scaled merged K, and
  removing the explicit `* scale` multiply from seed logits and route logits.
- Evidence: scai7/H200 small random smoke with
  `B=1, A=2, K_OUTER=2, ROWS=30, COLS=52, HEADS=2, D=128` measured Triton
  hard C4 at `0.0891` ms/call with rel-L2 `0.00687` versus the PyTorch
  reference. The full synthetic A=10 chunk timing measured `9.19` ms/call,
  slower than the maintained H200 baseline around `8.20` ms/call.
- Why not repeat directly: the removed scalar multiplies did not compensate
  for the extra Q/K scaling and changed fp16 intermediate rounding.
- Rule: keep the maintained unscaled merged-K layout and explicit route-logit
  scale unless a later implementation also changes intermediate precision or
  avoids storing merged K.

### Hard-C4 RT Assigned-Score Reuse

- Result: no speed improvement for the maintained RT-style H200 kernel.
- Change tested: while choosing the best seed, also keep an
  `assigned_scores` matrix containing the raw seed-key logit for the winning
  seed, then use that single matrix in the later masked cluster softmax instead
  of selecting among `s0`, `s1`, `s2`, and `s3` inside the static seed loop.
- Evidence: scai7/H200 small random smoke matched the PyTorch reference at the
  usual BF16/FP16 Triton tolerance. The full A=10 synthetic chunk timing
  measured `8.24` ms/call, not faster than the maintained baseline around
  `8.20` ms/call.
- Why not repeat directly: reducing visible source-level score selection did
  not reduce the compiled RT kernel's bottleneck enough to show a speedup.
- Rule: keep the explicit per-seed score matrices in the maintained RT-style
  kernel unless the merge loop is restructured more substantially. The older
  `bestlogit` result belongs to the retired simplemerge/token-major branch,
  not to the current RT-style kernel.

### Hard-C4 Head-Major Input K/V Rotation

- Result: slower than the maintained ordinary `[B, F, row, col, head, dim]`
  input K/V layout.
- Change tested: rotate K/V to `[B, head, F, row, col, dim]` before the hard-C4
  merge kernel so each fixed-head key row is read with contiguous key-column
  strides. The merge and route math, merged K/V layout, routing bias, and full
  compressed attention were otherwise unchanged.
- Evidence: scai7/H200 small random smoke matched the PyTorch reference at the
  usual BF16/FP16 Triton tolerance and looked faster on the tiny shape, but the
  realistic A=10 full synthetic chunk was slower. Pre-rotated kernel-only time
  was `8.35` ms/call; adding the K/V rotation made the wrapper `8.62` ms/call.
  The maintained baseline is around `8.20` ms/call for the same shape.
- Why not repeat directly: making K/V column reads contiguous did not offset
  the changed address arithmetic/scheduling, and explicit rotation adds extra
  HBM traffic.
- Rule: keep the standard input K/V layout for the maintained hard-C4 RT
  kernel unless a later design changes the full Q/K/V layout across the whole
  attention pipeline.

### Hard-C4 Wide K/V Merge Dot

- Result: much slower than separate `weights @ K` and `weights @ V` dots.
- Change tested: inside each seed cluster merge, build a logical wide
  `[K, V]` matrix and compute one `weights @ [K, V]` dot, then masked-store
  the first half to merged K and the second half to merged V.
- Evidence: scai7/H200 small random smoke compiled and matched the PyTorch
  reference at the usual tolerance but was not faster. The full A=10 synthetic
  chunk timing measured `10.18` ms/call, versus the maintained baseline around
  `8.20` ms/call.
- Why not repeat directly: the 256-wide output accumulator substantially
  increases register pressure and occupancy loss; reducing the number of dot
  calls did not help.
- Rule: keep separate merged-K and merged-V dot products unless the merge tile
  shape or accumulator layout is redesigned.

### Hard-C4 Static Key-Outer Route Unroll

- Result: much slower second-stage route kernel.
- Change tested: replace the route kernel's dynamic `tl.range(0, K_OUTER)`
  loop with `tl.static_range(0, 21)` because Wan 480p currently has
  `K_OUTER=21`.
- Evidence: scai7/H200 synthetic A=10 full chunk
  (`B=1, K_OUTER=21, ROWS=30, COLS=52, HEADS=12, D=128`) measured first-stage
  merge at `6.24` ms, but second-stage route increased to `6.25` ms and total
  two-stage time to `12.47` ms. The maintained dynamic-loop baseline route is
  about `1.94` ms and total is about `8.20` ms for the same shape.
- Why not repeat directly: fully unrolling 21 key-outer iterations bloats the
  route kernel and hurts instruction/cache/register behavior much more than it
  saves loop overhead.
- Rule: keep the route kernel's `tl.range(0, K_OUTER)` loop.

### Hard-C4 Stage-1 Online L32 Merge

- Result: much slower than the maintained one-pass 64-column merge.
- Change tested: replace the first-stage merge with an exact two-pass online
  design over two `BLOCK_L=32` key-column tiles. The first pass computes each
  seed's key-softmax log normalizer, and the second pass recomputes seed logits
  while online-accumulating per-cluster merged K/V and routing bias.
- Evidence: scai7/H200 synthetic A=10 full chunk
  (`B=1, K_OUTER=21, ROWS=30, COLS=52, HEADS=12, D=128`) measured the
  maintained stage-1 merge at `6.23` ms and the online L32 prototype at
  `26.91` ms. The temporary prototype also produced NaNs in the debug
  comparison, but the speed alone is already disqualifying.
- Why not repeat directly: avoiding the full 64-column score tile forces
  recomputing seed-key logits and serial online accumulation, which overwhelms
  any register-pressure benefit.
- Rule: keep the one-pass 64-column merge unless a future design avoids both
  score recomputation and large partial buffers.

### Hard-C4 Stage-1 Split-J Merge Tiles

- Result: slower than the maintained single `BLOCK_J=64` merge tile.
- Change tested: split query-column work into three exact-ish power-of-two
  tiles, `J=32 + 16 + 8`, with a `J_OFFSET` merge kernel. This reduces padded
  query-column work from 64 lanes to 56 lanes while keeping the same hard-C4
  assignment, merged K/V, routing bias, and second-stage full attention
  semantics.
- Evidence: scai7/H200 synthetic A=10 full chunk measured the maintained
  stage-1 merge at `6.22` ms and the split-J prototype at `8.40` ms.
- Why not repeat directly: smaller tiles reduce invalid-column arithmetic but
  add launches/program scheduling and less efficient small-M dot shapes; the
  tradeoff is worse on H200.
- Rule: keep `BLOCK_J=64` for stage-1 merge.

### Hard-C4 Stage-1 Log2-Domain Bias

- Result: slower than the maintained natural-log routing-bias representation.
- Change tested: convert seed logits into log2 units inside the merge kernel,
  compute/store routing bias in log2 units, and update the route kernel to
  subtract log2-domain bias directly. The math is equivalent up to fp rounding
  and primarily tries to remove repeated `log2(e)`/`ln(2)` scalar conversions
  from stage 1.
- Evidence: scai7/H200 smoke matched the PyTorch reference at the usual Triton
  tolerance, but the synthetic A=10 full chunk worsened stage-1 merge from
  about `6.24` ms to `7.12` ms and total two-stage time to `9.06` ms.
- Why not repeat directly: moving to log2 units did not reduce the dominant
  tensor-core merge work and appears to hurt compiled scheduling/rounding more
  than it saves scalar multiplies.
- Rule: keep natural-log routing bias in the maintained hard-C4 RT path.

### Hard-C4 RT Route Inner-Loop Removal

- Result: no useful speed improvement.
- Change tested: remove the route kernel's inner `tl.static_range(0, ROWS,
  BLOCK_K)` loop and compute the single Wan 480p key-row tile directly. This
  preserves the same compressed K/V layout, routing bias, and full compressed
  attention math.
- Evidence: scai7/H200 random-QKV smoke on 2026-08-13 matched the PyTorch
  reference at the usual BF16/Triton tolerance. The realistic
  `query_outer=10`, `key_outer=21`, `heads=12`, `dim=128` profile measured
  route at about `1.98` ms and full at about `7.40` ms, essentially tied with
  the maintained baseline (`1.99` ms route, `7.40` ms full).
- Why not repeat directly: the static loop already compiles away well enough;
  making the single iteration explicit changes readability but not the kernel
  bottleneck.
- Rule: keep the generic route loop unless the route tile shape changes.

### Hard-C4 RT Route-Q Cache Hint and Default K/V Dot Precision

- Result: no meaningful speed improvement.
- Change tested: add `eviction_policy="evict_last"` to the route kernel's Q
  tile load, and separately remove explicit `input_precision="ieee"` from the
  softmax-weight times K/V dot products while leaving seed/query-key logits in
  `ieee`. Both keep the hard-C4 formula, routing bias, and full compressed
  attention unchanged.
- Evidence: scai7/H200 random-QKV smoke on 2026-08-13 matched the PyTorch
  reference at the usual tolerance. The Q-load hint alone measured about
  `7.39` ms/call full on the realistic A=10 profile, the default-K/V-dot
  precision variant measured about `7.39` ms/call, and combining them measured
  about `7.39` ms/call. These are indistinguishable from the maintained
  baseline around `7.40` ms/call.
- Why not repeat directly: the route Q tile is already resident enough for the
  current scheduling, and the dot precision metadata is not the dominant
  tensor-core merge bottleneck.
- Rule: keep the maintained route Q load and explicit K/V dot precision unless
  a later kernel redesign changes the route tile or dot shapes.

### Hard-C4 RT `query_outer_chunk=11` With Conditional Int64 Offsets

- Result: stable but slower than the maintained `query_outer_chunk=10`.
- Change tested: cast only compressed K/V store/load offsets to int64 when
  `Q_OUTER > 10`, so the Wan 21-frame sequence could run as `11+10` chunks
  instead of `10+10+1`. The hard-C4 routing, merge, bias, and full compressed
  attention were otherwise unchanged.
- Evidence: scai7/H200 random-QKV A=11 profile on 2026-08-13 completed without
  illegal memory access, but measured about `9.33` ms/call full (`7.09` ms
  merge, `2.22` ms route). The maintained A=10 chunk is about `7.40` ms/call,
  so the saved small A=1 launch does not offset the larger int64-addressed
  chunk.
- Why not repeat directly: crossing the 32-bit compressed-K/V offset boundary
  makes the main chunk disproportionately slower.
- Rule: keep `query_outer_chunk=10` for hard-C4 unless the intermediate layout
  is redesigned so larger chunks avoid int64 compressed-K/V offsets.

### Hard-C4 RT Transposed Score Tile

- Result: much slower merge kernel.
- Change tested: compute first-stage seed-key logits as `[key_column,
  query_column] = K Q^T` instead of `[query_column, key_column] = Q K^T`,
  then reduce over the transposed axis and transpose the weights before
  `weights @ K/V`. The hard-C4 assignment, cluster softmax, routing bias, and
  full compressed attention were mathematically unchanged.
- Evidence: scai7/H200 random-QKV smoke on 2026-08-13 matched the PyTorch
  reference at the usual BF16/Triton tolerance, but the realistic A=10 profile
  slowed from about `7.40` ms/call full to about `22.99` ms/call. The merge
  stage alone rose to about `21.16` ms.
- Why not repeat directly: the transposed score/weight layout appears to make
  the merge dot path and register layout pathological.
- Rule: keep the maintained `[query_column, key_column]` score tile.

### Hard-C4 RT Merge-Store-Contiguous Intermediate Layout

- Result: faster on tiny smoke shapes, but slower on the realistic H200 chunk.
- Change tested: change only the physical compressed K/V and routing-bias
  workspace layout so merge stores for a fixed `(head, seed)` write
  `[query_column, dim]` contiguously. The route kernel was updated to read the
  same layout, so the hard-C4 merge, key-softmax assignment, routing bias, and
  full compressed attention were unchanged.
- Evidence: scai7/H200 small random smoke on 2026-08-13 matched the PyTorch
  reference and looked faster (`~0.04` ms vs the maintained `~0.10` ms on
  `query_outer=2`, `key_outer=2`, `heads=2`). The realistic A=10 profile
  reversed the result: full time increased to about `7.50` ms/call, with merge
  about `5.49` ms and route about `2.00` ms, versus the maintained baseline
  around `7.40` ms/call.
- Why not repeat directly: making merge writes contiguous hurts the larger
  real route/read and scheduling pattern enough to erase the tiny-shape win.
- Rule: keep the maintained compressed K/V layout for real Wan/H200 timing
  unless the route kernel is redesigned around the new physical layout.

### Hard-C4 RT Merge `maxnreg` Sweep

- Result: register caps do not improve the maintained H200 merge kernel.
- Change tested: keep the hard-C4 code unchanged and launch the merge kernel
  with Triton `maxnreg` values `32`, `48`, `64`, `80`, `96`, `128`, and `160`.
  The route kernel stayed at the default register allocation.
- Evidence: scai7/H200 random-QKV A=10 sweep on 2026-08-13 measured the
  default merge at about `5.43` ms and full merge+route at about `7.33` ms.
  `maxnreg` values up to `96` failed to compile or allocate registers; `128`
  slowed the merge to about `13.63` ms and `160` to about `9.48` ms.
- Why not repeat directly: the merge kernel needs roughly the default register
  budget for its WGMMA/tile live state; forcing fewer registers either fails or
  serializes/spills enough to dominate occupancy gains.
- Rule: do not cap `maxnreg` for the maintained hard-C4 merge kernel.

### Hard-C4 RT Static Tile-Origin Specialization

- Result: no speed improvement on the real Wan/H200 shape.
- Change tested: use `tl.constexpr` branches so the Wan 480p case
  (`COLS=52 < BLOCK_J=64`, `ROWS=30 < BLOCK_I=32`) compiles merge `j_start`
  and route `i_start` to zero, while retaining the generic program-id fallback
  for larger shapes. This changes only address arithmetic, not hard-C4 math.
- Evidence: scai7/H200 random-QKV smoke on 2026-08-13 matched the PyTorch
  reference. The realistic A=10 profile measured about `7.43` ms/call full,
  with route slightly faster but merge slower, versus the maintained baseline
  around `7.40` ms/call.
- Why not repeat directly: the program-id arithmetic is not a bottleneck, and
  the specialization perturbs merge scheduling enough to erase the tiny route
  gain.
- Rule: keep the straightforward program-id tile origins.

### Hard-C4 RT Bias-Only Log2 Storage

- Result: no meaningful speed improvement.
- Change tested: keep first-stage seed logits and cluster softmax in natural
  log units, but multiply the final routing bias by `log2(e)` before storing
  it. The route kernel then computes `qk * scale * log2(e) - bias_log2`
  instead of `(qk * scale - bias) * log2(e)`. This is algebraically the same
  full compressed attention up to floating-point rounding.
- Evidence: scai7/H200 random-QKV smoke on 2026-08-13 matched the PyTorch
  reference. The realistic A=10 profile measured about `7.40` ms/call full,
  with a tiny route improvement offset by a tiny merge slowdown, effectively
  tied with the maintained baseline.
- Why not repeat directly: moving only the bias unit conversion does not touch
  the dominant merge dot/store cost.
- Rule: keep routing bias in natural-log units in the maintained code.

### Hard-C4 RT Bias Expected-Value From Unnormalized Probs

- Result: slower than the maintained bias computation.
- Change tested: compute the routing-bias expected score as
  `sum(probs * scores) / sum(probs)` instead of
  `sum((probs / sum(probs)) * scores)`, while still using the normalized
  weights for `weights @ K/V`. This is algebraically equivalent and preserves
  key-softmax assignment, compressed K/V, routing bias, and full compressed
  attention up to rounding.
- Evidence: scai7/H200 random-QKV smoke on 2026-08-13 matched the PyTorch
  reference at the usual BF16/Triton tolerance. The realistic A=10 profile
  slowed to about `7.70` ms/call full, with merge about `5.70` ms, versus the
  maintained baseline around `7.40` ms/call.
- Why not repeat directly: the extra scalar division/dependency in the bias
  path hurts generated scheduling more than it saves.
- Rule: keep the maintained `weights * scores` expected-score computation.

### Hard-C4 RT Sequential Seed Assignment

- Result: slower than the maintained all-seed assignment tile.
- Change tested: compute the key-softmax assignment one seed at a time and
  reload each seed query inside the later merge loop. This removes the
  simultaneously live `s0..s3` score matrices and seed-query tiles, while
  preserving the same normalized route assignment, masked cluster softmax,
  routing bias, compressed K/V, and full compressed attention.
- Evidence: scai7/H200 smoke on 2026-08-13 matched the PyTorch reference at
  the usual BF16/Triton tolerance. The realistic A=10 stage profile measured
  merge at `5.81` ms, route at `1.97` ms, and full at `7.78` ms, slower than
  the maintained recompute-score version around `5.30` ms merge and `7.27`
  ms full.
- Why not repeat directly: the extra seed-query reloads and changed scheduling
  outweigh the lower visible live-score state.
- Rule: keep the maintained route-assignment structure and only recompute
  per-seed scores inside the merge loop.

## Current Best Direction

The best accepted direction so far is still:

- non-shared grid-local hard-route C4 compression,
- full compressed attention in the second stage,
- the Wan2.1-14B 720p specialization folds four seeds into a 64-row matrix
  operand and processes five 16-column query tiles in a runtime loop. Each
  program loads its 80-key K/V block once; the second stage remains full
  compressed attention. The production path on an exclusively idle H200
  averaged `122.771` ms/call over all 160 real-QKV calls versus `336.920`
  ms/call for same-run SDPA (`2.744x`). It retained rel-L2 `0.124030861` and
  MSE `0.004806307`, matching the previous exact Hard C4 path to the reported
  precision,
- local-with-ith seed selection for lower error than random-with-ith,
- adaptive greedy-cover-with-ith seed selection is the best precision-only
  seed strategy found so far, but is not the default fast path because it is
  much slower. Slurm job `368297` measured greedy-cover-with-ith at rel-L2
  `0.07632` and MSE `0.001804` versus local-with-ith at rel-L2 `0.09011` and
  MSE `0.002514` over all 90 self-attention calls. The larger three-prompt
  Slurm job `368827` confirmed the ranking over 270 calls: greedy-cover had
  rel-L2 `0.07967`, greedy-prob-cover `0.08065`, top-mean `0.08966`, and
  local-with-ith `0.09063`,
- prefilter12-greedy-with-ith is the best lower-complexity approximation found
  so far. It first ranks query rows by `q_i dot mean(K)` and keeps ith plus the
  best 11 candidates, then runs the same greedy-cover selection only inside
  that candidate set. Slurm job `369096` measured rel-L2 `0.08034`, close to
  exact greedy-cover's `0.07967`, while reducing analysis runtime from about
  `578.3` ms/call to `462.9` ms/call,
- inline local seed-row computation to avoid reading the seed table in the C4
  merge kernel,
- key-softmax assignment as the default hard C4 routing rule. Each seed's
  52 seed-key logits are normalized over the key dimension before each key
  chooses the seed with the largest normalized score; the later masked softmax
  merge still uses the original seed-key logits. Slurm job `369319` measured
  this full 270-call setting at rel-L2 `0.07237` and MSE `0.001534`, versus
  raw-dot assignment at rel-L2 `0.08927` and MSE `0.002335`,
- historical assigned-logit reuse (`bestlogit`) in the retired simplemerge
  branch. For the current RT-style H200 kernel, explicit per-seed score
  matrices are still the maintained faster path,
- corrected seed-3 best-logit update, which lowered dense rel-L2 from about
  0.09970 to about 0.09417 in Slurm job `366959`,
- specialized `bestlogit_simplemerge`, which matches corrected bestlogit output
  exactly while removing generic merge-kernel branches and improving real-QKV
  attention time from about 23.40 ms/call to about 23.04 ms/call in job
  `366959`,
- `ieee` logit precision for the simplemerge dot products, which job `366993`
  measured at 22.344 ms/call versus 22.456 ms/call for the previous default and
  22.351 ms/call for `bf16x3`, with identical dense rel-L2 `0.094166` and MSE
  `0.00259676`,
- no-force empty-seed handling as the default C4 assignment rule. Instead of
  forcing key `c` to seed `c`, empty seed clusters are left empty and masked in
  the full compressed-attention stage. Slurm job `367165` measured this at
  22.222 ms/call versus 22.890 ms/call for the old force-nonempty C4 path and
  22.757 ms/call for dense, while improving dense rel-L2 from `0.09759` to
  `0.09296` and MSE from `0.002715` to `0.002463`,
- token-major intermediate compressed-KV layout, which preserves the same
  full-attention math but makes merge stores more contiguous,
- standard K/V block layout for the current token-major `m32_n32` path, avoiding
  the older head-major experiment branch,
- historical route/merge Triton launch `num_stages=2` on H200 for the older
  RT-style hard-C4 backend; scai7 one-prompt real-Wan-QKV measured about `18.39` ms/call
  versus `18.54` ms/call for the previous `num_stages=3`, with unchanged rel-L2
  `0.073345` and MSE `0.0016324`,
- delayed V load in the current RT-style hard-C4 merge kernel. K is loaded
  before seed routing, while V is loaded only after the route masks are known.
  This preserves output exactly and reduced the 20-prompt H200 real-Wan-QKV
  attention benchmark from `18.45` to `17.91` ms/call, with unchanged rel-L2
  `0.07924` and MSE `0.001880`,
- normalized-route comparison in the merge kernel: keep original `s0..s3`
  logits for the later masked softmax merge, but compare `s - logZ` directly
  when assigning each key to a seed instead of materializing four normalized
  logit matrices. This preserves output and reduced the same 2,400-call H200
  benchmark further to `17.57` ms/call with unchanged rel-L2 `0.07924` and MSE
  `0.001880`,
- Triton dot `input_precision="ieee"` for the maintained H200 hard-C4 RT path.
  Older real-Wan-QKV timing had shown a tiny `tf32` advantage, but a later
  scai7 H200 GPU0 retest on 2026-08-13 measured `tf32` at about `7.737`
  ms/call on the A=10 random stage profile, essentially tied with and slightly
  slower than the maintained `ieee` profile at about `7.731` ms/call. The
  smoke error was unchanged, so this is a speed/config choice rather than an
  accuracy issue,
- intermediate merged-K/V and routing-bias stores use
  `eviction_policy="evict_first"`. This keeps the same hard-C4 output while
  reducing the scai7 H200 GPU0 A=10 random stage profile from about `7.73`
  ms/call full to about `7.39` ms/call full. `.cg` and `.cs` store cache
  modifiers did not beat this setting,
- historical recompute-each-seed-score implementation instead of keeping all
  seed-score matrices live through the masked cluster softmax. This preserves
  output at the same BF16/Triton tolerance and improved the H200 A=10 random
  profile to about `5.30` ms merge and `7.27` ms full. The 20-prompt real Wan
  QKV benchmark measured hard C4 at `15.90` ms/call versus dense SDPA at
  `18.56` ms/call with rel-L2 `0.07926` and MSE `0.001882`. The corresponding
  20-video generation run measured `8.15` s/video for hard C4 versus `14.40`
  s/video for SDPA,
- restored explicit 2D `s0..s3` score tiles as the current maintained fast
  path. Unlike the historical bullet above, this version reuses the four
  assignment logits for the masked cluster merge and launches the merge with
  `BLOCK_J=64`, four warps, `num_stages=1`, and IEEE dot precision. The full
  accepted result is recorded in "Restored Hard-C4 2D Score Tiles" below,
- mask only key columns in the stage-1 assignment softmax. The padded
  query-column lanes are never stored because later route and store masks still
  include the true query-column mask, so this preserves all real `52` columns
  while avoiding an unnecessary all-`-inf` softmax path. scai7/H200 measured
  the A=10 random profile at `5.17` ms merge and `7.14` ms full, and the
  20-prompt real Wan QKV benchmark at `15.65` ms/call for hard C4 versus
  `18.60` ms/call for dense SDPA, with rel-L2 `0.07926` and MSE `0.001882`,
- reusing the already-computed `s0..s3` seed-key logits in the maintained
  hard-C4 RT merge kernel is correct, but it is not a meaningful new speed
  direction by itself. The 2026-08-13 scai7/H200 real-Wan-QKV short benchmark
  measured `15.42` ms/call with rel-L2 `0.07951`, essentially matching the
  previous maintained hard-C4 timing/error. Keep the clearer reuse form, but
  do not expect it to solve the bottleneck; the compiler was likely already
  eliminating most of the apparent duplicate dot work or the bottleneck lies
  elsewhere,
- `query_outer_chunk=10`, which was the fastest confirmed safe chunk size in
  `analysis/hard_c4_nohead_chunk_confirm_20260808`,
- H200 merge-stage profiling shows the current hard-C4 RT bottleneck is not
  only intermediate HBM stores. For the A=10 synthetic full chunk
  (`B=1, K_OUTER=21, ROWS=30, COLS=52, HEADS=12, D=128`), scai7 measured
  full merge at `6.23` ms: seed dot `1.13` ms (`18.1%`), key-softmax routing
  `0.56` ms (`9.0%`), cluster softmax plus K/V merge dot `2.78` ms (`44.7%`),
  and intermediate K/V/bias stores `1.76` ms (`28.2%`). The no-store checksum
  version was `4.47` ms, so eliminating stores alone cannot reach the desired
  attention speed target,
- direct-route Triton kernels to avoid unnecessary reshape/copy overhead,
- only two maintained implementation paths: PyTorch `sdpa` as the readable
  correctness reference and `hard_c4` as the optimized Triton path. Old generic
  `triton`, `triton_c4`, `triton_c4_j16`, and long experiment-suffix aliases
  were removed from active code so they are not accidentally reused.
- expose this path only as `hard_c4`.

This direction improves the old hard C4 implementation substantially, but it
has not yet achieved a clear 2x attention speedup over dense attention while
preserving the desired error level.

## Row/Column Group Layout Exploration

- `qc1/kr2/C6` merges two adjacent key rows (104 keys) into six compressed
  tokens for each query column. Three-prompt real-Wan-QKV screening showed
  better dense error than current hard C4 (rel-L2 `0.06638`, MSE `0.001283`).
  The original Triton stage profile was not viable: A=10 random QKV measured
  `32.78` ms merge and `34.20` ms full with `BLOCK_J=64`. A later H200 merge
  sweep found `BLOCK_J=16`, which improved the A=10 stage profile to
  `11.20` ms merge, `1.61` ms route, and `12.86` ms full. However, the
  three-prompt/all-call real-QKV benchmark
  `hard_kr2_c6_tuned_realqkv_full3_20260813` still measured `27.03` ms/call,
  rel-L2 `0.06635`, and MSE `0.001281`. This is a quality-oriented layout, not
  a speed path, unless the first-stage merge is redesigned beyond tile tuning.
- `qc1/kr2/C4` reduces the second-stage compressed tokens by grouping two key
  rows while keeping four seeds. Tuning `BLOCK_J=16` improved the A=10 random
  profile to `7.16` ms merge, `1.01` ms route, and `8.22` ms full, but
  this is still slower than current hard C4. A first-six-call real-Wan-QKV
  benchmark measured `17.54` ms/call and rel-L2 `0.09034`, versus current hard
  C4 at `15.94` ms/call and rel-L2 `0.07951`. Do not replace the default hard
  C4 path with this layout unless a new merge kernel removes the large stage-1
  working-set cost. A later three-prompt/all-call H200 benchmark
  `hard_kr2_c4_realqkv_full3_20260813` confirmed the same conclusion:
  `17.46` ms/call, rel-L2 `0.08585`, and MSE `0.002146`, slower and less
  accurate than hard C4 even though it is much better than C2 at the same
  compressed-token density.
- `qc4/kr1/C8` keeps each key block at one image row (52 keys) but groups four
  query columns and uses eight seeds. This makes stage-1 merge faster than
  current hard C4 on A=10 random QKV (`4.57` ms merge versus `5.14` ms), but it
  doubles the second-stage compressed token count and makes route much slower.
  Route tuning found `BLOCK_K=32`, `BLOCK_N=256`, `route_num_warps=8` as the
  best tested setting, improving the A=10 random full profile from `8.96` ms
  to `8.44` ms. A first-six-call real-Wan-QKV benchmark measured better error
  (rel-L2 `0.07449`, MSE `0.000682`) but still no useful speedup: `18.04`
  ms/call, about equal to SDPA at `18.10` ms/call and slower than hard C4 in
  the same run (`17.23` ms/call). This is a precision-oriented layout, not a
  speed candidate.
- `qc2/kr1/C6` is also not a speed candidate. It groups two query columns and
  uses six seeds, which gives slightly better dense error than current hard C4
  but does not reduce the route workload enough. Route tuning again selected
  `BLOCK_K=32`, `BLOCK_N=256`, `route_num_warps=8`, giving an A=10 random
  profile of `6.59` ms merge, `3.94` ms route, and `10.50` ms full. A
  first-six-call real-Wan-QKV benchmark measured rel-L2 `0.07631` and MSE
  `0.000716`, but speed was still `21.96` ms/call, slower than SDPA (`18.10`
  ms/call) and hard C4 (`17.23` ms/call) in the same run. The key lesson is
  that grouping query columns shifts cost from merge into full compressed
  attention; even when error improves, the route tile count dominates.
- T-axis layouts are not promising for speed at comparable density. In the
  first-six-call real-Wan-QKV axis screen, using time as the key inner axis
  (`qh/kt`) gave rel-L2 `0.08439` at C2/density `0.09524`, already worse than
  current hard C4's rel-L2 `0.07951` while using more compressed tokens. Higher
  C improved error (`0.07085` at C3 and `0.06246` at C4), but densities rose to
  `0.14286` and `0.19048`, so route cost should be much higher. Using time as
  the query inner axis with width keys (`qt/kw`) was worse at equal density
  (C4 rel-L2 `0.09236`) and only became good at high density (C8 rel-L2
  `0.05709`, density `0.15385`). The T-axis route is therefore a precision
  knob, not a credible path to the speed target unless the second-stage token
  count is reduced by a different mechanism.
- The full3 H200 follow-up `axis_low_density_full3_20260813` confirmed this
  at low density. `qh_kt_c1` used time as the key inner axis with density
  `0.04762` and rel-L2 `0.15571`, worse than hard C2's rel-L2 `0.11824` even
  though hard C2 has lower density `0.03846`. `qh_kt_c2` raised density to
  `0.09524` but still had rel-L2 `0.10335`, worse than hard C3 at density
  `0.05769`. `qw_kt_c1/C2` and `qt_kw_c2` were worse still. Do not write a
  low-density T-axis Triton kernel for the current speed-quality target.
- `qc1/kr1/C3` is a speed-quality tradeoff, not a default replacement for hard
  C4 yet. The C3 Triton kernel uses the same `30q x 52k` layout but keeps
  three local seeds instead of four. The A=10 random-QKV profile on scai7/H200
  measured `3.78` ms merge, `1.63` ms route, and `5.44` ms full, faster than
  hard C4. A three-prompt/all-call real-Wan-QKV benchmark measured `11.83`
  ms/call for C3 versus `15.04` ms/call for flash and `15.65` ms/call for hard
  C4, but dense error worsened to rel-L2 `0.08771` and MSE `0.002239`, compared
  with hard C4 rel-L2 `0.07463` and MSE `0.001621` in the same run. Keep this
  as an experimental speed candidate; do not make it the default unless video
  samples show the quality loss is acceptable.
- `qc1/kr1/C2` and C1/MonarchRT-style compression define the current lower-C
  speed-quality curve. On scai7/H200, the three-prompt/all-call real-Wan-QKV
  benchmark measured C2 at `7.47` ms/call (`1.99x` faster than flash) with
  rel-L2 `0.11824` and MSE `0.004069`; C1/MonarchRT at `4.14` ms/call (`3.59x`
  faster than flash) with rel-L2 `0.17439` and MSE `0.008853`. This reaches
  the speed target only at a quality point that is much worse than hard C4.
  Use these as speed-quality anchors, not as default replacements.
- Changing the single C1 seed row does not fix the C1 quality gap. The H200
  screen `c1_seed_modes_screen_20260813` compared ith, center, top-mean, and
  top-lse seed choices. On the first eight real Wan self-attention calls, ith
  was best among cheap choices with rel-L2 `0.17263`; center and top-mean
  variants were worse. On one full prompt over all 90 calls, top-lse-all
  slightly reduced rel-L2 versus ith (`0.16200` versus `0.17164`) but remained
  far worse than C3/C4 and required a very expensive seed-selection path
  (`376.08` ms/call in PyTorch analysis versus `94.67` ms/call for ith).
  Treat C1 as a speed anchor, not a quality-preserving method.
- Most multi-column/multi-row groupings are not clear speed paths. The
  first-six screen `rowcol_multigroup_screen_20260813` made `qc2_kr2_c8` look
  worse than hard C4, while the follow-up full3 PyTorch run
  `rowcol_qc2kr2_c8_c10_full3_20260813` showed a small quality advantage:
  rel-L2 `0.07253` and MSE `0.001531`, slightly better than hard C4 at the same
  compressed-token density. A tuned H200 Triton prototype then confirmed the
  speed issue: `hard_qc2_kr2_c8_realqkv_full3_20260813` measured `23.55`
  ms/call versus dense SDPA `18.02` and hard C4 `15.40`, despite similar
  output to the PyTorch reference. Stage profiling measured `9.51` ms merge
  out of `11.16` ms for an A=10 chunk, so the larger `60q x 104k` merge
  working set dominates. Treat `qc2_kr2_c8` as a quality-oriented data point,
  not a path to the 3x speed target unless the first-stage merge is redesigned.
  Higher-density `qc2_kr2_c10` and `qc2_kr2_c12` remain quality knobs, not
  speed candidates.
- Wider key-row blocks show the same quality/cost tradeoff more sharply. The
  full3 H200 PyTorch screen `rowcol_wider_key_lowdensity_full3_20260813`
  measured `qc1_kr3_c6` at the same compressed-token density as C2 with rel-L2
  `0.07393`, close to hard C4, and `qc1_kr5_c10` at the same density with
  rel-L2 `0.06135`, better than hard C4. But the static cost model
  `wider_key_cost_20260813` estimates merge MAC at `1.5x` hard C4 for
  `qc1_kr3_c6` and `2.5x` hard C4 for `qc1_kr5_c10`. Treat these as strong
  quality-oriented evidence that more key rows help compression, not as
  current 3x speed candidates unless the first-stage merge is redesigned.
- The tuned `qc1/kr3/C5` Triton prototype confirms this guardrail for a
  lower-density setting. `hard_kr3_c5_tuned_realqkv_full3_20260813` measured
  rel-L2 `0.08240` and MSE `0.001977`, better than hard C3, at density
  `0.03205`, lower than C2. However, the H200 kernel still took `27.84`
  ms/call versus FlashAttention `14.56` ms/call and hard C3 `11.57` ms/call.
  Stage profiling showed A=21 full time `27.47` ms with merge about `90%` of
  the kernel. The need to pad the 156-key block to `BLOCK_L=256` makes the
  merge stage too expensive in this direct two-stage design.
- Splitting a 52-column key row into smaller key-column groups is also not a
  good same-density replacement for hard C4. The H200 screen
  `keycol_layout_screen_20260813` tested `kg2_c2`, where each key row is split
  into two `26`-key groups with C2 per group, and `kg4_c1`, where each row is
  split into four `13`-key groups with C1 per group. Both have the same
  compressed-token density as hard C4 (`4/52 = 0.07692`). On the full
  one-prompt / 120-call run, hard C4 had rel-L2 `0.07333` and MSE `0.001633`,
  while `kg2_c2` worsened to rel-L2 `0.10980` and MSE `0.003661`. The first-six
  call screen showed `kg4_c1` was worse still, rel-L2 `0.12889`. Increasing to
  `kg2_c3` nearly recovered hard-C4 error (rel-L2 `0.07737`) but raised density
  to `0.11538`, so the second-stage full-attention token count increases.
  Do not write a tuned key-column-group kernel for the same-density C2/C1
  variants unless a different seed or merge objective changes this quality
  result.
- Same-axis query/key blocks are not promising. The H200 first-six real-QKV
  screen `same_axis_layout_screen_20260813` tested `qw_kw_c4`, `qw_kw_c3`,
  `qh_kh_c3`, `qh_kh_c2`, `qt_kt_c3`, and `qt_kt_c2`, where query and key use
  the same inner axis W, H, or T. All were far worse than hard C4: rel-L2 ranged
  from `0.74037` to `0.81062`, while hard C4 in the same QKV calls stayed at
  rel-L2 `0.07951`. Even same-density `qw_kw_c4` (`4/52 = 0.07692`) had rel-L2
  `0.76274` and MSE `0.07149`. Do not write same-axis Triton kernels for the
  current speed-quality target.
- Local seed pattern changes are not enough to recover low-C quality. The
  screen `local_seed_pattern_screen_20260813` tested balanced, left, right, and
  wide local seed choices for C2/C3/C4 on the same `30q x 52k` layout. C2 did
  not improve (`0.10032` rel-L2 for balanced/left; `0.10276` for right/wide).
  C3-left improved only slightly (`0.08789` versus balanced `0.08824`), and
  C4-balanced stayed best (`0.07955`, versus left `0.07976`, wide `0.08041`,
  right `0.08219`). Do not spend time writing dedicated Triton kernels for
  these fixed local seed patterns unless a later full-video result contradicts
  the screen.
- Per-call/per-layer adaptive C is only a smooth speed-quality tradeoff. The
  profile `percall_c_sweep_full1_20260813` measured one prompt over all 120
  self-attention calls. Fixed C1/C2/C3/C4 averaged `4.23`/`7.58`/`11.84`/`15.42`
  ms/call with mean rel-L2 `0.183`/`0.122`/`0.088`/`0.075`. A fixed-layer oracle
  that chooses the smallest C within `1.2x` of each layer's average C4 error
  selected 2 C2 layers, 26 C3 layers, and 2 C4 layers, yielding `11.80` ms/call
  and mean rel-L2 `0.0868`, essentially a C3-like tradeoff. A looser `1.5x`
  schedule reached `10.32` ms/call with mean rel-L2 `0.0937`. This may be useful
  as a tunable knob, but it does not approach the `3x` speed target without
  accepting C1/C2-level error.
- A stricter per-call oracle confirms the same limit. Using
  `analysis/optimize_c_schedule.py` on `percall_c_sweep_full1.json`, the best
  MSE-minimizing schedule under a `3x`-vs-FlashAttention attention-time budget
  selected 93 C1 calls and 27 C2 calls, with no C3/C4 calls. It reached exactly
  `3.00x` versus FlashAttention but still had aggregate rel-L2 `0.15326` and
  mean MSE `0.00713`, much closer to fixed C1 (`0.18347`, `0.00952`) than to C3
  (`0.08814`, `0.00227`) or C4 (`0.07506`, `0.00163`). Reaching mean MSE around
  `0.0036` already drops to about `2x` versus FlashAttention. Treat dynamic C
  as a controllable quality/speed knob, not a path to the requested `3x` speed
  with C3/C4-level error.
- C1 seed changes do not rescue the `3x` target. C1/MonarchRT is fast enough,
  but the full-one-prompt C1 seed screen showed ith rel-L2 `0.17164`;
  `top_lse_all` improved this only to `0.16200` while requiring an expensive
  adaptive seed path, and `top_mean_local3` worsened to `0.20352`. Since the
  oracle dynamic-C result at exact `3x` is still rel-L2 `0.15326`, further C1
  seed-only tweaks are unlikely to reach C2/C3-quality error without giving up
  the speed target. Prefer new low-C block objectives over more single-seed
  selection sweeps.
- Low-C adaptive seed selection improves error but does not by itself solve the
  speed target. The H200 real-Wan-QKV screen
  `lowc_seed_modes_screen_20260813` used the same `30q x 52k` hard-route math
  and full compressed attention, but changed the seed rows for C2/C3/C4. On one
  prompt over all 90 self-attention calls, local C2/C3/C4 had rel-L2
  `0.11536`/`0.08275`/`0.07020`. `prefilter12_greedy` improved them to
  `0.10173`/`0.07475`/`0.06028`, respectively. This makes C3
  prefilter-greedy close to local C4 quality, so it is a possible
  quality/speed tradeoff if a cheap seed-selection kernel is later written.
  C2 remains much worse than C4, so it is still not a good 3x-quality
  candidate.
- For the C3 adaptive-seed direction, `prefilter8_greedy` is a better
  engineering target than `prefilter12_greedy`. The H200 full-one-prompt screen
  `lowc_prefilter_stage_profile_20260813` measured C3 local at rel-L2
  `0.08275`, C3 prefilter8-greedy at `0.07576`, and C3 prefilter12-greedy at
  `0.07475`. The PyTorch profiling path measured stage subtotals of `37.15s`
  for prefilter8 versus `43.80s` for prefilter12 over the same 90 attention
  calls. Candidate-logit computation and greedy selection dominate the adaptive
  overhead; for prefilter8 they were `25.5%` and `22.6%` of measured stage time.
  If a Triton adaptive-seed kernel is written, start with C3 prefilter8-greedy
  and optimize candidate logits plus greedy selection first.
- T-axis low-density layouts are not a speed candidate. The screen
  `axis_low_density_screen_20260813` compared all mixed axis choices near the
  C2/C3/C4 compressed densities. The current H-query/W-key C4 layout
  (`qh_kw_c4`) measured rel-L2 `0.07955` at density `0.07692`. The best T-axis
  result, `qh_kt_c3`, improved rel-L2 to `0.07085` only by raising density to
  `0.14286`, nearly twice hard C4. At comparable density, T-axis layouts were
  worse: `qt_kw_c4` rel-L2 `0.09236` at density `0.07692`, and `qh_kt_c2`
  rel-L2 `0.08439` at density `0.09524`. Do not write a T-axis Triton kernel
  for the current `3x` speed target unless the goal changes to a higher-density
  quality-oriented method.
- Hard-C4 logical IO model: intermediate K/V/bias IO is the largest theoretical
  remaining IO target, but eliminating it is not automatically a speed win.
  `analysis/estimate_attention_io.py` models one Wan 480p attention call with
  `outer=21`, `rows=30`, `cols=52`, `heads=12`, `dim=128`, and C4. With
  FlashAttention block-M `128`, the logical FlashAttention traffic is about
  `51.73` GB. The maintained two-stage hard C4 path is about `46.96` GB, of
  which intermediate compressed K/V/bias store+load is about `34.08` GB
  (`72.6%` of hard-C4 logical traffic). A perfect fused design that removes all
  intermediate store+load would make hard-C4 logical traffic about `0.249x` of
  FlashAttention logical traffic, but this is only an optimistic lower bound.
  A naive per-query-column fusion can lose the current merge stage's sharing
  across all 52 query columns and reload original K/V much more often. Pursue
  fusion only if the tile design keeps enough query-column and query-row work
  together to consume compressed K/V before writing it out. The same estimator
  gives no-intermediate traffic of `4.416x`, `1.230x`, `0.739x`, `0.494x`,
  `0.331x`, and `0.249x` versus FlashAttention for query-column tiles of
  `1`, `4`, `8`, `16`, `32`, and `52`. The same script's approximate live-state
  model explains why this is hard: `I_TILE=8,J_TILE=16` already needs about
  `152` KiB of live tensors, `I_TILE=16,J_TILE=16` about `249` KiB, and
  `I_TILE=30,J_TILE=16` about `419` KiB. Large column tiles reduce HBM traffic
  but create accumulator/register pressure; small column tiles keep live state
  manageable but reload K/V too often. With the corrected live-state model,
  one-pass `I_TILE=30,J_TILE=8` is the only default candidate that is both below
  FlashAttention logical IO and within the rough H200 shared-memory budget.
- One-pass hard-C4 with `J_TILE=1` is only a correctness prototype, not a speed
  path. Commit `c2aeee7` added `attention/hard_c4_onepass_triton.py`, which
  computes merge and second-stage attention in one kernel without writing
  `merged_k/v/bias`. It matches the no-intermediate PyTorch reference at the
  same numerical level as the two-stage Triton path, but it handles only one
  query column per program. On H200 synthetic QKV with shape
  `[1,2,30,52,1,128]`, `hard_c4_onepass_j1` measured `0.681` ms versus
  `0.042` ms for the two-stage kernel. The reason is exactly the IO-model
  warning: `J_TILE=1` reloads original K/V once per query column, losing the
  current merge stage's sharing across all 52 columns. Keep it as an equation
  and dataflow smoke test only; the real fused target remains
  `I_TILE=30,J_TILE=8`.
- A direct rank-3 `J_TILE=8` one-pass hard-C4 prototype also is not a speed
  path in its current form. Commit `65e1678` added `hard_c4_onepass_j8`, which
  keeps the natural `[query_row, query_column, dim]` accumulator inside one
  Triton program so K/V can be loaded once per eight query columns. It compiles
  and matches the streaming PyTorch reference at the same numerical level as
  the two-stage kernel, but the live tensor pressure is too high. On H200
  synthetic QKV `[1,2,30,52,1,128]`, the two-stage kernel measured `0.053` ms,
  `hard_c4_onepass_j1` measured `0.679` ms, and `hard_c4_onepass_j8` with
  `I_TILE=32` measured `2.871` ms. An `I_TILE` sweep confirmed the live-state
  diagnosis: `I_TILE=1/2/4/8` measured `0.871/0.461/0.463/0.624` ms. Smaller
  row tiles reduce the accumulator pressure, but they still reload K/V too
  often and remain much slower than the two-stage path. Do not treat this
  kernel as the fused solution; a usable one-pass design needs less live state
  without increasing K/V reloads enough to erase the intermediate-IO savings.
  A real Wan-QKV smoke confirmed this guardrail more strongly. Before the
  empty-token online-softmax fix, `hard_c4_onepass_j8` measured `315.08`
  ms/call on the first three attention calls and produced NaNs. After fixing
  empty compressed tokens, it produced the expected hard-C4-level error
  (rel-L2 `0.09311`, MSE `0.001100`), but was still `324.71` ms/call while
  hard C4 measured `15.35` ms/call and FlashAttention measured `14.53` ms/call.
  The `j8` prototype is therefore not exposed through the Wan dispatch path;
  keep it only as research code for debugging future fused designs.
- D-tiled two-pass no-merged-KV is not currently a good fused-kernel shortcut.
  It initially looked attractive because it stores only second-stage row
  max/sum and avoids a full `I x J x D` output accumulator. The corrected model
  in `analysis/estimate_attention_io.py` accounts for the fact that each output
  D chunk must recompute full-D route logits unless another intermediate is
  stored. Small `D_TILE` therefore repeats seed-Q/K traffic many times, while
  `D_TILE=128` avoids repeated route IO but exceeds the rough H200 live-state
  budget. In the default sweep there is no two-pass tile that is both below
  FlashAttention logical IO and within the 228 KiB budget. Do not pursue
  D-tiled two-pass hard C4 unless the design can reuse route logits without
  materializing another large intermediate.
- 2D key-patch grouping is not a better speed-quality frontier point. The H200
  screen `keypatch_layout_screen_20260813` replaced one full key row with 2D
  key patches such as `2x26`, `2x13`, `3x13`, `5x13`, and `5x26`, while keeping
  one-frame/one-column query blocks and the full compressed second-stage
  attention. On the first eight real Wan self-attention calls, same-density
  `kp2x26_c4` had rel-L2 `0.09306` versus hard C4 `0.08765`; `kp2x13_c2`
  rel-L2 `0.27752`; and `kp3x13_c3` rel-L2 `0.24694`. `kp2x26_c6` improved to
  rel-L2 `0.07548`, but only by raising compressed-token density to `0.11538`.
  Do not tune kernels for `kp2x13`, `kp3x13`, `kp5x13`, or same-density
  `kp2x26_c4`; treat `kp2x26_c6` as a quality knob only, not a `3x` speed
  candidate.
- 2D query-patch plus 2D key-patch grouping is much worse than the original
  row/column split. The H200 first-eight-call screen
  `patchpair_screen_20260813` tested `qp15x2_kp2x26_c2/c3/c4/c6`, where each
  block still has `30` query tokens and `52` key tokens but both sides are
  spatial patches instead of a full query column and a full key row. At the
  hard-C4 density `0.07692`, `qp15x2_kp2x26_c4` measured rel-L2 `0.28592` and
  MSE `0.01054`, versus hard C4 rel-L2 `0.08765` and MSE `0.000991` in the
  same screen. C6 raised density to `0.11538` but still measured rel-L2
  `0.17600`. This suggests the current method needs the full-column/full-row
  cross structure; do not write a Triton kernel for patch-pair layouts unless a
  different seed or objective first recovers quality.
- Query-patch-only grouping does not beat the original full-column query block.
  The H200 first-eight-call screen `querypatch_only_screen_20260813` used the
  same patch-pair screening code. Its sanity check `qp30x1_kp1x52_c4` matched
  the original hard-C4 quality, rel-L2 `0.08769` and MSE `0.000992`, confirming
  the implementation. With query patches changed to `15x2` and key blocks kept
  as original `1x52` rows, `qp15x2_kp1x52_c4` worsened to rel-L2 `0.09955` at
  the same density `0.07692`. `qp15x2_kp1x52_c6` reached rel-L2 `0.07207`, but
  at higher density `0.11538`. Do not tune a query-patch-only kernel unless a
  new design improves C4-density quality beyond the full-column query layout.
- Increasing hard-C4 `query_outer_chunk` is not a 3x speed path. The H200
  sweep `hard_c4_query_chunk_sweep_no21_20260813` measured the first 12 real Wan
  self-attention calls with chunks `1`, `3`, `7`, and `10`. The best setting was
  `chunk=7` at `16.08` ms/call, only `1.09x` faster than `chunk=1` at `17.57`
  ms/call and still slower than the same-run FlashAttention baseline at `14.97`
  ms/call. All chunk sizes had identical dense error, rel-L2 `0.11228`. This
  rules out Python chunking/kernel launch count as the primary bottleneck for
  the current implementation. `chunk=21` reached warmup then triggered a Triton
  illegal memory access; do not use full-query-outer chunking until that kernel
  issue is debugged.
- A-tiled hard-C4 merge is not a useful way to reuse original K/V. The tested
  kernel grouped adjacent query-outer blocks inside one merge program, so each
  key/value row could be loaded once and reused across multiple query frames,
  while keeping the exact hard-C4 assignment, key-softmax merge, routing bias,
  and full compressed second-stage attention. On scai7/H200 synthetic A=10
  stage profiling (`hard_c4_atile_profile_20260813`), the maintained baseline
  measured `5.21` ms merge, `1.94` ms route, and `7.33` ms full. A-tile
  variants were much slower despite matching the baseline output closely
  (rel-L2 about `6.1e-5`): `A_TILE=2` measured `15.45` ms merge and `21.04` ms
  full, `A_TILE=5` measured `24.02` ms merge and `18.53` ms full, and
  `A_TILE=10` measured `35.81` ms merge and `39.63` ms full. The K/V load
  reuse is outweighed by reduced program parallelism and much higher
  register/scheduling pressure. Do not pursue query-outer tiling inside the
  merge kernel unless the tile is restructured around a different physical
  layout.

### Temporal-Distance Adaptive C

- Result: not a new low-density speed-quality frontier point.
- Change tested: keep the original full query-column/full key-row grid and full
  compressed second-stage attention, but use more seeds for temporally nearby
  `(query frame, key frame)` pairs and fewer seeds for distant pairs.
- Evidence: the H200 first-eight-call screen
  `temporal_adaptive_c_screen_20260813` measured `ta_r3c4_far1` at density
  `0.03689`, rel-L2 `0.11670`, and MSE `0.001756`; `ta_r4c4_far1` at density
  `0.04134`, rel-L2 `0.11232`, and MSE `0.001627`; and the denser
  `ta_r1c4_far2` at density `0.04378`, rel-L2 `0.10727`, and MSE `0.001484`.
  These are smooth improvements over C1 but do not beat the matched C2
  first-eight reference around rel-L2 `0.10032` at density `0.03846`.
- Rule: do not write a temporal-adaptive-C Triton kernel unless a new temporal
  scoring rule first improves quality at C2-like density.

### Row-Major Contiguous 64/128-Key Blocks

- Result: useful quality signal at high C, but not a low-C speed-quality
  solution yet.
- Change tested: keep one frame/one query column as the query block, but split
  each frame's K/V tokens in row-major order into contiguous chunks such as
  64 or 128 tokens. Each key chunk is compressed with key-softmax assignment
  and full compressed second-stage attention.
- Evidence: the H200 first-eight-call screen `flat_keyblocks_screen_20260813`
  measured `flat64_c4/c5/c6` at rel-L2 `0.09435`/`0.08450`/`0.07711`, while
  same-call hard C4 was `0.08764`. The low-C 128-token screen
  `flat128_lowc_screen_20260813` measured `flat128_c4/c5/c6/c7` at rel-L2
  `0.14213`/`0.09707`/`0.08864`/`0.08233`; these do not clearly beat hard C4
  until C is increased.
- Strong quality signal: the full3/all-call run `flat128_keyblocks_full3_20260813`
  measured `flat128_c8` at density `0.06667`, rel-L2 `0.05938`, and MSE
  `0.001026`; `flat128_c10` at density `0.08333`, rel-L2 `0.05200`, and MSE
  `0.000787`; and `flat128_c12` at density `0.10000`, rel-L2 `0.04671`, and
  MSE `0.000635`. These are better than hard C4's full3 rel-L2 `0.07463`, but
  PyTorch analysis time was hundreds of ms/call.
- Why not tune naively: a direct two-stage merge over 128 keys has merge MAC
  roughly `C*128/(4*52)` times hard C4 for the same query-column tile. For
  C8/C10/C12 that is about `4.9x`/`6.2x`/`7.4x` hard-C4 merge work, so a direct
  copy of the current hard-C4 merge kernel is unlikely to reach the 3x target.
- Direct Triton evidence: `flat128_c8_triton_realqkv_full3_20260813` tested a
  direct two-stage `flat128_c8_triton` implementation on scai7/H200 full3
  real Wan QKV. It kept the good dense error, rel-L2 `0.05931` and MSE
  `0.001024`, but measured `49.74` ms/call versus FlashAttention `14.54`
  ms/call and SDPA `18.01` ms/call. `flat128_c8_stage_profile_20260813`
  measured merge `21.58` ms, route `1.78` ms, and full `23.49` ms for one
  `query_outer_chunk=10` call, so the direct path is about `92%` merge-bound.
- Rule: do not repeat `flat64_c4/c5/c6` or `flat128_c4/c5/c6/c7` as speed
  candidates. `flat128_c8/c10/c12` are worth revisiting only with a new merge
  design that reduces the large 128-key seed-dot/softmax cost.

### Repartitioned Query/Key Blocks

- Result: `60 x 64` and `64 x 64` C4 blocks are worse than hard C4 in dense
  error and should not get Triton tuning yet.
- Change tested: `q60_k64_c4` uses two adjacent image columns as the query
  block (`30*2=60` query tokens) and row-major contiguous 64-token K/V blocks.
  `q64_k64_c4` uses row-major contiguous 64-token chunks for both query and
  K/V blocks. Both keep key-softmax assignment and the full compressed
  second-stage attention.
- Evidence: H200 first-eight-call real Wan QKV screen
  `query_key_block_60_64_screen_20260813`. Both layouts have
  compressed-token density `0.06410`. `q60_k64_c4` measured rel-L2 `0.20142`,
  MSE `0.005231`, and `q64_k64_c4` measured rel-L2 `0.70185`, MSE `0.06352`.
  Same-screen hard C4 measured rel-L2 `0.08765`, MSE `0.000991`.
- Follow-up: `q60_k64_c4_x2r2` used two row/column intersection queries plus
  two deterministic random query seeds. In `q60_k64_x2r2_screen_20260813`, it
  measured rel-L2 `0.21348` and MSE `0.005877`, slightly worse than
  `q60_k64_c4`.
- Rule: do not pursue `q60_k64_c4` or `q64_k64_c4` kernels unless a new seed
  selection rule first recovers quality in PyTorch analysis.

### Hard-C4 Tensorized Seed Tile

- Result: better readability, but not a speed path.
- Change tested: replace explicit `q0/q1/q2/q3` seed-query loads in the hard-C4
  merge kernel with one tensorized seed tile, conceptually
  `[C * query_columns, head_dim]`, then reshape scores into
  `[query_columns, key_columns, C]` for key-softmax assignment and merge.
- Evidence: commit `b373773` standardized the real-QKV benchmark to the full
  360-call setting: 3 MovieGen prompts, all four Wan denoising timesteps, and
  30 self-attention layers per timestep. On scai7/H200,
  `hard_c4_tensor_seed_360_20260814` measured `35.31` ms/call for the
  tensorized Triton hard C4 path versus `17.99` ms/call for dense SDPA. The
  previous maintained hard C4 360-call run measured about `15.75` ms/call.
  Dense-output error stayed reasonable at rel-L2 `0.06719` and MSE
  `0.001358`, but the tensorized layout more than doubled the hard-C4 runtime.
- Likely reason: Triton can compile this expression, but carrying
  `[J, L, C]` block tensors through reductions, masks, and reshapes creates
  more register/scheduling pressure than four simple 2D score tiles.
- Follow-up tuning on 2026-08-14 improved the tensorized default by changing
  the score layout to `[C, query_columns, key_columns]`, using
  `tl.argmax(route_prob, axis=0)` for key-softmax assignment, and avoiding
  `0 * -inf` in the expected-score path. The final full 360-call scai7/H200 run
  `hard_c4_final_confirm_360_20260814` measured dense SDPA at `18.29` ms/call
  and tensorized hard C4 at `18.37` ms/call, with stage 1 `14.00` ms/call,
  stage 2 `4.25` ms/call, rel-L2 `0.07389`, and MSE `0.001589`.
- A later same-day cleanup removed a redundant `route_mask & key_mask`
  operation. Invalid padded keys already have zero seed probability, and the
  expected-score path still masks them to avoid `0 * -inf`, so the output is
  preserved. The matched full 360-call run `hard_c4_j8_nomask_360_20260814`
  measured dense SDPA at `18.27` ms/call and hard C4 at `18.35` ms/call, with
  stage 1 `13.96` ms/call, stage 2 `4.25` ms/call, rel-L2 `0.07389`, and MSE
  `0.001589`.
- A second same-day live-range cleanup delayed the V load until after
  `weights @ K` and reused a single converted weights tile for both K and V
  dots. This preserves the same key-softmax assignment, merged K/V, routing
  bias, and second-stage full attention. The matched full 360-call run
  `hard_c4_delayed_v_shared_weights_360_20260814` measured dense SDPA at
  `18.31` ms/call and hard C4 at `18.04` ms/call, with stage 1 `13.68`
  ms/call, stage 2 `4.24` ms/call, rel-L2 `0.07389`, and MSE `0.001589`.
  This was the final tensorized configuration before the 2D fast path was
  restored as the default.
- Failed follow-ups in the same sweep: split-tail/tail-tile launch, comparing
  route logits as `score - logsumexp(score)` instead of materialized
  key-softmax probabilities, sharing the `weights.to(dtype)` cast across K/V
  dots before the V-load cleanup, and clamping padded seed-query columns. None
  beat the then-current 360-call setting.
- Additional H200 retuning after the cleanup rejected `BLOCK_J=4`
  (`10.20` ms A=10 stage profile), `BLOCK_J=16` despite a promising synthetic
  profile because the full 360-call run was slower at `18.42` ms/call,
  `BLOCK_J=32` because the merge kernel exploded to `176.87` ms in the A=10
  profile, merge warp counts `1` and `4`, route warp counts `2` and `8`, route
  `num_stages=1`, and route `BLOCK_I=16/64`.
- Additional full 360-call H200 checks after the V-load cleanup rejected:
  early-storing routing bias and merged K (`18.77` ms/call, stage 1 `14.37`);
  reloading K before `weights @ K` to shorten the score K tile live range
  (`19.24` ms/call, stage 1 `14.58`); using `evict_first` on the delayed V
  load (`18.11` ms/call, stage 1 `13.71`, slower than `evict_last`);
  merge `num_stages=1` (`18.08` ms/call, not better); and merge warps `4`
  (`20.78` ms/call, stage 1 `16.44`).
- Rule: keep this form only if manual debugging/readability is the priority.
  Within the tensorized form, keep `BLOCK_J=8`, `BLOCK_I=32`, merge/route
  `num_stages=2`, route warps `4`, merge warps `2`, no duplicated key-mask AND
  in the route mask, delayed V load, and one shared converted weights tile for
  the two K/V merge dots.

### Restored Hard-C4 2D Score Tiles

- Result: accepted as the maintained fast path. Replacing the tensorized
  rank-3 score value with four explicit 2D score tiles restores the historical
  H200 performance advantage while preserving the current hard-C4 output.
- Change tested: use one `BLOCK_J=64` merge program for all 52 query columns,
  four 2D `q0/q1/q2/q3` and `s0/s1/s2/s3` tiles, reuse the assignment logits
  in the cluster merge, and write bias, merged K, and merged V sequentially.
  The contiguous seed window, key-softmax assignment, tie-left behavior,
  entropy routing bias, intermediate layout, and full compressed second-stage
  attention are unchanged.
- Evidence: the canonical scai7/H200 360-call real-Wan run
  `hard_c4_2d_seq_real360_20260814` measured hard C4 at `15.668` ms/call,
  stage 1 at `11.289` ms/call, and stage 2 at `4.250` ms/call. Dense SDPA in
  the same run was `18.391` ms/call. Error was unchanged from the tensorized
  default at rel-L2 `0.073895` and MSE `0.001590`. The final tensorized
  baseline was `18.04` ms/call with stage 1 `13.68` ms/call.
- Resource finding: on Triton 3.7.1/sm90, the tuned tensorized merge compiled
  to `255` registers/thread, zero reported spills, and `36,864` bytes of shared
  memory. The restored 2D merge compiled to `255` registers/thread, `56`
  reported spills, and `81,920` bytes of shared memory, yet remained much
  faster because it covers all columns in one program and gives Triton simpler
  2D reduction/MMA layouts. A zero-spill count alone was therefore not a useful
  predictor for this kernel.
- Follow-up screen: sequential bias/K/V stores reduced reported spills from
  `76` to `56` and improved a matched GPU1 A=10 merge profile from about
  `5.204` to `5.182` ms. Sharing one explicit BF16 weights cast slowed merge
  to about `5.260` ms, and recomputing two of four seed scores slowed it to
  about `5.36` ms, so those variants were rejected.
- Accepted follow-up: moving the compressed-K/V `store_base` address expression
  until immediately before the merged-K/V stores and replacing the fixed C4
  loop/address multiplier with literal `4` improved the canonical 360-call
  real-Wan benchmark without changing dense error. The confirmation run
  `hard_c4_storebase_late_literal4_confirm_real360_20260814` measured hard C4
  at `15.446` ms/call, stage 1 at `11.055` ms/call, stage 2 at `4.261`
  ms/call, and dense SDPA at `18.509` ms/call. Rel-L2 stayed `0.073895` and
  MSE stayed `0.001590`. A same-setting run on GPU1 measured an even lower
  `15.291` ms/call, but use the confirmation number for conservative tables.
- Rejected follow-up: removing the query-column mask from `route0`...`route3`
  and masking only key columns keeps real query-column outputs unchanged, but
  the 360-call run `hard_c4_storebase_late_literal4_keyroute_real360_20260814`
  slowed to `15.529` ms/call with stage 1 `11.134` ms/call. Keep the real
  query-column mask in the cluster route masks.
- Accepted follow-up: after route assignment, write invalid query/key positions
  to route id `-1` and compute the per-seed mask inside the merge loop as
  `route == seed_id`. This removes the four persistent `route0`...`route3`
  boolean tiles from the live state without changing the hard assignment,
  key-softmax weights, routing bias, or second-stage compressed attention.
  The canonical 360-call real-Wan run
  `hard_c4_route_mask_live_real360_20260814` measured hard C4 at `15.129`
  ms/call, stage 1 at `10.863` ms/call, stage 2 at `4.138` ms/call, and dense
  SDPA at `18.591` ms/call. Rel-L2 stayed `0.073895` and MSE stayed
  `0.001590`.
- Rejected follow-up: changing the in-kernel route id tile from `int32` to
  explicit `int8` did not improve the merge stage and made the code more
  verbose. The canonical 360-call run `hard_c4_route_int8_real360_20260814`
  measured hard C4 at `15.119` ms/call and dense SDPA at `18.617` ms/call, but
  stage 1 was `10.870` ms/call versus `10.863` ms/call for the maintained
  route-live-state cleanup. Treat the tiny total-time difference as run noise,
  not a merge-stage win.
- Rejected follow-up: changing the four one-shot seed-query loads from
  `evict_last` to `evict_first` was much slower. The canonical 360-call run
  `hard_c4_seed_evict_first_real360_20260814` measured hard C4 at `16.099`
  ms/call with stage 1 `11.820` ms/call, versus the maintained route-live-state
  cleanup at `15.129` ms/call and stage 1 `10.863` ms/call. Keep `evict_last`
  on seed-query loads; Triton/H200 benefits from retaining those tiles through
  the four seed score dots.
- Rejected follow-up: specializing the second-stage route kernel's compressed
  token indexing from `// C` and `* C` to literal `// 4` and `* 4` did not beat
  the maintained code. The canonical 360-call run
  `hard_c4_route_literal4_real360_20260814` measured hard C4 at `15.136`
  ms/call, stage 1 at `10.868` ms/call, stage 2 at `4.154` ms/call, and dense
  SDPA at `18.591` ms/call, versus the maintained route-live-state cleanup at
  `15.129` ms/call. The output error was unchanged, so treat the difference as
  no speed win rather than a precision change.
- Accepted follow-up: add an optional hard-C4 head chunk dispatch and set the
  Wan hard-C4 config to `query_outer_chunk=21`, `head_chunk=6` on H200. The
  kernels still read the original full-head Q/K/V tensors, but only process
  one head range at a time and store into the matching output heads. This
  preserves the hard-C4 math exactly because heads are independent, while
  replacing the old `10+10+1` query-outer schedule with two all-outer head
  chunks. The canonical 360-call run
  `hard_c4_head6_q21_fixed_real360_20260814` measured hard C4 at `14.897`
  ms/call, stage 1 at `10.734` ms/call, stage 2 at `4.079` ms/call, and dense
  SDPA at `18.613` ms/call. Error stayed rel-L2 `0.073895`, MSE `0.001590`,
  and max-abs `6.28125`. A bug in the first attempt loaded V with the local
  head instead of the global head for `head_start>0`; fixed head-chunk random
  QKV checks match the full-head path exactly.
- Accepted follow-up: retain only the winning seed's normalized log-probability
  after assignment and exponentiate that one tile once. For seed `c` and its
  routed key set `R_c`, subtracting the seed-wide log-normalizer is a row
  scalar, so it cancels from both the cluster weights and routing bias:
  `exp(s_l - n_c) / sum_R exp(s - n_c) = exp(s_l) / sum_R exp(s)` and
  `E_w[s - n_c] - log(sum_R exp(s - n_c)) = E_w[s] - log(sum_R exp(s))`.
  This removes four cluster-local masked maxima and replaces four full-tile
  exponentials with one, while allowing `s0`...`s3` to die immediately after
  hard assignment.
- Evidence: matched random-QKV A=10 profiles measured the preceding route-id
  kernel at `4.907`/`6.864` ms for merge/full and the winner-probability kernel
  at `4.397`/`6.347` ms; reversing benchmark order reproduced `4.921`/`6.876`
  versus `4.398`/`6.325` ms. On Triton 3.7.1/sm90, reported resources changed
  from `255` registers/thread, `46` spills, `81,920` bytes shared memory, and
  about `300k` PTX characters to `255`, `44`, `81,920`, and `275k`.
- Real-Wan evidence: the same-QKV canonical 360-call run
  `hard_c4_winner_prob_real360_20260814` measured the preceding kernel at
  `15.072` ms/call with stage 1 at `10.791` ms, and the winner-probability
  kernel at `13.831` ms/call with stage 1 at `9.595` ms. Dense SDPA was
  `18.345` ms/call. Rel-L2/MSE were `0.073895`/`0.00158953` for the preceding
  kernel and `0.073891`/`0.00158936` for the accepted kernel, i.e. no quality
  regression beyond normal floating-point rounding.
- Production-dispatch confirmation: the final same-run canonical profile
  `hard_c4_winner_prob_q21_head6_real360_20260814` used the maintained
  `query_outer_chunk=21`, `head_chunk=6` schedule and measured integrated hard
  C4 at `13.677` ms/call, with stage 1 at `9.516` ms and stage 2 at `4.072` ms,
  versus dense SDPA at `18.685` ms and the Triton FlashAttention reference at
  `15.195` ms. Rel-L2/MSE stayed `0.073891`/`0.00158936`. The older q10/full-head
  schedule measured `13.998` ms/call, and an earlier production-only
  confirmation measured `13.899` ms/call. These runs rule out an improvement
  caused only by the isolated profiler's launcher substitution.
- Head-chunk retest after winner-probability reuse: `head_chunk=4` is not the
  best production schedule for the current kernel. The full 360-call H200 run
  `hard_c4_winner_prob_q21_head4_real360_20260814` measured `13.888` ms/call,
  with stage 1 `9.680` ms, stage 2 `4.089` ms, rel-L2 `0.073891`, and MSE
  `0.00158936`, so it is slower than the maintained `head_chunk=6` result.
  `head_chunk=8` under the same `query_outer_chunk=21` setting triggered an
  illegal memory access because the active-head compressed-K/V workspace exceeds
  the current 32-bit element-offset range. Do not retest larger head chunks
  without first changing the intermediate address/layout design.
- Rejected follow-up: adding automatic int64 merged-K/V element offsets allowed
  `query_outer_chunk=21`, `head_chunk=12` to run, but did not improve speed.
  A correctness-only full-shape check showed `head_chunk=6` and `12` outputs
  were bitwise identical. The full 360-call H200 run
  `hard_c4_int64_q21_head12_real360_20260814` measured hard C4 at `14.350`
  ms/call, with stage 1 `10.067` ms and stage 2 `4.222` ms, versus the
  maintained `head_chunk=6` result at `13.677` ms/call, stage 1 `9.516` ms,
  and stage 2 `4.072` ms. The int64-offset production patch was reverted; do
  not reintroduce it unless a new workspace layout also reduces merge cost.
- Rejected follow-up: exact `52 x 52` merge tiles cannot be used directly in
  Triton because `tl.arange` requires a power-of-two range. Splitting query
  columns with `BLOCK_J=32` did compile, but the H200 stage profile
  `hard_c4_j32_stage_20260814` slowed to full `9.761` ms for
  `query_outer=21, heads=6`, versus the same-shape baseline around `6.574` ms.
  Keep `BLOCK_J=64` and `BLOCK_L=64`.
- Neutral follow-up, not retained: zeroing invalid `winner_logprob` lanes and
  computing the routing-bias expected value from unnormalized probabilities
  was mathematically equivalent on the maintained Wan calls, but did not beat
  the accepted total runtime. The 360-call run
  `hard_c4_zeroed_logprob_real360_20260814` measured `13.678` ms/call with
  rel-L2 `0.073890689`, versus the accepted `13.677` ms/call. Keep the
  accepted winner-prob implementation unless a future full run shows a real
  total-time win.
- Rejected follow-up: smaller route tiles or different warp/stage scheduling
  did not help. `BLOCK_K=16` slowed route to `2.112` ms, route warps `2`
  slowed route to `2.556` ms, merge warps `2` was catastrophic at `249.45`
  ms merge, merge warps `8` slowed merge to `7.159` ms, merge
  `num_stages=2` did not improve full time, and route `num_stages=1` slowed
  route to `2.379` ms in same-shape stage profiles. Keep
  `BLOCK_K=32`, merge/route warps `4`, merge stages `1`, and route stages `2`.
- Rejected follow-up: making K/V head-major before hard C4, and then changing
  the merged K/V workspace to a head-major layout, did not improve end-to-end
  speed. The 4-call real-QKV check `hard_c4_headmajor_kv_real4_20260814`
  measured `13.91` ms/call because the K/V rotation copy appeared as extra
  overhead. The standalone merged-layout stage profile
  `hard_c4_headmajor_merged_stage_20260814` also slowed to full `6.615` ms
  versus the baseline around `6.574` ms. Keep the current token-major K/V input
  and merged K/V layout unless a new design removes the rotation copy and
  shows a stage win.
- Rejected follow-up: storing routing bias in BF16 rather than FP32 reduced
  the bias workspace size, but made the route kernel slower. The H200 stage
  profile `hard_c4_bf16_bias_stage_20260814` measured merge `4.459` ms, route
  `2.734` ms, and full `7.181` ms for `query_outer=21, heads=6`, versus the
  FP32-bias baseline around route `2.016` ms and full `6.574` ms. Keep routing
  bias as FP32.
- Rejected follow-up: moving the input-V load from before the static seed loop
  into each seed iteration, after the merged-K store, did not improve live
  range enough to offset reading the same V tile four times. The H200 stage
  profile `hard_c4_perseed_vload_stage_20260814` measured merge `5.038` ms,
  route `2.002` ms, and full `7.086` ms for `query_outer=21, heads=6`, slower
  than the maintained winner-probability baseline. Keep one delayed V load
  after assignment but before the seed loop.
- Neutral follow-up, not retained: swapping the per-seed merge order from
  merged-K then merged-V to merged-V then merged-K produced only noise-level
  movement. The H200 stage profile `hard_c4_value_first_stage_20260814`
  measured merge `4.531` ms, route `2.002` ms, and full `6.577` ms for
  `query_outer=21, heads=6`, essentially tied with the same-GPU baseline
  `6.579` ms. Keep the current bias, merged-K, merged-V sequential order.
- Rejected follow-up: rewriting normalized merge weights as
  `probs * (1 / safe_sum)` instead of `probs / safe_sum[:, None]` made Triton
  generate a slower merge path. The H200 stage profile
  `hard_c4_recip_weights_stage_20260814` measured merge `4.790` ms, route
  `2.004` ms, and full `6.840` ms for `query_outer=21, heads=6`, worse than
  the same-GPU baseline `6.579` ms. Keep the direct elementwise divide.
- Neutral follow-up, not retained: explicitly casting merge and route
  `tl.program_id` values to `tl.int32` did not improve address arithmetic.
  The H200 stage profile `hard_c4_int32_pid_stage_20260814` measured merge
  `4.529` ms, route `2.000` ms, and full `6.591` ms for `query_outer=21,
  heads=6`, a noise-level/slower result than the same-GPU baseline. Keep the
  plain `tl.program_id` form.
- Rejected follow-up: adding a `BATCH == 1` constexpr branch to avoid
  `batch_head % ACTIVE_HEADS` and `batch_head // ACTIVE_HEADS` in merge and
  route kernels slowed the merge stage. The H200 stage profile
  `hard_c4_batch1_stage_20260814` measured merge `4.727` ms, route `1.992`
  ms, and full `6.778` ms for `query_outer=21, heads=6`. Keep the simple
  modulo/divide head decode; Triton schedules it better in the maintained
  kernel.
- Historical schedule note: before the later single-J/log2/D-mask cleanups, a
  real-QKV chunk schedule sweep did not find a replacement for
  `query_outer_chunk=21`, `head_chunk=6`. On idle H200 GPU4, the 12-call sweep
  `hard_c4_schedule_sweep12_gpu4_20260814` measured `q21_h4` slightly ahead
  of `q21_h6`, but the old standard 360-call recheck reversed it:
  `hard_c4_q21_h4_gpu4_real360_20260814` measured `13.651` ms/call, while
  `hard_c4_q21_h6_gpu4_real360_20260814b` measured `13.518` ms/call with the
  same dense error. Do not use those older schedule numbers as the current
  default after the later kernel changes.
- Accepted launch-grid follow-up: removing the dummy merge-kernel query-column
  grid axis improved the maintained schedule without changing the math. One
  `BLOCK_J=64` tile already covers all 52 real Wan query columns, so the merge
  grid only needs `(query_outer, key_outer, key_row, batch, head)` programs,
  not an additional one-element J axis. The standard 360-call confirmation
  `hard_c4_single_j_grid_confirm2_real360_20260814` measured hard C4 at
  `13.209` ms/call, stage 1 at `9.038` ms/call, stage 2 at `4.064` ms/call,
  and dense SDPA at `18.460` ms/call. Rel-L2, MSE, and max-abs stayed
  `0.073890689`, `0.001589365`, and `6.28125`. The immediately preceding
  maintained q21/head6 baseline measured `13.518` ms/call with stage 1
  `9.339` ms/call.
- Accepted log2-domain merge follow-up: compute the key-softmax assignment in
  log2 units and convert the entropy routing bias back to natural-log units
  only at the final bias store. This preserves the same hard-C4 assignment,
  merged K/V, routing bias definition, and full compressed second-stage
  attention while removing repeated large-tile `log2(e)` multiplications before
  `exp2`. The standard 360-call confirmations
  `hard_c4_log2_merge_real360_20260814` and
  `hard_c4_log2_merge_confirm2_real360_20260814` measured hard C4 at `12.861`
  and `12.885` ms/call, with stage 1 at `8.701` and `8.712` ms/call. Dense
  SDPA measured `18.496` and `18.506` ms/call. Rel-L2/MSE/max-abs were
  `0.073890634`, `0.001589362`, and `6.28125`, effectively unchanged from the
  preceding single-J result.
- Accepted log2-era stage-scheduling follow-up: after the log2-domain rewrite,
  merge `num_stages=2` became a small but repeatable improvement. The standard
  360-call confirmations `hard_c4_log2_merge_stages2_real360_20260814` and
  `hard_c4_log2_merge_stages2_confirm2_real360_20260814` measured hard C4 at
  `12.859` and `12.855` ms/call, with stage 1 at `8.696` and `8.702` ms/call.
  Rel-L2/MSE/max-abs stayed `0.073890634`, `0.001589362`, and `6.28125`.
- Accepted power-of-two D-mask cleanup: Wan hard C4 uses `D_HEAD=128`, so the
  D-lane mask in the merge and route kernels is all true. Dropping that mask
  and adding a static `D_HEAD == BLOCK_D` fast-path guard preserved output and
  made the code's specialization explicit. The standard 360-call confirmations
  `hard_c4_drop_dmask_real360_20260814` and
  `hard_c4_drop_dmask_confirm2_real360_20260814` measured hard C4 at `12.852`
  and `12.855` ms/call, with stage 1 at `8.697` and `8.701` ms/call. Error
  stayed rel-L2 `0.073890634`, MSE `0.001589362`, and max-abs `6.28125`.
- Accepted post-cleanup schedule update: after the log2-domain and D-mask
  cleanups, `head_chunk=4` became faster than `head_chunk=6` under the same
  `query_outer_chunk=21` full-360 setting. The standard confirmations
  `hard_c4_drop_dmask_q21_h4_real360_20260814` and
  `hard_c4_drop_dmask_q21_h4_confirm2_real360_20260814` measured hard C4 at
  `12.732` and `12.759` ms/call, with stage 1 at `8.528` and `8.547` ms/call,
  stage 2 at `4.084` and `4.087` ms/call, and dense SDPA at `18.747` and
  `18.765` ms/call. Error stayed rel-L2 `0.073886634`, MSE `0.001588173`, and
  max-abs `6.34375`. This is the current retained production dispatch.
- Rejected post-cleanup head-chunk follow-up: smaller or uneven head chunks did
  not beat the retained `head_chunk=4` schedule. The standard 360-call runs
  `hard_c4_drop_dmask_q21_h3_real360_20260814`,
  `hard_c4_drop_dmask_q21_h2_real360_20260814`, and
  `hard_c4_drop_dmask_q21_h5_real360_20260814` measured hard C4 at `13.223`,
  `12.986`, and `13.054` ms/call respectively. Their errors matched the
  retained path, but all were slower than the two `head_chunk=4` confirmations
  at `12.732` and `12.759` ms/call. Keep `head_chunk=4`.
- Rejected post-cleanup chunk-schedule follow-up: changing both query and head
  chunks did not find a better launch schedule. The full 360-call scai7/H200
  run `hard_c4_schedule_sweep_full360_20260814b` compared `q21_h4`,
  `q14_h6`, `q10_h8`, `q7_h12`, and `q21_h6` on the same real QKV stream.
  The run was lightly contended on GPU0, so treat absolute times as
  non-baseline, but the within-run ordering still did not justify a change:
  `q21_h4` measured `15.146` ms/call, `q21_h6` `15.149`, `q7_h12` `16.064`,
  `q14_h6` `16.276`, and `q10_h8` `16.478`. All schedules had the same dense
  error. Keep `query_outer_chunk=21`, `head_chunk=4`.
- Rejected prob-dot merge follow-up: delaying the routed-cluster normalization
  until after the K/V dot is algebraically equivalent in exact arithmetic, but
  it is not numerically safe for the BF16 Wan path. The standard 360-call run
  `hard_c4_probdot_q21_h4_real360_20260814` produced NaN rel-L2/MSE values,
  indicating that tiny routed-cluster mass values can be amplified after the
  BF16 dot. The timing was also slower under a contended H200 (`25.297`
  ms/call), so this change was reverted. Keep explicit normalized `weights`
  before the merged K/V dot.
- Rejected key-outer grouping follow-up: grouping two adjacent key outer blocks
  inside one merge program reduced repeated seed-Q loads, but it made each
  Triton program too long and cut parallelism. The real-QKV compile check
  `hard_c4_keyouter_group_smoke_20260814` measured one recorded call with
  stage 1 at `15.756` ms/call and total hard C4 at `20.177` ms/call, far worse
  than the current 360-call stage 1 near `8.54` ms/call. Keep one key outer per
  merge program.
- Rejected log2-era warp-scheduling follow-up: merge `num_warps=8` is still a
  bad schedule after the log2-domain rewrite. The standard 360-call run
  `hard_c4_log2_merge_warps8_real360_20260814` measured hard C4 at `19.313`
  ms/call, with stage 1 at `15.188` ms/call and dense SDPA at `18.136`
  ms/call. Error was unchanged, so this is a pure speed regression.
- Head-chunk integration guard: the new head-offset path must use
  `global_head = HEAD_START + head` for the input V load, just as it already
  does for Q and K. Using the chunk-local head there silently reads V from the
  wrong head for every nonzero `HEAD_START`. After fixing the index, head chunks
  `1`, `2`, `3`, `4`, and `6` were bitwise identical to the full-12-head launch
  on the complete A=10 random shape. At fixed `query_outer_chunk=10`, full-head
  launch remained fastest at `6.314` ms, versus `6.338` ms for chunk 6 and
  `6.717` ms for chunk 1. The maintained chunk 6 is valuable specifically
  because it safely enables the full `query_outer_chunk=21` schedule; that
  combined schedule was faster in the real 360-call run.
- Rejected conservative follow-up: selecting the raw winning score, computing
  four routed-cluster maxima, and then exponentiating one centered tile was
  bitwise identical to the preceding kernel for merged K/V, bias, and final
  output, but its A=10 merge time was `4.947` ms versus `4.892` ms. Keep it only
  as an isolated numerical reference.
- Numerical guardrail: the accepted fast form assumes the winning normalized
  probabilities do not all underflow inside a nonempty routed cluster. This was
  true for all 360 maintained Wan RMS-normalized BF16 calls. Use the stable
  isolated reference if future dispatch expands to adversarial or much
  higher-dynamic-range inputs.
- Rule: keep the maintained merge at `BLOCK_J=64`, merge warps `4`, merge
  `num_stages=2`, route warps `4`, route `num_stages=2`, late `store_base`
  computation, literal C4 address stride, masked route ids, `evict_last`
  seed-query loads, and one reused winning-probability tile instead of four
  persistent route-mask tiles and four cluster exponentials. Launch one merge
  program per `(query_outer, key_outer, key_row, batch, active_head)`, with no
  dummy J-grid axis. Keep the key-softmax/route comparison in log2 units and
  store routing bias in natural-log units. Keep the power-of-two head-dim
  fast path without D-lane masks for Wan's `D_HEAD=128`. On H200, pair it
  with `query_outer_chunk=21`, `head_chunk=4`. Treat the tensorized seed tile
  as debugging code, not the production speed path.

### Hard-C4 Route Key-Only Output Mask

- Result: accepted as a small route-stage cleanup.
- Change tested: after the compressed QK dot, mask only padded key tokens
  instead of combining the key mask with `row_mask`. The two padded query rows
  in the `BLOCK_I=32` tile reduce independently from valid rows and are never
  stored, so all valid outputs are bitwise unchanged.
- Evidence: matched H200 A=10 A/B profiles measured route at `1.998` ms for
  the original mask and `1.943` ms for the key-only mask. The canonical
  360-call real-Wan run `hard_c4_2d_route_keymask_real360_20260814` measured
  hard C4 at `15.612` ms/call, with stage 1 `11.293` ms/call and stage 2
  `4.146` ms/call, versus dense SDPA at `18.384` ms/call. Error remained
  rel-L2 `0.073895` and MSE `0.001590`.
- Rule: keep the key padding mask in route logits, but do not reintroduce the
  redundant padded-query-row mask.

### Hard-C4 Merge Key-Only Route-ID Domain

- Result: rejected.
- Change tested: in the merge kernel, mark the integer `route` tile valid for
  all real key columns with `key_col_mask` instead of the stricter
  `j_mask & key_mask`. The stored bias and merged K/V tensors are still masked
  by `j_mask`, and `winner_prob` still uses the real `j_mask & key_mask`
  domain, so valid output columns are mathematically unchanged.
- Evidence: the standard scai7/H200 360-call run
  `hard_c4_key_route_domain_real360_20260814` measured hard C4 at `14.498`
  ms/call, with stage 1 at `10.345` ms/call and stage 2 at `4.061` ms/call,
  versus dense SDPA at `18.391` ms/call. Error was unchanged at rel-L2
  `0.073890689`, MSE `0.001589365`, and max-abs `6.28125`, but the maintained
  `q21/head6` route-domain code measured `13.518` ms/call on the same clean
  H200 class setting.
- Likely reason: padded query-column lanes still participate in the same tile
  reductions and merge dots, so relaxing the route-ID validity mask does not
  remove meaningful work and appears to increase merge scheduling cost.
- Rule: keep `route = tl.where(real_mask, route, -1)` in the maintained
  winner-probability merge kernel.

### Hard-C4 Unmasked Winner Probability

- Result: rejected.
- Change tested: remove the `tl.where(real_mask, ..., 0.0)` around
  `winner_prob`, relying on the already-masked integer route IDs and masked
  output stores to drop invalid query/key lanes later. This preserves valid
  outputs because padded columns never contribute to a stored merged token.
- Evidence: the standard scai7/H200 360-call run
  `hard_c4_unmasked_winner_prob_real360_20260814` measured hard C4 at
  `13.915` ms/call, with stage 1 at `9.767` ms/call and stage 2 at `4.057`
  ms/call, versus dense SDPA at `18.378` ms/call. Error was unchanged at
  rel-L2 `0.073890689`, MSE `0.001589365`, and max-abs `6.28125`, but this
  remained slower than the maintained `q21/head6` result at `13.518` ms/call
  and stage 1 `9.339` ms/call.
- Likely reason: removing the select does not remove the full-tile exponent or
  merge-dot work, and it appears to make Triton keep a less favorable live
  probability tile.
- Rule: keep the explicit `real_mask` zeroing around `winner_prob` in the
  production hard-C4 merge.

### Hard-C4 Sequential Seed Winner Update

- Result: rejected.
- Change tested: compute each seed's key-softmax normalization and immediately
  update the winning seed/log-probability tile, instead of computing all four
  `s0..s3` score tiles and then routing. The key-softmax assignment, tie-left
  route rule, winner-probability reuse, merged K/V, entropy routing bias, and
  full compressed second-stage attention were unchanged.
- Evidence: the standard scai7/H200 360-call run
  `hard_c4_sequential_winner_real360_20260814` measured hard C4 at `13.983`
  ms/call, with stage 1 at `9.839` ms/call and stage 2 at `4.055` ms/call,
  versus dense SDPA at `18.372` ms/call. Error was unchanged at rel-L2
  `0.073890689`, MSE `0.001589365`, and max-abs `6.28125`, but it was slower
  than the maintained `q21/head6` result at `13.518` ms/call and stage 1
  `9.339` ms/call.
- Likely reason: the apparent shorter score lifetime does not translate into a
  better Triton schedule. Interleaving dot, reduction, and route updates per
  seed appears to reduce instruction-level scheduling freedom more than it
  reduces register pressure.
- Rule: keep the production merge organized as four explicit 2D score tiles,
  followed by the route update chain.

### Hard-C4 Unpadded Query-Column Tile

- Result: rejected at compile time.
- Change tested: set the hard-C4 merge `BLOCK_J` to the true Wan column count
  `52` instead of padding the query-column tile to `64`, while leaving all math
  and the second-stage full compressed attention unchanged.
- Evidence: the standard scai7/H200 run
  `hard_c4_blockj52_real360_20260814` failed during Triton compilation before
  any metrics were produced. Triton raised `ValueError: arange's range must be
  a power of 2` at `js = j_start + tl.arange(0, BLOCK_J)`.
- Rule: keep `HARD_RT_BLOCK_J=64` for this Triton implementation. Testing a
  true 52-column tile would require a different indexing/kernel structure, not
  just changing the tile constant.

### Hard-C4 Safe Winner Expected Score

- Result: rejected.
- Change tested: zero invalid `winner_logprob` lanes once after assignment and
  compute routing-bias expected score as `sum(weights * safe_winner_logprob)`
  instead of `sum(where(route_mask, weights * winner_logprob, 0))` inside each
  seed loop. The key-softmax assignment, cluster weights, routing bias value,
  merged K/V, and full compressed second-stage attention were unchanged for the
  maintained Wan calls.
- Evidence: the standard scai7/H200 360-call run
  `hard_c4_safe_expected_real360_20260814` measured hard C4 at `13.560`
  ms/call, with stage 1 at `9.416` ms/call and stage 2 at `4.056` ms/call,
  versus dense SDPA at `18.403` ms/call. Error stayed at rel-L2
  `0.073890689`, MSE `0.001589365`, and max-abs `6.28125`, but this did not
  beat the maintained `q21/head6` result at `13.518` ms/call and stage 1
  `9.339` ms/call.
- Likely reason: removing one per-seed masked product is offset by carrying an
  additional full `safe_winner_logprob` tile, so live state is not improved.
- Rule: keep the existing masked expected-score expression in the production
  hard-C4 merge.

### Fixed-Shape CUDA/WMMA Hard-C4 Merge

- Result: correct in aggregate but far slower than the maintained Triton
  merge, so it remains an isolated research prototype and is not dispatched.
- Change tested: specialize to SM90, BF16, `C=4`, `COLS=52`, and `D=128` with
  one 256-thread CTA per `(batch, head, query_outer, key_outer, key_row)`. Two
  four-warp groups reuse shared K/V, compute four `J=16` microtiles with WMMA,
  hard-route only the winning seed, and perform separate weight-times-K and
  weight-times-V WMMA epilogues. This removes Triton's rank-3 temporary
  layouts from the source-level dataflow.
- Evidence: on the full random A=10, K=21, H=12 H200 shape, artifact
  `hard_c4_cuda_wmma_fullshape_cachedattr_20260814` measured `47.946` ms for
  CUDA/WMMA versus `5.156` ms for the restored Triton merge. The finite bias
  mask matched exactly; merged-K/V aggregate rel-L2 was about `2.1e-4` and
  `2.3e-4`, but near-tie route differences produced maximum absolute outliers
  around `0.4`. The kernel used 138 registers/thread and 96 KiB dynamic shared
  memory.
- Why not repeat directly: legacy WMMA plus CTA-wide barriers and FP32 score
  round-trips through shared memory do not reproduce Triton's efficient SM90
  scheduling. A serious CUDA follow-up must use an SM90-native WGMMA/TMA,
  warp-specialized pipeline rather than translating the Triton operations.
- Rule: do not connect `analysis/hard_c4_cuda_wmma` to production dispatch.

### Post-q21/head4 Source-Level Follow-Up Sweep

- Result: rejected or neutral; no candidate was promoted to production or to a
  360-call real-QKV confirmation.
- Scope: after the retained single-J, log2-domain, D-mask, and
  `query_outer_chunk=21`/`head_chunk=4` updates, several exact scheduling
  changes were rechecked on one H200 with the complete random
  `B=1, Q_OUTER=K_OUTER=21, ROWS=30, COLS=52, HEADS=12, D=128` shape. Timings
  used independent workspaces and paired ABBA ordering; only four active heads
  were launched, matching one production head chunk.
- Delaying the sole input-V load until after the seed-zero merged-K store was
  bitwise identical, but the matched profile
  `value_after_k0_q21_h4_gpu2.json` slowed merge from `2.729` to `2.772`
  ms/chunk and merge-plus-route from `4.200` to `4.234` ms/chunk. Peeling seed
  zero costs more scheduling freedom than the shorter source-level V lifetime
  saves. Keep one V load after assignment and before the seed loop.
- A four-seed routing-bias prepass, intended to kill the full winner-logprob
  tile before all K/V dots, slowed merge from `2.729` to `2.905` ms/chunk and
  full from `4.151` to `4.329` ms/chunk in
  `bias_prepass_q21_h4_gpu1.json`. Rematerializing the cheap route/probability
  masks instead of retaining them also slowed merge from `2.739` to `2.879`
  ms/chunk and full from `4.152` to `4.291` ms/chunk in
  `remat_q21_h4_gpu1.json`. Keep the current per-seed bias/K/V sequence.
- Storing the entropy bias in log2 units and consuming it directly in route was
  effectively neutral to slightly slower on an idle H200. Artifact
  `log2_bias_q21_h4_gpu2.json` measured merge medians `2.734` ms for production
  and `2.734` ms for log2-bias storage, while merge-plus-route moved from
  `4.195` to `4.202` ms/chunk. Merged K/V were identical; the final output had
  rel-L2 `1.69e-4` and max-abs `1.22e-4` from the changed FP32 rounding path.
  Keep the FP32 workspace bias in natural-log units.
- Removing the route kernel's always-one row-grid axis and setting
  `disallow_acc_multi_buffer=True` were both exact but noise-level. In
  `route_exact_noacc_gpu1.json`, route medians were `1.378` ms for production,
  `1.375` ms for the loop attribute, and `1.372` ms for the 2-D grid; repeated
  simple timings did not preserve that ordering. Explicit route `maxnreg`
  limits were materially worse: `route_maxnreg_q21_h4_gpu1.json` measured the
  default at `1.356` ms, versus `1.713`, `1.490`, `1.488`, and `1.438` ms for
  limits `96`, `112`, `128`, and `144` respectively. Keep the default compiler
  register allocation and the straightforward route launch.
- A two-stream, double-buffered head-chunk pipeline preserved output but did
  not overlap usefully: `head_pipeline_q21_h4_abba.json` measured the simple
  sequential path at `12.339` ms/call and the pipeline at `12.459` ms/call in
  its least-contended window. Merge and route compete for the same HBM/tensor
  resources, so the extra workspace and stream/event machinery are not
  justified.
- Resource note: the then-retained pre-two-pair SM90 merge compiles at 255 registers/thread with
  40 bytes of stack and 81,920 bytes of shared memory; route compiles at 159
  registers/thread and 82,432 bytes of shared memory. These limits explain why
  plausible source-level live-range changes remain worth checking, but the
  results above show that Triton's existing schedule is already better than
  the obvious manual reorderings.
- Rule: stop source-level pointer/lifetime micro-tuning at this point. A
  material next step must reduce or bypass the intermediate merged-K/V traffic,
  for example through an SM90-native WGMMA/TMA producer-consumer design, rather
  than translating the current Triton operations directly to legacy CUDA WMMA.
### Hard-C4 Two-Pair Seed-Q Residency

- Result: accepted as the production merge schedule.
- Change tested: load `q0/q1`, consume their QK dots, then load `q2/q3` and
  consume their QK dots. All four 2D score tiles, key-softmax assignment,
  tie-left routing, cluster weights, entropy bias, merged K/V, intermediate
  layout, and full compressed second-stage attention remain unchanged.
- Resource evidence: Triton 3.7.1/sm90 reduced compiler-reported shared memory
  from `81,920` to `49,152` bytes per CTA and cubin size from `292,656` to
  `290,992` bytes. Both kernels use `255` registers/thread and a `48`-byte
  stack according to `cuobjdump`.
- Runtime evidence: the adjacent-launch A/B on the representative
  `query_outer=21`, `head_chunk=4` shape had a stable uncontended interval near
  `2.73` ms for the old schedule and `2.55` ms for the two-pair schedule. The
  canonical 360-call same-QKV A/B ran while GPU0 was shared: the old schedule
  measured `29.984` ms/call with stage 1 at `20.112` ms, and the two-pair
  schedule measured `29.318` ms/call with stage 1 at `19.501` ms. This is a
  `2.27%` total and `3.04%` stage-1 latency reduction under matched contention.
- Accuracy evidence: isolated merged K/V/bias were bitwise identical. The
  360-call run gave both variants rel-L2 `0.073890634`, MSE `0.001589362`, and
  max-abs `6.28125` versus dense.
- Rule: keep `query_outer_chunk=21`, `head_chunk=4`, four warps, two stages,
  and the two-pair Q-load/dot schedule. Keep the earlier idle-H200
  `12.732`/`12.759` ms results as the absolute baseline until this exact code is
  remeasured on an idle H200.

### Hard-C4 Q-Residency Follow-Ups

- Result: none beat the accepted two-pair schedule.
- Exact `1+1+1+1` loading reduced shared residency further but reached only
  about `1.053x` paired speedup over the old schedule, versus about `1.07x` for
  `2+2`. The `3+1` schedule reached about `1.011x`.
- Updating the winner after each two-seed pair was not faster than plain `2+2`
  and changed routing bias by up to `7.15e-7` through FP32 reassociation.
- Replacing the four-way static merge loop with a dynamic loop reached about
  `1.059x` over the old schedule; `loop_unroll_factor=2` reached only about
  `1.015x`. Keep the static loop.
- Balanced launch swizzles with query-outer groups `2`, `4`, and `7` did not
  improve beyond the unswizzled two-pair kernel. The grouped index arithmetic
  is therefore not retained.
- `num_ctas=2` failed in Triton's `TritonGPUPlanCTAPass` on sm90. Merge warps
  `2` remained pathological, merge warps `8` remained slower, and stages `1`
  versus `2` did not change the two-pair conclusion.

### Hard-C4 Low-Risk Kernel Follow-Ups

- Result: rejected or neutral; none changed the retained production code.
- A 3D merge launch grid removed block-id divisions but changed latency by only
  about `0.09%`. Triton block-pointer Q/K/V loads were about `0.915x` as fast
  as pointer-tensor loads. Pairing two seed QK dots into one larger MMA was
  about `0.601x` as fast as the maintained 2D dots.
- CUDA-stream overlap of merge and route head chunks was exact but only
  `1.005x` faster, indicating that the stages contend for the same GPU
  resources. A correctness-first shared-scratch SM90 WGMMA prototype matched
  merged K/V but ran at about `0.123x` the Triton merge speed; the register-only
  two-CTA WGMMA epilogue did not map accumulator fragments correctly.
- `head_chunk=12` again exceeded the current 32-bit intermediate element-offset
  range and raised an illegal memory access. On the heavily shared GPU,
  `head_chunk=6` was less affected by external preemption than `head_chunk=4`,
  but prior idle-H200 full-360 comparisons favor or tie `head_chunk=4`; do not
  change the production config from this contended result.

### Hard-C4 Post-Two-Pair H200 Confirmation

- Result: the retained `query_outer_chunk=21`, `head_chunk=4` dispatch remains
  the best measured production schedule after the two-pair seed-Q residency
  change.
- Evidence: two idle-H200 360-call confirmations measured hard C4 at `12.321`
  and `12.388` ms/call, with stage 1 at `8.109` and `8.124` ms/call and stage
  2 at `4.093` and `4.132` ms/call. A same-QKV `head_chunk=4` versus `6` run
  then measured `12.125` versus `12.464` ms/call, while dense SDPA measured
  `18.895` ms/call. The head-4 result is a `1.558x` attention speedup and kept
  rel-L2 `0.073891`, MSE `0.00158936`, and max-abs `6.28125` versus dense.
- Rule: use `head_chunk=4` for the current H200 hard-C4 production path.

### Hard-C4 Joint K/V Merge MMA

- Result: rejected despite bitwise-identical output.
- Change tested: concatenate each input K/V tile in the MMA output dimension
  and replace the separate `weights @ K` and `weights @ V` operations with one
  `weights @ [K|V]` operation per seed. This preserves assignment, normalized
  merge weights, entropy bias, intermediate values, and full compressed
  second-stage attention.
- Evidence: on an idle H200 with the production-size
  `[1, 21, 30, 52, 4, 128]` active-head chunk, merge latency increased from
  `2.540` to `3.640` ms and full latency increased from `3.951` to `5.000` ms.
  Maximum and mean output differences were both zero.
- Why not repeat: the wider `64 x 256` accumulator increases register/shared
  pressure enough to dominate any saving from reusing the weights operand.
  Keep separate, sequential K and V MMAs.

### Hard-C4 Route Loop Scheduling Follow-Ups

- Result: rejected; all outputs were bitwise identical to production.
- Adding a two-stage software pipeline to the key-outer loop changed route
  latency from about `1.323` to `1.328` ms. Setting
  `loop_unroll_factor=2` slowed it to `1.863` ms. Hopper automatic
  `warp_specialize=True` slowed it from `1.326` to `1.609` ms on the same
  production-size active-head chunk.
- Nsight Compute hardware-counter collection is unavailable to this account on
  scai7 (`ERR_NVGPUCTRPERM`), so source-level stage timings and logical traffic
  remain the available profiling methods.
- Rule: keep the simple key-outer `tl.range` loop in the route kernel.

### Hopper TMA, DSM, and Hand-Written WGMMA Follow-Ups

- Result: none beat the retained two-stage Triton merge.
- Triton tensor-descriptor/TMA loads were exact but slower. On the production
  `[1,21,30,52,4,128]` active-head chunk, merge changed from `2.540` ms to
  `2.621` ms with all TMA loads, `2.698` ms with only K/V TMA, and `3.185` ms
  with only Q TMA. Route TMA changed `1.323` ms to `1.328` ms.
- An eight-CTA SM90 cluster probe showed that repeated distributed shared
  memory exchange is too expensive for passing every compressed block from a
  merge producer to route consumers. Stable remote reads took `26.09` ms and
  synchronized remote reads/writes took `31.00`--`42.16` ms in the
  representative 630-iteration probe; local shared-memory work was much
  cheaper.
- The 96-KiB hand-written WGMMA merge was numerically correct: merged K/V were
  bitwise equal to Triton and finite bias differed by at most `7.15e-7`.
  Nevertheless, the unconstrained one-CTA version took `21.005` ms versus
  Triton's `2.541` ms, while the two-CTA/SM launch-bound version took `16.878`
  ms. `ptxas` explicitly reported serialized WGMMA instructions from
  insufficient registers in the two-CTA case.
- Rule: do not replace the production merge with the current TMA, DSM, or
  hand-written WGMMA prototypes. A future SM90 implementation must remove the
  repeated barriers/repacking and keep independent consumer accumulators in
  separate warpgroups; merely translating the Triton tile is not enough.

### Hard-C4 One-Pass and Partial-State Fusion

- Result: the best ordinary-Triton fusion remains `2.565x` slower than the
  retained two-stage kernel, so it is not dispatched.
- A warm-cache J8 load-only lower-bound probe took `1.280` ms per four-head
  chunk, while the original all-in-one J8 Triton kernel took more than `63`
  ms. This separates unavoidable Q/K/V traffic from the prototype's register,
  spill, reduction, and scheduling overhead.
- A tensor-core J1 rewrite removed spill and matched the two-stage output to
  rel-L2 about `0.00143`, but took `37.15` ms even with a four-row seed tile.
  It reloads the original K/V once per query column, so eliminating the
  compressed workspace alone is not sufficient.
- A partial-state design then fused merge plus compressed attention over small
  key-outer groups and stored only online-softmax `(max,sum,accumulator)`
  states. Packing four adjacent query columns into block-diagonal MMAs reduced
  the J1 result to `12.97` ms; a three-stage pipeline reduced it further to
  `9.940` ms. The matched retained two-stage path was `3.875` ms. Output versus
  two-stage had rel-L2 `0.002763`, MSE `5.33e-10`, and max-abs `6.64e-4`.
- The best `J4/I32/F3/4-warps/3-stages` point spent `9.804` ms in the fused
  stage and only `0.136` ms reducing seven partial states. Its workspace was
  `476,985,600` bytes per four-head chunk. J8 increased register pressure,
  I16 repeated too much merge work, and warp specialization did not help.
- Logical traffic explains why this fusion still loses. The retained
  two-stage path moves about `15.654` GB per four-head chunk, including an
  `11.360` GB compressed-K/V/bias round trip. The J4/F3 partial-state path
  removes that round trip but reloads original K/V for 13 query-column tiles,
  raising total logical traffic to about `22.356` GB. Its main stage performs
  about `1.262` TFLOP at `57.8` FLOP/byte and measured roughly `128.8` TFLOP/s
  plus `2.23` TB/s effective logical bandwidth.
- Rule: stop ordinary Triton one-pass tuning at this point. The remaining
  credible major optimization is an SM90-native intra-CTA design in which
  multiple consumer warpgroups own separate query-column accumulators while
  sharing one K/V tile. It must avoid both HBM materialization and the repeated
  original-K/V reads that make J1/J4 fusion slower.

### Wan2.1-14B Hard-C4 4x Target

- Result: the current exact key-softmax Hard C4 does not reach `4x` versus
  same-run SDPA on the 40-head, 128-dimension Wan2.1-14B model. On an idle H200
  using the first eight real QKV calls, SDPA measured `64.392` ms/call and the
  retained `head_chunk=4` Hard C4 measured `40.149` ms/call (`1.604x`). The
  dense error was rel-L2 `0.132054`, MSE `0.00224547`, max-abs `2.5510`.
- A route-contiguous compressed-K/V layout slowed the full synthetic call from
  about `39.1` to `40.2` ms. Joining K and V into one wider merge MMA slowed it
  further to `45.5` ms. Both keep the same Hard-C4 math.
- A no-intermediate ordinary-Triton kernel was finite and matched the retained
  output closely, but its best tested schedule was about `124.3` ms. Splitting
  query rows to reduce the accumulator still took `45.0` ms for only four
  active heads, because it repeated the merge and original K/V reads.
- Triton 3.7.1 failed its `PlanCTA` compiler pass for `num_ctas=2/4/8` on both
  maintained stages. A CUDA SM90 cluster/DSM prototype compiled and produced
  finite output, but repeated block/cluster barriers made the full 40-head call
  `3112` ms. It is not retained.
- Wider head launches require 64-bit temporary indexing. They were valid after
  that change but slower: `head_chunk=8/20/40` measured `41.24/42.68/42.75`
  ms on synthetic full-shape QKV, versus about `39.1` ms for the retained
  32-bit `head_chunk=4` schedule. The 64-bit index change also slowed H4 and was
  reverted.
- Rule: keep `hard_c4` with `query_outer_chunk=21`, `head_chunk=4` as the 14B
  default. Do not restore the route-contiguous, joined-KV, ordinary fused,
  multi-CTA Triton, or barrier-heavy DSM prototypes. Exceeding `4x` while
  preserving this exact algorithm requires a new SM90 warp-specialized kernel
  that uses TMA multicast/double buffering and keeps independent route
  accumulators in consumer warpgroups without HBM intermediates.

### Wan2.1-14B Hard-C4 2x Follow-Up

- Result: the exact two-stage Hard C4 remains `1.619x` versus same-run SDPA;
  none of the tested 14B-specific kernels reached the requested `2x` without
  changing precision or the algorithm. On an exclusively idle H200, one full
  Wan14B denoise trajectory recorded all 160 real-QKV calls. SDPA averaged
  `64.7963` ms/call and Hard C4 averaged `40.0181` ms/call. Hard C4 versus
  dense had rel-L2 `0.104546`, MSE `0.00345260`, and max-abs `11.4375`.
- The retained two-stage path moves about `15.654` GB per four-head chunk, or
  `156.54` GB for all 40 heads. Even treating H200's `4.8` TB/s peak HBM
  bandwidth as fully attainable gives a data-movement-only lower bound of
  about `32.61` ms. The same-run `2x` target is `32.40` ms. Therefore the
  existing merged-K/V materialization cannot reach `2x`; a winning kernel must
  remove that round trip while avoiding repeated original-K/V reads.
- Dispatch-only changes did not help. The retained `query_outer=21, head=4`
  schedule measured `38.75` ms on full synthetic 14B QKV. Matched schedules
  `query_outer=7, head=12` and `query_outer=2, head=40` measured `42.02` and
  `43.34` ms. Reusing one workspace across all ten head chunks changed
  `38.65` to only `38.60` ms.
- A key-frame-group kernel that kept the four seed-Q tiles resident slowed to
  `58.51` ms. A segmented merge/route fusion that wrote only partial outputs
  and log-sum-exp states measured `153.58` ms with BF16 partials and `147.91`
  ms with FP32 partials. The large persistent accumulators and cross-column
  MMAs outweighed its lower HBM traffic.
- PTX assembly reports `255` registers/thread, 48 KiB shared memory, and a
  small spill for the retained merge. Compiler caps of `224/192/160/128`
  registers slowed the four-head merge from `2.540` ms to
  `2.794/3.369/5.821/5.701` ms. Recomputing seed logits lowered shared memory
  to 40 KiB but still used 255 registers and produced an exact yet slightly
  slower `39.53` ms full call.
- Splitting the 64-column merge into 32- and 16-column programs measured
  `60.54` and `76.32` ms. Splitting assignment from K/V merge measured
  `55.60` ms. An intra-kernel global scratch live-range break measured
  `51.80` ms and increased spill. Casting transient seed scores to FP16 did
  not reduce the 255-register allocation, measured `39.40` ms, and changed
  the output by rel-L2 `0.00711`; it is rejected as both slower and less
  precise.
- Replacing the custom full compressed route with PyTorch fused SDPA was also
  slower. For the real 14B route shape, the broadcast-BF16-bias SDPA path
  extrapolated to `27.56` ms for 40 heads, versus about `13.3` ms for the
  maintained route kernel.
- Overlapping adjacent four-head merge and route chunks on separate CUDA
  streams did not hide either stage. On one clean H200 synthetic 14B run,
  sequential Hard C4 measured `39.34` ms/call and the exact double-buffered
  pipeline measured `39.42` ms/call; same-run SDPA was `61.09` ms/call, so
  speedup changed from `1.553x` to `1.550x`. Pipeline and sequential outputs
  were bit-identical. The two kernels compete for the same saturated device
  resources, so this orchestration is not retained.
- A Wan14B-specific tensorized merge stacked `32 query columns x 4 seeds` into
  one `128 x D` operand, reducing twelve small matrix products to three large
  products per program. After fixing token/dimension pointer broadcasting, it
  matched merged K/V exactly on the small smoke and matched routing bias to
  `2.42e-8` relative L2. On the full 14B shape, final output differed from the
  retained kernel by only `1.05e-4` relative L2, but latency increased from
  `39.33` to `71.23` ms/call. The `128 x 128` merged-K/V accumulators and a
  second query-column program caused substantially more register pressure and
  spill; reducing the number of dot instructions is not useful when it grows
  the live accumulator this much.
- Further exact scheduling probes also failed. Reordering programs to place
  `key_outer` inside `query_outer` changed the four-head call from `3.9183` to
  `3.9610` ms. Marking seed-Q loads `evict_first` changed it to `4.4715` ms.
  Pairing two seeds into one `128 x 128` merge accumulator changed merge from
  `2.5435` to `3.1246` ms and the full call to `4.4879` ms. A launch sweep
  reconfirmed `4 warps / 2 stages` as best (`2.5310` ms merge); 2 warps spilled
  catastrophically and 8 warps measured about `4.89` ms. NVIDIA's official
  Hopper CuTe FMHA on the same compressed-route shape without routing bias
  extrapolated to `13.97` ms for 40 heads, no better than the maintained
  bias-aware route at about `13.3` ms. The remaining latency is therefore in
  merge and its materialized workspace, not an inefficient route kernel.
- An exact L2-streamed design then divided the materialized workspace by
  query/key outer groups and carried FP32 online-softmax state between groups.
  It preserved the second-stage full attention and was bit-identical to the
  retained output. Small workspaces did not compensate for repeated route
  state traffic: `(query_outer_chunk, key_outer_chunk)` values `(21,3)`,
  `(21,7)`, and `(21,21)` measured `47.18`, `43.57`, and `39.56` ms/call,
  versus `39.14` ms retained. More aggressive query chunks `1/2/3/4/7` with
  one key outer at a time measured `121.27/74.62/78.29/73.25/67.70` ms. Do
  not reintroduce HBM-carried online route state; removing merged-K/V traffic
  must keep route accumulators on chip rather than replacing one round trip
  with many state round trips.
- A Hopper CUDA WMMA merge assigned one warp to each seed, kept scores,
  assignment, and merge weights in about 104 KiB shared memory, and generated
  merged K/V in `16 x 16` tiles to avoid the Triton kernel's 255-register live
  range. It compiled and was numerically valid (`2.80e-4` relative L2 versus
  retained on the 21-outer smoke), but the old warp-level WMMA API serialized
  too many small tiles: a four-head call measured `44.57` ms versus `3.914`
  ms retained. Do not replace Hopper warp-group MMA with warp-level WMMA; a
  future CUDA/CuTe implementation must use SM90 WGMMA and TMA directly.
- A direct CuTe SM90 WGMMA translation was also rejected. A minimal
  `64x128 @ 128x64` BF16-to-FP32 smoke reached `1.39e-7` relative L2 after
  adding the required async-shared proxy fence, but the exact full merge took
  `29.24` ms for one four-head chunk versus `2.54` ms for Triton. Repeated
  shared-memory staging, reductions, and barriers dominated the fast WGMMA
  instructions. The production Triton merge already emits about 100 WGMMA
  instructions and compiles to 255 registers/thread, 10 spills, and 48 KiB
  shared memory; this is a live-range/scheduling problem, not a missing Tensor
  Core path.
- Making the score and merge seed loops dynamic reduced the generated PTX from
  roughly 265k to 94k characters and removed spills, but still used 238--239
  registers/thread. Merge latency increased from `2.542` to `2.596--2.673`
  ms and full four-head latency increased from `3.917` to `3.975--4.049` ms.
  Packing the two-bit route into the low bits of the winner log-probability did
  not reduce the peak register allocation. Keep the static two-pair schedule.
- A no-intermediate SM90 `J_TILE=13` prototype assigned one query column to
  each of 13 route warps while a four-warp WGMMA group shared merge work. It
  used about 186 KiB shared memory and reduced original K/V reloads to four,
  but the 30x128 route accumulator and SIMT route dominated. On the same clean
  H200 four-head synthetic call, SDPA measured `6.144` ms, retained hard C4
  measured `3.902` ms (`1.575x`), and the prototype measured `58.379` ms
  (`0.105x`). Its first correctness version also differed from retained hard
  C4 by relative L2 `0.735`; the large performance loss makes further numerical
  debugging unjustified. A future fusion must distribute route accumulators
  across independently register-budgeted consumers and use tensor-core route
  math without dense cross-column padding.
- A transposed 720p grid reached `122.39` ms/call (`2.748x`) by changing the
  logical block to 80 query rows by 45 keys, but full-run rel-L2 increased from
  `0.12403` to `0.18232`. Fixed seed strides, per-layer stride selection, and
  limited fallbacks could not retain both `2.7x` and the original error. The
  transpose module was removed; do not trade the accepted grid geometry for
  this speedup.
- Splitting the exact 720p merge into independent query-column programs also
  failed. Full 40-head latency was `190.94/214.68/162.60` ms for 16/32/64
  query-column tiles versus `162.07` ms for the old padded kernel because each
  program reloaded K/V. Folding C into the matrix row dimension helped only
  after one runtime loop reused K/V across all five 16-column tiles; the
  candidate loop measured `119.48` ms/call and its production-path rerun
  measured `122.77` ms/call on real QKV.
- Rule: retain `attention/hard_c4_wan21_1_3b_480p_triton.py` for the 30 x 52
  Wan2.1-1.3B grid and `attention/hard_c4_wan21_14b_720p_triton.py` for the
  45 x 80 Wan2.1-14B grid, with
  `query_outer_chunk=21` and `head_chunk=4`. Do not restore the q-reuse,
  segmented-partial, score-recompute, split-assignment, transpose-grid,
  scratch, or FP16-score prototypes.

### Wan2.1-14B 720p Exact Hard-C4 3x Follow-Up

- Accepted: grouping seven query frames inside each merge program lets one
  exact 80-token K/V row serve all seven frames before it leaves the program.
  The 80 keys are split into exact 64- and 16-key tiles, K/V merging shares one
  matrix product, and the route-contiguous workspace feeds the unchanged full
  compressed attention. No grid sharing, sparse route, precision reduction,
  or algorithm change is used.
- On one clean H200 and all 160 real Wan2.1-14B 720p attention calls, same-run
  SDPA measured `336.7632` ms/call and grouped-reuse Hard C4 measured
  `105.7952` ms/call (`3.183x`). Dense-output error was rel-L2 `0.12403092`,
  MSE `0.00480631`, and max-abs `12.4570`, matching the previous exact method's
  error level. A clean synthetic production regression measured `103.3359`
  versus `332.7007` ms/call (`3.220x`) and was bit-identical to the optimized
  candidate.
- Rejected: adding per-query-outer remainder masks made query groups 7, 11,
  and 21 slower (`110.87/111.37/111.66` ms/call). Padding query group 11 to a
  divisible launch also measured `111.63` ms/call, versus about `106.25` for
  group 7 in that sweep.
- Rejected: splitting K and V merge work into concurrent streams duplicated
  assignment work and increased the exact full call from `105.06` to
  `138.64` ms/call. FP16 route accumulation did not compile and would also
  violate the precision-preservation requirement.
- Current rule: retain the grouped-reuse exact 720p kernel with query group 7,
  `query_outer_chunk=21`, and `head_chunk=8`. Keep key-dimension softmax
  assignment, grid-local compression, entropy routing bias, and the complete
  second-stage attention unchanged.
- A complete same-prompt/seed 50-step run recorded 2,000 calls per backend.
  SDPA measured `337.0198` ms/call and `878.6253` seconds/video; Hard C4
  measured `109.6744` ms/call and `425.7931` seconds/video. This is `3.073x`
  attention speedup and `2.064x` end-to-end speedup, excluding model loading.

### Wan2.1-14B 720p Exact Hard-C4 3.5x Follow-Up

- A `16 query columns x 16 key blocks` blocked intermediate layout was exact
  but neutral. On a clean H200 it measured `105.66` ms/call versus `105.44`
  ms/call for the route-contiguous layout; same-run SDPA was `330.35` ms/call.
  Merge improved only from `71.10` to `70.81` ms while route lost the gain.
- Pairwise seed-score residency (`2+2` tensorized seeds) was bitwise identical
  to the four-seed tensor, but the smaller `M=32` matrix shape reduced Hopper
  Tensor Core efficiency. It measured `150.61` ms/call versus `104.54`
  ms/call for production; same-run SDPA was `332.79` ms/call.
- An FP16 merged-K/V accumulator produced bitwise-identical BF16 output in the
  synthetic check, but changed latency only from `104.92` to `104.57` ms/call.
  It does not remove the merge kernel's dominant work and is not retained.
- An FP16 route output accumulator, with FP32 online-softmax state preserved,
  was both slower and less accurate: `105.50` versus `104.41` ms/call, with
  relative L2 `0.00223` versus the FP32 route output. Keep FP32 route
  accumulation.
- Storing only the routing bias before the merged-K/V dot was exact but
  neutral: `105.76` versus `105.60` ms/call, with merge `71.37` versus
  `71.20` ms. The compiler does not benefit from this source-level lifetime
  change.
- Nsight Compute hardware-counter profiling is unavailable to unprivileged
  users on `scai7` (`ERR_NVGPUCTRPERM`). Continue to use clean same-run stage
  timing, compiler resource metadata, and numerical checks for decisions.
- Accepted: process all 21 query frames per merge program, pipeline that loop
  with two stages, and launch route with one stage. On a clean H200 with
  Triton 3.7.1, the previous production kernel measured `101.8657` ms/call
  (`3.2605x` vs same-run SDPA), while the exact candidate measured `101.7201`
  ms/call (`3.2652x`). Same-run SDPA was `332.1315` ms/call, and candidate
  output was bitwise identical to production (`rel-L2=0`, `max-abs=0`). The
  gain is small but positive under the required same-run A/B, so this schedule
  is the maintained 720p default.
- Rejected: splitting the 40 heads across two clean H200s. Including Q/K/V P2P
  copies and output copy-back, two GPUs measured `103.9848` ms/call versus
  `101.0045` ms/call for one GPU. Both outputs were bitwise identical, but the
  two-GPU path reached only `3.1530x` versus SDPA, so transfer and coordination
  overhead outweighed head parallelism.
- Rejected: replacing the exact online maximum with a fixed log2 shift of 64.
  A synthetic random-QKV check looked finite and reached about `3.40x`, but a
  full four-step Wan2.1-14B 720p run exposed zero softmax denominators and NaN
  outputs. Across 320 CFG attention calls, same-run SDPA measured `332.2890`
  ms/call, exact production measured `101.9779` ms/call (`3.2584x`), and the
  invalid fixed-shift candidate measured `97.7007` ms/call (`3.4011x`). Keep
  the online maximum; synthetic finite-output checks are insufficient for
  validating softmax range approximations.

### Wan2.1-14B 720p Merge Residency Versus V Reload

- Rejected: defer every V tile load until after the merged-K dot for each
  query-column tile. This was bitwise identical and, with one launch stage,
  reduced compiler-reported shared memory from `77,824` to `40,960` bytes,
  but it reloads the same 80 x 128 V tile for every query tile.
- On an exclusively idle H200 with the production
  `Q_OUTER=K_OUTER=21`, `HEADS=10`, `D=128` shape, the retained
  `4 warps / 2 stages` merge measured `16.741` ms. Delayed V measured
  `17.351` ms with two stages and `20.758` ms with one stage. The one-stage
  candidate also increased spills from 12 to 50.
- Rule: keep K and V resident together and load each original K/V row once
  per merge program. Higher nominal CTA residency is not useful when it is
  purchased by repeated original-V traffic and spill.

### Wan2.1-14B 720p Head-Chunk Dispatch Recheck

- Rejected as noise: changing the pure 40-head token-output dispatch from
  `head_chunk=10` to 4, 5, 8, or 20. Every output was bitwise identical.
- The initial clean-H200 sweep measured `98.847` ms for chunk 5 versus
  `99.448` ms for chunk 10. A longer ABBA confirmation reversed the order:
  chunk 10 measured `100.904/101.174` ms and chunk 5 measured
  `101.207/101.348` ms. Same-run SDPA was `327.080` ms.
- Rule: retain `head_chunk=10` for the current pure 40-head fast runner; the
  sub-percent ordering is not stable enough to justify extra launches.

### Wan2.1-14B 720p Merge Warp Specialization

- Rejected as unsupported by Triton 3.7.1: enabling `warp_specialize=True`
  on the 21-frame merge loop failed in `NVGPUWarpSpecialization` with
  `unsupported load type for producer commit`. The seed-Q gather is not a
  producer load shape accepted by the current pass.
- The unchanged baseline still compiled at 255 registers/thread, 12 spills,
  and 102,400 bytes of shared memory and measured `16.739` ms for ten active
  heads. Do not retain an unused specialization flag in production.

### Wan2.1-14B 720p Merge LICM Scheduling

- Accepted for full-attention confirmation: set `disable_licm=True` on the
  21-frame merge loop. This changes only Triton scheduling; merged K/V/bias
  are bitwise identical.
- On an exclusively idle H200 with ten active heads, the first sweep changed
  merge from `16.747` to `16.500` ms. A 50-repeat ABBA confirmation measured
  baseline `16.764/16.798` ms and the candidate `16.527/16.530` ms, a stable
  reduction of about `1.52%`. Registers stayed at 255 and shared memory at
  102,400 bytes, while spills fell from 12 to 8.
- Complete 40-head attention then measured baseline `100.885/101.009` ms and
  the candidate `100.285/100.193` ms in ABBA order. Output was bitwise equal;
  same-run SDPA was `326.591` ms, so speedup improved from about `3.235x` to
  `3.258x`.
- A real Wan2.1-14B 720p 50-step video recorded core attention at `110.483`
  ms/call, full self-attention at `134.623` ms/call, and inference at
  `756.759` seconds for 4,000 calls. Versus the prior matching fast record
  (`110.793`, `134.873`, `755.718`), attention improved slightly while the
  single-video E2E difference was noise-level and not an E2E speedup claim.

### Wan2.1-14B 720p Route LICM Scheduling

- Rejected: applying `disable_licm=True` to the compressed-attention loop.
  Output was bitwise identical, but the baseline already compiled with zero
  spills, 185 registers/thread, and 115,200 bytes of shared memory.
- On an exclusively idle H200 with ten active heads, ABBA route timings were
  `7.656/7.714` ms for baseline and `7.881/8.094` ms with LICM disabled.
  Keep the compiler's default loop-invariant-code motion for route.

### Wan2.1-14B 720p Route Row Tiling

- Rejected: split the 45 query rows into two `BLOCK_I=32` programs to reduce
  per-CTA shared memory. It reduced shared memory from 115,200 to 78,336 bytes
  and registers/thread from 197 to 142, but each program reread all compressed
  K/V tokens.
- On an exclusively idle H200 with ten active heads, route increased from
  `7.609/7.876` ms at `BLOCK_I=64` to `14.297/14.238` ms at 32. The different
  MMA tile also was not bitwise identical. Keep one 64-row route program.

### Wan2.1-14B 720p Nested Merge LICM

- Rejected: additionally disable LICM on the inner five query-column tiles.
  This removed the remaining eight compiler-reported spills and kept output
  bitwise identical, but ABBA merge timings moved from `16.511/16.513` ms to
  `16.543/16.552` ms. Keep LICM disabled only on the outer 21-frame loop.

### Wan2.1-14B 720p Merge Accumulator Buffering

- Rejected as neutral: combine the accepted outer-loop LICM setting with
  `disallow_acc_multi_buffer=True`. Output remained bitwise identical and both
  variants compiled to the same 255 registers, eight spills, and 102,400 bytes
  of shared memory.
- ABBA merge timings were `16.510/16.577` ms without the hint and
  `16.530/16.532` ms with it. The ordering is noise-level, so keep the simpler
  compiler default.

### Wan2.1-14B 720p Merge Saturation And Addressing Probes

- Long-duration sampling on one exclusively idle H200 measured the maintained
  ten-head merge at `16.562` ms. During the steady interval, `nvidia-smi dmon`
  reported `100%` SM utilization, only `56-57%` memory-controller utilization,
  and about `690-692` W package power. The server blocks privileged Nsight
  counters, so the memory percentage is not a byte-accurate HBM measurement,
  but it rejects the assumption that this kernel is simply saturating HBM. The
  current limit includes high register pressure, reductions, and on-chip data
  movement.
- Rejected: compute routing bias before materializing the BF16 merge weights to
  shorten their FP32 live range. Merged K/V/bias were bitwise identical and
  compiler resources stayed at `102400` shared bytes, `255` registers/thread,
  and `8` spills. Four ABBA samples averaged `16.570` ms for production and
  `16.598` ms for the candidate.
- Rejected: reverse merge key-row/key-outer CTA order so route consumes the
  most recently written logical prefix first. Complete outputs were bitwise
  identical, but full 40-head attention averaged `100.318` ms in production
  and `100.367` ms with reversed scheduling. The L2-resident tail is too small
  relative to the materialized workspace.
- Rejected: pad the exact `64+16` key split to `64+32` or `64+64` in search of a
  more efficient WGMMA tail. The exact 16-wide tail measured `16.506` ms;
  32-wide and 64-wide tails measured `17.908` and `35.032` ms, increased spills,
  and changed MMA rounding.
- Rejected: shrink head chunks enough to use 32-bit workspace element offsets.
  Full attention measured `100.264` ms for the maintained ten-head/int64 path,
  `103.036` ms for two-head/int64, and `103.514` ms for two-head/int32. Computing
  token indices in int32 and widening only before the `D=128` multiplication
  was bitwise identical but also slower: four ABBA samples averaged `100.297`
  versus `101.886` ms.
- Keep the outer-loop LICM optimization as the only promoted change from these
  scheduling experiments. Future substantial gains need to reduce the
  reduction/register cost or remove the compressed workspace without reloading
  original K/V, rather than adding more HBM-oriented cache hints.

### Wan2.1-14B 720p C4 Fixed Partitions And Query Means

- Rejected: replacing content-based hard assignment with four fixed 20-key
  partitions. On AMD MI300X over all 40 layers, all four denoise steps, and
  heads 0/10/20/30, contiguous and interleaved partitions measured rel-L2
  `0.25697` and `0.22248`, versus `0.14807` for the maintained C4 assignment.
- Rejected: replacing local seed queries with four query-row segment means.
  Quartile means measured rel-L2 `0.31692`; retaining the index-matched query
  and adding three tertile means still measured `0.21550`. RoPE-positioned
  queries cannot be averaged across distant rows without losing the local
  geometry needed by the merge.
- Rule: keep exactly four compressed tokens, content-based key assignment,
  local position-preserving seeds, grid-local compression, entropy bias, and
  full second-stage attention. Do not implement fixed key partitions or
  global query-mean seeds in a production kernel.

### Wan2.1-14B 720p C4 Balanced Assignment

- Rejected: one to eight rounds of Sinkhorn score normalization before the
  same hard C4 argmax. Additional rounds reduced cluster-count CV from about
  `0.62` to `0.18`, but increased rel-L2 from `0.14807` to `0.15244--0.15464`.
  Equal-sized clusters are not the right objective because real attention
  mass is naturally uneven. Do not add Sinkhorn work to the production merge.

### Wan2.1-14B 720p C4 Scalar And Spatial Corrections

- Rejected as insufficient: a query-dependent diagonal key-variance
  correction. The best global scale reduced wide-seed rel-L2 only from
  `0.14579` to `0.14505`; larger scales became unstable and much worse.
- Rejected: adding a fixed spatial-continuity penalty to content-based key
  assignment. Every tested strength was neutral or worse. Key columns should
  not be tied to the ordered seed-row index by an artificial geometric prior.
- Rule: do not promote scalar curvature or spatial-prior tuning as an accuracy
  solution. A useful C4 correction must also model how the merged value changes
  with the query, not only alter routing mass.

### Wan2.1-14B 720p C4 Cosine Assignment

- Rejected: assigning each key to the seed with maximum cosine similarity.
  On the matched MI300X screen (prompt 0, all four denoise steps, layers
  0/13/26/39, heads 0/20), cosine assignment measured rel-L2 `0.14755` and
  MSE `0.003889`. Raw-dot assignment measured `0.15040` and `0.004041`, while
  the maintained key-softmax assignment measured `0.13539` and `0.003274`.
- Cosine reduced raw-dot error slightly, but left `21.36%` of clusters empty
  versus `0.78%` for key-softmax. Q/K normalization already makes token norms
  similar, so cosine normalization does not provide the seed-wise partition
  calibration supplied by the key-softmax normalizer.
- Rule: retain key-softmax assignment. Do not run the cosine variant through
  broader validation unless the assignment objective changes materially.

### Wan2.1-14B 720p C4 Dual-Probe Assignment

- Rejected: representing each of the four clusters with two fixed neighboring
  query probes while still emitting exactly four compressed K/V tokens. The
  probe pair was combined by averaging its key-softmax probabilities in log
  space, so no RoPE-positioned queries were averaged directly.
- On the matched MI300X 16-call screen, using the dual-probe score only for
  assignment measured rel-L2 `0.13901` and MSE `0.003452`, versus `0.13539`
  and `0.003274` for the maintained wide-seed key-softmax method. Using the
  mixture score for both assignment and merge was substantially worse at
  rel-L2 `0.16623` and MSE `0.004936`.
- Dual probes also doubled seed-key dot products from four to eight per block
  and increased the empty-cluster fraction from `0.78%` to `3.44%`. The route
  still had four tokens, but the extra merge work bought no accuracy benefit.
  Do not implement this direction in Triton or run broader validation.

### Wan2.1-14B 720p SAP Shuffle Before Hard C4

- Rejected after a real-QKV screening call (step 0, layer 0, head 0). Using
  SVG-2/SAP Euclidean clustering to permute Q and K/V separately, then applying
  the same-density fixed `45 x 80` Hard C4 grid, increased rel-L2 from
  `0.04441` to `0.55346` and MSE from `0.000660` to `0.102477`.
- Treating each of SAP's 300 query clusters and 1000 key clusters directly as
  a variable semantic grid was worse: rel-L2 `0.81692`, MSE `0.223263`. Its
  query clusters average 252 tokens, so one set of four seeds serves about
  5.6 times as many queries as the maintained 45-query grid.
- SAP uses semantic permutation to make selected sparse attention blocks
  contiguous; it does not imply that every semantic key block can be averaged
  into four K/V tokens. Hard C4 depends on its position-preserving query and
  key grids. Do not build a production kernel or broad validation around a
  full SAP Q/K shuffle unless the compression objective is redesigned.

### Wan2.1-14B 720p Processor And Runtime Ablation

- Rejected: treating the SVG Python/Torch/Triton environment as a Hard C4
  optimization. On one clean H200 with identical synthetic 720p QKV
  (`[1, 8, 75600, 128]`, five warmups, 30 repeats), the compatibility C4
  kernel measured `19.9886` ms in `monarch_rt` (Torch 2.8, Triton 3.7.1) but
  `47.6891` ms in the SVG environment (Torch 2.6, Triton 3.2.0), a `2.386x`
  regression. Same-shape SDPA was slightly faster in SVG (`65.3495` versus
  `66.3577` ms), so the regression is specific to the custom C4 compilation.
- Rejected as a speed optimization: source-only C4 specialization. Fixing C4
  shapes and removing the unused C8 branch changed `19.8620` to `20.0881` ms
  in `monarch_rt` (`1.14%` slower). Keep only where it improves readability.
- Compatibility only: removing `disallow_acc_multi_buffer=True`. Triton 3.2
  cannot compile the old kernel with this argument. In Triton 3.7.1, the
  before/after complete-kernel values (`20.0881` and `19.9886` ms) differ by
  less than `0.5%`; do not claim a performance gain from this edit.
- Accuracy tradeoff, not a speed optimization: wide seeds changed the
  complete kernel from `19.9886` to `20.0176` ms in `monarch_rt` and from
  `47.6891` to `47.8612` ms in SVG. Retain them because earlier real-QKV
  validation reduced MSE by `4.84--6.40%`, not because they are faster.
- Accepted independently: the SVG QK RMSNorm and RoPE kernels. At full Wan
  720p shape, QK RMSNorm changed `5.0978` to `0.3644` ms (`13.99x`) and RoPE
  changed `8.9069` to `0.8051` ms (`11.06x`), with relative L2 around
  `1.4e-5--1.6e-5`. In a matched five-step video run, enabling both reduced
  stable full self-attention from `303.1037` to `286.6075` ms/call and E2E
  from `160.7109` to `156.8724` seconds/video.
- Accepted independently: the SVG transformer-block elementwise kernels.
  LayerNorm, modulation, and gate-residual microbenchmarks reached `3.77x`,
  `3.73x`, and `7.81x`. Adding the SVG transformer forward to the matched
  five-step run reduced stable E2E from `156.8724` to `144.3506`
  seconds/video, although core attention remained statistically unchanged.
- Instrumentation only: a second nested CUDA-event timer added about
  `0.0501` ms per measured call (`0.20` seconds over 4,000 calls) in the
  event-overhead screen. It is small, but it is not an optimization.

### Wan2.1-14B 720p Exact QKV Preprocess And RoPE Fusion

- Accepted: fuse Q/K RMSNorm, complex RoPE, Q/K/V head-major layout conversion,
  and the final stores into one Triton path. The kernel preserves the existing
  SVG numerical order: normalized Q/K are first rounded to BF16, RoPE uses the
  same FP64 arithmetic, and rotated Q/K are rounded to BF16 on store. Q, K, V,
  and the complete GMA output were bitwise identical to the maintained path.
- The best full-shape launch uses four tokens per CTA and one warp. On one
  exclusively idle H200, preprocessing plus RoPE decreased from `2.7290` to
  `1.7810` ms (`1.532x`). The complete preprocess-plus-GMA ABBA mean decreased
  from `102.9180` to `102.1673` ms/call, saving `0.7507` ms/call (`1.00735x`).
  The same-run GMA core was `100.3511` ms/call versus SDPA at `326.5276`
  ms/call, or `3.2539x` faster.
- The component gain is larger than the complete-attention gain because merge
  and route dominate the remaining latency. Keep the fusion as an exact
  engineering optimization; it does not change C4 assignment or attention
  math.
- A clean one-video Wan2.1-14B 720p, 81-frame, 50-step run measured core GMA
  at `110.4709` ms/call, full self-attention at `133.5676` ms/call, and
  inference at `751.8179` seconds for 4,000 calls. The prior matching record
  was `110.483`, `134.623`, and `756.759`. Core GMA was unchanged as expected;
  full attention saved `1.0554` ms/call, or `4.22` seconds across 4,000 calls,
  while observed E2E improved by `4.94` seconds (`0.65%`). Treat the E2E value
  as a one-video confirmation, not a multi-video throughput estimate.

### Wan2.1-14B 720p Merge Tile And Workspace Index Probes

- Rejected: reduce the merge query-column tile from the maintained 16 columns
  to 8 or 4. Shared memory fell from `102400` bytes to `73728` or `69632`
  bytes and spills disappeared, but merge latency increased from `16.522` ms
  to `37.300` or `54.301` ms. The smaller MMA shapes were also not bitwise
  identical. Register or shared-memory reductions do not compensate for the
  lost tensor-core throughput.
- Rejected: use two-head chunks so workspace offsets fit in 32 bits. Complete
  40-head attention increased from about `100.05` ms with ten-head chunks to
  `103.49` ms with two-head/int32 chunks, despite bitwise-identical output.
  Same-run SDPA was `325.393` ms. Keep ten-head chunks; the 20-launch overhead
  is larger than the address-arithmetic saving.

### Wan2.1-14B 720p Merge Seed And Workspace Layout Probes

- Rejected: load the four seed-query rows in seed-major order and explicitly
  permute them back to the maintained query-column-major tensor layout. The
  merged K/V and entropy bias were bitwise identical, but the ten-head merge
  increased from `16.522/16.527` ms to `69.502/69.434` ms. Compiler-reported
  shared memory grew from `102400` to `126976` bytes and spills from 8 to 18.
- Rejected: keep the seed-major score tensor throughout assignment and
  reduction, avoiding the explicit permute. This also remained bitwise exact,
  but measured `66.140/66.238` ms versus `16.499/16.527` ms. It still required
  `126976` bytes of shared memory and 10 spills. Triton's axis-0 seed reduction
  needs an expensive layout conversion; the maintained interleaved
  `[query_column, seed]` layout is substantially better for this WGMMA shape.
- Rejected: store merged K and V as one interleaved workspace directly from the
  joint merge MMA, then have route load K/V at stride two. The final 40-head
  attention output was bitwise identical, but latency increased from
  `100.051/100.334` ms to `330.490/330.525` ms. Same-run SDPA was `325.988` ms.
  Eliminating the split and one allocation does not compensate for destroying
  route-contiguous K/V loads and the route kernel's WGMMA-friendly layout.

### Wan2.1-14B 720p Merge Query-Tile Scheduling Hints

- Rejected: set an explicit `loop_unroll_factor` on the five query-column
  tiles. All runnable variants were bitwise identical. In an eight-sample ABBA
  confirmation, factor 1 averaged `16.631` ms versus `16.617` ms for the
  compiler default and compiled to the same 102,400 shared bytes, 255
  registers/thread, and eight spills. Factor 2 increased latency to `26.860`
  ms with 135,168 shared bytes and 22 spills; factor 4 required 233,472 shared
  bytes, above the H200 limit of 232,448; factor 8 measured `17.436` ms with
  77,824 shared bytes and 24 spills. Keep the compiler's default unroll policy.
- Rejected: explicitly hoist the loop-invariant `tl.interleave(K, V)` layout
  conversion outside all 105 merge tiles. With the production two launch
  stages it also required 233,472 shared bytes and could not launch. Reducing
  to one stage made it bitwise exact and compile to the same resources as the
  one-stage baseline, but ABBA averages were `17.527` ms with the explicit
  hoist and `17.509` ms without it. The one-stage baseline itself was about
  5.4% slower than the maintained two-stage kernel, so neither change is kept.

- Rejected: explicitly unroll the outer 21-frame query loop. Every variant was
  bitwise identical, but factor 1 averaged `16.548` ms versus `16.528` ms for
  the compiler default. Factors 2, 4, and 8 increased merge latency to about
  `19.439`, `21.418`, and `25.027` ms while compiler-reported spills rose from
  8 to 100, 184, and 272. Shared memory remained 102,400 bytes. Keep the outer
  loop dynamic with the compiler's default unroll policy.
- Rejected: flatten all 105 merge tiles into one loop, including a
  query-column-major traversal that processes all 21 frames before moving to
  the next 16-column tile. Both flattened orders produced bitwise-identical
  merged K/V and entropy bias and removed the eight reported spills, but the
  maintained nested loop averaged `16.591` ms in the eight-sample run. The
  original frame-major flat order averaged `17.360` ms and the new
  query-column-major order averaged `17.534` ms. The extra quotient/remainder
  indexing and flatter schedule cost more than eliminating the spills.
- Rejected: calculate one scalar 64-bit workspace base per 16-column tile and
  add fixed column/seed strides instead of forming the vectorized logical
  indices directly. Merged K/V and bias were bitwise identical, and reported
  spills fell from eight to two, but eight-sample averages increased from
  `16.596` ms to `16.762` ms. The compiler's existing vector address lowering
  is faster than the explicit scalar-base decomposition.
- Rejected: increase the current 64-token route loop's software pipeline from
  three to four stages. Both produced bitwise-identical final output. On the
  ten-head 720p route shape, stage 3 averaged `7.875` ms while stage 4 averaged
  `8.904` ms and raised shared memory from 115,200 to 148,224 bytes. The
  one-head smoke also showed stages 1 and 2 behind stage 3, so retain three.
- Rejected as noise: explicitly keep only the 64-key main interleave, only the
  16-key tail interleave, or both K/V interleaves outside the nested merge
  loops. All variants produced bitwise-identical merged K/V/bias and compiled
  with the same 102,400 shared bytes, 255 registers, and eight spills. The
  tail-only ABBA averages were `16.609` ms for production and `16.606` ms for
  the candidate; main-only changed `16.668` to `16.673` ms, and both changed
  `16.780` to `16.773` ms. These `0.015--0.045%` shifts are noise, so keep the
  simpler expressions at their point of use.

### Wan2.1-14B 720p Score-Quantile C4 Merge

- Rejected: use the row-matched query, or its average with the left neighbor,
  to rank each 80-key grid and split it into four equal-count or equal-attention-
  mass groups. The same one or two probe logits supplied the within-group
  softmax weights, while entropy bias and full second-stage attention stayed
  unchanged.
- Across three prompts, four denoise steps, layers 0/13/26/39, and heads 0/20,
  the best equal-mass variant measured rel-L2 `0.17010` and MSE `0.006051`.
  The matched wide-seed Hard C4 reference measured `0.12269` and `0.003149`.
  Equal-count variants were worse (`0.19227` rel-L2 or higher).
- Although one- and two-probe variants use only 25% and 50% of the original
  four seed-key dot products, a scalar key ranking discards how keys respond
  to other local query directions. Do not revisit scalar score quantiles unless
  the grouping retains a multi-query response profile.

### Wan2.1-14B 720p Four-Response Profile Clustering

- Rejected: represent every key by its four existing seed-query logits and
  cluster those four-dimensional profiles into four groups using deterministic
  farthest-point initialization, optionally followed by a Lloyd update. This
  reused the original C4 QK products and kept entropy bias and full route.
- Across the same three-prompt 16-call screen, the best cluster-adaptive probe
  mixture measured rel-L2 `0.16418` and MSE `0.005638`, versus `0.12269` and
  `0.003149` for wide-seed Hard C4. Uniform, mean-probe, owner-probe, and
  z-scored profile variants were all worse.
- Similar response vectors do not necessarily form a good log-sum-exp tangent
  group. Preserve owner-based hard assignment, where each key belongs to the
  seed with the strongest normalized response, unless a new objective directly
  models the dense attention output.

### Wan2.1-14B 720p Local Query-Centroid Seeds

- Rejected as insufficient: initialize four wide local seeds, assign 11--21
  neighboring query rows to them in query space, and replace each seed with
  its one-step centroid before the unchanged C4 merge. Both raw centroids and
  norm-preserving centroids were tested, including two Lloyd steps.
- The best norm-preserving C15 centroid plus `1.05` entropy-bias scale measured
  rel-L2 `0.12103` and MSE `0.003064`, only `2.69%` lower MSE than the wide-seed
  reference (`0.12269`, `0.003149`) and much worse than discrete adaptive seeds.
- Averaging RoPE-positioned query vectors weakens useful directional structure.
  Keep actual query rows as C4 anchors rather than replacing them with local
  centroids.

### Wan2.1-14B 720p C4 Norm And Seed-Blend Corrections

- Rejected: bias the within-cluster merge logits by a global multiple of
  `log ||V_l||`. On top of the best sampled-key-response seeds, all tested
  negative and positive scales were neutral or worse; the closest positive
  scale increased MSE from `0.002073` to `0.002085`.
- Rejected: rescale each merged key toward the weighted RMS norm of its source
  keys. Even a quarter-power correction increased MSE by `21.31%`, and stronger
  corrections degraded rapidly. The norm contraction of a softmax key centroid
  is part of the correct first-order log-sum-exp tangent, not missing energy.
- Insufficient: move selected real query seeds 10--50% toward their local query
  centroids while preserving norm. The best 10% blend changed rel-L2 only from
  `0.09955` to `0.09943` and MSE from `0.002073` to `0.002068` on the screen.
  This `0.24%` MSE change is too small for broad validation or production.

### Wan2.1-14B 720p Exact Proxy-Cover Seed Selection

- Rejected: after fixing the row-matched query and its left neighbor, enumerate
  all pairs for the remaining two seeds and minimize nearest-seed distance in a
  sampled key-response space. The search used 15 local candidate query rows and
  either 16 or 32 sampled keys, while the final C4 merge, entropy bias, and full
  second-stage attention were unchanged.
- Across three prompts, four denoise steps, layers 0/13/26/39, and heads 0/20,
  the best exact-cover variant measured rel-L2 `0.10374` and MSE `0.002251`.
  Greedy response-farthest measured `0.09955`/`0.002073`, and the best
  response-span rule measured `0.09882`/`0.002042` on the same QKV calls.
- Minimizing proxy-space coverage is not aligned closely enough with final
  dense-output error. Do not repeat exact medoid/coverage enumeration as a seed
  objective; retain response span or farthest selection as the low-cost proxy
  baselines.

### Wan2.1-14B 720p Block-Local Seed Selection Diagnostics

- Accepted as a quality diagnostic: select four real query rows independently
  for every key block by spanning sampled key-response profiles. Across three
  prompts, all 40 layers, all four denoise steps, and heads 0/10/20/30, the
  best 31-candidate multi-anchor rule measured rel-L2 `0.10006` and MSE
  `0.001918`, versus `0.13193` and `0.003334` for fixed wide seeds. It is not
  yet a production default because its PyTorch selector computes more probe
  responses than the maintained Triton merge.
- Rejected: choose new seeds from responses to a provisional four-key C4
  merge. Although this pilot-refine rule reduced candidate search to only four
  provisional keys, its best three-prompt screen measured rel-L2 `0.09039`
  and MSE `0.001709`, worse than the sampled block-response selector
  (`0.07773` and `0.001264`) on the same 16-call screen.
- Rejected: use every one of the 80 keys to form local attention-times-value
  outputs before selecting seeds. Full-key span selection measured rel-L2
  `0.08073` and MSE `0.001363`, while the 16-probe value-output rule measured
  `0.07683` and `0.001234`. More probes increase QK work but do not improve
  the seed objective.
- Rejected: unanchored value-output farthest selection. Removing the fixed
  row-matched query increased rel-L2 to `0.10671`; retaining one local anchor
  is important even when later seeds are selected globally.

### Wan2.1-14B 720p Value-Aware C4 Merge

- Promising diagnostic, not yet production: after the ordinary key-softmax
  assignment, form one provisional merged value and add a globally scaled,
  cluster-normalized negative value-distance to the assignment score. This
  changes no route tokens and adds no seed-key dot products. A scale near
  `0.20--0.25` reduced the best three-prompt screen from rel-L2 `0.07773` and
  MSE `0.001264` to about `0.07677` and `0.001233`.
- Combining value-output seed selection with value-distance reassignment
  reached rel-L2 `0.07594` and MSE `0.001206` on the same screen. Mixing each
  owner distribution with a small cluster-mass-weighted combination of all
  four existing seed distributions improved the screen further to rel-L2
  `0.07412` and MSE `0.001149`. This reuses the existing C4 logits and keeps
  entropy bias plus full second-stage attention.
- The adaptive mixture remained beneficial across three prompts, all 40
  layers, all four denoise steps, and heads 0/10/20/30. It measured rel-L2
  `0.09682` and MSE `0.001796`, compared with `0.10006` and `0.001918` for the
  same block-response seeds without value-aware merge changes. The broader
  reduction is `3.2%` in rel-L2 and `6.4%` in MSE, so the screen improvement
  is real but not large enough to approach the SVG-EAR quality target.
- Do not promote these merge changes until broad real-QKV validation and a
  fused-kernel runtime comparison show that the extra value reduction and
  `4 x 4 x 80` mixture do not erase the SDPA speedup.

### Wan2.1-14B 720p Anchor-Free Portfolio And Mean-Query Merge

- Rejected after broad validation: compare the anchored C4 span proposal with
  an unanchored proposal in every grid using a four-channel pooled-value proxy,
  then add `0.2` times the already available value-similarity score to the
  within-cluster merge logits. The three-prompt screen improved rel-L2 from
  `0.065776` to `0.064846` and MSE from `0.0009080` to `0.0008832`.
- The apparent gain did not generalize. Across three prompts, all 40 layers,
  all four denoise steps, and heads 0/10/20/30, the maintained anchored method
  measured rel-L2 `0.089400` and MSE `0.00151195`; the portfolio measured
  `0.089799` and `0.00152479`, worsening MSE by `0.85%`. Do not promote a
  per-grid anchored/free selector from a small layer/head screen.
- Rejected: average all 45 queries in a query column, compute one additional
  mean-query/key product, and interpolate it into the merge logits. Even a
  `0.10` interpolation worsened the screen, while `0.50` increased MSE by
  `15.2%`. Averaging queries destroys multimodal softmax structure and is not
  a useful replacement for seed-conditioned merge weights.

### Wan2.1-14B 720p Response-Sketch Value Fitting

- Rejected: reuse the sampled-key value responses from response-cover seed
  selection as additional target equations for the four merged values. This
  adds no query-to-original-key dot products, keeps the entropy bias, and
  leaves the full second-stage attention unchanged.
- On prompt 0 with all four denoise steps, layers 0/13/26/39, and heads 0/20,
  the compact response-cover reference measured rel-L2 `0.06871` and MSE
  `0.0009243`. Even the lightest eight-query sketch at weight `0.10` degraded
  to `0.10247` and `0.0020557`; larger sketch counts or weights were much
  worse.
- Sampled-key responses are useful for ranking query coverage but are not
  accurate full-key attention-output targets. Do not reuse them directly in
  the merged-value regression.

### Wan2.1-14B 720p Adaptive Value Targets And Key Log-Mass Fitting

- Rejected: select exact additional value-regression queries from the complete
  45-row query block using either compressed-route entropy or farthest-point
  coverage. Four adaptive targets reduced the three-prompt MSE by only `0.38%`;
  increasing the count to 8 or 16 did not improve the prompt-0 result. The
  extra full-key query products are not justified by this gain.
- Rejected: remove 25--100% of the peak sharpening from value weights while
  retaining it for merged keys. Results were numerically neutral or slightly
  worse because the subsequent value regression already absorbs the initial
  value-weight choice.
- Rejected: fit each merged key to exact per-cluster log-partition values at
  the four existing seeds or at those seeds plus four existing anchors. On
  prompt 0, four denoise steps, layers 0/13/26/39, and heads 0/20, the reference
  measured rel-L2 `0.0635433` and MSE `0.000790534`. The lightest correction
  measured `0.0635905` and `0.000791709`; stronger corrections worsened MSE by
  up to `20.7%`. A multi-anchor log-mass fit harms unseen-query generalization
  even though it adds no seed-key products, so do not extend this direction.
- Insufficient: regularize the fitted merged value toward the cluster value
  distribution induced by the log-mean-exp of the four existing seed logits.
  A 10% mix consistently reduced the three-prompt MSE from `0.000907986` to
  `0.000905106`, only `0.32%`, while requiring another value aggregation.
- Rejected: project the unconstrained fitted merged value back into each
  cluster's value convex hull using Euclidean Frank--Wolfe steps. One step
  increased MSE by `69%`, and additional steps were worse. Euclidean value
  distance is not the correct metric for final attention-output quality.
- Rejected: run Frank--Wolfe directly against the existing local probe-output
  loss. On prompt 0, even the best tested setting increased MSE from
  `0.000790534` to `0.00109749`. Improving the conditional 80-key probe target
  does not reliably improve the final global routed output.

### Wan2.1-14B 720p Uniform-Key Global Value Calibration

- Rejected: approximate the dense targets for 16 route-selected calibration
  queries by uniformly sampling only 8, 16, 24, or 32 of each grid's 80
  original keys. This reduced the added QK/PV work from `16/45 = 35.6%` of
  dense attention to between `3.6%` and `14.2%`, while keeping C4 and the full
  second-stage route unchanged.
- On prompt 0 over four denoise steps, layers 0/13/26/39, and heads 0/20, the
  full-key calibration measured rel-L2 `0.04540` and MSE `0.0004036`. Uniform
  8/16/24/32-key targets instead measured rel-L2
  `0.65994`/`0.62989`/`0.57045`/`0.49885`; even 32 samples were unusable.
- Global softmax output is not preserved by deterministic uniform key
  quadrature: omitted high-logit keys distort both the partition function and
  value numerator. Do not repeat uniform per-grid key sampling for calibration
  targets without a principled importance-weighted estimator.

### Wan2.1-14B 720p C4-Proposal Importance Calibration

- Rejected: use the expanded C4 attention distribution as an importance
  proposal, sample 64/128/256/512 original keys for each of 16 route-selected
  calibration queries, and fit the four merged values to those sampled dense
  targets. This keeps C=4 and the full route stage, and adds at most about
  `0.24%` of dense vector QK/PV work at 512 samples.
- Across three prompts, four denoise steps, layers 0/13/26/39, and heads 0/20,
  exact full-key calibration measured rel-L2 `0.04726` and MSE `0.00046885`.
  Importance sampling with 64/128/256/512 samples measured rel-L2
  `0.37002`/`0.27593`/`0.21117`/`0.16419` and MSE
  `0.02860`/`0.01591`/`0.00932`/`0.00564`.
- The approximate C4 proposal misses enough high-logit original keys that the
  self-normalized estimator has prohibitive variance. Do not repeat random
  importance sampling from the current C4 distribution as a dense-target
  replacement; reducing the full-key calibration cost needs a deterministic
  low-rank or block-summary estimator instead.

### Wan2.1-14B 720p Flash-Style Hard-C4 Fusion Revisit

- The retained route kernel already uses FlashAttention-style online max/sum
  accumulation. The remaining fusion opportunity is to consume each merged
  K/V block before writing `merged_k`, `merged_v`, and entropy bias to HBM.
- On an exclusively idle H200 at commit `34540bd`, the exact native shape
  `[1,10,75600,128]` measured `16.841` ms for merge, `7.491` ms for route,
  `25.081` ms for the complete ten-head call, and `83.511` ms for matched SDPA.
  Four ten-head chunks therefore correspond to about `100.32` ms for Hard C4
  and `334.04` ms for SDPA (`3.33x`).
- Long stress samples separate the bottlenecks. Merge reported about `99%` SM
  activity, `12%` occupancy, `23%` MMA activity, and `40%` DRAM activity. Route
  reported about `99%` SM activity, `12%` occupancy, `37%` MMA activity, and
  `84-85%` DRAM activity. Thus workspace reads make route bandwidth-bound, but
  merge is already limited by register pressure, reductions, and on-chip
  scheduling; fusion cannot remove its arithmetic.
- One ten-head chunk materializes `63,504,000` compressed tokens. BF16 merged
  K/V plus FP32 bias cost `32.768` GB in one direction and `65.536` GB for the
  store/read round trip, or `262.144` GB over all 40 heads. This traffic is the
  correct target, but its cost overlaps merge and route computation and must
  not be subtracted directly from wall time using peak HBM bandwidth.
- The corrected current-dataflow IO model finds no ordinary one-pass tile that
  both moves less logical traffic than retained C4 and fits the rough H200
  on-chip budget. Full-row `J=4` needs about `185` KiB live state but moves
  `716.9` GB (`2.17x` retained C4) because it reloads original K/V. `J=8`
  still moves `391.7` GB (`1.19x`) and needs about `331` KiB. `J=16` finally
  drops below retained traffic but needs about `622` KiB, far beyond one SM.
- A legacy equation-level one-pass prototype was rerun at the current
  `21x45x80`, one-head shape as a dataflow diagnostic. `J=1` measured
  `224.12` ms and `J=8,I=4` measured `135.31` ms, versus `2.739` ms for the
  retained current one-head implementation. The prototype predates the current
  wide seed ordering, so these are not quality-comparison numbers, but they
  reconfirm the structural failure: small J reloads original K/V, while larger
  J creates oversized route accumulators and inefficient cross-column work.
- Hard C4 performs about `21.46` algorithmic TFLOP per 40-head call, versus
  `117.05` TFLOP for dense attention. The retained kernel realizes about
  `214` TFLOP/s. An optimistic fused design that somehow matched the measured
  dense FlashAttention throughput of about `352` TFLOP/s would take about
  `61.0` ms: at most `1.65x` faster than retained C4 and about `5.48x` versus
  matched SDPA. This is a hardware-aware upper bound, not an achieved result;
  C=4 reductions and block-diagonal route structure are less tensor-core
  friendly than dense FlashAttention.
- Rule: do not replace the retained two-stage kernel with an ordinary Triton
  one-pass implementation. A credible `1.4-1.65x` additional gain requires an
  SM90-native warp-specialized producer/consumer kernel: TMA-multicast one K/V
  tile to independently register-budgeted query-column consumers, retain route
  accumulators on chip, and use tensor-core route math without dense
  cross-column padding. Until that exists, the measured gain from writing C4
  "in FlashAttention style" is negative rather than positive.

## Weighted-Span Selector Findings (research/importance-weighted-seed-span)

Findings from the importance-weighted span seed-selection work on Wan2.1-1.3B
480p. Quality numbers are rel-L2 against dense attention on a fixed 16-point
real-QKV capture (steps 0/6/13/19, layers 0/10/20/29, heads 0/6, prompt 0);
speed numbers are A100, 21 frames, 12 heads, head_chunk 4. The accepted
direction is span selection over value-output response profiles with the
objective scaled by exp(alpha * z) at alpha=0.3, key-softmax assignment
unchanged, run as split probe/Gram-Schmidt Triton kernels at BLOCK_J=2
(rel-L2 0.06670 vs key-row C4's 0.09742, 2.48x its e2e attention time).

### Dimension-Pooled Similarity Scoring in Span Selection

- Result: pooling the Q/K head dimension used for candidate-vs-probe scoring
  destroys most of the selector's benefit; pooling the value features does not.
- Evidence: on the 16-point capture, scoring at 64/32/16 dims measured rel-L2
  +12.9% / +27.2% / +38.6% versus full 128-dim scoring, with 16-dim scoring
  (0.09210) nearly back at the fixed-seed baseline (0.09742). Value features
  pooled 128 -> 32 for the retrieval profiles cost +0.14% and are the default.
- Why not repeat: q.k similarity is a discriminative signal; mean-pooling
  collapses exactly the directional detail that separates candidates, and
  every downstream decision inherits the loss. Value pooling is forgiving
  because retrieval errors are averaged again downstream.
- Rule: pool value features, never the scoring dimension.

### Alternative Seed-Selection and Assignment Rules Against Weighted Span

- Result: no tested alternative beat weighted span with key-softmax assignment
  (0.06644 in PyTorch on the capture harness).
- Evidence, same harness and identical C=4 density: importance-only cheap
  seeds 0.09032; one Lloyd step of key-space tightness reassignment 0.07393
  with hybrids monotonically worse as tightness weight grows; per-cluster
  capacity caps a null (cap=30 bit-equal to unconstrained, cap=20 +0.7%).
  On the 3-prompt 80-call screen, MMR was monotone in lambda with even pure
  diversity +3.0% worse than span, and mass-weighted facility location had a
  real interior temperature optimum but remained +4.5% worse than span.
- Why not repeat: moving keys away from their max-likelihood seed loses
  first-order routed mass that outweighs second-order tightness gains; span's
  residual norm already encodes importance, so explicit importance terms
  mostly double-count.
- Rule: keep span + key-softmax; treat exp(alpha * z) objective scaling as the
  one importance mechanism that pays (alpha=0.3, -9.9% on the capture).

### Alpha Evaluated at Narrow Candidate Pools

- Result: the importance weight looks negligible when the candidate pool is
  narrow and large when it is full-width; early narrow-pool measurements
  under-stated it by an order of magnitude.
- Evidence: alpha's benefit over alpha=0 grew monotonically with pool width,
  from -0.09% at 5 candidates to -8.3% at the full 30 rows, with the optimal
  alpha drifting 0.1 -> 0.2 -> 0.3 as the pool widened.
- Rule: evaluate seed-selection objectives at full pool width; a narrow window
  makes candidates near-identical in attention mass and silences any
  importance-dependent term.

### Seed Reuse Across Steps, Layers and Heads

- Result: adaptive seed choices cannot be cached.
- Evidence: consecutive denoising steps agree on the full seed set 7-11% of
  the time outside layer 0 (layer 0: 84%); cross-layer agreement is 0.1%;
  cross-head cluster-partition agreement is 33% against a 25% random
  baseline; the induced key partition agrees 42% across steps -- structure,
  but far from cacheable.
- Rule: recompute selection every call; any refresh schedule is a measured
  quality tradeoff, not a free win.

### Masked-All-Keys Probe Scoring in the Selector Kernel

- Result: scoring candidates against all keys under a probe mask, to avoid a
  data-dependent gather, made the selector kernel 317x the cost of key-row C4
  and slower than the PyTorch selector it replaced.
- Evidence: every hot tile ran at BLOCK_L=64 wide instead of 16, spilling
  registers throughout; gathering the 16 probe rows into a [16, D] tile took
  the same kernel from 15,226 ms to 216 ms at identical output.
- Why not repeat: the standing caution against data-dependent gathers applies
  to per-lane gathers that break tile reuse; a gather uniform across the tile
  is one indexed load and is cheap.
- Rule: do not trade tile width to avoid a tile-uniform gather.

### Pre-Pooled Probe Values for the Split Selector

- Result: pre-pooling V to 32 dims once per call, so the Gram-Schmidt kernel
  gathers pooled rows instead of pooling per program, changed nothing.
- Evidence: rel-L2 identical at 0.06670; selector 63.1 -> 63.4 ms (noise).
  The targeted probe-V gathers were already L2-resident, because probe
  columns concentrate on high-mass keys shared across many groups, so the
  estimated HBM traffic was never actually paid.
- Rule: reverted; logical-traffic estimates must be checked against cache
  behaviour before buying complexity with them.

## Owner-Preserving Conditional Medoid Seed Refinement (2026-09-04)

- Result: insufficient and inconsistent improvement; do not promote
  `gs_medoid_256` or launch a 20-video run from this screen. It improved one
  captured call slightly and worsened the other relative to its own
  `gs_256` starting point.
- Change tested: start from globally mass-weighted Hellinger-Span C4
  (`gs_256`), keep the first row anchor, and make one owner-preserving medoid
  pass over the other three seeds. Candidate-first conditional KL is minimized
  against each fixed cell's route-mass-weighted logit barycenter, with rho and
  the original local route probabilities fixed. This remains training-free;
  merge and full compressed routing are unchanged.
- Matched evidence: frozen `call020_holdout` and `call100_holdout` QKV,
  heads `0,10,20,30`, all 21 query frames, router frame 10, Uniform16 local
  probes and Uniform256 global normalization. These are attention-output
  errors against dense, not video PSNR or runtime measurements:

  | Capture | Initial `gs_256` MSE | Refined MSE | Change vs initial GS |
  | --- | ---: | ---: | ---: |
  | call020 | 0.005324654 | 0.005294956 | -0.5577% |
  | call100 | 0.007269767 | 0.007288324 | +0.2553% |

- Diagnostics, call020/call100 respectively: each contains `302,400` selector
  groups; `56,111`/`51,338` seed slots changed in `51,816`/`46,520` groups.
  Both had **zero probe-owner violations and zero anchor violations**.
  Mean weighted probe-local log-partition gains were
  `+0.002827383`/`+0.002615924` in natural-log units. Raw negative-gain counts
  were `5`/`1`, with minima `-1.907349e-6`/`-4.768372e-7`, consistent with
  FP32 roundoff; do not report strict per-group numerical monotonicity.
- Why not repeat: improving the fixed-rho, fixed-cell, probe-local objective
  does not guarantee improvement for other query frames, the actual full-key
  partitions, or final global attention output. The observed gains do not
  justify extra refinement cost or broader generation. Preserve this as a
  bounded negative result, not evidence against all variational formulations.
- Artifact: `C:\GMA\experiments\owner_preserving_medoid_screen_20260904c\screen.json`.

## Singleton-Complement Mass-Span Seed Mask (2026-09-04)

- Result: rejected. `gs_bulk_256` worsened both captured calls relative to
  its unmasked mass-weighted Hellinger-Span starting point `gs_256`; do not
  promote the mask or launch a 20-video run from this screen.
- Change tested: remove the already-exact query-column singleton from local
  probe softmax, centering, and mass estimation when it occurs in Uniform16.
  This leaves 15 or 16 valid probes and affects 20% of query columns, without
  extra QK products or compressed slots. RF10, the fixed first seed, candidate
  bank, global Uniform256 normalizer, and production merge/route are unchanged.
- Matched evidence: frozen `call020_holdout` and `call100_holdout`, heads
  `0,10,20,30`, all 21 query frames. MSE is attention-output error against
  dense, not video quality or runtime:

  | Capture | Initial `gs_256` MSE | Masked MSE | Change vs initial GS |
  | --- | ---: | ---: | ---: |
  | call020 | 0.005324654 | 0.005389178 | +1.2118% |
  | call100 | 0.007269767 | 0.007353496 | +1.1518% |

- Interpretation: the fixed-cell global-gap derivation correctly assigns
  zero deficit to the inherited singleton, but it does not guarantee that
  masking its probe improves the span surrogate or actual attention output.
  Production ownership still uses full-block seed normalization before
  singleton removal. Reduced probe coverage and temporal sharing also remain
  mismatches; these captures do not establish which caused the degradation.
  Both masked results beat the older RF10 heuristic, but lose to their own
  mass-GS baseline, so they do not demonstrate an improvement of the mask.
- Rule: preserve the mathematical observation, reject this unchanged mask
  as a next candidate. No 20-video metrics or latency improvement are claimed.
- Derivation: [singleton-complement seed selection](singleton_complement_seed_selection.md).
  Artifact: `C:\GMA\experiments\singleton_complement_screen_20260904c\screen.json`.

## Partition-Mass Selector Launch Optimization (2026-09-04)

- Result: failed the no-slower-than-RF10 runtime gate; do not promote to a
  20-video run. The best tested launch, `J=2`, four warps, and
  `reuse_owner_logp=True`, measured `134.390244 ms` per complete attention
  call versus matched RF10 `127.597534 ms`: `+6.792709 ms`, or `+5.3235%`.
- Change tested: reuse each probe's winning log probability inside its owned
  cell, removing redundant selected-seed profile storage/gathers, and sweep
  J1/2/4/8 with four/eight warps. The partition-mass objective, C4 budget,
  original merge, and full compressed route are unchanged.
- Evidence: frozen `call100_holdout`, all 40 BF16 heads, RF10 router,
  C32/G45, Uniform16/local and Uniform256/global probes, SQ1,
  `head_chunk=8`, `query_outer_chunk=21`; two warmups, four interleaved
  repeats of five calls. Compilation and dispatch-context setup are excluded.
  Original J4/eight-warps partition attention measured `138.695813 ms`.
  Best-launch seeds exactly matched that original table (`4.0/4` shared,
  all anchors/ranges/distinctness valid), and full-output relative-L2 and
  maximum absolute difference were both zero on this capture.
- Scope: complete selector/merge/route plus contiguous output, not model
  projections/RoPE or video inference. SDPA was **not measured** in this run;
  do not derive a speedup from a different run's SDPA timing.
- Rule: the optimization preserves this capture's output and reduces
  partition-selector overhead, but does not meet the runtime target. Do not
  repeat the same launch sweep or infer video-quality eligibility from parity.
- Artifact: `C:\GMA\experiments\partition_mass_launch_20260904c\screen.json`.

## Independent Two-Frame Mass-GS C16: Video-Quality Failure (2026-09-04)

- Result: do not promote. The frozen joint-C16 selector completed the
  canonical 20-video four-step test but reached only **11.467133951 dB
  PSNR, 0.347324362 SSIM, and 0.508273922 LPIPS**. All three metrics are
  worse than both RF10 and single-frame mass-GS evaluated against the same
  fixed dense20260826 reference; PSNR remains below the >12 dB target.
- Change tested: C16/g16 candidate bank, independent frames-5/15 Hellinger
  residual projections, shared four row indices with the matched first
  anchor, Uniform16 local probes and G256 global normalization. The
  original BF16-to-FP32 TF32x3 normalizer, hard C4+SQ1 merge and full global
  compressed route remain unchanged. This is training-free and pure
  all-layer, without frame/head exceptions or dense rescue.
- Same-reference quality, 20 videos x 81 frames, seed 0:

  | Method | PSNR dB (higher better) | SSIM (higher better) | LPIPS (lower better) |
  | --- | ---: | ---: | ---: |
  | Hellinger C32 RF10 | 11.529624325 | 0.357959972 | 0.505947175 |
  | Mass-GS C32 RF10 | 11.610300592 | 0.357684962 | 0.500155182 |
  | Joint mass-GS C16 | 11.467133951 | 0.347324362 | 0.508273922 |

  Joint-minus-RF10 deltas are **-0.062490374 dB / -0.010635609 /
  +0.002326747**; joint-minus-mass-GS deltas are **-0.143166640 dB /
  -0.010360600 / +0.008118740**, in PSNR/SSIM/LPIPS order. The reference is
  `/local2/denghaoran/GridMergeAttention/experiments/wan14b_720p_4step_dense_flash_matched_extended_ema_20260826/flash/videos`.
  Do not substitute the earlier dense20260829 RF10 metrics.
- Why the fixed-QKV screen was insufficient: full-40-head output MSE
  improved by 1.60%/4.04% versus mass-GS on call020/call100, but the complete
  generated-video test worsened. This records a proxy/outcome mismatch,
  not a demonstrated causal failure mechanism or a rejection of every
  temporal seed-selection formulation. Do not repeat or promote the
  unchanged C16 selector solely on those capture-level gains.
- Runtime on the completed video run: all20 complete operator
  **135.744561 ms/call**, full attention module **159.099348 ms/call**,
  model inference **39.650595 s/video**. Steady19 values are
  **135.478054 ms / 158.578163 ms / 39.545298 s**, respectively. Each video
  has 160 core and full calls, for 3,200 of each all20. Inference excludes
  loading, random-noise preparation, CPU export and video encoding.
- Separate matched frozen-QKV timing: joint C16 **126.255144 ms/call**,
  **2.625825x versus matched SDPA** (331.523956 ms), including selection,
  merge, route and contiguous output but excluding projections/RoPE.
  This is **not video speedup**; never combine the video-run 135.744561 ms
  with historical or frozen-QKV SDPA timing. Joint C32 measured
  143.875339 ms versus RF10 128.664966 ms in the matched benchmark,
  failed the no-slowdown gate, and received **no 20-video test**.
- Run integrity: `wan14b_720p_4step_joint_frame_mass_span_extended_20260904e`
  completed on scai7 H200 GPU5 with `valid: true`, generation/evaluation
  exit code 0 and empty purity logs. All 20 prompt/seed pairs, 1,620 evaluated
  frames, per-video records and runtime aggregates passed validation;
  runtime and manifest identify the checkpoint as `raw_state_dict`.
  The preceding `...20260904d` attempt was aborted when a foreign process
  entered GPU6 during model loading: zero videos, zero runtime records,
  exit 143, timing invalid. Its retained artifacts are excluded from results.
- Artifacts: `C:\GMA\experiments\wan14b_720p_4step_joint_frame_mass_span_extended_20260904e\validation_report.json`,
  `quality.json`, `runtime.json`, and `evaluation_protocol.json` in that run;
  controls in `C:\GMA\experiments\wan14b_720p_4step_entropy_mass_span_extended_20260904b\quality.json`;
  matched benchmark in `C:\GMA\experiments\joint_frame_mass_native40_benchmark_20260904d\screen.json`.
  See [joint-frame mass selection](joint_frame_mass_seed_selection.md) for
  the separate screen, parity, timing and video-result evidence.

## RF0 Anchored Partition-Deficit C16: Final Video Regression (2026-09-04)

- Setting: canonical Wan14B 720p four-step, 20 videos, seed 0, fixed RF0,
  Uniform16 probes, C16/geometry16, G256, four real seeds, unchanged SQ1 and
  full compressed attention in every layer/call. The first seed remains the
  matched-row anchor; this is not the isolated first-medoid proposal.
- Final PSNR/SSIM/LPIPS: **11.377814069 / 0.344944231 / 0.516835021**,
  versus newly matched Uniform16 Hellinger RF0 **11.520875884 /
  0.358054837 / 0.509484093**. All three metrics worsen and PSNR >12 is not
  reached. Historical RF10 and A8U8 controls are not substituted.
- Runtime all20: complete operator **134.628901 ms/call**, full module
  **157.827592 ms/call**, inference **39.446249 s/video**, 160 calls/video.
  Hellinger RF0 is 136.912539 ms / 160.148765 ms / 39.776745 s on the same
  H200. Full-module latency is 1.45% lower, but this does not rescue quality.
  No matched full-video dense runtime is available; SDPA video speedup is
  deliberately unreported. Separate frozen-QKV speedups are not substituted.
- Integrity: run `wan14b_720p_4step_partition_c16_rf0_extended_20260904l`
  has generation/evaluation exit 0, empty purity logs, matching canonical
  prompt/seed pairs, 20 x 81 evaluated frames, raw checkpoint weights, and
  validated source/config snapshots. Runtime and quality audits are separate
  `validated_summary.json` and `validated_quality_summary.json`; all 20 copied
  MP4 hashes match the remote artifacts under `C:\GMA\experiments`.
- The earlier 480-call dense-trajectory MSE improved 6.80%, yet video quality
  declined. Do not repeat or promote this unchanged selector based solely on
  that surrogate. The layer-33/head-30 regression and mass-probe sampling
  bias are diagnostic leads, not proven causes or authorization for hybrid
  dense exceptions. See [RF0 results](rf0_seed_screen_results.md) and
  [stratified diagnosis](rf0_seed_failure_localization.md).

## Original Mass-GS C32 RF0: Mixed Quality, Joint Target Unmet (2026-09-04)

- This is not an all-metric regression. The original C32/g45 Mass-GS
  selector, with only RF10 changed to RF0, completed canonical 20-video
  validation: PSNR **11.540888084**, SSIM **0.354963445**, LPIPS
  **0.505831757**. Relative to matched Uniform16 RF0 Hellinger, the changes
  are **+0.020012200 dB / -0.003091391 / -0.003652336**. PSNR wins 14/20
  and LPIPS wins 12/20, while SSIM wins 8/20. No significance claim is made.
- The PSNR/LPIPS gains do not meet PSNR >12. Complete-operator latency is
  **142.071851 ms/call all20** versus Hellinger 136.912539 ms; full module
  **165.419669 ms/call**, inference **47.151744 s/video**. The first Mass
  video has large overhead of unmeasured composition. Steady19 diagnostic
  times are **137.157585 ms / 160.354711 ms / 39.815906 s**; the operator
  point estimate is still 0.332% slower than Hellinger. The strict joint
  quality/speed objective is not established. No full-video SDPA speedup
  is inferred from separate frozen-QKV timings.
- Original Uniform16/G256, J4/four-warps, FP32 TF32x3 normalizer on BF16
  inputs, and full compressed routing are retained in every layer/call.
  Optional precision and complement variants are off. Generation and
  evaluation exit zero with empty purity logs; source/config, canonical
  prompt pairs, 20 x 81 frames, raw weights and 3,200 calls are validated.
- Artifact: `wan14b_720p_4step_mass_gs_c32_rf0_extended_20260904r`, under
  the canonical experiment directory. See [RF0 results](rf0_seed_screen_results.md).
  Preserve this mixed result distinctly from the failed partition C16 run
  and from the historical RF10 improvement. Do not repeat the unchanged
  selector expecting it to bridge the remaining gap to 12 dB.

## RF0 Local Conditional Oracles: No Automatic Gain from Exact Information (2026-09-04)

- Scope is a CPU float64 local diagnostic, not video or complete attention:
  three saved L33/step0 captures, RF0 queries only, four heads, 1,800 fixed
  banks. Same C32/g45 candidate rows, anchored four real seeds and final
  full80/SQ1 mathematical projection. Native FP8 and global route are excluded.
- Mass P16/G256 weighted conditional-output MSE is 0.000171910935.
  Replacing only its denominator with saved all-key exact-Z gives
  0.000197351773, **14.7988% worse**. Its P16 numerator is not exact block mass.
- The new conditional-Jeffreys P16/G256 pivot improves the local proxy to
  0.000167386418. Giving that pivot P80 plus exact-Z instead yields
  0.000191265445, **14.2658% worse than its own P16/G256 version**. This is
  not a rejection of the P16 Jeffreys candidate or proof that better
  information is inherently harmful. These are fixed greedy heuristics.
- Do not repeat these unchanged oracle substitutions expecting an automatic
  fix, or promote the small P16 gain to video/latency evidence. Mass already
  beats Hellinger locally in this scope, opposite to the known all-frame L33
  native regression. First distinguish row-index transfer across query frames,
  global reweighting and native merge precision. Keep RF0 and the original
  Mass-GS default; no dense exception is introduced.
- Artifact: `rf0_conditional_jeffreys_banks_20260904s`, with source/input
  receipts and independent scalar/reduction validation. No GPU, video quality
  or attention timing was measured. See the
  [derivation and nine-method results](conditional_jeffreys_seed_selection.md).
