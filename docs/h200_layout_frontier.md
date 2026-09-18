# H200 GridMerge Layout Frontier

Generated from existing scai7/H200 `metrics.json` files with at least 90
recorded self-attention calls. Tuned-kernel rows are valid for speed
comparison. Analysis-only rows are PyTorch screening paths and should be used
only as quality signals.

For a broader coverage table across all local H200 layout screens, including
first-six/first-eight-call negative results, see `docs/h200_layout_coverage.md`.

## Tuned Kernel Frontier

| Variant | Experiment | Calls | Mean ms | Speedup vs Flash | Rel-L2 | MSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `monarch_rt_triton` | `hard_c2_full3_20260813` | 360 | 4.139 | 3.59x | 0.17439 | 0.008853 |
| `hard_c2` | `hard_c2_full3_20260813` | 360 | 7.468 | 1.99x | 0.11824 | 0.004069 |
| `hard_c3` | `hard_c2_full3_20260813` | 360 | 11.766 | 1.26x | 0.08771 | 0.002239 |
| `hard_c4` | `hard_c2_full3_20260813` | 360 | 15.567 | 0.95x | 0.07463 | 0.001621 |
| `hard_qc2_kr2_c8` | `hard_qc2_kr2_c8_realqkv_full3_20260813` | 360 | 23.555 | 0.62x | 0.07246 | 0.001528 |

## Oracle Dynamic-C Schedule

The one-prompt per-call sweep `percall_c_sweep_full1` was reprocessed with
`analysis/optimize_c_schedule.py` to ask whether choosing C per attention call
can reach a target speed while improving over fixed C1. This is an oracle
planning result, not a deployed runtime policy.

| Variant | Mean ms | Speedup vs Flash | Rel-L2 | MSE | Choices |
| --- | ---: | ---: | ---: | ---: | --- |
| `flash_attn_triton` | 14.954 | 1.00x | 0.00118 | 3.85e-7 | fixed |
| fixed `c1` | 4.229 | 3.54x | 0.18347 | 0.009518 | all C1 |
| fixed `c2` | 7.578 | 1.97x | 0.12219 | 0.004338 | all C2 |
| oracle `3x_vs_flash` | 4.985 | 3.00x | 0.15326 | 0.007133 | 93 C1, 27 C2 |
| oracle `2.5x_vs_flash` | 5.981 | 2.50x | 0.13088 | 0.005201 | 65 C1, 49 C2, 6 C3 |
| oracle `2x_vs_flash` | 7.474 | 2.00x | 0.10925 | 0.003625 | 40 C1, 52 C2, 28 C3 |
| oracle `1.5x_vs_flash` | 9.969 | 1.50x | 0.08713 | 0.002306 | 19 C1, 40 C2, 36 C3, 25 C4 |

The signal is useful but limited: a strict `3x` budget can only upgrade about a
quarter of the calls from C1 to C2, so the error remains much closer to C1 than
to C2. Dynamic C is a speed/quality knob, not by itself a solution to the
`3x` speed plus hard-C4-like quality target.

## Target Gap

Using the one-prompt per-call sweep as the most detailed speed/error trace,
FlashAttention is `14.954` ms/call, so a `3x` attention speedup requires about
`4.985` ms/call or less.

Current measured options separate cleanly into two groups:

- Speed-feasible but too inaccurate: fixed C1 is `4.229` ms/call and `3.54x`
  versus FlashAttention, but rel-L2 is `0.18347`. The best oracle `3x` dynamic-C
  schedule is still rel-L2 `0.15326`.
- Accuracy-feasible but too slow: fixed C3/C4 are rel-L2 `0.08814`/`0.07506`,
  but `11.841`/`15.423` ms/call. Even the oracle schedule at C3-like error
  (`1.5x_vs_flash`, rel-L2 `0.08713`) is `9.969` ms/call.

Therefore, reaching the stated goal needs either:

- a low-C layout with C1/C2-like runtime but much lower error, or
- a C3/C4-quality layout whose kernel cost is reduced by roughly `2x` beyond
  the current tuned C3/C4 kernels.

Existing evidence narrows the search. C1 seed changes only improved full-run
rel-L2 to about `0.162`, still far from C2/C3. Temporal-key `qf1_kf7_c5/c7`
has the strongest low-density quality signal, but a naive two-stage kernel is
merge-bound because it compresses `364` keys per block. Future experiments
should therefore avoid more fixed C1 seed tweaks and avoid naive larger-key
merge kernels; they need a different low-C block objective or a genuinely
cheaper large-key merge design.

## Important Dominated Tuned Kernels

These kernels are not on the speed/quality frontier, but they are useful
negative evidence for layout choices.

| Variant | Experiment | Calls | Mean ms | Speedup vs Flash | Rel-L2 | MSE | Density | Block q x k |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hard_kr2_c4` | `hard_kr2_c4_realqkv_full3_20260813` | 360 | 17.457 | 0.84x | 0.08585 | 0.002146 | 0.03846 | 30 x 104 |
| `hard_kr2_c6` | `hard_kr2_c6_tuned_realqkv_full3_20260813` | 360 | 27.030 | 0.54x | 0.06635 | 0.001281 | 0.05769 | 30 x 104 |
| `hard_kr3_c5` | `hard_kr3_c5_tuned_realqkv_full3_20260813` | 360 | 27.844 | 0.52x | 0.08240 | 0.001977 | 0.03205 | 30 x 156 |

## Best Analysis-Only Quality Candidates

| Variant | Experiment | Calls | Mean ms | Rel-L2 | MSE | Density | Block q x k |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `flat128_c12` | `flat128_keyblocks_full3_20260813` | 360 | 437.988 | 0.04671 | 0.000635 | 0.10000 | 30 x 128 |
| `flat128_c10` | `flat128_keyblocks_full3_20260813` | 360 | 375.613 | 0.05200 | 0.000787 | 0.08333 | 30 x 128 |
| `qc1_kr2_c8` | `rowcol_candidate_full3_20260813` | 360 | 270.921 | 0.05567 | 0.000902 | 0.07692 | 30 x 104 |
| `qc1_kr5_c10` | `rowcol_wider_key_lowdensity_full3_20260813` | 360 | 270.471 | 0.06135 | 0.001096 | 0.03846 | 30 x 260 |
| `c4_prefilter12_greedy` | `lowc_seed_modes_screen_20260813` | 90 | 561.931 | 0.05739 | 0.000427 | - | - |
| `c4_greedy_cover` | `lowc_seed_modes_screen_20260813` | 90 | 725.987 | 0.05775 | 0.000433 | - | - |
| `qc2_kr2_c12` | `rowcol_qc2kr2_c8_c10_full3_20260813` | 360 | 236.356 | 0.05495 | 0.000879 | 0.11538 | 60 x 104 |
| `flat128_c8` | `flat128_keyblocks_full3_20260813` | 360 | 310.554 | 0.05938 | 0.001026 | 0.06667 | 30 x 128 |
| `qc1_kr2_c7` | `rowcol_candidate_full3_20260813` | 360 | 244.193 | 0.06093 | 0.001081 | 0.06731 | 30 x 104 |
| `qc2_kr2_c10` | `rowcol_qc2kr2_c8_c10_full3_20260813` | 360 | 201.634 | 0.06338 | 0.001169 | 0.09615 | 60 x 104 |
| `qc2_kr1_c7` | `rowcol_candidate_full3_20260813` | 360 | 224.427 | 0.06336 | 0.001169 | 0.13462 | 60 x 52 |
| `qc1_kr5_c9` | `rowcol_kr5_c8_c9_full3_20260813` | 360 | 244.012 | 0.06491 | 0.001226 | 0.03462 | 30 x 260 |
| `c3_prefilter12_greedy` | `lowc_seed_modes_screen_20260813` | 90 | 467.921 | 0.06624 | 0.000569 | - | - |
| `qc1_kr2_c6` | `rowcol_kr2_c6_left_full3_20260813` | 360 | 217.132 | 0.06638 | 0.001283 | 0.05769 | 30 x 104 |
| `qc1_kr5_c8` | `rowcol_kr5_c8_c9_full3_20260813` | 360 | 223.905 | 0.07041 | 0.001443 | 0.03077 | 30 x 260 |
| `qc2_kr2_c8` | `rowcol_qc2kr2_c8_c10_full3_20260813` | 360 | 164.297 | 0.07253 | 0.001531 | 0.07692 | 60 x 104 |

## Current Conclusion

- The only measured setting around 3x versus FlashAttention is C1/MonarchRT,
  but its dense error is high: rel-L2 around 0.17.
- C2 gives about 2x versus FlashAttention with rel-L2 around 0.118.
- C3/C4 improve quality but drop to roughly 1.3x or below versus
  FlashAttention.
- `qc1/kr2/C4` has the same compressed-token density as C2 (`4/104 = 2/52`)
  and much better error than C2, but its tuned Triton kernel is too slow:
  `17.46` ms/call versus C2 `7.40` ms/call and hard C4 `15.45` ms/call in
  the full3 H200 run. The larger key-row merge working set dominates.
- `qc1/kr2/C6` is a stronger quality point than hard C4, but still not a
  speed point. Retuning its merge tile from `BLOCK_J=64` to `BLOCK_J=16`
  improved the A=10 stage profile from about `34.2` ms full to `12.86` ms
  full, but the three-prompt/all-call real-QKV benchmark still measured
  `27.03` ms/call, slower than dense SDPA and FlashAttention.
- `qc1/kr3/C5` has even lower compressed-token density than C2 and improves
  error over hard C3, but the tuned Triton kernel is also slow. Best tested
  merge config was `BLOCK_J=16, warps=8`; A=21 profile was `27.47` ms full and
  real Wan full3 measured `27.84` ms/call. The `BLOCK_L=256` merge working set
  dominates.
- Some analysis-only row/key-group layouts, especially `qc1_kr2_c8`, are much
  better in dense error at comparable compressed-token density, but existing
  `kr2` Triton evidence shows the larger merge working set is not currently a
  speed path.
- Row-major contiguous 128-token key blocks are a strong quality signal but
  not yet a speed path. The full3 screen `flat128_keyblocks_full3_20260813`
  kept one full query column and compressed each frame into 13 contiguous
  128-token key chunks. `flat128_c8` reached rel-L2 `0.05938` at density
  `0.06667`, already better than hard C4 at lower density. `flat128_c10` and
  `flat128_c12` improved to rel-L2 `0.05200` and `0.04671`, but their PyTorch
  analysis times were `375.61`/`437.99` ms/call and their merge MAC scales as
  roughly `C*128/(4*52)` versus hard C4. This is a promising partition for
  quality, but it needs a new merge design before it can be treated as a
  speed candidate. A direct Triton `flat128_c8_triton` implementation confirmed
  this: on full3/all-call real Wan QKV it measured rel-L2 `0.05931` and MSE
  `0.001024`, but took `49.74` ms/call versus FlashAttention `14.54` ms/call.
  Stage profiling with the real `query_outer_chunk=10` shape measured merge
  `21.58` ms, route `1.78` ms, and full `23.49` ms per query chunk, so about
  `92%` of the time is merge.
- Wider key-row blocks improve quality strongly in PyTorch analysis. The most
  striking low-density point is `qc1_kr5_c10`: it has the same
  compressed-token density as C2 (`0.03846`) but rel-L2 `0.06135`, much better
  than hard C4. The static cost model explains why this is not immediately a
  speed path: its merge MAC is `2.5x` hard C4 even though its route MAC is only
  `0.5x` hard C4. The updated cost table now records this explicitly:
  `qc1_kr5_c10` has two-stage logical IO `0.497x` FlashAttention, but total
  MAC `0.538x` FlashAttention with the merge stage dominating. A naive
  two-stage `kr5/c10` Triton kernel would therefore optimize the wrong side of
  the tradeoff unless the 260-key merge itself is redesigned.
- The new full3 `kr5` C sweep fills the useful gap below C10. In
  `rowcol_kr5_c8_c9_full3_20260813`, `qc1_kr5_c8` reached rel-L2 `0.07041` at
  density `0.03077`, and `qc1_kr5_c9` reached rel-L2 `0.06491` at density
  `0.03462`. Both are below C2 density (`0.03846`) and better than hard C4
  error (`0.07463`) in the same full3 setup. This is the strongest current
  quality signal for the multi-row-key direction, but the cost table still
  shows merge MAC `2.00x`/`2.25x` hard C4, so the next useful speed work would
  need a specialized 260-key merge design rather than a direct copy of the
  existing `kr2/kr3` kernels. A direct two-stage `hard_kr5_c8` Triton
  prototype confirmed that concern. The first version compiled only after
  splitting the 260-key merge tile into `256 + 16` and measured `178.30`
  ms/call on one real Wan QKV call. Reusing seed logits and retuning the merge
  kernel to `BLOCK_J=8, num_warps=8` improved a first-three-call H200 run to
  `84.38` ms/call, but this is still much slower than FlashAttention `14.80`
  ms/call and hard C4 `17.26` ms/call in the same run.
- The static cost model in `docs/gridmerge_layout_cost_model.md` made
  `qc2_kr2_c8` look like the best multi-row/column kernel candidate because it
  keeps C4-like main MAC and lowers idealized two-stage logical IO. The tuned
  H200 Triton kernel did not realize that speedup: three-prompt/all-call real
  Wan QKV measured `23.55` ms/call, slower than dense SDPA at `18.02` ms/call
  and hard C4 at `15.40` ms/call. It remains a quality-oriented data point
  because its rel-L2 `0.07246` is slightly better than hard C4 `0.07463`, but
  the current implementation is not a speed path.
- Key-column grouped layouts were also screened. At the same density as hard
  C4, `kg2_c2` splits each row into two `26`-key groups and worsened full-run
  rel-L2 to 0.10980, while `kg4_c1` was already bad in the first-six-call
  screen. `kg2_c3` nearly matched hard-C4 error but only by increasing density
  to 0.11538, so it is not a 3x speed candidate.
- The transpose row/column swap is also not on the frontier. The full H200
  three-prompt run `transpose_monarchrt_full3_h200_20260813` measured
  transpose C1 at rel-L2 `0.17649` with density `0.03333`; this is worse than
  non-transpose C1/MonarchRT despite being denser. Transpose C4 measured
  rel-L2 `0.11976` with density `0.13333`, roughly hard-C2-level error at much
  higher density, so a tuned transpose C4 kernel is not justified.
- Same-axis blocks are not viable for the current target. The H200 first-six
  screen `same_axis_layout_screen_20260813` tested W/W, H/H, and T/T query-key
  inner axes; all had rel-L2 around 0.74-0.81 versus dense, far worse than
  hard C4 at about 0.08 in the same calls.
- Low-density T-axis mixed blocks are also not viable. The full3 H200 run
  `axis_low_density_full3_20260813` measured `qh_kt_c1` at rel-L2 `0.15571`
  with density `0.04762`, worse than hard C2's rel-L2 `0.11824` despite higher
  compressed-token density. `qh_kt_c2` raised density to `0.09524` but still
  had rel-L2 `0.10335`, worse than C3 at lower density. Other low-density
  T-axis choices were worse.
- A broader first-eight-call mixed-axis screen also did not rescue temporal
  layouts. In `mixed_axis_broad_screen_20260813`, the best temporal-query point
  was `qt_kh_c4` with rel-L2 `0.08643` at density `0.13333`; the original
  `qh_kw_c4` reference was rel-L2 `0.08769` at density `0.07692`. At matched
  density `0.05769`, `qt_kw_c3` was rel-L2 `0.11945`, much worse than
  original `qh_kw_c3` at `0.10021`.
- Temporal-distance adaptive C on the original row/column grid gives a smooth
  C1-to-C2 quality tradeoff, but not a new frontier. It keeps full compressed
  second-stage attention and changes only the number of seeds used for a
  `(query frame, key frame)` pair. In the first-eight-call H200 screen
  `temporal_adaptive_c_screen_20260813`, the best variants near C2 density were
  `ta_r3c4_far1` with density `0.03689`, rel-L2 `0.11670`, and MSE `0.001756`,
  and `ta_r4c4_far1` with density `0.04134`, rel-L2 `0.11232`, and MSE
  `0.001627`. The matched C2 first-eight reference is around rel-L2 `0.10032`
  at density `0.03846`, so adaptive C does not justify a tuned kernel yet.
- Temporal key grouping has one useful quality signal but not a speed path yet.
  In `temporal_grouped_promising_full3_20260813`, `qf1_kf3_c3` compressed the
  same spatial row across three key frames into C3, keeping C1 density
  `0.01923` while improving rel-L2 to `0.10417`. A tuned Triton prototype
  reached `15.09` ms/call in `hard_kf3_c3_tuned_realqkv_full3_20260813`, but
  hard C3 was both faster (`11.65` ms/call) and more accurate (`0.08771`
  rel-L2), so the backend was removed.
- A longer temporal-key block is the strongest density/quality signal so far,
  but not yet a speed path. In `temporal_kf7_full3_20260813`, `qf1_kf7_c7`
  groups the same spatial key row across seven key frames and compresses
  `364` keys to C7. It keeps C1 compressed-token density `0.01923` while
  reaching rel-L2 `0.07408` and MSE `0.001597`, essentially hard-C4 quality.
  `qf1_kf7_c5` is also close to hard C3 quality with density `0.01374`,
  rel-L2 `0.08704`, and MSE `0.002205`. The updated static cost model shows
  why direct kernelization is hard: `qf1_kf7_c7` has only `0.25x` C4 route MAC
  but `1.75x` C4 merge MAC, so a straightforward two-stage kernel is likely
  merge-bound.
- Grouping temporal query frames was screened as a way to amortize the large
  kf7 merge, but the quality fell off. In the first-eight-call H200 screen
  `temporal_qf_kf7_screen_20260813`, `qf3_kf7_c7` had density `0.01923` but
  rel-L2 `0.11772`, much worse than `qf1_kf7_c7` and hard C4. `qf7_kf7_c7`
  was worse again at rel-L2 `0.15221`. The static model says qf7/kf7 could be
  cheap, but sharing one compression across seven query frames loses too much
  query-frame-specific alignment.
- Wide query-column grouping is also not promising. In
  `rowcol_wide_query_screen_20260813`, `qc4_kr2_c8` had C4 density `0.07692`
  but rel-L2 `0.25104`, and `qc13_kr2_c8` had rel-L2 `0.55868`. Increasing C
  did not recover the quality: `qc4_kr2_c12` was still rel-L2 `0.14143`.
  The static IO/MAC model looks attractive for these variants, but sharing one
  compressed K/V set across many query columns destroys too much local
  alignment.
- Moderate two-column query grouping also does not improve the frontier. The
  H200 first-eight-call screen `rowcol_qc2_more_screen_20260813` tested
  `qc2_kr2_c5/c6/c7`, `qc2_kr3_c8/c9`, and `qc2_kr5_c10/c12`. At the same
  density as hard C3, `qc2_kr2_c6` had rel-L2 `0.16483` and `qc2_kr3_c9` had
  rel-L2 `0.11884`, both worse than same-axis reference C3 screens. The best
  low-density point was `qc2_kr5_c12` with density `0.04615` and rel-L2
  `0.09830`, but its static merge MAC is `1.5x` hard C4, so it is still a
  quality-oriented, merge-heavy point rather than a 3x speed candidate.
- Non-contiguous key-row grouping is worse than contiguous key rows. The H200
  first-eight-call screen `rowcol_interleaved_keyrow_screen_20260813` grouped
  rows by stride, for example `(0,15),(1,16),...` for `kr2`. It worsened every
  matched comparison: `qc1_kr2_c4` rel-L2 went from `0.10032` contiguous to
  `0.12114` interleaved; `qc1_kr2_c6` went from `0.08339` to `0.08906`; and
  `qc1_kr3_c6` went from `0.08992` to `0.11115`. This supports keeping key
  rows spatially contiguous when widening key blocks.
- 2D key patches did not improve the speed-quality frontier. The H200
  first-eight-call screen `keypatch_layout_screen_20260813` tested patches like
  `2x26`, `2x13`, `3x13`, and `5x26`. At hard-C4 density, `kp2x26_c4` reached
  rel-L2 `0.09306`, worse than hard C4's `0.08765` in the same calls, while
  `kp2x13_c2` and `kp3x13_c3` were much worse. `kp2x26_c6` reached good error
  (rel-L2 `0.07548`) only by raising density to `0.11538`, so patching keys is
  a quality knob rather than a 3x speed candidate.
- 2D query-patch plus 2D key-patch blocks are worse. The H200 first-eight-call
  screen `patchpair_screen_20260813` kept the same block sizes as hard C4 with
  `qp15x2_kp2x26` (`30` query tokens and `52` key tokens), but changed both
  sides from full column/row lines to spatial patches. At the same C4 density
  `0.07692`, `qp15x2_kp2x26_c4` measured rel-L2 `0.28592`, far worse than
  hard C4's rel-L2 `0.08765` in the same screen. Even increasing to C6
  density `0.11538` only reached rel-L2 `0.17600`, so patch-pair partitioning
  should not get Triton tuning.
- Query-patch-only grouping is also not a speed frontier point. The H200
  first-eight-call screen `querypatch_only_screen_20260813` sanity-checked
  `qp30x1_kp1x52_c4` and reproduced hard-C4 quality, rel-L2 `0.08769`. Changing
  only the query block to `15x2` while keeping key rows as `1x52` worsened C4
  to rel-L2 `0.09955` at the same density. `qp15x2_kp1x52_c6` improved to
  rel-L2 `0.07207`, but only by raising density to `0.11538`, so it is a
  high-density quality knob rather than a 3x candidate.
- Repartitioning query blocks to `60 x 64` or `64 x 64` is not promising at
  C4. The H200 first-eight-call screen `query_key_block_60_64_screen_20260813`
  tested `q60_k64_c4`, where the query block is two adjacent image columns
  (`30*2=60` query tokens) and the key block is row-major contiguous 64 tokens,
  plus `q64_k64_c4`, where both sides are row-major contiguous 64-token chunks.
  Both have compressed-token density `0.06410`. `q60_k64_c4` measured rel-L2
  `0.20142`, and `q64_k64_c4` measured rel-L2 `0.70185`, versus same-screen
  hard C4 rel-L2 `0.08765`. These layouts should not get Triton tuning unless
  a new seed rule first recovers quality. A follow-up seed rule
  `q60_k64_c4_x2r2`, using two row/column intersection queries plus two
  deterministic random query seeds, did not help: rel-L2 worsened to `0.21348`
  in `q60_k64_x2r2_screen_20260813`.
- Hard-C4 `query_outer_chunk` tuning is not enough to explain the gap. The H200
  12-call sweep `hard_c4_query_chunk_sweep_no21_20260813` measured chunk
  `1/3/7/10` at `17.57`/`17.75`/`16.08`/`16.72` ms/call, with identical dense
  error for all chunks. `chunk=7` was best, but still slower than the same-run
  FlashAttention baseline at `14.97` ms/call. This suggests the outer Python
  chunk loop and launch count are secondary; the two-stage intermediate K/V/bias
  traffic remains the main target. `chunk=21` currently triggers a Triton
  illegal memory access during warmup and should not be used without debugging.
- The no-intermediate fused hard-C4 direction has a narrow feasible tile range.
  The updated IO estimator shows that tiny column tiles reload K/V too often,
  while large column tiles exceed the approximate H200 per-block shared-memory
  budget. In the default sweep, `I_TILE=30,J_TILE=8` is the only tile that both
  stays under FlashAttention logical IO (`0.739x`) and fits the rough H200
  228-KiB budget (`222` KiB live state, `97.5%`). This makes `full query rows x
  8 query columns` the first plausible no-intermediate fused-kernel target.
- A synthetic Triton live-state probe for that target ran successfully on H200.
  `q_tile=240` pads to a `256 x 128` fp32 accumulator (`128` KiB) and compiled
  cleanly. This is not a full hard-C4 runtime result, but it means the next
  step can be a real one-pass fused prototype instead of more estimator-only
  work.
- A no-intermediate streaming PyTorch reference now exists for hard C4 in
  `attention/hard_route_pytorch.py`. It computes one `(key_outer, key_row)` block
  at a time and immediately updates the second-stage softmax accumulator instead
  of materializing `merged_k`, `merged_v`, and `bias`. A small CUDA tensor check
  on scai7 matched the existing PyTorch reference with max_abs `3.6e-7` and
  rel-L2 `3.8e-7`, so this is the equation-level reference for the fused Triton
  prototype.
- A first one-pass Triton prototype, `hard_c4_onepass_j1`, now matches that
  streaming reference well enough for smoke testing, but it is not a speed
  candidate. On H200 synthetic QKV `[1,2,30,52,1,128]`, it measured `0.681`
  ms versus `0.042` ms for the current two-stage hard C4 kernel because it
  handles only one query column per program and reloads K/V for every column.
  It should be used to debug the one-pass equations only. The next useful
  kernel target is still `J_TILE=8`.
- A direct `J_TILE=8` one-pass prototype was tested next and is also not the
  final path. It keeps the natural `[row,column,dim]` accumulator live inside
  one Triton program. This compiles and matches the reference, but H200 timing
  on synthetic QKV `[1,2,30,52,1,128]` was `2.871` ms versus `0.053` ms for the
  two-stage kernel. The likely issue is register/local-memory pressure from the
  `30 x 8 x 128` fp32 accumulator plus merge temporaries. A faster fused design
  needs less live state, not just a wider column tile. Reducing the row tile
  helps but does not fix the path: `I_TILE=1/2/4/8` measured
  `0.871/0.461/0.463/0.624` ms, still far slower than the two-stage kernel.
  A first-three-call real Wan-QKV smoke was worse: after fixing empty-token
  online softmax, `hard_c4_onepass_j8` produced hard-C4-level error
  (rel-L2 `0.09311`) but still measured `324.71` ms/call. It is now treated as
  research code only, not a usable backend.
- The IO estimator now also includes a conservative two-pass no-merged-KV
  design. Once D-tiled output accumulation is accounted for, there is no
  two-pass tile that is both below FlashAttention logical IO and within the
  rough H200 228-KiB live-state budget. Small `D_TILE` repeats full-D route
  logits and rereads seed Q/K many times; `D_TILE=128` avoids that repeated IO
  but needs too much live state. The current no-intermediate kernel target is
  therefore still one-pass full-query-rows `J_TILE=8`.
- See `docs/hard_c4_loop_io_notes.md` for the loop/reload accounting behind
  this conclusion. In short, the failed one-pass J8 prototype removes
  intermediate writes but raises original K/V reloads to about `1176x` per
  token, versus about `256x` for the local FlashAttention kernel and about
  `21x` in the maintained hard-C4 merge stage.
- Query-outer tiling inside the merge program was also tested and rejected. It
  preserves the exact hard-C4 algorithm and tries to reuse K/V across adjacent
  query frames, but the H200 A=10 stage profile slowed from `7.33` ms full for
  the maintained kernel to `21.04` ms with `A_TILE=2`; larger tiles were worse.
  The loss of parallelism/register behavior outweighs the K/V-load saving.
