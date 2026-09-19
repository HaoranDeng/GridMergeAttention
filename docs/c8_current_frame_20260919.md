# Q45K80 C8 without RF0 selector reuse

Removing RF0 reuse improves mean sampled attention-output relative L2 from
**8.2463% to 7.0008%**: a **1.2456 percentage-point** reduction, or **15.10%**
less relative error. It wins 319/384 paired cases (83.1%). Complete H40
attention takes **320.517 ms**, versus **336.503 ms** for matched Flash SDPA
(**1.050x speedup**) and **160.932 ms** for original r2 RF0.

This is an opt-in kernel at `kernels.experimental.c8_current.create`.
The default C8 remains r2, and `slow_reference` remains explicit.

## Isolated change

The control is the shipped **Hellinger r2** Q45K80 C8, not the recent
anchor-Fisher spatial-axis reference. Candidate geometry C32/g45, alpha 0.2,
A8U8 probe policy, exact-eight capacity, BF16 boundaries, merge, and full
compressed routing are retained. Each query frame computes its own probes
and selected row indices. The merge epilogue prefetches the next frame's
selected Q rows instead of advancing the same row indices to the next frame.

There are no additional representatives, raw keys, dense rescue calls,
learned parameters, or layer/step exceptions.

## Matched numerical comparison

The frozen external dense trajectory uses three canonical prompts, four
StepDistill steps, heads 0/10/20/30, and zero-based layers
**0, 5, 10, 15, 20, 25, 30, 35**. The seed policy remains base seed 0 plus
zero-based prompt index. Sampled query coordinates remain the Cartesian
product of frames 0/10/20, rows 0/22/44, and columns 0/40/79: 27 queries per
head. Both candidates consume identical BF16 Q/K/V and are compared to
the same saved Flash SDPA output. Candidate outputs never advance the model.

| Layer | RF0 relative L2 | Current-frame relative L2 |
|---|---:|---:|
| 0 | 2.6488% | 2.7215% |
| 5 | 5.5889% | 4.9634% |
| 10 | 11.2938% | 8.4710% |
| 15 | 7.7162% | 6.6714% |
| 20 | 11.9555% | 9.7448% |
| 25 | 9.5670% | 8.2674% |
| 30 | 7.0050% | 6.1288% |
| 35 | 10.1954% | 9.0377% |
| Mean of 384 head cases | 8.2463% | 7.0008% |
| Pooled relative L2 | 7.7763% | 6.3278% |

Frame 0 is unchanged (8.2435%). The sampled frame-10 error changes from
8.3842% to 6.5979%, and frame-20 from 8.1004% to 6.5404%. Layer 0 regresses
slightly; seven of the eight layer means improve. This supports a useful
but uneven numerical benefit from current-frame selection.

These are QKV output-error measurements. **PSNR, SSIM, LPIPS, VBench, and
end-to-end video quality have not been measured for this variant.** The
numbers above must not be presented as gains in those metrics. The four-step
capture protocol is preserved rather than silently changed to 50 steps.

## Full-call runtime

One exclusively idle scai7 H200, GPU6, with continuous foreign-process
monitoring. Seven paired repetitions alternate method order. BF16 input
shape is `[1, 75600, 40, 128]`; timing uses CUDA events. Every complete
candidate call includes selector, probe, Q packing, merge, and full route.
Dense includes its contiguous token-major output conversion. Compilation
and warmup are excluded. The pre-existing H40 layer-0 capture is used only
for implementation validation and timing, not quality ranking.

| Method | mean_attention_ms | speedup_vs_sdpa |
|---|---:|---:|
| Flash SDPA | 336.503 | 1.000x |
| Original r2 RF0 | 160.932 | 2.091x |
| Current-frame C8 | 320.517 | 1.050x |

Current-frame C8 is 4.75% lower latency than dense and 99.16% higher latency
than RF0 r2. A single diagnostic selector measurement is 162.292 ms for all
21 frames versus 7.458 ms for RF0; these stage measurements are not substituted
for the seven-repeat complete-call mean. The speed gate was satisfied on
this H40 capture; runtime on other shapes/hardware is unmeasured.

## Validation and provenance

- RF0 seeds and output exactly match shipped r2.
- Broadcasting RF0 seeds through the new frame-aware merge exactly reproduces
  the one-frame seed-bank control.
- r2 versus `slow_reference` relative L2: 0.02199% on the H40 validation input.
- Reversing query-frame order reverses the selected seed banks exactly.
- Independent PyTorch merge/route reference relative errors are 0.1926%,
  0.1988%, and 0.2108% for the sampled validation lines. The engineering test
  gate is 1.5%; this is not a user-specified universal tolerance.
- All-tied scores retain exactly eight slots, with seven zero-KV empty slots
  and positive-infinity route bias; final outputs remain finite.
- Both timing and quality GPU guards completed with `valid: true`.

Local artifacts: `C:/GMA/experiments/c8_current_20260919/`.
The incremental exploratory workbook row is in `C:/GMA/tmp.xlsx`.
Remote artifacts have the same directory name under
`/local2/denghaoran/GridMergeAttention/experiments/`.
The numerical capture source is `rf0_spatial_c8_8layers_20260919_v1`, with its
prompt manifest and per-capture metadata. The checked-in compact evidence is
`kernels/experimental/c8_current/evidence.json`.

Run the benchmark as a module under `analysis.guard_axis_job`. Use
`--timing-only` for the H40 comparison, `--quality-only` for the complete
eight-layer replay, or neither for both. Missing captures cause a hard error.
The CUDA source builder preserves measured source bytes across platforms.

## Matched SVG-EAR supplement

SVG-EAR top_p=0.85 was replayed on the **same 96 captures and 384 head cases**,
including layer 0. No layer or timestep uses a dense fallback. The comparison
uses the same saved BF16 dense outputs and identical 27 query coordinates.

| Method | Mean case relative L2 | Pooled relative L2 |
|---|---:|---:|
| Hellinger C8 r2 RF0 | 8.2463% | 7.7763% |
| Hellinger C8 current-frame | 7.0008% | 6.3278% |
| SVG-EAR p=0.85 | 6.8446% | 17.4802% |

EAR has a 0.1561 percentage-point advantage in mean case error and wins
298/384 cases against current-frame C8. Its pooled error is much higher:
layer 0 accounts for 84.26% of EAR's total squared error. The largest
energy-weighted errors are in head 30 of that layer, including a 45.81%
relative-L2 case at prompt 1, step 2. These cases remain in every aggregate.
Mean case error weights all cases equally; pooled relative L2 divides total
squared error by total reference energy before taking the square root.

EAR uses Q/K centroid counts 300/1000, 50 initial k-means iterations, two
update iterations, and effective min_kc_ratio=0.05. It recomputes assignments
and retained regions on each attention call, using the previous step's
centroids as the starting point. It does not reuse RF0 query-row indices.
There is one processor per sampled layer, reset for each prompt. Initial
centroid RNG uses seed 0 per prompt and logical H40 draws; rows for heads
0/10/20/30 are selected before gathering captured tensors. Clustering and
its convergence test run on the four available heads, so this is a matched
H4 numerical replay rather than an assertion of bitwise H40 clustering.

Realized attention-map density is **28.1139%**, weighted by query/key cluster
sizes; sparsity is 71.8861%. The value 0.85 is a mass threshold, not density.

The supplement also measures **H4**, with three warmed paired CUDA-event
repeats per capture and identical centroid/RNG state restored before each
repeat. EAR mean_attention_ms is **29.930**, matched H4 Flash SDPA is
**31.716**, and speedup_vs_sdpa is **1.060x**. Initial clustering is included
on step 0, with centroid updates on steps 1-3. These H4 times **cannot be
compared directly with the H40 C8 times above**.

The exclusive GPU6 guard completed successfully. Raw evidence is `ear.json`
and `ear_guard/` in the same canonical artifact directory. Compact evidence
is `kernels/experimental/c8_current/ear_evidence.json`, and the replay script
is `analysis/compare_ear_c8_current.py`. No video metrics were measured.
