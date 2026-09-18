# RF0 seed-selection screens

Completed on 2026-09-04. The first two sections report attention diagnostics;
the final sections report the separate canonical 20-video comparison. The
runs exited successfully on an exclusively clean H200.
All methods use fixed RF0, Uniform16, four seed queries, and unchanged SQ1,
merge, entropy correction, and full global attention over compressed entries.

## All-layer dense-trajectory diagnostic

Artifact: `native_seed_trajectory_rf0_20260904g/screen.json` and its
`validated_summary.json`, under the canonical experiment directory. The
validator checked all 480 ordered calls, canonical prompt manifests, method
identity, source hash, array lengths, error aggregation, exit code and purity.
Coverage is three prompts, four actual warped steps, 40 layers, all 21 query
frames, and heads 0/10/20/30. Each candidate is an off-path diagnostic: only
dense output advances the external reference trajectory.

| RF0 method | Output MSE | Change vs Hellinger | Calls with lower MSE / 480 |
| --- | ---: | ---: | ---: |
| Hellinger C32/g45 | 0.004669802173 | reference | — |
| Mass-GS C32/g45 | 0.004531736547 | -2.96% | 461 |
| Mass-GS C16/g16 | 0.004435673606 | -5.01% | 438 |
| Partition-deficit C16/g16 | 0.004352409845 | -6.80% | 428 |

MSE is the sum of squared output errors divided by the number of scalar
outputs. Win counts are strict per-call comparisons, not a statistical test.
Partition-deficit reduces prompt-aggregated MSE by 7.94%, 2.51%, and 9.59%.
Its step-aggregated changes are -0.004%, -5.12%, -10.61%, and -11.43%; the
first step is effectively tied, rather than showing a uniform gain.
Sampled-head times include compilation on the first call and are not used
for a speed comparison.

A later [stratified diagnosis](rf0_seed_failure_localization.md) found that
step 0 improves on 99/120 calls, but severe layer-33/sample-head-30 regression
nearly cancels the other layers' gains. This recurs across the three prompts
and is not explained by the smaller C16 bank alone. Neither the aggregate
gain nor the two layer-20 captures below establish uniform behavior.

## Matched full-40-head benchmark

Artifact: `rf0_native40_seed_benchmark_20260904h/screen.json`. The two frozen
captures are the same holdout prompt at layer 20, steps 0 and 2. Their stored
FP16 tensors are converted to BF16 for all methods and SDPA. This broadens
head coverage, but does not add layer or prompt coverage. Timings include
selection, merge, route and contiguous output, with no cached seed tables.
Four repeats of five calls follow two warmups, in alternating method order.

| RF0 method | Call020 MSE | Call100 MSE | Call020 ms | Call100 ms | SDPA speedup 020 / 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hellinger C32/g45 | 0.005066637 | 0.009249786 | 128.348 | 129.088 | 2.586x / 2.582x |
| Mass-GS C32/g45 | 0.004431024 | 0.008380183 | 128.110 | 128.476 | 2.591x / 2.594x |
| Mass-GS C16/g16 | 0.004450542 | 0.008522902 | 122.968 | 123.500 | 2.699x / 2.699x |
| Partition-deficit C16/g16 | 0.004487841 | 0.008475157 | 125.777 | 125.749 | 2.639x / 2.650x |
| Dense SDPA Flash | reference | reference | 331.932 | 333.268 | 1x / 1x |

Each speedup uses only that capture's matched forced-Flash SDPA measurement.
Partition-deficit is 2.00%/2.59% faster than Hellinger RF0 in this benchmark.
Mass C32 has the lowest error on these two frozen full-head captures, unlike
the aggregate sampled-head trajectory ranking. Neither screen alone is a
reliable substitute for final video quality.

Both completed screens use a documented partition-only four-warp launch and
owner-log-probability reuse override. Their timing must not be attributed to
an unmodified eight-warp production launch. The subsequent production port
passed direct parity against that measured override before candidate video
generation, as detailed below.

## Production parity and completed control generation

`partition_rf0_production_parity_20260904k/verification.json` verifies both
full-40-head captures: the first complete production call is unpatched, the
observation-only call records the actual RF0 input and real returned seed
table, and the final call replays the historical four-warp/reuse recipe.
All complete outputs and both real seed tables are bitwise equal. The unchanged
kernel hashes match the earlier benchmark. Both MSE values are also unchanged.
The run exited zero with no foreign GPU process. No timing is reported for
this compilation/observation-heavy parity diagnostic.

A new canonical 20-video Uniform16 RF0 Hellinger control has completed:
`wan14b_720p_4step_hellinger_uniform16_rf0_extended_20260904i`. The old A8U8 RF0
run is not substituted. The new control has 20 unique canonical outputs,
3,200 core and full attention calls, raw checkpoint weights, exit zero and an
empty purity log. All 20 locally copied MP4 hashes match the remote originals.
`validated_summary.json` audits the frozen source/config snapshots, prompt
identity and runtime aggregation. The complete-run means are:

| Scope | Core attention ms/call | Full attention module ms/call | Inference seconds/video |
| --- | ---: | ---: | ---: |
| All20, primary | 136.912539 | 160.148765 | 39.776745 |
| Steady19, diagnostic | 136.703199 | 159.781693 | 39.696035 |

Core means selector + merge + route. Full module includes projections and
RoPE. Inference excludes loading and MP4 export. A matched dense video runtime
has not been supplied, so full-video SDPA speedup is unavailable; the frozen
capture speedups above must not be substituted.

The control's quality evaluation subsequently completed with exit zero and
an empty purity log. All 20 paired videos decoded to exactly 81 frames. Its
`quality.json` and `validated_quality_summary.json` record:

| RF0 control | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Hellinger C32/g45 Uniform16 | 11.520875884 | 0.358054837 | 0.509484093 |

These are equal-weight means across 20 videos, equivalent to frame weighting
because every video has 81 frames. PSNR/SSIM use RGB data range 255; LPIPS uses
AlexNet v0.1. The reference is the audited raw-weight dense20260826 run, and
all paired prompt hashes/seed 0 match. The earlier RF10 and A8U8 controls are
not substituted.

## Completed RF0 candidate video comparison: negative result

The anchored partition C16/g16 RF0 candidate completed its separate canonical
20-video run at 2026-09-04 18:35:10 PDT:
`wan14b_720p_4step_partition_c16_rf0_extended_20260904l`. It uses the production
port verified above, not the separate untested medoid-initialization proposal.
Both generation and quality exited zero with empty purity logs. The frozen
source/config snapshots, raw checkpoint branch, canonical prompt identities,
3,200 core/full calls, and all 20 paired 81-frame videos passed the runtime and
quality validators. All 20 copied MP4 hashes match the remote originals.

| RF0 method | PSNR | SSIM | LPIPS | Core ms/call | Full module ms/call | Inference s/video |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hellinger C32/g45 Uniform16 | 11.520875884 | 0.358054837 | 0.509484093 | 136.912539 | 160.148765 | 39.776745 |
| Partition-deficit C16/g16 | 11.377814069 | 0.344944231 | 0.516835021 | 134.628901 | 157.827592 | 39.446249 |

These are the complete 20-video means, not the steady-state subset. Candidate
steady19 diagnostics are 134.446000 ms core, 157.471071 ms full module and
39.362828 s inference. Every video has 160 attention calls. No matched dense
video runtime is available; a full-video SDPA speedup remains unreported.

The candidate has lower core/module/inference latency, but loses all three
quality metrics: PSNR -0.143061815, SSIM -0.013110606, and LPIPS +0.007350928.
Full-module latency is 1.45% lower than the RF0 control. This is a negative
quality result, **not an improved GCA method**, and PSNR >12 is not achieved.
The fixed-QKV MSE gains above did not transfer to these generated videos.

The subsequent [L33 replication and mass-sketch diagnosis](rf0_mass_probe_diagnostic.md)
preserved original BF16 QKV and same-call dense outputs over three prompts and
all four steps. It reproduces the earlier L33 regression. A fixed-sample check
also finds severe normalized-importance distortion on head30: mean TV 0.2764,
versus 0.0102 on the other sampled heads. Missing mass inside the G256 sampling
phase, rather than phase coverage alone, is the main diagnostic lead. The
subsequent exact-normalizer oracle changed the selected seeds but reduced
step-0 partition output MSE by only 0.103%; it did not repair the failure.
No output/PSNR fix is established and no dense exception is introduced.
Historical RF10 results remain unchanged. No paper files were modified.

## Original Mass-GS C32 at RF0: completed 20-video test

Run `wan14b_720p_4step_mass_gs_c32_rf0_extended_20260904r` tests the original
Mass-GS C32/g45 selector with only the reference-frame setting changed from
RF10 to RF0. The resolved configs otherwise match: Uniform16, G256, J4,
four-warp GS, the original FP32 TF32x3 global normalizer on BF16 inputs,
normalizer Q32/eight warps, and unchanged SQ1/merge/entropy/full compressed
route. Optional native-precision, TF32 and singleton-complement branches are
off. This is not partition-deficit or its smaller C16 candidate bank. Current
and historical whole-source hashes differ; algorithmic equivalence of the
default Mass-GS path is not a claim of identical source files.

Generation and evaluation completed on exclusively clean scai7 H200 GPU5,
both with exit zero and empty purity logs. The validator checked the frozen
source/config hashes, raw checkpoint branch, canonical prompts 0–19 and seed
0, 3,200 attention calls, and 20 paired videos of 81 frames each. All 20 local
MP4 hashes match the remote originals. Metrics use the same audited dense
reference and framewise protocol as Hellinger RF0 above.

| RF0 method | PSNR dB | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Hellinger C32/g45 Uniform16 | 11.520875884 | 0.358054837 | 0.509484093 |
| Original Mass-GS C32/g45 | 11.540888084 | 0.354963445 | 0.505831757 |
| Mass minus Hellinger | +0.020012200 | -0.003091391 | -0.003652336 |

Mass wins PSNR on 14/20 videos, SSIM on 8/20 and LPIPS on 12/20, with no
ties. Thus the historical RF10 PSNR/LPIPS gains partially transfer to RF0,
but the PSNR gain is small and SSIM falls. These descriptive comparisons do
not establish statistical significance or generalization. PSNR remains below
12, and there is no uniform quality winner across these three metrics.

| Method / scope | Complete operator ms/call | Full module ms/call | Inference s/video |
| --- | ---: | ---: | ---: |
| Hellinger all20, primary | 136.912539 | 160.148765 | 39.776745 |
| Mass-GS all20, primary | 142.071851 | 165.419669 | 47.151744 |
| Hellinger steady19, diagnostic | 136.703199 | 159.781693 | 39.696035 |
| Mass-GS steady19, diagnostic | 137.157585 | 160.354711 | 39.815906 |

Each video has 160 calls. Complete operator includes selection, merge and
route; full module also includes projections/RoPE. Inference excludes model
loading and MP4 export. Mass's first video took 186.532661 seconds versus
39.815906 seconds averaged over its remaining 19; the extra first-video
cost was not independently decomposed into compilation and other components.
All20 is retained as primary, not replaced by steady19. The Mass point estimate
is 3.768% slower in complete-operator time all20 and 0.332% slower steady19.
This does not establish a substantial steady-state slowdown, but it does not
prove the strict no-slowdown requirement either. Full-video SDPA speedup is
unavailable without a matched SDPA runtime; frozen-QKV ratios are not reused.

The completed test supports a small PSNR/LPIPS trade-off, not achievement of
the joint PSNR >12 and no-slowdown objective. Source artifacts are
`quality.json`, `runtime.json`, `evaluation_protocol.json`,
`source_provenance.json`, `validated_quality_summary.json`, and
`video_copy_validation.json` in the run directory under the canonical local
experiments folder. The existing exploratory workbook contains the final
metrics, separate timing scopes and formula-based RF0 differences. Historical
RF10 results and the paper repository remain unchanged.
