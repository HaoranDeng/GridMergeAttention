# Joint-frame mass-weighted seed selection

Status: **not promoted after the completed canonical 20-video test**
(2026-09-04). Joint C16 reaches PSNR **11.467133951 dB**, SSIM
**0.347324362**, and LPIPS **0.508273922**: all three are worse than both
same-reference RF10 and single-frame mass-GS, and PSNR remains below 12.
The earlier positive fixed-QKV screens did not translate into a video-quality
gain. In the full-40-head benchmark, C16 still lowers captured-output MSE by
1.60%/4.04% versus mass-GS and averages **126.255144 ms/call**, or
**2.625825x versus matched Flash SDPA**. That is a separate frozen-QKV
measurement, not a video-generation speedup. C32 fails the no-slowdown gate
and was not tested on 20 videos. Neither joint variant is a validated GCA
default. The training-free, pure all-layer operator is unchanged, and the
paper repository remains unchanged.

## Motivation from the paper

The paper's oracle averages the entropy-regularized projection gap over all
query frames and rows. The current RF10 selector instead chooses one set of
row indices from query frame 10. Those indices are reused over all 21 frames,
but the merge uses the actual query vectors of each current frame.

For fixed disjoint cells, the complete variational gap of query i at frame a
can be written as

\[
\Delta_{ai}=-\log\left(\sum_b\rho_{aib}e^{-\epsilon_{aib}}\right)
\leq\sum_b\rho_{aib}\epsilon_{aib},
\]

where rho is dense block mass and epsilon is the local restricted gap.
For the measured SQ1 operator, that local gap includes the block's four bulk
cells and its disjoint singleton. No additional singleton term is needed in
this full-block expression. This is the same fixed-cell identity
used by [mass-weighted selection](entropy_mass_seed_selection.md), not a
guarantee about seed search.

The proposed change approximates the temporal average using two deterministic
midpoint frames, a=5 and a=15 for T=21. It keeps their distributions separate.
It does not average query vectors before softmax.

## Shared indices, separate projections

Fix a key block and query column. At sampled frame a, candidate row i has
Uniform16 probe logits x_ai, distribution p_ai=softmax(x_ai), and centered
square-root feature phi_ai. Define

\[
d_{ai}=\operatorname{LSE}(x_{ai})
-\operatorname{LSE}(q_{ai}K_{\mathrm{global}}^\top/\sqrt d),
\qquad
v_{ai}=\exp(d_{ai}-\max_{a',i'}d_{a'i'}).
\]

The global sketch contains the same 256 flat key indices for both frames.
Each frame/query needs its own normalizer. The common maximum spans both
frames and candidates, preserving relative frame masses. Separately
normalizing candidate weights within each frame would change the criterion.
The equal quadrature factors are common to both frames and all candidates,
so they cancel from the pivot ranking; these weights are not reported as
absolute attention masses.

Let Pi_a(S) project onto the selected feature span in frame a. Starting with
the geometrically matched row, choose three additional shared row indices by

\[
s_t=\arg\max_{i\notin S_{t-1}}
\sum_{a\in\{5,15\}}v_{ai}
\left\|(I-\Pi_a(S_{t-1}))\phi_{ai}\right\|_2^2.
\]

Each pivot updates the residuals separately in the two frames. This allows
different projection coefficients at different frames, consistent with
sharing only row indices in GCA. Concatenating features and applying one
projection would instead impose common coefficients, a different surrogate.

## Controlled screen

All variants keep Uniform16 local probes, G256 normalization, four real seed
rows including the fixed first row, and the unchanged production C4+SQ1
merge and global compressed attention.

| Variant | Candidate bank | Router frames | Purpose |
| --- | --- | --- | --- |
| gs_256 | C32, geometry 45 | 10 | Existing mass-GS reference |
| gs_c16_256 | C16, geometry 16 | 10 | Reduced-bank control |
| gs_joint2_c16_256 | C16, geometry 16 | 5, 15 | Equal local-QK budget candidate |
| gs_joint2_c32_256 | C32, geometry 45 | 5, 15 | Temporal benefit without reducing bank |

Geometry 16 uses radius 4, four initial uniform rows, and farthest filling.
It is not a truncation or guaranteed subset of the old C32 bank. The matching
C16 single-frame control isolates the temporal change from this bank change.
The C16 two-frame variant has the same local candidate-probe QK count as
C32 one-frame. A second global sketch adds arithmetic and all-frame scoring
may affect implementation latency; neither equal cost nor equal speed is
assumed. C32 two-frame is a quality diagnostic, not a speed-qualified method.

## First matched fixed-QKV screen

Run `joint_frame_mass_screen_20260904d` completed on an exclusively idle
scai7 H200. The matched inputs are `call020_holdout` and `call100_holdout`,
heads 0, 10, 20, and 30, with the unchanged production operator evaluated
on all 21 query frames. Every method is compared against the same dense
attention output for its capture. The reference implementation supplies seed
rows; these measurements are not native selector latency or video quality.

| Method | call020 MSE | Change vs mass-GS | call100 MSE | Change vs mass-GS |
| --- | ---: | ---: | ---: | ---: |
| gs_256 | 0.005324654 | reference | 0.007269767 | reference |
| gs_c16_256 | 0.005234714 | -1.69% | 0.007057184 | -2.92% |
| gs_joint2_c16_256 | 0.004973360 | -6.60% | 0.006177632 | -15.02% |
| gs_joint2_c32_256 | 0.004853555 | -8.85% | 0.006106533 | -16.00% |

Reducing the bank alone helps aggregate MSE in these captures. It therefore
accounts for part, but not all, of the joint-C16 result. Holding that C16 bank
fixed, adding the two independent-frame profiles lowers MSE by **4.99% and
12.46%** relative to `gs_c16_256`. The C32 joint control also improves without
changing the bank. These two same-bank contrasts support a temporal-profile
benefit within this screen; they do not prove a general optimum for two
frames or the chosen frame locations.

### Selected-frame effects versus transfer to other frames

For a fair temporal-transfer diagnostic, define the common off-router set
U = {0,...,20} \ {5,10,15}. These 18 frames are not used to select seeds by
either RF10 or the two-frame variants. Merely excluding frames 5 and 15
would leave the RF10 reference's own sampled frame inside the comparison.
For each subset, average the recorded frame MSEs before computing the
relative change; do not average framewise percentage changes.

The following percentages compare each joint variant with `gs_256`:

| Frame subset | Joint C16, call020 | Joint C16, call100 | Joint C32, call020 | Joint C32, call100 |
| --- | ---: | ---: | ---: | ---: |
| Common off-router U, 18 frames | -6.39% | -14.34% | -7.76% | -14.87% |
| New sampled frames 5 and 15 | -24.98% | -30.94% | -35.26% | -35.12% |
| Old sampled frame 10 | +43.40% | +18.48% | +41.09% | +16.28% |
| Boundary frame 0 | -7.22% | -29.87% | -9.26% | -22.62% |
| Boundary frame 20 | -5.68% | -3.64% | -6.01% | -5.64% |

Joint C16 improves 18/18 and 15/18 common off-router frames; joint C32
improves all 18 in both captures. Against the reduced-bank C16/RF10 control,
joint C16 improves the U mean by 4.32% and 11.62%, with 18/18 and 17/18
individual U frames improved. Thus the benefit is not confined to the two
new sampled frames.

There is nevertheless a pronounced selected-frame effect: the previous
RF10 advantage at frame 10 is lost, while frames 5 and 15 improve much more
than the off-router mean. In call020, frame-10 MSE rises from 0.003521474
to 0.005049875 for joint C16; in call100 it rises from 0.004990003 to
0.005912391. This is a real tradeoff, not numerical noise. Joint C16 has
three additional regressions in call100 versus `gs_256`: frame 8 (+2.27%),
11 (+1.27%), and 12 (+5.36%). No joint variant regresses boundary frame 0
or 20 relative to `gs_256` in these two captures.

The controls are not uniformly ordered. C16/RF10 alone improves only 7/18
U frames in call100 despite improving its aggregate, and has small boundary
regressions at call020/frame0 (+0.70%) and call100/frame20 (+0.49%). Joint
C32 improves overall MSE by a further 2.41% and 1.15% versus joint C16,
but its call100/frame0 error is 10.33% higher than joint C16's, while still
below the mass-GS baseline. More candidate rows do not dominate every frame.

### Interpretation and acceptance checks at the screen stage

This was positive development-screen evidence for independently projected
temporal profiles: both same-bank comparisons improve overall MSE, and
improvements extend beyond the router frames. It warranted additional
fixed-configuration captures and a native implementation/parity benchmark.
It does not justify changing or tuning the two sampled frames after seeing
these curves.

The two capture files have been used in earlier selector development, so
their historical `holdout` filenames do not make this an untouched final
test set. The recorded frame MSE averages all four selected heads; it cannot
establish that each head improves. Adjacent frames and heads are not
independent statistical replicates, and no significance claim is made from
their win counts.

Before a 20-video comparison, the same frozen candidate had to pass native
numerical checks and matched complete-attention timing including selection
and both global normalizers. The existing mass-GS implementation reached
11.610300592 dB on 20 videos; the present attention-MSE percentages cannot
be converted into an expected video-PSNR gain or establish PSNR above 12.
Joint C16 is the equal-local-QK-budget candidate. Joint C32 remains an
additional-cost quality diagnostic unless its own complete-attention timing
meets the same acceptance condition. Neither inherits another variant's
runtime or video-quality result.

Artifact: `C:\GMA\experiments\joint_frame_mass_screen_20260904d\screen.json`.
Per-variant metadata records the actual candidate bank, frame list, probe
counts, anchor, and projection/mass rules. No native timing, SDPA speedup,
or generated-video metrics for either joint variant are reported here.

## Additional eight-head validation

Run `joint_frame_mass_transfer_screen_20260904d` evaluates the same frozen
four variants on historical `call020_train` and `call100_train` captures,
using heads 0, 5, 10, 15, 20, 25, 30, and 35 and all 21 query frames. These
historical `train` filenames identify archived QKV tensors used read-only
for validation here: no training, parameter fitting, or frame retuning was
performed. The run completed on an exclusively idle scai7 H200. It also
records an all-frame MSE for each selected head.

These results are kept separate from the four-head screen. Do not average
the four-head and eight-head raw MSEs without specifying weighting and
interpretation.

| Method | call020_train MSE | Change vs mass-GS | call100_train MSE | Change vs mass-GS |
| --- | ---: | ---: | ---: | ---: |
| gs_256 | 0.004242380 | reference | 0.006699690 | reference |
| gs_c16_256 | 0.004218245 | -0.57% | 0.006319649 | -5.67% |
| gs_joint2_c16_256 | 0.004092036 | -3.54% | 0.005931572 | -11.46% |
| gs_joint2_c32_256 | 0.004024181 | -5.14% | 0.006168847 | -7.92% |

At a fixed C16 bank, the joint selector improves overall MSE by 2.99% and
6.14% relative to C16/RF10. Both C32 same-bank temporal comparisons also
improve. Thus the additional captures retain a temporal-profile benefit,
although its magnitude varies and joint C32 no longer beats joint C16 in
the second capture.

Using the same common off-router set U excluding frames 5, 10, and 15:

| Capture and variant | U mean change vs mass-GS | Improved U frames | Improved all-frame heads | Frame-10 change |
| --- | ---: | ---: | ---: | ---: |
| call020_train, joint C16 | -3.52% | 18/18 | 5/8 | +42.12% |
| call100_train, joint C16 | -11.35% | 18/18 | 8/8 | +7.88% |
| call020_train, joint C32 | -4.29% | 18/18 | 5/8 | +40.81% |
| call100_train, joint C32 | -7.46% | 18/18 | 8/8 | +11.75% |

Joint C16 also lowers the U mean relative to the C16/RF10 control by 2.53%
and 5.81%. Its sampled-frame 5/15 mean improves by 18.88% and 20.38%
against mass-GS, larger than the off-router gains. Its frame-0/frame-20
changes are -5.41%/-5.25% in the first capture and -12.25%/-15.57% in the
second. Both joint C32 boundary-frame comparisons also improve. There is
still no joint-variant boundary regression versus mass-GS in these four
capture groups, but frame 10 consistently gives up its RF10 advantage.

### Head-level heterogeneity

The first transfer capture exposes regressions hidden by the aggregate:
joint C16 worsens heads **5 (+13.98%), 30 (+6.43%), and 35 (+8.15%)**
relative to mass-GS, while the other five heads improve. At the same C16
bank, these same three heads also regress when adding joint-frame selection;
the effect cannot be attributed solely to reducing the bank. Joint C32
regresses the same three heads by +4.87%, +0.27%, and +0.15%, respectively,
despite improving the overall mean. In the second transfer capture, both
joint variants improve all eight heads versus mass-GS.

The saved `frame_mse` averages heads, and `head_mse` averages frames. These
separate marginal summaries do not reveal per-head error on U, or whether
a regressing head's error is concentrated at frame 10. Do not infer
head-by-frame consistency from them. No per-head exception or fitted
selection schedule is introduced to address these observations.

The aggregate and off-router consistency across the two screens support
native development of the frozen joint-C16 candidate, while the head-level
regressions remain a meaningful limitation. This is still fixed-QKV
validation, not evidence of a generated-video gain or qualified runtime.
Keep the actual candidate's native parity, complete-attention latency,
and eventual same-reference 20-video result separate from these screens.

Artifact: `C:\GMA\experiments\joint_frame_mass_transfer_screen_20260904d\screen.json`.

## Full-40-head native quality and timing

Run `joint_frame_mass_native40_benchmark_20260904d` completed with
`exit_code=0` and an empty `purity.log` on an exclusively idle H200. It uses
the same archived `call020_holdout` and `call100_holdout` tensors as the first
screen, now evaluating the native dispatcher's output on **all 40 heads and
all 21 query frames**. This expands head coverage; it is not two new captures
or independent generated-video evidence.

The complete operator retains four seed-conditioned bulk cells plus the
inherited disjoint singleton, one global compressed softmax, and the same
four shared row indices. Both joint variants use frames 5/15, P16/G256,
independent per-frame projections, one common frame/candidate mass maximum,
and four GS warps. The original BF16-to-FP32 TF32x3 global normalizer is used;
optional native-BF16/TF32-single normalizer optimizations are disabled.

### Native output error against matched dense attention

| Method | call020 MSE | Change vs mass-GS | call100 MSE | Change vs mass-GS |
| --- | ---: | ---: | ---: | ---: |
| Hellinger C32 RF10 | 0.005058141 | +15.99% | 0.009085572 | +11.75% |
| Mass-GS C32 RF10 | 0.004360697 | reference | 0.008130175 | reference |
| Joint mass-GS C16 | 0.004291085 | -1.60% | 0.007801440 | -4.04% |
| Joint mass-GS C32 | 0.004129943 | -5.29% | 0.007350972 | -9.58% |

C16 improves the full-head output MSE in both captures, but its gains are
substantially smaller than the 6.60%/15.02% gains in the original four-head
subset. Report the full-head numbers as this benchmark's quality result;
do not transfer subset gains to the full operator or pool these overlapping
measurements as independent observations. C32 lowers MSE further, so its
rejection below is a speed-budget decision, not a claim that it fails to
improve the fixed-QKV quality proxy.

### Matched complete-attention latency

Each timed method produces contiguous `[1,75600,40,128]` output from the
same BF16 QKV. Timing includes selection, both frame-specific global
normalizers where applicable, merge, route, and the contiguous output. It
excludes model QKV/output projections and RoPE, so it is neither full model
attention-module latency nor end-to-end video generation time. SDPA is
PyTorch's forced `FLASH_ATTENTION` backend on the same inputs.

After two warmup calls, the benchmark records four samples of five calls
per method, alternating forward/reverse method order. The table reports the
mean milliseconds per call; speedup is the matched SDPA mean divided by
that method's mean for the same capture.

| Method | call020 ms/call | call020 speedup vs SDPA | call100 ms/call | call100 speedup vs SDPA |
| --- | ---: | ---: | ---: | ---: |
| Hellinger C32 RF10 | 128.402 | 2.583x | 128.928 | 2.570x |
| Mass-GS C32 RF10 | 128.233 | 2.587x | 128.806 | 2.572x |
| Joint mass-GS C16 | 125.979 | 2.633x | 126.532 | 2.619x |
| Joint mass-GS C32 | 143.521 | 2.311x | 144.230 | 2.297x |
| Dense SDPA Flash | 331.702 | 1.000x | 331.346 | 1.000x |

C16 is 2.423/2.396 ms faster than RF10 (-1.89%/-1.86%) and
2.254/2.274 ms faster than mass-GS. C32 is 15.119/15.302 ms slower than
RF10 (+11.77%/+11.87%) and does not qualify for promotion under the
no-slowdown requirement. The sample standard deviations for C16 are
0.171/0.128 ms, versus 0.585/0.174 ms for RF10. These four timing samples
characterize this benchmark, not general performance confidence intervals.

### Numerical checks and promotion scope

Native-versus-Torch selector parity was checked on heads 0/10/20/30, not
all 40 heads. C16 selects the same seed sets in both captures: call020 has
rare ordering differences (slot agreement 0.99999839), while call100 is
bitwise seed-identical. The resulting C16 attention output is bitwise
identical to the Torch-table path in both checked captures. All checked
tables retain the matched first anchor, distinct in-range rows, and the
intended query frames. Dispatcher output is bitwise identical to direct
native-table output for all checked variants.

C32 also passes the recorded numerical gate, but it is not fully identical:
call100 has same-set fraction 0.99999672 and output relative L2
0.0001030518 versus the Torch-table path. Its maximum absolute output
difference is 0.037109375. Do not conflate a passed tolerance check with
universal exact seed parity, or imply that four-head parity establishes
every head and every future attention call.

This benchmark qualified the frozen joint-C16 method for the canonical
20-video test: it improved the full-head fixed-QKV proxy and passed the
matched native timing screen. The completed video test below subsequently
failed the quality goal. C32 stays a non-promoted quality diagnostic and
received no 20-video test because its native latency failed the speed gate.
No frame tuning, head-specific exception, dense rescue, or extra
compressed-slot budget was introduced.

Source: `C:\GMA\experiments\joint_frame_mass_native40_benchmark_20260904d\screen.json`.
Implementation: [native benchmark](../analysis/benchmark_joint_frame_mass_seeds.py).
The canonical exploratory workbook records this result separately from
video experiments and from the earlier four/eight-head screens.

## Canonical 20-video result: quality failure

Run `wan14b_720p_4step_joint_frame_mass_span_extended_20260904e`
completed on scai7 H200 GPU5. Its validation report is `valid: true`:
generation and quality evaluation both exited with code 0 and empty purity
logs. All 20 canonical files `000-00.mp4` through `019-00.mp4` have aligned
prompt/seed manifests and 81 evaluated frames each, for 1,620 frames total.
The run uses seed 0, four denoising steps, the unchanged C16/g16 frames-5/15
selector and original TF32x3 global normalizer in every attention call.
Both runtime and run manifest record the loaded weights as `raw_state_dict`;
the requested EMA flag does not select a separate EMA state in this checkpoint.

The fixed dense reference is
`/local2/denghaoran/GridMergeAttention/experiments/wan14b_720p_4step_dense_flash_matched_extended_ema_20260826/flash/videos`.
RF10 and mass-GS below are their existing evaluations against exactly this
same reference, not the earlier dense20260829 comparison. PSNR is mean
framewise dB, SSIM is framewise RGB with range 255, and LPIPS is AlexNet
v0.1 on full-resolution frames without resizing or cropping.

| Method | PSNR dB (higher better) | SSIM (higher better) | LPIPS (lower better) |
| --- | ---: | ---: | ---: |
| Hellinger C32 RF10 | 11.529624325 | 0.357959972 | 0.505947175 |
| Mass-GS C32 RF10 | 11.610300592 | 0.357684962 | 0.500155182 |
| Joint mass-GS C16 | 11.467133951 | 0.347324362 | 0.508273922 |

Relative to RF10, joint C16 changes PSNR/SSIM/LPIPS by
**-0.062490374 dB / -0.010635609 / +0.002326747**. Relative to mass-GS,
the changes are **-0.143166640 dB / -0.010360600 / +0.008118740**.
Every direction is worse. This fails the >12 dB goal despite the earlier
fixed-QKV improvements; the available measurements do not establish the
mechanism of that mismatch. This round does not promote joint C16 or C32.

### Video-run runtime, separate from the matched-QKV speedup

Every video records 160 complete attention-operator calls and 160 full
attention-module calls. There are 3,200 calls of each kind in all20 and
3,040 in steady19 (zero-based videos 1 through 19). The following means
were recomputed from individual records and checked against runtime.json:

| Scope | Complete operator ms/call | Full attention module ms/call | Model inference s/video |
| --- | ---: | ---: | ---: |
| All20 | 135.744561 | 159.099348 | 39.650595 |
| Steady19 | 135.478054 | 158.578163 | 39.545298 |

The operator includes seed selection, merge and route; the full module also
includes projections/RoPE. Inference time excludes model loading, random-noise
preparation, CPU export and video encoding. Export is recorded separately
(mean 0.589408 s/video all20, 0.586881 s/video steady19).

Separately, pooling equal call counts from the two matched frozen-QKV
captures gives C16 **126.255144 ms/call** and SDPA **331.523956 ms/call**,
hence **2.625825x versus SDPA**. This benchmark includes the complete
selector/merge/route and contiguous output, but excludes model projections
and RoPE. It is not video end-to-end speedup. Do not divide the video-run
135.744561 ms by a historical or fixed-QKV SDPA latency to claim one.

The predecessor `wan14b_720p_4step_joint_frame_mass_span_extended_20260904d`
was aborted during model loading when a foreign process entered GPU6.
It produced **zero videos and zero runtime records**, exited 143, and has
invalid timing. Its artifacts are retained for audit, not mixed into the
clean GPU5 `e` run or any quality/runtime aggregate.

Sources: `C:\GMA\experiments\wan14b_720p_4step_joint_frame_mass_span_extended_20260904e\validation_report.json`,
`quality.json`, `runtime.json`, and `evaluation_protocol.json` in that run;
same-reference controls in
`C:\GMA\experiments\wan14b_720p_4step_entropy_mass_span_extended_20260904b\quality.json`;
aborted-run provenance in
`C:\GMA\experiments\wan14b_720p_4step_joint_frame_mass_span_extended_20260904d\aborted_run.json`.

## Limitations

- Two fixed midpoint frames are quadrature, not an unbiased estimate of the
  21-frame mean. RoPE, first-frame behavior, or changing ownership can make
  this approximation poor.
- Mass-weighted Hellinger-span novelty is still a surrogate for the actual
  hard-partition simplex gap. Its pivot score is not the exact decrease of
  total population reconstruction error or the entropy-regularized objective.
- Selecting four common indices does not imply a common linear projection
  or fixed conditional distributions across frames.
- A fixed-QKV output improvement is not a generated-video improvement. The
  completed joint-C16 test demonstrates this distinction: all three video
  fidelity metrics regress despite its capture-level gains. The required
  end state remains PSNR above 12 on 20 canonical four-step videos and a
  matched complete-attention runtime no slower than RF10; this round fails it.

Implementation: [reference screen](../analysis/screen_partition_population_seeds.py).

## Implementation checks

The dedicated suite passed all 11 tests with actual PyTorch 2.8.0 on CPU,
with CUDA devices hidden. Eight tests check the vectorized helper through
an independent NumPy adapter/reference. Three additional tests execute the
actual Torch helper and table builder, including FP32/FP64 helper checks,
all three new table variants on a synthetic [1,21,45,16,4] head/frame/row/
column/dimension grid, and 12 manually reconstructed QK-to-seed tuples.
Checks cover the fixed anchor, distinct rows, candidate membership, common
mass shifts, independent projections, and candidate-to-output indexing.

This is synthetic arithmetic and indexing validation only. It does not
measure real captured-QKV error, native Triton parity, full attention latency,
SDPA speedup, or generated-video quality. The local and remote source hashes
were checked equal. The validation report is stored in
`C:\GMA\experiments\joint_frame_mass_cpu_validation_20260904d`.
