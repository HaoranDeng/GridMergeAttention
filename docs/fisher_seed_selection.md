# Geometric Fisher seed selection

Exploratory RF0/C4 method, 2026-09-05. No training or calibration and no
production-default or paper change. Best completed video PSNR is 11.617539,
still below the requested 12 and without an established no-slowdown result.

For the existing 32 real-query candidates and Uniform16 key probes, write
their untempered logits as z_i. Set z_bar = mean_i z_i and
rho = softmax(z_bar). This is the geometric probability barycenter: rho
minimizes the mean reverse KL, mean_i KL(rho || softmax(z_i)), on the
probe support. It is not a C4 or video-level optimum.

The log-partition F(z) = logsumexp(z) has Hessian

    J(z_bar) = diag(rho) - rho rho^T.

Keep the same real-row anchor z_a. Define

    d_i = z_i - z_a
    x_i = sqrt(rho) * (d_i - dot(rho, d_i)).

Thus ||x_i||^2 = d_i^T J(z_bar) d_i. Three greedy Gram–Schmidt pivots
select complementary directions in this local quadratic approximation to
the entropy-regularized dual objective. Return the corresponding real query
rows, not the barycenter itself. All four rows feed the unchanged hard
ownership, SQ1 merge and full compressed attention.

This local curvature approximation is invariant to logit offsets but is
not a global KL bound and can miss large logit excursions. RF0 temporal
reuse, probe sampling, hard reassignment and output error remain distinct
approximations. A lower local objective or fixed-QKV MSE does not establish
better video PSNR.

## Initial three-capture screen

The prototype uses original BF16 Q/K with FP32 unquantized products, the
existing C32/G45 bank, all 21 query frames and heads 0/10/20/30 at L33.
It computes real key-block curvature, unlike earlier query-only proxies.
Three fixed choices of rho were compared in one clean-GPU screen l:

| Rule | Prompt 0 MSE | Prompt 1 MSE | Prompt 2 MSE |
| --- | ---: | ---: | ---: |
| Geometric mean of probabilities | .001488045978 | .001024960312 | .001194386655 |
| Arithmetic mean of probabilities | .001763694312 | .001140781629 | .001057356250 |
| Uniform probabilities | .001636884148 | .001507644931 | .000911993240 |

The geometric version advanced to native implementation. The FP32 native
implementation (run o) had pooled MSE .001316367900 and did not preserve
the prototype improvement. One FP64-curvature implementation (run p) had
pooled MSE .001257260255, 3.17% below matched Hellinger RF0. Native and
prototype seeds differ; these are separate output measurements, not a
numerical equivalence claim. Both still use BF16 merge and route.

Run n used the FP64-curvature implementation in all 40 layers and completed
20 canonical videos. Full-resolution, all-frame CPU PSNR is 11.617539025,
versus 11.529624325 for historical Hellinger RF10 on the identical dense
reference: +0.087914700 dB, 11 wins and 9 losses. SSIM, LPIPS and VBench
have not been measured for this candidate. This comparison does not isolate
the effect of seeds because the RF policy and merge-score dtype also differ.
The first three videos alone would have rejected it: all three are worse,
despite the positive full-set mean. A prefix screen is not evidence of a
20-video failure, and is not used as the next candidate's quality conclusion.
Complete
GCA latency was 150.026928 ms/call, or 143.250582 excluding the first video;
the full wrapper averaged 187.416967 and 174.757949 ms/call respectively.
There is no matched SDPA speedup measurement. No quality/speed success is
inferred from the single-layer output screen.

## Population quadratic reduction

Run r screens three additional fixed rules on the same captured QKV, without
changing merge, route, RF0 or C4. Let x_i be the current residual Fisher
features, and pi_i either one or the G256 block-mass estimate. An ideal
unit direction u_s = x_s / ||x_s|| reduces the total quadratic residual by

    gain(s) = sum_i pi_i (x_i dot x_s)^2 / ||x_s||^2.

Choose the real query maximizing this gain and update the residuals. This
is a one-step optimum for the stated quadratic projection among the
remaining real directions, not an optimum of the hard-partition operator.
The implementation uses the same 1e-12 norm safeguard and FP64 curvature.
The three rules are mass-weighted largest residual, unweighted total
reduction, and mass-weighted total reduction. The original G256 sampling
locations are unchanged. There is no extra dataset, fitting or calibration.
Unlike the rejected exact partition lookahead, this only projects tangent
features; it does not simulate every candidate's hard repartition.

All three completed. Pooled MSE was .001287574758 for mass-weighted largest
residual, .001634376119 for unweighted total reduction, and .001425986600
for mass-weighted total reduction. None improves on native geometric Fisher;
none advances to videos.

## Value-direction gradients and secants

For F(z)=logsumexp(z), attention on the supplied support is
V^T grad F(z). Run s compared anchor-relative value responses, Fisher
gradient changes V^T J(z_bar)(z_i-z_anchor), and mass-weighted Fisher
gradient changes. Values were centered within the U16 block, so a constant
value offset cannot change the selection. A finite-difference check verifies
the gradient expression, but it does not establish a hard-merge error bound.
Pooled complete-output MSE was .001407342784, .001357303118 and
.001292839353 respectively. None advances to videos.

Run t compares two different profile geometries: Fisher curvature at the
retained anchor, and signed Jeffreys secants. For the latter,

    x_ij = sign(log p_ij - log p_aj)
           * sqrt((p_ij-p_aj)(log p_ij-log p_aj)).

Its squared norm exactly equals KL(p_i||p_a)+KL(p_a||p_i), before any GS
projection. This is not the earlier conditional-partition Jeffreys pivot;
there is no trial hard ownership or partition objective in this selector.
No exact KL identity is claimed for later projected residuals.

Both t variants were rejected: anchor-Fisher MSE .001322050826 and secant
MSE .001436232422, worse than the matched Hellinger RF0 reference.

## Full-grid candidate population

Separate prospective run ab keeps C32 and the original geometric Fisher-GS
rule, but estimates its curvature on 40 equally spaced keys or all80 keys.
This is a quality-first probe-quadrature comparison, not a claimed speedup.
It preserves RF0, C4, real anchor, BF16 merge and full compressed route.
Earlier P80 conditional-Jeffreys/value-response negatives are not a direct
test of this unchanged Fisher rule. No result is asserted before completion.

Run fisher_probes_bf16_screen_20260905ab completed cleanly. P40 yielded
MSE 0.001683848039 and P80 0.001697500843; neither is advanced. More
accurate RF0 probe support did not improve the all-frame output here.

## Expanded-resource video decision

The user expanded the resource allowance to all exclusively idle scai6 H200s
and at most one scai7 GPU. Two bounded native/video candidates are reopened:
anchor-curvature Fisher and antipodal Fisher. Their prototype L33 MSE
(0.001322050826 and 0.001342134317) does not beat original Fisher32.
This is not a promotion or a claim that they meet a QKV quality gate.
It is a direct test of uncertain transfer from one-layer MSE to full20-video
PSNR, now using concurrent clean GPUs. Full45 previously improved L33 MSE
but slightly worsened video PSNR, so the single-layer ranking is not treated
as a definitive video ranking. Verify each native operator first; retain
RF0/C4/BF16/all-layer conditions and score every canonical video.

A third bounded video candidate is the already-screened mass-weighted
Fisher pivot (run r, prototype MSE 0.001287574758, below Hellinger RF0 but
above geometric Fisher32). Native selection reuses the original G256
normalizer and multiplies Fisher squared residual scores by estimated
block mass. The unchanged global normalizer is FP32/log2; local curvature,
LSE and GS arithmetic are FP64. This native implementation therefore needs
its own output check; no exact prototype equivalence is assumed.

The full metric suite for geometric Fisher32 completed in run
fisher_geometric_video_metrics_20260905ag: PSNR 11.61753902506619,
SSIM 0.36275850326285947, LPIPS 0.5070626009209657, all20 videos and
1620 full-resolution frames. It improves PSNR/SSIM over matched-reference
historical HF RF10, but slightly worsens LPIPS; PSNR remains below12.
The completed Hellinger RF0 BF16 control g has PSNR 11.499407266792963
against the same reference (psnr_20_20260905ah.json). Its SSIM/LPIPS remain
unmeasured as a complete20-video aggregate. No default or paper change.

The expanded-resource 20-video runs ae/af completed on clean scai6 GPUs2/3.
Anchor-Fisher ae reached PSNR 11.636408427866783 (+0.018869403 vs Fisher32,
+0.106784103 vs matched-reference historical Hellinger RF10). Antipodal af
reached 11.57873104652467. Neither reaches12, and the small gain is one
deterministic seed0 benchmark rather than a general performance guarantee.
All3200 attention calls/all40 layers were audited in each run. Observed
GCA latency was 153.064542451ms for ae and 154.265625949ms for af; CPU
scoring and other-generation workloads overlapped, so these timings are
provisional, not evidence of no slowdown. Full metric evaluations al/am
are recorded separately. No default or paper promotion.

Mass-Fisher aj completed its full20-video/all3200-call check with PSNR
11.546159342735535, below both anchor-Fisher and geometric Fisher32.
It is not promoted. Its separate full-resolution metric suite is ar;
concurrent CPU scoring makes generation timing provisional as before.

Anchor-Fisher full metrics al completed validly at2026-09-05T15:38:41Z:
PSNR11.636408427866783, SSIM0.3635055850386802 and
LPIPS0.5050196406649955 over20 videos/1620 full-resolution frames.
All three improve slightly on the matched-reference historical Hellinger
RF10 (11.529624325204969,0.3579599716779665,0.5059471754012285).
The strict PSNR>12 goal and no-slowdown requirement remain unfulfilled.

Run u keeps the geometric Fisher rule but uses all 45 real rows instead of
the C32/g45 subset. This changes the rows entering both the mean logit and
the pivot search, not the final four representatives or SQ1 route capacity.
The FP64-feature PyTorch control with 32 candidates has MSE .001257257061;
all 45 candidates gives .001205759505, 4.10% lower. The two native/output
implementations still need separate evaluation. Native run v retains the
same BF16 products and FP64 feature rule, padding the internal candidate
tile to 64 with invalid rows excluded from selection and the mean divisor.
Run v completed with native MSE .001205757827, preserving the output
improvement. Seed-set agreement with the prototype is above 99.995% on
each capture, but is not exact identity or a proof of numerical equivalence.
The historical v report's top-level `methods` list was stale; its records,
summary and stored seeds correctly identify only `fisher_candidates45`.
Raw results are retained unchanged, and the reporting list is corrected in
future screen code.

Full-grid run w began 2026-09-05 11:08:48 UTC on clean scai6 H200 GPU3,
from frozen commit 9c086d2. It uses all 40 layers, all four steps and the
canonical 20 videos. CPU PSNR scoring overlaps generation, so generation
timings are provisional observations, not a matched speed benchmark.
Run w completed all 20 videos with PSNR 11.601630453, compared with
11.617539025 for Fisher C32 and 11.529624325 for the historical Hellinger
RF10 matched-dense-reference evaluation. Full-grid selection is not promoted.
Observed GCA latency was 186.614931 ms/call (179.553071 excluding the first
video); these overlapped-scoring timings are provisional, not a matched
speed benchmark. SSIM, LPIPS and VBench remain unmeasured for this run.

The preceding SQ1 partition-deficit candidate was rejected on the first
three canonical videos: PSNR 10.372796591 versus historical Hellinger RF10
10.578613115, lower on all three. Its generation was deliberately stopped;
five videos had completed, but only the predetermined three were screened.
This is not a completed 20-video result. The separate first-seed SQ1 medoid
CPU check improved the four-seed quadrature objective by only about 2.5%
and was not advanced to a new GPU implementation.
