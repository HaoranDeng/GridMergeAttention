# RF0-only temporal phase averaging for seed selection

Exploratory hypothesis, 2026-09-06 UTC. This is a selection-only change to
Anchor32. RF0, C32/g45, Uniform16, four real seed rows, matched row anchor,
SQ1, BF16 merging and full compressed attention remain fixed. The paper and
production default are unchanged. The completed 20-video result below is
slightly worse than Anchor32; matched attention timing remains unmeasured.

## Account for index reuse without reading other query frames

The selected row indices are reused across query frames, but the selector
observes only frame zero. Test one specific component of that mismatch:
known temporal rotary position changes under a fixed-content model. This
does not assume that real query contents are constant or claim that temporal
transfer has been established as the main error source.

For post-RoPE RF0 query q_i and fixed probe keys K, define virtual queries

    q_i(a) = R_time(a) q_i, a=0,...,T-1,
    q_eff_i = mean_a q_i(a),
    z_i(a) = K q_i(a) / sqrt(d),
    z_eff_i = K q_eff_i / sqrt(d).

Only the temporal rotary pairs change. The model's existing frequencies
are omega_m = 10000^(-2m/44), m=0,...,21, for head dimension 128.
In each interleaved pair, let c_m=mean_a cos(a omega_m) and
s_m=mean_a sin(a omega_m). Apply

    (x,y) -> (c_m x - s_m y, s_m x + c_m y).

The remaining 84 spatial channels are unchanged. This uses a small fixed
coefficient table and one linear pass over RF0, not T materialized query
frames. T comes from the number of key frames, not a tuned parameter.
An existing absolute starting position cancels: the added rotation is the
relative positive offset a. Averaging phases also attenuates frequencies;
it is not just selecting the middle frame's position.

## Entropy-regularized interpretation and its boundary

For the virtual probe distributions p_i(a)=softmax(z_i(a)),

    mean_a KL(rho || p_i(a))
      = KL(rho || softmax(z_eff_i))
        + mean_a F(z_i(a)) - F(z_eff_i).

Thus softmax(z_eff_i) is the unique reverse-KL barycenter on this finite
virtual support. This identity is exact before the implementation's BF16
rounding. It is neither the arithmetic mean of those probabilities nor a
claim about the real distributions at later video frames.

Use these effective probe logits in unchanged Anchor-Fisher GS: retain
the matched real row, compute the shared Fisher curvature at its effective
logits, then select three residual-norm pivots. Return original row indices.
The actual merge reloads all four original queries from the current frame;
it never merges with q_eff. The barycenter identity does not make this GS
rule, its induced hard partition, or generated videos optimal. Motion,
content changes, key-probe sampling and finite precision remain errors.

## Difference from recorded attempts

The existing temporal_transport_router first reads real queries from all
available frames, rotates each by R(reference-a), then averages them.
This candidate reads only RF0 and applies an analytic mean of positive
virtual offsets. It does not change to RF10 or introduce a multi-frame
query-content estimator. Pre-run bounded code/history inspection found no prior
implementation or complete 20-video result for this exact rule.

Anchor32 to all 45 real candidates changed complete 20-video PSNR only from
11.636408427866783 to 11.636455906092582. Earlier full-support and alternative
geometry screens did not establish a video-quality solution. These findings
motivate a different hypothesis, but do not prove its correctness.

## Validation protocol

First check the relative rotation, independent explicit virtual average,
barycenter identity, RF0-only input boundary and original-input preservation.
Then compare the wrapper's native selections against an independent FP64
Anchor-Fisher reference on original BF16 inputs. The small preliminary
implementation computes the transform in FP64 and casts it once to BF16;
the existing selector keeps FP32 QK accumulation and FP64 Fisher/GS.
This cast is declared, not treated as exact original-Q logits.

Only a complete canonical four-step/all-40-layer/20-video PSNR measurement
can evaluate the quality target. No prefix decides promotion or rejection.
Full attention latency must later be measured against Hellinger on matched
inputs; unchanged QK count is not evidence of unchanged wall-clock time.
Do not optimize kernels or change the paper on an unverified quality claim.

The analytic module passed 14 actual CPU tests on scai7 in 0.490 seconds,
with CUDA disabled and numerical libraries restricted to one thread.
This includes all five Torch checks (zero skips), independent NumPy/complex
rotations, the reverse-KL identity and wrapper behavior. Separately, 39
configuration/dispatch/launcher checks passed locally. These CPU tests alone
do not establish native GPU agreement or complete-video quality; both were
subsequently evaluated below.

The native check is prespecified for T=1 and T=21, totaling 158,400 seed
banks. It uses an independent explicit virtual-time sum, then BF16 rounding.
Every nonzero transformed result must match bitwise; the sign of an exact
zero may differ and is counted separately. Original Q/K and all spatial
channels must remain bitwise unchanged. Dense integer fixtures are retained.
Because the BF16 phase transform can introduce small dyadic values, integer
keys alone do not imply exact FP32 accumulation. Before the first GPU run,
the dot gate is defined by the standard forward-error bound

    u = 2^-24, gamma_127 = 127u / (1 - 127u),
    abs(dot_FP32 - dot_FP64) <= gamma_127 * sum_d abs(q_d k_d).

The BF16-by-integer products are exactly representable in FP32. All dots
must satisfy this bound, with zero bound requiring exact equality. Raw
nonzero rounding differences are recorded, not hidden or counted as exact
arithmetic. This replaces an unnecessarily strict pre-run exact-dot check;
no GPU result was used to choose the bound. Final real-row selections must
still agree exactly with the independent
FP64 Anchor-Fisher reference. FP32 score audits check completeness and
audit-on/off row equality only, not FP64 score-value equivalence.

## Native gate passed

Run `phase_averaged_native_check_20260906bu`, frozen at `1e24faca`,
completed validly on one clean scai7 H200 at
`2026-09-06T07:58:41.579966+00:00`. The worker exited 0 and no foreign GPU
process was observed. All 158,400 native row banks match the independent
FP64 reference exactly, including anchor, membership, uniqueness and
audit-on/off gates. Original Q/K and spatial channels are unchanged.

T1 is bitwise identical. T21 has 171 signed-zero differences and zero
nonzero transform differences. Its 211,854 nonzero FP32 dot-rounding
differences have maximum absolute error 1.52587890625e-5 and maximum
forward-bound ratio 0.00424093754945709, with zero bound violations.
These observations use the gates fixed before GPU execution; no tolerance
or fixture was changed after seeing the result.

All 52 frozen source files were archived and independently hash-verified,
and parent completion validation passed again locally. The results SHA256
is `8cc212386122387a94948617a9004f2af11a4d45856ed008c40ca5757d6f5c15`.

## Complete 20-video result

The same immutable source, `1e24faca9ea1a6de647e93277d51714c269cacee`, ran
`wan14b_720p_4step_phase_averaged_fisher_rf0_bf16score_extended_20260906bv`.
Its worker started on scai7 GPU7 at `2026-09-06T08:01:08.971032+00:00`.
It completed validly at `2026-09-06T08:29:18.595925+00:00`, with worker exit
code 0 and no foreign process in 327 sampled GPU checks.

The run used the canonical prompts 0--19 and seed 0, the raw non-EMA
checkpoint, four denoising steps and 20 videos at 720p with 81 frames each.
All 3,200 self-attention calls used the candidate: 40 layers with 80 calls
per layer. The four merge queries remained original current-frame real rows;
only the RF0 selection reference received the analytic phase transform.

Full-resolution PSNR against the canonical dense reference was
**11.588652184850831 dB**, averaged over all 20 per-video means, each covering
81 frames (1,620 frames total). This is **-0.04775624301595194 dB** relative
to Anchor32's 11.636408427866783 dB. The tested variant did not improve
Anchor32 or reach the target strictly above 12 dB. The reverse-KL barycenter
identity remains valid for its virtual support, but this result does not
support a video-quality benefit from this phase-only selection reference.

Observed complete GCA time was **147.4337753534317 ms/call** over all 3,200
calls, or **141.48059215043722 ms/call** over the final 19 videos. The full
wrapper measured 191.88020540714263 ms/call, or 172.91082680350857 ms/call
over those final 19 videos. These are instrumented generation timings with
concurrent CPU PSNR scoring, not matched timing against SDPA or Hellinger.
No speedup or no-slowdown conclusion follows. SSIM, LPIPS and VBench remain
unmeasured. Neither the paper nor the production default is promoted.

All 20 videos, 514 frozen source files and supporting records were archived
and locally hash-verified (551 primary files). Prompt identities, all 40
paired video input hashes, per-video frame counts and aggregate means were
verified. The scai7 dense video bytes and evaluator/decoder sources also
match the earlier scai6 comparisons. `tmp.xlsx` records the result at row
770. The `psnr_20.json` SHA256 is
`d9b46a0d53b23cd462d9d399a11903b83c1e183314d7efb3da79a0614f85db7f`;
the primary archive SHA256 is
`04ca3d8eac62fdb98a3ac99070724d42e5e43f39eb6454484e18f4803b98ddc6`.
