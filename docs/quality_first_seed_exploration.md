# Quality-first seed selection

The user clarified the research order on 2026-09-04 (local time): first find
a simple, training-free RF0 method approaching and exceeding PSNR 12 on the
canonical four-step, 20-video suite; then optimize its implementation toward
the original no-slower-than-Hellinger requirement. Early kernel latency is
recorded but is not a quality-screen rejection gate. The final speed target
has not been waived. The paper and production default remain unchanged.

## Common BF16-score substrate

Remove only `_fp8score` from the existing Hellinger and Mass RF0 modes. Keep
C32/g45, Uniform16, the matched real anchor, SQ1, merge/layout flags and full
compressed global attention unchanged. Jeffreys uses the existing isolated
selector substitution under the Mass mode. All candidate video runs must
replace attention at every layer and call, with no dense rescue.

`analysis/quality_first_mode_contract.py` defines the exact three RF0 modes
and fail-closed configuration checks. These declarations do not prove that a
runtime used the declared method: generation instrumentation must separately
verify all 40 layers and the actual selector. BF16-score does not mean FP64
or exact merging; stored moments, weights and outputs still have rounding.
Do not silently add `_directbulk`, which changes normalization/rounding order
and still uses subtractive residual calculations and the existing threshold.

## Completed fixed-seed diagnostic

Run `quality_first_bf16_20260905b` uses the original three canonical step-0
L33 captures, with saved heads 0, 10, 20 and 30. Every complete native output
contains all 75,600 queries and all four saved heads. Original FP8 outputs and
native seed tables replicate the immutable `conditional_jeffreys_native_20260904x`
receipts. Disabling FP8 leaves every seed row unchanged. This isolates the
effect of the score cast for these inputs, not its effect on a generated video.

Complete output MSE against the saved same-call Flash reference:

| Method | Original FP8 scores | BF16 scores |
| --- | ---: | ---: |
| Hellinger RF0 | 0.0054893184 | 0.0012984453 |
| Mass RF0 | 0.0263625232 | 0.0012696144 |
| Conditional Jeffreys RF0 | 0.0264592497 | 0.0012548176 |
| Hellinger RF10 control | 0.0055866124 | 0.0013657875 |

The BF16 pooled ranking is not uniform across query frames. On RF0 alone,
over all 80 columns, MSE is 0.0003362035 for Hellinger, 0.0004790202 for Mass
and 0.0007160857 for Jeffreys. Neither this diagnostic nor the earlier local
probe score is a substitute for canonical video PSNR/SSIM/LPIPS/VBench.

The second part reconstructs ideal entropy-corrected merging with FP64
products of the original BF16-valued Q/K. It uses the exact native seed row
tables, all 945 key blocks and 10 fixed RF0 query columns, all 45 rows, all
four heads. RF10 changes **row selection only**: merge always reloads the
selected real queries from the current query frame (RF0 for this subset).

| Method | Ideal FP64 MSE | Native BF16 MSE | Native FP8 MSE |
| --- | ---: | ---: | ---: |
| Hellinger RF0 | 0.0001635575 | 0.0001778544 | 0.0052245700 |
| Mass RF0 | 0.0000612587 | 0.0012180607 | 0.0199570459 |
| Conditional Jeffreys RF0 | 0.0000590657 | 0.0012170613 | 0.0199509902 |
| Hellinger RF10 control | 0.0001208625 | 0.0001314414 | 0.0063734344 |

These columns compare exactly the same selected query coordinates against
the same true-FP64 dense reference. They must not be compared directly with
the all-21-frame table to attribute numerical error. The native-versus-ideal
residual combines merge arithmetic, finite-precision ownership, residual
thresholds and route arithmetic; it does not uniquely identify one cause.

Let rho_b be true dense block mass, rho_tilde_b compressed block mass, mu_b
dense conditional output and mu_tilde_b compressed conditional output. The
saved diagnostic verifies the signed vector decomposition

    O_tilde - O = A + B
    A = sum_b rho_b (mu_tilde_b - mu_b)
    B = sum_b (rho_tilde_b - rho_b) mu_tilde_b.

Squared error is ||A||^2 + ||B||^2 + 2<A,B>; the cross term may be negative.
The old local proxy sum rho_b ||delta_b||^2 differs from ||A||^2. The previous
15-of-945-block cache additionally concentrated over 97% of head30's weight
on five sampled rows. The new all-block reference removes this sampling
confound. Exact dense mass is used only in off-path diagnostics, never inside
a candidate inference operator.

The first run `quality_first_bf16_20260905a` was stopped after an independent
audit found an RF10 ideal-reference frame-index error. Its incomplete files
and sources are preserved; they are not promoted. Run b fixes the frame
semantics. Nine Torch CPU tests pass, including independent scalar
reconstruction, SQ1/ownership, variational identities, signed error
decomposition and an executed RF10 row-only regression.

Run b completed on exclusively idle scai6 H200 GPU 3 with no foreign compute
processes. `results.json` SHA256 is
`5a7cb6e15e37bb8f40ffec82d67b16ccb53b5d5ffae357bc98e5d9b8bb557b03`.
The original inputs, source snapshots, actual selected outputs and full
per-head/per-frame error sums are retained under the run directory.

`mean_attention_ms` and `speedup_vs_sdpa` are **unmeasured** in this diagnostic.
PSNR, SSIM, LPIPS and VBench are also unmeasured. No runtime is borrowed from
the earlier FP8 experiment. The goal is not achieved by these MSE reductions.

## Unpromoted affine candidate

`attention/affine_hellinger_seed_triton.py` explores an anchored affine span:

    u_i = sqrt(softmax(z_i)), x_i = u_i - u_anchor.

Keep the mandatory real matched anchor and choose three additional real
queries by residual norm after projection onto previously selected
differences. This removes per-profile mean centering and the original anchor
projection. There is no fitted parameter, additional data, mass weighting,
global normalizer or change to merge/route. The two projection updates are a
structural observation, not measured speed evidence.

The motivation is the local expansion KL(p || q) =
2 ||sqrt(p) - sqrt(q)||^2 plus higher-order terms on the interior of the
simplex. Affine residuals around an existing anchor preserve full
root-probability differences, including broad profiles that per-profile
centering can suppress. This is a local surrogate, not an exact minimizer
of the hard-partition objective, a monotonic greedy guarantee or a proof of
better video quality. Native parity and video evaluation remain required.

The native BF16 screen `affine_hellinger_bf16_20260905c` subsequently completed
on the same three L33 inputs. Every prompt passed all 57,600 score checks,
1,800 rank checks, all 302,400 structural group checks and audit-on/off row
equality. Hellinger/Mass/Jeffreys BF16 controls reproduced run b. Affine's
complete-output MSE was 0.0016213308, 24.8671% above BF16 Hellinger and
27.7026% above BF16 Mass; it was worse than both on all three inputs. This
candidate is not promoted or prioritized for the next video run. The screen
does not prove a video-quality outcome. Its runtime, PSNR, SSIM, LPIPS and
VBench remain unmeasured. Result SHA256:
`b8fc9ca4dc089a0ec2a34bfb68fe9c3e90ff0e4cc2f0808c9bd7f379ee05c7ea`.

## Video validation

The first quality-first video baseline is Mass RF0 with BF16 scores, retaining
the same seed criterion and all-layer C4/SQ1 algorithm. It tests the common
substrate and is not evidence of a new seed-selection method by itself.
`analysis/launch_quality_first_videos.py` owns the clean single-H200 admission
and continuous foreign-process monitor; `analysis/run_quality_first_videos.py`
freezes source/config, verifies assets and instruments all 3,200 attention
calls. The matched checkpoint is the canonical
`Wan2.1-T2V-14B-StepDistill-CfgDistill/distill_model.pt`, not Self-Forcing.
Each worker requires the same absolute model asset root at prepare/execute.
Generation timing is instrumented and must be labeled accordingly.

For quality comparison, use the canonical
`wan14b_720p_4step_dense_flash_matched_extended_ema_20260826/flash/videos`.
The existing Uniform16 RF10 run `wan14b_720p_4step_hellinger_uniform16_20260904`
was scored against a different dense run under
`wan14b_720p_4step_first_frame_global_hellinger_opt_20260829/dense_flash/videos`.
Both reference prompt manifests match, but all 20 encoded video hashes and
sizes differ. This does not establish decoded equality or inequality; do not
compare the previously published RF10 PSNR directly to a newly scored BF16
result against the 20260826 reference. Rescore that RF10 output against the
canonical target, or evaluate the new output against both targets with clear
labels. The absolute goal remains PSNR **above 12**, not merely above the
historical RF10 score. SSIM and LPIPS are reported without inventing additional
noninferiority thresholds. Historical RF10 timing lacks complete purity/exit
receipts and is not a final no-slowdown proof.

The read-only `quality_first_reference_audit.json` receipt pins all 20 encoded
videos in each dense reference and in the historical Uniform16 RF10 output.
It also records their matching prompt identities and the limitations of the
historical checkpoint/runtime provenance. Its SHA256 is
`5d75c179120fa9e2f25b8f4a9b860578da3f0489eb56aea412a0925d7f8714a5`.

After Mass BF16 generation completes, evaluate it and the historical RF10
output against the same canonical reference with the unchanged full-resolution
framewise RGB PSNR, SSIM and AlexNet LPIPS protocol, all 20 videos and 81
frames each. Do not infer video PSNR from the L33 MSE screen. If Mass exceeds
12 and beats matched-reference RF10, the video PSNR milestone is met; the
simple-method and final complete-attention speed requirements still apply.
If it does not, prioritize a matched Hellinger RF0 BF16 video control to
isolate seed selection from score precision and reference-frame policy.
Jeffreys has only a small, inconsistent diagnostic advantage over Mass, and
affine is not prioritized after its replicated negative result. Speed remains
recorded, but is not the admission gate for these quality-first experiments.

The initial Mass BF16 video attempt (`...20260905d`) failed before the first
attention call: FlashInfer loaded the shared home-directory RoPE cache, whose
binary required unavailable `libcudart.so.13`. It produced no video metric.
On scai6, restore the historical Mass run's explicit
`FLASHINFER_WORKSPACE_BASE=/local2/denghaoran/flashinfer_scai6_cuda129` and record
it in the process receipt. Verify that cache's dynamic dependencies before a
fresh generation attempt. This restores the original RoPE implementation;
do not substitute a different position-encoding operator or change attention
math to work around this environment failure. The failed run is retained.

The fresh `...20260905e` attempt restores that exact historical cache. The
RoPE binary loads successfully and links to CUDA 12; the generation source,
Mass selector, model, configuration, and merge/route sources are unchanged
from attempt d. Its source-provenance SHA256 is
`bf4380917f4d320893bf34810c46132abf8d85510b2c12ab04712fb4e7385a1e`.

Run e completed successfully at 2026-09-05 07:48:54 UTC. All 20 canonical
videos were generated, with 3,200 complete GCA calls, 16,000 head-chunk
dispatches and 80 calls in each of the 40 layers. Its 294 five-second GPU
monitor samples contain no foreign process; the purity log is empty and the
parent and worker report success. Recomputing from the 20 per-video records
gives:

| Timing scope | All 20 videos | Videos 1--19, excluding startup |
| --- | ---: | ---: |
| Complete GCA, ms/call | 140.533749545 | 133.578181410 |
| Full Wan self-attention module, ms/call | 177.946099319 | 165.150768661 |
| Inference, s/video | 49.427723703 | 40.519356367 |
| Export, s/video (separate) | 6.050866233 | 6.232972311 |

The GCA measurement includes selection, merge and route; the full-module
measurement additionally includes projections and RoPE. Both include the
generation instrumentation overhead. `speedup_vs_sdpa` is unmeasured: no
same-run SDPA denominator was collected, and historical Hellinger timing is
not a final no-slowdown proof. Runtime SHA256 is
`4672b88a9afb001f679e95b6d4ace22c8bd194752764d7a9a6f5993384101d42`.
The canonical local copy contains all 20 videos, 34 primary support files and
456 frozen source files with verified remote/source-provenance fingerprints.
The paired video evaluation is a separate run,
`quality_first_video_evaluation_20260905f`; generation completion alone does
not establish the PSNR milestone.

Run f completed at 2026-09-05 08:07:54 UTC with exit 0, an empty purity log
and 207 monitor samples containing only its selected worker. Both methods
were rescored against the same canonical 20260826 dense reference, using
all 20 videos, all 81 frames per video and full-resolution 720p RGB decoding:

| Method | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Mass-GS RF0, BF16 merge scores | 11.533208174 | 0.356532663 | 0.505449248 |
| Hellinger Uniform16 RF10, historical FP8 merge scores | 11.529624325 | 0.357959972 | 0.505947175 |

Mass minus Hellinger is +0.003583848 dB PSNR, -0.001427309 SSIM and
-0.000497928 LPIPS. Mass wins on 9/20 PSNR pairs, 7/20 SSIM pairs and 11/20
LPIPS pairs. It does **not** reach PSNR 12 and is not promoted as a better
seed method. Hellinger's recomputed summaries and all per-video values
replicate the historical same-reference evaluation exactly. The comparison
still changes both RF policy and merge-score precision; it does not isolate
seed-selection quality. The matched Hellinger RF0 BF16 control is the next
video experiment. Neither this evaluation nor the L33 error reduction proves
the final no-slowdown requirement. VBench remains unmeasured.

Run f's result SHA256 is
`03a5b344329f6d3e535bdf342139457c96841970532615adbefb5b304d372673`.
The complete results, input fingerprints, frozen evaluator, logs and parent
completion are retained in the canonical local experiment directory.

An isolated [SQ1-aware partition-deficit prototype](sq1_partition_deficit_seed_selection.md)
is being developed separately. It corrects the represented family used by
the old deficit score and has mathematical CPU tests, but no native parity,
full-output quality, video quality or speed result yet. It is not used by
either completed video method or by the matched Hellinger control.

The matched Hellinger RF0 BF16 control was launched as
`wan14b_720p_4step_hellinger_rf0_bf16score_extended_20260905g` on the same
clean scai6 H200 GPU 3. It uses the frozen generation code at `5649116`, the
same checkpoint, canonical prompts and restored FlashInfer cache as Mass e.
Its source-provenance SHA256 is
`6e150a32842aa85af48c4528bdb5f46264448396e6b93dfb7f6e90201cafb8d8`.
Generation completed at 2026-09-05 08:32:50 UTC with exit 0, empty purity log
and 268 five-second monitor samples containing no foreign process. All 20
videos, 3,200 complete calls, 16,000 head-chunk dispatches and 80 calls per
layer were verified. Runtime, including generation instrumentation, is:

| Timing scope | All 20 videos | Videos 1--19, excluding startup |
| --- | ---: | ---: |
| Complete GCA, ms/call | 140.477673237 | 133.594829000 |
| Full Wan self-attention module, ms/call | 177.815482726 | 165.133731927 |
| Inference, s/video | 49.848335857 | 40.511225402 |
| Export, s/video (separate) | 1.806690821 | 1.790898334 |

No same-run SDPA denominator was measured. These independently generated
QKV streams do not establish the final matched-input Hellinger speed gate.
Runtime SHA256 is
`96a08f52eb11971794e9d8b182256200230925dfd210147d30ff7afc292daca6`.
All 20 videos and 456 fingerprint-verified frozen source files are archived
locally. The transfer archive SHA256 is
`451652089dd673308a848272abc8e7d7c983f2979a0852cf9fb89183268a6663`.

`run_quality_first_evaluation.py` now admits this exact g source only with
explicit `--candidate-method hellinger_rf0 --candidate-dir .../g/videos`.
Its 25 CPU tests preserve the original Mass e contract and reject wrong
Hellinger labels, source bytes, incomplete generation, altered prompt pairing
and changed full-resolution metric settings. The two candidates use separate
frozen evaluation runs; this extension does not evaluate g before it finishes.
The paired evaluation `quality_first_hellinger_video_evaluation_20260905h`
uses frozen code at `97c14f6` and is in progress. PSNR, SSIM and LPIPS must
come from its completed results; VBench remains unmeasured.

Update 08:52 UTC: evaluation h was deliberately stopped after three candidate
videos to prioritize new seed methods following the user's instruction.
Its parent exited after SIGTERM and stopped only its worker. Partial h
metrics are not a completed 20-video result; generation g remains complete.
No precision-control replacement run is planned.

SQ1 screen i completed on the clean H200 with exit 0 and both native
score/rank checks passing on all three captures. Pooled full-output MSE:
Hellinger RF0 0.00129844528807, Mass RF0 0.00126961438557,
SQ1 partition RF0 0.00129624704054. SQ1 improves two prompts and worsens
one; its 0.17% mean advantage over Hellinger is not convincing evidence of
video improvement. Result SHA256:
`8d900cc84b25d089a181d7de0bda6ea80a4ce94d1aa96c0f134f082864bc0da3`.
The next decision uses end-to-end PSNR, with the fixed first three canonical
videos as an early rejection check. Neither this screen nor a partial video
run meets the 20-video target. Runtime and SDPA speedup remain unmeasured.

## Candidate-history checks during video generation

The singleton-complement probe mask is already implemented and was rejected
after two RF10, FP8-score frozen-QKV screens (+1.2118% and +1.1518% MSE).
It is not a new candidate; no RF0/BF16 video evidence was found for it. See
[the original scope and derivation](singleton_complement_seed_selection.md).

Current G256 indices are `295*v`, so they visit only 16 of 80 column
positions. Integer midpoints `((2*v+1)*75600)//512` visit all 80 columns,
with three or four probes per column, at the same total budget. No completed
alternative-location G256 experiment was found in the inspected repository
and scai6 artifacts. This is only a possible quadrature ablation, not a
demonstrated improvement or SQ1 correction. The
[existing sampling audit](rf0_mass_probe_diagnostic.md) found that within-phase subsampling, rather
than column-phase mismatch, dominated head30 importance error. Existing
exact-normalizer interventions also do not establish that fixing the
denominator improves full BF16 Mass output or video quality. Leave the
running comparison and production probe policies unchanged.
