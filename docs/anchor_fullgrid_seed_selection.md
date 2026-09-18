# Anchor-Fisher over all real grid rows

Prospective RF0-only candidate,2026-09-05. This is not the previously tested
geometric-Fisher full45 variant. The retained real anchor defines the
curvature rho=softmax(z_anchor), independent of the candidate population.
Replacing C32/g45 with all45 real rows therefore leaves the metric itself
unchanged and removes only the candidate-subset approximation. Uniform16,
C4+SQ1, current-frame actual-Q merge and full compressed route stay fixed.

For delta_i=z_i-z_anchor, select three residual-norm GS pivots in
sqrt(rho)*(delta_i-<rho,delta_i>), retaining the real anchor. No extra
frame information, learned parameter or synthesized query is used.
Considering all45 rows simplifies the candidate definition, but raises
selection work and does not guarantee a better greedy span or hard merge.
The previous geometric45 video loss is not reused as an Anchor45 result.

The native tile pads45 candidates to64. Invalid rows must never enter
anchor curvature, pivots or basis extraction. Original BF16 QK products
accumulate inFP32; curvature and GS useFP64. The matched C32 control and
full45 prototype, actual padded-kernel arithmetic, and native complete
output are checked separately before any full20-video claim.

## Completed full20 result: effectively flat

`wan14b_720p_4step_fisher_anchor_full45_rf0_bf16score_extended_20260905av`,
frozen 3a17200, completed 2026-09-05 22:57 UTC with all20 canonical videos,
all40 layers/all4 steps, the raw_state_dict checkpoint and no observed
foreign GPU process. PSNR is **11.636455906092582**, versus C32 Anchor-Fisher
**11.636408427866783**: only +0.000047478225799 dB, not a meaningful gain.
This does not reach the PSNR>12 target and is not promoted over Anchor32.

Instrumented complete GCA attention averaged 181.932258644104 ms per call
(steady last19: 172.0760617958872 ms); full-module mean was
223.8601205587387 ms. CPU PSNR evaluation overlapped generation. No matched
Hellinger/SDPA speedup is established. Completed full20/1620-frame metrics in
`fisher_anchor_full45_video_metrics_20260905aw` are SSIM 0.36402780797145295
and LPIPS 0.5079060522126563. Relative to Anchor32, SSIM is slightly higher
but LPIPS is worse; this does not change the decision. Local archives contain all20
videos plus frozen source/support; the exploratory workbook row is 554.
