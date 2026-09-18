# Three directed-axis C4 video candidates

User scope: twenty videos each, 50 denoising steps, PSNR/SSIM/LPIPS against
existing dense outputs. Reuse existing C8 results; generate neither baseline.
Evaluate and report immediately after the first five videos of each method,
then continue to twenty. The user requires the complete attention operator
to beat matched dense before that method is submitted. W/H may launch first;
it does not need to wait for T/W and T/H to pass their own speed gates.
Use all exclusively idle H200s on scai6 and scai7 (explicit override of the
single-GPU default). No frames are inspected by a model.

Candidates are Q=T21/K=W80, Q=W80/K=H45, Q=T21/K=H45, each exactly four
merged representatives per key group. They are the top three C4 candidates
from the earlier QKV screen. Only T/W beat C8 r2 in that screen's mean error;
video superiority is an open question.

`attention/axis_current_c4_triton.py` implements current-line anchor-Fisher
selection, up to 32 candidates with the existing g45 geometry, Uniform16
key probes, matched-anchor first seed and three Gram–Schmidt pivots.
Normalized seed logits assign every key to one of four slots. Stable softmax
weights merge K/V; negative entropy is subtracted in the full compressed
attention. Empty slots have zero K/V and +infinity bias. There are no
singleton routes or dense self-attention exceptions. Every one of 40 layers
uses the same candidate on all 4000 self-attention calls per video.

This is a simple prototype kernel, not a speed-optimized result. Fisher
reductions use FP32 (optional FP64 diagnostic), merged K/V and merge weights
use BF16, and accumulation and entropy use FP32. An optional probe-logit
precompute was measured and disabled because it was slower. `validate_axis_current_c4`
checks seed identities, empty slots, sampled independent reference outputs,
real frozen QKV, and complete 40-head calls before video inference.

The runner uses Wan2.1 T2V 14B, 1280×720, 81 frames, UniPC flow_shift=5,
50 steps, guidance=5, seed0, canonical MovieGenVideoBench prompts0–19,
and the existing SVG norm/RoPE/forward handling. It audits 100 calls in each
layer per video and calculates framewise PSNR/SSIM/LPIPS(Alex) on all 81
decoded frames. Output filenames are 000-00.mp4 through 019-00.mp4.
First-five evaluation uses verified subset views, strict frame/video counts,
and executes before starting video 005. First-five metrics are written to
`quality_first5.json`; all-twenty metrics go to `quality.json`.

The original slow Triton generation jobs were intentionally stopped. They
are not resumed or counted as optimized results. Native CUDA trials adapt
the existing C4 MMA selector and WGMMA/TMA merger to all three directed axes,
including current-line anchor-Fisher weighting and H45 tail masking. The
CUDA compiler cache is isolated by host because scai6 uses CUDA 12.9 and
scai7 uses CUDA 13.0. All complete-call timings include layout transforms,
selection, merging, and full compressed attention on all forty heads.

Initial native CUDA microbenchmarks measured W/H at about 182 ms against
326 ms matched Flash SDPA (1.79x). Temporal variants have not yet passed:
the best current synthetic trials are about 334 ms for T/W and 586 ms for
T/H. These are engineering diagnostics, not video quality results or final
launch receipts. Triton streaming fusion, the first CUDA streaming fusion,
and FP16 residual-Gram selection were slower and are retained only as
rejected engineering controls. Do not retry those implementations unchanged.
No optimized video job is submitted without source-bound numerical validation,
real captured QKV speed checks, and an exclusive-GPU receipt. The launch gate
is enforced independently in both launcher and runner.

The reused C8 video baseline is the existing standalone c79666e Hellinger
RF0 implementation from 20260901; it is not a newly generated C8 r2 run.
Verified prompt manifests establish first-five pairing. Baseline means:
PSNR 15.6897659930, SSIM 0.5397033106, LPIPS 0.3955052650.
Historical dense attention timing is reported explicitly as reused timing.

Remote evidence lives at
`/local2/denghaoran/GridMergeAttention/experiments/axis_video_c4_20260918`;
canonical local evidence at `C:/GMA/experiments/axis_video_c4_20260918`.
`axis_video_status.py` produces compact progress and, when available,
matched quality deltas. Guard processes require clean GPUs before launching,
monitor foreign processes throughout, and reject contaminated timings.
Final candidate metrics belong in the existing `C:/GMA/tmp.xlsx` sheet.
