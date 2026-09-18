# Three directed-axis C4 video candidates

User scope: five videos each, 50 denoising steps, PSNR/SSIM/LPIPS against
existing dense outputs. Reuse existing C8 results; generate neither baseline.
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
50 steps, guidance=5, seed0, canonical MovieGenVideoBench prompts0–4,
and the existing SVG norm/RoPE/forward handling. It audits 100 calls in each
layer per video and calculates framewise PSNR/SSIM/LPIPS(Alex) on all 81
decoded frames. Output filenames are 000-00.mp4 through 004-00.mp4.

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
