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

W/H was submitted after the user explicitly authorized it to proceed first.
The final real-QKV gate measured a median 168.1098 ms versus 322.1030 ms
matched dense (1.916x), with seven repetitions. The corresponding mean is
168.1416 ms and the mean-to-mean speedup is 1.9106x. Numerical validation
found zero mismatches in sampled seed cells, preserved empty slots, and
0.00165786 relative L2 against the independent frozen-QKV reference. The
20-video job is `qw_kh_c4_current_cuda_fast_v1_20`, guarded on scai7 GPU4,
using the frozen `axis_video_wh_fast20_source_20260918` source. Its first
video has started. Later temporal-kernel edits only touch the separate
`axis_video_source_20260918` checkout. A five-minute read-only heartbeat
reports first-five metrics and final completion without changing the job.
The launch/gate record is in row 1392 of `C:/GMA/tmp.xlsx`; its runtime
columns currently refer to the QKV gate, pending actual video-run averages.

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

The matched C8 r2 control uses the same real full40-head BF16 capture and
the same clean scai7 GPU1 as the temporal screen, including head-major input
and output conversions. Seven interleaved repetitions measured mean C8
166.7812 ms versus dense 323.7991 ms (1.9415x); medians were 167.1760 and
320.2604 ms. The exclusive-GPU guard passed. This replays QKV only and does
not regenerate baseline videos. A separate profiled call attributes 7.4054 ms
to probe plus selection, 86.4937 ms to merging, and 59.3748 ms to routing;
profiled component times must not be substituted for the complete-call mean.

The new current-line candidates do not have the same selection reuse as
C8 r2: C8 selects from reference query frame 0 and reuses the selected row
indices across the 21 query frames, whereas each axis candidate selects
independently for every current query line. The candidate selection rules
also differ (Hellinger versus anchor-Fisher). Consequently these timings
are comparisons of actual implementations, not an isolated C4/C8 ablation.
The current temporal FMA selector trial measured median T/W 326.3134 ms
versus dense 324.2460 ms and T/H 574.6920 ms versus dense 326.6316 ms.
Neither trial qualifies for video launch.

For N=75600, complete-call merged K/V representative count per head is
`(N / query_line_length) * (N / key_line_length) * C`. It counts generated
representatives across all query groups, not simultaneously allocated
workspace or a FLOP prediction:

| Variant | Query groups | Merged tokens per query | Total representatives/head | Relative to C8 |
| --- | ---: | ---: | ---: | ---: |
| H45/W80 C8 | 1680 | 7560 | 12700800 | 1.0000 |
| W80/H45 C4 | 945 | 6720 | 6350400 | 0.5000 |
| T21/W80 C4 | 3600 | 3780 | 13608000 | 1.0714 |
| T21/H45 C4 | 3600 | 6720 | 24192000 | 1.9048 |

Temporal selector option 4 removes only fully invalid candidate register
groups 24..31 from scalar Fisher/Gram-Schmidt work; all21 actual candidates
and the validity/tie masks remain. T/W passed validation with zero sampled
seed mismatches, correct empty clusters, and real-QKV relative L2 0.001622715.
Seven full-call repetitions measured mean 316.0781 ms versus dense
322.9933 ms (1.02188x); medians 316.0521 and 323.8596 ms. Both exclusive-GPU
guards passed. The submitted twenty-video job is
`qt_kw_c4_current_cuda_trim_v1_20`, on scai7 GPU1, guarded by
`run_tw_cuda_trim_v1`. Source `axis_video_temporal_trim_source_20260918` is
now frozen for this run. The read-only heartbeat covers both submitted
methods and reports each method's first-five quality as soon as available.
T/W remains substantially slower than C8 r2 despite passing the dense gate.
Option 3 retains all32 padded
positions and allows FMA accumulation; neither changes the four-slot
capacity, empty-cluster policy, or full compressed attention stage.

The corresponding T/H option4 trial measured median 554.8137 ms versus
dense 327.0671 ms on a clean scai6 H200. It still fails the speed condition
and has no submitted video job. W/H has completed two videos with measured
mean attention 178.0764 ms. Its approximate 1.8646x speedup uses reused
historical dense video timing and must not be confused with a matched gate.
