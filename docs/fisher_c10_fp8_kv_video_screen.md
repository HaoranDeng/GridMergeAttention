# Fisher C10 FP8 intermediate K/V, four-step quality screen

This explicitly requested quantization experiment compares scaled FP8 stored
bulk K/V with the existing BF16 Fisher C10 video operator. It uses 15 canonical
MovieGen prompts (indices 0–14), seed 0, 81 frames, 720p, the same raw StepDistill
checkpoint and all four denoising steps. Every self-attention layer/call uses
the same operator: 2,400 complete calls and 24,000 four-head merge/route calls.
No VBench is requested. The required result for this screen is video PSNR.

`fisher_geometric_c10_rf0_fp8kv` enables the existing scaled-FP8 storage path:
each merged K and V vector gets its own FP32 absmax/448 scale (floor 1e-8),
computed from the current FP32 merge result. Vectors are stored as E4M3 FP8,
then decoded into BF16 operands inside the full compressed attention route.
Q/K score and merge matrix operands remain BF16. FP32 bias, full route support,
query-column singleton, Fisher selection and ten real seeds are preserved.
The singleton retains the original BF16 representation. No calibration data,
learning, sparse route or dense fallback is introduced.

The experiment uses the existing sixteen-physical-slot C10 video adapter,
including its six empty padding slots, J4 and one merge loop stage. It is not
the later native ten-slot kernel. Runtime must not be compared directly with
the native C10 294.13 ms figure. The 128-dimensional K/V payload is halved;
two FP32 scales add 8 bytes per K/V vector pair, versus 512 bytes for its BF16
payload. Bias storage is unchanged. This is a storage calculation, not a
measured whole-call speedup.

The baseline is `wan14b_720p_4step_fisher_geometric_c10_rf0_j4_20260909_r2`.
The controller verifies source equality (or the explicit FP8 storage patch's
before/after hashes) with its frozen source, identical
model configuration/checkpoint selection and matching prompt identities before
launch. A single clean H200 is checked before preparation and immediately
before worker launch, then monitored throughout GPU execution.

Before loading the model, the existing C10 native gate checks seed selection,
full support and ties. The additional FP8 gate observes the actual producer
and consumer buffers, verifies FP8 K/V and independent FP32 scales, and checks
complete decoded compressed attention plus BF16 singleton against an independent
reference on three queries. Synthetic tensor error is not video PSNR.

After generation succeeds, `analysis/evaluate_fp8_video_psnr.py` uses CPU-only
decoding to compare dense/BF16, dense/FP8 and BF16/FP8. It validates prompt/seed
identity and exactly 81 equal-sized RGB frames per pair, hashes the videos,
and writes per-video and aggregate results incrementally. PSNR uses RGB uint8
data_range=255 and the mean of frame PSNRs, matching the existing evaluator.

Artifacts: `C:/GMA/experiments/fisher_c10_fp8_kv_4step_20260909` and the same
experiment name beneath `/local2/denghaoran/GridMergeAttention/experiments`.
Video run: `wan14b_720p_4step_fisher_geometric_c10_rf0_fp8kv_20260909_r2`.

Completed at 2026-09-10 03:12:16 UTC: all 15 videos and all 81 frames per video.
CPU contracts pass (51 tests); native FP8 validation and completion/timing audits
pass. GPU monitoring found no foreign processes. All 2,400 complete calls and
24,000 four-head merge/route calls used the requested operator.

| Paired quality measure | Mean PSNR (dB) |
| --- | ---: |
| BF16 C10 vs dense | 12.558194 |
| FP8 C10 vs dense | 12.514763 |
| FP8 minus BF16, both vs dense | -0.043431 |
| FP8 vs BF16 directly | 17.475492 |

The mean PSNR loss against dense is small. Twelve videos decrease and three
increase; per-video differences range from -0.256266 to +0.254166 dB. This does
not establish visual equivalence: direct FP8-vs-BF16 PSNR is 17.48 dB. No VBench,
SSIM or LPIPS was run for this PSNR screen.

Across all 15 videos, recorded mean core attention is 584.531903 ms/call, full
attention is 639.178736 ms/call, and inference is 125.638794 seconds/video.
These include first-video compilation and host audit overhead. There is no
matched SDPA/Flash timing baseline and no speedup claim. This sixteen-slot
adapter is not comparable to the retained native ten-slot 294.13 ms kernel.
Results are in `psnr.json`, `run_r2/runtime.json` and `run_r2/completion.json`;
all 15 downloaded FP8 video hashes match the evaluator's recorded hashes.
The exploratory workbook records this result in row 1119.

The first launch stopped before model loading: the historical scaled-FP8 path
rejected separate K/V accumulation. That path previously scaled only the
interleaved K/V accumulation output. The follow-up patch adds independent scale
calculation and FP8 conversion immediately before each separate K/V store;
all extra arithmetic is guarded by `SCALED_FP8_KV`. BF16 matrix accumulation,
residual correction, ownership and bias stay unchanged. Two corresponding
unsupported-combination guards are lifted. The controller records this exact
storage patch instead of claiming those two source files are byte-identical.
The retry uses a fresh video directory ending in `_r2`, retaining the first
failure's frozen source/logs. It must pass native FP8 validation before any
model generation; that validation and the subsequent video PSNR screen passed.
