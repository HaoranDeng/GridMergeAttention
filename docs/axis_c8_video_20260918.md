# W80-query / H45-key exact-C8 video experiment

The directed-axis C8 implementation retains the current-line anchor-Fisher
operator used by the W/H C4 experiment, extending capacity to exactly eight
merged representatives. It uses a matched anchor plus seven Gram-Schmidt
pivots, C32/g45 candidate geometry, and Uniform16 key probes. Every key group
participates in full compressed attention. There are no singleton, residual,
raw-key, or dense-self-attention exceptions.

`attention.axis_current_c8_cuda.AxisCurrentC8` defaults to `implementation='r2'`:
CUDA MMA selection, WGMMA merging, packed-owner comparisons derived from C8 r2,
and capacity-generic Triton routing. `implementation='slow_reference'` preserves
full FP32 winner ordering. The separate mathematical reference uses FP64 Fisher
selection and FP32 logits/reductions, with BF16 merge representation boundaries.
The existing packaged C8 and all C4 operators are unchanged.

## Validation

- CPU literal checks passed for eight distinct seeds, exact capacity, zero-Q
  seven-empty-slot behavior, dense uniform identity, and constant V.
- Synthetic native-shape tests include the H45 partial tile and two outer frames.
- Six real QKV cases cover canonical prompts 0-2, layers 0 and 33, varied heads.
  Relative L2 against the independent reference ranges from 0.1199% to 0.2066%.
  Maximum r2 versus slow_reference relative L2 is 0.03296%.
- Full H40 outputs for heads 0/10/20/30/39 match separate single-head calls in
  the checked real-QKV capture. Finite-output and empty-slot checks passed.
- The implementation diagnostic thresholds are recorded in the result JSON;
  they are not user-specified universal tolerances or video-quality guarantees.

## Matched speed gate

Measured on an exclusively idle scai6 H200 GPU4, UUID
`GPU-923ad71f-2808-6bf7-701e-dc79b7aaa38f`, with no foreign GPU processes during
the guarded run. Inputs are full BF16 `[1,40,75600,128]` captured model QKV.
Seven alternating CUDA-event measurements per method and capture cover four
prompt/layer/timestep cases. C8 timing includes input/output layouts, selection,
merging, and full routing. Both methods receive the same tensors.

| Capture | C8 median ms | Dense Flash SDPA median ms | Speedup |
|---|---:|---:|---:|
| prompt000, call000, layer0 | 318.4869 | 328.6688 | 1.03197x |
| prompt000, call033, layer33 | 318.1515 | 323.9670 | 1.01828x |
| prompt001, call073, layer33 | 317.9200 | 322.4009 | 1.01409x |
| prompt002, call120, layer0 | 319.7760 | 322.6921 | 1.00912x |

Across all 28 samples per method, **mean_attention_ms = 318.905657**,
dense mean = **325.516781 ms**, **speedup_vs_sdpa = 1.020731x**. This is a small
kernel advantage; end-to-end video speed and quality remain separate results.
Tuning stopped as soon as the user's faster-than-dense condition was satisfied.

Selected settings: head chunk 4; merge tile count 21; selector warps 4;
route tokens 64, warps 4, stages 2; fast math off. Eight slots per H45 line
give 13,440 compressed K/V tokens per query, a compressed-token fraction 8/45.
This fraction does not classify GMA as sparse attention.

## Video run

Twenty canonical MovieGenVideoBench prompts (indices 0-19), seed 0, Wan2.1 T2V
14B, 1280x720, 81 frames, 50 UniPC steps, flow shift 5, guidance 5. All 40 layers
use this operator on all 4,000 self-attention calls per video. The driver checks
prompt hashes and reuses existing matched dense outputs without regeneration.
PSNR/SSIM/LPIPS are evaluated after video 5 and again after video 20. VBench is
unmeasured. No model is used to inspect videos or extracted frames.

Launch requires matching source hashes, numerical validation, a speed receipt
showing each real-QKV case faster than dense, and a fresh clean-H200 check.
The guard stops and invalidates timing if a foreign process appears.

Artifacts: `/local2/denghaoran/GridMergeAttention/experiments/axis_video_c8_20260918`.
Frozen source: `/local2/denghaoran/GridMergeAttention/axis_wh_c8_source_20260918`.
Video directory: `qw_kh_c8_current_r2_v1_20`. Local copies belong under
`C:/GMA/experiments/axis_video_c8_20260918`; exploratory results use `C:/GMA/tmp.xlsx`.
