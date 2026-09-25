# Wan2.2 SVG-EAR and standard GMA-C4 comparison

The user requested scai6 and at most two GPUs, and explicitly selected identical
dense warm-up for both methods: the first 10 of 50 steps and the first layer.
This overrides the ordinary all-layer GMA default for this comparison only.

- Model: Wan-AI/Wan2.2-T2V-A14B-Diffusers, revision
  `5be7df9619b54f4e2667b2755bc6a756675b5cd7`.
- Output: 1280 by 720, 81 frames, 50 UniPC steps, checkpoint flow shift 3,
  guidance 4 for the high-noise expert and 3 for the low-noise expert.
- Expert boundary: 0.875. Both experts receive the same attention operator and
  share the global denoising timestep. Warm-up does not restart at the switch.
- Prompts: first 20 canonical MovieGenVideoBench_extended prompts, seed 0,
  one video per prompt, Tokyo-woman first, verified prompt manifests.
- SVG-EAR: top-p 0.85, 300 query clusters, 1,000 key clusters, minimum key-cluster
  ratio 0.05, initial/continuing k-means iterations 50/2. Published upstream
  operator pinned to `f89aedaf169ac2ae5b186bda674e53c3dc08c476`.
- GMA-C4: existing standard Hard C4, four fixed seed queries, entropy bias,
  full compressed attention and head chunk 8. No learned parameters.
- The two methods use 880 dense and 3,120 replacement calls per video, for
  4,000 total self-attention calls including both CFG passes.

All methods use the same native Wan2.2 transformer forward, separate Triton Q/K
normalization and SVG rotary kernel. The isolated Diffusers 0.35.2 installation
supports Wan2.2 without changing the shared 0.34 environment. A layout adapter
converts the newer real rotary embeddings to SVG's equivalent rotary layout
and supplies the shared pipeline timestep to each processor. Native model
forward is retained because the older SVG whole-model patch failed the tiny
BF16 numerical check. A native-versus-adapted test exercises both experts.
The initial run completed GMA denoising but failed in FP32 VAE decoding because
global flash-only SDPA has no compatible FP32 kernel. The retry uses math SDPA
only inside VAE decode, restores flash-only SDPA for transformer calls, and
requires a full 81-frame 720p checkpoint-VAE decode preflight. The first failed
run is preserved; the corrected run uses the `_r1` artifact directory suffix.

GPU 4 generates Dense SDPA while GPU 0 generates GMA-C4 followed by SVG-EAR.
GPU 5 became occupied during setup, so the final launch rechecks GPUs 4 and 0.
Paired quality and VBench evaluation then use the two GPUs. Each GPU stage
requires an immediately clean H200, polls for foreign
processes, and invalidates contaminated runs. A busy device causes a stop.
The account uses at most two GPUs on scai6 throughout.

Paired PSNR, SSIM and AlexNet LPIPS use the fresh Dense SDPA reference and all
1,620 frames. Six custom-input VBench dimensions are reported separately,
without claiming the official aggregate VBench score. Attention core/module
times use CUDA events. End-to-end inference excludes model loading and video
export. Steady results exclude the first video. Method-to-Dense runtime ratios
are marked approximate across two physical H200s; no identical-QKV replay is
claimed. SVG-EAR reports both sparse-call and all-call realized density.

This uses the paper's SVG-EAR hyperparameters and warm-up, with the project's
canonical prompts; it is not a literal reproduction of the paper's prompt set.

Remote artifacts:
`/local2/denghaoran/GridMergeAttention/experiments/wan22_720p_50step_ear_c4_20260908`.
Corrected run: the same path with `_r1` appended.
Local copies:
`C:\GMA\experiments\wan22_720p_50step_ear_c4_20260908`.
Corrected run: the same path with `_r1` appended.
Dense/SVG-EAR baseline results go to `C:\GMA\experiments.xlsx`.
The explicitly requested GMA hybrid comparison goes to `C:\GMA\tmp.xlsx`.

Sources: [SVG-EAR paper](https://arxiv.org/html/2603.08982v2),
[Wan2.2 checkpoint](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers).
