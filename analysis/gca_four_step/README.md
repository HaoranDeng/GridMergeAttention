# Q80K90Even four-step T2V rerun

User-requested 2026-09-25 campaign: Q80K90 C4/C6/C8/C10, 20 canonical
MovieGen prompts per capacity. This uses the frozen native even-capacity
kernel from `~/workspace/gma_q80k90_even_v2`, now included under
`kernels/experimental/q80k90_even`. The source hash of its Python interface
is `31e042f401e7c5aa64175a94b9cff87938114ab068daa944ce487efda9b07513`.

## Protocol

- Wan2.1-T2V-14B-StepDistill-CfgDistill, raw checkpoint weights, BF16.
- 1280x720, 81 frames, 4 distilled steps; actual warped timesteps are
  1000, 937.5, 833.3333129882812, 625. There is one guidance branch.
- All 40 self-attention layers use the chosen operator, 160 calls/video.
  Exact capacities, RF0 selection, entropy bias, no singleton/residual routes.
- Generation resets the diffusion seed to zero for each prompt. This is
  explicit rather than relying on the older generator's base-seed-plus-index
  behavior. Both Dense and GCA use the same input and checkpoint protocol.
- 20 new matched Dense videos are split across the two workers. These are
  engineering/quality references, not additional external comparison methods.
- The launch gate uses the existing numerical-screen protocol: prompts 0--2,
  seed equal to prompt index, all four steps, layers 0/5/10/15/20/25/30/35,
  all 40 heads. Each capacity has 96 real-QKV cases against the previous
  validated Q80K90 operator and same-GPU Flash SDPA. The 0.01 relative-L2
  bound is an engineering gate, not a user-specified universal tolerance.
  Existing observed maxima were approximately 0.0018. All finite checks
  and an average speed advantage over same-GPU Dense must pass.
- Numerical-gate measurements are diagnostics; per-video CUDA-event timings
  are recorded independently. Model loading, gate, export and quality work
  are excluded from inference time. The first video of each method is
  identified as warmup, and both all-video and steady-state summaries remain.
- PSNR/SSIM/LPIPS use the unchanged hash-frozen CPU evaluator, all 81 decoded
  frames at original resolution, LPIPS AlexNet. Prompt, seed and video hashes
  are checked before pairing. No model-based video/frame inspection is used.
- VBench is not executed by this runner and must remain pending until run.
- This is a four-step campaign; do not relabel these timings or its 160-call
  arithmetic as full 50-step results. External SVG-EAR and original Q45K80C8
  comparisons require matched protocols and remain pending in this campaign.

## Dispatch and outputs

Two workers use separate exclusive H200s. Worker 0 runs C4 then C10; worker 1
runs C6 then C8. A worker starts only after three clean GPU snapshots and
continues monitoring compute-process ownership. Foreign GPU use terminates
only this worker's process group, preserving the foreign job and marking the
campaign invalid. Launches never wait for a busy GPU to become idle.

```
python analysis/gca_four_step/run.py supervise --root /path/to/fresh/campaign --shard 0 --gpu GPU-UUID
CUDA_VISIBLE_DEVICES='' python analysis/gca_four_step/run.py quality --root /path/to/fresh/campaign --shard 0
```

Use the corresponding shard 1 on a second GPU. Set the Python environment to
the repository's validated `monarch_rt` environment with CUDA/CUTLASS support.
The supervisor sets private extension and Triton cache directories. Never
reuse a completed/failed destination without explicit recovery and provenance.

The 2026-09-26 campaign lives at
`/local2/denghaoran/GridMergeAttention/experiments/gca_q80k90_4step_20260926`.
Local artifacts are under `C:/GMA/experiments/gca_q80k90_4step_20260926`.
Each worker writes `gate.json`, `protocol.json`, `stage.json`, GPU monitoring,
and completion/failure markers. Each capacity writes per-video runtime and
quality JSON alongside its videos. Dense has a runtime file per worker.
No paper files or historical results are modified.
