# H200 Wan2.1-14B 720p Four-Step Comparison

This entry point reproduces the pure all-layer comparison used on 2026-08-26:

- Dense SDPA forced to cuDNN attention
- Dense SDPA forced to FlashAttention
- Hard C4 with fixed local seed rows
- Hellinger-Span C4 top-1
- SVG
- SVG-2 at `top_p=0.9`
- SVG-EAR at `top_p=0.9`

All methods use the same StepDistill checkpoint, 20 prompts, seed 0,
1280x720 output, 81 frames, and four denoising steps. SVG-2 and SVG-EAR
record realized weighted attention-map density for every attention call.
`top_p` is not reported as density.

## Launch From Main

Use two exclusively idle H200 GPUs:

```bash
git switch main
git pull --ff-only origin main

DENSE_GPU=3 METHOD_GPU=4 \
bash scripts/generate/run_wan14b_720p_4step_h200_comparison.sh
```

To run only selected methods:

```bash
GPU=4 METHODS="hard_c4 hellinger_top1" \
bash analysis/run_wan14b_720p_4step_method_queue.sh

GPU=3 BACKENDS="flash" \
bash analysis/run_dense_sdpa_backends_720p_4step.sh
```

The scripts reject a busy GPU and monitor for foreign compute processes during
each run. Override `WAN_MODEL_ROOT`, `PROMPTS`, `ENV_ROOT`, `SVG_ROOT`,
`RUN_ID`, or `OUTPUT_ROOT` when the scai7 installation paths differ.

## Outputs

Each method directory contains `manifest.json`, `runtime.json`,
`generate.log`, `purity.log`, and 20 videos. SVG-2 and SVG-EAR additionally
contain `density.jsonl` with 3,200 records and `results/density.json` with
call-weighted aggregate and per-video density/sparsity.

The dense summary command is:

```bash
python analysis/summarize_dense_sdpa_backends.py --root <dense-output-root>
```
