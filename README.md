# training-free-videogen

Minimal Wan2.1 text-to-video generation with token-merging attention.

This repository keeps only the inference path needed to run Wan2.1 with the
token-merging attention module. Large model weights and checkpoints are not
stored here.

## Layout

- `attention/`: dense SDPA attention plus token-merging attention.
- `wan/`: minimal Wan2.1 model, T5 text encoder, tokenizer, and VAE runtime.
- `pipeline/`: few-step text-to-video pipeline.
- `generate.py`: single-video-generation entry point.

## Run

Set `WAN_MODEL_ROOT` to a directory containing `Wan2.1-T2V-1.3B/`, then run:

```bash
python generate.py \
  --config configs/token_merging_wan_fewstep.yaml \
  --checkpoint /path/to/self_forcing_dmd.pt \
  --prompt_path prompts/MovieGenVideoBench_extended.txt \
  --output_dir videos/token_merging_wan \
  --num_videos 20 \
  --use_ema
```

On `amdhpc`, the helper script uses the configured conda environment and expects
repo-local `wan_models/` and `checkpoints/` links or directories:

```bash
CONDA_ENV=<your-env> sbatch amd_scripts/generate_20_token_merging_wan.sh
```

Override token-merging size without editing YAML:

```bash
CONDA_ENV=<your-env> W_REDUCE=4 sbatch amd_scripts/generate_20_token_merging_wan.sh
```
