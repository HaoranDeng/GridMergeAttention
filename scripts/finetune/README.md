# Finetuning scripts

These scripts are additive alternatives to the existing `amd_scripts/` jobs. They
run from the repository root and expose common settings as environment variables.

Log in to Weights & Biases once with `wandb login`, or set `WANDB_API_KEY`. Then run:

```bash
CONDA_ENV=<environment> WANDB_PROJECT=training-free-videogen \
  sbatch scripts/finetune/wan_syn_dense_448p.sh
```

For a local/offline W&B run, add `WANDB_MODE=offline`. Override settings such as
`MAX_STEPS`, `GRADIENT_ACCUMULATION_STEPS`, `DATA_DIR`, and `OUTPUT_DIR` through
environment variables.
