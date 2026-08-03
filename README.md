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

## Token-Merging Variants

The dense baseline is selected by `configs/dense_wan_fewstep.yaml`, which sets
`token_merging_args.enable: false`.

Token merging is controlled by the `token_merging_args` block in each config:

```yaml
token_merging_args:
  enable: true
  f_tied: 1
  h_reduce: 1
  w_reduce: 1
  layout: query_column_key_row
  q_init: ith
  num_merge_tokens: 1
  seed_mode: random_with_ith
  seed: 0
```

Supported local layouts:

- `query_column_key_row`: default layout. A block fixes one query column and one
  key row, so the default 480p token grid gives a `30q x 52k` block.
- `query_column_key_row_select`: same block shape, but each query chooses among
  several candidate compressed KV tokens.
- `query_row_key_column`: transposed layout. A block fixes one query row and one
  key column, giving a `52q x 30k` block.
- `query_row_key_column_select`: transposed version of the select method.

`num_merge_tokens` selects the method:

- `num_merge_tokens: 1` is the original one-token MonarchRT-style compression.
  Each block compresses its key group into one KV token.
- `num_merge_tokens: C > 1` with `layout: query_column_key_row` or
  `query_row_key_column` uses the hard multi-seed method. The block samples `C`
  seed queries, assigns every key to its best seed by dot product, and compresses
  one KV token per seed.
- `num_merge_tokens: C > 1` with a `_select` layout uses the select method. Each
  seed independently compresses the full key group, then each query selects the
  best compressed KV candidate using the dot-product-plus-entropy objective.

For the current experiments, `seed_mode: random_with_ith` keeps the original
index-matched query as one seed and samples the remaining seeds
deterministically from the same block.

Useful configs:

- `configs/token_merging_wan_fewstep.yaml`: one-token MonarchRT-style baseline.
- `configs/token_merging_wan_two_seed_fewstep.yaml`: hard C=2.
- `configs/token_merging_wan_four_seed_fewstep.yaml`: hard C=4.
- `configs/token_merging_wan_six_seed_fewstep.yaml`: hard C=6.
- `configs/token_merging_wan_thirteen_seed_fewstep.yaml`: hard C=13.
- `configs/token_merging_wan_select_c2_fewstep.yaml`: select C=2.
- `configs/token_merging_wan_select_c4_fewstep.yaml`: select C=4.
- `configs/token_merging_wan_select_c6_fewstep.yaml`: select C=6.
- `configs/token_merging_wan_transpose_c2_fewstep.yaml`: transpose hard C=2.
- `configs/token_merging_wan_transpose_c4_fewstep.yaml`: transpose hard C=4.
- `configs/token_merging_wan_transpose_c6_fewstep.yaml`: transpose hard C=6.
- `configs/token_merging_wan_transpose_select_c2_fewstep.yaml`: transpose select C=2.
- `configs/token_merging_wan_transpose_select_c4_fewstep.yaml`: transpose select C=4.
- `configs/token_merging_wan_transpose_select_c6_fewstep.yaml`: transpose select C=6.

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
