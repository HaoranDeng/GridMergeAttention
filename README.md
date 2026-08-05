# GridMergeAttention

Minimal Wan2.1 text-to-video generation with grid-based token-merging attention.

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
  layout: hard_route
  q_init: ith
  num_merge_tokens: 1
  seed_mode: random_with_ith
  seed: 0
  hard_route_backend: sdpa
```

Supported local layouts:

- `hard_route`: default GridMerge layout. A block fixes one query column and one
  key row, so the default 480p token grid gives a `30q x 52k` block.
- `hard_route_shared`: experimental C-seed hard-route variant. It compresses
  each key outer once using the matching query outer as seeds, then shares those
  compressed KV tokens across all query outers. This is faster but is a slight
  method change from `hard_route`.
- `select_route`: same block shape, but each query chooses among several
  candidate compressed KV tokens.
- `hard_route_transpose`: transposed layout. A block fixes one query row and one
  key column, giving a `52q x 30k` block.
- `select_route_transpose`: transposed version of the select-route method.

`num_merge_tokens` selects the method:

- `num_merge_tokens: 1` is the original one-token MonarchRT-style compression.
  Each block compresses its key group into one KV token.
- `num_merge_tokens: C > 1` with `layout: hard_route` or
  `hard_route_transpose` uses the hard-route method. The block samples `C` seed
  queries, assigns every key to its best seed by dot product, and compresses one
  KV token per seed.
- `num_merge_tokens: C > 1` with `layout: select_route` or
  `select_route_transpose` uses the select-route method. Each seed independently
  compresses the full key group, then each query selects the best compressed KV
  candidate using the dot-product-plus-entropy objective.

For the current experiments, `seed_mode: random_with_ith` keeps the original
index-matched query as one seed and samples the remaining seeds
deterministically from the same block.
`seed_mode: fast_random_with_ith` keeps that index-matched seed but chooses the
remaining seeds with a lightweight deterministic hash instead of scoring every
candidate row.

`hard_route_backend` switches the hard-route KV-compression implementation:

- `sdpa`: default PyTorch reference path for hard assignment, KV compression,
  and final SDPA routing.
- `triton`: fused Triton hard-assignment/KV-compression kernel followed by the
  same SDPA routing path. The original PyTorch path remains available by
  switching this value back to `sdpa`.

`hard_route_routing_backend` separately switches the second compressed-attention
step:

- `sdpa`: default PyTorch routing over compressed KV tokens.
- `triton`: experimental fused Triton routing kernel for `q @ K_merged - bias`.

Useful configs:

- `configs/token_merging_wan_fewstep.yaml`: one-token MonarchRT-style baseline.
- `configs/token_merging_wan_hard_route_c2_fewstep.yaml`: hard-route C=2.
- `configs/token_merging_wan_hard_route_c2_triton_fewstep.yaml`: hard-route C=2 with fused Triton KV compression.
- `configs/token_merging_wan_hard_route_c4_fewstep.yaml`: hard-route C=4.
- `configs/token_merging_wan_hard_route_c4_triton_fewstep.yaml`: hard-route C=4 with fused Triton KV compression.
- `configs/token_merging_wan_hard_route_c4_triton_chunk7_fewstep.yaml`: hard-route C=4 with fused Triton KV compression and safe query-outer chunking.
- `configs/token_merging_wan_hard_route_c4_triton_chunk7_route_triton_fewstep.yaml`: hard-route C=4 with fused Triton KV compression, safe query-outer chunking, and fused Triton routing.
- `configs/token_merging_wan_hard_route_c4_fastseed_triton_chunk7_route_triton_fewstep.yaml`: same as above, with lightweight deterministic seed selection.
- `configs/token_merging_wan_hard_route_c4_shared_triton_fewstep.yaml`: experimental shared-compression hard-route C=4.
- `configs/token_merging_wan_hard_route_c4_shared_triton_route_triton_fewstep.yaml`: experimental shared-compression hard-route C=4 with fused Triton routing.
- `configs/token_merging_wan_hard_route_c4_shared_fastseed_triton_route_triton_fewstep.yaml`: shared-compression hard-route C=4 with lightweight deterministic seed selection.
- `configs/token_merging_wan_hard_route_c6_fewstep.yaml`: hard-route C=6.
- `configs/token_merging_wan_hard_route_c6_triton_fewstep.yaml`: hard-route C=6 with fused Triton KV compression.
- `configs/token_merging_wan_hard_route_c13_fewstep.yaml`: hard-route C=13.
- `configs/token_merging_wan_select_route_c2_fewstep.yaml`: select-route C=2.
- `configs/token_merging_wan_select_route_c4_fewstep.yaml`: select-route C=4.
- `configs/token_merging_wan_select_route_c6_fewstep.yaml`: select-route C=6.
- `configs/token_merging_wan_gridmerge_one_transpose_fewstep.yaml`: one-token GridMerge transpose.
- `configs/token_merging_wan_hard_route_c2_transpose_fewstep.yaml`: hard-route C=2 transpose.
- `configs/token_merging_wan_hard_route_c4_transpose_fewstep.yaml`: hard-route C=4 transpose.
- `configs/token_merging_wan_hard_route_c6_transpose_fewstep.yaml`: hard-route C=6 transpose.
- `configs/token_merging_wan_hard_route_c6_transpose_triton_fewstep.yaml`: hard-route C=6 transpose with fused Triton KV compression.
- `configs/token_merging_wan_select_route_c2_transpose_fewstep.yaml`: select-route C=2 transpose.
- `configs/token_merging_wan_select_route_c4_transpose_fewstep.yaml`: select-route C=4 transpose.
- `configs/token_merging_wan_select_route_c6_transpose_fewstep.yaml`: select-route C=6 transpose.

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
