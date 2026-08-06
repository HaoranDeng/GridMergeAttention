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
- `hard_route_group_shared`: experimental C-seed hard-route variant. It
  compresses once per query-outer group and shares the compressed KV inside that
  group. `query_outer_chunk` controls the group size.
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
For `hard_route_shared`, the deterministic seed-row table is cached per device
and token-grid shape, so repeated layers and denoise steps reuse the same seed
indices without recomputing them.

`hard_route_backend` switches the hard-route KV-compression implementation:

- `sdpa`: default PyTorch reference path for hard assignment, KV compression,
  and final SDPA routing.
- `triton`: fused Triton hard-assignment/KV-compression kernel followed by the
  same SDPA routing path. The original PyTorch path remains available by
  switching this value back to `sdpa`.
- `triton_c4`: C=4-only Triton KV-compression kernel that reuses the seed-key
  dot products between hard assignment and merge.
- `triton_c4_j16`: same C=4-specialized kernel with a wider query-column tile.
- `triton_c4_direct_j16`: shared-compression C=4 kernel that gathers seed
  queries inside Triton instead of materializing the seed-query tensor. This is
  faster than materializing the seed-query tensor.
- `triton_c4_direct_route_j16`: same direct shared C=4 kernel, but writes
  compressed K/V and routing bias directly in the layout consumed by routing
  SDPA.
- `triton_c4_direct_route_j32_w2`: route-layout direct shared C=4 kernel with
  `block_j=32` and `num_warps=2`. This is the fastest current
  `hard_route_shared` merge backend in the default 480p C=4 benchmark.
- `triton_c4_direct_route_j64`: same route-layout direct shared C=4 kernel
  with a wider query-column tile; kept as a stable comparison point.

`hard_route_routing_backend` separately switches the second compressed-attention
step:

- `sdpa`: default PyTorch routing over compressed KV tokens.
- `triton`: experimental fused Triton routing kernel for `q @ K_merged - bias`.
- `triton_m128_n32`: faster fused Triton routing kernel using `block_m=128`
  and `block_n=32`. This is the current fastest routing backend for the
  default C=4 shared setup.
- `triton_m256_n32`: fused Triton routing kernel using `block_m=256` and
  `block_n=32`; kept as a stable comparison point.

Useful configs:

- `configs/token_merging_wan_fewstep.yaml`: one-token MonarchRT-style baseline.
- `configs/token_merging_wan_hard_route_c2_fewstep.yaml`: hard-route C=2.
- `configs/token_merging_wan_hard_route_c2_triton_fewstep.yaml`: hard-route C=2 with fused Triton KV compression.
- `configs/token_merging_wan_hard_route_c4_fewstep.yaml`: hard-route C=4.
- `configs/token_merging_wan_hard_route_c4_triton_fewstep.yaml`: hard-route C=4 with fused Triton KV compression.
- `configs/token_merging_wan_hard_route_c4_triton_chunk7_fewstep.yaml`: hard-route C=4 with fused Triton KV compression and safe query-outer chunking.
- `configs/token_merging_wan_hard_route_c4_group_shared_chunk7_triton_fewstep.yaml`: experimental group-shared C=4, sharing one compression across each group of seven query outers.
- `configs/token_merging_wan_hard_route_c4_group_shared_chunk7_triton_c4_fewstep.yaml`: group-shared C=4 with the C=4-specialized Triton kernel.
- `configs/token_merging_wan_hard_route_c4_triton_chunk7_route_triton_fewstep.yaml`: hard-route C=4 with fused Triton KV compression, safe query-outer chunking, and fused Triton routing.
- `configs/token_merging_wan_hard_route_c4_fastseed_triton_chunk7_route_triton_fewstep.yaml`: same as above, with lightweight deterministic seed selection.
- `configs/token_merging_wan_hard_route_c4_shared_triton_fewstep.yaml`: experimental shared-compression hard-route C=4.
- `configs/token_merging_wan_hard_route_c4_shared_triton_c4_fewstep.yaml`: shared-compression C=4 with the C=4-specialized Triton kernel.
- `configs/token_merging_wan_hard_route_c4_shared_triton_c4_j16_fewstep.yaml`: stable shared-compression C=4 config, using the direct route-layout Triton merge kernel and `triton_m256_n32` routing.
- `configs/token_merging_wan_hard_route_c4_shared_triton_c4_direct_j16_fewstep.yaml`: direct shared-compression C=4 backend before route-layout output.
- `configs/token_merging_wan_hard_route_c4_shared_triton_c4_direct_route_j16_fewstep.yaml`: explicit direct route-layout shared-compression C=4 backend.
- `configs/token_merging_wan_hard_route_c4_shared_triton_c4_direct_route_j16_routing_triton_fewstep.yaml`: direct route-layout shared-compression C=4 backend with fast Triton routing.
- `configs/token_merging_wan_hard_route_c4_shared_triton_c4_direct_route_j16_routing_triton_m256_fewstep.yaml`: direct route-layout shared-compression C=4 backend with `triton_m256_n32` routing.
- `configs/token_merging_wan_hard_route_c4_shared_triton_c4_direct_route_j32_w2_routing_triton_m128_fewstep.yaml`: current fastest shared-compression C=4 config, using the `j32_w2` route-layout merge kernel and `triton_m128_n32` routing.
- `configs/token_merging_wan_hard_route_c4_shared_triton_c4_direct_route_j64_routing_triton_m256_fewstep.yaml`: wider `j64` route-layout merge tile with `triton_m256_n32` routing.
- `configs/token_merging_wan_hard_route_c4_shared_triton_c4_direct_route_j64_routing_triton_m128_fewstep.yaml`: wider `j64` route-layout merge tile with `triton_m128_n32` routing.
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
