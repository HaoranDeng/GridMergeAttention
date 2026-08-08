# GridMergeAttention

Minimal Wan2.1 text-to-video generation with grid-based token-merging
attention. The repository keeps the inference code, configs, and attention
experiments needed to compare dense attention, MonarchRT-style one-token
GridMerge, and multi-token GridMerge routing methods.

Large Wan model weights and checkpoints are not stored here.

## Layout

- `attention/`: dense attention plus GridMerge attention variants.
- `wan/`: minimal Wan2.1 model, text encoder, tokenizer, and VAE runtime.
- `pipeline/`: few-step text-to-video pipeline.
- `configs/`: dense, MonarchRT/GridMerge, hard-route, select-route, and
  transpose experiment configs.
- `generate.py`: video generation entry point.
- `docs/negative_results.md`: failed directions and constraints that should be
  checked before starting new attention experiments.

## Attention Variants

Dense baseline:

- `configs/dense_wan_fewstep.yaml`

One-token GridMerge / MonarchRT-style compression:

- `configs/token_merging_wan_fewstep.yaml`
- `configs/token_merging_wan_h_reduce2_fewstep.yaml`
- `configs/token_merging_wan_h_reduce4_fewstep.yaml`
- `configs/token_merging_wan_w_reduce2_fewstep.yaml`
- `configs/token_merging_wan_w_reduce4_fewstep.yaml`
- `configs/token_merging_wan_w_reduce13_fewstep.yaml`

Multi-token hard routing:

- `configs/token_merging_wan_hard_route_c2_fewstep.yaml`
- `configs/token_merging_wan_hard_route_c4_fewstep.yaml`
- `configs/token_merging_wan_hard_route_c6_fewstep.yaml`
- `configs/token_merging_wan_hard_route_c13_fewstep.yaml`
- `configs/token_merging_wan_hard_route_c4_fastseed_full_core_fewstep.yaml`

Select routing:

- `configs/token_merging_wan_select_route_c2_fewstep.yaml`
- `configs/token_merging_wan_select_route_c4_fewstep.yaml`
- `configs/token_merging_wan_select_route_c6_fewstep.yaml`

Transpose layouts:

- `configs/token_merging_wan_gridmerge_one_transpose_fewstep.yaml`
- `configs/token_merging_wan_hard_route_c2_transpose_fewstep.yaml`
- `configs/token_merging_wan_hard_route_c4_transpose_fewstep.yaml`
- `configs/token_merging_wan_hard_route_c6_transpose_fewstep.yaml`
- `configs/token_merging_wan_select_route_c2_transpose_fewstep.yaml`
- `configs/token_merging_wan_select_route_c4_transpose_fewstep.yaml`
- `configs/token_merging_wan_select_route_c6_transpose_fewstep.yaml`

For the current recommended hard C4 path, use
`configs/token_merging_wan_hard_route_c4_fastseed_full_core_fewstep.yaml`.
It keeps non-shared grid-local KV compression and runs full compressed
attention after token merging.

## Token-Merging Args

Token merging is controlled by `token_merging_args`:

```yaml
token_merging_args:
  enable: true
  f_tied: 1
  h_reduce: 1
  w_reduce: 1
  layout: hard_route
  q_init: ith
  query_outer_chunk: 10
  num_merge_tokens: 4
  seed_mode: local_with_ith
  seed: 0
  hard_route_backend: triton_c4_direct_route_j32_w2_exp2merge_forcekeys_fp16kvbias_s1_fastdot_bestlogit_tokenmajor_inline_local_seed
  hard_route_routing_backend: triton_direct_m32_n16_w1_s1_exp2_tokenmajor
```

Supported layouts:

- `hard_route`: default GridMerge layout. For default 480p/81-frame Wan tokens,
  each block is `30 queries x 52 keys`.
- `select_route`: each seed independently compresses a full key group, then
  each query selects the best compressed KV candidate.
- `hard_route_transpose`: transposed layout, with `52 queries x 30 keys` per
  block.
- `select_route_transpose`: transpose version of select routing.

`num_merge_tokens: 1` runs one-token GridMerge / MonarchRT-style compression.
`num_merge_tokens: C > 1` runs C-seed hard-route or select-route compression.

## Guardrails

The main quality-preserving path should follow these rules:

- Keep grid-local compression. Do not use global/share-all compression.
- After token merging, keep full compressed attention. Do not replace the
  second step with sparse, sliding-window, or local-only routing.
- Keep the routing bias. No-bias variants produced bad videos.
- Do not use quantization for the current memory-bound optimization direction.

Rejected directions and benchmark evidence are recorded in
`docs/negative_results.md`.

## Run

Set `WAN_MODEL_ROOT` to a directory containing `Wan2.1-T2V-1.3B/`, then run:

```bash
python generate.py \
  --config configs/token_merging_wan_hard_route_c4_fastseed_full_core_fewstep.yaml \
  --checkpoint /path/to/self_forcing_dmd.pt \
  --prompt_path prompts/MovieGenVideoBench_extended.txt \
  --output_dir videos/gridmerge_hard_c4 \
  --num_videos 20 \
  --use_ema
```
