# GridMergeAttention

Minimal Wan2.1 text-to-video generation with the best current hard C4
GridMergeAttention path. The repository keeps:

- dense attention for baseline comparison,
- hard C4 PyTorch reference implementation,
- hard C4 Triton implementation for speed.

Large Wan model weights and checkpoints are not stored here.

## Layout

- `attention/`: dense attention plus hard C4 GridMergeAttention.
- `wan/`: minimal Wan2.1 model, text encoder, tokenizer, and VAE runtime.
- `pipeline/`: few-step text-to-video pipeline.
- `configs/`: dense baseline and hard C4 PyTorch/Triton configs.
- `generate.py`: video generation entry point.
- `docs/negative_results.md`: failed routes and constraints that should not be
  repeated.

## Configs

Dense baseline:

- `configs/dense_wan_fewstep.yaml`

Hard C4 GridMergeAttention:

- `configs/token_merging_wan_hard_c4_pytorch_fewstep.yaml`
- `configs/token_merging_wan_hard_c4_triton_fewstep.yaml`

The PyTorch config is the readable correctness reference:

```yaml
token_merging_args:
  enable: true
  layout: hard_route
  num_merge_tokens: 4
  seed_mode: local_with_ith
  hard_route_backend: sdpa
```

The Triton config is the current best speed/quality path:

```yaml
token_merging_args:
  enable: true
  layout: hard_route
  query_outer_chunk: 10
  num_merge_tokens: 4
  seed_mode: local_with_ith
  hard_route_backend: hard_c4
```

## Guardrails

The maintained hard C4 path follows these rules:

- Keep grid-local compression.
- Use `local_with_ith` C4 seeds.
- After token merging, keep full compressed attention.
- Keep the entropy/routing bias.
- Do not use global/share-all compression, no-bias routing, sparse/windowed
  second-stage attention, or quantization for the current path.

Rejected directions and benchmark evidence are recorded in
`docs/negative_results.md`.

## Run

Set `WAN_MODEL_ROOT` to a directory containing `Wan2.1-T2V-1.3B/`, then run:

```bash
python generate.py \
  --config configs/token_merging_wan_hard_c4_triton_fewstep.yaml \
  --checkpoint /path/to/self_forcing_dmd.pt \
  --prompt_path prompts/MovieGenVideoBench_extended.txt \
  --output_dir videos/gridmerge_hard_c4 \
  --num_videos 20 \
  --use_ema
```
