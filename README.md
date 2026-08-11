# GridMergeAttention

Training-free Token Merging for text-to-video diffusion. This repository
contains three maintained inference paths:

- bidirectional Wan2.1 with dense attention, a hard-C4 PyTorch correctness
  reference, and a fused hard-C4 Triton implementation for speed;
- causal, block-wise Self-Forcing with a growing K/V cache and one-token
  GridMergeAttention;
- Diffusers HunyuanVideo with joint video/text hard-route Token Merging.

Model weights and Self-Forcing checkpoints are not stored in the repository.

## Supported paths

| Path | Generation semantics | Token-Merging support | Entry point |
| --- | --- | --- | --- |
| Wan2.1 | All latent frames are denoised bidirectionally | Hard C4 with PyTorch and fused Triton implementations | `generate.py` |
| Self-Forcing | Seven causal three-frame blocks, with current-block Q and a growing visible K/V prefix | Cache-aware hard C1 | `generate.py` through the causal config |
| HunyuanVideo | Standard Diffusers denoising with joint `[video / text]` attention | PyTorch hard C1/C2/C4/C6; video K/V are merged while text remains exact | `generate_hunyuan.py` |

`C` is the number of compressed video K/V tokens retained per local routing
block. It is not the attention head dimension or the original number of keys.

For Hunyuan, video queries attend jointly to compressed video keys and exact
text keys under one shared softmax. Text-query attention remains exact and
dense over the original video and text tokens. The separate Hunyuan text
refiner is not replaced because it has no video-token grid.

## Ordinary Wan configurations

The dense baseline is:

- `configs/dense_wan_fewstep.yaml`

The maintained hard-C4 GridMergeAttention configurations are:

- `configs/token_merging_wan_hard_c4_pytorch_fewstep.yaml`;
- `configs/token_merging_wan_hard_c4_triton_fewstep.yaml`.

The PyTorch configuration is the readable correctness reference:

```yaml
token_merging_args:
  enable: true
  layout: hard_route
  num_merge_tokens: 4
  seed_mode: local_with_ith
  hard_route_backend: sdpa
```

The Triton configuration selects the fused hard-C4 kernel:

```yaml
token_merging_args:
  enable: true
  layout: hard_route
  query_outer_chunk: 10
  num_merge_tokens: 4
  seed_mode: local_with_ith
  hard_route_backend: hard_c4
```

## Repository layout

- `attention/`: dense attention, ordinary-Wan hard C4, causal C1, and Hunyuan
  joint video/text Token Merging.
- `wan/`: the Wan2.1 model, causal model, text encoder, tokenizer, and VAE
  runtime.
- `pipeline/`: bidirectional Wan, causal Self-Forcing, and Hunyuan pipeline
  integration.
- `configs/`: the dense baseline, ordinary-Wan hard-C4 PyTorch/Triton,
  Self-Forcing, and Hunyuan generation configurations.
- `scripts/generate/`: single- and multi-GPU generation launchers.
- `generate.py`: Wan and Self-Forcing generation entry point.
- `generate_hunyuan.py`: HunyuanVideo generation and timing entry point.
- `tests/`: fast contract tests and opt-in real-model parity/E2E tests.
- [`docs/negative_results.md`](docs/negative_results.md): rejected routes,
  benchmark evidence, and constraints that should not be repeated.

## Requirements

Generation requires a CUDA-capable PyTorch installation and the model-specific
Python dependencies.

For Wan and Self-Forcing:

```bash
python -m pip install -r requirements.txt
```

For HunyuanVideo:

```bash
python -m pip install -r requirements-hunyuan.txt
```

The Hunyuan requirements intentionally do not pin or install PyTorch because
the PyTorch build must match the target CUDA toolchain. Hunyuan requires
Diffusers 0.34.0, while the Wan stack currently uses Diffusers 0.31.0.

Wan generation additionally requires the `Wan2.1-T2V-1.3B` model and the
Self-Forcing checkpoint. Set `WAN_MODEL_ROOT`, `WAN_MODEL_DIR`, `CHECKPOINT`,
and `PROMPT_PATH` as appropriate for the selected entry point. Hunyuan defaults
to the public `tencent/HunyuanVideo` model at revision `refs/pr/18`.

## Generate with bidirectional Wan

The PyTorch hard-C4 configuration is the readable correctness reference:

```bash
python generate.py \
  --config configs/token_merging_wan_hard_c4_pytorch_fewstep.yaml \
  --checkpoint "$CHECKPOINT" \
  --prompt_path "$PROMPT_PATH" \
  --num_videos 1 \
  --num_samples 1 \
  --use_ema
```

Set `WAN_MODEL_ROOT` before running the command. Use
`configs/dense_wan_fewstep.yaml` for the dense few-step baseline or
`configs/token_merging_wan_hard_c4_triton_fewstep.yaml` for the fused hard-C4
backend.

## Generate with causal Self-Forcing

The dedicated launcher selects
`configs/token_merging_self_forcing_causal.yaml`. Set `WAN_MODEL_DIR`,
`CHECKPOINT`, and `PROMPT_PATH`, then run:

```bash
NUM_VIDEOS=1 \
NUM_SAMPLES=1 \
bash scripts/generate/generate_token_merging_self_forcing.sh
```

The configured 21 latent frames are generated as seven causal three-frame
blocks. Each block runs all four denoising steps and one clean-context commit:

```text
Q frames:   3, 3, 3,  3,  3,  3,  3
K/V frames: 3, 6, 9, 12, 15, 18, 21
```

`SAMPLE_BATCH_SIZE` defaults to one because the complete 30-layer BF16 K/V
cache is large. `NUM_SAMPLES` controls the total outputs per prompt; increase
`SAMPLE_BATCH_SIZE` only after measuring memory on the target GPU.

## Generate with HunyuanVideo

The single-worker launcher uses the generation geometry and offload policy in
the selected YAML. This example runs hard C2:

```bash
CONFIG=configs/token_merging_hunyuan_hard_route_c2_480p.yaml \
NUM_VIDEOS=1 \
NUM_SAMPLES=1 \
bash scripts/generate/generate_token_merging_hunyuan.sh
```

The supplied Hunyuan configs generate 480x720, 129-frame videos in BF16 with
model CPU offload and VAE tiling. The first run may download tokenizers, text
encoders, the transformer, and the VAE. Use `LOCAL_FILES_ONLY=1` only after the
complete pipeline is cached.

The reference one-token config is
`configs/token_merging_hunyuan_480p.yaml`. The multi-token experiment configs
are:

- `configs/token_merging_hunyuan_hard_route_c2_480p.yaml`;
- `configs/token_merging_hunyuan_hard_route_c4_480p.yaml`;
- `configs/token_merging_hunyuan_hard_route_c6_480p.yaml`.

Only `hard_route` is supported. Select routing and transposed layouts are
rejected explicitly.

## Run the Hunyuan hard C2/C4/C6 sweep

Reserve the GPUs first, then provide their IDs explicitly:

```bash
GPU_IDS="0 1" \
PYTHON_BIN="$(command -v python)" \
NUM_VIDEOS=-1 \
NUM_SAMPLES=5 \
bash scripts/generate/launch_hunyuan_hard_configs_multi_gpu.sh
```

For each of C2, C4, and C6, the launcher starts one prompt shard per listed
GPU. It waits for every shard to finish before moving to the next C value,
resumes valid existing videos, records per-worker timing, aggregates a summary,
and stops the sweep if a worker fails. Every worker loads its own Hunyuan
pipeline, so account for host RAM as well as GPU memory.

The primary timing field,
`average_pipeline_generation_seconds_per_video`, is the weighted average of
newly generated videos. It includes prompt encoding, denoising, and VAE decode,
but excludes model loading and MP4 export. The summary separately reports
`effective_multi_gpu_seconds_per_video`, which measures actual parallel
throughput across the worker window. Resume runs measure only videos generated
by that invocation.

Override `PROMPT_PATH`, `MODEL_ID`, `NUM_VIDEOS`, `NUM_SAMPLES`, `SEED`, or
`FPS` through environment variables. Extra CLI arguments supplied to the
launcher are forwarded to every worker.

## Configuration summary

| Configuration | Purpose |
| --- | --- |
| `configs/dense_wan_fewstep.yaml` | Bidirectional Wan dense baseline |
| `configs/token_merging_wan_hard_c4_pytorch_fewstep.yaml` | Bidirectional Wan hard-C4 PyTorch reference |
| `configs/token_merging_wan_hard_c4_triton_fewstep.yaml` | Bidirectional Wan fused hard-C4 Triton backend |
| `configs/token_merging_self_forcing_causal.yaml` | Causal Self-Forcing with cache-aware hard C1 |
| `configs/token_merging_hunyuan_480p.yaml` | Hunyuan joint-attention hard C1 reference |
| `configs/token_merging_hunyuan_hard_route_c{2,4,6}_480p.yaml` | Hunyuan hard-route experiment family |

Ordinary Wan C2/C6 configurations were removed upstream and remain
unsupported. Hunyuan C2/C4/C6 are independent implementations and do not use
the ordinary-Wan backend.

## Validation

The implementation is covered at several boundaries:

- The Self-Forcing real-checkpoint E2E test strictly loaded the EMA checkpoint,
  observed all 1,050 expected cache-aware Token-Merging calls, decoded the
  video, and passed its shape, finiteness, and cache-index assertions.
- The dense Self-Forcing comparison against MonarchRT passed with
  `MAE=0.0066297227`, `RMSE=0.0085105132`, maximum error `0.046875`, and cosine
  similarity `0.99988222` across two causal blocks.
- The real-checkpoint Hunyuan dense-probe comparison observed all 120 attention
  calls across two denoising steps and matched untouched Diffusers exactly at
  every recorded latent boundary.
- The tiny Diffusers Hunyuan test passed exact processor parity and exercised
  hard C2/C4/C6 through real dual- and single-stream transformer blocks.

These checks establish checkpoint compatibility, scheduling, cache behavior,
joint video/text attention integration, and pipeline parity. They do not by
themselves establish a speedup or video-quality result.

## Tests

Run the default suite from the repository root:

```bash
python tests/run_tests.py
```

The default suite runs lightweight CPU contracts and skips expensive opt-in
tests. It covers causal scheduling and cache boundaries, checkpoint schema,
generation routing and resume behavior, ordinary-Wan hard-C4 compatibility,
Hunyuan joint attention, processor installation, multi-GPU launcher behavior,
and timing aggregation.

Run the checkpoint-free tiny Diffusers Hunyuan comparison with:

```bash
RUN_HUNYUAN_TINY_PARITY=1 \
PYTHONPATH=. \
python -m unittest tests.test_hunyuan_diffusers_tiny_parity -v
```

Real-checkpoint tests require an exclusive GPU. Commands and detailed failure
interpretations are in [`tests/README.md`](tests/README.md) and
[`tests/TEST_EXPLANATIONS.md`](tests/TEST_EXPLANATIONS.md).

## Detailed documentation

- [`attention/README.md`](attention/README.md): hard-C4 equations, tensor
  shapes, and backend design.
- [`SELF_FORCING_TOKEN_MERGING_IMPLEMENTATION.md`](SELF_FORCING_TOKEN_MERGING_IMPLEMENTATION.md):
  causal port, execution-path comparison, file-by-file changes, validation,
  and current limitations.
- [`HUNYUAN_TOKEN_MERGING_IMPLEMENTATION.md`](HUNYUAN_TOKEN_MERGING_IMPLEMENTATION.md):
  joint video/text equations, Diffusers integration, launchers, timing
  definitions, and validation.
- [`tests/README.md`](tests/README.md): runnable test commands and latest
  recorded results.
- [`tests/TEST_EXPLANATIONS.md`](tests/TEST_EXPLANATIONS.md): what every test
  proves and why it is needed.
- [`docs/negative_results.md`](docs/negative_results.md): rejected optimization
  routes and benchmark evidence.

## Guardrails and current limits

- The maintained ordinary-Wan hard-C4 path keeps grid-local compression and
  uses `local_with_ith` C4 seeds.
- After token merging, keep full compressed attention and the entropy/routing
  bias.
- Do not use global/share-all compression, no-bias routing, sparse/windowed
  second-stage attention, or quantization for the maintained hard-C4 path.
- Cached causal Self-Forcing is validated only for `hard_route` C1 with the
  complete visible prefix.
- Hunyuan supports non-transposed `hard_route` C1/C2/C4/C6. Processor geometry
  is fixed at installation time and must match the requested video geometry.
- Passing parity tests proves execution semantics, not speed or video quality.
  Benchmark runtime, peak memory, and VBench quality on the target hardware.
- Full model generation is GPU- and memory-intensive; use GPUs with sufficient
  available memory.

Rejected directions and benchmark evidence are recorded in
[`docs/negative_results.md`](docs/negative_results.md).

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
