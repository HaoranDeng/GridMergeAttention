# Grid Compressed Attention (GCA)

Retained fastest measured C4/C8/C10 BF16 kernels:
[kernel snapshots and usage](kernels/best/README.md).

## Default method

`generate.py` defaults to
`configs/token_merging_wan_720p_fisher_c4_exact_capacity_fewstep.yaml` when
`--config` is omitted. It selects four real RF0 seed rows from a C32/g45
candidate bank using Uniform-16 probes, then uses current-frame queries for
merging.

Each key block produces exactly four merged K/V representatives. There is no
SQ1, singleton peeling, raw-K/V side path, residual token, or fifth route
entry. The same operator replaces every self-attention call; there are no
learned corrections or dense rescue layers.

Historical C4+SQ1 and C8+SQ1 experiments are retired and are not runnable
methods or valid defaults. See the [exact-capacity contract](docs/exact_capacity_contract.md).

## Other maintained paths

Training-free Token Merging for text-to-video diffusion. This repository
contains three maintained inference paths:

- bidirectional Wan2.1 with dense attention, a hard-C4 PyTorch correctness
  reference, and the default optimized two-stage hard-C4 Triton backend;
- causal, block-wise Self-Forcing with a growing K/V cache and one-token
  GridMergeAttention;
- Diffusers HunyuanVideo with joint video/text hard-route Token Merging.

Model weights and Self-Forcing checkpoints are not stored in the repository.

The ordinary-Wan attention layer intentionally exposes only four maintained
choices: dense SDPA, dense FlashAttention Triton, original MonarchRT, and the
hard-C4 implementation. Older C2/C3, flat, KR/QC, one-pass, and
other kernel prototypes have been removed from the runnable code.

## Supported paths

| Path | Generation semantics | Token-Merging support | Entry point |
| --- | --- | --- | --- |
| Wan2.1 | All latent frames are denoised bidirectionally | Hard C4 with PyTorch and two-stage Triton implementations | `generate.py` |
| Self-Forcing | Seven causal three-frame blocks, with current-block Q and a growing visible K/V prefix | Cache-aware hard C1 | `generate.py` through the causal config |
| HunyuanVideo | Standard Diffusers denoising with joint `[video / text]` attention | PyTorch hard C1/C2/C4/C6; video K/V are merged while text remains exact | `generate_hunyuan.py` |

`C` is the complete merged-K/V capacity. C4 means exactly four representatives
and C8 means exactly eight. C4/C8 may never append a singleton or other `C+1`
route. It is not the head dimension or original key count.

For Hunyuan, video queries attend jointly to compressed video keys and exact
text keys under one shared softmax. Text-query attention remains exact and
dense over the original video and text tokens. The separate Hunyuan text
refiner is not replaced because it has no video-token grid.

## Ordinary Wan configurations

The dense baseline is:

- `configs/dense_wan_fewstep.yaml`

Besides the exact-capacity Fisher C4 default above, the maintained legacy hard-C4 configurations are:

- `configs/token_merging_wan_hard_c4_pytorch_fewstep.yaml`;
- `configs/token_merging_wan_hard_c4_triton_fewstep.yaml`;
- `configs/token_merging_wan_14b_hard_c4_triton_720p_fewstep.yaml`;
- `configs/token_merging_wan_14b_hard_c4_triton_720p_50step.yaml`.

The Wan 2.1 14B 720p baseline is
`configs/dense_wan_14b_720p_fewstep.yaml`. It uses a `45 x 80` spatial token
grid (`75,600` total tokens for 21 latent frames), while the 480p path uses
`30 x 52` (`32,760` total tokens).

Paired dense and hard-C4 50-step schedules are provided by
`configs/dense_wan_14b_720p_50step.yaml` and
`configs/token_merging_wan_14b_hard_c4_triton_720p_50step.yaml`. They use 50
evenly spaced training indices from 1000 through 20; `warp_denoising_step`
maps those indices onto the shifted flow schedule. Wan 14B has 40 transformer
blocks, so one prompt executes 2,000 self-attention calls instead of the 160
calls made by the four-step configs.

These two configs intentionally keep the existing
`Wan2.1-T2V-14B-StepDistill-CfgDistill` checkpoint so dense and hard C4 remain
directly comparable. They are a 50-step evaluation of that distilled model,
not the official base-model inference recipe, which also requires the base
`Wan2.1-T2V-14B` weights, classifier-free guidance, and a UniPC flow solver.

The original one-token MonarchRT comparison configurations are:

- `configs/token_merging_wan_monarchrt_h_reduce2_fewstep.yaml`;
- `configs/token_merging_wan_monarchrt_w_reduce4_fewstep.yaml`;
- `configs/token_merging_wan_monarchrt_triton_fewstep.yaml`.

Hard C4 now uses key-softmax assignment by default: seed-key logits are
normalized over all keys inside each block (80 at 720p; 52 at 480p) before each
key chooses its seed.
The later masked softmax merge and full compressed attention stay unchanged.

The PyTorch configuration is the readable correctness reference:

```yaml
token_merging_args:
  enable: true
  layout: hard_route
  num_merge_tokens: 4
  seed_mode: local_with_ith
  hard_route_backend: sdpa
```

The Triton configuration selects the default 2D-score-tile hard-C4 kernel:

```yaml
token_merging_args:
  enable: true
  layout: hard_route
  query_outer_chunk: 21
  head_chunk: 4
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

The default `hard_c4` Triton backend is an H100/H200-oriented fast path. The
480p kernel uses a padded `64 x 64` merge tile. The 720p kernel folds four
seeds into a `64 x 128` matrix tile, loops over five 16-column query tiles,
and uses a `64 x 64` full-attention route tile. Use the `sdpa` hard-C4
reference on GPUs that cannot launch these resource footprints.

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
`configs/token_merging_wan_hard_c4_triton_fewstep.yaml` for the hard-C4 Triton
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
| `configs/token_merging_wan_hard_c4_triton_fewstep.yaml` | Bidirectional Wan default optimized hard-C4 Triton backend |
| `configs/token_merging_wan_monarchrt_triton_fewstep.yaml` | Original one-token MonarchRT forward Triton backend |
| `configs/token_merging_wan_monarchrt_h_reduce2_fewstep.yaml` | Original one-token MonarchRT h-reduce comparison |
| `configs/token_merging_wan_monarchrt_w_reduce4_fewstep.yaml` | Original one-token MonarchRT w-reduce comparison |
| `configs/token_merging_self_forcing_causal.yaml` | Causal Self-Forcing with cache-aware hard C1 |
| `configs/token_merging_hunyuan_480p.yaml` | Hunyuan joint-attention hard C1 reference |
| `configs/token_merging_hunyuan_hard_route_c{2,4,6}_480p.yaml` | Hunyuan hard-route experiment family |

Ordinary Wan exposes only the hard-C4 PyTorch reference and the default
2D-score-tile hard-C4 Triton backend. Ordinary Wan C2/C6 remain unsupported.
Hunyuan C2/C4/C6 are independent implementations and do not use the
ordinary-Wan backend.

## Key experimental results

Unless noted otherwise, these numbers use real Wan2.1 BF16 self-attention QKV
from `configs/dense_wan_fewstep.yaml` and
`prompts/MovieGenVideoBench_extended.txt`. Error metrics compare attention
outputs against dense SDPA.

For MonarchRT paper-style sparsity, use the rank-1 Monarch block accounting:

```text
paper_density  = (b_q + b_k) / (b_q * b_k)
paper_sparsity = 1 - paper_density
```

where `b_q` and `b_k` are the query and key counts inside one Monarch block.
Do not confuse this with the engineering shortcut
`compressed_tokens_per_query / dense_tokens_per_query`. For a C-seed hard-route
comparison, report `C * (b_q + b_k) / (b_q * b_k)` as an approximate
paper-density analogue, because hard C4 is not exactly the original one-token
MonarchRT rank-1 block.

Wan 480p has a `30 x 52` latent grid per frame. The common paper-style
sparsity values are:

| Variant | Block shape | Approx. paper density | Approx. paper sparsity |
| --- | ---: | ---: | ---: |
| One-token MonarchRT `h_reduce=1,w_reduce=1` | `30 x 52` | 5.26% | 94.74% |
| One-token MonarchRT `h_reduce=2,w_reduce=1` | `15 x 52` | 8.59% | 91.41% |
| One-token MonarchRT `h_reduce=1,w_reduce=4` | `30 x 13` | 11.03% | 88.97% |
| Hard C2, rank-2 analogue | `30 x 52`, C=2 | 10.51% | 89.49% |
| Hard C4, rank-4 analogue | `30 x 52`, C=4 | 21.03% | 78.97% |

The full three-prompt key-assignment comparison on MI210 covered all 270 Wan
self-attention calls. Key-softmax assignment is the maintained default because
it substantially reduced dense-output error versus raw dot-product assignment,
although the standalone PyTorch analysis path was slower:

| Assignment | Calls | Mean ms/call | rel-L2 vs dense | MSE vs dense | Max abs | Changed keys vs raw dot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense SDPA | 270 | 66.82 | 0 | 0 | 0 | - |
| Raw dot | 270 | 514.28 | 0.08927 | 0.002335 | 8.2969 | 0.00% |
| Key-softmax | 270 | 544.11 | 0.07237 | 0.001534 | 6.6172 | 54.34% |

The following H200 block-layout table summarizes the most important current
layout comparisons. `compressed density` counts how many compressed K/V tokens
each query attends to relative to dense attention. `paper density` uses the
MonarchRT-style parameter/count estimate `C * (b_q + b_k) / (b_q * b_k)`.
The Hard C4 and MonarchRT rows are full three-prompt runs over all 360 recorded
Wan self-attention calls. The other GridMerge layout rows are first-eight-call
screens.

| Variant | q block | k block | C | Compressed density | Paper density | Paper sparsity | Mean ms/call | rel-L2 vs dense | MSE vs dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hard C4 Triton | 30 | 52 | 4 | 7.69% | 21.03% | 78.97% | 12.85 | 0.07389 | 0.001589 |
| Flat64 C4 | 30 | 64 | 4 | 6.41% | 19.58% | 80.42% | 211.58 | 0.09435 | 0.001147 |
| Q60 K64 C4 | 60 | 64 | 4 | 6.41% | 12.92% | 87.08% | 173.31 | 0.20142 | 0.005231 |
| Q60 K64 C4, x2r2 seeds | 60 | 64 | 4 | 6.41% | 12.92% | 87.08% | 173.28 | 0.21348 | 0.005877 |
| Q64 K64 C4 | 64 | 64 | 4 | 6.41% | 12.50% | 87.50% | 139.88 | 0.70185 | 0.063520 |
| MonarchRT h1 w1 | 30 | 52 | 1 | 1.92% | 5.26% | 94.74% | 4.28 | 0.17439 | 0.008853 |
| MonarchRT h2 w1 | 15 | 52 | 1 | 1.92% | 8.59% | 91.41% | 10.32 | 0.16295 | 0.007730 |
| MonarchRT h1 w2 | 30 | 26 | 1 | 3.85% | 7.18% | 92.82% | 11.96 | 0.16024 | 0.007474 |

A full three-prompt H200 stage profile recorded all 360 Wan self-attention
calls for the same method family. The table combines the latest confirmation
run for each variant; use the paired A/B numbers below for the merge-optimization
delta. Stage 1 is K/V compression or merge; Stage 2 is full attention over
original or compressed K/V tokens. Dense SDPA and FlashAttention have no merge
stage, so their Stage 1 time is zero. Stage profiling uses explicit per-stage
synchronization, so `other` includes Python, reshape, allocation, scatter, and
synchronization overhead.

| Variant | Custom kernel | Total ms/call | Stage 1 ms/call | Stage 2 ms/call | Other ms/call | Stage 1 share | rel-L2 vs dense | MSE vs dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense SDPA | No | 18.49 | 0.00 | 18.49 | 0.00 | 0.00% | 0 | 0 |
| FlashAttention Triton | Yes | 15.20 | 0.00 | 15.18 | 0.01 | 0.00% | 0.00109 | 0.000000343 |
| Hard C4 Triton | Yes | 12.85 | 8.70 | 4.06 | 0.09 | 67.70% | 0.07389 | 0.001589 |
| Flat64 C4 | No | 230.63 | 160.51 | 53.65 | 16.46 | 69.60% | 0.07965 | 0.001847 |
| Q60 K64 C4 | No | 184.86 | 96.87 | 43.42 | 44.57 | 52.40% | 0.15237 | 0.006759 |
| Q60 K64 C4, x2r2 seeds | No | 186.32 | 96.82 | 43.37 | 46.13 | 51.96% | 0.22277 | 0.014446 |
| Q64 K64 C4 | No | 155.18 | 94.71 | 43.43 | 17.05 | 61.03% | 0.74345 | 0.160896 |
| MonarchRT h1 w1 | Yes | 4.14 | 2.98 | 1.09 | 0.07 | 71.98% | 0.17439 | 0.008853 |
| MonarchRT h2 w1 | Yes | 10.00 | 5.73 | 4.15 | 0.11 | 57.33% | 0.16295 | 0.007730 |
| MonarchRT h1 w2 | Yes | 11.63 | 8.90 | 2.11 | 0.62 | 76.54% | 0.16024 | 0.007474 |

The current default hard-C4 merge uses four explicit 2D score tiles and one
`BLOCK_J=64` program for all 52 real query columns. This avoids carrying a
rank-3 `[C, query_columns, key_columns]` value through assignment and merge,
while preserving the same contiguous four-seed window, key-softmax assignment,
routing bias, and full compressed second-stage attention.

The merge loads the four seed-Q tiles in two pairs. It consumes the first pair
with the two corresponding QK dots before loading the second pair, while still
keeping the same four score tiles and the same downstream routing/merge order.
On Triton 3.7.1/sm90 this reduces compiler-reported shared memory from `81,920`
to `49,152` bytes per CTA; registers remain `255` per thread. A same-call
360-call A/B on a shared H200 measured the old schedule at `29.984` ms/call
with stage 1 at `20.112` ms, and the two-pair schedule at `29.318` ms/call
with stage 1 at `19.501` ms. That is a `2.27%` total-attention and `3.04%`
merge-stage latency reduction under the same contention. Merged K, merged V,
and routing bias were bitwise identical in the isolated comparison; the full
run kept rel-L2/MSE/max-abs at `0.0738906`/`0.00158936`/`6.28125`.

The last uncontended absolute production-dispatch confirmations before this
load-scheduling change,
`hard_c4_drop_dmask_q21_h4_real360_20260814` and
`hard_c4_drop_dmask_q21_h4_confirm2_real360_20260814` use the maintained
`query_outer_chunk=21`, `head_chunk=4` schedule. They measured hard C4 at
`12.732` and `12.759` ms/call versus dense SDPA at `18.747` and `18.765`
ms/call (`1.47x` speedup). Stage 1 was `8.528` and `8.547` ms/call, and stage
2 was `4.084` and `4.087` ms/call. Rel-L2 was `0.073887` and MSE was
`0.00158817`. They remain the absolute latency baseline until the two-pair
schedule can be rerun on an idle H200. The preceding power-of-two-D-mask-free `head_chunk=6`
confirmations averaged `12.854` ms/call with stage 1 at `8.699` ms/call. The
log2-domain, merge-`num_stages=2` confirmations averaged `12.857` ms/call, and
the merge-`num_stages=1` log2 confirmations averaged `12.873` ms/call. The
older single-J production baseline measured `13.209` ms/call with stage 1 at
`9.038` ms/call.

The follow-up `head_chunk=3`, `2`, and `5` standard runs measured `13.223`,
`12.986`, and `13.054` ms/call, so they did not beat the retained
`head_chunk=4` confirmations. A later same-QKV schedule sweep over
`q14_h6`, `q10_h8`, `q7_h12`, and `q21_h6` also did not beat `q21_h4`; the
run was lightly contended, so it is recorded as a negative scheduling result
rather than a new absolute baseline. A prob-dot merge variant that postponed
cluster normalization until after the K/V dot was also rejected: it is
algebraically equivalent in exact arithmetic, but the BF16 Wan path produced
NaN error metrics from tiny routed-cluster masses. The default therefore keeps
explicit normalized cluster weights before merging K/V.

The log2-domain merge computes the key-softmax assignment in log2 units and
converts the entropy routing bias back to natural-log units at the final bias
store; this keeps the same assignment/merge objective while removing repeated
large-tile `log2(e)` multiplications. The earlier single-J merge-grid cleanup
removes the dummy query-column launch axis because one `BLOCK_J=64` tile
already covers all 52 real Wan columns. In the paired same-QKV merge A/B run
`hard_c4_winner_prob_real360_20260814`, the preceding masked-route-id kernel
measured `15.072` ms/call with stage 1 at `10.791` ms/call, while the accepted
kernel measured `13.831` and `9.595` ms/call respectively. That is an `8.2%`
end-to-end and `11.1%` merge-stage reduction; rel-L2/MSE remained
`0.07389`/`0.001589` for both.

After assignment, the merge now keeps only the winning seed's normalized
log-probability. Exponentiating that value once and renormalizing it inside
each routed cluster gives the same cluster weights while the normalized
probabilities remain representable: the seed-wide log-normalizer is one scalar
per query row and cancels. The same shift also cancels in the entropy routing
bias. This removes four cluster-local masked maxima and replaces four full
score-tile exponentials with one. It retains the
accepted 2D score layout, masked route ids, key-only route padding mask, late
`store_base`, literal C4 stride, and full compressed second-stage attention.
The identity is exact algebraically; the implementation changes FP32 rounding
slightly. All 360 maintained RMS-normalized BF16 calls kept nonempty routed
cluster probability mass representable and showed no quality regression. A
cluster-max-centered numerical reference remains isolated for broader or
adversarial input validation, but was slower on H200.

The intermediate route-live-state confirmation
`hard_c4_route_mask_live_real360_20260814` measured hard C4 at `15.129`
ms/call versus dense SDPA at `18.591` ms/call. The address-live-range run
`hard_c4_storebase_late_literal4_confirm_real360_20260814` measured hard C4 at
`15.446` ms/call versus dense SDPA at `18.509` ms/call.
The intermediate route-mask confirmation `hard_c4_2d_route_keymask_real360_20260814`
measured hard C4 at `15.612` ms/call versus dense SDPA at `18.384` ms/call.
The earlier 2D fast-path confirmation `hard_c4_2d_seq_real360_20260814`
measured hard C4 at `15.668` ms/call versus dense SDPA at `18.391` ms/call.
The final tensorized baseline, including delayed V loading and a shared
converted weights tile, measured `18.04` ms/call with the same error. Restoring
the 2D merge and the subsequent live-range/probability-reuse cleanups therefore
lower end-to-end attention latency by about `24.2%` versus that final
tensorized baseline.
The tensorized form remains documented as a readability/debugging experiment
in `docs/negative_results.md`.

The `flat64_c4` row above is the original full 360-call PyTorch analysis
path. Commit `17ce1eb` added a dedicated two-stage Triton kernel for the same
layout. A 12-call H200 smoke run (`flat64_c4_triton_12call_20260814`) measured
`24.71` ms/call with `17.59` ms/call in merge and `5.04` ms/call in full
compressed attention; its rel-L2 and MSE versus dense were `0.11944` and
`0.002183` on those first 12 calls. This smoke is for kernel debugging and is
not a replacement for the full 360-call table.

An older three-prompt / three-denoise-step script compared the maintained hard
C4 route against original one-token MonarchRT `w_reduce=4`, recording 270
self-attention calls. Keep this as historical context only; new full real-QKV
comparisons should use the 360-call setting above. These two have the same
compressed token density `4/52 = 1/13`, but not the same MonarchRT paper-style
sparsity. Hard C4 produced much lower attention-output error, while MonarchRT
`w_reduce=4` remained faster in that older analysis script:

| Variant | Calls | Compressed-token density | Paper-style sparsity | Mean ms/call | rel-L2 vs dense | MSE vs dense | Max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense SDPA | 270 | 100.00% | 0.00% | 65.28 | 0 | 0 | 0 |
| Hard C4 | 270 | 7.69% | approx. 78.97% | 643.52 | 0.07241 | 0.001536 | 6.6250 |
| MonarchRT `w_reduce=4` | 270 | 7.69% | 88.97% | 190.38 | 0.14350 | 0.006034 | 11.7344 |

A smaller MI300X smoke compared original MonarchRT h-reduce variants and the
transpose layout over the first four self-attention calls. Treat this table as
a directional check, not a full quality benchmark:

| Variant | Calls | Paper-style sparsity | Mean ms/call | rel-L2 vs dense | MSE vs dense | Max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense SDPA | 4 | 0.00% | 22.96 | 0 | 0 | 0 |
| MonarchRT `h_reduce=2` | 4 | 91.41% | 38.00 | 0.11421 | 0.001682 | 4.2344 |
| MonarchRT `h_reduce=4` | 4 | 85.58% with padded height 32 | 88.55 | 0.12438 | 0.001995 | 4.2422 |
| Transpose C1 | 4 | 94.74% | 20.73 | 0.17528 | 0.003961 | 3.7891 |
| Transpose C4 | 4 | approx. 78.97% | 139.13 | 0.11520 | 0.001711 | 2.8191 |

The corresponding video samples were generated and copied to
`C:\ICLR 2026 Video`: 20 hard-C4 samples, 20 MonarchRT `w_reduce=4` samples,
and 20 MonarchRT `h_reduce=2` samples.

### Wan 2.1 14B at 720p

The dedicated 720p kernel uses the real `21 x 45 x 80` Wan token grid. A clean
H200 run compared both backends on the same BF16 QKV stream for one prompt,
four denoising steps, and all 40 self-attention layers (`160` calls):

| Variant | Calls | Mean attention ms/call | Speedup vs SDPA | Rel L2 vs dense | MSE vs dense | Max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense SDPA | 160 | 336.76 | 1.00x | 0 | 0 | 0 |
| Previous padded Hard C4 | 160 | 164.83 | 2.04x | 0.124030857 | 0.004806307 | 12.4570 |
| Hard C4 720p tensorized loop | 160 | 122.77 | 2.74x | 0.124030861 | 0.004806307 | 12.4570 |
| **Hard C4 720p grouped reuse** | **160** | **105.80** | **3.18x** | **0.124030915** | **0.004806311** | **12.4570** |

The accepted 720p merge folds the four seeds into the matrix row dimension,
so each iteration computes one `[4 * 16, 128] x [128, 128]` product. The current
kernel additionally groups seven query frames per program, loading each exact
80-token K/V row once and reusing it across those frames. The 80 keys are
handled as exact 64- and 16-key tiles, K and V are merged together, and the
route-contiguous workspace is consumed with a 64-token online-softmax loop.
The route kernel still performs full attention over all `21 * 45 * 4 = 3780`
compressed tokens. Seed selection, key-dimension softmax assignment, merged
K/V, routing bias, and the second-stage full attention are unchanged.

The real-QKV run above reduced latency by `13.8%` from the previous production
kernel while preserving its error level. A separate clean-H200 synthetic
regression measured `103.34` ms/call versus `332.70` ms/call for same-run SDPA
(`3.22x`) and was bit-identical to the optimized candidate. Production uses
`query_outer_chunk=21` and `head_chunk=8` for this 720p specialization.

The 720p seed window was subsequently widened without changing C=4, the
number of seed-key dot products, key-softmax assignment, entropy bias, or the
full second-stage attention. Interior key row `i` now uses
`[i - 2, i, i + 1, i + 3]`; boundary rows retain the old contiguous window.
Real Wan2.1-14B QKV validation sampled one query frame in every layer and
denoising step (`160` calls) and compared the old and new Triton kernels:

| Validation | Old MSE | Wide-seed MSE | MSE change | Old rel-L2 | Wide rel-L2 | Old / wide max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt 0, eight spread heads | 0.00387558 | 0.00368812 | -4.84% | 0.160932 | 0.156992 | 3.8516 / 4.3867 |
| Prompt 1, heads 0-7 | 0.00376100 | 0.00352042 | -6.40% | 0.161586 | 0.156333 | 6.1523 / 6.1484 |

The prompt-1 MI300X timing used the complete 40-head attention: SDPA ran all
heads together, while Hard C4 ran the production five sequential 8-head
chunks. The first call was excluded as warmup and `159` calls were averaged.

| Backend | Mean attention ms/call | Speedup vs same-run SDPA |
| --- | ---: | ---: |
| Dense SDPA | 359.07 | 1.00x |
| Old contiguous-seed Hard C4 | 181.95 | 1.973x |
| **Wide-seed Hard C4** | **182.56** | **1.967x** |

The wide window costs `0.34%` latency in this run while reducing MSE by
`6.40%`, so it is the maintained 720p seed pattern. The 480p kernel remains
unchanged.

A same-prompt, same-seed 50-step run measured all 2,000 self-attention calls
and excluded checkpoint loading from end-to-end time:

| Variant | Steps | Mean attention ms/call | Attention seconds | Inference seconds/video | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense SDPA | 50 | 337.02 | 674.04 | 878.63 | 1.00x |
| **Hard C4 720p grouped reuse** | **50** | **109.67** | **219.35** | **425.79** | **3.073x attention / 2.064x end-to-end** |

Attention accounted for `76.7%` of dense inference time and `51.5%` of Hard
C4 inference time. Video encoding took less than `0.65` seconds in both runs
and is reported separately from inference.

For context, the matching 14B 480p run measured `64.80` ms/call for SDPA and
`40.02` ms/call for hard C4 (`1.62x`), with rel-L2 `0.10455`. These numbers
must not be mixed with the 1.3B, 360-call benchmark because the model depth,
head count, and token length differ.

An earlier H200 hard-C4 Triton generation baseline used `num_stages=2`, a
delayed merge-kernel V load, direct `s - logZ` route comparisons, and
`input_precision="tf32"`. Its 20-prompt real-Wan-QKV benchmark covered all
2,400 self-attention calls:

| Variant | Calls | Mean attention ms/call | Attention speedup vs dense | Rel L2 vs dense | MSE vs dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense SDPA | 2,400 | 18.63 | 1.00x | 0 | 0 |
| FlashAttention Triton | 2,400 | 15.14 | 1.23x | 0.00112 | 0.000000374 |
| Hard C4 Triton, chunk 10 | 2,400 | 17.72 | 1.05x | 0.07924 | 0.001880 |

The corresponding clean H200 run generated 20 videos per backend. The samples,
logs, runtime summary, and attention metrics were copied to
`C:\ICLR 2026 Video\h200_current_baseline_20video_20260812_1720`:

| Variant | Videos | Total seconds | Seconds/video | End-to-end speedup vs dense |
| --- | ---: | ---: | ---: | ---: |
| Dense SDPA | 20 | 533.35 | 26.67 | 1.00x |
| FlashAttention Triton | 20 | 164.27 | 8.21 | 3.25x |
| Hard C4 Triton, chunk 10 | 20 | 170.33 | 8.52 | 3.13x |

Use the 2,400-call attention benchmark above as the kernel-speed comparison;
the end-to-end table includes the full `generate.py` path and video export.

A forward-only Triton backend for original one-token MonarchRT is available as
`hard_route_backend: monarch_rt` with `num_merge_tokens: 1`. On one H200, a
one-prompt real-Wan-QKV benchmark over all 120 self-attention calls gave:

| Variant | Calls | Mean attention ms/call | Attention speedup vs dense | Rel L2 vs dense | MSE vs dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense SDPA | 120 | 18.79 | 1.00x | 0 | 0 |
| MonarchRT Triton, chunk 21 | 120 | 4.30 | 4.36x | 0.17699 | 0.009506 |

The flash-attention load profile on MI210 used one real Wan QKV call with
`Q_len=K_len=32760`, 12 heads, and head dimension 128. PyTorch SDPA measured
about 63 ms/call. Standalone load-checksum kernels measured 284.29, 142.93,
72.22, 36.30, and 18.85 ms for block-M 16, 32, 64, 128, and 256 respectively.
These standalone kernels isolate global loads but are not sub-timings from the
fused SDPA kernel; use them only as memory-traffic diagnostics.

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
