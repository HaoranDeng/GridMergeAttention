# HunyuanVideo Token Merging: Implementation Record

This document records the implementation added to generate HunyuanVideo with
this repository's GridMerge Token-Merging attention. It describes the actual
execution path, the mathematical boundary between merged and exact attention,
the files changed, environment requirements, validation, and current limits.

## Status

- Production implementation: complete for Diffusers HunyuanVideo text-to-video,
  including the hard-route C2, C4, and C6 experiment family.
- Fast CPU equation/installation/generation/launcher tests: passed.
- Tiny real-Diffusers transformer parity plus hard C2/C4/C6 block execution:
  both passed in the existing `SVG` Conda environment (`Ran 2 tests in
  7.558s`, `OK`; runtime varies on the shared filesystem).
- Complete fast regression suite: 38 tests, all 33 enabled tests passed and 5
  expensive opt-in tests were skipped as expected.
- Real-checkpoint dense-probe pipeline parity: passed exactly on an A100 using
  the cached 12.821B-parameter transformer. All 120 attention calls across two
  denoising steps were observed; final and per-step latent errors were zero.
  Peak allocated CUDA memory was 23.894 GiB.
- Full approximate 480p MP4 generation with hard C2/C4/C6 has not yet been
  executed. Only the transformer weights are currently cached, so the first
  full-pipeline run may still download tokenizers, text encoders, and the VAE.

## Previous and new execution paths

This repository previously had no Hunyuan generation path. Its production
entry point constructed either ordinary Wan or causal Self-Forcing Wan:

```text
generate.py
    -> TokenMergingWanPipeline or CausalTokenMergingWanPipeline
    -> repository-native Wan model
```

The new Hunyuan path is separate and leaves both Wan paths unchanged:

```text
generate_hunyuan.py
    -> load_hunyuan_token_merging_pipeline
    -> Diffusers HunyuanVideoPipeline (ordinary scheduler/denoising loop)
    -> HunyuanVideoTransformer3DModel
    -> every dual-stream and single-stream block's attn processor
    -> HunyuanTokenMergingAttnProcessor2_0
    -> hunyuan_token_merging_attn
```

The standard Hunyuan model has 20 dual-stream transformer blocks and 40
single-stream blocks. The installer discovers both lists rather than
hard-coding those counts. Hunyuan's separate text-refiner attention remains
dense because it contains no video-token grid.

## Why Hunyuan needs a different Token-Merging core

Wan self-attention receives only video Q/K/V. Diffusers Hunyuan attention is
joint attention over a sequence ordered as:

```text
[video tokens | text tokens]
```

The text stream is also updated and passed into later transformer blocks.
Treating the entire sequence as a rectangular video grid would merge text
tokens into arbitrary spatial groups. Running video attention and text
attention independently would also be wrong because two separate softmaxes do
not preserve the competition between video and text keys.

For the one-token GridMerge case, the implemented rule is therefore:

```text
video queries:
    GridMerge the video keys/values
    keep text keys/values unmerged
    concatenate compressed-video logits and exact-text logits
    apply one shared softmax

text queries:
    exact dense attention over original [video | text] keys/values
```

The Hunyuan boolean key mask is applied to both paths. Prompt padding can never
contribute to either video or text output. Spatial padding added for reduction
factors is also marked invalid and cropped from the final video result.

Keeping text-query attention exact is inexpensive relative to full video
attention because the text sequence is small. More importantly, it avoids
compounding an unvalidated approximation through Hunyuan's recurrently updated
text stream.

For the requested multi-token hard-routing runs (C2, C4, and C6), each video
block instead does the following:

```text
choose C deterministic seed-query rows
    -> score every video key against the C seeds
    -> hard-assign every video key to its highest-scoring seed
    -> softmax and merge K/V within each assigned cluster
    -> apply the hard-route entropy correction
    -> concatenate C compressed-video logits with exact-text logits
    -> apply one shared softmax
```

This is the same hard-assignment/merge equation used by the repository's Wan
hard-route attention. The Hunyuan-specific part is preserving exact text keys
inside that same normalization and keeping all text-query attention dense.
Select routing and transposed layouts are deliberately not enabled.

## File-by-file changes

### `attention/hunyuan_token_merging.py` (new)

Implements joint Hunyuan attention using the public tensor contract
`[batch, sequence, heads, head_dim]`.

It:

- validates video/text stream boundaries and the boolean key mask;
- derives temporal length from the video sequence and runtime spatial grid;
- reuses the existing GridMerge merge weights, entropy correction, block
  transforms, and spatial pad/crop helpers;
- normalizes compressed video keys and exact text keys in one softmax;
- evaluates exact text-query SDPA over the original joint sequence;
- supports bounded `query_outer_chunk` execution;
- supports the non-transposed `layout: hard_route` path for one token and for
  deterministic multi-token C2/C4/C6 routing;
- keeps select routing and transposed layouts disabled.

### `attention/hunyuan_attention_processor.py` (new)

Provides `HunyuanTokenMergingAttnProcessor2_0`, a drop-in Diffusers attention
processor. It preserves the upstream Hunyuan behavior around attention:

- regular and added Q/K/V projections;
- QK normalization;
- RoPE on video tokens only;
- separate dual-stream projections and shared single-stream projections;
- video/text stream splitting;
- `to_out` and `to_add_out` projections.

Only the joint SDPA primitive is replaced with the new Hunyuan Token-Merging
core. The call is module-qualified so the differential test can replace that
single primitive in a test subprocess without adding a production skip path.

### `pipeline/token_merging_hunyuan.py` (new)

Adds three public functions:

- `derive_hunyuan_video_grid` converts requested pixels/frames into the
  transformer's post-VAE, post-patch `(F, H, W)` token grid;
- `install_hunyuan_token_merging_attention` installs one processor in every
  dual- and single-stream main block;
- `load_hunyuan_token_merging_pipeline` loads the real transformer, scheduler,
  and Diffusers pipeline, installs processors, enables VAE tiling, and applies
  the selected CPU-offload/device policy.

For the supplied 480x720, 129-frame configuration, VAE compression and 2x2
transformer patches produce a `33 x 30 x 45` attention grid. Invalid pixel or
temporal geometry fails during installation, before a 13B denoising run starts.

The pipeline stores `_hunyuan_token_merging_config` as runtime diagnostic
metadata, including the derived grid and every installed processor.

### `pipeline/__init__.py`

Pipeline exports are now lazy. Importing the Hunyuan integration does not first
construct/import the unrelated Wan stack, while existing
`from pipeline import TokenMergingWanPipeline` usage remains supported.

### `attention/__init__.py`

Exports `hunyuan_token_merging_attn`.

### `generate_hunyuan.py` (new)

Adds a dedicated text-to-video CLI. It:

- reads the Hunyuan YAML configuration and prompt file;
- installs Token Merging for the configured generation geometry;
- generates samples sequentially with deterministic per-output seeds;
- supports resume through `--skip_existing`;
- supports deterministic prompt sharding across independent GPU workers;
- keeps every prompt and all of its missing sample indices on one worker;
- deduplicates repeated VBench prompts and can emit the required
  `<prompt>-<sample>.mp4` filenames;
- exports videos at the configured FPS using atomic partial-file replacement;
- accepts runtime overrides for model ID, `w_reduce`, and query chunk size.

It intentionally does not modify or overload the Wan-oriented `generate.py`.

### `configs/token_merging_hunyuan_480p.yaml` (new)

Defines the reference `tencent/HunyuanVideo` Diffusers revision, flow shift 7,
480x720x129 generation, BF16, whole-model CPU offload, VAE tiling, and the
validated one-token GridMerge settings.

`w_reduce: 1` follows this repository's existing GridMerge convention: each
key row's full width is merged to one routed token. Increasing `w_reduce`
creates more, narrower width groups and therefore reduces the compression.

### Hard-route experiment configurations (new)

The following configurations hold the Hunyuan model, resolution, 129-frame
length, 50-step denoising schedule, and all other Token-Merging settings fixed.
Only `num_merge_tokens` changes:

```text
configs/token_merging_hunyuan_hard_route_c2_480p.yaml  -> C=2
configs/token_merging_hunyuan_hard_route_c4_480p.yaml  -> C=4
configs/token_merging_hunyuan_hard_route_c6_480p.yaml  -> C=6
```

All three use `layout: hard_route` and `seed_mode: random_with_ith`. There are
no Hunyuan select-route experiment configurations.

### `scripts/generate/generate_token_merging_hunyuan.sh` (new)

Adds a reproducible launcher. By default, Hugging Face downloads and temporary
files stay on the shared project filesystem:

```text
HF_HOME=/xuanwu-tank/east/zrl/huggingface_cache
TMPDIR=/xuanwu-tank/east/zrl/tmp
```

### `scripts/generate/launch_hunyuan_hard_configs_multi_gpu.sh` (new)

Mirrors the existing six-config Wan launcher while running only the requested
hard-route variants. For each of C2, C4, and C6 it:

1. starts one independent Hunyuan process per explicitly reserved GPU;
2. balances unfinished prompt groups across those workers;
3. waits for all workers to finish before starting the next C value;
4. resumes valid existing MP4s and writes VBench-compatible filenames;
5. records one log per configuration and GPU;
6. aborts the sweep if any worker fails.

The default allows Hugging Face downloads because only transformer weights are
known to be cached. Set `LOCAL_FILES_ONLY=1` only after the complete Diffusers
pipeline is available locally.

### `requirements-hunyuan.txt` (new)

Records the Hunyuan-specific userspace packages. It deliberately does not pin
or install PyTorch because the PyTorch build must match the cluster CUDA
toolchain. The existing `etm` environment has Diffusers 0.31.0, which predates
the required Hunyuan classes; the existing `SVG` environment already has the
needed Diffusers 0.34.0/Transformers/Accelerate stack.

No `svg.*` package is imported by this implementation. `SVG` is only an
available Conda environment name.

### Tests (new and extended)

- `test_hunyuan_token_merging.py` checks the joint equations, shared
  video/text normalization, prompt masking, exact C2/C4/C6 partitions, parity
  of the multi-token video core with Wan hard routing, and explicit rejection
  of unsupported layouts.
- `test_hunyuan_pipeline_integration.py` checks runtime grid derivation and
  C2/C4/C6 option propagation into both main block families without touching
  the text refiner.
- `test_hunyuan_generation_routing.py` runs the CLI with a fake pipeline and
  checks geometry propagation, per-sample seeds, resume behavior, prompt
  sharding, VBench deduplication/names, and atomic output without model weights.
- `test_hunyuan_multi_gpu_launcher.py` proves that the sweep contains exactly
  hard C2/C4/C6, that its configs differ only where intended, and that the
  launcher refuses to run without an explicit GPU reservation.
- `test_hunyuan_diffusers_tiny_parity.py` compares a tiny untouched Diffusers
  Hunyuan transformer with an identical transformer using the production
  processor/core in an exact one-key-per-group limit, then smoke-tests hard
  C2/C4/C6 through real dual- and single-stream Diffusers blocks.
- `test_hunyuan_pipeline_reference_parity.py` and its runner compare the full
  real checkpoint and Diffusers denoising pipeline using a test-only dense core
  probe. These were written before implementation and now bind to the
  production installer and processor.

## Validation commands

Fast equation and installation tests:

```bash
TOKEN_MERGING_FORCE_TORCH_RMSNORM=1 \
/home/zrlu/miniforge3/envs/etm/bin/python -m unittest \
  tests.test_hunyuan_token_merging \
  tests.test_hunyuan_pipeline_integration \
  tests.test_hunyuan_generation_routing \
  tests.test_hunyuan_multi_gpu_launcher -v
```

Tiny whole-transformer Diffusers parity (no checkpoint or GPU):

```bash
RUN_HUNYUAN_TINY_PARITY=1 \
PYTHONPATH=/xuanwu-tank/east/zrl/training-free-videogen \
/home/zrlu/miniforge3/envs/SVG/bin/python -m unittest \
  tests.test_hunyuan_diffusers_tiny_parity -v
```

Generation after activating/installing a Hunyuan-capable environment:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/generate/generate_token_merging_hunyuan.sh
```

Hard-only C2/C4/C6 multi-GPU VBench sweep:

```bash
GPU_IDS="0 3 5 6" \
PYTHON_BIN=/home/zrlu/miniforge3/envs/SVG/bin/python \
bash scripts/generate/launch_hunyuan_hard_configs_multi_gpu.sh
```

The defaults generate five samples for every prompt into
`videos/token_merging_hunyuan_hard_configs/{hard_c2,hard_c4,hard_c6}`. Override
`GPU_IDS`, `NUM_SAMPLES`, `NUM_VIDEOS`, `PROMPT_PATH`, or `OUTPUT_ROOT` with
environment variables. Use only GPUs reserved for this run: every worker loads
its own Hunyuan pipeline and may also consume substantial host RAM.

The real-checkpoint dense pipeline parity command is documented in
`tests/README.md` and `tests/TEST_EXPLANATIONS.md`.

## Current limits and remaining empirical work

- Only non-transposed `hard_route` is enabled. The one-token case and hard
  C2/C4/C6 are covered by equation/integration tests. Select routing and
  transposed layouts are rejected rather than silently applying unvalidated
  joint-attention math.
- Processor geometry is fixed when installed. A caller that changes output
  resolution or frame count on the same loaded pipeline must reinstall the
  processors with the new geometry; the supplied CLI uses one fixed geometry.
- Passing parity proves integration semantics, not speed or video quality.
- The real 480p C2/C4/C6 path has not yet been GPU-smoked or memory-profiled.
  Multi-token seed logits require more temporary memory than the one-token
  path, so start with C2 and a short/single-prompt run before launching a full
  VBench sweep on 40 GiB cards.
- The real approximate core still needs a full-checkpoint GPU generation,
  peak-memory/runtime measurement, and dense-vs-Token-Merging quality
  evaluation (for example VBench) before making an acceleration-quality claim.
