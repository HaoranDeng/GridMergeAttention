# Self-Forcing Token Merging: Implementation Record

This document records the changes made to apply this repository's Token
Merging attention inside Self-Forcing's causal, blockwise Wan inference. It
also records the old execution path, the new execution path, validation
results, environment findings, and current limitations.

## Status

- Implementation: complete for the requested Wan2.1 T2V configuration.
- Fast tests: 17 passed; the two opt-in GPU tests are excluded by default.
- Real-checkpoint GPU test: passed on an NVIDIA A100-SXM4-40GB after strict EMA
  loading, all seven causal Token Merging prefixes, VAE decoding, and final
  trace/shape/cache assertions (`Ran 1 test in 208.966s`, `OK`).
- Dense Self-Forcing reference parity: passed against MonarchRT for two causal
  blocks using identical stochastic inputs (`MAE 0.0066297227`, cosine
  `0.99988222`, `Ran 1 test in 185.466s`, `OK`).
- No fallback or environment workaround remains in `wan/modules/model.py`.

## Problem in the previous execution path

The original shell script selected `configs/token_merging_wan_fewstep.yaml`.
`generate.py` then always constructed `TokenMergingWanPipeline`, whose wrapper
always instantiated `WanModel`:

```python
self.model = WanModel.from_pretrained(str(model_dir)).eval()
self.uniform_timestep = True
```

That path loaded the Self-Forcing EMA checkpoint and few-step schedule, but it
processed all 21 latent frames together through ordinary bidirectional Wan
self-attention. Replacing `WanModel`'s attention with GridMerge did not add the
causal temporal blocks, per-layer K/V caches, growing visible prefix, or clean
context commit that define Self-Forcing inference.

In other words, the old behavior was approximately:

```text
Self-Forcing checkpoint + DMD/few-step schedule
    -> ordinary WanModel
    -> all 21 latent frames in each model call
    -> bidirectional Token Merging self-attention
```

## New execution path

The dedicated causal config now selects this path:

```text
generate.py
    -> build_pipeline(causal: true)
    -> CausalTokenMergingWanPipeline
    -> WanDiffusionWrapper(is_causal=True)
    -> CausalWanModel
    -> CausalWanSelfAttention
    -> token_merging_attn_with_kv_cache
```

For the configured 21 latent frames and three frames per block, generation is:

```text
block starts: 0, 3, 6, 9, 12, 15, 18

per block:
    denoise at configured step 1
    denoise at configured step 2
    denoise at configured step 3
    denoise at configured step 4
    rerun the final clean prediction at context_noise=0 to commit K/V
```

The query always contains the current three-frame block. The visible K/V
prefix grows as follows:

```text
Q frames:   3, 3, 3,  3,  3,  3,  3
K/V frames: 3, 6, 9, 12, 15, 18, 21
```

With 30 transformer layers and five model evaluations per block, the complete
run should call cached Token Merging `7 * 5 * 30 = 1,050` times.

## File-by-file changes

### `attention/token_merging.py`

The public Token Merging dispatcher previously required equal Q, K, and V
sequence lengths. Cached causal attention instead has a short current-block Q
and a longer visible-prefix K/V.

Changes:

- K and V must still have equal lengths.
- Q may have a different temporal length, while batch, head count, and head
  dimension must match K/V.
- The result buffer is allocated from `q_blocks`, not `v_blocks`, so output has
  the current query length rather than the prefix length.
- Added `token_merging_attn_with_kv_cache`. It writes the current RoPE-applied
  K/V into the selected cache interval and attends only through `end_idx`.
  Future allocated cache slots are not visible.
- Cached causal Token Merging is deliberately restricted to the validated
  `layout: hard_route` and `num_merge_tokens: 1` GridMerge path.

### `attention/__init__.py`

Exports `token_merging_attn_with_kv_cache` for use by causal Wan attention.

### `wan/modules/causal_model.py` (new)

Adds the causal Wan execution classes adapted from MonarchRT's Self-Forcing
structure:

- `CausalWanSelfAttention`
- `CausalWanAttentionBlock`
- `CausalHead`
- `CausalWanModel`

Important behavior:

- Every self-attention layer receives its own persistent K/V cache.
- Repeated denoising calls for one block overwrite the same cache interval.
- Advancing to the next block extends the visible prefix.
- RoPE uses `current_start_frame`, so each temporal block receives its global
  frame offset.
- Per-frame timestep tensors remain `[batch, current_block_frames]`.
- Cross-attention K/V for prompt context is cached per transformer layer.
- Token Merging is configured on every causal self-attention layer.
- The learned module names and tensor shapes match ordinary `WanModel`, so the
  same checkpoint state dict can load strictly even though execution differs.
- A cacheless call raises an error instead of silently falling back to
  non-Self-Forcing behavior.

### `wan/modules/__init__.py`

Exports `CausalWanModel`.

### `utils/wan_wrapper.py`

`WanDiffusionWrapper` now accepts:

```python
is_causal=False
local_attn_size=-1
sink_size=0
```

When `is_causal=True`, it constructs `CausalWanModel`; otherwise it preserves
the previous `WanModel` path. Causal calls retain the full `[B, F]` timestep
tensor and forward the self-attention cache, cross-attention cache, and
`current_start_frame`.

### `pipeline/causal_token_merging_wan.py` (new)

Adds `CausalTokenMergingWanPipeline`.

It:

- constructs the wrapper with `is_causal=True`;
- sets `num_frame_per_block` and Token Merging arguments on the causal model;
- allocates per-layer self-attention and cross-attention caches lazily using
  the actual batch size, dtype, and device;
- resets caches between videos and reallocates them when a final sample batch
  has a different size;
- denoises only the current temporal block;
- evaluates every configured denoising step without indexing beyond the final
  step;
- commits clean context at `context_noise` after every block;
- assembles all generated blocks and decodes the completed latent video.

### `pipeline/__init__.py`

Exports both the causal and bidirectional pipelines.

### `generate.py`

Adds `build_pipeline(config, device)`:

```text
causal: true  -> CausalTokenMergingWanPipeline
causal: false -> TokenMergingWanPipeline
```

The startup message now distinguishes causal Self-Forcing from bidirectional
Wan generation.

After merging later performance work from `origin/main`, the entry point also
retains prompt-chunk batching, optional batched VAE decoding, uint8 CPU
transfer, and configurable fast video encoding. These are combined with, not
substituted for, sample batching, skip-existing resume behavior, balanced
prompt sharding, atomic output replacement, and causal pipeline routing.

### `configs/token_merging_self_forcing_causal.yaml` (new)

Declares the Self-Forcing requirements explicitly:

```yaml
causal: true
num_frame_per_block: 3
independent_first_frame: false
context_noise: 0
denoising_step_list: [1000, 750, 500, 250]
warp_denoising_step: true
```

It also enables the validated one-token `hard_route` GridMerge configuration
and uses a full visible prefix with `local_attn_size: -1`.

### `scripts/generate/generate_token_merging_self_forcing.sh`

- Selects `token_merging_self_forcing_causal.yaml` instead of the ordinary Wan
  few-step config.
- Passes `--sample_batch_size`.
- Defaults `SAMPLE_BATCH_SIZE` to one because the full 30-layer BF16 K/V cache
  is approximately 5.6 GiB per sample before weights and activations.
- Verifies that the selected config file exists.

### `pipeline/token_merging_wan.py`

The existing bidirectional path had been changed to evaluate the complete
denoising list but still accessed `index + 1` unconditionally on the final
iteration. It now re-noises only when another timestep exists. This keeps the
old noncausal path usable and ensures the final denoising step is evaluated.
The 2026-08-09 upstream hard-C4 simplification removed denoising-step attention
policies, so there is no longer a policy hook to update before those calls.

### `tests/` (new)

Added the built-in `unittest` suite and detailed explanations:

- `test_token_merging_cache.py`
- `test_causal_model_contract.py`
- `test_causal_attention_dispatch.py`
- `test_causal_pipeline_schedule.py`
- `test_bidirectional_pipeline_schedule.py`
- `test_generation_routing.py`
- `test_generation_batching.py`
- `test_self_forcing_e2e.py`
- `test_self_forcing_reference_parity.py`
- `reference_parity_runner.py`
- `_support.py`
- `run_tests.py`
- `compare_latents.py`
- `README.md`
- `TEST_EXPLANATIONS.md`

See `tests/TEST_EXPLANATIONS.md` for each test's purpose and failure meaning.

## Validation completed

Fast suite command:

```bash
/home/zrlu/miniforge3/envs/etm/bin/python tests/run_tests.py
```

Latest default result:

```text
Ran 19 tests in 5.595s
OK (skipped=2)
```

This means 17 fast tests passed and the real-checkpoint Token-Merging test plus
the MonarchRT differential test were skipped.

The first real-checkpoint attempt established that:

- the causal pipeline constructed successfully;
- the real Wan components loaded;
- `generator_ema` loaded strictly into the causal model;
- caches were allocated;
- the model reached the first causal attention layer on GPU.

It then failed before Token Merging because FlashInfer attempted to execute
`ninja`, but the test had been launched with the environment's Python by
absolute path while `etm/bin` was absent from `PATH`.

## Environment finding: `ninja`

`ninja` was already installed. It was not a missing package:

```text
/home/zrlu/miniforge3/envs/etm/bin/ninja
version: 1.13.0.git.kitware.jobserver-pipe-1
```

Also available:

```text
nvcc: /usr/local/cuda/bin/nvcc
c++:  /usr/bin/c++
```

The original failing launch used:

```text
/home/zrlu/miniforge3/envs/etm/bin/python ...
```

Using an interpreter by absolute path does not activate its Conda environment
or prepend `etm/bin` to subprocess `PATH`. FlashInfer therefore could not find
`ninja`. Running through the following form supplies the complete environment:

```text
/home/zrlu/miniforge3/bin/conda run --no-capture-output -n etm python ...
```

A standalone FlashInfer RoPE JIT call subsequently compiled and executed
successfully through `conda run`. No package installation or Wan code fallback
was required. The first compilation took approximately 15 minutes and emitted
no progress output.

The next real-checkpoint attempt reached cached Token Merging but eventually
used all of a 48 GB GPU. The cause was the test's use of:

```python
mock.patch.object(..., side_effect=traced_cache_attention)
```

`MagicMock` records its complete argument list for every call. One current
three-frame BF16 Q, K, and V set is roughly 41 MiB, and retaining those tensors
across as many as 1,050 calls can exceed 40 GiB. The test now patches the plain
tracing function with `new=traced_cache_attention`, which records only small
integer frame-count tuples and does not retain attention tensors. Sparse stage
and visible-prefix progress messages were also added.

With the corrected tracer, the next run strictly loaded the EMA checkpoint and
completed generation through all required attention prefixes:

```text
(Q=3, K/V=3), (3, 6), (3, 9), (3, 12), (3, 15), (3, 18), (3, 21)
```

It reached VAE decoding using approximately 22.25 GiB. At that point a new
root-owned AlphaFold process (PID 1971611) was also using 16.57 GiB on the same
40 GB GPU. The VAE needed another 1.14 GiB and failed because only about 681
MiB remained. This was GPU contention, not retained tracing tensors or a
FlashInfer/Conda failure. A truly exclusive GPU is still required for the final
decode and assertions.

The final rerun started on an idle A100-SXM4-40GB and remained exclusive. It
completed strict checkpoint loading, all 1,050 cached Token Merging calls, VAE
decoding, latent/video shape and finiteness checks, and final cache-index
assertions:

```text
Ran 1 test in 208.966s
OK
```

## Real-checkpoint integration command

Replace `0` with a free GPU index on the selected cluster:

```bash
cd /xuanwu-tank/east/zrl/training-free-videogen
FREE_GPU=0

CUDA_VISIBLE_DEVICES="${FREE_GPU}" \
RUN_SELF_FORCING_E2E=1 \
WAN_MODEL_ROOT=/xuanwu-tank/east/zrl/training-free-videogen/wan_models \
SELF_FORCING_CONFIG=/xuanwu-tank/east/zrl/training-free-videogen/configs/token_merging_self_forcing_causal.yaml \
SELF_FORCING_CHECKPOINT=/xuanwu-tank/east/zrl/training-free-videogen/checkpoints/self_forcing_dmd.pt \
TOKEN_MERGING_FORCE_TORCH_RMSNORM=1 \
/home/zrlu/miniforge3/bin/conda run --no-capture-output -n etm \
python -m unittest tests.test_self_forcing_e2e -v
```

The test intentionally emits little output between the initial test name and
its final result. During that interval it constructs several large components,
loads the 5.3 GB EMA checkpoint, performs any required FlashInfer JIT work, and
runs 1,050 cached Token Merging attention calls.

## Dense Self-Forcing parity with MonarchRT

An additional opt-in differential test now compares the dense causal baseline
to the source MonarchRT implementation. It makes no production-code change and
does not add a special attention skip path. Instead, it uses the configuration
switches that already exist:

```text
current repository: token_merging_args.enable = false
MonarchRT:           monarch_args.enable       = false
```

`tests/test_self_forcing_reference_parity.py` creates one deterministic BF16
fixture containing prompt embeddings, initial latent noise, and every
re-noising tensor. `tests/reference_parity_runner.py` launches each repository
in a fresh process to prevent their identically named `wan`, `pipeline`, and
`utils` packages from colliding.

The current side uses `etm`; the reference side uses the existing
`monarch_rt` environment. No packages were added. The test uses the real 1.3B
generator and strictly loads the same `generator_ema` checkpoint in both
processes. It substitutes a fixed text encoder and identity VAE only in the
test process so that the comparison ends at generator latents and does not
include tokenization, T5, VAE, or MP4 differences.

The initial attempt to load MonarchRT under `etm` stopped on its unrelated
`easydict` import. After learning that the `monarch_rt` environment exists, the
reference subprocess was changed to use
`/home/zrlu/miniforge3/envs/monarch_rt/bin/python`. A second setup failure
entered MonarchRT's training-only manual RoPE branch because the standalone
runner had not copied this line from MonarchRT's real `inference.py`:

```python
torch.set_grad_enabled(False)
```

The runner now applies that real inference precondition to both sides. No RoPE
or Wan model source was patched. MonarchRT's broad package initializers also
eagerly imported every unrelated pipeline and Torch quantization modules,
causing several minutes of shared-filesystem reads. The test runner now skips
only those package initializers while importing and executing the exact causal
pipeline, wrapper, model, and their direct dependencies.

The default test uses six latent frames: two three-frame blocks are the minimum
that exercise a growing causal prefix. It verifies:

- strict EMA loading on both sides;
- identical warped denoising schedules;
- exact replay of six transition-noise tensors;
- final global/local cache index 9,360 in all 30 layers;
- matching latent shapes;
- bounded MAE, RMSE, maximum error, cosine similarity, and elementwise error.

Latest result on an exclusive A100-SXM4-40GB:

```text
current:   torch 2.13.0+cu130 / CUDA 13.0
MonarchRT: torch 2.8.0+cu128  / CUDA 12.8
shape:     (1, 6, 16, 60, 104)
MAE:       0.0066297227
RMSE:      0.0085105132
max error: 0.046875
cosine:    0.99988222

Ran 1 test in 185.466s
OK
```

Small BF16 differences are expected from the two native PyTorch/CUDA stacks.
The aggregate and elementwise tolerances are recorded in
`tests/TEST_EXPLANATIONS.md`.

Run the test with:

```bash
CUDA_VISIBLE_DEVICES=3 \
RUN_SELF_FORCING_PARITY=1 \
MONARCHRT_ROOT=/xuanwu-tank/east/zrl/MonarchRT \
MONARCHRT_PYTHON=/home/zrlu/miniforge3/envs/monarch_rt/bin/python \
SELF_FORCING_CHECKPOINT=/xuanwu-tank/east/zrl/training-free-videogen/checkpoints/self_forcing_dmd.pt \
SELF_FORCING_PARITY_ARTIFACT_DIR=/tmp/self_forcing_reference_parity \
/home/zrlu/miniforge3/bin/conda run --no-capture-output -n etm \
python -m unittest tests.test_self_forcing_reference_parity -v
```

## Post-merge reconciliation with `origin/main`

On 2026-08-07, `origin/main` was merged after it added a separate set of
ordinary-Wan performance changes. Its intent was not causal Self-Forcing. It
added:

- multiple prompt/sample chunks per generation call;
- optional batched VAE decoding;
- uint8 conversion before CPU transfer;
- faster configurable video encoding;
- denoising-step-aware hard-route backend policies for ordinary `WanModel`.

Accepting the remote conflict regions verbatim left an inconsistent hybrid:
`build_pipeline()` still existed but `main()` directly constructed
`TokenMergingWanPipeline`; resume/sharding arguments remained but their loop
was gone; the final status message referenced undefined counters; and the
ordinary pipeline iterated over `timesteps[:-1]`, undoing the complete-step
fix.

The manual reconciliation keeps both feature sets:

```text
remaining-sample scan and balanced sharding
    -> split each prompt into --sample_batch_size chunks
    -> concatenate up to --batch_size chunks per pipeline call
    -> build_pipeline(causal config selects Self-Forcing)
    -> uint8 CPU transfer and configurable fast encoding
    -> atomic final output replacement
```

For the ordinary bidirectional pipeline, every configured timestep is still a
model evaluation. Re-noising occurs only between adjacent evaluations.

Four regression tests were added for the merge boundary:

- `main()` must call `build_pipeline()`;
- batching resumed work must preserve the exact missing sample indices;
- a simulated partial run must batch and save exactly the missing outputs with
  the correct prompt/output ordering and fast-encoding arguments;
- four configured ordinary-Wan steps must produce four model calls and only
  three scheduler transitions.

### 2026-08-09 reconciliation with the simplified `origin/main`

The later upstream merge narrowed ordinary Wan attention to the maintained
hard-C4 implementations (a PyTorch reference and fused Triton kernel) and removed
the old generic common/GridMerge/select/transposed/shared modules and C2/C6
Wan configurations. The reconciliation does not restore that removed
experimental stack:

```text
ordinary bidirectional Wan -> upstream hard C4 PyTorch or fused Triton
causal Self-Forcing        -> isolated one-token causal GridMerge + KV cache
Hunyuan                    -> independent joint video/text hard C2/C4/C6
```

General grid/seed utilities needed by the latter two paths now live in the
private `attention/_grid_utils.py`; the causal one-token primitive lives in
`attention/causal_gridmerge.py`. `attention/token_merging.py` dispatches only
upstream Wan C4 or causal C1. A compatibility test compares the C4 dispatcher
directly with upstream's PyTorch reference, checks that the fused backend is
still selected, and requires removed Wan C2/C6 paths to remain rejected.

The obsolete Wan six-configuration launchers were removed because every
configuration they referenced was deleted by upstream. The Hunyuan hard-only
C2/C4/C6 launcher is independent and remains available.

The complete fast suite after reconciliation passed:

```text
Ran 19 tests in 5.595s
OK (skipped=2)
```

The two opt-in tests represented by those skips were then rerun separately on
an exclusive A100-SXM4-40GB (GPU 3 on `scai3`). The real-checkpoint causal
Token-Merging test observed all seven expected prefixes from `(Q=3, KV=3)`
through `(Q=3, KV=21)` and passed:

```text
Ran 1 test in 104.973s
OK
```

The dense two-block comparison against MonarchRT also passed. It reproduced
the previously recorded numerical baseline across the native `etm` and
`monarch_rt` environments:

```text
shape:     (1, 6, 16, 60, 104)
MAE:       0.0066297227
RMSE:      0.0085105132
max error: 0.046875
cosine:    0.99988222

Ran 1 test in 44.733s
OK
```

## Current limitations

- The new generation entry point supports text-to-video Self-Forcing. Initial
  latent video extension and the I2V `[1, N, N, ...]` layout are not exposed.
- Cached causal Token Merging is validated only for `hard_route` with one merge
  token. Other experimental layouts fail explicitly instead of silently being
  treated as supported.
- The dedicated configuration uses a full causal prefix. Rolling local cache
  support follows MonarchRT's indexing structure but is not the requested or
  fully validated generation configuration.
- The real-checkpoint integration test requires an exclusive GPU; unrelated
  workloads entering the same device can still make VAE decoding fail.
