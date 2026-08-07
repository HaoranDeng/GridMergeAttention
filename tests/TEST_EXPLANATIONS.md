# Detailed Test Explanations

This suite is an executable specification for the claim:

> Token Merging is applied inside Self-Forcing's causal, block-wise inference.

A generated MP4 is not sufficient evidence for that claim. The implementation
must select the causal pipeline, process temporal blocks, maintain the correct
KV prefix, commit clean context, and call Token Merging from causal
self-attention. The tests verify these requirements separately and then verify
their integration with the real checkpoint.

Before the causal implementation was added, the fast suite was intentionally
red and each failure identified a missing contract. The implemented suite now
passes; the real-checkpoint test remains opt-in because of its GPU and memory
cost.

## 1. GridMerge with causal Q/KV shapes

File: `test_token_merging_cache.py`

### `test_gridmerge_core_already_supports_short_q_and_long_kv`

This test creates synthetic attention tensors with two query frames and six
key/value frames:

```text
Q:   current 2-frame block
K/V: visible 6-frame prefix
```

It calls the low-level `gridmerge_one` implementation and compares its output
against the independent, slow PyTorch equations in `_support.py`.

Why this matters:

- Self-Forcing does not normally have equal temporal Q and K/V lengths.
- Q contains only the current temporal block.
- K/V contains all committed blocks plus the current block.
- Testing the low-level operation separately tells us whether the GridMerge
  mathematics or the surrounding dispatcher is the limiting component.

This test already passes. The core GridMerge einsums use separate query-outer
and key-outer dimensions, so they can represent different temporal lengths.

### `test_dispatcher_accepts_short_q_and_long_kv`

This test sends the same tensors through the public `token_merging_attn`
dispatcher. It must produce the same result as the independent reference.

The dispatcher now validates that K and V have equal lengths while allowing Q
to contain fewer complete frames. This test guards that causal requirement
against regression.

## 2. Cache-aware Token Merging

File: `test_token_merging_cache.py`

### `test_cache_write_and_attention_match_independent_reference`

The synthetic cache represents:

```text
Cache capacity:         8 frames
Previously committed:  4 frames
New current block:      2 frames
Visible prefix:         6 frames
Unused future capacity: 2 frames
```

The expected cache-aware API writes the new RoPE-applied K and V into
`start_idx:end_idx`, attends only through `end_idx`, and returns output for the
current Q block.

The test verifies:

1. the new K/V tensors overwrite exactly the current block's cache positions;
2. previous committed cache positions remain part of the visible prefix;
3. output matches an independent GridMerge calculation over
   `concat(previous_kv, new_kv)`;
4. unused future cache slots remain unchanged.

This catches off-by-one cache indices, stale current-block K/V, and attention
over allocated-but-not-visible cache capacity.

### `test_unused_future_cache_cannot_change_current_output`

Two caches receive the same visible prefix and current K/V. Their unused future
regions are filled with opposite extreme values:

```text
Cache A future: +1,000,000
Cache B future: -1,000,000
```

The two outputs must be identical. If they differ, the attention function read
past `end_idx`, which is direct future-information leakage.

This is stronger than a shape assertion because it verifies that invisible
cache values cannot numerically affect current output.

## 3. Checkpoint-compatible causal model

File: `test_causal_model_contract.py`

### `test_tiny_causal_and_bidirectional_models_have_identical_state_schema`

This constructs tiny `WanModel` and `CausalWanModel` instances with matching
dimensions. It compares every state-dict key and tensor shape and then performs
a strict state-dict load from the ordinary model into the causal model.

Why this matters:

- causal masks and KV caches should change execution, not learned parameters;
- `self_forcing_dmd.pt` stores tensors rather than the Python `forward` method;
- the causal class must keep names such as
  `blocks.0.self_attn.q.weight` unchanged;
- cache state should remain runtime state, not persistent checkpoint state.

This is a small, fast analogue of loading the real multi-gigabyte checkpoint.

### `test_token_merging_configuration_reaches_every_causal_block`

This applies `model.token_merging_args` to a small multi-layer causal model and
checks every transformer's causal self-attention object.

It prevents two silent mistakes:

- implementing the configuration property only on ordinary `WanModel`;
- enabling Token Merging on only some transformer layers.

Passing means all causal self-attention layers receive the requested layout,
merge count, reduction factors, and block size.

## 4. Causal wrapper behavior

File: `test_causal_model_contract.py`

### `test_wrapper_selects_causal_model_and_preserves_frame_timesteps`

The test replaces heavyweight model loading with a tiny recording model. It
constructs `WanDiffusionWrapper(is_causal=True)` and verifies that:

- `CausalWanModel.from_pretrained` is selected;
- `uniform_timestep` is false;
- a `[batch, frames]` timestep tensor reaches the model unchanged;
- causal cache arguments and `current_start_frame` can pass through the wrapper.

The current bidirectional wrapper reduces timesteps with `timestep[:, 0]`.
Retaining that behavior would remove frame-aware causal modulation even if a
causal model class existed.

## 5. Causal attention dispatch

File: `test_causal_attention_dispatch.py`

### `test_current_block_queries_attend_to_growing_prefix_through_gridmerge`

This constructs a tiny causal self-attention layer and simulates three
two-frame blocks. RoPE is replaced with an identity only to isolate dispatch and
cache indexing. A spy replaces cache-aware Token Merging and records semantic
frame counts.

The expected trace is:

```text
(Q=2, visible KV=2)
(Q=2, visible KV=4)
(Q=2, visible KV=6)
```

The test also checks the final global and local cache indices.

Why this matters:

A class can be named `CausalWanModel` while still invoking ordinary full-video
attention internally. This test checks the actual attention boundary: current
Q stays fixed, visible K/V grows, and the cache-aware Token-Merging function is
the selected implementation.

## 6. Self-Forcing temporal schedule

File: `test_causal_pipeline_schedule.py`

These tests use fake generator, scheduler, text encoder, and VAE objects. They
exercise pipeline control flow without loading Wan weights.

### `test_seven_blocks_run_four_steps_and_one_clean_commit_each`

For 21 latent frames and three frames per block, there must be seven temporal
blocks. Each block must execute:

```text
t=1000
t=750
t=500
t=250
t=context_noise=0 clean-cache commit
```

Therefore the expected counts are:

```text
Generator calls:     7 * (4 denoising + 1 commit) = 35
Scheduler re-noises: 7 * 3 transitions            = 21
```

The test verifies:

- every generator input contains only three frames;
- `current_start_frame` is `0, 3, 6, 9, 12, 15, 18`;
- all four denoising timesteps are evaluated;
- no `index + 1` access occurs after the final timestep;
- the fifth call receives the fourth call's clean prediction;
- KV and text cross-attention caches are supplied;
- seven predictions are assembled into one 21-frame latent.

This catches full-video generation, the existing terminal-step bug, a missing
clean-context rerun, and incorrect block assembly.

### `test_cache_is_reallocated_when_sample_batch_size_changes`

This runs the same pipeline first with batch size two and then with batch size
one. Every cache tensor must be reallocated for the second batch.

This matters because `--sample_batch_size`, `--skip_existing`, and resumed
generation can produce a smaller last batch. Reusing a cache allocated for the
previous batch can cause shape errors or mix stale sample state.

## 7. User-facing routing and configuration

File: `test_generation_routing.py`

### `test_build_pipeline_routes_causal_and_bidirectional_configs`

This checks the small `build_pipeline(config, device)` entry-point decision:

```text
causal: true  -> CausalTokenMergingWanPipeline
causal: false -> TokenMergingWanPipeline
```

This prevents a correct causal implementation from remaining unused because
`generate.py` still constructs the old pipeline unconditionally.

### `test_self_forcing_config_declares_all_causal_requirements`

This checks that `configs/token_merging_self_forcing_causal.yaml` explicitly
contains:

```yaml
causal: true
num_frame_per_block: 3
independent_first_frame: false
context_noise: 0
denoising_step_list: [1000, 750, 500, 250]
warp_denoising_step: true
token_merging_args:
  enable: true
  layout: hard_route
  num_merge_tokens: 1
```

The current repository does not merge MonarchRT's default configuration, so
these values cannot be assumed to arrive from another YAML file.

### `test_self_forcing_script_uses_causal_config_and_batch_one`

This checks that the shell script selects the dedicated causal config and
passes a default `--sample_batch_size 1`.

The full BF16 self-attention KV cache for Wan2.1-1.3B is approximately 5.6 GiB
per sample across thirty layers. Batch five could use roughly 28 GiB for K/V
alone, before model weights, activations, the text encoder, or VAE.

## 8. Real-checkpoint GPU integration

File: `test_self_forcing_e2e.py`

This expensive test is disabled unless `RUN_SELF_FORCING_E2E=1` is set. It:

1. loads the dedicated causal configuration;
2. constructs the causal Token-Merging pipeline;
3. strictly loads `generator_ema` from `self_forcing_dmd.pt`;
4. confirms the runtime model is `CausalWanModel`;
5. generates one sample in BF16 on a GPU;
6. records every cache-aware Token-Merging call;
7. checks latent/video shapes and finite values;
8. checks final cache indices.

For each transformer layer, the expected trace is:

```text
(Q=3, KV=3)   repeated 5 times
(Q=3, KV=6)   repeated 5 times
(Q=3, KV=9)   repeated 5 times
(Q=3, KV=12)  repeated 5 times
(Q=3, KV=15)  repeated 5 times
(Q=3, KV=18)  repeated 5 times
(Q=3, KV=21)  repeated 5 times
```

Five calls mean four denoising evaluations plus one clean-context commit. With
seven blocks and thirty layers, the total expected number of cache-aware
Token-Merging calls is 1,050.

Expected output includes a `[1, 21, 16, 60, 104]` latent and an 81-frame decoded
video. This test is the final integrated proof that the real checkpoint,
causal schedule, caches, and Token Merging operate together.

## 9. Dense causal differential test against MonarchRT

Files:

- `test_self_forcing_reference_parity.py`
- `reference_parity_runner.py`

This opt-in GPU test is disabled unless `RUN_SELF_FORCING_PARITY=1` is set. It
answers a narrower question than the Token-Merging E2E test:

> With both acceleration methods disabled, did we port MonarchRT's
> Self-Forcing generator semantics correctly?

The default fixture has six latent frames, which produces two three-frame
blocks. One block would test only the initial prefix; the second block is
needed to expose a wrong cache offset, missing clean-context commit, or failure
to attend to committed history.

### Inputs held constant

The parent test creates all stochastic inputs once on CPU in BF16:

- `[1, 6, 16, 60, 104]` initial latent noise;
- `[1, 512, 4096]` already-encoded prompt context;
- six `[3, 16, 60, 104]` re-noising tensors, corresponding to three
  transitions for each of two blocks.

The tensors themselves are replayed. Merely resetting a seed would be weaker
because RNG implementations can differ between PyTorch/CUDA versions.

Both subprocesses strictly load `generator_ema` from the same checkpoint. The
current subprocess uses `etm`; the reference subprocess uses
`/home/zrlu/miniforge3/envs/monarch_rt/bin/python`. Separate processes are
required because both repositories define top-level packages named `wan`,
`pipeline`, and `utils`.

The reference runner skips only MonarchRT's broad package initializers, which
eagerly import unrelated training pipelines and configurations. It still
executes the exact reference `pipeline/causal_inference.py`,
`utils/wan_wrapper.py`, `wan/modules/causal_model.py`, and their direct
dependencies. This avoids minutes of unrelated shared-filesystem reads without
changing causal execution.

### Attention and non-generator controls

The test uses existing public switches:

```text
current:   token_merging_args.enable = false
MonarchRT: monarch_args.enable       = false
```

No skip path was added to production code. MonarchRT's dense attention globals
are redirected, in its short-lived test subprocess only, to the same SDPA
helper API used by the current repository. This isolates pipeline/model/cache
differences from FlashAttention-versus-SDPA algorithm differences. Each helper
still executes in its native PyTorch/CUDA environment, so small BF16 rounding
differences remain expected.

A fixed text-encoder test double returns the saved prompt embeddings, and an
identity VAE leaves output at latent level. T5 tokenization and video decoding
are deliberately excluded: they are not part of the causal generator parity
question and would add memory, runtime, and MP4 quantization differences.

The runner also calls `torch.set_grad_enabled(False)`, exactly as MonarchRT's
`inference.py` does. This both prevents autograd allocation and selects
MonarchRT's intended inference RoPE path.

### Structural assertions

For each implementation, the test verifies:

- the strict checkpoint load has no missing or unexpected keys;
- the warped four-step schedule is identical;
- exactly six saved re-noising tensors are consumed;
- all thirty self-attention caches end at `6 * 30 * 52 = 9,360` tokens;
- both global and local cache indices reach that same end;
- returned latent shape is `[1, 6, 16, 60, 104]`.

These checks make a deceptively similar output harder to obtain through the
wrong schedule or a one-block/noncausal execution path.

### Numerical assertions

Because the real environments differ (`torch 2.13.0+cu130` versus
`2.8.0+cu128`), bitwise equality is not a valid requirement. The test checks
four complementary metrics by default:

```text
MAE       <= 0.0100
RMSE      <= 0.0125
max error <= 0.0625
cosine    >= 0.9995
```

It additionally applies elementwise `assert_close` with `rtol=0.01` and
`atol=0.05`. All bounds can be overridden with the
`SELF_FORCING_PARITY_*` environment variables defined in the test file.
Aggregate bounds prevent a uniformly shifted result from passing just because
the elementwise absolute tolerance is generous; the maximum bound catches a
small corrupted region.

The measured two-block result was:

```text
MAE       0.0066297227
RMSE      0.0085105132
max error 0.046875
cosine    0.99988222

Ran 1 test in 185.466s
OK
```

This is strong evidence that the dense causal port preserves MonarchRT's
Self-Forcing pipeline semantics before Token Merging is enabled. It does not by
itself prove Token Merging dispatch; that is covered by the cache/attention
unit tests and the real Token-Merging E2E test.

## 10. Standalone MonarchRT latent comparison

Script: `compare_latents.py`

This is a command-line comparison tool rather than an automatically discovered
unit test. Run both implementations with Token Merging disabled and save their
returned latent tensors. The runs must share:

- EMA checkpoint;
- prompt embeddings;
- initial noise;
- every per-step re-noising tensor;
- denoising timesteps;
- temporal block size.

The script reports mean absolute error, maximum absolute error, RMSE, and cosine
similarity, then uses `torch.testing.assert_close`.

It can compare an early prefix with `--prefix-frames 3`. Prefix comparison is
useful because backend rounding differences have not yet compounded through
later temporal blocks.

This establishes that the dense causal port matches MonarchRT before Token
Merging is introduced. Without that baseline, a difference could come from
RoPE offsets, timestep modulation, cache indexing, or the Self-Forcing schedule
rather than Token Merging.

## Supporting files

### `_support.py`

This contains the independent GridMerge equations, state-dict comparison, and
focused messages for missing implementation contracts. It also
works around the repository's eager `torch.cuda.current_device()` T5 default
during CPU-only test discovery. That workaround does not mock model execution.

### `run_tests.py`

This discovers all `test_*.py` files using Python's built-in `unittest`, so the
suite needs no additional testing package in the `etm` environment.

### `reference_parity_runner.py`

This is an internal subprocess entry point, not a separately discovered test.
It selects exactly one repository, installs fixed test dependencies, strictly
loads the checkpoint, runs dense causal inference, and saves latents plus
schedule/cache metadata for the parent differential test.

## What passing the complete suite means

No individual observation is sufficient:

```text
Checkpoint loads              != causal inference
Causal-looking class name     != causal attention
GridMerge function is called  != correct visible prefix
Valid MP4 is produced         != Self-Forcing semantics
```

The desired evidence is the conjunction:

```text
causal pipeline routing
+ three-frame block generation
+ growing KV prefix
+ clean-context cache commits
+ cache-aware GridMerge dispatch
+ no future-cache influence
+ strict Self-Forcing checkpoint loading
+ dense-causal parity with MonarchRT
```

Passing these tests supports the correctness claim. Video quality, latency, and
memory improvement remain separate empirical questions and should be evaluated
with VBench and runtime/memory profiling.
