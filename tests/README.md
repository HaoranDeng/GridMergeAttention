# Video Token-Merging Tests

For a test-by-test explanation of what each assertion proves and why it is
needed, see [`TEST_EXPLANATIONS.md`](TEST_EXPLANATIONS.md).

These tests are designed to prove three separate properties:

1. generation follows Self-Forcing's causal, block-wise, KV-cached inference;
2. the causal self-attention calls GridMerge with current-block Q and a growing
   visible K/V prefix;
3. Hunyuan's joint video/text Token-Merging equations preserve masks and one
   shared normalization for one-token and hard C2/C4/C6 routing, and the
   processor/pipeline integration can reproduce untouched Diffusers Hunyuan
   when the attention core is exact.

They use Python's built-in `unittest` because the `etm` environment currently
does not include `pytest`.

## Fast suite

Run from the repository root:

```bash
/home/zrlu/miniforge3/envs/etm/bin/python tests/run_tests.py
```

Equivalently:

```bash
/home/zrlu/miniforge3/envs/etm/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

The suite was initially written as a failing executable specification. The
implemented contracts it now protects are:

- `wan.modules.causal_model.CausalWanModel`;
- `wan.modules.causal_model.CausalWanSelfAttention`;
- `attention.token_merging.token_merging_attn_with_kv_cache`;
- `pipeline.causal_token_merging_wan.CausalTokenMergingWanPipeline`;
- unequal Q and K/V sequence lengths in `token_merging_attn`.

The individual test files document the behavior they require:

- `test_token_merging_cache.py` checks the GridMerge equations, cache writes,
  unequal temporal lengths, and exclusion of unused future cache entries.
- `test_wan_hard_c4_compatibility.py` checks that the merged dispatcher still
  exactly matches main's maintained PyTorch hard-C4 path, preserves its new
  fused-backend dispatch, and rejects the Wan C2/C6 paths removed upstream.
- `test_causal_model_contract.py` checks checkpoint-compatible state-dict
  schemas, propagation of Token-Merging configuration, causal model selection,
  and preservation of per-frame timesteps.
- `test_causal_attention_dispatch.py` checks that causal self-attention sends a
  fixed-size current Q block through cache-aware Token Merging while K/V grows.
- `test_causal_pipeline_schedule.py` checks seven temporal blocks, all four
  denoising steps, three re-noising transitions, the clean-context commit, cache
  passing, and cache reallocation when the sample batch size changes.
- `test_bidirectional_pipeline_schedule.py` checks that the ordinary-Wan path
  evaluates every configured denoising step without re-noising after the final
  model call.
- `test_generation_routing.py` checks that `generate.py` both defines and
  actually uses causal routing, preserves resumed sample indices while prompt
  chunks are batched, contains every causal setting, and uses a safe default
  sample batch size of one.
- `test_generation_batching.py` runs the merged entry point with a fake
  pipeline and checks that only missing samples are batched, mapped back to the
  correct filenames, converted to uint8, and passed to atomic fast encoding.
- `test_self_forcing_e2e.py` is the opt-in real-checkpoint GPU test.
- `test_self_forcing_reference_parity.py` is the opt-in dense causal
  differential test against MonarchRT.
- `test_hunyuan_pipeline_reference_parity.py` is the opt-in differential
  test for the Hunyuan pipeline. It compares an untouched Diffusers
  Hunyuan run with our installed Hunyuan processor using a test-only exact
  dense attention core.
- `test_hunyuan_token_merging.py` checks the approximate joint video/text
  equations, mask behavior, and exact hard C2/C4/C6 partitions directly on
  CPU. Hunyuan remains independent of main's Wan-only hard-C4 backend.
- `test_hunyuan_pipeline_integration.py` checks runtime grid derivation and
  processor installation in both Hunyuan main block families.
- `test_hunyuan_generation_routing.py` checks CLI geometry propagation,
  deterministic samples, resume behavior, prompt sharding, VBench filenames,
  and atomic output routing with a fake pipeline.
- `test_hunyuan_multi_gpu_launcher.py` checks that the Hunyuan experiment
  launcher contains only hard C2/C4/C6, shards prompts across explicit GPUs,
  and rejects a launch with no reserved GPU list.
- `test_hunyuan_timing_summary.py` checks weighted per-video pipeline latency,
  effective parallel multi-GPU throughput, model-load-inclusive wall time, and
  atomic timing-summary output.
- `test_hunyuan_diffusers_tiny_parity.py` is the opt-in, checkpoint-free parity
  and hard C2/C4/C6 execution test against a tiny real Diffusers 0.34 Hunyuan
  transformer.

## Real-checkpoint GPU test

After the fast suite passes, run one full prompt with batch size one:

```bash
RUN_SELF_FORCING_E2E=1 \
WAN_MODEL_ROOT=/xuanwu-tank/east/zrl/training-free-videogen/wan_models \
SELF_FORCING_CONFIG=/xuanwu-tank/east/zrl/training-free-videogen/configs/token_merging_self_forcing_causal.yaml \
SELF_FORCING_CHECKPOINT=/xuanwu-tank/east/zrl/training-free-videogen/checkpoints/self_forcing_dmd.pt \
TOKEN_MERGING_FORCE_TORCH_RMSNORM=1 \
/home/zrlu/miniforge3/bin/conda run --no-capture-output -n etm \
python -m unittest tests.test_self_forcing_e2e -v
```

The test loads the EMA checkpoint strictly, generates one video, and records
every cache-aware GridMerge call. For each transformer layer, the expected
frame trace is:

```text
(Q=3, KV=3)   repeated 5 times
(Q=3, KV=6)   repeated 5 times
...
(Q=3, KV=21)  repeated 5 times
```

Five calls per block means four denoising calls plus one clean-context cache
commit. With seven blocks and thirty layers, this is 1,050 cache-aware
Token-Merging calls.

Latest real-checkpoint result on an exclusive A100-SXM4-40GB:

```text
Ran 1 test in 104.973s
OK
```

## Dense Self-Forcing parity with MonarchRT

The parity test runs this repository in `etm` and the reference repository in
its `monarch_rt` Conda environment. It disables Token Merging through our
existing `token_merging_args.enable` switch and disables Monarch through
MonarchRT's existing `monarch_args.enable` switch. It does not add a test-only
branch to either production pipeline.

The default comparison uses two three-frame causal blocks. Both subprocesses
strictly load the same `generator_ema` state and receive byte-identical prompt
embeddings, initial noise, and all six per-transition re-noising tensors. T5
and VAE are replaced only in the test process because the comparison is at the
generator-latent boundary.

Run on an exclusive GPU:

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

Omit `SELF_FORCING_PARITY_ARTIFACT_DIR` to use an automatically cleaned
temporary directory. Set `SELF_FORCING_PARITY_FRAMES=21` for all seven blocks;
six frames is the default because it is sufficient to test a growing prefix.

Latest two-block result on an exclusive A100-SXM4-40GB:

```text
current:   torch 2.13.0+cu130 / CUDA 13.0
MonarchRT: torch 2.8.0+cu128  / CUDA 12.8
shape:     (1, 6, 16, 60, 104)
MAE:       0.0066297227
RMSE:      0.0085105132
max error: 0.046875
cosine:    0.99988222

Ran 1 test in 44.733s
OK
```

The default bounds account for BF16/kernel rounding across the two native
PyTorch/CUDA stacks while still detecting schedule or cache divergence. See
`TEST_EXPLANATIONS.md` for every invariant and configurable bound.

The standalone comparison helper can also inspect saved artifacts:

```bash
/home/zrlu/miniforge3/envs/etm/bin/python tests/compare_latents.py \
  /tmp/self_forcing_reference_parity/current_latents.pt \
  /tmp/self_forcing_reference_parity/monarchrt_latents.pt \
  --atol 5e-2
```

For debugging accumulated numerical differences, compare one generated prefix
at a time:

```bash
/home/zrlu/miniforge3/envs/etm/bin/python tests/compare_latents.py \
  /tmp/self_forcing_reference_parity/current_latents.pt \
  /tmp/self_forcing_reference_parity/monarchrt_latents.pt \
  --prefix-frames 3 --atol 5e-2
```

## Hunyuan pipeline parity with Diffusers

This test was added before the Hunyuan production implementation. The finished
implementation now satisfies its public interface and uses it as a numerical
regression baseline:

```text
pipeline.token_merging_hunyuan.install_hunyuan_token_merging_attention
attention.hunyuan_token_merging.hunyuan_token_merging_attn
```

The reference subprocess uses Diffusers' untouched Hunyuan attention. The
current subprocess installs our processor, then monkeypatches only its
low-level Token-Merging function to exact joint SDPA inside the test process.
That makes the two outputs directly comparable without adding a dense or skip
path to production code.

Both subprocesses load the same real Hunyuan transformer checkpoint in BF16 on
the same visible GPU. They receive fixed prompt embeddings, one padded text
token, fixed initial latents, and two denoising steps, then compare both
per-step and final latent output. Text encoding, VAE decoding, and MP4 encoding
are excluded so the test isolates the generator and denoising pipeline, just
like the Self-Forcing/MonarchRT parity test.

The existing `SVG` Conda environment is used only because it already contains
PyTorch and Diffusers 0.34.0 with `HunyuanVideoPipeline`. The test does not
import the Sparse-VideoGen repository or any `svg.*` module. `etm` remains the
test orchestrator, but its Diffusers 0.31.0 cannot execute HunyuanVideo.

First run the checkpoint-free whole-transformer parity test on CPU:

```bash
RUN_HUNYUAN_TINY_PARITY=1 \
PYTHONPATH=/xuanwu-tank/east/zrl/training-free-videogen \
/home/zrlu/miniforge3/envs/SVG/bin/python \
-m unittest tests.test_hunyuan_diffusers_tiny_parity -v
```

Then, on a GPU with the real model available, run:

```bash
CUDA_VISIBLE_DEVICES=3 \
RUN_HUNYUAN_PIPELINE_PARITY=1 \
HUNYUAN_PYTHON=/home/zrlu/miniforge3/envs/SVG/bin/python \
HUNYUAN_MODEL=/path/to/local/HunyuanVideo \
HUNYUAN_REVISION=refs/pr/18 \
HUNYUAN_LOCAL_FILES_ONLY=1 \
HUNYUAN_PARITY_ARTIFACT_DIR=/xuanwu-tank/east/zrl/hunyuan_pipeline_parity \
/home/zrlu/miniforge3/envs/etm/bin/python \
-m unittest tests.test_hunyuan_pipeline_reference_parity -v
```

`HUNYUAN_MODEL` may instead be a Hugging Face model ID. In that case omit
`HUNYUAN_LOCAL_FILES_ONLY=1`; downloads use `HF_HOME`, which defaults in this
test to `/xuanwu-tank/east/zrl/huggingface_cache`. The test deliberately does
not initiate a multi-gigabyte model download unless a remote ID is explicitly
provided.

With their environment flags omitted, both Hunyuan parity tests are skipped
during normal test discovery.
