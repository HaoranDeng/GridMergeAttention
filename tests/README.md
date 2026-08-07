# Self-Forcing Token-Merging Tests

For a test-by-test explanation of what each assertion proves and why it is
needed, see [`TEST_EXPLANATIONS.md`](TEST_EXPLANATIONS.md).

These tests are designed to prove two separate properties:

1. generation follows Self-Forcing's causal, block-wise, KV-cached inference;
2. the causal self-attention calls GridMerge with current-block Q and a growing
   visible K/V prefix.

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
- `test_causal_model_contract.py` checks checkpoint-compatible state-dict
  schemas, propagation of Token-Merging configuration, causal model selection,
  and preservation of per-frame timesteps.
- `test_causal_attention_dispatch.py` checks that causal self-attention sends a
  fixed-size current Q block through cache-aware Token Merging while K/V grows.
- `test_causal_pipeline_schedule.py` checks seven temporal blocks, all four
  denoising steps, three re-noising transitions, the clean-context commit, cache
  passing, and cache reallocation when the sample batch size changes.
- `test_generation_routing.py` checks that `generate.py` selects the causal
  pipeline, the dedicated config contains every causal setting, and the shell
  script uses that config with a safe sample batch size of one.
- `test_self_forcing_e2e.py` is the opt-in real-checkpoint GPU test.
- `test_self_forcing_reference_parity.py` is the opt-in dense causal
  differential test against MonarchRT.

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
Ran 1 test in 208.966s
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

Ran 1 test in 185.466s
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
