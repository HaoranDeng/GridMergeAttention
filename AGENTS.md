# GridMergeAttention repository rules

## Default I2V benchmark inputs

- Since the user's2026-09-24 instruction, future I2V20 experiments default to
  `prompts/i2v/vbench_i2v_20_sparkler_v1.json` and paired
  `prompts/vbench_i2v_20.{txt,images.txt}`. Keep samples000-018 unchanged;
  sample019 is `a person holding a sparkler in their hand` and its matching
  conditioning image. Use base seed0 plus the original prompt index.
- Verify image/prompt hashes. Every method and Dense reference must use the
  same input version. Do not silently reuse the old019 Dense result.
- Keep the old20 available explicitly as `vbench_i2v_20_legacy`. The019
  replacement was chosen after seeing results; preserve its post-hoc identity.
  Adopting it as the future default does not make old comparisons prespecified.
- Do not rewrite the inputs of active, frozen or restored jobs. Existing50-video
  jobs retain their original50 inputs, seeds and resume semantics.
- See `experiments/q80k90_i2v_20260924/README.md` for the published experiment
  code, full50-step protocol, gates, metrics and versioned input preparation.

## Default C8 and numerical acceptance

- GMA-C8 defaults to the r2 implementation. Keep the previous C8 available as
  `slow_reference`, invoked explicitly for comparisons and diagnostics.
- The user requires the same mathematical GMA operator and numerically close
  results, not bitwise identity. Do not require identical seeds, assignments,
  intermediates, or output bits as the default acceptance gate.
- R2 is explicitly approved as the default, including its finite-precision
  arithmetic and packed winner comparison. Near-tied assignments may differ;
  this is not permission for a materially different operator or large errors.
- Record numerical errors and chosen tolerances against `slow_reference`.
  Preserve finite outputs and empty-cluster semantics, and investigate
  material numerical or video-quality regressions. No universal numeric
  tolerance has been specified by the user; do not invent one on their behalf.

## Exact C4/C8 capacity invariant

- `C4` means exactly four merged K/V representatives per key block.
- `C8` means exactly eight merged K/V representatives per key block.
- C4 and C8 must not append, stream, retain, or route a singleton/raw/residual
  K/V token. In particular, `4+1` must never be called C4 and `8+1` must
  never be called C8.
- Do not add C4/C8 configs, adapters, kernels, tests, or reports that enable
  `SQ1`, singleton peeling, or any `C+1` route.
- Historical C4+SQ1 and C8+SQ1 artifacts are retired controls. They must not
  be used as active implementations, baselines, defaults, or starting points.
