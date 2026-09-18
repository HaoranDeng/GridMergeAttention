# GridMergeAttention repository rules

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
