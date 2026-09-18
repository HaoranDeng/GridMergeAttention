# GridMergeAttention repository rules

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
