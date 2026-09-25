# Exact C4/C8 capacity contract

This contract is mandatory for every active implementation, configuration,
experiment, result, and report in this repository.

- **C4 means exactly four merged K/V representatives per key block.**
- **C8 means exactly eight merged K/V representatives per key block.**
- C4 and C8 must not append, stream, retain, or route a singleton, SQ1,
  raw-K/V, residual, or other `C+1` token.
- `4+1` is not C4. `8+1` is not C8. Such methods must not be introduced under
  another spelling or through a hidden adapter.
- Historical C4+SQ1 and C8+SQ1 measurements are retired controls. They do not
  define the current method and must not be used as active baselines, defaults,
  performance claims, or starting points for optimization.

The executable contract lives in `attention/capacity_contract.py`. Public C4
and C8 wrappers validate it, and tests reject singleton counts for both
capacities.

## Fisher arithmetic policy

Active Fisher selectors use FP32 Gram--Schmidt arithmetic. The exact-capacity
contract is unchanged: C4 still returns four representatives and C8 still
returns eight, with no singleton/raw/residual route. FP64 selector sources are
kept only as numerical references and must not be production defaults for new
Fisher variants.
