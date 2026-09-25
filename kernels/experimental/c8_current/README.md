# Q45K80 C8 with current-frame selection

This opt-in Hopper kernel removes RF0 seed reuse from the shipped Hellinger
C8 r2. It selects independently for all 21 query frames. It keeps C32/g45,
alpha 0.2, the existing eight adaptive plus eight uniform probes (A8U8),
exactly eight merged representatives, the entropy bias, and full compressed
attention. This is a paired ablation of r2, so its probe policy is deliberately
preserved. It is separate from the recent anchor-Fisher spatial-axis control.

```python
from kernels.experimental.c8_current import create

attention = create(device="cuda")
output = attention(q, k, v)
# Optional matched RF0 control using the same merge implementation:
rf0_output = attention(q, k, v, policy="rf0")
```

Inputs are contiguous inference-only BF16 `[1, 75600, 40, 128]` or
`[1, 21, 45, 80, 40, 128]`, in token-major order, on Hopper SM90a.
`output=` accepts a separate contiguous output buffer. Production uses H40;
`heads=1` is a diagnostic specialization. Each instance owns about 33 GiB
of scratch and must be called serially on one stream. As with r2, the merge
workspace is reused in chunks of five heads. Probe scratch is reused across
query frames; the selected uint8 rows are retained separately for each frame.

The merge is derived from the approved r2 implementation. Its epilogue now
prefetches the next query frame's selected rows while storing merged K/V.
There is no extra representative, singleton, residual, retained raw key,
dense attention call, trained parameter, or calibration dependency.
All empty clusters retain zero K/V and positive-infinity route bias.

The default `kernels.best.create_c8` continues to select r2. Its explicit
`slow_reference` remains available. Removing RF0 is an algorithmic ablation,
not an arithmetic-equivalent replacement for r2.

`analysis/build_c8_current_sources.py` reproduces the CUDA derivatives from
the retained r2 sources. `analysis/benchmark_c8_current.py` provides complete
H40 paired timing, comparison against shipped r2 and `slow_reference`, and
the fixed-eight-layer quality replay. `analysis/validate_c8_current.py`
checks independent frame selection, independent sampled merge/route agreement,
and empty-cluster behavior. Run GPU commands under `analysis.guard_axis_job`.

The QKV protocol uses layers **0, 5, 10, 15, 20, 25, 30, 35**, three prompts,
four steps, heads 0/10/20/30, and 27 sampled queries per head. Missing captures
are an error. QKV relative-L2 results are not PSNR, SSIM, LPIPS, or VBench.
