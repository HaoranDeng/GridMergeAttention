"""Explicit, bounded-roundoff contract for named C8 arithmetic experiments.

All other sources continue to require bitwise identity. Thresholds are fixed
before measurement, and cover the full intermediates and full H40 output.
"""
import math


NUMERICAL_SOURCES = frozenset({
    'fisher_c8_reuse_exponential_serial_v30.cu',
    'fisher_c8_reuse_exponential_rotated_v30.cu',
    'fisher_c8_polynomial_exp5_rotated_v31.cu',
    'fisher_c8_polynomial_exp6_rotated_v31.cu',
})


def compare_tensor(actual, reference, *, kind, chunk=1 << 26):
    import torch

    assert actual.shape == reference.shape and actual.dtype == reference.dtype
    assert kind in ('mk', 'mv', 'bias', 'output')
    limits = ({'relative_l2': 1e-5, 'max_abs_over_reference_max': 1e-4}
              if kind == 'bias' else
              {'relative_l2': 1e-3, 'max_abs_over_reference_max': 1e-2})
    a, b = actual.reshape(-1), reference.reshape(-1)
    err2 = torch.zeros((), device=a.device, dtype=torch.float64)
    ref2 = torch.zeros_like(err2)
    maxerr = torch.zeros((), device=a.device, dtype=torch.float32)
    maxref = torch.zeros_like(maxerr)
    bad = torch.zeros((), device=a.device, dtype=torch.int64)
    finite_count = torch.zeros_like(bad)
    for start in range(0, a.numel(), chunk):
        av, bv = a[start:start+chunk].float(), b[start:start+chunk].float()
        fa, fb = torch.isfinite(av), torch.isfinite(bv)
        # Equal infinities are valid bias sentinels. NaNs always fail.
        bad += ((fa != fb) | (~fa & (av != bv))).sum()
        finite_count += fb.sum()
        av, bv = torch.where(fa, av, 0.), torch.where(fb, bv, 0.)
        diff = av.double()-bv.double()
        err2 += diff.square().sum()
        ref2 += bv.double().square().sum()
        maxerr = torch.maximum(maxerr, diff.abs().max().float())
        maxref = torch.maximum(maxref, bv.abs().max())
    e2, r2, em, rm = (float(x) for x in (err2, ref2, maxerr, maxref))
    rel = math.sqrt(e2/r2) if r2 > 0 else (0. if e2 == 0 else math.inf)
    peak = em/rm if rm > 0 else (0. if em == 0 else math.inf)
    bad_count = int(bad)
    valid = all(math.isfinite(x) for x in (e2, r2, em, rm, rel, peak))
    return {'passed': valid and bad_count == 0 and rel <= limits['relative_l2']
            and peak <= limits['max_abs_over_reference_max'],
            'thresholds': limits, 'relative_l2': rel if math.isfinite(rel) else None,
            'max_abs': em, 'reference_abs_max': rm,
            'max_abs_over_reference_max': peak if math.isfinite(peak) else None,
            'invalid_or_mismatched_special_values': bad_count,
            'reference_finite_count': int(finite_count)}
