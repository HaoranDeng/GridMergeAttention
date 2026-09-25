"""Smoke test for the C8-Hunyuan port: does it run and produce finite output?

Not a numerical-accuracy check against any reference -- see the port's
report for what remains open. This only checks: builds, runs without a CUDA
error, output shape matches input, and output has no NaN/Inf.
"""
import sys
import torch

sys.path.insert(0, '.')
from gma_c8h_r0 import create, _SHAPE, _FLAT  # noqa: E402


def main():
    torch.manual_seed(0)
    dev = 'cuda'
    print('Building extensions (probe/select/merge/route)...')
    op = create(device=dev)
    print('Built. Resources:', op.resources())

    for shape_name, shape in [('native', _SHAPE), ('flat', _FLAT)]:
        q = torch.randn(shape, device=dev, dtype=torch.bfloat16)
        k = torch.randn(shape, device=dev, dtype=torch.bfloat16)
        v = torch.randn(shape, device=dev, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        out = op(q, k, v, scale=128 ** -0.5)
        torch.cuda.synchronize()
        assert out.shape == q.shape, f'{shape_name}: shape mismatch {out.shape} vs {q.shape}'
        finite = torch.isfinite(out).all().item()
        nan = torch.isnan(out).any().item()
        inf = torch.isinf(out).any().item()
        print(f'{shape_name}: shape={tuple(out.shape)} finite_all={finite} '
              f'any_nan={nan} any_inf={inf} mean={out.float().mean().item():.6f} '
              f'std={out.float().std().item():.6f} max_abs={out.float().abs().max().item():.4f}')
        assert finite, f'{shape_name}: output has non-finite values'
    print('SMOKE TEST PASSED')


if __name__ == '__main__':
    main()
