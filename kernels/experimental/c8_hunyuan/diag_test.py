"""Localize which stage (probe/select/merge/route) first produces NaN/Inf."""
import torch
import sys
sys.path.insert(0, '.')
from gma_c8h_r0 import create, _SHAPE  # noqa: E402

torch.manual_seed(0)
dev = 'cuda'
op = create(device=dev)

q = torch.randn(_SHAPE, device=dev, dtype=torch.bfloat16)
k = torch.randn(_SHAPE, device=dev, dtype=torch.bfloat16)
v = torch.randn(_SHAPE, device=dev, dtype=torch.bfloat16)
scale = 128 ** -0.5

op.probe.run(q[:, :1], k, op.probes, scale)
torch.cuda.synchronize()
probes = op.probes
print('probes: min', probes.min().item(), 'max', probes.max().item())
# probes should be valid column indices in [0, 44]
bad_probes = ((probes < 0) | (probes > 44)).sum().item()
print('bad probe indices (out of [0,44]):', bad_probes, '/', probes.numel())

op.gs.run(q[:, :1], k, op.probes, op.candidates, op.anchors, op.seeds, 0.2, scale)
torch.cuda.synchronize()
seeds = op.seeds.to(torch.int32)
print('seeds: min', seeds.min().item(), 'max', seeds.max().item())
bad_seeds = ((seeds < 0) | (seeds > 29)).sum().item()
print('bad seed indices (out of [0,29]):', bad_seeds, '/', seeds.numel())

op.merge.pack(q, op.packed_q)
torch.cuda.synchronize()
print('packed_q finite:', torch.isfinite(op.packed_q).all().item())

for head in range(0, 24, 8):
    op.merge.run(op.packed_q, k, v, op.seeds, op.mk, op.mv, op.bias,
                 head, 8, op.tiles, scale, op.empty, 0, 1)
    torch.cuda.synchronize()
    # .any() avoids .sum()'s int64 promotion, which would allocate a
    # same-shape int64 temp (~96 GiB for mk/mv) and OOM.
    mk_nan = torch.isnan(op.mk).any().item()
    mk_inf = torch.isinf(op.mk).any().item()
    mv_nan = torch.isnan(op.mv).any().item()
    mv_inf = torch.isinf(op.mv).any().item()
    bias_nan = torch.isnan(op.bias).any().item()
    bias_inf = torch.isinf(op.bias).any().item()
    bias_all_finite = torch.isfinite(op.bias).all().item()
    print(f'head={head}: mk_nan={mk_nan} mk_inf={mk_inf} mv_nan={mv_nan} mv_inf={mv_inf} '
          f'bias_nan={bias_nan} bias_inf={bias_inf} bias_all_finite={bias_all_finite} '
          f'bias_min={op.bias.min().item()}')
    if not bias_all_finite:
        finite_mask = torch.isfinite(op.bias)
        n_finite = finite_mask.sum(dtype=torch.int64).item() if finite_mask.numel() < 2**31 else -1
        print('  bias_finite_count (approx)', n_finite, '/', op.bias.numel())
        if finite_mask.any().item():
            print('  bias_max_finite', op.bias[finite_mask].max().item())
        del finite_mask
    output = torch.empty_like(q)
    op.route.run(q, op.mk, op.mv, op.bias, output, head, 8, scale)
    torch.cuda.synchronize()
    out_nan = torch.isnan(output).any().item()
    out_inf = torch.isinf(output).any().item()
    print(f'  route head={head}: any_nan={out_nan} any_inf={out_inf}')
    del output
