"""RF0 real reverse-KL medoid initialization, then anchored Fisher-GS.

This isolated prototype changes seed selection only. All four returned rows
refer to actual queries. The matched key row remains eligible but is not
forced. No synthetic mean, partition refinement, global mass estimate, value
tensor, additional query frame, learned parameter, or route change is used.
"""
from __future__ import annotations


def medoid_indices(logits, *, return_costs=False):
    """Return four real candidate slots from [banks,candidates,probes].

First minimize KL(softmax(z_s) || softmax(mean_candidates(z))). This is the
exact discrete minimizer of the average reverse KL on this fixed probe cell.
Then use softmax(z_first) as the shared Fisher curvature, subtract z_first,
and select three residual-norm GS pivots. All scoring and projections use
FP64, stable log-softmax probabilities, and the existing 1e-12 safeguard.
Exact ties select the first unused candidate-table slot. This is not an
optimizer or monotone update of the final hard-partition attention objective.
"""
    import torch

    if (logits.ndim != 3 or logits.shape[0] < 1 or logits.shape[1] < 4
            or logits.shape[2] < 1 or not logits.is_floating_point()):
        raise ValueError("require nonempty floating logits [banks,candidates>=4,probes]")
    if not torch.isfinite(logits).all():
        raise ValueError("finite logits required")
    z = logits.double()
    logp = torch.log_softmax(z, dim=-1)
    log_center = torch.log_softmax(z.mean(dim=1), dim=-1)
    probabilities = logp.exp()
    costs = (probabilities * (logp - log_center[:, None])).sum(dim=-1)
    if not torch.isfinite(costs).all():
        raise ValueError("nonfinite medoid costs")
    first = costs.argmin(dim=-1)
    batch = torch.arange(z.shape[0], device=z.device)
    rho = probabilities[batch, first]
    delta = z - z[batch, first, None]
    residual = (delta - (delta * rho[:, None]).sum(dim=-1, keepdim=True)) * rho[:, None].sqrt()
    used = torch.arange(z.shape[1], device=z.device)[None] == first[:, None]
    chosen = [first]
    for step in range(3):
        pick = residual.square().sum(dim=-1).masked_fill(used, -float("inf")).argmax(dim=-1)
        chosen.append(pick)
        used[batch, pick] = True
        if step < 2:
            basis = residual[batch, pick]
            basis = basis / (basis.square().sum(dim=-1, keepdim=True) + 1e-12).sqrt()
            residual = residual - (residual * basis[:, None]).sum(dim=-1, keepdim=True) * basis[:, None]
    selected = torch.stack(chosen, dim=-1)
    return (selected, costs) if return_costs else selected


def select(q, k, *, bank_chunk=2048):
    """Return {'barycentric_medoid32': uint8 [1,1,21,45,80,heads,4]}.

Inputs are the original BF16 Q/K [1,75600,heads,128]. C32/g45 and Uniform16
probe columns 0,5,...,75 are fixed. Only RF0 Q is inspected, independently
for every key frame/row, query column and head. QK is computed with FP32
operands/accumulation, then promoted to FP64 before scaling and scoring.
CUDA callers must disable TF32. This is not a true FP64-QK oracle.

The caller shares only these indices across query frames and must merge
with each frame's original Q. Slot zero is a REAL medoid, not a dummy or a
forced matched row. No synthetic-seed merge adapter should be enabled.
"""
    import torch
    from attention.adaptive_seed_pytorch import multiscale_candidate_table

    if (q.ndim != 4 or q.shape[0] != 1 or q.shape[1] != 75600
            or q.shape[2] < 1 or q.shape[3] != 128 or q.dtype != torch.bfloat16):
        raise ValueError("require original BF16 Q [1,75600,heads,128]")
    if k.shape != q.shape or k.dtype != q.dtype or k.device != q.device:
        raise ValueError("Q/K must share original BF16 standard shape and device")
    if not isinstance(bank_chunk, int) or isinstance(bank_chunk, bool) or bank_chunk < 1:
        raise ValueError("bank_chunk must be a positive integer")
    if q.is_cuda and torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("disable TF32 to preserve FP32 QK products")

    table, _ = multiscale_candidate_table(45, 32, q.device, geometry_count=45)
    heads = q.shape[2]
    total = 21 * 45 * 80 * heads
    output = torch.empty((total, 4), device=q.device, dtype=torch.uint8)
    probes = torch.arange(0, 80, 5, device=q.device)
    for start in range(0, total, bank_chunk):
        ids = torch.arange(start, min(start + bank_chunk, total), device=q.device)
        head = ids % heads
        column = (ids // heads) % 80
        row = (ids // (heads * 80)) % 45
        key_frame = ids // (heads * 80 * 45)
        real = table[row].long()
        # No query-frame offset: non-RF0 Q values do not enter selection.
        qq = q[0, real * 80 + column[:, None], head[:, None]].float()
        kk = k[0, key_frame[:, None] * 3600 + row[:, None] * 80 + probes,
               head[:, None]].float()
        logits = torch.bmm(qq, kk.transpose(1, 2)).double() * (128 ** -0.5)
        slots = medoid_indices(logits)
        output[start:start + len(ids)] = real.gather(1, slots).to(torch.uint8)
    return {"barycentric_medoid32": output.reshape(1, 1, 21, 45, 80, heads, 4)}


__all__ = ["medoid_indices", "select"]
