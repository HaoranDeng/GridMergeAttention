"""Isolated RF0 prototype: one analytic zero query and three real query rows.

Slot zero in the returned table is only a placeholder. The caller MUST use
MeanSeedKernel with zero_queries(q); passing the table to ordinary four-real-
query merge implements a different method. This module does not merge/route,
change a production default, or claim a video-quality or timing improvement.
"""
from __future__ import annotations


def zero_indices(logits):
    """Three real candidate slots selected about the uniform/zero reference.

For probe logits z, use x = (z - mean(z)) / sqrt(P), the Fisher embedding
at zero logits. Greedy residual-norm GS chooses three distinct slots without
    forcing a real anchor. Exact ties follow candidate-table order. This is
    the mean-seed Fisher algebra with center zero, implemented independently
    to avoid importing a new helper through an older frozen source namespace.
"""
    import torch
    if (logits.ndim != 3 or logits.shape[0] < 1 or logits.shape[1] < 3
            or logits.shape[2] < 1 or not logits.is_floating_point()):
        raise ValueError("expected nonempty floating logits [banks,candidates>=3,probes]")
    if not torch.isfinite(logits).all():
        raise ValueError("finite logits required")
    z = logits.double()
    residual = (z - z.mean(dim=-1, keepdim=True)) / (z.shape[-1] ** 0.5)
    used = torch.zeros(z.shape[:2], device=z.device, dtype=torch.bool)
    batch = torch.arange(z.shape[0], device=z.device)
    chosen = []
    for step in range(3):
        pick = residual.square().sum(dim=-1).masked_fill(used, -float("inf")).argmax(dim=-1)
        chosen.append(pick)
        used[batch, pick] = True
        if step < 2:
            basis = residual[batch, pick]
            basis = basis / (basis.square().sum(dim=-1, keepdim=True) + 1e-12).sqrt()
            residual = residual - (residual * basis[:, None]).sum(dim=-1, keepdim=True) * basis[:, None]
    return torch.stack(chosen, dim=-1)


def _validate_query(q):
    import torch

    if (q.ndim != 4 or q.shape[0] != 1 or q.shape[1] != 21 * 45 * 80
            or q.shape[2] < 1 or q.shape[3] != 128 or q.dtype != torch.bfloat16):
        raise ValueError("require original BF16 Q [1,75600,heads,128]")


def zero_queries(q):
    """Return contiguous BF16 [1,21,80,heads,128] zeros for MeanSeedKernel.

No query values are read, averaged, normalized, or reused across frames.
The analytic seed is exactly zero at every query frame and column.
"""
    _validate_query(q)
    return q.new_zeros((1, 21, 80, q.shape[2], 128)).contiguous()


def select(q, k, *, bank_chunk=2048):
    """Return {'zero_fisher32': uint8 [1,1,21,45,80,heads,4]}.

Only RF0 query vectors are read. Each key frame/row, query column and head
gets three real rows from the original C32/g45 table and Uniform16 probes.
BF16 inputs are cast to FP32 for QK products, then promoted to FP64 before
scaling and GS. CUDA callers must disable TF32, as the guarded screen does.
Slot zero stores the matched row as a DUMMY identifier, NOT a real anchor.
Only slots 1:4 must be distinct; any may equal the dummy identifier.
"""
    import torch
    from attention.adaptive_seed_pytorch import multiscale_candidate_table

    _validate_query(q)
    if k.shape != q.shape or k.dtype != q.dtype or k.device != q.device:
        raise ValueError("Q/K must share the original BF16 standard shape and device")
    if not isinstance(bank_chunk, int) or isinstance(bank_chunk, bool) or bank_chunk < 1:
        raise ValueError("bank_chunk must be a positive integer")
    if q.is_cuda and torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("zero-anchor FP32 QK prototype requires allow_tf32=False")

    table, _ = multiscale_candidate_table(45, 32, q.device, geometry_count=45)
    heads = q.shape[2]
    total = 21 * 45 * 80 * heads
    output = torch.empty((total, 4), dtype=torch.uint8, device=q.device)
    probes = torch.arange(0, 80, 5, device=q.device)
    for start in range(0, total, bank_chunk):
        ids = torch.arange(start, min(start + bank_chunk, total), device=q.device)
        head = ids % heads
        column = (ids // heads) % 80
        row = (ids // (heads * 80)) % 45
        key_frame = ids // (heads * 80 * 45)
        real = table[row].long()
        # No query-frame offset: non-RF0 Q cannot affect selection.
        qq = q[0, real * 80 + column[:, None], head[:, None]].float()
        kk = k[0, key_frame[:, None] * 3600 + row[:, None] * 80 + probes,
               head[:, None]].float()
        logits = torch.bmm(qq, kk.transpose(1, 2)).double() * (128 ** -0.5)
        slots = zero_indices(logits)
        chosen = real.gather(1, slots)
        output[start:start + len(ids)] = torch.cat((row[:, None], chosen), dim=1).to(torch.uint8)

    return {"zero_fisher32": output.reshape(1, 1, 21, 45, 80, heads, 4)}


__all__ = ["select", "zero_indices", "zero_queries"]
