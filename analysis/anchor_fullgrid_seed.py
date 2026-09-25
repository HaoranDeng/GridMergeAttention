"""Compare C32 and all 45 RF0 rows under the same anchor Fisher metric.

Unlike geometric Fisher's candidate-average reference, rho is fixed by the
actual retained row r. Expanding the candidate set therefore changes only
the pivot search, not the curvature or final C4 compression capacity.
This remains a greedy local surrogate, not an optimal hard-partition rule.
"""

METHODS = ("fisher_anchor_candidates32", "fisher_anchor_candidates45")


def anchor_features(z, anchor):
    """Return FP64 Fisher features and rho for [banks, candidates, probes]."""
    import torch

    if z.ndim != 3 or z.shape[0] < 1 or z.shape[1] < 4 or z.shape[2] < 1:
        raise ValueError("require logits [banks, candidates>=4, probes]")
    if not z.is_floating_point():
        raise TypeError("logits must have floating-point dtype")
    anchor = torch.as_tensor(anchor, device=z.device)
    if (anchor.shape != (z.shape[0],) or anchor.is_floating_point()
            or anchor.is_complex() or anchor.dtype == torch.bool):
        raise ValueError("anchor must contain one integer candidate slot per bank")
    anchor = anchor.long()
    if not torch.isfinite(z).all() or ((anchor < 0) | (anchor >= z.shape[1])).any():
        raise ValueError("finite logits and in-range anchor slots are required")
    z = z.double()
    batch = torch.arange(z.shape[0], device=z.device)
    anchor_z = z[batch, anchor]
    # Match select_population(secant=True)'s anchor branch numerically;
    # exp(log_softmax) is the same categorical distribution as softmax.
    rho = torch.log_softmax(anchor_z, dim=-1).exp()
    delta = z - anchor_z[:, None]
    feature = (delta - (delta * rho[:, None]).sum(-1, keepdim=True))
    return feature * rho[:, None].sqrt(), rho


def anchor_indices(z, anchor):
    """Pick four distinct real-candidate slots, with the given anchor first."""
    import torch

    residual, _ = anchor_features(z, anchor)
    anchor = torch.as_tensor(anchor, device=z.device, dtype=torch.long)
    batch = torch.arange(z.shape[0], device=z.device)
    used = torch.arange(z.shape[1], device=z.device)[None] == anchor[:, None]
    selected = [anchor]
    for _ in range(3):
        score = residual.square().sum(-1).masked_fill(used, -float("inf"))
        pick = score.argmax(-1)
        selected.append(pick)
        used[batch, pick] = True
        basis = residual[batch, pick]
        basis = basis / (basis.square().sum(-1, keepdim=True) + 1e-12).sqrt()
        residual = residual - (residual * basis[:, None]).sum(-1, keepdim=True) * basis[:, None]
    return torch.stack(selected, dim=-1)


def select(q, k, *, bank_chunk=2048):
    """Return both uint8 RF0 tables [B,1,key_frames,45,80,heads,4].

    For standard original BF16 Q/K [1,75600,H,128], each output has shape
    [1,1,21,45,80,H,4]. Only RF0 query vectors are read. The caller reuses
    these *row indices* and must load each frame's actual Q during merge.

    Q/K are cast from BF16 to FP32 for dot products, then logits are promoted
    to FP64 before scaling, the anchor profile, and GS. Uniform16 probes are
    fixed at columns 0,5,...,75. No value tensor, global normalizer, mean Q,
    trial partition, or cross-frame Q statistic is used. TF32 must be off.
    """
    import torch
    from attention.adaptive_seed_pytorch import multiscale_candidate_table

    if not isinstance(bank_chunk, int) or isinstance(bank_chunk, bool) or bank_chunk < 1:
        raise ValueError("bank_chunk must be a positive integer")
    if q.ndim != 4 or k.ndim != 4 or min(q.shape) < 1 or min(k.shape) < 1:
        raise ValueError("Q/K must be nonempty [B,frames*3600,H,D]")
    if (q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]
            or q.shape[1] % 3600 or k.shape[1] % 3600):
        raise ValueError("Q/K must share batch/head/dim and complete 45x80 frames")
    if q.device != k.device or q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise ValueError("Q/K must be original BF16 tensors on the same device")
    if q.is_cuda and torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("disable TF32 to preserve FP32 dot products")

    batches, _, heads, head_dim = q.shape
    frames = k.shape[1] // 3600
    table32, anchor32 = multiscale_candidate_table(45, 32, q.device, geometry_count=45)
    table45 = torch.arange(45, device=q.device)[None].expand(45, -1)
    anchor45 = torch.arange(45, device=q.device)
    tables = ((table32, anchor32), (table45, anchor45))
    total = batches * frames * 45 * 80 * heads
    outputs = {name: torch.empty((total, 4), device=q.device, dtype=torch.uint8) for name in METHODS}
    probes = torch.arange(0, 80, 5, device=q.device)
    for start in range(0, total, bank_chunk):
        ids = torch.arange(start, min(start + bank_chunk, total), device=q.device)
        head = ids % heads
        column = (ids // heads) % 80
        row = (ids // (heads * 80)) % 45
        frame = (ids // (heads * 80 * 45)) % frames
        batch = ids // (heads * 80 * 45 * frames)
        kk = k[batch[:, None], frame[:, None] * 3600 + row[:, None] * 80 + probes,
               head[:, None]].float()
        for name, (table, anchors) in zip(METHODS, tables):
            real = table[row].long()
            qq = q[batch[:, None], real * 80 + column[:, None], head[:, None]].float()
            z = torch.bmm(qq, kk.transpose(1, 2)).double() * (head_dim ** -0.5)
            slots = anchor_indices(z, anchors[row])
            outputs[name][start:start + len(ids)] = real.gather(1, slots).to(torch.uint8)
    return {name: rows.reshape(batches, 1, frames, 45, 80, heads, 4)
            for name, rows in outputs.items()}


__all__ = ["METHODS", "anchor_features", "anchor_indices", "select"]
