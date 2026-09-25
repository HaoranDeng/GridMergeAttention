"""RF0-only Fisher-GS on concatenated neighboring-column features.

Every column has its own geometric Fisher reference. A row is represented
by the concatenation of those features, so GS uses one projection coefficient
across the columns. Only the selected real row indices are shared: the caller
must still merge each column/frame using its own actual queries and SQ1.
This is a local quadratic surrogate, not a hard-partition error guarantee.
"""


def joint_indices(z, anchor):
    """Return four candidate slots from logits [banks, J, candidates, probes].

    J is 2 or 4; anchor is [banks]. Curvature and GS use FP64 after promoting
    the supplied logits. Features are concatenated without a 1/sqrt(J) scale,
    and every projection uses the existing 1e-12 denominator safeguard.
    Exact ties choose the first unused candidate slot. No logits, queries,
    or probability distributions are averaged across columns.
    """
    import torch

    if (z.ndim != 4 or z.shape[0] < 1 or z.shape[1] not in (2, 4)
            or z.shape[2] < 4 or z.shape[3] < 1):
        raise ValueError("require nonempty logits [banks, 2|4, candidates>=4, probes]")
    if not z.is_floating_point():
        raise TypeError("logits must have floating-point dtype")
    anchor = torch.as_tensor(anchor, device=z.device)
    if (anchor.shape != (z.shape[0],) or anchor.is_floating_point()
            or anchor.is_complex() or anchor.dtype == torch.bool):
        raise ValueError("anchor must contain one integer candidate slot per bank")
    anchor = anchor.long()
    if not torch.isfinite(z).all() or ((anchor < 0) | (anchor >= z.shape[2])).any():
        raise ValueError("logits must be finite and anchor slots must be in range")

    z = z.double()
    banks, width, candidates, probes = z.shape
    rho = torch.softmax(z.mean(dim=2), dim=-1)
    anchor_z = z.gather(2, anchor[:, None, None, None].expand(banks, width, 1, probes))
    delta = z - anchor_z
    residual = (delta - (delta * rho[:, :, None]).sum(dim=-1, keepdim=True))
    residual = residual * rho[:, :, None].sqrt()
    batch = torch.arange(banks, device=z.device)
    columns = torch.arange(width, device=z.device)
    used = torch.arange(candidates, device=z.device)[None] == anchor[:, None]
    selected = [anchor]
    for step in range(3):
        score = residual.square().sum(dim=(1, 3)).masked_fill(used, -float("inf"))
        pick = score.argmax(dim=-1)
        selected.append(pick)
        used[batch, pick] = True
        if step < 2:
            basis = residual[batch[:, None], columns[None], pick[:, None]]
            basis = basis / (basis.square().sum(dim=(1, 2)) + 1e-12).sqrt()[:, None, None]
            # One coefficient per candidate, summed over BOTH columns/probes.
            projection = (residual * basis[:, :, None]).sum(dim=(1, 3))
            residual = residual - projection[:, None, :, None] * basis[:, :, None]
    return torch.stack(selected, dim=-1)


def select(q, k, *, joint_width=4, bank_chunk=512):
    """Select RF0 rows; return {'fisher_jointJ': [B,1,F,45,80,H,4]}.

    Inputs are flattened original Q/K [B, frames*45*80, H, D]. Only the
    first query frame is read; every supplied key frame and key row is used.
    C32/g45 and Uniform16 columns 0,5,...,75 are fixed. The output is uint8,
    with four identical real row indices in every consecutive J-column group.

    As in select_refined, QK products use FP32 operands/accumulators before
    conversion to FP64 for the scale, Fisher reference, and joint GS. This
    is not a true FP64-QK oracle. No global normalizer, Q mean, KV merge, or
    route is evaluated here. bank_chunk changes only batching, not grouping.
    """
    import torch
    from attention.adaptive_seed_pytorch import multiscale_candidate_table

    if joint_width not in (2, 4) or isinstance(joint_width, bool):
        raise ValueError("joint_width must be 2 or 4")
    if not isinstance(bank_chunk, int) or isinstance(bank_chunk, bool) or bank_chunk < 1:
        raise ValueError("bank_chunk must be a positive integer")
    if q.ndim != 4 or k.ndim != 4 or min(q.shape) < 1 or min(k.shape) < 1:
        raise ValueError("Q/K must be nonempty [B, frames*3600, H, D]")
    if (q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]
            or q.shape[1] % 3600 or k.shape[1] % 3600):
        raise ValueError("Q/K must share batch/head/dim and complete 45x80 frames")
    if q.device != k.device or q.dtype != k.dtype or not q.is_floating_point():
        raise ValueError("Q/K must share device and floating-point dtype")

    batch_count, _, heads, head_dim = q.shape
    frames = k.shape[1] // 3600
    groups = 80 // joint_width
    table, anchors = multiscale_candidate_table(45, 32, q.device, geometry_count=45)
    total = batch_count * frames * 45 * groups * heads
    result = torch.empty((total, 4), device=q.device, dtype=torch.uint8)
    probe_columns = torch.arange(0, 80, 5, device=q.device)
    group_columns = torch.arange(joint_width, device=q.device)
    for start in range(0, total, bank_chunk):
        ids = torch.arange(start, min(start + bank_chunk, total), device=q.device)
        head = ids % heads
        group = (ids // heads) % groups
        row = (ids // (heads * groups)) % 45
        frame = (ids // (heads * groups * 45)) % frames
        batch = ids // (heads * groups * 45 * frames)
        real = table[row].long()
        columns = group[:, None] * joint_width + group_columns
        # No query-frame offset: these are RF0 row/column vectors only.
        qq = q[batch[:, None, None], real[:, :, None] * 80 + columns[:, None],
               head[:, None, None]].float()
        kk = k[batch[:, None], frame[:, None] * 3600 + row[:, None] * 80 + probe_columns,
               head[:, None]].float()
        logits = torch.bmm(qq.reshape(len(ids), 32 * joint_width, head_dim), kk.transpose(1, 2))
        logits = logits.double().reshape(len(ids), 32, joint_width, 16).permute(0, 2, 1, 3)
        slots = joint_indices(logits * (head_dim ** -0.5), anchors[row])
        result[start:start + len(ids)] = real.gather(1, slots).to(torch.uint8)

    result = result.reshape(batch_count, frames, 45, groups, heads, 4)
    result = result[:, :, :, :, None].expand(-1, -1, -1, -1, joint_width, -1, -1)
    result = result.reshape(batch_count, 1, frames, 45, 80, heads, 4)
    return {f"fisher_joint{joint_width}": result}


__all__ = ["joint_indices", "select"]
