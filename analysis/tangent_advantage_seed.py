"""RF0 real-seed GS in exact tangent-advantage space, not logit space.

For seed s and target query i on the fixed Uniform16 support,
    D[s, i] = KL(p_s || p_i) = F(z_i) - (p_s @ z_i + H(p_s)),
    A[s, i] = exp(-D[s, i]),
    X[s, i] = max(A[s, i] - A[anchor, i], 0).
X is the extra partition fraction represented by adding the seed tangent
to the anchor tangent envelope. Its coordinates are TARGET QUERIES. GS
seeks complementary advantage patterns, not the failed greedy-L1 tangent
coverage objective. It is a surrogate: actual hard cells, frame reuse and
output error do not inherit a monotonicity or optimality guarantee.

No new QK products, temperature, weights, fitting, or merge changes. Forming
the advantage matrix costs O(R^2 P), then three GS pivots cost O(C R^2).
"""

METHODS = ("tangent_advantage_gs",)


def tangent_advantage_features(z, anchor):
    """Return (X, D), both [bank, SEED candidate, TARGET candidate]."""
    import torch

    if z.ndim != 3 or z.shape[1] < 4 or z.shape[2] < 1:
        raise ValueError("expected [bank, candidates>=4, probes>=1]")
    anchor = torch.as_tensor(anchor, device=z.device, dtype=torch.long)
    if anchor.shape != (z.shape[0],):
        raise ValueError("one anchor candidate slot is required per bank")
    if not bool(torch.isfinite(z).all()) or bool(
        ((anchor < 0) | (anchor >= z.shape[1])).any()
    ):
        raise ValueError("invalid logits or anchor slot")
    logp = torch.log_softmax(z.double(), dim=-1)
    probability = logp.exp()
    # Axis 1 is the candidate supplying p_s; axis 2 supplies target log p_i.
    self_logp = (probability * logp).sum(-1)
    directed_kl = self_logp[:, :, None] - torch.bmm(
        probability, logp.transpose(1, 2)
    )
    directed_kl = directed_kl.clamp_min(0)
    # Exact self KL is zero; remove harmless reduction noise on the diagonal.
    directed_kl.diagonal(dim1=1, dim2=2).zero_()
    represented = (-directed_kl).exp()
    batch = torch.arange(z.shape[0], device=z.device)
    features = (represented - represented[batch, anchor, None]).clamp_min(0)
    features[batch, anchor] = 0
    return features, directed_kl


def tangent_advantage_indices(z, anchor):
    """Four distinct real candidate slots, fixed anchor followed by GS."""
    import torch

    residual, _ = tangent_advantage_features(z, anchor)
    anchor = torch.as_tensor(anchor, device=z.device, dtype=torch.long)
    batch = torch.arange(z.shape[0], device=z.device)
    used = torch.arange(z.shape[1], device=z.device)[None] == anchor[:, None]
    picks = [anchor]
    for _ in range(3):
        score = residual.square().sum(-1).masked_fill(used, -float("inf"))
        pick = score.argmax(-1)
        picks.append(pick)
        used[batch, pick] = True
        basis = residual[batch, pick]
        basis = basis / (basis.square().sum(-1, keepdim=True) + 1e-12).sqrt()
        residual = residual - (residual * basis[:, None]).sum(
            -1, keepdim=True
        ) * basis[:, None]
    return torch.stack(picks, dim=-1)


def _probe_banks(q, k, ids, table):
    """Gather RF0 query rows and each bank's own key frame/row U16 logits."""
    import torch

    heads = q.shape[2]
    head, column = ids % heads, (ids // heads) % 80
    row = (ids // (heads * 80)) % 45
    frame = ids // (heads * 80 * 45)
    real = table[row].long()
    query = q[0, real * 80 + column[:, None], head[:, None]].float()
    key = k[
        0,
        frame[:, None] * 3600 + row[:, None] * 80
        + torch.arange(0, 80, 5, device=q.device),
        head[:, None],
    ].float()
    logits = torch.bmm(query, key.transpose(1, 2)).double() * (128 ** -0.5)
    return logits, real, row


def select(q, k):
    """Return standard [1,1,21,45,80,heads,4] uint8 real-row seed tables."""
    import torch
    from attention.adaptive_seed_pytorch import multiscale_candidate_table

    if (
        q.ndim != 4 or q.shape != k.shape or q.shape[0] != 1
        or q.shape[1] != 75600 or q.shape[3] != 128 or q.shape[2] < 1
        or q.device != k.device or q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
    ):
        raise ValueError("expected matching BF16 [1,75600,heads,128] Q/K")
    table, anchors = multiscale_candidate_table(45, 32, q.device, geometry_count=45)
    heads = q.shape[2]
    total = 21 * 45 * 80 * heads
    result = torch.empty((total, 4), device=q.device, dtype=torch.uint8)
    with torch.inference_mode():
        for start in range(0, total, 1024):
            ids = torch.arange(start, min(start + 1024, total), device=q.device)
            logits, real, row = _probe_banks(q, k, ids, table)
            slots = tangent_advantage_indices(logits, anchors[row])
            result[start:start + len(ids)] = real.gather(1, slots).to(torch.uint8)
    return {METHODS[0]: result.reshape(1, 1, 21, 45, 80, heads, 4)}

