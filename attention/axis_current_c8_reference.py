"""Independent FP64 Fisher selection / FP32 exact-capacity C8 reference."""
import itertools
from analysis.quick_axis_c4_qkv import candidate_table, SAMPLES

def prepare(k, v, qa, ka):
    import torch
    remaining = [axis for axis in range(3) if axis != ka]
    # Key groups preserve both fixed physical coordinates in lexicographic order.
    kb = k.permute(*remaining, ka, 3).reshape(-1, k.shape[ka], k.shape[-1]).contiguous()
    vb = v.permute(*remaining, ka, 3).reshape_as(kb).contiguous()
    coords = torch.cartesian_prod(*[torch.arange(k.shape[a], device=k.device) for a in remaining])
    anchor = coords[:, remaining.index(qa)]
    table, anchor_slots = candidate_table(k.shape[qa])
    candidate = torch.tensor(table, device=k.device)[anchor]
    slots = torch.tensor(anchor_slots, device=k.device)[anchor]
    probes = torch.arange(16, device=k.device) * k.shape[ka] // 16
    if k.shape[ka] < 16:
        probes = torch.arange(k.shape[ka], device=k.device)
    return kb, vb, candidate, slots, probes


def select_fisher(query, kb, candidates, anchor_slots, probes, scale):
    import torch
    z_all = torch.einsum('id,gpd->gip', query, kb[:, probes]) * scale
    z = z_all.gather(1, candidates[..., None].expand(-1, -1, len(probes))).double()
    groups = torch.arange(len(kb), device=query.device)
    anchor_z = z[groups, anchor_slots]
    rho = anchor_z.softmax(-1)
    delta = z - anchor_z[:, None]
    residual = (delta - (delta * rho[:, None]).sum(-1, keepdim=True)) * rho[:, None].sqrt()
    used = torch.zeros(candidates.shape, dtype=torch.bool, device=query.device)
    used[groups, anchor_slots] = True
    selected = [candidates[groups, anchor_slots]]
    for step in range(1, 8):
        score = residual.square().sum(-1).masked_fill(used, -float('inf'))
        picked = score.argmax(-1)
        selected.append(candidates[groups, picked])
        used[groups, picked] = True
        if step < 7:
            direction = residual[groups, picked]
            basis = direction / (direction.square().sum(-1, keepdim=True) + 1e-12).sqrt()
            residual -= (residual * basis[:, None]).sum(-1, keepdim=True) * basis[:, None]
    return torch.stack(selected, -1)


def line_attention(query, reference_query, kb, vb, candidates, slots, probes, sampled_indices, *, quantize=True):
    import torch
    scale = query.shape[-1] ** -.5
    seeds = select_fisher(reference_query, kb, candidates, slots, probes, scale)
    z = torch.bmm(query[seeds], kb.transpose(-1, -2)) * scale
    owner = z.log_softmax(-1).argmax(-2)
    assigned = owner[:, None] == torch.arange(8, device=query.device)[None, :, None]
    active = assigned.any(-1)
    selected = z.masked_fill(~assigned, -float('inf'))
    maximum = selected.amax(-1, keepdim=True)
    maximum = torch.where(active[..., None], maximum, 0.0)
    weights = torch.exp(selected - maximum)
    weights /= weights.sum(-1, keepdim=True).clamp_min(1e-30)
    entropy = -(weights * weights.clamp_min(1e-30).log()).sum(-1)
    # Match the BF16 representation boundary, preserving FP32 entropy.
    if quantize:
        merge_weights = weights.to(torch.bfloat16).float()
        mk = torch.bmm(merge_weights, kb).to(torch.bfloat16).float()
        mv = torch.bmm(merge_weights, vb).to(torch.bfloat16).float()
    else:
        mk, mv = torch.bmm(weights, kb), torch.bmm(weights, vb)
    logits = query[sampled_indices] @ mk.flatten(0, 1).T * scale
    logits += entropy.flatten()[None]
    logits.masked_fill_(~active.flatten()[None], -float('inf'))
    output = logits.softmax(-1) @ mv.flatten(0, 1)
    return output, {'empty': int((~active).sum()), 'slots': active.numel()}


def evaluate(q, k, v, qa, ka, policy, samples=SAMPLES, quantize=True):
    import torch
    kb, vb, candidates, slots, probes = prepare(k, v, qa, ka)
    other = [a for a in range(3) if a != qa]
    rows, empty, total = {}, 0, 0
    for fixed in itertools.product(*[samples[a] for a in other]):
        index = [slice(None)] * 3
        for axis, value in zip(other, fixed):
            index[axis] = value
        line = q[tuple(index)]
        reference_index = list(index)
        if policy == 'rf0':
            if qa == 0:
                raise ValueError('RF0 cannot supply four temporal query seeds')
            reference_index[0] = 0
        reference_line = q[tuple(reference_index)]
        out, counts = line_attention(line, reference_line, kb, vb, candidates, slots, probes,
                                    list(samples[qa]), quantize=quantize)
        empty += counts['empty']; total += counts['slots']
        for inner, value in enumerate(samples[qa]):
            coordinate = list(index); coordinate[qa] = value
            rows[tuple(coordinate)] = out[inner]
    return torch.stack([rows[c] for c in itertools.product(*samples)]), empty / total
