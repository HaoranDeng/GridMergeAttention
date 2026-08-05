import torch

from .common import HASH_MOD


def _global_seed_indices(seq_len, seed_ratio, seed_mode, seed, device):
    """Choose global query positions used as key/value merge anchors."""
    seed_ratio = float(seed_ratio)
    if not 0 < seed_ratio <= 1:
        raise ValueError(f"global_seed_ratio must be in (0, 1], got {seed_ratio!r}.")

    num_seeds = max(1, min(seq_len, int(round(seq_len * seed_ratio))))
    positions = torch.arange(seq_len, device=device, dtype=torch.long)
    seed_mode = (seed_mode or "random").lower()

    if seed_mode == "uniform":
        if num_seeds == 1:
            return positions[:1]
        return torch.linspace(0, seq_len - 1, num_seeds, device=device).round().long()

    if seed_mode != "random":
        raise ValueError("global token merging supports seed_mode='random' or 'uniform'.")

    # Deterministic hash scores avoid constructing a CPU generator or relying on
    # global RNG state. Top-k positions are the sampled query seeds.
    scores = (positions * 1103515245 + int(seed) * 12345 + 12345).remainder(HASH_MOD)
    scores = (scores * (scores + 60493)).remainder(HASH_MOD)
    return scores.topk(num_seeds, sorted=False).indices


def global_attention(
    q,
    k,
    v,
    scale,
    seed_ratio,
    seed_mode,
    seed,
    assign_chunk,
    query_chunk,
):
    # q/k/v: [B, N, H, D]. Pick M ~= seed_ratio * N query tokens as global
    # seeds, assign every key to one seed, merge K/V per seed, then let every
    # query attend to those M merged tokens.
    batch, seq_len, heads, dim = q.shape
    scale_sqrt = scale**0.5
    q = (q * scale_sqrt).permute(0, 2, 1, 3).contiguous()
    k = (k * scale_sqrt).permute(0, 2, 1, 3).contiguous()
    v = v.permute(0, 2, 1, 3).contiguous()

    seed_idx = _global_seed_indices(seq_len, seed_ratio, seed_mode, seed, q.device)
    num_seeds = seed_idx.numel()
    seed_q = q[:, :, seed_idx]
    assign_chunk = max(1, int(assign_chunk or 512))
    query_chunk = max(1, int(query_chunk or 512))

    assignments = torch.empty(batch, heads, seq_len, dtype=torch.long, device=q.device)
    assigned_scores = torch.empty(batch, heads, seq_len, dtype=torch.float32, device=q.device)

    for start in range(0, seq_len, assign_chunk):
        end = min(start + assign_chunk, seq_len)
        logits = torch.einsum("bhmd,bhkd->bhmk", seed_q, k[:, :, start:end]).float()
        scores, indices = logits.max(dim=2)
        assignments[:, :, start:end] = indices
        assigned_scores[:, :, start:end] = scores

    # Keep the number of merged tokens fixed: each sampled seed owns at least
    # its matching key position, matching the non-empty-cluster behavior of the
    # block multi-seed path.
    forced_scores = (seed_q * k[:, :, seed_idx]).sum(dim=-1).float()
    seed_targets = seed_idx.view(1, 1, num_seeds).expand(batch, heads, num_seeds)
    seed_labels = torch.arange(num_seeds, device=q.device).view(1, 1, num_seeds)
    assignments.scatter_(2, seed_targets, seed_labels.expand_as(seed_targets))
    assigned_scores.scatter_(2, seed_targets, forced_scores)

    max_scores = torch.full(
        (batch, heads, num_seeds),
        torch.finfo(torch.float32).min,
        device=q.device,
    )
    max_scores.scatter_reduce_(2, assignments, assigned_scores, reduce="amax", include_self=True)
    shifted_scores = assigned_scores - max_scores.gather(2, assignments)
    exp_scores = shifted_scores.exp()

    denom = torch.zeros(batch, heads, num_seeds, dtype=torch.float32, device=q.device)
    denom.scatter_add_(2, assignments, exp_scores)
    denom = denom.clamp_min(torch.finfo(torch.float32).tiny)

    merged_k_num = torch.zeros(batch, heads, num_seeds, dim, dtype=torch.float32, device=q.device)
    merged_v_num = torch.zeros_like(merged_k_num)
    expanded_assignments = assignments.unsqueeze(-1).expand(-1, -1, -1, dim)
    merge_weights = exp_scores.unsqueeze(-1)
    merged_k_num.scatter_add_(2, expanded_assignments, merge_weights * k.float())
    merged_v_num.scatter_add_(2, expanded_assignments, merge_weights * v.float())

    merged_k = (merged_k_num / denom.unsqueeze(-1)).to(q.dtype)
    merged_v = (merged_v_num / denom.unsqueeze(-1)).to(v.dtype)

    merge_logprob = shifted_scores - denom.gather(2, assignments).log()
    routing_bias = torch.zeros(batch, heads, num_seeds, dtype=torch.float32, device=q.device)
    routing_bias.scatter_add_(2, assignments, exp_scores * merge_logprob)
    routing_bias = routing_bias / denom

    out = torch.empty_like(v)
    for start in range(0, seq_len, query_chunk):
        end = min(start + query_chunk, seq_len)
        routing_logits = torch.einsum("bhqd,bhmd->bhqm", q[:, :, start:end], merged_k)
        routing_logits = routing_logits.float() - routing_bias.unsqueeze(2)
        routing_weights = torch.softmax(routing_logits, dim=-1).to(v.dtype)
        out[:, :, start:end] = torch.einsum("bhqm,bhmd->bhqd", routing_weights, merged_v)

    return out.permute(0, 2, 1, 3).contiguous()


__all__ = ["global_attention"]
