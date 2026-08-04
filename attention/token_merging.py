import torch
from einops import rearrange


_HASH_MOD = 2147483647
_BLOCK_SEED_MODES = {"random", "random_with_ith"}

_LAYOUT_COLUMN_KEY_ROW = "query_column_key_row"
_LAYOUT_ROW_KEY_COLUMN = "query_row_key_column"
_LAYOUT_COLUMN_KEY_ROW_SELECT = "query_column_key_row_select"
_LAYOUT_ROW_KEY_COLUMN_SELECT = "query_row_key_column_select"
_LAYOUT_GLOBAL = "global"

_LOCAL_LAYOUTS = {
    _LAYOUT_COLUMN_KEY_ROW,
    _LAYOUT_ROW_KEY_COLUMN,
    _LAYOUT_COLUMN_KEY_ROW_SELECT,
    _LAYOUT_ROW_KEY_COLUMN_SELECT,
}
_SUPPORTED_LAYOUTS = _LOCAL_LAYOUTS | {_LAYOUT_GLOBAL}
_COLUMN_KEY_ROW_LAYOUTS = {_LAYOUT_COLUMN_KEY_ROW, _LAYOUT_COLUMN_KEY_ROW_SELECT}

_Q_INIT_ALIASES = {
    "identity": "ith",
    "i": "ith",
    "uniform": "mean",
    "avg": "mean",
    "average": "mean",
    "first": "1st",
}


def _normalize_q_init(q_init):
    q_init = (q_init or "ith").lower()
    return _Q_INIT_ALIASES.get(q_init, q_init)


def _scale_qk(q, k, scale):
    scale_sqrt = scale**0.5
    return q * scale_sqrt, k * scale_sqrt


def _validate_block_seed_mode(seed_mode, context):
    if seed_mode not in _BLOCK_SEED_MODES:
        raise ValueError(
            f"{context} currently supports seed_mode='random' or 'random_with_ith'."
        )
    return seed_mode == "random_with_ith"


def _merge_query_row(q, key_rows, q_init):
    """Choose query rows used to merge each key row's columns."""
    q_init = _normalize_q_init(q_init)

    if q_init == "ith":
        if q.size(2) != key_rows:
            raise ValueError("q_init=ith needs query_row and key_row to have the same size.")
        return q

    if q_init == "mean":
        return q.mean(dim=2, keepdim=True).expand(-1, -1, key_rows, -1, -1, -1)

    if q_init == "1st":
        return q[:, :, :1].expand(-1, -1, key_rows, -1, -1, -1)

    raise ValueError(f"unsupported q_init={q_init!r}; expected mean, 1st, or ith.")


def _merge_query_column(q, key_columns, q_init):
    """Choose query columns used to merge each key column's rows."""
    q_init = _normalize_q_init(q_init)

    if q_init == "ith":
        if q.size(3) != key_columns:
            raise ValueError("q_init=ith needs query_column and key_column to have the same size.")
        return q

    if q_init == "mean":
        return q.mean(dim=3, keepdim=True).expand(-1, -1, -1, key_columns, -1, -1)

    if q_init == "1st":
        return q[:, :, :, :1].expand(-1, -1, -1, key_columns, -1, -1)

    raise ValueError(f"unsupported q_init={q_init!r}; expected mean, 1st, or ith.")


def _video_to_blocks(x, f_tied, h_reduce, w_reduce, height, width):
    """[B, F*H*W, heads, dim] -> [B, outer, row, col, heads, dim]."""
    batch, seq_len, heads, dim = x.shape
    if height % h_reduce or width % w_reduce:
        raise ValueError("height/width must be divisible by h_reduce/w_reduce.")
    if seq_len % (height * width):
        raise ValueError("sequence length must be a multiple of height * width.")

    frames = seq_len // (height * width)
    if frames % f_tied:
        raise ValueError("number of frames must be divisible by f_tied.")

    return rearrange(
        x,
        "b (outer ft hr row wr col) head dim -> b (outer hr wr) (ft row) col head dim",
        ft=f_tied,
        hr=h_reduce,
        row=height // h_reduce,
        wr=w_reduce,
        col=width // w_reduce,
    ).contiguous()


def _blocks_to_video(x, f_tied, h_reduce, w_reduce):
    """[B, outer, row, col, heads, dim] -> [B, F*H*W, heads, dim]."""
    return rearrange(
        x,
        "b (outer hr wr) (ft row) col head dim -> b (outer ft hr row wr col) head dim",
        ft=f_tied,
        hr=h_reduce,
        wr=w_reduce,
    ).contiguous()


def _hashed_candidate_scores(base, num_choices):
    """Deterministic scores for choosing pseudo-random seed indices."""
    candidate = torch.arange(num_choices, device=base.device).view(
        *((1,) * base.ndim),
        num_choices,
    )
    scores = (
        base.unsqueeze(-1).remainder(_HASH_MOD)
        + candidate * 374761393
    ).remainder(_HASH_MOD)
    scores = (scores * (scores + 1274126177)).remainder(_HASH_MOD)
    scores = (scores * (scores + 60493)).remainder(_HASH_MOD)
    return scores * num_choices + candidate, candidate


def _seed_indices_from_base(base, num_choices, num_seeds, include_seed=None):
    """Map a per-block hash base to deterministic seed indices.

    The num_seeds==2 path preserves the original two-seed sampling rule.
    """
    if include_seed is not None:
        if num_seeds == 1:
            return include_seed.unsqueeze(-1).long()

        scores, candidate = _hashed_candidate_scores(base, num_choices)
        scores = scores.masked_fill(candidate == include_seed.unsqueeze(-1), -1)
        random_seeds = scores.topk(num_seeds - 1, dim=-1).indices.long()
        return torch.cat((include_seed.unsqueeze(-1).long(), random_seeds), dim=-1)

    if num_seeds == 2:
        seed0 = base.remainder(num_choices)
        step = (base.div(num_choices, rounding_mode="floor").remainder(num_choices - 1)) + 1
        seed1 = (seed0 + step).remainder(num_choices)
        return torch.stack((seed0, seed1), dim=-1).long()

    scores, _ = _hashed_candidate_scores(base, num_choices)
    return scores.topk(num_seeds, dim=-1).indices.long()


def _random_seed_rows(
    query_outer,
    key_outer,
    key_rows,
    query_columns,
    query_rows,
    num_seeds,
    seed,
    include_ith=False,
):
    """Deterministic pseudo-random query-row seeds for each block."""
    if num_seeds > query_rows:
        raise ValueError("num_merge_tokens cannot exceed query rows per block.")

    device = query_outer.device
    key_row = torch.arange(key_rows, device=device).view(1, 1, key_rows, 1)
    query_col = torch.arange(query_columns, device=device).view(1, 1, 1, query_columns)
    base = (
        query_outer.view(-1, 1, 1, 1) * 73856093
        + key_outer.view(1, -1, 1, 1) * 19349663
        + key_row * 83492791
        + query_col * 2654435761
        + int(seed)
    )
    ith_seed = None
    if include_ith:
        ith_seed = key_row.expand(query_outer.numel(), key_outer.numel(), key_rows, query_columns)
    return _seed_indices_from_base(base, query_rows, num_seeds, include_seed=ith_seed)


def _random_seed_columns(
    query_outer,
    key_outer,
    query_rows,
    key_columns,
    query_columns,
    num_seeds,
    seed,
    include_ith=False,
):
    """Deterministic pseudo-random query-column seeds for transposed blocks."""
    if num_seeds > query_columns:
        raise ValueError("num_merge_tokens cannot exceed query columns per block.")

    device = query_outer.device
    query_row = torch.arange(query_rows, device=device).view(1, 1, query_rows, 1)
    key_col = torch.arange(key_columns, device=device).view(1, 1, 1, key_columns)
    base = (
        query_outer.view(-1, 1, 1, 1) * 73856093
        + key_outer.view(1, -1, 1, 1) * 19349663
        + query_row * 83492791
        + key_col * 2654435761
        + int(seed)
    )
    ith_seed = None
    if include_ith:
        ith_seed = key_col.expand(query_outer.numel(), key_outer.numel(), query_rows, key_columns)
    return _seed_indices_from_base(base, query_columns, num_seeds, include_seed=ith_seed)


def _gather_seed_queries(q, seed_rows):
    # q: [B, A, I, J, H, D], seed_rows: [A, F, K, J, C].
    query_outer = torch.arange(q.size(1), device=q.device).view(-1, 1, 1, 1, 1)
    query_col = torch.arange(q.size(3), device=q.device).view(1, 1, 1, -1, 1)
    return q[:, query_outer, seed_rows, query_col]


def _gather_seed_queries_transpose(q, seed_cols):
    # q: [B, A, I, J, H, D], seed_cols: [A, F, I, L, C].
    query_outer = torch.arange(q.size(1), device=q.device).view(-1, 1, 1, 1, 1)
    query_row = torch.arange(q.size(2), device=q.device).view(1, 1, -1, 1, 1)
    return q[:, query_outer, query_row, seed_cols]


def _assign_keys_to_seeds(seed_logits):
    # seed_logits: [B, H, A, F, K, J, C, L].
    assignments = seed_logits.argmax(dim=-2)
    key_col = torch.arange(seed_logits.size(-1), device=seed_logits.device)

    for seed_idx in range(seed_logits.size(-2)):
        has_seed = (assignments == seed_idx).any(dim=-1)
        best_key = seed_logits[..., seed_idx, :].argmax(dim=-1)
        force_seed_shape = (1,) * (assignments.ndim - 1) + (-1,)
        force_seed = key_col.view(*force_seed_shape) == best_key.unsqueeze(-1)
        assignments = torch.where(
            has_seed.unsqueeze(-1),
            assignments,
            torch.where(force_seed, seed_idx, assignments),
        )

    return torch.nn.functional.one_hot(assignments, num_classes=seed_logits.size(-2)).movedim(-1, -2).bool()


def _hard_merge_weights(seed_logits, dtype):
    """Assign each key to one seed, then softmax within each hard cluster."""
    assignment_mask = _assign_keys_to_seeds(seed_logits)
    cluster_has_keys = assignment_mask.any(dim=-1)

    masked_logits = seed_logits.masked_fill(~assignment_mask, torch.finfo(seed_logits.dtype).min)
    merge_logits = masked_logits - masked_logits.amax(dim=-1, keepdim=True)
    merge_logits = merge_logits.masked_fill(~assignment_mask, torch.finfo(seed_logits.dtype).min)
    merge_weights = torch.softmax(merge_logits, dim=-1).masked_fill(~assignment_mask, 0).to(dtype)

    merge_logprob = merge_logits - torch.logsumexp(merge_logits, dim=-1, keepdim=True)
    merge_logprob = merge_logprob.masked_fill(~assignment_mask, 0)
    return merge_weights, merge_logprob, cluster_has_keys


def _soft_merge_weights(merge_logits, dtype):
    """Softmax all keys for each seed and keep the corresponding log-probs."""
    merge_logits = merge_logits.float()
    merge_logits = merge_logits - merge_logits.amax(dim=-1, keepdim=True)
    merge_weights = torch.softmax(merge_logits, dim=-1).to(dtype)
    merge_logprob = merge_logits - torch.logsumexp(merge_logits, dim=-1, keepdim=True)
    return merge_weights, merge_logprob


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
    scores = (positions * 1103515245 + int(seed) * 12345 + 12345).remainder(2147483647)
    scores = (scores * (scores + 60493)).remainder(2147483647)
    return scores.topk(num_seeds, sorted=False).indices


def _global_token_merging_attn(
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


def _one_step_query_column_key_row(q, k, v, scale, q_init):
    # Dimension letters used below:
    #   b batch, a query_outer, f key_outer, i query_row, j query_column,
    #   k key_row, l key_column, h head, d/e head_dim.
    #
    # q:   [B, A, I, J, H, D]
    # k/v: [B, F, K, L, H, D]
    # A single fixed (a, f, j, k) tile is 30 queries x 52 keys by default.
    q, k = _scale_qk(q, k, scale)

    # 1. For each key row k, use the selected query row to merge key columns l.
    selected_q = _merge_query_row(q, k.size(2), q_init)
    merge_logits = torch.einsum("bakjhd,bfklhd->bhafkjl", selected_q, k)
    merge_weights, merge_logprob = _soft_merge_weights(merge_logits, k.dtype)

    merged_keys = torch.einsum("bhafkjl,bfklhd->bafjkhd", merge_weights, k)
    merged_values = torch.einsum("bhafkjl,bfklhe->bafjkhe", merge_weights, v)

    # 2. Route each query row i to merged tokens indexed by (f, k).
    # The merge-logprob correction keeps this exactly equivalent to the old
    # one-step implementation while writing it as token merging.
    routing_bias = (merge_weights * merge_logprob).sum(dim=-1, keepdim=True)
    routing_bias = routing_bias.transpose(-2, -3)
    routing_logits = torch.einsum("bafjkhd,baijhd->bhafjki", merged_keys, q)
    routing_logits = routing_logits - routing_bias

    routing_weights = rearrange(routing_logits, "b h a f j k i -> b h a j i (f k)")
    routing_weights = torch.softmax(routing_weights.float(), dim=-1).to(q.dtype)
    routing_weights = rearrange(
        routing_weights,
        "b h a j i (f k) -> b h a f j k i",
        f=k.size(1),
        k=k.size(2),
    )

    return torch.einsum("bhafjki,bafjkhe->baijhe", routing_weights, merged_values)


def _multi_seed_query_column_key_row(
    q,
    k,
    v,
    scale,
    num_merge_tokens,
    seed_mode,
    seed,
    query_outer_offset,
):
    # q:   [B, A, I, J, H, D]
    # k/v: [B, F, K, L, H, D]
    # A fixed (a, f, j, k) block has I queries x L keys. This path merges
    # the L keys into several hard-assigned merged tokens instead of one.
    include_ith = _validate_block_seed_mode(seed_mode, "multi-token merging")
    if num_merge_tokens > k.size(3):
        raise ValueError("num_merge_tokens cannot exceed key columns per block.")

    q, k = _scale_qk(q, k, scale)

    query_outer = torch.arange(
        query_outer_offset,
        query_outer_offset + q.size(1),
        device=q.device,
    )
    key_outer = torch.arange(k.size(1), device=q.device)
    seed_rows = _random_seed_rows(
        query_outer,
        key_outer,
        k.size(2),
        q.size(3),
        q.size(2),
        num_merge_tokens,
        seed,
        include_ith=include_ith,
    )
    seed_queries = _gather_seed_queries(q, seed_rows)

    seed_logits = torch.einsum("bafkjchd,bfklhd->bhafkjcl", seed_queries, k).float()
    merge_weights, merge_logprob, cluster_has_keys = _hard_merge_weights(seed_logits, k.dtype)

    merged_keys = torch.einsum("bhafkjcl,bfklhd->bafjkchd", merge_weights, k)
    merged_values = torch.einsum("bhafkjcl,bfklhe->bafjkche", merge_weights, v)

    routing_bias = (merge_weights * merge_logprob).sum(dim=-1)
    routing_bias = rearrange(routing_bias, "b h a f k j c -> b h a f j k c 1")

    routing_logits = torch.einsum("bafjkchd,baijhd->bhafjkci", merged_keys, q)
    routing_logits = routing_logits - routing_bias
    cluster_has_keys = rearrange(cluster_has_keys, "b h a f k j c -> b h a f j k c 1")
    routing_logits = routing_logits.masked_fill(
        ~cluster_has_keys,
        torch.finfo(routing_logits.dtype).min,
    )

    routing_weights = rearrange(routing_logits, "b h a f j k c i -> b h a j i (f k c)")
    routing_weights = torch.softmax(routing_weights.float(), dim=-1).to(q.dtype)
    routing_weights = rearrange(
        routing_weights,
        "b h a j i (f k c) -> b h a f j k c i",
        f=k.size(1),
        k=k.size(2),
        c=num_merge_tokens,
    )

    return torch.einsum("bhafjkci,bafjkche->baijhe", routing_weights, merged_values)


def _select_seed_query_column_key_row(
    q,
    k,
    v,
    scale,
    num_merge_tokens,
    seed_mode,
    seed,
    query_outer_offset,
):
    # q:   [B, A, I, J, H, D]
    # k/v: [B, F, K, L, H, D]
    # Each seed compresses all L keys in a fixed (a, f, j, k) block. Each
    # query then selects the seed whose compressed distribution maximizes
    # q_i dot K_c + H(alpha_c), the softmax variational objective.
    include_ith = _validate_block_seed_mode(seed_mode, "selective token merging")

    q, k = _scale_qk(q, k, scale)

    query_outer = torch.arange(
        query_outer_offset,
        query_outer_offset + q.size(1),
        device=q.device,
    )
    key_outer = torch.arange(k.size(1), device=q.device)
    seed_rows = _random_seed_rows(
        query_outer,
        key_outer,
        k.size(2),
        q.size(3),
        q.size(2),
        num_merge_tokens,
        seed,
        include_ith=include_ith,
    )
    seed_queries = _gather_seed_queries(q, seed_rows)

    merge_logits = torch.einsum("bafkjchd,bfklhd->bhafkjcl", seed_queries, k)
    merge_weights, merge_logprob = _soft_merge_weights(merge_logits, k.dtype)

    merged_keys = torch.einsum("bhafkjcl,bfklhd->bafjkchd", merge_weights, k)
    merged_values = torch.einsum("bhafkjcl,bfklhe->bafjkche", merge_weights, v)

    entropy = -(merge_weights.float() * merge_logprob).sum(dim=-1)
    entropy = rearrange(entropy, "b h a f k j c -> b h a f j k c 1")

    candidate_scores = torch.einsum("bafjkchd,baijhd->bhafjkci", merged_keys, q).float()
    candidate_scores = candidate_scores + entropy
    selected_scores, selected_seed = candidate_scores.max(dim=-2)

    routing_weights = rearrange(selected_scores, "b h a f j k i -> b h a j i (f k)")
    routing_weights = torch.softmax(routing_weights.float(), dim=-1).to(q.dtype)
    routing_weights = rearrange(
        routing_weights,
        "b h a j i (f k) -> b h a f j k i",
        f=k.size(1),
        k=k.size(2),
    )

    selected_seed = torch.nn.functional.one_hot(
        selected_seed,
        num_classes=num_merge_tokens,
    ).movedim(-1, -2).to(q.dtype)
    routing_weights = routing_weights.unsqueeze(-2) * selected_seed

    return torch.einsum("bhafjkci,bafjkche->baijhe", routing_weights, merged_values)


def _multi_seed_query_row_key_column(
    q,
    k,
    v,
    scale,
    num_merge_tokens,
    seed_mode,
    seed,
    query_outer_offset,
):
    # Transposed multi-token path:
    # q:   [B, A, I, J, H, D]
    # k/v: [B, F, K, L, H, D]
    # A fixed (a, f, i, l) block has J queries x K keys.
    include_ith = _validate_block_seed_mode(seed_mode, "transposed multi-token merging")
    if num_merge_tokens > q.size(3):
        raise ValueError("num_merge_tokens cannot exceed query columns per block.")

    q, k = _scale_qk(q, k, scale)

    query_outer = torch.arange(
        query_outer_offset,
        query_outer_offset + q.size(1),
        device=q.device,
    )
    key_outer = torch.arange(k.size(1), device=q.device)
    seed_cols = _random_seed_columns(
        query_outer,
        key_outer,
        q.size(2),
        k.size(3),
        q.size(3),
        num_merge_tokens,
        seed,
        include_ith=include_ith,
    )
    seed_queries = _gather_seed_queries_transpose(q, seed_cols)

    seed_logits = torch.einsum("bafilchd,bfklhd->bhafilck", seed_queries, k).float()
    merge_weights, merge_logprob, cluster_has_keys = _hard_merge_weights(seed_logits, k.dtype)

    merged_keys = torch.einsum("bhafilck,bfklhd->bafilchd", merge_weights, k)
    merged_values = torch.einsum("bhafilck,bfklhe->bafilche", merge_weights, v)

    routing_bias = (merge_weights * merge_logprob).sum(dim=-1)
    routing_bias = routing_bias.unsqueeze(-1)

    routing_logits = torch.einsum("bafilchd,baijhd->bhafilcj", merged_keys, q)
    routing_logits = routing_logits - routing_bias
    routing_logits = routing_logits.masked_fill(
        ~cluster_has_keys.unsqueeze(-1),
        torch.finfo(routing_logits.dtype).min,
    )

    routing_weights = rearrange(routing_logits, "b h a f i l c j -> b h a i j (f l c)")
    routing_weights = torch.softmax(routing_weights.float(), dim=-1).to(q.dtype)
    routing_weights = rearrange(
        routing_weights,
        "b h a i j (f l c) -> b h a f i l c j",
        f=k.size(1),
        l=k.size(3),
        c=num_merge_tokens,
    )

    return torch.einsum("bhafilcj,bafilche->baijhe", routing_weights, merged_values)


def _select_seed_query_row_key_column(
    q,
    k,
    v,
    scale,
    num_merge_tokens,
    seed_mode,
    seed,
    query_outer_offset,
):
    # Transposed selective path. Each seed compresses all K key rows in a fixed
    # (a, f, i, l) block; each query column picks the best compressed KV via
    # q_j dot K_c + H(alpha_c).
    include_ith = _validate_block_seed_mode(seed_mode, "transposed selective token merging")

    q, k = _scale_qk(q, k, scale)

    query_outer = torch.arange(
        query_outer_offset,
        query_outer_offset + q.size(1),
        device=q.device,
    )
    key_outer = torch.arange(k.size(1), device=q.device)
    seed_cols = _random_seed_columns(
        query_outer,
        key_outer,
        q.size(2),
        k.size(3),
        q.size(3),
        num_merge_tokens,
        seed,
        include_ith=include_ith,
    )
    seed_queries = _gather_seed_queries_transpose(q, seed_cols)

    merge_logits = torch.einsum("bafilchd,bfklhd->bhafilck", seed_queries, k)
    merge_weights, merge_logprob = _soft_merge_weights(merge_logits, k.dtype)

    merged_keys = torch.einsum("bhafilck,bfklhd->bafilchd", merge_weights, k)
    merged_values = torch.einsum("bhafilck,bfklhe->bafilche", merge_weights, v)

    entropy = -(merge_weights.float() * merge_logprob).sum(dim=-1).unsqueeze(-1)

    candidate_scores = torch.einsum("bafilchd,baijhd->bhafilcj", merged_keys, q).float()
    candidate_scores = candidate_scores + entropy
    selected_scores, selected_seed = candidate_scores.max(dim=-2)

    routing_weights = rearrange(selected_scores, "b h a f i l j -> b h a i j (f l)")
    routing_weights = torch.softmax(routing_weights.float(), dim=-1).to(q.dtype)
    routing_weights = rearrange(
        routing_weights,
        "b h a i j (f l) -> b h a f i l j",
        f=k.size(1),
        l=k.size(3),
    )

    selected_seed = torch.nn.functional.one_hot(
        selected_seed,
        num_classes=num_merge_tokens,
    ).movedim(-1, -2).to(q.dtype)
    routing_weights = routing_weights.unsqueeze(-2) * selected_seed

    return torch.einsum("bhafilcj,bafilche->baijhe", routing_weights, merged_values)


def _one_step_query_row_key_column(q, k, v, scale, q_init):
    # Symmetric layout to _one_step_query_column_key_row:
    #   q:   [B, A, I, J, H, D]
    #   k/v: [B, F, K, L, H, D]
    # A single fixed (a, f, i, l) tile is 52 queries x 30 keys by default:
    # fixed query row i routes across query columns j, while fixed key column l
    # merges key rows k.
    q, k = _scale_qk(q, k, scale)

    # 1. For each key column l, use the selected query column to merge key rows k.
    selected_q = _merge_query_column(q, k.size(3), q_init)
    merge_logits = torch.einsum("bailhd,bfklhd->bhafilk", selected_q, k)
    merge_weights, merge_logprob = _soft_merge_weights(merge_logits, k.dtype)

    merged_keys = torch.einsum("bhafilk,bfklhd->bafilhd", merge_weights, k)
    merged_values = torch.einsum("bhafilk,bfklhe->bafilhe", merge_weights, v)

    # 2. Route each query column j to merged tokens indexed by (f, l).
    routing_bias = (merge_weights * merge_logprob).sum(dim=-1, keepdim=True)
    routing_logits = torch.einsum("bafilhd,baijhd->bhafilj", merged_keys, q)
    routing_logits = routing_logits - routing_bias

    routing_weights = rearrange(routing_logits, "b h a f i l j -> b h a i j (f l)")
    routing_weights = torch.softmax(routing_weights.float(), dim=-1).to(q.dtype)
    routing_weights = rearrange(
        routing_weights,
        "b h a i j (f l) -> b h a f i l j",
        f=k.size(1),
        l=k.size(3),
    )

    return torch.einsum("bhafilj,bafilhe->baijhe", routing_weights, merged_values)


def _local_attention_impl(layout, num_merge_tokens):
    if num_merge_tokens > 1:
        return {
            _LAYOUT_COLUMN_KEY_ROW: _multi_seed_query_column_key_row,
            _LAYOUT_COLUMN_KEY_ROW_SELECT: _select_seed_query_column_key_row,
            _LAYOUT_ROW_KEY_COLUMN: _multi_seed_query_row_key_column,
            _LAYOUT_ROW_KEY_COLUMN_SELECT: _select_seed_query_row_key_column,
        }[layout]

    if layout in _COLUMN_KEY_ROW_LAYOUTS:
        return _one_step_query_column_key_row
    return _one_step_query_row_key_column


def token_merging_attn(
    q,
    k,
    v,
    f_tied,
    h_reduce,
    w_reduce,
    h,
    w,
    sm_scale=None,
    q_init=None,
    query_outer_chunk=None,
    layout="query_column_key_row",
    num_merge_tokens=1,
    seed_mode="random",
    seed=0,
    global_seed_ratio=0.1,
    global_assign_chunk=None,
    global_query_chunk=None,
):
    """Grid-based token-merging attention for Wan video self-attention."""
    # Incoming q/k/v from Wan self-attention:
    #   q/k/v: [batch, sequence, head, dim]
    # Default 480p/81-frame Wan:
    #   sequence = frame * height * width = 21 * 30 * 52 = 32760.
    # After _to_token_merging_blocks:
    #   q/k/v: [batch, 21, 30, 52, head, dim]
    if q.size(1) != k.size(1) or k.size(1) != v.size(1):
        raise ValueError("token-merging attention expects self-attention q/k/v lengths to match.")

    if sm_scale is None:
        sm_scale = q.size(-1) ** -0.5
    query_outer_chunk = max(1, int(query_outer_chunk or 1))
    if layout not in _SUPPORTED_LAYOUTS:
        raise ValueError(
            "layout must be 'query_column_key_row', 'query_row_key_column', "
            "'query_column_key_row_select', 'query_row_key_column_select', or 'global', "
            f"got {layout!r}."
        )
    if layout == _LAYOUT_GLOBAL:
        return _global_token_merging_attn(
            q,
            k,
            v,
            sm_scale,
            global_seed_ratio,
            seed_mode,
            seed,
            global_assign_chunk,
            global_query_chunk,
        )

    num_merge_tokens = int(num_merge_tokens)
    if num_merge_tokens < 1:
        raise ValueError(f"num_merge_tokens must be >= 1, got {num_merge_tokens!r}.")
    if num_merge_tokens > 1 and layout not in _LOCAL_LAYOUTS:
        raise ValueError(
            "multi-token merging is currently implemented only for query/key row/column layouts."
        )

    q_blocks = _video_to_blocks(q, f_tied, h_reduce, w_reduce, h, w)
    k_blocks = _video_to_blocks(k, f_tied, h_reduce, w_reduce, h, w)
    v_blocks = _video_to_blocks(v, f_tied, h_reduce, w_reduce, h, w)
    one_step = _local_attention_impl(layout, num_merge_tokens)

    out = torch.empty_like(v_blocks)
    for start in range(0, q_blocks.size(1), query_outer_chunk):
        end = min(start + query_outer_chunk, q_blocks.size(1))
        if num_merge_tokens > 1:
            out[:, start:end] = one_step(
                q_blocks[:, start:end],
                k_blocks,
                v_blocks,
                sm_scale,
                num_merge_tokens,
                seed_mode,
                seed,
                start,
            )
        else:
            out[:, start:end] = one_step(
                q_blocks[:, start:end],
                k_blocks,
                v_blocks,
                sm_scale,
                q_init,
            )

    return _blocks_to_video(out, f_tied, h_reduce, w_reduce)


__all__ = ["token_merging_attn"]
