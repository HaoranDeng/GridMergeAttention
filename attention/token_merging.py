import torch
from einops import rearrange


_Q_INIT_ALIASES = {
    "identity": "ith",
    "i": "ith",
    "uniform": "mean",
    "avg": "mean",
    "average": "mean",
    "first": "1st",
}


def _merge_query_row(q, key_rows, q_init):
    """Choose query rows used to merge each key row's columns."""
    q_init = (q_init or "ith").lower()
    q_init = _Q_INIT_ALIASES.get(q_init, q_init)

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
    q_init = (q_init or "ith").lower()
    q_init = _Q_INIT_ALIASES.get(q_init, q_init)

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


def _random_seed_rows(
    query_outer,
    key_outer,
    key_rows,
    query_columns,
    query_rows,
    num_seeds,
    seed,
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

    if num_seeds == 2:
        row0 = base.remainder(query_rows)
        step = (base.div(query_rows, rounding_mode="floor").remainder(query_rows - 1)) + 1
        row1 = (row0 + step).remainder(query_rows)
        return torch.stack((row0, row1), dim=-1).long()

    query_row = torch.arange(query_rows, device=device).view(1, 1, 1, 1, query_rows)
    scores = (
        base.unsqueeze(-1).remainder(2147483647)
        + query_row * 374761393
    ).remainder(2147483647)
    scores = (scores * (scores + 1274126177)).remainder(2147483647)
    scores = (scores * (scores + 60493)).remainder(2147483647)
    scores = scores * query_rows + query_row
    return scores.topk(num_seeds, dim=-1).indices.long()


def _gather_seed_queries(q, seed_rows):
    # q: [B, A, I, J, H, D], seed_rows: [A, F, K, J, C].
    query_outer = torch.arange(q.size(1), device=q.device).view(-1, 1, 1, 1, 1)
    query_col = torch.arange(q.size(3), device=q.device).view(1, 1, 1, -1, 1)
    return q[:, query_outer, seed_rows, query_col]


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


def _one_step_query_column_key_row(q, k, v, scale, q_init):
    # Dimension letters used below:
    #   b batch, a query_outer, f key_outer, i query_row, j query_column,
    #   k key_row, l key_column, h head, d/e head_dim.
    #
    # q:   [B, A, I, J, H, D]
    # k/v: [B, F, K, L, H, D]
    # A single fixed (a, f, j, k) tile is 30 queries x 52 keys by default.
    scale_sqrt = scale**0.5
    q = q * scale_sqrt
    k = k * scale_sqrt

    # 1. For each key row k, use the selected query row to merge key columns l.
    selected_q = _merge_query_row(q, k.size(2), q_init)
    merge_logits = torch.einsum("bakjhd,bfklhd->bhafkjl", selected_q, k)
    merge_logits = merge_logits.float()
    merge_logits = merge_logits - merge_logits.amax(dim=-1, keepdim=True)
    merge_weights = torch.softmax(merge_logits, dim=-1).to(k.dtype)

    merged_keys = torch.einsum("bhafkjl,bfklhd->bafjkhd", merge_weights, k)
    merged_values = torch.einsum("bhafkjl,bfklhe->bafjkhe", merge_weights, v)

    # 2. Route each query row i to merged tokens indexed by (f, k).
    # The merge-logprob correction keeps this exactly equivalent to the old
    # one-step implementation while writing it as token merging.
    merge_logprob = merge_logits - torch.logsumexp(merge_logits, dim=-1, keepdim=True)
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
    if seed_mode != "random":
        raise ValueError("multi-token merging currently supports seed_mode='random' only.")
    if num_merge_tokens > k.size(3):
        raise ValueError("num_merge_tokens cannot exceed key columns per block.")

    scale_sqrt = scale**0.5
    q = q * scale_sqrt
    k = k * scale_sqrt

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
    )
    seed_queries = _gather_seed_queries(q, seed_rows)

    seed_logits = torch.einsum("bafkjchd,bfklhd->bhafkjcl", seed_queries, k)
    seed_logits = seed_logits.float()
    assignment_mask = _assign_keys_to_seeds(seed_logits)
    cluster_has_keys = assignment_mask.any(dim=-1)

    masked_logits = seed_logits.masked_fill(~assignment_mask, torch.finfo(seed_logits.dtype).min)
    merge_logits = masked_logits - masked_logits.amax(dim=-1, keepdim=True)
    merge_logits = merge_logits.masked_fill(~assignment_mask, torch.finfo(seed_logits.dtype).min)
    merge_weights = torch.softmax(merge_logits, dim=-1).masked_fill(~assignment_mask, 0).to(k.dtype)

    merged_keys = torch.einsum("bhafkjcl,bfklhd->bafjkchd", merge_weights, k)
    merged_values = torch.einsum("bhafkjcl,bfklhe->bafjkche", merge_weights, v)

    merge_logprob = merge_logits - torch.logsumexp(merge_logits, dim=-1, keepdim=True)
    merge_logprob = merge_logprob.masked_fill(~assignment_mask, 0)
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


def _one_step_query_row_key_column(q, k, v, scale, q_init):
    # Symmetric layout to _one_step_query_column_key_row:
    #   q:   [B, A, I, J, H, D]
    #   k/v: [B, F, K, L, H, D]
    # A single fixed (a, f, i, l) tile is 52 queries x 30 keys by default:
    # fixed query row i routes across query columns j, while fixed key column l
    # merges key rows k.
    scale_sqrt = scale**0.5
    q = q * scale_sqrt
    k = k * scale_sqrt

    # 1. For each key column l, use the selected query column to merge key rows k.
    selected_q = _merge_query_column(q, k.size(3), q_init)
    merge_logits = torch.einsum("bailhd,bfklhd->bhafilk", selected_q, k)
    merge_logits = merge_logits.float()
    merge_logits = merge_logits - merge_logits.amax(dim=-1, keepdim=True)
    merge_weights = torch.softmax(merge_logits, dim=-1).to(k.dtype)

    merged_keys = torch.einsum("bhafilk,bfklhd->bafilhd", merge_weights, k)
    merged_values = torch.einsum("bhafilk,bfklhe->bafilhe", merge_weights, v)

    # 2. Route each query column j to merged tokens indexed by (f, l).
    merge_logprob = merge_logits - torch.logsumexp(merge_logits, dim=-1, keepdim=True)
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
):
    """One-step token-merging attention for Wan video tokens."""
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
    if layout not in {"query_column_key_row", "query_row_key_column"}:
        raise ValueError(
            "layout must be 'query_column_key_row' or 'query_row_key_column', "
            f"got {layout!r}."
        )
    num_merge_tokens = int(num_merge_tokens)
    if num_merge_tokens < 1:
        raise ValueError(f"num_merge_tokens must be >= 1, got {num_merge_tokens!r}.")
    if num_merge_tokens > 1 and layout != "query_column_key_row":
        raise ValueError(
            "multi-token merging is currently implemented only for "
            "layout='query_column_key_row'."
        )

    q_blocks = _video_to_blocks(q, f_tied, h_reduce, w_reduce, h, w)
    k_blocks = _video_to_blocks(k, f_tied, h_reduce, w_reduce, h, w)
    v_blocks = _video_to_blocks(v, f_tied, h_reduce, w_reduce, h, w)
    if num_merge_tokens > 1:
        one_step = _multi_seed_query_column_key_row
    else:
        one_step = (
            _one_step_query_column_key_row
            if layout == "query_column_key_row"
            else _one_step_query_row_key_column
        )

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
