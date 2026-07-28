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

    q_blocks = _video_to_blocks(q, f_tied, h_reduce, w_reduce, h, w)
    k_blocks = _video_to_blocks(k, f_tied, h_reduce, w_reduce, h, w)
    v_blocks = _video_to_blocks(v, f_tied, h_reduce, w_reduce, h, w)
    one_step = (
        _one_step_query_column_key_row
        if layout == "query_column_key_row"
        else _one_step_query_row_key_column
    )

    out = torch.empty_like(v_blocks)
    for start in range(0, q_blocks.size(1), query_outer_chunk):
        end = min(start + query_outer_chunk, q_blocks.size(1))
        out[:, start:end] = one_step(
            q_blocks[:, start:end],
            k_blocks,
            v_blocks,
            sm_scale,
            q_init,
        )

    return _blocks_to_video(out, f_tied, h_reduce, w_reduce)


__all__ = ["token_merging_attn"]
