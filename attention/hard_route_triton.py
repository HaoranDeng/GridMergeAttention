import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised only when Triton is missing.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _hard_route_merge_kernel(
        seed_ptr,
        key_ptr,
        value_ptr,
        merged_key_ptr,
        merged_value_ptr,
        routing_bias_ptr,
        valid_ptr,
        A: tl.constexpr,
        F: tl.constexpr,
        K_ROWS: tl.constexpr,
        J: tl.constexpr,
        C_SEEDS: tl.constexpr,
        H: tl.constexpr,
        L: tl.constexpr,
        D_HEAD: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_J: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        group_id = tl.program_id(0)
        query_col_block = tl.program_id(1)

        head = group_id % H
        tmp = group_id // H
        key_row = tmp % K_ROWS
        tmp = tmp // K_ROWS
        key_outer = tmp % F
        tmp = tmp // F
        query_outer = tmp % A
        batch = tmp // A

        query_offsets = query_col_block * BLOCK_J + tl.arange(0, BLOCK_J)
        key_offsets = tl.arange(0, BLOCK_L)
        dim_offsets = tl.arange(0, BLOCK_D)
        valid_query = query_offsets < J
        valid_key = key_offsets < L
        valid_dim = dim_offsets < D_HEAD

        key_base = key_ptr + (((batch * F + key_outer) * K_ROWS + key_row) * L * H + head) * D_HEAD
        key_block = tl.load(
            key_base + key_offsets[:, None] * H * D_HEAD + dim_offsets[None, :],
            mask=valid_key[:, None] & valid_dim[None, :],
            other=0.0,
        )
        value_base = value_ptr + (((batch * F + key_outer) * K_ROWS + key_row) * L * H + head) * D_HEAD
        value_block = tl.load(
            value_base + key_offsets[:, None] * H * D_HEAD + dim_offsets[None, :],
            mask=valid_key[:, None] & valid_dim[None, :],
            other=0.0,
        )

        best_logits = tl.full((BLOCK_J, BLOCK_L), -float("inf"), tl.float32)
        assignments = tl.zeros((BLOCK_J, BLOCK_L), tl.int32)
        for seed_idx in tl.static_range(0, BLOCK_C):
            seed_offsets = (
                (((((batch * A + query_outer) * F + key_outer) * K_ROWS + key_row)
                  * J + query_offsets[:, None]) * C_SEEDS + seed_idx)
                * H + head
            ) * D_HEAD + dim_offsets[None, :]
            seed_vec = tl.load(
                seed_ptr + seed_offsets,
                mask=(seed_idx < C_SEEDS) & valid_query[:, None] & valid_dim[None, :],
                other=0.0,
            )
            logits = tl.dot(seed_vec, tl.trans(key_block), input_precision="ieee")
            logits = tl.where(
                (seed_idx < C_SEEDS) & valid_query[:, None] & valid_key[None, :],
                logits,
                -float("inf"),
            )
            update = logits > best_logits
            best_logits = tl.where(update, logits, best_logits)
            assignments = tl.where(update, seed_idx, assignments)

        # Preserve the Python reference behavior: if a seed owns no key after
        # argmax assignment, force its best key to that seed.
        for seed_idx in tl.static_range(0, BLOCK_C):
            seed_offsets = (
                (((((batch * A + query_outer) * F + key_outer) * K_ROWS + key_row)
                  * J + query_offsets[:, None]) * C_SEEDS + seed_idx)
                * H + head
            ) * D_HEAD + dim_offsets[None, :]
            seed_vec = tl.load(
                seed_ptr + seed_offsets,
                mask=(seed_idx < C_SEEDS) & valid_query[:, None] & valid_dim[None, :],
                other=0.0,
            )
            logits = tl.dot(seed_vec, tl.trans(key_block), input_precision="ieee")
            active_seed = seed_idx < C_SEEDS
            active = active_seed & valid_query[:, None] & valid_key[None, :]
            has_seed = tl.sum(tl.where((assignments == seed_idx) & active, 1, 0), axis=1) > 0
            best_logit = tl.max(tl.where(active, logits, -float("inf")), axis=1)
            force_key = active & (~has_seed[:, None]) & (logits == best_logit[:, None])
            assignments = tl.where(force_key, seed_idx, assignments)

        for seed_idx in tl.static_range(0, BLOCK_C):
            seed_offsets = (
                (((((batch * A + query_outer) * F + key_outer) * K_ROWS + key_row)
                  * J + query_offsets[:, None]) * C_SEEDS + seed_idx)
                * H + head
            ) * D_HEAD + dim_offsets[None, :]
            seed_vec = tl.load(
                seed_ptr + seed_offsets,
                mask=(seed_idx < C_SEEDS) & valid_query[:, None] & valid_dim[None, :],
                other=0.0,
            )
            logits = tl.dot(seed_vec, tl.trans(key_block), input_precision="ieee")
            active = (
                (seed_idx < C_SEEDS)
                & valid_query[:, None]
                & valid_key[None, :]
                & (assignments == seed_idx)
            )
            has_keys = tl.sum(tl.where(active, 1, 0), axis=1) > 0

            row_max = tl.max(tl.where(active, logits, -float("inf")), axis=1)
            probs = tl.where(active, tl.exp(logits - row_max[:, None]), 0.0)
            denom = tl.sum(probs, axis=1)
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            weights = probs / safe_denom[:, None]
            log_denom = row_max + tl.log(safe_denom)

            merged_key = tl.dot(weights.to(key_block.dtype), key_block)
            merged_value = tl.dot(weights.to(value_block.dtype), value_block)
            routing_bias = tl.sum(
                tl.where(active, weights * (logits - log_denom[:, None]), 0.0),
                axis=1,
            )
            routing_bias = tl.where(has_keys, routing_bias, 0.0)

            merged_offset = (
                ((((((batch * A + query_outer) * F + key_outer) * J + query_offsets[:, None])
                   * K_ROWS + key_row) * C_SEEDS + seed_idx) * H + head)
                * D_HEAD + dim_offsets[None, :]
            )
            tl.store(
                merged_key_ptr + merged_offset,
                merged_key,
                mask=(seed_idx < C_SEEDS) & valid_query[:, None] & valid_dim[None, :],
            )
            tl.store(
                merged_value_ptr + merged_offset,
                merged_value,
                mask=(seed_idx < C_SEEDS) & valid_query[:, None] & valid_dim[None, :],
            )

            bias_offset = (
                ((((((batch * H + head) * A + query_outer) * F + key_outer)
                   * K_ROWS + key_row) * J + query_offsets) * C_SEEDS + seed_idx)
            )
            tl.store(
                routing_bias_ptr + bias_offset,
                routing_bias,
                mask=(seed_idx < C_SEEDS) & valid_query,
            )
            tl.store(
                valid_ptr + bias_offset,
                has_keys,
                mask=(seed_idx < C_SEEDS) & valid_query,
            )

    @triton.jit
    def _biased_attention_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        bias_ptr,
        out_ptr,
        Q_LEN: tl.constexpr,
        K_LEN: tl.constexpr,
        D_HEAD: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        group_id = tl.program_id(0)
        query_block_id = tl.program_id(1)

        query_offsets = query_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        key_offsets = tl.arange(0, BLOCK_N)
        dim_offsets = tl.arange(0, BLOCK_D)

        query_base = query_ptr + group_id * Q_LEN * D_HEAD
        key_base = key_ptr + group_id * K_LEN * D_HEAD
        value_base = value_ptr + group_id * K_LEN * D_HEAD
        bias_base = bias_ptr + group_id * K_LEN
        out_base = out_ptr + group_id * Q_LEN * D_HEAD

        q = tl.load(
            query_base + query_offsets[:, None] * D_HEAD + dim_offsets[None, :],
            mask=(query_offsets[:, None] < Q_LEN) & (dim_offsets[None, :] < D_HEAD),
            other=0.0,
        )

        max_score = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        normalizer = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for key_start in range(0, K_LEN, BLOCK_N):
            keys = key_start + key_offsets
            k = tl.load(
                key_base + keys[:, None] * D_HEAD + dim_offsets[None, :],
                mask=(keys[:, None] < K_LEN) & (dim_offsets[None, :] < D_HEAD),
                other=0.0,
            )
            v = tl.load(
                value_base + keys[:, None] * D_HEAD + dim_offsets[None, :],
                mask=(keys[:, None] < K_LEN) & (dim_offsets[None, :] < D_HEAD),
                other=0.0,
            )
            bias = tl.load(bias_base + keys, mask=keys < K_LEN, other=float("inf"))

            logits = tl.dot(q, tl.trans(k)) - bias[None, :]
            logits = tl.where(
                (query_offsets[:, None] < Q_LEN) & (keys[None, :] < K_LEN),
                logits,
                -float("inf"),
            )

            block_max = tl.max(logits, axis=1)
            next_max = tl.maximum(max_score, block_max)
            p = tl.exp(logits - next_max[:, None])
            alpha = tl.exp(max_score - next_max)

            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            normalizer = normalizer * alpha + tl.sum(p, axis=1)
            max_score = next_max

        out = acc / normalizer[:, None]
        tl.store(
            out_base + query_offsets[:, None] * D_HEAD + dim_offsets[None, :],
            out,
            mask=(query_offsets[:, None] < Q_LEN) & (dim_offsets[None, :] < D_HEAD),
        )


def triton_biased_attention(query, key, value, bias, block_m=16, block_n=128):
    """Fused Triton attention for hard-route compressed KV routing.

    query: [groups, query_tokens, dim]
    key/value: [groups, compressed_tokens, dim]
    bias: [groups, compressed_tokens]

    The computed logits are query @ key.T - bias, matching biased_sdpa.
    """
    if triton is None:
        raise RuntimeError("hard_route_backend='triton' requires Triton.")
    if not query.is_cuda:
        raise RuntimeError("hard_route_backend='triton' requires CUDA/ROCm tensors.")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("hard_route_backend='triton' requires fp16 or bf16 query tensors.")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    bias = bias.contiguous()

    groups, query_len, dim = query.shape
    _, key_len, key_dim = key.shape
    if key_dim != dim or value.shape != key.shape or bias.shape != (groups, key_len):
        raise ValueError("query/key/value/bias shapes are inconsistent for Triton hard-route attention.")

    block_d = triton.next_power_of_2(dim)
    if block_d > 128:
        raise ValueError(f"Triton hard-route attention supports head dim <= 128, got {dim}.")

    out = torch.empty_like(query)
    grid = (groups, triton.cdiv(query_len, block_m))
    _biased_attention_kernel[grid](
        query,
        key,
        value,
        bias,
        out,
        Q_LEN=query_len,
        K_LEN=key_len,
        D_HEAD=dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out


def triton_hard_route_merge(seed_queries, key, value):
    """Fuse hard assignment, per-cluster softmax, KV merge, and routing bias.

    seed_queries: [B, A, F, K, J, C, H, D]
    key/value:    [B, F, K, L, H, D]

    Returns:
        merged_key/value: [B, A, F, J, K, C, H, D]
        routing_bias:     [B, H, A, F, K, J, C]
        cluster_has_keys: [B, H, A, F, K, J, C]
    """
    if triton is None:
        raise RuntimeError("hard_route_backend='triton' requires Triton.")
    if not seed_queries.is_cuda:
        raise RuntimeError("hard_route_backend='triton' requires CUDA/ROCm tensors.")
    if seed_queries.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("hard_route_backend='triton' requires fp16 or bf16 tensors.")

    seed_queries = seed_queries.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    batch, query_outer, key_outer, key_rows, query_columns, num_seeds, heads, dim = seed_queries.shape
    if key.shape[:3] != (batch, key_outer, key_rows) or value.shape != key.shape:
        raise ValueError("seed_queries/key/value shapes are inconsistent for Triton hard-route merge.")
    key_columns = key.size(3)
    if key.size(4) != heads or key.size(5) != dim:
        raise ValueError("seed_queries/key head dimensions are inconsistent for Triton hard-route merge.")

    block_c = triton.next_power_of_2(num_seeds)
    target_block_j = 16 if key_columns <= 32 else 8
    block_j = min(target_block_j, triton.next_power_of_2(query_columns))
    block_l = triton.next_power_of_2(key_columns)
    block_d = triton.next_power_of_2(dim)
    if block_c > 16:
        raise ValueError(f"Triton hard-route merge supports at most 16 seeds, got {num_seeds}.")
    if block_l > 128:
        raise ValueError(f"Triton hard-route merge supports at most 128 key tokens per block, got {key_columns}.")
    if block_d > 128:
        raise ValueError(f"Triton hard-route merge supports head dim <= 128, got {dim}.")

    merged_key = torch.empty(
        (batch, query_outer, key_outer, query_columns, key_rows, num_seeds, heads, dim),
        device=key.device,
        dtype=key.dtype,
    )
    merged_value = torch.empty_like(merged_key)
    routing_bias = torch.empty(
        (batch, heads, query_outer, key_outer, key_rows, query_columns, num_seeds),
        device=key.device,
        dtype=torch.float32,
    )
    cluster_has_keys = torch.empty(
        (batch, heads, query_outer, key_outer, key_rows, query_columns, num_seeds),
        device=key.device,
        dtype=torch.bool,
    )

    grid = (
        batch * query_outer * key_outer * key_rows * heads,
        triton.cdiv(query_columns, block_j),
    )
    _hard_route_merge_kernel[grid](
        seed_queries,
        key,
        value,
        merged_key,
        merged_value,
        routing_bias,
        cluster_has_keys,
        A=query_outer,
        F=key_outer,
        K_ROWS=key_rows,
        J=query_columns,
        C_SEEDS=num_seeds,
        H=heads,
        L=key_columns,
        D_HEAD=dim,
        BLOCK_C=block_c,
        BLOCK_J=block_j,
        BLOCK_L=block_l,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return merged_key, merged_value, routing_bias, cluster_has_keys


__all__ = ["triton_biased_attention", "triton_hard_route_merge"]
