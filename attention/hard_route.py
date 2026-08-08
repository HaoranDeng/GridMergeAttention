from dataclasses import dataclass

import torch
from einops import rearrange

from .common import (
    biased_sdpa,
    gather_seed_queries,
    seed_rows_by_mode,
    validate_block_seed_mode,
)
from .hard_route_triton import (
    triton_biased_attention,
    triton_direct_route_biased_attention,
    triton_hard_route_merge,
    triton_hard_route_merge_c4,
    triton_hard_route_merge_c4_direct_route,
)


_SEED_ROWS_CACHE = {}
_DIRECT_ROUTE_C4_BASE = "triton_c4_direct_route_j32_w2_exp2merge_forcekeys_fp16kvbias_s1_fastdot"
_DIRECT_ROUTE_ATTENTION_CONFIGS = {
    # backend name -> (BLOCK_M, BLOCK_N, num_warps, num_stages)
    "triton_direct_m16_n16_w1_s1_exp2": (16, 16, 1, 1),
    "triton_direct_m16_n32_w1_s1_exp2": (16, 32, 1, 1),
    "triton_direct_m16_n64_w1_s1_exp2": (16, 64, 1, 1),
    "triton_direct_m16_n64_w2_s1_exp2": (16, 64, 2, 1),
    "triton_direct_m32_n16_w1_s1_exp2": (32, 16, 1, 1),
    "triton_direct_m32_n16_w1_s2_exp2": (32, 16, 1, 2),
    "triton_direct_m32_n16_w1_s3_exp2": (32, 16, 1, 3),
    "triton_direct_m32_n32_w1_s1_exp2": (32, 32, 1, 1),
    "triton_direct_m32_n32_w1_s2_exp2": (32, 32, 1, 2),
    "triton_direct_m32_n32_w1_s3_exp2": (32, 32, 1, 3),
    "triton_direct_m32_n64_w1_s1_exp2": (32, 64, 1, 1),
    "triton_direct_m32_n64_w2_s1_exp2": (32, 64, 2, 1),
}
_DIRECT_ROUTE_KEY_OUTER_BLOCKS = {
    "f2": 2,
    "f3": 3,
    "f4": 4,
    "f5": 5,
    "f6": 6,
    "f7": 7,
}


@dataclass(frozen=True)
class DirectRouteMergeConfig:
    supported: bool
    inline_local_seed: bool = False
    key_outer_block: int = 1
    use_assigned_logit: bool = False
    use_bf16_merged_kv: bool = False


def _remove_suffix(text, suffix):
    if text.endswith(suffix):
        return text[: -len(suffix)], True
    return text, False


def uses_inline_local_seed_backend(backend):
    backend = str(backend or "").lower()
    return backend.endswith("_inline_local_seed")


def _parse_direct_route_backend(backend):
    """Parse compact backend suffixes used by the direct Triton C4 fast path."""
    backend = (backend or "").lower()
    backend, inline_local_seed = _remove_suffix(backend, "_inline_local_seed")
    backend, token_major_output = _remove_suffix(backend, "_tokenmajor")

    key_outer_block = 1
    use_assigned_logit = False
    use_bf16_merged_kv = False
    changed = True
    while changed:
        changed = False
        for suffix, block in _DIRECT_ROUTE_KEY_OUTER_BLOCKS.items():
            token = f"_{suffix}"
            backend, matched = _remove_suffix(backend, token)
            if matched:
                key_outer_block = block
                changed = True
        backend, matched = _remove_suffix(backend, "_bestlogit")
        if matched:
            use_assigned_logit = True
            changed = True
        backend, matched = _remove_suffix(backend, "_bf16kv")
        if matched:
            use_bf16_merged_kv = True
            changed = True

    return DirectRouteMergeConfig(
        supported=token_major_output and backend == _DIRECT_ROUTE_C4_BASE,
        inline_local_seed=inline_local_seed,
        key_outer_block=key_outer_block,
        use_assigned_logit=use_assigned_logit,
        use_bf16_merged_kv=use_bf16_merged_kv,
    )


def _cached_seed_rows(
    device,
    query_outer_offset,
    query_outer_count,
    key_outer_count,
    key_rows,
    query_columns,
    query_rows,
    num_seeds,
    seed,
    seed_mode,
):
    cache_key = (
        str(device),
        int(query_outer_offset),
        int(query_outer_count),
        int(key_outer_count),
        int(key_rows),
        int(query_columns),
        int(query_rows),
        int(num_seeds),
        int(seed),
        str(seed_mode),
    )
    seed_rows = _SEED_ROWS_CACHE.get(cache_key)
    if seed_rows is not None:
        return seed_rows

    query_outer = torch.arange(
        query_outer_offset,
        query_outer_offset + query_outer_count,
        device=device,
    )
    key_outer = torch.arange(key_outer_count, device=device)
    seed_rows = seed_rows_by_mode(
        query_outer,
        key_outer,
        key_rows,
        query_columns,
        query_rows,
        num_seeds,
        seed,
        seed_mode,
    )
    _SEED_ROWS_CACHE[cache_key] = seed_rows
    return seed_rows


def _build_seed_rows(
    q,
    k,
    query_outer_offset,
    num_merge_tokens,
    seed,
    seed_mode,
):
    return _cached_seed_rows(
        q.device,
        query_outer_offset,
        q.size(1),
        k.size(1),
        k.size(2),
        q.size(3),
        q.size(2),
        num_merge_tokens,
        seed,
        seed_mode,
    )


def assign_keys_to_seeds(seed_logits):
    """Return a boolean [B, H, A, F, K, J, C, L] key-to-seed assignment mask."""
    num_seeds = seed_logits.size(-2)
    num_keys = seed_logits.size(-1)

    best_seed = seed_logits.argmax(dim=-2)
    assigned = torch.nn.functional.one_hot(
        best_seed,
        num_classes=num_seeds,
    ).movedim(-1, -2).bool()

    has_key = assigned.any(dim=-1)
    key_idx = torch.arange(num_keys, device=seed_logits.device)
    key_idx = key_idx.view(*((1,) * has_key.ndim), num_keys)

    # Empty clusters are rare. Keep every seed alive by routing key 0 to any
    # seed that received no key; key 0 can be shared only in this fallback.
    force_first_key = (~has_key).unsqueeze(-1) & (key_idx == 0)
    return assigned | force_first_key


def _log_softmax_over_assigned_keys(logits, assigned):
    masked_logits = logits.masked_fill(~assigned, torch.finfo(logits.dtype).min)
    logprob = torch.log_softmax(masked_logits, dim=-1)
    return logprob.masked_fill(~assigned, 0)


def hard_route_merge_weights(seed_logits, dtype):
    """Assign each key to one seed, then softmax within each routed cluster."""
    assigned = assign_keys_to_seeds(seed_logits)
    merge_logprob = _log_softmax_over_assigned_keys(seed_logits, assigned)
    merge_weights = merge_logprob.exp().masked_fill(~assigned, 0).to(dtype)
    cluster_has_keys = assigned.any(dim=-1)
    return merge_weights, merge_logprob, cluster_has_keys


def _torch_hard_route_merge(seed_queries, k, v, scale):
    # seed_queries: [B, A, F, K, J, C, H, D]
    # k/v:          [B, F, K, L, H, D]
    # seed_logits:  [B, H, A, F, K, J, C, L]
    seed_logits = torch.einsum("bafkjchd,bfklhd->bhafkjcl", seed_queries, k)
    seed_logits = seed_logits.float() * float(scale)
    merge_weights, merge_logprob, cluster_has_keys = hard_route_merge_weights(seed_logits, k.dtype)

    # merged_*:    [B, A, F, J, K, C, H, D]
    # routing_bias: [B, H, A, F, K, J, C]
    merged_keys = torch.einsum("bhafkjcl,bfklhd->bafjkchd", merge_weights, k)
    merged_values = torch.einsum("bhafkjcl,bfklhe->bafjkche", merge_weights, v)
    routing_bias = (merge_weights * merge_logprob).sum(dim=-1)
    return merged_keys, merged_values, routing_bias, cluster_has_keys


def _merge_kv_with_backend(seed_queries, k, v, backend, scale):
    backend = (backend or "sdpa").lower()
    if backend == "sdpa":
        return _torch_hard_route_merge(seed_queries, k, v, scale)
    if backend == "triton":
        return triton_hard_route_merge(seed_queries, k, v, scale=scale)
    if backend == "triton_c4":
        return triton_hard_route_merge_c4(seed_queries, k, v, scale=scale)
    if backend == "triton_c4_j16":
        return triton_hard_route_merge_c4(seed_queries, k, v, scale=scale, block_j_target=16)
    raise ValueError(
        f"unsupported hard_route_backend={backend!r}; expected 'sdpa', 'triton', "
        "'triton_c4', 'triton_c4_j16', or a direct-route C4 backend."
    )


def _direct_route_merge(q, seed_rows, k, v, config, scale, assume_contiguous=False):
    return triton_hard_route_merge_c4_direct_route(
        q,
        seed_rows,
        k,
        v,
        scale=scale,
        block_j_target=32,
        num_warps=2,
        bias_dtype=torch.float16,
        merged_dtype=torch.bfloat16 if config.use_bf16_merged_kv else torch.float16,
        use_exp2=True,
        force_seed_keys=True,
        num_stages=1,
        logit_input_precision="tf32",
        inline_local_seed=config.inline_local_seed,
        key_outer_block=config.key_outer_block,
        use_assigned_logit=config.use_assigned_logit,
        assume_contiguous=assume_contiguous and config.inline_local_seed,
    )


def hard_route_compressed_attention(query, key, value, bias, backend, scale):
    backend = (backend or "sdpa").lower()
    if backend == "sdpa":
        return biased_sdpa(query, key, value, bias, scale=scale)
    if backend == "triton":
        return triton_biased_attention(query, key, value, bias, scale=scale)
    raise ValueError(
        f"unsupported hard_route_routing_backend={backend!r}; expected 'sdpa', "
        "'triton', or a direct-route backend."
    )


def hard_route_direct_compressed_attention(
    q,
    key,
    value,
    bias,
    backend,
    scale,
    output=None,
    assume_contiguous=False,
):
    backend = (backend or "").lower()
    key_token_major = backend.endswith("_tokenmajor")
    if not key_token_major:
        return None
    backend = backend[: -len("_tokenmajor")]
    direct_config = _DIRECT_ROUTE_ATTENTION_CONFIGS.get(backend)
    if direct_config is None:
        return None

    block_m, block_n, num_warps, num_stages = direct_config
    return triton_direct_route_biased_attention(
        q,
        key,
        value,
        bias,
        block_m=block_m,
        block_n=block_n,
        scale=scale,
        num_warps=num_warps,
        num_stages=num_stages,
        use_exp2=True,
        out=output,
        assume_contiguous=assume_contiguous,
    )


def _restore_attention_blocks(out, q):
    return rearrange(
        out,
        "(b h a j) i d -> b a i j h d",
        b=q.size(0),
        h=q.size(4),
        a=q.size(1),
        j=q.size(3),
    )


def _run_reference_path(q, k, v, scale, seed_rows, hard_route_backend, hard_route_routing_backend):
    """Readable PyTorch/reference path: gather seeds, merge KV, then attend."""
    query = rearrange(q, "b a i j h d -> (b h a j) i d")
    seed_queries = gather_seed_queries(q, seed_rows)
    merged_keys, merged_values, routing_bias, _ = _merge_kv_with_backend(
        seed_queries,
        k,
        v,
        hard_route_backend,
        scale,
    )
    key = rearrange(merged_keys, "b a f j k c h d -> (b h a j) (f k c) d")
    value = rearrange(merged_values, "b a f j k c h d -> (b h a j) (f k c) d")
    bias = rearrange(routing_bias, "b h a f k j c -> (b h a j) (f k c)")
    out = hard_route_compressed_attention(
        query,
        key,
        value,
        bias,
        hard_route_routing_backend,
        scale,
    )
    return _restore_attention_blocks(out, q)


def _run_direct_path(q, k, v, scale, seed_rows, direct_config, hard_route_routing_backend, output):
    """Optimized C4 path: Triton merge writes routing-ready token-major KV."""
    q = q.contiguous()
    key, value, bias, _ = _direct_route_merge(
        q,
        seed_rows,
        k,
        v,
        direct_config,
        scale,
        assume_contiguous=True,
    )
    direct_out = hard_route_direct_compressed_attention(
        q,
        key,
        value,
        bias,
        hard_route_routing_backend,
        scale,
        output=output if output is not None and output.is_contiguous() else None,
        assume_contiguous=True,
    )
    if direct_out is not None:
        return direct_out

    query = rearrange(q, "b a i j h d -> (b h a j) i d")
    out = hard_route_compressed_attention(
        query,
        key,
        value,
        bias,
        hard_route_routing_backend,
        scale,
    )
    return _restore_attention_blocks(out, q)


def hard_route_attention(
    q,
    k,
    v,
    scale,
    num_merge_tokens,
    seed_mode,
    seed,
    query_outer_offset,
    hard_route_backend="sdpa",
    hard_route_routing_backend="sdpa",
    output=None,
):
    # q:   [B, A, I, J, H, D]
    # k/v: [B, F, K, L, H, D]
    # A fixed (a, f, j, k) block has I queries x L keys. This path merges
    # the L keys into several hard-assigned merged tokens instead of one.
    validate_block_seed_mode(seed_mode, "multi-token merging")
    direct_backend = _parse_direct_route_backend(hard_route_backend)
    if direct_backend.inline_local_seed and seed_mode != "local_with_ith":
        raise ValueError("inline-local-seed hard-route backend requires seed_mode='local_with_ith'.")
    if num_merge_tokens > k.size(3):
        raise ValueError("num_merge_tokens cannot exceed key columns per block.")

    seed_rows = None
    if not direct_backend.inline_local_seed:
        seed_rows = _build_seed_rows(
            q,
            k,
            query_outer_offset,
            num_merge_tokens,
            seed,
            seed_mode,
        )

    if direct_backend.supported:
        return _run_direct_path(
            q,
            k,
            v,
            scale,
            seed_rows,
            direct_backend,
            hard_route_routing_backend,
            output,
        )

    if seed_rows is None:
        seed_rows = _build_seed_rows(
            q,
            k,
            query_outer_offset,
            num_merge_tokens,
            seed,
            seed_mode,
        )
    return _run_reference_path(
        q,
        k,
        v,
        scale,
        seed_rows,
        hard_route_backend,
        hard_route_routing_backend,
    )


__all__ = [
    "assign_keys_to_seeds",
    "hard_route_attention",
    "hard_route_compressed_attention",
    "hard_route_merge_weights",
    "uses_inline_local_seed_backend",
]
