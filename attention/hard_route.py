import torch
from einops import rearrange

from .common import (
    biased_sdpa,
    gather_seed_queries,
    random_seed_rows,
    validate_block_seed_mode,
)
from .hard_route_triton import (
    triton_biased_attention,
    triton_direct_route_biased_attention,
    triton_grid_biased_attention,
    triton_grid_outer_exact_local_biased_attention,
    triton_grid_outer_local_biased_attention,
    triton_hard_route_merge,
    triton_hard_route_merge_c4,
    triton_hard_route_merge_c4_direct_route,
)


_SEED_ROWS_CACHE = {}


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
    include_ith,
    fast_with_include,
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
        bool(include_ith),
        bool(fast_with_include),
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
    seed_rows = random_seed_rows(
        query_outer,
        key_outer,
        key_rows,
        query_columns,
        query_rows,
        num_seeds,
        seed,
        include_ith=include_ith,
        fast_with_include=fast_with_include,
    )
    _SEED_ROWS_CACHE[cache_key] = seed_rows
    return seed_rows


def assign_keys_to_seeds(seed_logits):
    # seed_logits: [B, H, A, F, K, J, C, L].
    num_seeds = seed_logits.size(-2)
    num_keys = seed_logits.size(-1)

    assignments = seed_logits.argmax(dim=-2)
    assignment_mask = torch.nn.functional.one_hot(
        assignments,
        num_classes=num_seeds,
    ).movedim(-1, -2).bool()

    has_seed = assignment_mask.any(dim=-1)
    key_idx = torch.arange(num_keys, device=seed_logits.device)
    key_idx = key_idx.view(*((1,) * has_seed.ndim), num_keys)
    # Empty clusters are rare; give each empty seed the first key without a
    # Python loop. In this fallback only, key 0 may be shared by multiple seeds.
    force_mask = (~has_seed).unsqueeze(-1) & (key_idx == 0)

    return assignment_mask | force_mask


def hard_route_merge_weights(seed_logits, dtype):
    """Assign each key to one seed, then softmax within each routed cluster."""
    assignment_mask = assign_keys_to_seeds(seed_logits)
    cluster_has_keys = assignment_mask.any(dim=-1)

    masked_logits = seed_logits.masked_fill(~assignment_mask, torch.finfo(seed_logits.dtype).min)
    merge_logprob = torch.log_softmax(masked_logits, dim=-1)
    merge_weights = merge_logprob.exp().masked_fill(~assignment_mask, 0).to(dtype)
    merge_logprob = merge_logprob.masked_fill(~assignment_mask, 0)
    return merge_weights, merge_logprob, cluster_has_keys


def _torch_merge(seed_queries, k, v, scale):
    seed_logits = torch.einsum("bafkjchd,bfklhd->bhafkjcl", seed_queries, k).float()
    seed_logits = seed_logits * float(scale)
    merge_weights, merge_logprob, cluster_has_keys = hard_route_merge_weights(seed_logits, k.dtype)

    merged_keys = torch.einsum("bhafkjcl,bfklhd->bafjkchd", merge_weights, k)
    merged_values = torch.einsum("bhafkjcl,bfklhe->bafjkche", merge_weights, v)
    routing_bias = (merge_weights * merge_logprob).sum(dim=-1)
    return merged_keys, merged_values, routing_bias, cluster_has_keys


def _merge(seed_queries, k, v, backend, scale):
    backend = (backend or "sdpa").lower()
    if backend == "sdpa":
        return _torch_merge(seed_queries, k, v, scale)
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


def _direct_route_merge(q, seed_rows, k, v, backend, scale):
    backend = (backend or "").lower()
    if backend == "triton_c4_direct_route_j16_w2":
        return triton_hard_route_merge_c4_direct_route(
            q,
            seed_rows,
            k,
            v,
            scale=scale,
            block_j_target=16,
            num_warps=2,
        )
    if backend == "triton_c4_direct_route_j16_w1":
        return triton_hard_route_merge_c4_direct_route(
            q,
            seed_rows,
            k,
            v,
            scale=scale,
            block_j_target=16,
            num_warps=1,
        )
    if backend == "triton_c4_direct_route_j32_w2":
        return triton_hard_route_merge_c4_direct_route(
            q,
            seed_rows,
            k,
            v,
            scale=scale,
            block_j_target=32,
            num_warps=2,
        )
    if backend == "triton_c4_direct_route_j32_w2_exp2merge":
        return triton_hard_route_merge_c4_direct_route(
            q,
            seed_rows,
            k,
            v,
            scale=scale,
            block_j_target=32,
            num_warps=2,
            use_exp2=True,
        )
    if backend == "triton_c4_direct_route_j32_w2_exp2merge_forcekeys":
        return triton_hard_route_merge_c4_direct_route(
            q,
            seed_rows,
            k,
            v,
            scale=scale,
            block_j_target=32,
            num_warps=2,
            use_exp2=True,
            force_seed_keys=True,
        )
    if backend == "triton_c4_direct_route_j32_w2_exp2merge_forcekeys_bf16bias":
        return triton_hard_route_merge_c4_direct_route(
            q,
            seed_rows,
            k,
            v,
            scale=scale,
            block_j_target=32,
            num_warps=2,
            bias_dtype=torch.bfloat16,
            use_exp2=True,
            force_seed_keys=True,
        )
    if backend == "triton_c4_direct_route_j32_w2_exp2merge_forcekeys_bf16bias_s1":
        return triton_hard_route_merge_c4_direct_route(
            q,
            seed_rows,
            k,
            v,
            scale=scale,
            block_j_target=32,
            num_warps=2,
            bias_dtype=torch.bfloat16,
            use_exp2=True,
            force_seed_keys=True,
            num_stages=1,
        )
    if backend == "triton_c4_direct_route_j32_w2_exp2merge_forcekeys_bf16bias_s1_fastdot":
        return triton_hard_route_merge_c4_direct_route(
            q,
            seed_rows,
            k,
            v,
            scale=scale,
            block_j_target=32,
            num_warps=2,
            bias_dtype=torch.bfloat16,
            use_exp2=True,
            force_seed_keys=True,
            num_stages=1,
            logit_input_precision="tf32",
        )
    if backend == "triton_c4_direct_route_j16_w1_exp2merge_forcekeys_bf16bias_s1_fastdot":
        return triton_hard_route_merge_c4_direct_route(
            q,
            seed_rows,
            k,
            v,
            scale=scale,
            block_j_target=16,
            num_warps=1,
            bias_dtype=torch.bfloat16,
            use_exp2=True,
            force_seed_keys=True,
            num_stages=1,
            logit_input_precision="tf32",
        )
    if backend == "triton_c4_direct_route_j16_w2_exp2merge_forcekeys_bf16bias_s1_fastdot":
        return triton_hard_route_merge_c4_direct_route(
            q,
            seed_rows,
            k,
            v,
            scale=scale,
            block_j_target=16,
            num_warps=2,
            bias_dtype=torch.bfloat16,
            use_exp2=True,
            force_seed_keys=True,
            num_stages=1,
            logit_input_precision="tf32",
        )
    if backend == "triton_c4_direct_route_j32_w1_exp2merge_forcekeys_bf16bias_s1_fastdot":
        return triton_hard_route_merge_c4_direct_route(
            q,
            seed_rows,
            k,
            v,
            scale=scale,
            block_j_target=32,
            num_warps=1,
            bias_dtype=torch.bfloat16,
            use_exp2=True,
            force_seed_keys=True,
            num_stages=1,
            logit_input_precision="tf32",
        )
    if backend == "triton_c4_direct_route_j32_w4_exp2merge_forcekeys_bf16bias_s1_fastdot":
        return triton_hard_route_merge_c4_direct_route(
            q,
            seed_rows,
            k,
            v,
            scale=scale,
            block_j_target=32,
            num_warps=4,
            bias_dtype=torch.bfloat16,
            use_exp2=True,
            force_seed_keys=True,
            num_stages=1,
            logit_input_precision="tf32",
        )
    if backend == "triton_c4_direct_route_j32_w2_exp2merge_forcekeys_bf16bias_s2":
        return triton_hard_route_merge_c4_direct_route(
            q,
            seed_rows,
            k,
            v,
            scale=scale,
            block_j_target=32,
            num_warps=2,
            bias_dtype=torch.bfloat16,
            use_exp2=True,
            force_seed_keys=True,
            num_stages=2,
        )
    if backend == "triton_c4_direct_route_j32_w2_exp2merge_forcekeys_bf16bias_s4":
        return triton_hard_route_merge_c4_direct_route(
            q,
            seed_rows,
            k,
            v,
            scale=scale,
            block_j_target=32,
            num_warps=2,
            bias_dtype=torch.bfloat16,
            use_exp2=True,
            force_seed_keys=True,
            num_stages=4,
        )
    if backend == "triton_c4_direct_route_j32_w2_exp2merge_bf16bias":
        return triton_hard_route_merge_c4_direct_route(
            q,
            seed_rows,
            k,
            v,
            scale=scale,
            block_j_target=32,
            num_warps=2,
            bias_dtype=torch.bfloat16,
            use_exp2=True,
        )
    if backend == "triton_c4_direct_route_j32_w2_bf16bias":
        return triton_hard_route_merge_c4_direct_route(
            q,
            seed_rows,
            k,
            v,
            scale=scale,
            block_j_target=32,
            num_warps=2,
            bias_dtype=torch.bfloat16,
        )
    if backend == "triton_c4_direct_route_j32_w1":
        return triton_hard_route_merge_c4_direct_route(
            q,
            seed_rows,
            k,
            v,
            scale=scale,
            block_j_target=32,
            num_warps=1,
        )
    if backend == "triton_c4_direct_route_j32_w4":
        return triton_hard_route_merge_c4_direct_route(
            q,
            seed_rows,
            k,
            v,
            scale=scale,
            block_j_target=32,
            num_warps=4,
        )
    if backend == "triton_c4_direct_route_j64_w1":
        return triton_hard_route_merge_c4_direct_route(
            q,
            seed_rows,
            k,
            v,
            scale=scale,
            block_j_target=64,
            num_warps=1,
        )
    if backend == "triton_c4_direct_route_j64_w2":
        return triton_hard_route_merge_c4_direct_route(
            q,
            seed_rows,
            k,
            v,
            scale=scale,
            block_j_target=64,
            num_warps=2,
        )
    if backend == "triton_c4_direct_route_j64_w4":
        return triton_hard_route_merge_c4_direct_route(
            q,
            seed_rows,
            k,
            v,
            scale=scale,
            block_j_target=64,
            num_warps=4,
        )
    return None


def hard_route_compressed_attention(query, key, value, bias, backend, scale):
    backend = (backend or "sdpa").lower()
    if backend == "sdpa":
        return biased_sdpa(query, key, value, bias, scale=scale)
    if backend == "triton":
        return triton_biased_attention(query, key, value, bias, scale=scale)
    if backend == "triton_m32_n32_s1":
        return triton_biased_attention(
            query,
            key,
            value,
            bias,
            block_m=32,
            block_n=32,
            scale=scale,
            num_stages=1,
        )
    if backend == "triton_m128_n32":
        return triton_biased_attention(
            query,
            key,
            value,
            bias,
            block_m=128,
            block_n=32,
            scale=scale,
        )
    if backend == "triton_m128_n32_s1":
        return triton_biased_attention(
            query,
            key,
            value,
            bias,
            block_m=128,
            block_n=32,
            scale=scale,
            num_stages=1,
        )
    if backend == "triton_m32_n128":
        return triton_biased_attention(
            query,
            key,
            value,
            bias,
            block_m=32,
            block_n=128,
            scale=scale,
        )
    if backend == "triton_m32_n128_w2_s1":
        return triton_biased_attention(
            query,
            key,
            value,
            bias,
            block_m=32,
            block_n=128,
            scale=scale,
            num_warps=2,
            num_stages=1,
        )
    if backend == "triton_m32_n128_s1":
        return triton_biased_attention(
            query,
            key,
            value,
            bias,
            block_m=32,
            block_n=128,
            scale=scale,
            num_stages=1,
        )
    if backend == "triton_m32_n128_w2_s1_exp2":
        return triton_biased_attention(
            query,
            key,
            value,
            bias,
            block_m=32,
            block_n=128,
            scale=scale,
            num_warps=2,
            num_stages=1,
            use_exp2=True,
        )
    if backend == "triton_m32_n256_s1":
        return triton_biased_attention(
            query,
            key,
            value,
            bias,
            block_m=32,
            block_n=256,
            scale=scale,
            num_stages=1,
        )
    if backend == "triton_m32_n64_s1":
        return triton_biased_attention(
            query,
            key,
            value,
            bias,
            block_m=32,
            block_n=64,
            scale=scale,
            num_stages=1,
        )
    if backend == "triton_m32_n64_w1_s1":
        return triton_biased_attention(
            query,
            key,
            value,
            bias,
            block_m=32,
            block_n=64,
            scale=scale,
            num_warps=1,
            num_stages=1,
        )
    if backend == "triton_m32_n64_w2_s1":
        return triton_biased_attention(
            query,
            key,
            value,
            bias,
            block_m=32,
            block_n=64,
            scale=scale,
            num_warps=2,
            num_stages=1,
        )
    if backend == "triton_m32_n64_w1_s1_exp2":
        return triton_biased_attention(
            query,
            key,
            value,
            bias,
            block_m=32,
            block_n=64,
            scale=scale,
            num_warps=1,
            num_stages=1,
            use_exp2=True,
        )
    if backend == "triton_m32_n64_w2_s1_exp2":
        return triton_biased_attention(
            query,
            key,
            value,
            bias,
            block_m=32,
            block_n=64,
            scale=scale,
            num_warps=2,
            num_stages=1,
            use_exp2=True,
        )
    if backend == "triton_m64_n64_w2_s1":
        return triton_biased_attention(
            query,
            key,
            value,
            bias,
            block_m=64,
            block_n=64,
            scale=scale,
            num_warps=2,
            num_stages=1,
        )
    if backend == "triton_m256_n32":
        return triton_biased_attention(
            query,
            key,
            value,
            bias,
            block_m=256,
            block_n=32,
            scale=scale,
        )
    raise ValueError(
        f"unsupported hard_route_routing_backend={backend!r}; expected "
        "'sdpa', 'triton', 'triton_m128_n32', 'triton_m128_n32_s1', "
        "or 'triton_m256_n32'."
    )


def hard_route_direct_compressed_attention(q, key, value, bias, backend, scale, output=None):
    backend = (backend or "").lower()
    direct_configs = {
        "triton_direct_m16_n64_w1_s1_exp2": (16, 64, 1),
        "triton_direct_m16_n128_w1_s1_exp2": (16, 128, 1),
        "triton_direct_m32_n32_w1_s1_exp2": (32, 32, 1),
        "triton_direct_m32_n64_w1_s1_exp2": (32, 64, 1),
        "triton_direct_m32_n64_w2_s1_exp2": (32, 64, 2),
        "triton_direct_m32_n128_w1_s1_exp2": (32, 128, 1),
        "triton_direct_m32_n128_w2_s1_exp2": (32, 128, 2),
        "triton_direct_m64_n64_w1_s1_exp2": (64, 64, 1),
        "triton_direct_m64_n128_w1_s1_exp2": (64, 128, 1),
    }
    direct_config = direct_configs.get(backend)
    if direct_config is not None:
        block_m, block_n, num_warps = direct_config
        return triton_direct_route_biased_attention(
            q,
            key,
            value,
            bias,
            block_m=block_m,
            block_n=block_n,
            scale=scale,
            num_warps=num_warps,
            num_stages=1,
            use_exp2=True,
            out=output,
        )
    return None


def hard_route_grid_compressed_attention(query, key, value, bias, backend, scale):
    backend = (backend or "sdpa").lower()
    if backend == "triton_grid_outer_exact_r4_m128_n64_w4_s3_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=4,
            block_m=128,
            block_n=64,
            scale=scale,
            num_stages=3,
        )
    if backend == "triton_grid_m128_n32_s1":
        return triton_grid_biased_attention(
            query,
            key,
            value,
            bias,
            block_m=128,
            block_n=32,
            scale=scale,
            num_stages=1,
        )
    if backend == "triton_grid_m128_n32_s1_nobias":
        return triton_grid_biased_attention(
            query,
            key,
            value,
            bias,
            block_m=128,
            block_n=32,
            scale=scale,
            num_stages=1,
            use_bias=False,
        )
    if backend == "triton_grid_m128_n32_s1_exp2":
        return triton_grid_biased_attention(
            query,
            key,
            value,
            bias,
            block_m=128,
            block_n=32,
            scale=scale,
            num_stages=1,
            use_exp2=True,
        )
    if backend == "triton_grid_m128_n64_s1_exp2":
        return triton_grid_biased_attention(
            query,
            key,
            value,
            bias,
            block_m=128,
            block_n=64,
            scale=scale,
            num_stages=1,
            use_exp2=True,
        )
    if backend == "triton_grid_m128_n32_s1_exp2_nobias":
        return triton_grid_biased_attention(
            query,
            key,
            value,
            bias,
            block_m=128,
            block_n=32,
            scale=scale,
            num_stages=1,
            use_bias=False,
            use_exp2=True,
        )
    if backend == "triton_grid_outer_local_r2_m128_n32_s1_exp2":
        return triton_grid_outer_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=2,
            block_m=128,
            block_n=32,
            scale=scale,
        )
    if backend == "triton_grid_outer_local_r2_m128_n64_s1_exp2":
        return triton_grid_outer_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=2,
            block_m=128,
            block_n=64,
            scale=scale,
        )
    if backend == "triton_grid_outer_local_r3_m128_n32_s1_exp2":
        return triton_grid_outer_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=3,
            block_m=128,
            block_n=32,
            scale=scale,
        )
    if backend == "triton_grid_outer_local_r3_m128_n64_s1_exp2":
        return triton_grid_outer_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=3,
            block_m=128,
            block_n=64,
            scale=scale,
        )
    if backend == "triton_grid_outer_local_r3_m64_n64_s1_exp2":
        return triton_grid_outer_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=3,
            block_m=64,
            block_n=64,
            scale=scale,
        )
    if backend == "triton_grid_outer_exact_r2_m128_n64_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=2,
            block_m=128,
            block_n=64,
            scale=scale,
        )
    if backend == "triton_grid_outer_exact_r3_m128_n64_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=3,
            block_m=128,
            block_n=64,
            scale=scale,
        )
    if backend == "triton_grid_outer_exact_r3_m64_n64_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=3,
            block_m=64,
            block_n=64,
            scale=scale,
        )
    if backend == "triton_grid_outer_exact_r3_m128_n32_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=3,
            block_m=128,
            block_n=32,
            scale=scale,
        )
    if backend == "triton_grid_outer_exact_r3_m128_n128_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=3,
            block_m=128,
            block_n=128,
            scale=scale,
        )
    if backend == "triton_grid_outer_exact_r3_m128_n64_w2_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=3,
            block_m=128,
            block_n=64,
            scale=scale,
            num_warps=2,
        )
    if backend == "triton_grid_outer_exact_r3_m128_n64_w8_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=3,
            block_m=128,
            block_n=64,
            scale=scale,
            num_warps=8,
        )
    if backend == "triton_grid_outer_exact_r3_m128_n64_w4_s2_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=3,
            block_m=128,
            block_n=64,
            scale=scale,
            num_stages=2,
        )
    if backend == "triton_grid_outer_exact_r3_m256_n64_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=3,
            block_m=256,
            block_n=64,
            scale=scale,
        )
    if backend == "triton_grid_outer_exact_r3_m256_n128_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=3,
            block_m=256,
            block_n=128,
            scale=scale,
        )
    if backend == "triton_grid_outer_local_r4_m128_n32_s1_exp2":
        return triton_grid_outer_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=4,
            block_m=128,
            block_n=32,
            scale=scale,
        )
    if backend == "triton_grid_outer_local_r4_m128_n64_s1_exp2":
        return triton_grid_outer_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=4,
            block_m=128,
            block_n=64,
            scale=scale,
        )
    if backend == "triton_grid_outer_local_r4_m64_n64_s1_exp2":
        return triton_grid_outer_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=4,
            block_m=64,
            block_n=64,
            scale=scale,
        )
    if backend == "triton_grid_outer_exact_r4_m128_n64_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=4,
            block_m=128,
            block_n=64,
            scale=scale,
        )
    if backend == "triton_grid_outer_exact_r4_m128_n32_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=4,
            block_m=128,
            block_n=32,
            scale=scale,
        )
    if backend == "triton_grid_outer_exact_r4_m128_n32_w4_s2_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=4,
            block_m=128,
            block_n=32,
            scale=scale,
            num_stages=2,
        )
    if backend == "triton_grid_outer_exact_r4_m128_n32_w4_s3_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=4,
            block_m=128,
            block_n=32,
            scale=scale,
            num_stages=3,
        )
    if backend == "triton_grid_outer_exact_r4_m128_n32_w4_s4_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=4,
            block_m=128,
            block_n=32,
            scale=scale,
            num_stages=4,
        )
    if backend == "triton_grid_outer_exact_r4_m128_n128_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=4,
            block_m=128,
            block_n=128,
            scale=scale,
        )
    if backend == "triton_grid_outer_exact_r4_m128_n128_w4_s2_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=4,
            block_m=128,
            block_n=128,
            scale=scale,
            num_stages=2,
        )
    if backend == "triton_grid_outer_exact_r4_m128_n128_w4_s3_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=4,
            block_m=128,
            block_n=128,
            scale=scale,
            num_stages=3,
        )
    if backend == "triton_grid_outer_exact_r4_m128_n64_w1_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=4,
            block_m=128,
            block_n=64,
            scale=scale,
            num_warps=1,
        )
    if backend == "triton_grid_outer_exact_r4_m128_n64_w1_s2_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=4,
            block_m=128,
            block_n=64,
            scale=scale,
            num_warps=1,
            num_stages=2,
        )
    if backend == "triton_grid_outer_exact_r4_m128_n64_w1_s3_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=4,
            block_m=128,
            block_n=64,
            scale=scale,
            num_warps=1,
            num_stages=3,
        )
    if backend == "triton_grid_outer_exact_r4_m128_n64_w2_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=4,
            block_m=128,
            block_n=64,
            scale=scale,
            num_warps=2,
        )
    if backend == "triton_grid_outer_exact_r4_m128_n64_w2_s2_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=4,
            block_m=128,
            block_n=64,
            scale=scale,
            num_warps=2,
            num_stages=2,
        )
    if backend == "triton_grid_outer_exact_r4_m128_n64_w2_s3_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=4,
            block_m=128,
            block_n=64,
            scale=scale,
            num_warps=2,
            num_stages=3,
        )
    if backend == "triton_grid_outer_exact_r4_m128_n64_w8_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=4,
            block_m=128,
            block_n=64,
            scale=scale,
            num_warps=8,
        )
    if backend == "triton_grid_outer_exact_r4_m128_n64_w8_s2_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=4,
            block_m=128,
            block_n=64,
            scale=scale,
            num_warps=8,
            num_stages=2,
        )
    if backend == "triton_grid_outer_exact_r4_m128_n64_w4_s2_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=4,
            block_m=128,
            block_n=64,
            scale=scale,
            num_stages=2,
        )
    if backend == "triton_grid_outer_exact_r4_m128_n64_w4_s4_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=4,
            block_m=128,
            block_n=64,
            scale=scale,
            num_stages=4,
        )
    if backend == "triton_grid_outer_exact_r4_m128_n64_s1_exp2_nobias":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=4,
            block_m=128,
            block_n=64,
            scale=scale,
            use_bias=False,
        )
    if backend == "triton_grid_outer_exact_r4_m128_n32_s1_exp2_nobias":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=4,
            block_m=128,
            block_n=32,
            scale=scale,
            use_bias=False,
        )
    if backend == "triton_grid_outer_exact_r4_m64_n64_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=4,
            block_m=64,
            block_n=64,
            scale=scale,
        )
    if backend == "triton_grid_outer_local_r4_m128_n32_s1_exp2_nobias":
        return triton_grid_outer_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=4,
            block_m=128,
            block_n=32,
            scale=scale,
            use_bias=False,
        )
    if backend == "triton_grid_outer_local_r5_m128_n32_s1_exp2":
        return triton_grid_outer_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=5,
            block_m=128,
            block_n=32,
            scale=scale,
        )
    if backend == "triton_grid_outer_local_r5_m128_n64_s1_exp2":
        return triton_grid_outer_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=5,
            block_m=128,
            block_n=64,
            scale=scale,
        )
    if backend == "triton_grid_outer_local_r5_m64_n64_s1_exp2":
        return triton_grid_outer_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=5,
            block_m=64,
            block_n=64,
            scale=scale,
        )
    if backend == "triton_grid_outer_exact_r5_m128_n64_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=5,
            block_m=128,
            block_n=64,
            scale=scale,
        )
    if backend == "triton_grid_outer_exact_r5_m128_n32_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=5,
            block_m=128,
            block_n=32,
            scale=scale,
        )
    if backend == "triton_grid_outer_exact_r5_m128_n32_w4_s2_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=5,
            block_m=128,
            block_n=32,
            scale=scale,
            num_stages=2,
        )
    if backend == "triton_grid_outer_exact_r5_m128_n64_w2_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=5,
            block_m=128,
            block_n=64,
            scale=scale,
            num_warps=2,
        )
    if backend == "triton_grid_outer_exact_r5_m128_n64_w2_s2_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=5,
            block_m=128,
            block_n=64,
            scale=scale,
            num_warps=2,
            num_stages=2,
        )
    if backend == "triton_grid_outer_exact_r5_m128_n64_w8_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=5,
            block_m=128,
            block_n=64,
            scale=scale,
            num_warps=8,
        )
    if backend == "triton_grid_outer_exact_r5_m128_n64_w4_s2_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=5,
            block_m=128,
            block_n=64,
            scale=scale,
            num_stages=2,
        )
    if backend == "triton_grid_outer_exact_r5_m128_n64_w4_s3_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=5,
            block_m=128,
            block_n=64,
            scale=scale,
            num_stages=3,
        )
    if backend == "triton_grid_outer_exact_r5_m128_n64_w4_s4_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=5,
            block_m=128,
            block_n=64,
            scale=scale,
            num_stages=4,
        )
    if backend == "triton_grid_outer_exact_r5_m64_n64_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=5,
            block_m=64,
            block_n=64,
            scale=scale,
        )
    if backend == "triton_grid_outer_exact_r5_m128_n128_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=5,
            block_m=128,
            block_n=128,
            scale=scale,
        )
    if backend == "triton_grid_outer_exact_r5_m256_n64_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=5,
            block_m=256,
            block_n=64,
            scale=scale,
        )
    if backend == "triton_grid_outer_exact_r5_m256_n128_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=5,
            block_m=256,
            block_n=128,
            scale=scale,
        )
    if backend == "triton_grid_outer_local_r6_m128_n32_s1_exp2":
        return triton_grid_outer_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=6,
            block_m=128,
            block_n=32,
            scale=scale,
        )
    if backend == "triton_grid_outer_local_r6_m128_n64_s1_exp2":
        return triton_grid_outer_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=6,
            block_m=128,
            block_n=64,
            scale=scale,
        )
    if backend == "triton_grid_outer_local_r6_m64_n64_s1_exp2":
        return triton_grid_outer_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=6,
            block_m=64,
            block_n=64,
            scale=scale,
        )
    if backend == "triton_grid_outer_exact_r6_m128_n64_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=6,
            block_m=128,
            block_n=64,
            scale=scale,
        )
    if backend == "triton_grid_outer_exact_r6_m64_n64_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=6,
            block_m=64,
            block_n=64,
            scale=scale,
        )
    if backend == "triton_grid_outer_exact_r6_m128_n128_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=6,
            block_m=128,
            block_n=128,
            scale=scale,
        )
    if backend == "triton_grid_outer_exact_r6_m256_n64_s1_exp2":
        return triton_grid_outer_exact_local_biased_attention(
            query,
            key,
            value,
            bias,
            local_radius=6,
            block_m=256,
            block_n=64,
            scale=scale,
        )
    raise ValueError(
        f"unsupported grid hard_route_routing_backend={backend!r}; expected "
        "'triton_grid_m128_n32_s1', 'triton_grid_m128_n32_s1_nobias', "
        "'triton_grid_m128_n32_s1_exp2', 'triton_grid_m128_n64_s1_exp2', "
        "'triton_grid_m128_n32_s1_exp2_nobias', "
        "'triton_grid_outer_local_r{2,3,4,5,6}_m128_n32_s1_exp2', "
        "'triton_grid_outer_local_r{2,3,4,5,6}_m128_n64_s1_exp2', "
        "'triton_grid_outer_local_r{3,4,5,6}_m64_n64_s1_exp2', "
        "'triton_grid_outer_exact_r{2,3,4,5,6}_m128_n64_s1_exp2', "
        "'triton_grid_outer_exact_r4_m128_n32_s1_exp2', "
        "'triton_grid_outer_exact_r4_m128_n32_w4_s{2,3,4}_exp2', "
        "'triton_grid_outer_exact_r4_m128_n128_s1_exp2', "
        "'triton_grid_outer_exact_r4_m128_n128_w4_s{2,3}_exp2', "
        "'triton_grid_outer_exact_r4_m128_n64_w1_s{1,2,3}_exp2', "
        "'triton_grid_outer_exact_r4_m128_n64_w2_s{1,2,3}_exp2', "
        "'triton_grid_outer_exact_r4_m128_n64_w8_s{1,2}_exp2', "
        "'triton_grid_outer_exact_r4_m128_n64_w4_s{2,3,4}_exp2', "
        "'triton_grid_outer_exact_r4_m128_n{32,64}_s1_exp2_nobias', "
        "'triton_grid_outer_exact_r5_m128_n32_s1_exp2', "
        "'triton_grid_outer_exact_r5_m128_n32_w4_s2_exp2', "
        "'triton_grid_outer_exact_r{4,5,6}_m64_n64_s1_exp2', "
        "'triton_grid_outer_exact_r{5,6}_m128_n128_s1_exp2', "
        "'triton_grid_outer_exact_r{5,6}_m256_n64_s1_exp2', "
        "'triton_grid_outer_exact_r5_m256_n128_s1_exp2', "
        "'triton_grid_outer_exact_r5_m128_n64_w{2,8}_s1_exp2', "
        "'triton_grid_outer_exact_r5_m128_n64_w2_s2_exp2', "
        "'triton_grid_outer_exact_r5_m128_n64_w4_s{2,3,4}_exp2', "
        "'triton_grid_outer_local_r4_m128_n32_s1_exp2_nobias', "
        "or 'triton_grid_outer_local_r6_m128_n32_s1_exp2'."
    )


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
    include_ith = validate_block_seed_mode(seed_mode, "multi-token merging")
    fast_with_include = seed_mode == "fast_random_with_ith"
    if num_merge_tokens > k.size(3):
        raise ValueError("num_merge_tokens cannot exceed key columns per block.")

    seed_rows = _cached_seed_rows(
        q.device,
        query_outer_offset,
        q.size(1),
        k.size(1),
        k.size(2),
        q.size(3),
        q.size(2),
        num_merge_tokens,
        seed,
        include_ith,
        fast_with_include,
    )

    direct_merge = _direct_route_merge(q, seed_rows, k, v, hard_route_backend, scale)
    if direct_merge is None:
        query = rearrange(q, "b a i j h d -> (b h a j) i d")
        seed_queries = gather_seed_queries(q, seed_rows)
        merged_keys, merged_values, routing_bias, cluster_has_keys = _merge(
            seed_queries,
            k,
            v,
            hard_route_backend,
            scale,
        )
        key = rearrange(merged_keys, "b a f j k c h d -> (b h a j) (f k c) d")
        value = rearrange(merged_values, "b a f j k c h d -> (b h a j) (f k c) d")
        bias = rearrange(routing_bias, "b h a f k j c -> (b h a j) (f k c)")
        del cluster_has_keys
    else:
        key, value, bias, cluster_has_keys = direct_merge
        del cluster_has_keys
        direct_out = hard_route_direct_compressed_attention(
            q,
            key,
            value,
            bias,
            hard_route_routing_backend,
            scale,
            output=output if output is not None and output.is_contiguous() else None,
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
    return rearrange(
        out,
        "(b h a j) i d -> b a i j h d",
        b=q.size(0),
        h=q.size(4),
        a=q.size(1),
        j=q.size(3),
    )


__all__ = [
    "assign_keys_to_seeds",
    "hard_route_attention",
    "hard_route_compressed_attention",
    "hard_route_grid_compressed_attention",
    "hard_route_merge_weights",
]
