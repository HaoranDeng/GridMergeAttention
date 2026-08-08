import re
from dataclasses import dataclass, fields
from functools import lru_cache

import torch

from .hard_route_triton import (
    triton_direct_route_biased_attention,
    triton_direct_route_biased_attention_g2,
    triton_direct_route_indirect_value_attention,
    triton_direct_route_recompute_value_attention,
    triton_direct_route_seed_key_attention,
    triton_direct_route_top_key_attention,
    triton_direct_route_weight_value_attention,
    triton_hard_route_merge_c4_direct_route,
)


_DIRECT_ROUTE_C4_RE = re.compile(
    r"^triton_c4_direct_route_j(?P<j>\d+)_w2_exp2merge_forcekeys_fp16kvbias_s1_fastdot$"
)
_DIRECT_ROUTE_ATTENTION_RE = re.compile(
    r"^triton_direct(?:_g(?P<g>\d+))?_m(?P<m>\d+)_n(?P<n>\d+)_w(?P<w>\d+)_s(?P<s>\d+)_exp2$"
)
_DIRECT_ROUTE_KEY_TILE_RE = re.compile(r"^(?P<base>.+)_l(?P<l>\d+)$")

_DIRECT_ROUTE_KEY_OUTER_BLOCKS = {
    "f1": 1,
    "f2": 2,
    "f3": 3,
    "f4": 4,
    "f5": 5,
    "f6": 6,
    "f7": 7,
}
_DIRECT_ROUTE_FLAG_SUFFIXES = (
    ("_fp32scale", "use_fp32_key_scale"),
    ("_fp32bias", "use_fp32_routing_bias"),
    ("_seedkeyaffine", ("use_seed_key", "use_seed_key_scale", "use_seed_key_affine_scale")),
    ("_seedkeyls", ("use_seed_key", "use_seed_key_scale", "use_seed_key_lsq_scale")),
    ("_seedkeyproj", ("use_seed_key", "use_seed_key_scale")),
    ("_seedkeymatch", ("use_seed_key", "use_seed_key_bias_match")),
    ("_seedkey", "use_seed_key"),
    ("_indirecttop3seedvalue", "use_indirect_top3_seed_value"),
    ("_indirecttop3value", "use_indirect_top3_value"),
    ("_indirecttop2seedvalue", "use_indirect_top2_seed_value"),
    ("_indirecttop2value", "use_indirect_top2_value"),
    ("_indirectseedvalue", "use_indirect_seed_value"),
    ("_indirectvalue", "use_indirect_top_value"),
    ("_indirectkv", "use_indirect_top_kv"),
    ("_indirectkey", "use_indirect_top_key"),
    ("_topkv", "use_top_kv"),
    ("_topkeyaffine", ("use_top_key", "use_top_key_affine_scale")),
    ("_topkeyscale", ("use_top_key", "use_top_key_scale")),
    ("_topkey", "use_top_key"),
    ("_top3seedvalue", "use_top3_seed_value"),
    ("_top3value", "use_top3_value"),
    ("_top2seedvalue", "use_top2_seed_value"),
    ("_top2value", "use_top2_value"),
    ("_topvalue", "use_top_value"),
    ("_seedvalue", "use_seed_value"),
    ("_stripe", "use_stripe_assign"),
    ("_recomputevalue", "use_recompute_value"),
    ("_recompute", "use_recompute_merge"),
    ("_mergeweights", "use_merge_weight_value"),
    ("_maxbias", "use_max_bias"),
    ("_bestlogit", "use_assigned_logit"),
    ("_maskassign", "use_mask_assign"),
    ("_bf16kv", ("use_bf16_merged_key", "use_bf16_merged_value")),
    ("_bf16k", "use_bf16_merged_key"),
    ("_bf16v", "use_bf16_merged_value"),
    ("_bf16bias", "use_bf16_routing_bias"),
)
_INDIRECT_KEY_FLAGS = ("use_indirect_top_key", "use_indirect_top_kv")
_ORIGINAL_VALUE_FLAGS = (
    "use_indirect_top_kv",
    "use_indirect_top3_seed_value",
    "use_indirect_top3_value",
    "use_indirect_top2_seed_value",
    "use_indirect_top2_value",
    "use_indirect_top_value",
    "use_indirect_seed_value",
)
_INDIRECT_VALUE_FLAGS = (
    "use_indirect_top3_seed_value",
    "use_indirect_top3_value",
    "use_indirect_top2_seed_value",
    "use_indirect_top2_value",
    "use_indirect_top_value",
    "use_indirect_seed_value",
)
_WEIGHTED_INDIRECT_VALUE_FLAGS = (
    "use_indirect_top3_seed_value",
    "use_indirect_top3_value",
    "use_indirect_top2_seed_value",
    "use_indirect_top2_value",
)


@dataclass(frozen=True)
class DirectRouteMergeConfig:
    """Options encoded in one direct-route C4 backend string."""

    supported: bool
    inline_local_seed: bool = False
    block_j_target: int = 32
    block_l_target: int = 0
    key_outer_block: int = 1
    use_key_outer_kernel: bool = False
    use_assigned_logit: bool = False
    use_bf16_merged_key: bool = False
    use_bf16_merged_value: bool = False
    use_bf16_routing_bias: bool = False
    use_fp32_routing_bias: bool = False
    use_mask_assign: bool = False
    use_recompute_merge: bool = False
    use_recompute_value: bool = False
    use_merge_weight_value: bool = False
    use_stripe_assign: bool = False
    use_max_bias: bool = False
    use_seed_key: bool = False
    use_seed_key_bias_match: bool = False
    use_seed_key_scale: bool = False
    use_fp32_key_scale: bool = False
    use_seed_key_lsq_scale: bool = False
    use_seed_key_affine_scale: bool = False
    use_indirect_top3_seed_value: bool = False
    use_indirect_top3_value: bool = False
    use_indirect_top2_seed_value: bool = False
    use_indirect_top2_value: bool = False
    use_indirect_seed_value: bool = False
    use_indirect_top_value: bool = False
    use_indirect_top_kv: bool = False
    use_indirect_top_key: bool = False
    use_top_kv: bool = False
    use_top_key: bool = False
    use_top_key_scale: bool = False
    use_top_key_affine_scale: bool = False
    use_top3_seed_value: bool = False
    use_top3_value: bool = False
    use_top2_seed_value: bool = False
    use_top2_value: bool = False
    use_top_value: bool = False
    use_seed_value: bool = False

    def _has_any(self, flags):
        return any(getattr(self, flag) for flag in flags)

    @property
    def uses_indirect_key(self):
        return self._has_any(_INDIRECT_KEY_FLAGS)

    @property
    def uses_original_value(self):
        return self._has_any(_ORIGINAL_VALUE_FLAGS)

    @property
    def uses_indirect_value_attention(self):
        return self._has_any(_INDIRECT_VALUE_FLAGS)

    @property
    def uses_weighted_indirect_value(self):
        return self._has_any(_WEIGHTED_INDIRECT_VALUE_FLAGS)


def _remove_suffix(text, suffix):
    if text.endswith(suffix):
        return text[: -len(suffix)], True
    return text, False


def _set_option(options, field):
    if isinstance(field, tuple):
        for item in field:
            options[item] = True
    else:
        options[field] = True


def uses_inline_local_seed_backend(backend):
    return str(backend or "").lower().endswith("_inline_local_seed")


def _default_direct_route_options():
    return {
        field.name: field.default
        for field in fields(DirectRouteMergeConfig)
        if field.name not in {"supported", "inline_local_seed", "block_j_target"}
    }


@lru_cache(maxsize=64)
def parse_direct_route_attention_backend(backend):
    match = _DIRECT_ROUTE_ATTENTION_RE.fullmatch(backend or "")
    if match is None:
        return None

    block_m = int(match.group("m"))
    block_n = int(match.group("n"))
    num_warps = int(match.group("w"))
    num_stages = int(match.group("s"))
    group_block = int(match.group("g") or 1)
    if group_block not in {1, 2}:
        return None
    if block_m not in {16, 32}:
        return None
    if block_n not in {16, 32, 64}:
        return None
    if num_warps not in {1, 2}:
        return None
    if num_stages not in {1, 2, 3}:
        return None
    return block_m, block_n, num_warps, num_stages, group_block


def parse_direct_route_backend(backend):
    """Parse compact backend suffixes used by the direct Triton C4 fast path."""
    backend = (backend or "").lower()
    backend, inline_local_seed = _remove_suffix(backend, "_inline_local_seed")
    backend, token_major_output = _remove_suffix(backend, "_tokenmajor")

    options = _default_direct_route_options()

    while True:
        matched_any_suffix = False
        for suffix, block in _DIRECT_ROUTE_KEY_OUTER_BLOCKS.items():
            backend, matched = _remove_suffix(backend, f"_{suffix}")
            if matched:
                options["key_outer_block"] = block
                options["use_key_outer_kernel"] = True
                matched_any_suffix = True
                break

        if matched_any_suffix:
            continue

        match = _DIRECT_ROUTE_KEY_TILE_RE.fullmatch(backend)
        if match is not None:
            options["block_l_target"] = int(match.group("l"))
            backend = match.group("base")
            continue

        for suffix, field in _DIRECT_ROUTE_FLAG_SUFFIXES:
            backend, matched = _remove_suffix(backend, suffix)
            if matched:
                _set_option(options, field)
                matched_any_suffix = True
                break

        if not matched_any_suffix:
            break

    match = _DIRECT_ROUTE_C4_RE.fullmatch(backend)
    supported = token_major_output and match is not None
    block_j_target = int(match.group("j")) if match is not None else 32
    # Triton tl.arange requires power-of-two block sizes in this kernel.
    if block_j_target not in {16, 32, 64}:
        supported = False
    if options["block_l_target"] not in {0, 64, 128}:
        supported = False

    return DirectRouteMergeConfig(
        supported=supported,
        inline_local_seed=inline_local_seed,
        block_j_target=block_j_target,
        **options,
    )


def _parse_tokenmajor_attention_backend(backend):
    backend = (backend or "").lower()
    backend, token_major = _remove_suffix(backend, "_tokenmajor")
    if not token_major:
        return None
    return parse_direct_route_attention_backend(backend)


def merge_kv_direct_c4(q, seed_rows, key, value, config, scale, assume_contiguous=False):
    return triton_hard_route_merge_c4_direct_route(
        q,
        seed_rows,
        key,
        value,
        scale=scale,
        block_j_target=config.block_j_target,
        block_l_target=config.block_l_target,
        num_warps=2,
        bias_dtype=(
            torch.float32
            if config.use_fp32_routing_bias
            else (torch.bfloat16 if config.use_bf16_routing_bias else torch.float16)
        ),
        merged_key_dtype=torch.bfloat16 if config.use_bf16_merged_key else torch.float16,
        merged_value_dtype=torch.bfloat16 if config.use_bf16_merged_value else torch.float16,
        use_exp2=True,
        force_seed_keys=True,
        num_stages=1,
        logit_input_precision="tf32",
        inline_local_seed=config.inline_local_seed,
        key_outer_block=config.key_outer_block,
        use_key_outer_kernel=config.use_key_outer_kernel,
        use_assigned_logit=config.use_assigned_logit,
        use_mask_assign=config.use_mask_assign,
        use_recompute_merge=config.use_recompute_merge,
        use_recompute_value=config.use_recompute_value,
        use_merge_weight_value=config.use_merge_weight_value,
        use_stripe_assign=config.use_stripe_assign,
        use_max_bias=config.use_max_bias,
        use_seed_key=config.use_seed_key,
        use_seed_key_bias_match=config.use_seed_key_bias_match,
        use_seed_key_scale=config.use_seed_key_scale,
        use_seed_key_lsq_scale=config.use_seed_key_lsq_scale,
        use_seed_key_affine_scale=config.use_seed_key_affine_scale,
        use_indirect_top3_seed_value=config.use_indirect_top3_seed_value,
        use_indirect_top3_value=config.use_indirect_top3_value,
        use_indirect_top2_seed_value=config.use_indirect_top2_seed_value,
        use_indirect_top2_value=config.use_indirect_top2_value,
        use_indirect_seed_value=config.use_indirect_seed_value,
        use_indirect_top_value=config.use_indirect_top_value,
        use_indirect_top_kv=config.use_indirect_top_kv,
        use_indirect_top_key=config.use_indirect_top_key,
        use_top_kv=config.use_top_kv,
        use_top_key=config.use_top_key,
        use_top_key_scale=config.use_top_key_scale,
        use_top_key_affine_scale=config.use_top_key_affine_scale,
        use_top3_seed_value=config.use_top3_seed_value,
        use_top3_value=config.use_top3_value,
        use_top2_seed_value=config.use_top2_seed_value,
        use_top2_value=config.use_top2_value,
        use_top_value=config.use_top_value,
        use_seed_value=config.use_seed_value,
        key_scale_dtype=torch.float32 if config.use_fp32_key_scale else None,
        assume_contiguous=assume_contiguous and config.inline_local_seed,
    )


def direct_seed_key_attention(
    q,
    value,
    bias,
    key_scale,
    backend,
    scale,
    output=None,
    assume_contiguous=False,
):
    config = _parse_tokenmajor_attention_backend(backend)
    if config is None:
        return None

    block_m, block_n, num_warps, num_stages, group_block = config
    if group_block != 1:
        return None
    return triton_direct_route_seed_key_attention(
        q,
        value,
        bias,
        key_scale=key_scale,
        block_m=block_m,
        block_n=block_n,
        scale=scale,
        num_warps=num_warps,
        num_stages=num_stages,
        use_exp2=True,
        out=output,
        assume_contiguous=assume_contiguous,
    )


def direct_top_key_attention(
    q,
    source_key,
    source_value,
    key_index,
    value,
    bias,
    key_scale,
    backend,
    scale,
    indirect_seed_value=False,
    output=None,
    assume_contiguous=False,
):
    config = _parse_tokenmajor_attention_backend(backend)
    if config is None:
        return None

    block_m, block_n, num_warps, num_stages, group_block = config
    if group_block != 1:
        return None
    return triton_direct_route_top_key_attention(
        q,
        source_key,
        source_value,
        key_index,
        value,
        bias,
        key_scale=key_scale,
        block_m=block_m,
        block_n=block_n,
        scale=scale,
        num_warps=num_warps,
        num_stages=num_stages,
        use_exp2=True,
        indirect_seed_value=indirect_seed_value,
        out=output,
        assume_contiguous=assume_contiguous,
    )


def direct_indirect_value_attention(
    q,
    key,
    source_value,
    value_index,
    value_weight,
    bias,
    backend,
    scale,
    indirect_seed_value=False,
    indirect_top3_seed_value=False,
    indirect_top3_value=False,
    indirect_top2_value=False,
    indirect_top2_seed_value=False,
    output=None,
    assume_contiguous=False,
):
    config = _parse_tokenmajor_attention_backend(backend)
    if config is None:
        return None

    block_m, block_n, num_warps, num_stages, group_block = config
    if group_block != 1:
        return None
    return triton_direct_route_indirect_value_attention(
        q,
        key,
        source_value,
        value_index,
        value_weight,
        bias,
        block_m=block_m,
        block_n=block_n,
        scale=scale,
        num_warps=num_warps,
        num_stages=num_stages,
        use_exp2=True,
        indirect_seed_value=indirect_seed_value,
        indirect_top3_seed_value=indirect_top3_seed_value,
        indirect_top3_value=indirect_top3_value,
        indirect_top2_value=indirect_top2_value,
        indirect_top2_seed_value=indirect_top2_seed_value,
        out=output,
        assume_contiguous=assume_contiguous,
    )


def direct_recompute_value_attention(
    q,
    key,
    source_key,
    source_value,
    bias,
    backend,
    scale,
    output=None,
    assume_contiguous=False,
):
    config = _parse_tokenmajor_attention_backend(backend)
    if config is None:
        return None

    block_m, block_n, num_warps, num_stages, group_block = config
    if group_block != 1:
        return None
    return triton_direct_route_recompute_value_attention(
        q,
        key,
        source_key,
        source_value,
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


def direct_weight_value_attention(
    q,
    key,
    source_value,
    value_weight,
    bias,
    backend,
    scale,
    output=None,
    assume_contiguous=False,
):
    config = _parse_tokenmajor_attention_backend(backend)
    if config is None:
        return None

    block_m, block_n, num_warps, num_stages, group_block = config
    if group_block != 1:
        return None
    return triton_direct_route_weight_value_attention(
        q,
        key,
        source_value,
        value_weight,
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


def direct_compressed_attention(
    q,
    key,
    value,
    bias,
    backend,
    scale,
    output=None,
    assume_contiguous=False,
):
    config = _parse_tokenmajor_attention_backend(backend)
    if config is None:
        return None

    block_m, block_n, num_warps, num_stages, group_block = config
    if group_block == 2:
        return triton_direct_route_biased_attention_g2(
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


__all__ = [
    "DirectRouteMergeConfig",
    "direct_compressed_attention",
    "direct_indirect_value_attention",
    "direct_recompute_value_attention",
    "direct_seed_key_attention",
    "direct_top_key_attention",
    "direct_weight_value_attention",
    "merge_kv_direct_c4",
    "parse_direct_route_attention_backend",
    "parse_direct_route_backend",
    "uses_inline_local_seed_backend",
]
