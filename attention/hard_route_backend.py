import re
from dataclasses import dataclass
from functools import lru_cache

import torch

from .hard_route_triton import (
    triton_direct_route_biased_attention,
    triton_direct_route_biased_attention_packed_kv,
    triton_hard_route_merge_c4_direct_route,
)


_DIRECT_ROUTE_C4_BASE = "triton_c4_direct_route_j32_w2_exp2merge_forcekeys_fp16kvbias_s1_fastdot"
_DIRECT_ROUTE_ATTENTION_RE = re.compile(
    r"^triton_direct_m(?P<m>\d+)_n(?P<n>\d+)_w(?P<w>\d+)_s(?P<s>\d+)_exp2$"
)

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
    ("_bestlogit", "use_assigned_logit"),
    ("_maskassign", "use_mask_assign"),
    ("_packkv", "use_packed_kv"),
    ("_bf16kv", ("use_bf16_merged_key", "use_bf16_merged_value")),
    ("_bf16k", "use_bf16_merged_key"),
    ("_bf16v", "use_bf16_merged_value"),
    ("_bf16bias", "use_bf16_routing_bias"),
)


@dataclass(frozen=True)
class DirectRouteMergeConfig:
    """Options encoded in one direct-route C4 backend string."""

    supported: bool
    inline_local_seed: bool = False
    key_outer_block: int = 1
    use_key_outer_kernel: bool = False
    use_assigned_logit: bool = False
    use_bf16_merged_key: bool = False
    use_bf16_merged_value: bool = False
    use_bf16_routing_bias: bool = False
    use_mask_assign: bool = False
    use_packed_kv: bool = False


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


@lru_cache(maxsize=64)
def parse_direct_route_attention_backend(backend):
    match = _DIRECT_ROUTE_ATTENTION_RE.fullmatch(backend or "")
    if match is None:
        return None

    block_m = int(match.group("m"))
    block_n = int(match.group("n"))
    num_warps = int(match.group("w"))
    num_stages = int(match.group("s"))
    if block_m not in {16, 32}:
        return None
    if block_n not in {16, 32, 64}:
        return None
    if num_warps not in {1, 2}:
        return None
    if num_stages not in {1, 2, 3}:
        return None
    return block_m, block_n, num_warps, num_stages


def parse_direct_route_backend(backend):
    """Parse compact backend suffixes used by the direct Triton C4 fast path."""
    backend = (backend or "").lower()
    backend, inline_local_seed = _remove_suffix(backend, "_inline_local_seed")
    backend, token_major_output = _remove_suffix(backend, "_tokenmajor")

    options = {
        "key_outer_block": 1,
        "use_key_outer_kernel": False,
        "use_assigned_logit": False,
        "use_bf16_merged_key": False,
        "use_bf16_merged_value": False,
        "use_bf16_routing_bias": False,
        "use_mask_assign": False,
        "use_packed_kv": False,
    }

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

        for suffix, field in _DIRECT_ROUTE_FLAG_SUFFIXES:
            backend, matched = _remove_suffix(backend, suffix)
            if matched:
                _set_option(options, field)
                matched_any_suffix = True
                break

        if not matched_any_suffix:
            break

    return DirectRouteMergeConfig(
        supported=token_major_output and backend == _DIRECT_ROUTE_C4_BASE,
        inline_local_seed=inline_local_seed,
        **options,
    )


def merge_kv_direct_c4(q, seed_rows, key, value, config, scale, assume_contiguous=False):
    return triton_hard_route_merge_c4_direct_route(
        q,
        seed_rows,
        key,
        value,
        scale=scale,
        block_j_target=32,
        num_warps=2,
        bias_dtype=torch.bfloat16 if config.use_bf16_routing_bias else torch.float16,
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
        pack_kv=config.use_packed_kv,
        assume_contiguous=assume_contiguous and config.inline_local_seed,
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
    backend = (backend or "").lower()
    backend, token_major = _remove_suffix(backend, "_tokenmajor")
    if not token_major:
        return None

    config = parse_direct_route_attention_backend(backend)
    if config is None:
        return None

    block_m, block_n, num_warps, num_stages = config
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


def direct_compressed_attention_packed_kv(
    q,
    packed_kv,
    bias,
    backend,
    scale,
    output=None,
    assume_contiguous=False,
):
    backend = (backend or "").lower()
    backend, token_major = _remove_suffix(backend, "_tokenmajor")
    if not token_major:
        return None

    config = parse_direct_route_attention_backend(backend)
    if config is None:
        return None

    block_m, block_n, num_warps, num_stages = config
    return triton_direct_route_biased_attention_packed_kv(
        q,
        packed_kv,
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
    "direct_compressed_attention_packed_kv",
    "merge_kv_direct_c4",
    "parse_direct_route_attention_backend",
    "parse_direct_route_backend",
    "uses_inline_local_seed_backend",
]
