"""Backend plumbing for the supported hard-route C4 attention path.

The readable algorithm lives in hard_route_pytorch.py. This file only keeps the
production Triton path that survived the experiments:

    exact best-logit C4 merge -> full compressed attention

Rejected representative-key/value, global/shared compression, no-bias, stripe,
merge-weight reconstruction, recompute-value, key-outer, and grouped-routing
paths are documented in docs/negative_results.md and are intentionally not
accepted by the parser here.
"""

import re
from dataclasses import dataclass
from functools import lru_cache

import torch

from .hard_route_triton import (
    triton_direct_route_biased_attention,
    triton_hard_route_merge_c4_direct_route,
)


_SUPPORTED_MERGE_RE = re.compile(
    r"^triton_c4_direct_route_j32_w2_exp2merge_forcekeys_fp16kvbias_s1_fastdot"
    r"_bestlogit_simplemerge_ieee_tokenmajor_inline_local_seed$"
)
_SUPPORTED_ROUTING_BACKEND = re.compile(
    r"^triton_direct_m32_n32_w1_s1_exp2_tokenmajor$"
)


@dataclass(frozen=True)
class DirectRouteMergeConfig:
    """Parsed options for the single supported direct-route C4 backend."""

    supported: bool
    inline_local_seed: bool = False
    block_j_target: int = 32
    merge_num_warps: int = 2
    merge_num_stages: int = 1
    logit_input_precision: str = "tf32"
    use_assigned_logit: bool = False
    use_simple_merge: bool = False


def uses_inline_local_seed_backend(backend):
    return str(backend or "").lower().endswith("_inline_local_seed")


@lru_cache(maxsize=16)
def parse_direct_route_attention_backend(backend):
    """Return the only supported full compressed-attention tile config."""
    if _SUPPORTED_ROUTING_BACKEND.fullmatch(backend or "") is None:
        return None
    return 32, 32, 1, 1


@lru_cache(maxsize=16)
def parse_direct_route_backend(backend):
    """Parse the direct C4 merge backend.

    Unsupported historical experiment suffixes deliberately return
    supported=False instead of exposing half-dead code paths.
    """
    supported = _SUPPORTED_MERGE_RE.fullmatch((backend or "").lower()) is not None
    return DirectRouteMergeConfig(
        supported=supported,
        inline_local_seed=supported,
        logit_input_precision="ieee" if supported else "tf32",
        use_assigned_logit=supported,
        use_simple_merge=supported,
    )


def merge_kv_direct_c4(q, seed_rows, key, value, config, scale, assume_contiguous=False):
    if not config.supported:
        raise ValueError("unsupported hard-route C4 backend; use the documented bestlogit_simplemerge path.")

    return triton_hard_route_merge_c4_direct_route(
        q,
        seed_rows,
        key,
        value,
        scale=scale,
        block_j_target=config.block_j_target,
        num_warps=config.merge_num_warps,
        bias_dtype=torch.float16,
        merged_key_dtype=torch.float16,
        merged_value_dtype=torch.float16,
        use_exp2=True,
        force_seed_keys=True,
        num_stages=config.merge_num_stages,
        logit_input_precision=config.logit_input_precision,
        inline_local_seed=config.inline_local_seed,
        use_assigned_logit=config.use_assigned_logit,
        use_simple_merge=config.use_simple_merge,
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
    config = parse_direct_route_attention_backend((backend or "").lower())
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


def direct_route_only_feature(config):
    return None


def run_direct_route_attention(
    q,
    source_key,
    source_value,
    key,
    value,
    bias,
    key_scale,
    config,
    routing_backend,
    scale,
    output=None,
    assume_contiguous=False,
):
    del source_key, source_value, key_scale, config
    return direct_compressed_attention(
        q,
        key,
        value,
        bias,
        routing_backend,
        scale,
        output=output,
        assume_contiguous=assume_contiguous,
    )


__all__ = [
    "DirectRouteMergeConfig",
    "direct_compressed_attention",
    "direct_route_only_feature",
    "merge_kv_direct_c4",
    "parse_direct_route_attention_backend",
    "parse_direct_route_backend",
    "run_direct_route_attention",
    "uses_inline_local_seed_backend",
]
