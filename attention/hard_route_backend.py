"""Backend plumbing for the supported hard-route C4 attention path.

The readable algorithm lives in hard_route_pytorch.py. This file only keeps the
production Triton path that survived the experiments:

    exact best-logit C4 merge -> full compressed attention

Rejected representative-key/value, global/shared compression, no-bias, stripe,
merge-weight reconstruction, recompute-value, key-outer, and grouped-routing
paths are documented in docs/negative_results.md and are intentionally not
accepted by the parser here.
"""

from dataclasses import dataclass
from functools import lru_cache

from .hard_route_triton import (
    triton_direct_route_biased_attention,
    triton_hard_route_merge_c4_direct_route,
)


HARD_C4_BACKEND = "hard_c4"
LEGACY_HARD_C4_BACKEND = (
    "triton_c4_direct_route_j32_w2_exp2merge_forcekeys_fp16kvbias_s1_fastdot"
    "_bestlogit_simplemerge_ieee_tokenmajor_inline_local_seed"
)
DIRECT_ROUTING_BACKEND = "triton_direct_m32_n32_w1_s1_exp2_tokenmajor"


@dataclass(frozen=True)
class DirectRouteMergeConfig:
    """Parsed options for the single supported direct-route C4 backend."""

    supported: bool
    block_j_target: int = 32
    merge_num_warps: int = 2
    merge_num_stages: int = 1
    logit_input_precision: str = "ieee"


@lru_cache(maxsize=16)
def parse_direct_route_attention_backend(backend):
    """Return the only supported full compressed-attention tile config."""
    if (backend or "").lower() != DIRECT_ROUTING_BACKEND:
        return None
    return 32, 32, 1, 1


@lru_cache(maxsize=16)
def parse_direct_route_backend(backend):
    """Parse the direct C4 merge backend.

    Unsupported historical experiment suffixes deliberately return
    supported=False instead of exposing half-dead code paths.
    """
    backend = (backend or "").lower()
    return DirectRouteMergeConfig(supported=backend in {HARD_C4_BACKEND, LEGACY_HARD_C4_BACKEND})


def merge_kv_direct_c4(q, key, value, config, scale, assume_contiguous=False):
    if not config.supported:
        raise ValueError("unsupported hard-route C4 backend; use hard_route_backend='hard_c4'.")

    return triton_hard_route_merge_c4_direct_route(
        q,
        key,
        value,
        scale=scale,
        block_j_target=config.block_j_target,
        num_warps=config.merge_num_warps,
        num_stages=config.merge_num_stages,
        logit_input_precision=config.logit_input_precision,
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


__all__ = [
    "DirectRouteMergeConfig",
    "DIRECT_ROUTING_BACKEND",
    "HARD_C4_BACKEND",
    "LEGACY_HARD_C4_BACKEND",
    "direct_compressed_attention",
    "merge_kv_direct_c4",
    "parse_direct_route_attention_backend",
    "parse_direct_route_backend",
]
