"""Backend plumbing for the supported hard-route C4 attention path.

The readable algorithm lives in hard_route_pytorch.py. This file only keeps the
production Triton path that survived the experiments:

    exact best-logit C4 merge -> full compressed attention

Rejected representative-key/value, global/shared compression, no-bias, stripe,
merge-weight reconstruction, recompute-value, key-outer, and grouped-routing
paths are documented in docs/negative_results.md and are intentionally not
accepted by the parser here.
"""

from .hard_route_triton import (
    triton_direct_route_biased_attention,
    triton_hard_route_merge_c4_direct_route,
)


HARD_C4_BACKEND = "hard_c4"
DIRECT_ROUTING_BACKEND = "triton_direct_m32_n32_w1_s1_exp2_tokenmajor"
MERGE_BLOCK_J = 32
MERGE_NUM_WARPS = 2
MERGE_NUM_STAGES = 1
MERGE_LOGIT_INPUT_PRECISION = "ieee"
ROUTING_BLOCK_M = 32
ROUTING_BLOCK_N = 32
ROUTING_NUM_WARPS = 1
ROUTING_NUM_STAGES = 1


def is_direct_route_attention_backend(backend):
    """Return whether the requested routing backend is the supported full-attention kernel."""
    return (backend or "").lower() == DIRECT_ROUTING_BACKEND


def is_hard_c4_backend(backend):
    """Return whether the requested merge backend is the supported C4 path."""
    return (backend or "").lower() == HARD_C4_BACKEND


def merge_kv_direct_c4(q, key, value, scale, assume_contiguous=False):
    return triton_hard_route_merge_c4_direct_route(
        q,
        key,
        value,
        scale=scale,
        block_j_target=MERGE_BLOCK_J,
        num_warps=MERGE_NUM_WARPS,
        num_stages=MERGE_NUM_STAGES,
        logit_input_precision=MERGE_LOGIT_INPUT_PRECISION,
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
    if not is_direct_route_attention_backend(backend):
        raise ValueError(
            f"hard_route_backend='hard_c4' requires "
            f"hard_route_routing_backend={DIRECT_ROUTING_BACKEND!r}."
        )

    return triton_direct_route_biased_attention(
        q,
        key,
        value,
        bias,
        block_m=ROUTING_BLOCK_M,
        block_n=ROUTING_BLOCK_N,
        scale=scale,
        num_warps=ROUTING_NUM_WARPS,
        num_stages=ROUTING_NUM_STAGES,
        use_exp2=True,
        out=output,
        assume_contiguous=assume_contiguous,
    )


__all__ = [
    "DIRECT_ROUTING_BACKEND",
    "HARD_C4_BACKEND",
    "MERGE_BLOCK_J",
    "MERGE_LOGIT_INPUT_PRECISION",
    "MERGE_NUM_STAGES",
    "MERGE_NUM_WARPS",
    "ROUTING_BLOCK_M",
    "ROUTING_BLOCK_N",
    "ROUTING_NUM_STAGES",
    "ROUTING_NUM_WARPS",
    "direct_compressed_attention",
    "is_direct_route_attention_backend",
    "is_hard_c4_backend",
    "merge_kv_direct_c4",
]
