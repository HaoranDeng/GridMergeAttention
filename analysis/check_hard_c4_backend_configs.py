import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from attention.hard_route_backend import (  # noqa: E402
    DIRECT_ROUTING_BACKEND,
    HARD_C4_BACKEND,
    LEGACY_HARD_C4_BACKEND,
    parse_direct_route_attention_backend,
    parse_direct_route_backend,
)


SUPPORTED_MERGE_BACKEND = HARD_C4_BACKEND
SUPPORTED_ROUTING_BACKEND = DIRECT_ROUTING_BACKEND

REMOVED_MERGE_PREFIX = "triton_c4_direct_route_j32_w2_exp2merge_forcekeys_fp16kvbias_s1_fastdot"
REMOVED_MERGE_SUFFIXES = [
    "_bestlogit_tokenmajor_inline_local_seed",
    "_bestlogit_recomputevalue_tokenmajor_inline_local_seed",
    "_bestlogit_mergeweights_tokenmajor_inline_local_seed",
    "_bestlogit_compactweights_tokenmajor_inline_local_seed",
    "_bestlogit_topkey_tokenmajor_inline_local_seed",
    "_bestlogit_topvalue_tokenmajor_inline_local_seed",
    "_seedkey_bestlogit_tokenmajor_inline_local_seed",
    "_bestlogit_maxbias_tokenmajor_inline_local_seed",
    "_stripe_tokenmajor_inline_local_seed",
    "_f2_bestlogit_tokenmajor_inline_local_seed",
    "_bestlogit_simplemerge_tokenmajor_inline_local_seed",
    "_bestlogit_simplemerge_bf16kv_tokenmajor_inline_local_seed",
    "_bestlogit_simplemerge_bf16x3_tokenmajor_inline_local_seed",
]
REMOVED_MERGE_BACKENDS = [REMOVED_MERGE_PREFIX + suffix for suffix in REMOVED_MERGE_SUFFIXES]

REMOVED_ROUTING_BACKENDS = [
    "triton_direct_g2_m32_n16_w1_s1_exp2_tokenmajor",
    "triton_direct_m32_n16_w1_s1_exp2_tokenmajor",
    "triton_direct_m16_n64_w1_s1_exp2_tokenmajor",
]


def main():
    config = parse_direct_route_backend(SUPPORTED_MERGE_BACKEND)
    assert config.supported
    assert config.block_j_target == 32
    assert config.merge_num_warps == 2
    assert config.merge_num_stages == 1
    assert config.logit_input_precision == "ieee"
    assert parse_direct_route_backend(LEGACY_HARD_C4_BACKEND).supported

    assert parse_direct_route_attention_backend(SUPPORTED_ROUTING_BACKEND) == (32, 32, 1, 1)

    for backend in REMOVED_MERGE_BACKENDS:
        if parse_direct_route_backend(backend).supported:
            raise AssertionError(f"removed merge backend is still supported: {backend}")

    for backend in REMOVED_ROUTING_BACKENDS:
        if parse_direct_route_attention_backend(backend) is not None:
            raise AssertionError(f"removed routing backend is still supported: {backend}")

    print("hard_c4_backend_config_check_ok")


if __name__ == "__main__":
    main()
