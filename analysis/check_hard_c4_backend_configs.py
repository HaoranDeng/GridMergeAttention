import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from attention.hard_route_triton import (  # noqa: E402
    HARD_C4_BACKEND,
    MERGE_BLOCK_J,
    MERGE_LOGIT_INPUT_PRECISION,
    MERGE_NUM_STAGES,
    MERGE_NUM_WARPS,
    ROUTING_BLOCK_M,
    ROUTING_BLOCK_N,
    ROUTING_NUM_STAGES,
    ROUTING_NUM_WARPS,
)


SUPPORTED_MERGE_BACKEND = HARD_C4_BACKEND

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
    "_bestlogit_simplemerge_ieee_tokenmajor_inline_local_seed",
    "_bestlogit_simplemerge_bf16kv_tokenmajor_inline_local_seed",
    "_bestlogit_simplemerge_bf16x3_tokenmajor_inline_local_seed",
]
REMOVED_MERGE_BACKENDS = [REMOVED_MERGE_PREFIX + suffix for suffix in REMOVED_MERGE_SUFFIXES]


def main():
    assert SUPPORTED_MERGE_BACKEND == "hard_c4"
    assert MERGE_BLOCK_J == 32
    assert MERGE_NUM_WARPS == 2
    assert MERGE_NUM_STAGES == 1
    assert MERGE_LOGIT_INPUT_PRECISION == "ieee"

    assert ROUTING_BLOCK_M == 32
    assert ROUTING_BLOCK_N == 32
    assert ROUTING_NUM_WARPS == 1
    assert ROUTING_NUM_STAGES == 1

    for backend in REMOVED_MERGE_BACKENDS:
        if backend == SUPPORTED_MERGE_BACKEND:
            raise AssertionError(f"removed merge backend is still supported: {backend}")

    print("hard_c4_backend_config_check_ok")


if __name__ == "__main__":
    main()
