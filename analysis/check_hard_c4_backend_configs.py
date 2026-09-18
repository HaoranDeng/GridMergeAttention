import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from attention.hard_c4_wan21_1_3b_480p_triton import (  # noqa: E402
    HARD_C4_BACKEND,
    HARD_RT_BLOCK_I,
    HARD_RT_BLOCK_J,
    HARD_RT_BLOCK_K,
    HARD_RT_BLOCK_L,
    HARD_RT_C,
    HARD_RT_INPUT_PRECISION,
    HARD_RT_MERGE_NUM_STAGES,
    HARD_RT_MERGE_NUM_WARPS,
    HARD_RT_ROUTE_NUM_STAGES,
    HARD_RT_ROUTE_NUM_WARPS,
)


def main():
    assert HARD_C4_BACKEND == "hard_c4"
    assert HARD_RT_BLOCK_I == 32
    assert HARD_RT_BLOCK_J == 64
    assert HARD_RT_BLOCK_K == 32
    assert HARD_RT_BLOCK_L == 64
    assert HARD_RT_C == 4
    assert HARD_RT_INPUT_PRECISION == "ieee"
    assert HARD_RT_MERGE_NUM_WARPS == 4
    assert HARD_RT_ROUTE_NUM_WARPS == 4
    assert HARD_RT_MERGE_NUM_STAGES == 2
    assert HARD_RT_ROUTE_NUM_STAGES == 2
    print("hard_c4_backend_config_check_ok")


if __name__ == "__main__":
    main()
