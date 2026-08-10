import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from attention.hard_fused_triton import (  # noqa: E402
    HARD_C4_BACKEND,
    HARD_FUSED_BLOCK_M,
    HARD_FUSED_BLOCK_N,
    HARD_FUSED_INPUT_PRECISION,
    HARD_FUSED_NUM_STAGES,
    HARD_FUSED_NUM_WARPS,
)


def main():
    assert HARD_C4_BACKEND == "hard_c4"
    assert HARD_FUSED_BLOCK_M == 32
    assert HARD_FUSED_BLOCK_N == 64
    assert HARD_FUSED_INPUT_PRECISION == "ieee"
    assert HARD_FUSED_NUM_WARPS == 4
    assert HARD_FUSED_NUM_STAGES == 1
    print("hard_c4_backend_config_check_ok")


if __name__ == "__main__":
    main()
