"""Verify v60's storing-lane operand mapping independently of its CUDA code."""
import argparse
import json
from pathlib import Path


def audit():
    original, grouped = {}, {}
    for warp in range(4):
        for lane in range(32):
            if lane % 4 == 0:
                for half in (0, 1):
                    row = 16 * warp + lane // 4 + 8 * half
                    column, seed = 2 * (row // 16) + row % 2, (row % 16) // 2
                    # Dot comes from the original storing lane. The reciprocal
                    # and log come from the alternating lane via broadcast.
                    original[column, seed] = (warp, lane, half, lane + half)
            if lane % 4 < 2:
                half = lane % 2
                row = 16 * warp + lane // 4 + 8 * half
                column, seed = 2 * (row // 16) + row % 2, (row % 16) // 2
                assert (column, seed) not in grouped
                grouped[column, seed] = (warp, lane - half, half, lane)
    assert original == grouped
    assert set(grouped) == {(column, seed) for column in range(8) for seed in range(8)}
    return dict(passed=True, unique_bias_outputs=64,
                scope='Exact address and dot/reciprocal/log source-lane mapping. GPU bitwise gates still required.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = audit()
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
