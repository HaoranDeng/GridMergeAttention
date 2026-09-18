"""Integer layout and worst-case pending-read model for v58 half stores."""
import argparse
import collections
import json
from pathlib import Path


def audit():
    # Logical output is [query, column, head, key_block, seed, dimension].
    # Compare descriptor addresses with that independent contiguous layout.
    writes = set()
    for warp in range(4):
        for part in range(2):
            for row in range(16):
                for dimension in range(64):
                    local = row * 64 + dimension
                    swizzled = local ^ (((local >> 6) & 7) << 3)
                    address = warp * 2048 + part * 1024 + swizzled
                    assert address not in writes
                    writes.add(address)
                    column, seed = 2 * warp + row // 8, row % 8
                    for query, group, head, key in ((0, 0, 0, 0), (20, 9, 4, 944)):
                        actual_column = group * 8 + column
                        expected = (((((query * 80 + actual_column) * 5 + head)
                                      * 945 + key) * 8 + seed) * 128
                                    + part * 64 + dimension)
                        descriptor = (part * 64 + dimension + 128 * seed
                                      + 1024 * key + 1024 * 945 * head
                                      + 1024 * 945 * 5 * (query * 80 + actual_column))
                        assert descriptor == expected
    assert writes == set(range(8192))
    # The row permutation used by stmatrix remains bijective inside each warp.
    for warp in range(4):
        rows = [((row >> 4) * 2 + (row & 1)) * 8 + ((row & 15) >> 1)
                - warp * 16 for row in range(warp * 16, warp * 16 + 16)]
        assert sorted(rows) == list(range(16))

    cases = []
    for allowed in (0, 1):
        for frames in (1, 2, 21):
            for spans in (1, 2, 10):
                pending = []
                issued = []
                maximum = 0

                def stage(span, query, kind, part):
                    nonlocal pending, maximum
                    # Assume no spontaneous completion; only wait guarantees
                    # retire old groups. Retain the newest `allowed` groups.
                    pending = pending[-allowed:] if allowed else []
                    assert all(group[3] != part for group in pending), pending
                    item = (span, query, kind, part)
                    pending.append(item)
                    issued.append(item)
                    maximum = max(maximum, len(pending))

                for span in range(spans):
                    for query in range(frames):
                        if query:
                            stage(span, query - 1, 'V', 1)
                        stage(span, query, 'K', 0)
                        stage(span, query, 'K', 1)
                        stage(span, query, 'V', 0)
                    stage(span, frames - 1, 'V', 1)
                pending = []  # kernel's final wait_group.read 0
                expected = [(s, q, kind, part) for s in range(spans)
                            for q in range(frames) for kind in ('K', 'V')
                            for part in (0, 1)]
                assert collections.Counter(issued) == collections.Counter(expected)
                cases.append(dict(allowed=allowed, frames=frames, spans=spans,
                                  groups=len(issued), maximum_pending=maximum))
    return dict(passed=True, staging_bytes=16384, dynamic_shared_bytes=57344,
                cases=cases, scope='Integer geometry and FIFO reuse only; GPU bitwise gates required.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = audit()
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
