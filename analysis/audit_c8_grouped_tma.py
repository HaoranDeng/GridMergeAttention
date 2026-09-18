"""Independent box/source geometry checks for v59 grouped TMA issuers."""
import argparse
import json
from pathlib import Path
from audit_c8_half_tma import audit as half_schedule_audit


def audit():
    assert half_schedule_audit()['passed']
    cases = []
    for warps in (2, 4):
        for halves in (False, True):
            storage = {}
            width = 64 if halves else 128
            for warp in range(4):
                group = warp // warps
                for part in range(2 if halves else 1):
                    for row in range(16):
                        for dim in range(width):
                            index = row * width + dim
                            swizzle = index ^ (((index >> 6) & 7) << 3)
                            base = (group * warps * 2048 + part * warps * 1024
                                    + (warp % warps) * 1024) if halves else warp * 2048
                            address = base + swizzle
                            assert address not in storage
                            storage[address] = (2 * warp + row // 8, row % 8,
                                                part * 64 + dim if halves else dim)
            assert set(storage) == set(range(8192))
            for first_warp in range(0, 4, warps):
                for part in range(2 if halves else 1):
                    base = first_warp * 2048 + (part * warps * 1024 if halves else 0)
                    # Flatten the tensor-map box, independently of the producer's
                    # per-warp row/part allocation. SW128 repeats every 1 KiB.
                    for column in range(warps * 2):
                        for seed in range(8):
                            for dim in range(width):
                                flat = (column * 8 + seed) * width + dim
                                swizzled = flat ^ (((flat >> 6) & 7) << 3)
                                actual = storage[base + swizzled]
                                expected = (first_warp * 2 + column, seed,
                                            part * 64 + dim if halves else dim)
                                assert actual == expected
                threads = [tid for tid in range(128) if (tid // 32) // warps == first_warp // warps]
                assert len(threads) == 32 * warps and first_warp * 32 in threads
            cases.append(dict(warps=warps, halves=halves, staging_bytes=16384,
                              store_instructions_per_query=(16 if halves else 8)//warps,
                              barrier_threads=warps*32))
    return dict(passed=True, cases=cases,
                scope='Source/box coverage and group membership; GPU correctness gates remain required.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = audit()
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
