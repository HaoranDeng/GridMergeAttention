"""Tuning-only sweep over merge tile count and route block/warp/stage knobs,
no algorithm change. Reuses benchmark_axis_c8.py's single-capture measurement.
"""
import json
import os
import subprocess
import sys

REPO = '/local2/denghaoran/GridMergeAttention_axis_c8_opt'


def run(env_overrides):
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_overrides.items()})
    env['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libstdc++.so.6'
    env['PYTHONPATH'] = '.'
    out = subprocess.run(
        [sys.executable, 'analysis/benchmark_axis_c8.py', '--output', '/tmp/sweep_out.json',
         '--single', '--repeats', '5'],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=180)
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith('{') and 'method_median_ms' in line:
            return json.loads(line)['method_median_ms']
    print(out.stdout[-2000:], file=sys.stderr)
    print(out.stderr[-2000:], file=sys.stderr)
    return None


def main():
    print('--- tiles sweep ---', flush=True)
    tile_results = {}
    for tiles in (21, 42, 63, 105, 126, 21 * 3):
        ms = run({'AXIS_C8_CUDA_TILES': tiles})
        tile_results[tiles] = ms
        print(f'tiles={tiles:4d}  median_ms={ms}', flush=True)
    best_tiles = min(tile_results, key=lambda k: tile_results[k] if tile_results[k] else 1e9)

    print(f'--- route sweep (tiles={best_tiles}) ---', flush=True)
    route_results = {}
    for tokens in (32, 64, 128):
        for warps in (2, 4, 8):
            for stages in (1, 2, 3):
                key = (tokens, warps, stages)
                ms = run({'AXIS_C8_CUDA_TILES': best_tiles, 'AXIS_C8_ROUTE_TOKENS': tokens,
                          'AXIS_C8_ROUTE_WARPS': warps, 'AXIS_C8_ROUTE_STAGES': stages})
                route_results[key] = ms
                print(f'tokens={tokens:3d} warps={warps} stages={stages}  median_ms={ms}', flush=True)
    best_route = min(route_results, key=lambda k: route_results[k] if route_results[k] else 1e9)
    print(f'BEST: tiles={best_tiles} route={best_route} -> {route_results[best_route]} ms', flush=True)


if __name__ == '__main__':
    main()
