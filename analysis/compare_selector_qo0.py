"""Quantify the output delta from fixing selection to a single reference
query-outer frame (qo=0) instead of recomputing per (qo, ko), by running the
full merged_attention pipeline with the original vs modified selector source
on the same real captured QKV, merge/route held fixed.
"""
import argparse
import ctypes
import hashlib
import os
import statistics
import time
from pathlib import Path

import torch

ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libstdc++.so.6', mode=ctypes.RTLD_GLOBAL)
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '9.0a')
os.environ.setdefault('MAX_JOBS', '1')

CAPTURES = Path('/local2/denghaoran/GridMergeAttention/experiments/full40_bf16_dense_capture_20260905bb/captures')


def compiled_selector(source_path, o, r, l, h, ah, warps=4):
    from torch.utils.cpp_extension import load
    flags = ['-O3', '--expt-relaxed-constexpr', '-lineinfo', f'-DAXIS_OUTER={o}', f'-DAXIS_ROWS={r}',
             f'-DAXIS_COLS={l}', f'-DAXIS_HEADS={h}', f'-DAXIS_ACTIVE_HEADS={ah}',
             '-DAXIS_C8_R2=1', f'-DAXIS_SELECTOR_WARPS={warps}']
    digest = hashlib.sha256(Path(source_path).read_bytes() + ' '.join(flags).encode()).hexdigest()[:16]
    cutlass = Path(torch.__file__).resolve().parent.parent / 'flashinfer/data/cutlass/include'
    return load(name=f'axis_c8_sel_{digest}', sources=[str(source_path)],
                extra_include_paths=[str(cutlass)], extra_cuda_cflags=flags,
                extra_ldflags=['-lcuda'], verbose=False)


def run_full(selector_mod, merge_mod, q, k, v, o, r, l, h, chunk=4):
    out = torch.empty_like(q)
    packed = torch.empty_like(q)
    merge_mod.pack(q, packed)
    candidates_anchors_cache = {}
    from analysis.quick_axis_c4_qkv import candidate_table
    table, anchors = candidate_table(r)
    table = [row + [0] * (32 - len(row)) for row in table]
    candidates = torch.tensor(table, device=q.device, dtype=torch.int32)
    anchors_t = torch.tensor(anchors, device=q.device, dtype=torch.int32)
    shape = (1, o, l, chunk, o, r, 8)
    seeds = torch.empty(shape, device=q.device, dtype=torch.uint8)
    mk = torch.empty((*shape, 128), device=q.device, dtype=torch.bfloat16)
    mv = torch.empty_like(mk)
    bias = torch.empty(shape, device=q.device, dtype=torch.float32)
    empty = torch.empty(0, device=q.device, dtype=torch.float32)
    scale = 128 ** -.5
    for hs in range(0, h, chunk):
        selector_mod.run(q, k, candidates, anchors_t, seeds, hs, scale)
        merge_mod.run(packed, k, v, seeds, mk, mv, bias, hs, chunk, 21, scale, empty, 0, 1)
        import triton
        from kernels.best.c4.route_reference import _hard_c4_720p_route_kernel
        _hard_c4_720p_route_kernel[(1, 1 * chunk, o * l)](
            q, k, v, mk, mv, bias, bias, bias, bias, bias, out,
            Q_OUTER=o, K_OUTER=o, TOTAL_HEADS=h, ACTIVE_HEADS=chunk, HEAD_START=hs,
            D_HEAD=128, HEAD_MAJOR=False, OUTPUT_HEAD_MAJOR=False, ROWS=r, COLS=l, C=8,
            BLOCK_I=triton.next_power_of_2(r), BLOCK_TOKENS=64,
            LOOP_STAGES=2, INPUT_PRECISION='ieee', EXACT_INTRA_FRAME=False, EXACT_NEIGHBOR_WINDOW_COLS=0,
            RECORD_DISTORTION=False, QUERY_OUTER_OFFSET=0, SINGLETON_TOP_R=0,
            SINGLETON_QUERY_COLUMN=False, SINGLETON_SECONDARY_FRAME_STRIDE=1,
            SINGLETON_PRIMARY_ROW_MOD4_OMIT2=False, QUERY_COLUMN_MAJOR=False,
            PACKED_SINGLETON_KV=False, SCALED_FP8_KV=False, COMPRESSED_FP8=False,
            SCALE=scale, num_warps=4, num_stages=2)
    return out


def main():
    torch.set_grad_enabled(False)
    payload = torch.load(CAPTURES / 'prompt000_call000_layer0.pt', map_location='cpu', mmap=True, weights_only=True)
    q, k, v = [payload[name].permute(0, 2, 1, 3).to('cuda').contiguous() for name in ('q', 'k', 'v')]
    del payload
    b, h, n, d = q.shape
    grid = (21, 45, 80)
    order = (0, 2, 4, 3, 1, 5)
    relaid = [x.reshape(b, h, *grid, d).permute(order).contiguous() for x in (q, k, v)]
    o, r, l = grid[0], grid[2], grid[1]
    chunk = 4

    import attention.axis_current_c8_cuda as m
    merge_mod, = [_compiled_pair[1] for _compiled_pair in [m._compiled(o, r, l, h, chunk, 'r2')]]

    sel_orig = compiled_selector('/local2/denghaoran/GridMergeAttention_axis_c8_opt/attention/axis_current_c8_selector_original.cu', o, r, l, h, chunk)
    sel_new = compiled_selector('/local2/denghaoran/GridMergeAttention_axis_c8_opt/attention/axis_current_c8_selector.cu', o, r, l, h, chunk)

    out_orig = run_full(sel_orig, merge_mod, *relaid, o, r, l, h, chunk)
    out_new = run_full(sel_new, merge_mod, *relaid, o, r, l, h, chunk)
    torch.cuda.synchronize()

    diff = (out_new.float() - out_orig.float())
    rel_l2 = diff.norm() / out_orig.float().norm()
    max_abs = diff.abs().max()
    print(f'relative_L2(new_qo0_vs_original) = {rel_l2.item():.6%}')
    print(f'max_abs_diff = {max_abs.item():.6f}')
    print(f'finite: orig={torch.isfinite(out_orig).all().item()} new={torch.isfinite(out_new).all().item()}')


if __name__ == '__main__':
    main()
