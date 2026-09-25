"""Sweep the C4 Triton route schedule and anchor the real streaming-read ceiling.

No route configuration sweep was ever recorded for C4: block 64 / warps 4 /
stages 3 was inherited. The route is a third of the call, so this screens the
schedule directly. `route_stages` is a pipelining depth and cannot change
arithmetic; `route_block` and `route_warps` change the online-softmax tile and
the reduction width, so they may change results. Every arm is gated on bitwise
equality of the complete H40 output against the retained configuration, and
non-equal arms are reported with their error but never promoted.

The bandwidth anchor is a real grid-stride vectorised read, not a torch
reduction, because a reduction understates the achievable streaming ceiling.
"""
import argparse, json, os, statistics, sys, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.launch_quality_first_videos import require_clean, gpu_processes, now

CAPTURE = '/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer{}.pt'
RETAINED = ROOT / 'kernels/best/c4/fisher_c4_guarded_softmax.cu'
SELECTOR = ROOT / 'kernels/best/c4/fisher_c4_fused_selector_register_basis_loop.cu'
BASE = (64, 4, 3)

READ_KERNEL = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void stream_read(const uint4* __restrict__ p, size_t n, uint4* sink) {
  uint4 acc = make_uint4(0, 0, 0, 0);
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = p[i];
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  // Never true at runtime, but the compiler cannot prove it, so the loads stay.
  if (acc.x == 0xdeadbeefu && acc.y == 0x1234u) *sink = acc;
}
void run(torch::Tensor a, torch::Tensor sink) {
  size_t n = a.numel() * a.element_size() / sizeof(uint4);
  stream_read<<<8192, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const uint4*)a.data_ptr(), n, (uint4*)sink.data_ptr());
}
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--layer', action='append', default=[])
    p.add_argument('--blocks', type=int, default=8)
    args = p.parse_args()
    layers = args.layer or ['00']
    assert not args.output.exists()
    initial = require_clean(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = initial['gpu']['uuid']
    os.environ['TORCH_EXTENSIONS_DIR'] = str(args.output.parent / 'torch_extensions')
    os.environ['TRITON_CACHE_DIR'] = str(args.output.parent / 'triton_cache')
    os.environ['OMP_NUM_THREADS'] = '4'
    args.output.parent.mkdir(parents=True, exist_ok=True)

    stop = threading.Event(); errors = []; samples = []; query_errors = []

    def monitor():
        while not stop.is_set():
            try:
                ps = gpu_processes(initial['gpu']['uuid'])
                samples.append({'utc': now(), 'processes': ps})
                if any(r['pid'] != os.getpid() for r in ps):
                    errors.append('foreign GPU process')
            except Exception as exc:
                query_errors.append(repr(exc))
            stop.wait(3)

    threading.Thread(target=monitor, daemon=True).start()
    result = {'initial_gpu': initial, 'layers': layers, 'base_config': BASE,
              'scope': 'C4 Triton route schedule sweep plus a real streaming-read '
                       'bandwidth anchor. Non-bitwise arms are recorded, never promoted.'}

    def save():
        result.update(monitor_errors=errors, monitor_query_errors=query_errors,
                      monitor_samples=samples)
        args.output.write_text(json.dumps(result, indent=2))

    try:
        import torch
        from torch.utils.cpp_extension import load_inline
        from kernels.best.c4.operator import NativeFisherAttention
        torch.set_grad_enabled(False); torch.set_num_threads(4)

        def timed(fn, repeats=5):
            for _ in range(2):
                fn()
            torch.cuda.synchronize()
            values = []
            for _ in range(repeats):
                assert not errors, errors
                s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                s.record(); y = fn(); e.record(); e.synchronize()
                values.append(s.elapsed_time(e)); del y
            return values

        # --- real streaming-read ceiling -----------------------------------
        reader = load_inline(name='c4_stream_read',
                             cpp_sources='void run(torch::Tensor a, torch::Tensor sink);',
                             cuda_sources=READ_KERNEL, functions=['run'],
                             extra_cuda_cflags=['-O3'], verbose=False)
        big = torch.empty(4 << 30, dtype=torch.uint8, device='cuda')
        big.fill_(7)
        sink = torch.zeros(4, dtype=torch.int32, device='cuda')
        vals = timed(lambda: reader.run(big, sink), repeats=10)
        best = min(vals)
        result['read_anchor'] = {
            'bytes': big.numel(), 'mean_ms': statistics.fmean(vals), 'best_ms': best,
            'mean_tb_s': big.numel() / statistics.fmean(vals) / 1e9,
            'best_tb_s': big.numel() / best / 1e9}
        print('READ_ANCHOR', json.dumps(result['read_anchor']), flush=True)
        del big, sink
        torch.cuda.empty_cache()
        save()

        op = NativeFisherAttention(device='cuda', capacity=4,
                                   selector_backend='fused_cooperative_loop',
                                   selector_source=SELECTOR, merge_source=RETAINED)
        scale = 128 ** -.5
        configs = [BASE, (64, 4, 1), (64, 4, 2), (64, 4, 4), (64, 8, 3),
                   (128, 4, 3), (128, 4, 1), (128, 4, 2), (128, 8, 3),
                   (256, 4, 1), (256, 4, 2), (32, 4, 3)]

        def use(cfg):
            op.route_block, op.route_warps, op.route_stages = cfg

        result['per_layer'] = {}
        for layer in layers:
            payload = torch.load(CAPTURE.format(layer), map_location='cpu', weights_only=True)
            q, k, v = [payload[n].reshape(1, 21, 45, 80, 40, 128).cuda() for n in ('q', 'k', 'v')]
            del payload
            row = {}
            result['per_layer'][layer] = row
            use(BASE)
            reference = op(q, k, v).clone()
            out = torch.empty_like(q)
            seeds = op.select(q, k, scale)
            op.merge.pack(q, op.packed)
            op.merge.run(op.packed, k, v, seeds, op.mk, op.mv, op.bias, 0, 5,
                         op.tiles, scale, op.empty, 0, 1)
            torch.cuda.synchronize()

            for cfg in configs:
                name = f'b{cfg[0]}_w{cfg[1]}_s{cfg[2]}'
                entry = {'config': list(cfg)}
                try:
                    use(cfg)
                    produced = op(q, k, v).clone()
                    difference = produced.float() - reference.float()
                    entry['correctness'] = {
                        'bitwise_equal': bool(torch.equal(produced, reference)),
                        'max_abs': float(difference.abs().max()),
                        'relative_l2': float((difference.square().sum()
                                              / reference.float().square().sum()).sqrt()),
                        'finite': bool(torch.isfinite(produced).all())}
                    del produced, difference
                    # Route stage cost on its own, five heads.
                    entry['route_h5_ms'] = statistics.fmean(
                        timed(lambda: op.route(q, k, v, out, 0, scale, active_heads=5)))
                except Exception as exc:
                    entry['error'] = repr(exc)
                    print('CONFIG', layer, name, 'FAILED', repr(exc)[:120], flush=True)
                    row[name] = entry; save(); continue
                print('CONFIG', layer, name, 'route_h5', round(entry['route_h5_ms'], 4),
                      'bitwise', entry['correctness']['bitwise_equal'],
                      'relL2', f"{entry['correctness']['relative_l2']:.3e}", flush=True)
                row[name] = entry
                save()

            # Paired full-call timing for the best bitwise-equal arm.
            ok = {n: e for n, e in row.items()
                  if 'route_h5_ms' in e and e['correctness']['bitwise_equal']}
            if ok:
                best_name = min(ok, key=lambda n: ok[n]['route_h5_ms'])
                if tuple(ok[best_name]['config']) != BASE:
                    pairs = {'base': [], 'best': []}
                    for block in range(args.blocks):
                        order = ('base', 'best') if block % 2 == 0 else ('best', 'base')
                        for arm in order:
                            use(BASE if arm == 'base' else tuple(ok[best_name]['config']))
                            pairs[arm] += timed(lambda: op(q, k, v))
                    row['paired'] = {
                        'best_name': best_name,
                        'base_mean_ms': statistics.fmean(pairs['base']),
                        'best_mean_ms': statistics.fmean(pairs['best']),
                        'delta_ms': statistics.fmean(pairs['best']) - statistics.fmean(pairs['base']),
                        'wins': sum(int(b < a) for b, a in zip(pairs['best'], pairs['base'])),
                        'count': len(pairs['best'])}
                    print('PAIRED', layer, json.dumps(row['paired']), flush=True)
                else:
                    row['paired'] = {'best_name': best_name,
                                     'note': 'inherited configuration is already the best'}
            del reference, q, k, v, out, seeds
            torch.cuda.empty_cache()
            save()
        result['valid'] = not errors
        print('SUMMARY_DONE', flush=True)
        save()
    finally:
        stop.set()
        save()


if __name__ == '__main__':
    main()
