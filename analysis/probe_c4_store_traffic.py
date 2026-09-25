"""Separate C4 merge store traffic from merge issue cost on an exclusive H200.

The probe merge sources are byte-identical to the retained guarded-softmax merge
except for the compressed K/V tensor-map extents, so every WGMMA, reduction,
stmatrix and TMA issue is preserved while the store traffic is divided by 21
(frame alias) or effectively removed (full alias). Probe outputs are garbage by
construction; this script makes no correctness claim and promotes nothing. It
also anchors achievable device read and write bandwidth in the same process.
"""
import argparse, json, os, statistics, sys, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.launch_quality_first_videos import require_clean, gpu_processes, now

CAPTURE = '/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer{}.pt'
RETAINED = ROOT / 'kernels/best/c4/fisher_c4_guarded_softmax.cu'
SELECTOR = ROOT / 'kernels/best/c4/fisher_c4_fused_selector_register_basis_loop.cu'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--layer', default='00')
    p.add_argument('--probe', type=Path, action='append', default=[],
                   help='Probe merge source; may be repeated')
    p.add_argument('--blocks', type=int, default=6, help='ABBA/BAAB block count')
    args = p.parse_args()
    assert not args.output.exists()
    initial = require_clean(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = initial['gpu']['uuid']
    os.environ['TORCH_EXTENSIONS_DIR'] = str(args.output.parent / 'torch_extensions')
    os.environ['TRITON_CACHE_DIR'] = str(args.output.parent / 'triton_cache')
    os.environ['OMP_NUM_THREADS'] = '4'
    args.output.parent.mkdir(parents=True, exist_ok=True)

    stop = threading.Event(); errors = []; samples = []

    def monitor():
        while not stop.is_set():
            try:
                ps = gpu_processes(initial['gpu']['uuid'])
                samples.append({'utc': now(), 'processes': ps})
                if any(r['pid'] != os.getpid() for r in ps):
                    errors.append('foreign GPU process')
            except Exception as exc:
                errors.append(repr(exc))
            stop.wait(3)

    threading.Thread(target=monitor, daemon=True).start()
    result = {
        'initial_gpu': initial,
        'scope': 'Timing-only C4 merge store-traffic ablation; probe outputs are '
                 'deliberately wrong and no candidate is promoted',
        'layer': args.layer,
        'retained_merge': str(RETAINED),
        'probes': [str(x) for x in args.probe],
    }

    def save():
        result.update(monitor_errors=errors, monitor_samples=samples)
        args.output.write_text(json.dumps(result, indent=2))

    try:
        import torch
        from kernels.best.c4.operator import NativeFisherAttention, extension
        torch.set_grad_enabled(False); torch.set_num_threads(4)
        payload = torch.load(CAPTURE.format(args.layer), map_location='cpu', weights_only=True)
        q, k, v = [payload[n].reshape(1, 21, 45, 80, 40, 128).cuda() for n in ('q', 'k', 'v')]
        del payload

        # Achievable bandwidth anchor in this process, on this device.
        def bandwidth():
            n = 2 << 30  # 2 GiB of bfloat16 pairs
            a = torch.empty(n // 2, dtype=torch.bfloat16, device='cuda')
            b = torch.empty_like(a)
            a.fill_(1.0)

            def timed(fn, repeats=10):
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                vals = []
                for _ in range(repeats):
                    s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    s.record(); fn(); e.record(); e.synchronize()
                    vals.append(s.elapsed_time(e))
                return statistics.fmean(vals)

            copy_ms = timed(lambda: b.copy_(a))
            fill_ms = timed(lambda: b.fill_(0.0))
            read_ms = timed(lambda: torch.sum(a.view(torch.int16).to(torch.int32)[:1]) if False else a.sum())
            out = {
                'bytes_per_tensor': a.numel() * 2,
                'copy_ms': copy_ms,
                'copy_tb_s': 2 * a.numel() * 2 / copy_ms / 1e9,
                'fill_ms': fill_ms,
                'write_tb_s': a.numel() * 2 / fill_ms / 1e9,
                'reduce_ms': read_ms,
                'read_tb_s': a.numel() * 2 / read_ms / 1e9,
            }
            del a, b
            torch.cuda.empty_cache()
            return out

        result['bandwidth_anchor'] = bandwidth(); save()
        print('BANDWIDTH', json.dumps(result['bandwidth_anchor']), flush=True)

        op = NativeFisherAttention(
            device='cuda', capacity=4, selector_backend='fused_cooperative_loop',
            selector_source=SELECTOR, merge_source=RETAINED)
        modules = {'retained': op.merge}
        for probe in args.probe:
            modules[probe.stem] = extension(4, source_path=probe, name_suffix='_probe')
        result['resources'] = {name: m.resources() for name, m in modules.items()}
        result['source_sha256'] = {name: m.source_sha256 for name, m in modules.items()}
        print('RESOURCES', json.dumps(result['resources']), flush=True)

        seeds = op.select(q, k, 128 ** -.5)
        op.merge.pack(q, op.packed)
        torch.cuda.synchronize()

        def merge_chunk(module):
            module.run(op.packed, k, v, seeds, op.mk, op.mv, op.bias, 0, 5, op.tiles,
                       128 ** -.5, op.empty, 0, 1)

        def timed(fn, repeats=5):
            for _ in range(2):
                fn()
            torch.cuda.synchronize()
            vals = []
            for _ in range(repeats):
                assert not errors, errors
                s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                s.record(); fn(); e.record(); e.synchronize()
                vals.append(s.elapsed_time(e))
            return vals

        names = list(modules)
        merge_samples = {n: [] for n in names}
        for block in range(args.blocks):
            order = names if block % 2 == 0 else names[::-1]
            for name in order:
                merge_samples[name] += timed(lambda n=name: merge_chunk(modules[n]))
        result['merge_h5_ms'] = {
            n: {'mean_ms': statistics.fmean(s), 'samples_ms': s}
            for n, s in merge_samples.items()}
        print('MERGE', json.dumps({n: v['mean_ms'] for n, v in result['merge_h5_ms'].items()}), flush=True)
        save()

        # Stage context measured with the retained merge only.
        op.merge = modules['retained']
        out = torch.empty_like(q)
        stages = {
            'selector': lambda: op.select(q, k, 128 ** -.5),
            'pack': lambda: op.merge.pack(q, op.packed),
            'merge_h5': lambda: merge_chunk(modules['retained']),
            'route_h5': lambda: op.route(q, k, v, out, 0, 128 ** -.5, active_heads=5),
            'full_call': lambda: op(q, k, v),
        }
        result['stages'] = {n: {'samples_ms': (s := timed(f)), 'mean_ms': statistics.fmean(s)}
                            for n, f in stages.items()}
        print('STAGES', json.dumps({n: v['mean_ms'] for n, v in result['stages'].items()}), flush=True)
        result['workspace_bytes'] = {
            'mk': op.mk.numel() * 2, 'mv': op.mv.numel() * 2, 'bias': op.bias.numel() * 4}
        result['valid'] = not errors
        save()
    finally:
        stop.set()
        save()


if __name__ == '__main__':
    main()
