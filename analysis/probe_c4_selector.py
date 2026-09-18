"""Decompose the C4 fused selector on an exclusive H200.

The probe selector sources are byte-identical to the retained register-basis
selector except that one stage is replaced by a cheap stand-in that still
consumes its inputs. Probe seeds are wrong by construction; this script makes
no correctness claim and promotes nothing.
"""
import argparse, json, os, statistics, sys, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.launch_quality_first_videos import require_clean, gpu_processes, now

CAPTURE = '/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer{}.pt'
RETAINED = ROOT / 'kernels/best/c4/fisher_c4_fused_selector_register_basis_loop.cu'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--layer', default='00')
    p.add_argument('--probe', type=Path, action='append', default=[])
    p.add_argument('--blocks', type=int, default=6)
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
    result = {'initial_gpu': initial, 'layer': args.layer,
              'scope': 'Timing-only C4 selector stage ablation; probe seeds are '
                       'deliberately wrong and no candidate is promoted',
              'retained_selector': str(RETAINED),
              'probes': [str(x) for x in args.probe]}

    def save():
        result.update(monitor_errors=errors, monitor_samples=samples)
        args.output.write_text(json.dumps(result, indent=2))

    try:
        import torch
        from kernels.best.c4.fused_selector import NativeFusedFisherSelector
        torch.set_grad_enabled(False); torch.set_num_threads(4)
        payload = torch.load(CAPTURE.format(args.layer), map_location='cpu', weights_only=True)
        q, k = [payload[n].reshape(1, 21, 45, 80, 40, 128).cuda() for n in ('q', 'k')]
        del payload

        selectors = {'retained': NativeFusedFisherSelector(
            4, device='cuda', loop_frames=True, cooperative=True, source_path=RETAINED)}
        for probe in args.probe:
            selectors[probe.stem] = NativeFusedFisherSelector(
                4, device='cuda', loop_frames=True, cooperative=True, source_path=probe)
        result['source_sha256'] = {n: s.extension.source_sha256 for n, s in selectors.items()}
        print('READY', json.dumps(result['source_sha256']), flush=True)

        def timed(fn, repeats=5):
            for _ in range(2):
                fn()
            torch.cuda.synchronize()
            vals = []
            for _ in range(repeats):
                assert not errors, errors
                s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                s.record(); y = fn(); e.record(); e.synchronize()
                vals.append(s.elapsed_time(e)); del y
            return vals

        names = list(selectors)
        sel_samples = {n: [] for n in names}
        for block in range(args.blocks):
            order = names if block % 2 == 0 else names[::-1]
            for name in order:
                sel_samples[name] += timed(
                    lambda n=name: selectors[n](q[:, :1], k, 128 ** -.5))
        result['selector_ms'] = {n: {'mean_ms': statistics.fmean(s), 'samples_ms': s}
                                 for n, s in sel_samples.items()}
        print('SELECTOR', json.dumps({n: v['mean_ms'] for n, v in result['selector_ms'].items()}),
              flush=True)
        result['valid'] = not errors
        save()
    finally:
        stop.set()
        save()


if __name__ == '__main__':
    main()
