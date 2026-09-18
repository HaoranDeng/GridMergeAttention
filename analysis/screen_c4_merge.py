"""Paired complete-call screen for a candidate C4 merge against the retained one.

Both operators share one workspace and one selector, so the only difference is
the merge CUDA source. Complete H40 outputs must match the retained merge
bitwise before any timing is reported; timing is interleaved ABBA/BAAB on an
exclusively clean H200 with a foreign-process monitor.
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
    p.add_argument('--layer', action='append', default=[])
    p.add_argument('--candidate', type=Path, action='append', required=True,
                   help='Candidate merge source; may be repeated. Each is screened '
                        'against the retained merge in its own ABBA block set.')
    p.add_argument('--blocks', type=int, default=8)
    p.add_argument('--allow-mismatch', action='store_true',
                   help='Record output differences instead of refusing to time')
    args = p.parse_args()
    layers = args.layer or ['00']
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
        'initial_gpu': initial, 'layers': layers,
        'scope': 'Complete H40 C4 call; candidate merge source against the retained '
                 'guarded-softmax merge, same selector, workspace and Triton route',
        'target_ms': 80, 'candidate_sources': [str(x) for x in args.candidate],
        'timing_protocol': f'{args.blocks} alternating ABBA/BAAB blocks, '
                           '5 complete calls per method per block per layer',
    }

    def save():
        result.update(monitor_errors=errors, monitor_samples=samples)
        args.output.write_text(json.dumps(result, indent=2))

    try:
        import torch
        from kernels.best.c4.operator import NativeFisherAttention, extension
        torch.set_grad_enabled(False); torch.set_num_threads(4)
        op = NativeFisherAttention(
            device='cuda', capacity=4, selector_backend='fused_cooperative_loop',
            selector_source=SELECTOR, merge_source=RETAINED)
        retained = op.merge
        candidates = {c.stem: extension(4, source_path=c, name_suffix='_c' + str(i))
                      for i, c in enumerate(args.candidate)}
        result['resources'] = {'retained': retained.resources(),
                               **{n: m.resources() for n, m in candidates.items()}}
        result['source_sha256'] = {'retained': retained.source_sha256,
                                   **{n: m.source_sha256 for n, m in candidates.items()}}
        print('RESOURCES', json.dumps(result['resources']), flush=True)
        save()

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

        result['per_layer'] = {}
        for layer in layers:
            payload = torch.load(CAPTURE.format(layer), map_location='cpu', weights_only=True)
            q, k, v = [payload[n].reshape(1, 21, 45, 80, 40, 128).cuda() for n in ('q', 'k', 'v')]
            del payload
            row = {}
            # Register the row before timing so an aborted run still keeps every
            # correctness verdict and every candidate already measured.
            result['per_layer'][layer] = row
            op.merge = retained
            reference = op(q, k, v).clone()
            out = torch.empty_like(q)
            for name, candidate in candidates.items():
                entry = {}
                op.merge = candidate
                produced = op(q, k, v).clone()
                difference = produced.float() - reference.float()
                entry['correctness'] = {
                    'bitwise_equal': bool(torch.equal(produced, reference)),
                    'max_abs': float(difference.abs().max()),
                    'relative_l2': float((difference.square().sum()
                                          / reference.float().square().sum()).sqrt()),
                    'finite': bool(torch.isfinite(produced).all())}
                print('CORRECTNESS', layer, name, json.dumps(entry['correctness']), flush=True)
                del produced, difference
                if not entry['correctness']['bitwise_equal'] and not args.allow_mismatch:
                    entry['timing_skipped'] = 'not bitwise equal to the retained merge'
                    row[name] = entry
                    continue
                pairs = {'retained': [], 'candidate': []}
                for block in range(args.blocks):
                    order = ('retained', 'candidate') if block % 2 == 0 else ('candidate', 'retained')
                    for arm in order:
                        op.merge = retained if arm == 'retained' else candidate
                        pairs[arm] += timed(lambda: op(q, k, v))
                entry['timing_ms'] = {a: {'mean_ms': statistics.fmean(s), 'samples_ms': s}
                                      for a, s in pairs.items()}
                entry['paired_delta_ms'] = (entry['timing_ms']['candidate']['mean_ms']
                                            - entry['timing_ms']['retained']['mean_ms'])
                entry['paired_wins'] = sum(int(c < r) for c, r in
                                           zip(pairs['candidate'], pairs['retained']))
                entry['paired_count'] = len(pairs['candidate'])
                print('LAYER', layer, name, 'delta_ms', round(entry['paired_delta_ms'], 4),
                      'wins', entry['paired_wins'], '/', entry['paired_count'], flush=True)
                row[name] = entry
                save()
            del reference
            result['per_layer'][layer] = row
            save()
            del q, k, v, out
            torch.cuda.empty_cache()

        result['summary'] = {}
        for name in candidates:
            rows = [r[name] for r in result['per_layer'].values()
                    if name in r and 'paired_delta_ms' in r[name]]
            if not rows:
                continue
            result['summary'][name] = {
                'candidate_mean_ms': statistics.fmean(
                    [x for r in rows for x in r['timing_ms']['candidate']['samples_ms']]),
                'retained_mean_ms': statistics.fmean(
                    [x for r in rows for x in r['timing_ms']['retained']['samples_ms']]),
                'paired_wins': sum(r['paired_wins'] for r in rows),
                'paired_count': sum(r['paired_count'] for r in rows)}
            s = result['summary'][name]
            s['delta_ms'] = s['candidate_mean_ms'] - s['retained_mean_ms']
        print('SUMMARY', json.dumps(result['summary']), flush=True)
        result['valid'] = not errors
        save()
    finally:
        stop.set()
        save()


if __name__ == '__main__':
    main()
