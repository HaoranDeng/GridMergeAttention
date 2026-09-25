"""Paired complete-call screen for a candidate C4 selector source.

Both selectors share one operator and workspace, so the only difference is the
selector CUDA source. Seeds must match bitwise and the complete H40 output must
match bitwise before any timing is reported.
"""
import argparse, json, os, statistics, sys, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.launch_quality_first_videos import require_clean, gpu_processes, now

CAPTURE = '/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer{}.pt'
RETAINED_MERGE = ROOT / 'kernels/best/c4/fisher_c4_guarded_softmax.cu'
RETAINED_SELECTOR = ROOT / 'kernels/best/c4/fisher_c4_fused_selector_register_basis_loop.cu'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--layer', action='append', default=[])
    p.add_argument('--candidate', type=Path, action='append', required=True)
    p.add_argument('--merge-source', type=Path, default=RETAINED_MERGE)
    p.add_argument('--blocks', type=int, default=10)
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
    result = {'initial_gpu': initial, 'layers': layers,
              'merge_source': str(args.merge_source),
              'candidates': [str(c) for c in args.candidate],
              'scope': 'Complete H40 C4 call; candidate selector source against the '
                       'retained register-basis selector, same merge and route.'}

    def save():
        result.update(monitor_errors=errors, monitor_query_errors=query_errors,
                      monitor_samples=samples)
        args.output.write_text(json.dumps(result, indent=2))

    try:
        import torch
        from kernels.best.c4.operator import NativeFisherAttention
        from kernels.best.c4.fused_selector import NativeFusedFisherSelector
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

        op = NativeFisherAttention(device='cuda', capacity=4,
                                   selector_backend='fused_cooperative_loop',
                                   selector_source=RETAINED_SELECTOR,
                                   merge_source=args.merge_source)
        retained = op.native_selector
        cands = {c.stem: NativeFusedFisherSelector(4, device='cuda', loop_frames=True,
                                                  cooperative=True, source_path=c)
                 for c in args.candidate}
        result['source_sha256'] = {'retained': retained.extension.source_sha256,
                                   **{n: s.extension.source_sha256 for n, s in cands.items()}}
        print('READY', json.dumps(result['source_sha256']), flush=True)
        save()

        result['arms'] = {}
        for layer in layers:
            payload = torch.load(CAPTURE.format(layer), map_location='cpu', weights_only=True)
            q, k, v = [payload[n].reshape(1, 21, 45, 80, 40, 128).cuda() for n in ('q', 'k', 'v')]
            del payload
            op.native_selector = retained
            reference = op(q, k, v).clone()
            reference_seeds = retained(q[:, :1], k, 128 ** -.5).clone()
            for name, selector in cands.items():
                entry = result['arms'].setdefault(name, {})
                op.native_selector = selector
                produced_seeds = selector(q[:, :1], k, 128 ** -.5).clone()
                produced = op(q, k, v).clone()
                difference = produced.float() - reference.float()
                correctness = {
                    'seeds_bitwise_equal': bool(torch.equal(produced_seeds, reference_seeds)),
                    'seed_mismatches': int((produced_seeds != reference_seeds).sum()),
                    'output_bitwise_equal': bool(torch.equal(produced, reference)),
                    'max_abs': float(difference.abs().max()),
                    'finite': bool(torch.isfinite(produced).all())}
                del produced, produced_seeds, difference
                row = {'correctness': correctness}
                print('CORRECTNESS', name, layer, json.dumps(correctness), flush=True)
                if not (correctness['seeds_bitwise_equal'] and correctness['output_bitwise_equal']):
                    row['timing_skipped'] = 'not bitwise equal to the retained selector'
                    entry[layer] = row; save(); continue
                op.native_selector = selector
                row['selector_ms'] = statistics.fmean(
                    timed(lambda: selector(q[:, :1], k, 128 ** -.5)))
                op.native_selector = retained
                row['retained_selector_ms'] = statistics.fmean(
                    timed(lambda: retained(q[:, :1], k, 128 ** -.5)))
                pairs = {'retained': [], 'candidate': []}
                for block in range(args.blocks):
                    order = ('retained', 'candidate') if block % 2 == 0 else ('candidate', 'retained')
                    for arm in order:
                        op.native_selector = retained if arm == 'retained' else selector
                        pairs[arm] += timed(lambda: op(q, k, v))
                row['timing_ms'] = {a: {'mean_ms': statistics.fmean(s), 'samples_ms': s}
                                    for a, s in pairs.items()}
                row['paired_delta_ms'] = (row['timing_ms']['candidate']['mean_ms']
                                          - row['timing_ms']['retained']['mean_ms'])
                row['paired_wins'] = sum(int(c < r) for c, r in
                                         zip(pairs['candidate'], pairs['retained']))
                row['paired_count'] = len(pairs['candidate'])
                print('LAYER', name, layer, 'selector', round(row['selector_ms'], 4),
                      'vs', round(row['retained_selector_ms'], 4),
                      'call_delta', round(row['paired_delta_ms'], 4),
                      'wins', row['paired_wins'], '/', row['paired_count'], flush=True)
                entry[layer] = row
                save()
            del reference, reference_seeds, q, k, v
            torch.cuda.empty_cache()

        result['summary'] = {}
        for name, entry in result['arms'].items():
            rows = [r for r in entry.values() if 'paired_delta_ms' in r]
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
        print('SUMMARY_DONE', flush=True)
        save()
    finally:
        stop.set()
        save()


if __name__ == '__main__':
    main()
