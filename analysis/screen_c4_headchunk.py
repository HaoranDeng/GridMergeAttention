"""Screen a ten-head chunk against the shipped five-head chunk for C4.

The 2026-09-09 campaign measured `head10` at -0.70 ms against its own control and
never banked it; the retained package still ships head_chunk=5. The chunk size
changes only the compressed workspace stride and the number of launches, so
complete H40 output must stay bitwise equal. Ten-head chunks double the
compressed workspace to ~32.5 GB, so operators are built and freed one at a time.
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
    p.add_argument('--candidate', type=Path, action='append', required=True)
    p.add_argument('--head-chunk', type=int, default=10)
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
    result = {'initial_gpu': initial, 'layers': layers, 'head_chunk': args.head_chunk,
              'candidates': [str(c) for c in args.candidate],
              'scope': 'Complete H40 C4 call with a larger head chunk against the '
                       'shipped five-head chunk; bitwise gate then paired ABBA.'}

    def save():
        result.update(monitor_errors=errors, monitor_query_errors=query_errors,
                      monitor_samples=samples)
        args.output.write_text(json.dumps(result, indent=2))

    try:
        import torch
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

        base = NativeFisherAttention(device='cuda', capacity=4,
                                     selector_backend='fused_cooperative_loop',
                                     selector_source=SELECTOR, merge_source=RETAINED)
        result['base_resources'] = base.merge.resources()
        print('BASE', json.dumps(result['base_resources']), flush=True)

        captures = {}
        for layer in layers:
            payload = torch.load(CAPTURE.format(layer), map_location='cpu', weights_only=True)
            captures[layer] = [payload[n].reshape(1, 21, 45, 80, 40, 128).cuda()
                               for n in ('q', 'k', 'v')]
            del payload
        reference = {layer: base(*captures[layer]).clone() for layer in layers}
        torch.cuda.synchronize()

        result['arms'] = {}
        for candidate in args.candidate:
            name = candidate.stem
            entry = {}
            wide = NativeFisherAttention(device='cuda', capacity=4,
                                         selector_backend='fused_cooperative_loop',
                                         selector_source=SELECTOR,
                                         merge_source=candidate,
                                         head_chunk=args.head_chunk)
            entry['resources'] = wide.merge.resources()
            entry['workspace_bytes'] = int(wide.mk.numel() * 2 * 2 + wide.bias.numel() * 4)
            print('ARM', name, json.dumps(entry['resources']),
                  'workspace_GB', round(entry['workspace_bytes'] / 1e9, 2), flush=True)
            entry['per_layer'] = {}
            for layer in layers:
                q, k, v = captures[layer]
                produced = wide(q, k, v).clone()
                difference = produced.float() - reference[layer].float()
                correctness = {
                    'bitwise_equal': bool(torch.equal(produced, reference[layer])),
                    'max_abs': float(difference.abs().max()),
                    'relative_l2': float((difference.square().sum()
                                          / reference[layer].float().square().sum()).sqrt()),
                    'finite': bool(torch.isfinite(produced).all())}
                del produced, difference
                row = {'correctness': correctness}
                print('CORRECTNESS', name, layer, json.dumps(correctness), flush=True)
                if not correctness['bitwise_equal']:
                    row['timing_skipped'] = 'not bitwise equal to the five-head chunk'
                    entry['per_layer'][layer] = row; save(); continue
                pairs = {'base': [], 'wide': []}
                for block in range(args.blocks):
                    order = ('base', 'wide') if block % 2 == 0 else ('wide', 'base')
                    for arm in order:
                        op = base if arm == 'base' else wide
                        pairs[arm] += timed(lambda o=op: o(q, k, v))
                row['timing_ms'] = {a: {'mean_ms': statistics.fmean(s), 'samples_ms': s}
                                    for a, s in pairs.items()}
                row['paired_delta_ms'] = (row['timing_ms']['wide']['mean_ms']
                                          - row['timing_ms']['base']['mean_ms'])
                row['paired_wins'] = sum(int(w < b) for w, b in zip(pairs['wide'], pairs['base']))
                row['paired_count'] = len(pairs['wide'])
                print('LAYER', name, layer, 'delta_ms', round(row['paired_delta_ms'], 4),
                      'wins', row['paired_wins'], '/', row['paired_count'], flush=True)
                entry['per_layer'][layer] = row
                save()
            rows = [r for r in entry['per_layer'].values() if 'paired_delta_ms' in r]
            if rows:
                entry['summary'] = {
                    'wide_mean_ms': statistics.fmean(
                        [x for r in rows for x in r['timing_ms']['wide']['samples_ms']]),
                    'base_mean_ms': statistics.fmean(
                        [x for r in rows for x in r['timing_ms']['base']['samples_ms']]),
                    'paired_wins': sum(r['paired_wins'] for r in rows),
                    'paired_count': sum(r['paired_count'] for r in rows)}
                entry['summary']['delta_ms'] = (entry['summary']['wide_mean_ms']
                                                - entry['summary']['base_mean_ms'])
                print('SUMMARY', name, json.dumps(entry['summary']), flush=True)
            result['arms'][name] = entry
            save()
            del wide
            torch.cuda.empty_cache()
        result['valid'] = not errors
        print('SUMMARY_DONE', flush=True)
        save()
    finally:
        stop.set()
        save()


if __name__ == '__main__':
    main()
