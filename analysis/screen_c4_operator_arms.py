"""Complete-call ABBA of operator *configurations* against a base stack.

`screen_c4_full_stack.py` swaps kernels on one workspace, which cannot express a
change of `head_chunk` or `tiles`: those alter the workspace shape or the launch
grid, so every arm here is its own operator. The base is built twice - the
second copy is the A/A null arm - and each candidate is built, measured on every
layer, and freed before the next, because a ten-head workspace is 32.5 GB.

Arms are JSON overlays on the base configuration, so head-chunk, tile-count and
kernel-source sweeps all run through one instrument. Bitwise equality against
the base is recorded per layer and, unless `--allow-mismatch`, required.
"""
import argparse, json, os, socket, statistics, sys, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.launch_quality_first_videos import require_clean, gpu_processes, now

CAPTURE = '/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer{}.pt'
DEFAULT_BASE = {'merge_source': 'kernels/best/c4/fisher_c4_guarded_softmax.cu',
                'selector_source': 'kernels/best/c4/fisher_c4_fused_selector_register_basis_loop.cu',
                'head_chunk': 5, 'tiles': 21}


def build(cfg, NativeFisherAttention):
    return NativeFisherAttention(device='cuda', capacity=4,
                                 selector_backend='fused_cooperative_loop',
                                 selector_source=ROOT / cfg['selector_source'],
                                 merge_source=ROOT / cfg['merge_source'],
                                 head_chunk=cfg['head_chunk'], tiles=cfg['tiles'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base', default='{}', help='JSON overlay on the retained configuration.')
    p.add_argument('--arm', action='append', required=True,
                   help='Repeatable JSON overlay on the base; must carry "name".')
    p.add_argument('--layer', action='append', default=[])
    p.add_argument('--blocks', type=int, default=15)
    p.add_argument('--allow-mismatch', action='store_true')
    a = p.parse_args()
    layers = a.layer or ['00', '19', '39']
    base_cfg = dict(DEFAULT_BASE, **json.loads(a.base))
    arms = []
    for text in a.arm:
        overlay = json.loads(text)
        name = overlay.pop('name')
        arms.append((name, dict(base_cfg, **overlay)))
    assert not a.output.exists()
    initial = require_clean(a.gpu)
    host = socket.gethostname().split('.')[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = initial['gpu']['uuid']
    os.environ['TORCH_EXTENSIONS_DIR'] = str(a.output.parent / ('torch_extensions_' + host))
    os.environ['TRITON_CACHE_DIR'] = str(a.output.parent / ('triton_cache_' + host))
    os.environ['OMP_NUM_THREADS'] = '4'
    a.output.parent.mkdir(parents=True, exist_ok=True)

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
    result = {'initial_gpu': initial, 'host': host, 'layers': layers, 'base': base_cfg,
              'arm_configs': {n: c for n, c in arms},
              'scope': 'Complete-call paired timing of operator configurations against a '
                       'base stack, each arm its own operator, A/A null pair in the same '
                       'square, bitwise equality recorded per layer.'}

    def save():
        result.update(monitor_errors=errors, monitor_samples=samples)
        a.output.write_text(json.dumps(result, indent=2))

    def stats(pairs, x, y):
        n = min(len(pairs[x]), len(pairs[y]))
        deltas = [pairs[y][i] - pairs[x][i] for i in range(n)]
        mean = statistics.fmean(deltas)
        sd = statistics.stdev(deltas) if n > 1 else float('nan')
        return {'arm_a': x, 'arm_b': y, 'a_mean_ms': statistics.fmean(pairs[x]),
                'b_mean_ms': statistics.fmean(pairs[y]), 'delta_mean_ms': mean,
                'delta_sd_ms': sd, 'n': n,
                't': mean / (sd / n ** .5) if sd and n > 1 else float('nan'),
                'b_faster_count': sum(1 for d in deltas if d < 0)}

    try:
        import torch
        from kernels.best.c4.operator import NativeFisherAttention
        torch.set_grad_enabled(False); torch.set_num_threads(4)
        ops = {'base': build(base_cfg, NativeFisherAttention),
               'base_bis': build(base_cfg, NativeFisherAttention)}
        result['base_resources'] = ops['base'].merge.resources()
        result['base_sha256'] = {'merge': ops['base'].merge.source_sha256,
                                 'selector': ops['base'].native_selector.extension.source_sha256}
        print('BASE', json.dumps(result['base_resources']), json.dumps(result['base_sha256']), flush=True)
        save()

        captures = {}
        for layer in layers:
            payload = torch.load(CAPTURE.format(layer), map_location='cpu', weights_only=True)
            captures[layer] = [payload[n].reshape(1, 21, 45, 80, 40, 128).cuda()
                               for n in ('q', 'k', 'v')]
            del payload
        reference = {layer: ops['base'](*captures[layer]).clone() for layer in layers}
        for layer in layers:
            assert torch.equal(ops['base_bis'](*captures[layer]), reference[layer]), \
                'the null arm must reproduce the base result exactly'
        torch.cuda.synchronize()

        def timed(op, q, k, v):
            s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            s.record(); y = op(q, k, v); e.record(); e.synchronize()
            del y
            return s.elapsed_time(e)

        def square(x, y, q, k, v):
            pairs = {x: [], y: []}
            for block in range(a.blocks):
                order = (x, y, y, x) if block % 2 == 0 else (y, x, x, y)
                for arm in order:
                    assert not errors, errors
                    pairs[arm].append(timed(ops[arm], q, k, v))
            return pairs

        result['arms'] = {}
        for name, cfg in arms:
            ops['cand'] = build(cfg, NativeFisherAttention)
            entry = {'config': cfg, 'resources': ops['cand'].merge.resources(),
                     'workspace_bytes': int(ops['cand'].mk.numel() * 2 * 2 + ops['cand'].bias.numel() * 4),
                     'sha256': {'merge': ops['cand'].merge.source_sha256,
                                'selector': ops['cand'].native_selector.extension.source_sha256},
                     'per_layer': {}}
            print('ARM', name, json.dumps(entry['resources']), 'workspace_GB',
                  round(entry['workspace_bytes'] / 1e9, 2), flush=True)
            for layer in layers:
                q, k, v = captures[layer]
                produced = ops['cand'](q, k, v).clone()
                difference = produced.float() - reference[layer].float()
                row = {'correctness': {
                    'bitwise_equal': bool(torch.equal(produced, reference[layer])),
                    'max_abs': float(difference.abs().max()),
                    'relative_l2': float((difference.square().sum()
                                          / reference[layer].float().square().sum()).sqrt()),
                    'finite': bool(torch.isfinite(produced).all())}}
                del produced, difference
                print('CORRECTNESS', name, layer, json.dumps(row['correctness']), flush=True)
                if not row['correctness']['bitwise_equal'] and not a.allow_mismatch:
                    row['timing_skipped'] = 'not bitwise equal to the base; pass --allow-mismatch'
                    entry['per_layer'][layer] = row; save(); continue
                for _ in range(3):
                    timed(ops['base'], q, k, v); timed(ops['cand'], q, k, v)
                torch.cuda.synchronize()
                row['effect'] = stats(square('base', 'cand', q, k, v), 'base', 'cand')
                row['null'] = stats(square('base', 'base_bis', q, k, v), 'base', 'base_bis')
                row['candidate_delta_ms'] = row['effect']['delta_mean_ms'] - row['null']['delta_mean_ms']
                row['base_call_ms'] = row['effect']['a_mean_ms']
                row['candidate_call_ms'] = row['effect']['b_mean_ms']
                print('LAYER', name, layer, 'base', round(row['base_call_ms'], 3), 'candidate',
                      round(row['candidate_call_ms'], 3), 'delta (offset corrected)',
                      round(row['candidate_delta_ms'], 4), 'effect t', round(row['effect']['t'], 2),
                      'wins', row['effect']['b_faster_count'], '/', row['effect']['n'],
                      'null', round(row['null']['delta_mean_ms'], 4), flush=True)
                entry['per_layer'][layer] = row
                save()
            rows = [r for r in entry['per_layer'].values() if 'candidate_delta_ms' in r]
            if rows:
                entry['summary'] = {
                    'pooled_delta_ms': statistics.fmean(r['candidate_delta_ms'] for r in rows),
                    'base_call_ms': statistics.fmean(r['base_call_ms'] for r in rows),
                    'candidate_call_ms': statistics.fmean(r['candidate_call_ms'] for r in rows),
                    'wins': sum(r['effect']['b_faster_count'] for r in rows),
                    'n': sum(r['effect']['n'] for r in rows),
                    'bitwise_all_layers': all(r['correctness']['bitwise_equal'] for r in rows)}
                print('SUMMARY', name, json.dumps(entry['summary']), flush=True)
            result['arms'][name] = entry
            save()
            del ops['cand']
            torch.cuda.empty_cache()
        result['valid'] = not errors
        print('SUMMARY_DONE', flush=True)
        save()
    finally:
        stop.set(); save()


if __name__ == '__main__':
    main()
