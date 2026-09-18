"""Selector-stage timing of selector-source variants against a configurable base.

`screen_c4_selector_stage.py` measures one candidate against the retained FP64
selector. The base is now the FP32 GS register-pipelined selector on the
desc_select merge, and the next candidates are a stack of bitwise-safe
rewrites of that selector, so this instrument takes the base as an argument
and any number of variants.

Per layer: the selector stage is timed in place inside a complete call (events
around `op.select`), arms interleaved ABBA/BAAB, an A/A null arm (a second
module of the base source) in the same square. Every arm's seeds are compared
bitwise with the base's and the complete output likewise; unless
`--allow-mismatch` both must be equal. Registers / spills / shared / CTAs per
SM are recorded per arm (`gate-the-instrument`).
"""
import argparse, json, os, socket, statistics, sys, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.launch_quality_first_videos import require_clean, gpu_processes, now

CAPTURE = '/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer{}.pt'
BASE_SELECTOR = ROOT / 'native_fisher/fisher_c4_selector_fp32gs_regpipe.cu'
MERGE = ROOT / 'native_fisher/fisher_c4_desc_select.cu'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant', type=Path, action='append', required=True)
    p.add_argument('--base-selector', type=Path, default=BASE_SELECTOR)
    p.add_argument('--merge-source', type=Path, default=MERGE)
    p.add_argument('--layer', action='append', default=[])
    p.add_argument('--blocks', type=int, default=15)
    p.add_argument('--allow-mismatch', action='store_true')
    a = p.parse_args()
    layers = a.layer or ['00', '19', '39']
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
    result = {'initial_gpu': initial, 'host': host, 'layers': layers,
              'base_selector': str(a.base_selector), 'merge_source': str(a.merge_source),
              'variants': [str(v) for v in a.variant],
              'scope': 'Selector stage timed in place inside complete C4 calls on the promoted '
                       'merge, selector module swapped per arm, interleaved ABBA/BAAB with an '
                       'A/A null arm; seeds and output compared bitwise with the base per layer; '
                       'resources recorded per arm.'}

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
        from kernels.best.c4.fused_selector import NativeFusedFisherSelector
        torch.set_grad_enabled(False); torch.set_num_threads(4)
        op = NativeFisherAttention(device='cuda', capacity=4, selector_backend='fused_cooperative_loop',
                                   selector_source=a.base_selector, merge_source=a.merge_source)

        def selector(path):
            return NativeFusedFisherSelector(4, device='cuda', loop_frames=True, cooperative=True,
                                             source_path=path)

        sel = {'base': op.native_selector, 'null': selector(a.base_selector)}
        for v in a.variant:
            sel[v.stem.replace('fisher_c4_selector_', '')] = selector(v)
        result['selector_sha256'] = {n: s.extension.source_sha256 for n, s in sel.items()}
        result['resources'] = {n: s.extension.resources() for n, s in sel.items()}
        result['merge_sha256'] = op.merge.source_sha256
        assert result['selector_sha256']['null'] == result['selector_sha256']['base']
        print('SOURCES', json.dumps(result['selector_sha256']), flush=True)
        print('RESOURCES', json.dumps(result['resources']), flush=True)
        save()

        scale = 128 ** -.5

        def call(arm, q, k, v):
            """One complete call; returns (output, selector stage ms inside it)."""
            op.native_selector = sel[arm]
            s0, s1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            s0.record()
            seeds = op.select(q, k, scale); op.seeds = seeds
            s1.record()
            op.merge.pack(q, op.packed)
            out = torch.empty_like(q)
            for head in range(0, 40, op.head_chunk):
                op.merge.run(op.packed, k, v, seeds, op.mk, op.mv, op.bias, head,
                             op.head_chunk, op.tiles, scale, op.empty, 0, 1)
                op.route(q, k, v, out, head, scale, active_heads=op.head_chunk)
            torch.cuda.synchronize()
            return out, s0.elapsed_time(s1)

        result['correctness'] = {}
        result['arms'] = {}
        for layer in layers:
            payload = torch.load(CAPTURE.format(layer), map_location='cpu', weights_only=True)
            q, k, v = [payload[n].reshape(1, 21, 45, 80, 40, 128).cuda() for n in ('q', 'k', 'v')]
            del payload
            reference, _ = call('base', q, k, v)
            reference = reference.clone()
            reference_seeds = op.seeds.clone()
            row = {}
            for arm in sel:
                if arm == 'base':
                    continue
                produced, _ = call(arm, q, k, v)
                seeds = op.seeds
                difference = produced.float() - reference.float()
                row[arm] = {'seeds_bitwise_equal': bool(torch.equal(seeds, reference_seeds)),
                            'seed_mismatches': int((seeds != reference_seeds).sum()),
                            'seed_total': int(reference_seeds.numel()),
                            'output_bitwise_equal': bool(torch.equal(produced, reference)),
                            'finite': bool(torch.isfinite(produced).all()),
                            'max_abs': float(difference.abs().max()),
                            'relative_l2': float((difference.square().sum()
                                                  / reference.float().square().sum()).sqrt())}
                del produced, difference
                print('CORRECTNESS', layer, arm, json.dumps(row[arm]), flush=True)
            result['correctness'][layer] = row
            assert row['null']['seeds_bitwise_equal'] and row['null']['output_bitwise_equal'], \
                'the null arm must reproduce the base exactly'
            if not a.allow_mismatch:
                for arm, c in row.items():
                    assert c['seeds_bitwise_equal'] and c['output_bitwise_equal'], \
                        '%s is not bitwise; pass --allow-mismatch' % arm
            del reference, reference_seeds
            save()

            for _ in range(3):
                for arm in sel:
                    call(arm, q, k, v)
            torch.cuda.synchronize()

            def square(x, y):
                pairs = {x: [], y: []}
                for block in range(a.blocks):
                    order = (x, y, y, x) if block % 2 == 0 else (y, x, x, y)
                    for arm in order:
                        assert not errors, errors
                        out, ms = call(arm, q, k, v)
                        del out
                        pairs[arm].append(ms)
                return pairs

            entry = {'null': stats(square('base', 'null'), 'base', 'null')}
            print('NULL', layer, json.dumps({k: entry['null'][k] for k in
                  ('a_mean_ms', 'b_mean_ms', 'delta_mean_ms', 'delta_sd_ms', 't', 'b_faster_count', 'n')}),
                  flush=True)
            for arm in sel:
                if arm in ('base', 'null'):
                    continue
                e = stats(square('base', arm), 'base', arm)
                e['delta_offset_corrected_ms'] = e['delta_mean_ms'] - entry['null']['delta_mean_ms']
                entry[arm] = e
                print('ARM', layer, arm, json.dumps({k: e[k] for k in
                      ('a_mean_ms', 'b_mean_ms', 'delta_mean_ms', 'delta_sd_ms', 't',
                       'b_faster_count', 'n', 'delta_offset_corrected_ms')}), flush=True)
            result['arms'][layer] = entry
            del q, k, v
            torch.cuda.empty_cache()
            save()

        summary = {}
        for arm in sel:
            if arm in ('base', 'null'):
                continue
            per = {l: result['arms'][l][arm]['delta_offset_corrected_ms'] for l in result['arms']}
            summary[arm] = {'pooled_selector_delta_ms_per_call': statistics.fmean(per.values()),
                            'per_layer_ms': per,
                            'wins': sum(result['arms'][l][arm]['b_faster_count'] for l in result['arms']),
                            'n': sum(result['arms'][l][arm]['n'] for l in result['arms']),
                            'seeds_bitwise_all_layers': all(result['correctness'][l][arm]['seeds_bitwise_equal']
                                                            for l in result['correctness']),
                            'output_bitwise_all_layers': all(result['correctness'][l][arm]['output_bitwise_equal']
                                                             for l in result['correctness']),
                            'resources': result['resources'][arm]}
        summary['base_selector_ms_per_call'] = statistics.fmean(
            result['arms'][l]['null']['a_mean_ms'] for l in result['arms'])
        result['summary'] = summary
        print('SUMMARY', json.dumps(summary), flush=True)
        result['valid'] = not errors
        print('SUMMARY_DONE', flush=True)
        save()
    finally:
        stop.set(); save()


if __name__ == '__main__':
    main()
