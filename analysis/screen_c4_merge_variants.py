"""Price merge-source variants per chunk, inside complete calls, on the promoted stack.

Base merge is `fisher_c4_desc_select.cu` with the FP32 GS register-pipelined
selector - the pair that ships. Each variant is a merge source swapped on the
same workspace; the eight merge launches of a complete call are timed with
events and summed, arms interleaved ABBA/BAAB, with an A/A null arm (a second
module of the base source) in the same square. Numerics are recorded per layer
for every arm and, unless `--allow-mismatch`, required bitwise.

Registers, spills, shared bytes and CTAs per SM are recorded for every arm, so
a variant that changed occupancy is visible in the receipt rather than silently
timed as if it were the same kernel (`gate-the-instrument`).
"""
import argparse, json, os, socket, statistics, sys, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.launch_quality_first_videos import require_clean, gpu_processes, now

CAPTURE = '/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer{}.pt'
BASE_MERGE = ROOT / 'native_fisher/fisher_c4_desc_select.cu'
SELECTOR = ROOT / 'native_fisher/fisher_c4_selector_fp32gs_regpipe.cu'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant', type=Path, action='append', required=True)
    p.add_argument('--base-merge', type=Path, default=BASE_MERGE)
    p.add_argument('--selector', type=Path, default=SELECTOR)
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
              'base_merge': str(a.base_merge), 'selector': str(a.selector),
              'variants': [str(v) for v in a.variant],
              'scope': 'Per-chunk merge time inside complete C4 calls on the promoted stack, '
                       'merge module swapped per arm, interleaved ABBA/BAAB with an A/A null '
                       'arm; numerics and resources recorded per arm.'}

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
        from kernels.best.c4.operator import NativeFisherAttention, extension
        torch.set_grad_enabled(False); torch.set_num_threads(4)
        op = NativeFisherAttention(device='cuda', capacity=4, selector_backend='fused_cooperative_loop',
                                   selector_source=a.selector, merge_source=a.base_merge)
        mods = {'base': op.merge,
                'null': extension(4, source_path=a.base_merge, name_suffix='_bis')}
        for v in a.variant:
            mods[v.stem] = extension(4, source_path=v, name_suffix='_' + v.stem[-12:])
        result['resources'] = {n: m.resources() for n, m in mods.items()}
        result['merge_sha256'] = {n: m.source_sha256 for n, m in mods.items()}
        result['selector_sha256'] = op.native_selector.extension.source_sha256
        assert result['merge_sha256']['null'] == result['merge_sha256']['base']
        print('RESOURCES', json.dumps(result['resources']), flush=True)
        print('SOURCES', json.dumps(result['merge_sha256']), flush=True)
        save()

        scale = 128 ** -.5

        def call(arm, q, k, v):
            """One complete call; returns (output, summed merge ms over the eight chunks)."""
            module = mods[arm]
            marks = [torch.cuda.Event(enable_timing=True) for _ in range(16)]
            seeds = op.select(q, k, scale); op.seeds = seeds
            module.pack(q, op.packed)
            out = torch.empty_like(q)
            i = 0
            for head in range(0, 40, op.head_chunk):
                marks[i].record(); i += 1
                module.run(op.packed, k, v, seeds, op.mk, op.mv, op.bias, head,
                           op.head_chunk, op.tiles, scale, op.empty, 0, 1)
                marks[i].record(); i += 1
                op.route(q, k, v, out, head, scale, active_heads=op.head_chunk)
            marks[i - 1].synchronize()
            return out, sum(marks[j].elapsed_time(marks[j + 1]) for j in range(0, 16, 2))

        result['correctness'] = {}
        result['arms'] = {}
        for layer in layers:
            payload = torch.load(CAPTURE.format(layer), map_location='cpu', weights_only=True)
            q, k, v = [payload[n].reshape(1, 21, 45, 80, 40, 128).cuda() for n in ('q', 'k', 'v')]
            del payload
            reference, _ = call('base', q, k, v)
            reference = reference.clone()
            bias_reference = op.bias.clone()   # last chunk's bias: masked-token agreement
            row = {}
            for arm in mods:
                if arm == 'base':
                    continue
                produced, _ = call(arm, q, k, v)
                difference = produced.float() - reference.float()
                row[arm] = {'output_bitwise_equal': bool(torch.equal(produced, reference)),
                            'finite': bool(torch.isfinite(produced).all()),
                            'max_abs': float(difference.abs().max()),
                            'relative_l2': float((difference.square().sum()
                                                  / reference.float().square().sum()).sqrt()),
                            'last_chunk_bias_inf_agreement': bool(torch.equal(
                                torch.isinf(op.bias), torch.isinf(bias_reference))),
                            'last_chunk_bias_max_abs_finite': float(
                                (op.bias - bias_reference)[torch.isfinite(bias_reference)
                                                           & torch.isfinite(op.bias)].abs().max())}
                del produced, difference
                print('CORRECTNESS', layer, arm, json.dumps(row[arm]), flush=True)
            result['correctness'][layer] = row
            assert row['null']['output_bitwise_equal'], 'the null arm must reproduce the base output'
            if not a.allow_mismatch:
                for arm, c in row.items():
                    assert c['output_bitwise_equal'], '%s is not bitwise; pass --allow-mismatch' % arm
            del reference, bias_reference
            save()

            for _ in range(3):
                for arm in mods:
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
            for arm in mods:
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
        for arm in mods:
            if arm in ('base', 'null'):
                continue
            per = {l: result['arms'][l][arm]['delta_offset_corrected_ms'] for l in result['arms']}
            summary[arm] = {'pooled_merge_delta_ms_per_call': statistics.fmean(per.values()),
                            'per_layer_ms': per,
                            'wins': sum(result['arms'][l][arm]['b_faster_count'] for l in result['arms']),
                            'n': sum(result['arms'][l][arm]['n'] for l in result['arms']),
                            'bitwise_all_layers': all(result['correctness'][l][arm]['output_bitwise_equal']
                                                      for l in result['correctness']),
                            'max_relative_l2': max(result['correctness'][l][arm]['relative_l2']
                                                   for l in result['correctness']),
                            'resources': result['resources'][arm]}
        summary['base_merge_ms_per_call'] = statistics.fmean(
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
