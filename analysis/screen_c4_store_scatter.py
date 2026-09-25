"""Price the C4 compressed K/V store scatter, in call, with a null pair.

The retained merge emits sixteen 1 KiB runs 9.7 MB apart per TMA store. The
probe writes the same bytes through the same box, the same shared staging and
the same issue count, but to one contiguous 16 KiB run. The difference is the
scatter penalty and nothing else; the probe's output is deliberately wrong.

Instrument choices, each forced by a recorded failure:
  * per-chunk merge timed INSIDE a complete call, because the isolated H5 screen
    is demoted for anything touching cross-tile store-pipeline state;
  * arms interleaved, because a non-interleaved stage screen reports drift;
  * an A/A null pair built from a second module of the SAME source in the same
    square, because ABBA carries a per-window position offset the ordering does
    not cancel.
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
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--blocks', type=int, default=20)
    args = p.parse_args()
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
    result = {'initial_gpu': initial, 'layer': args.layer, 'probe': str(args.probe),
              'scope': 'Timing-only C4 store-scatter pricing, per-chunk merge measured '
                       'inside a complete call, interleaved, with an A/A null pair. '
                       'The probe writes to the wrong addresses by construction.'}

    def save():
        result.update(monitor_errors=errors, monitor_query_errors=query_errors,
                      monitor_samples=samples)
        args.output.write_text(json.dumps(result, indent=2))

    try:
        import torch
        from kernels.best.c4.operator import NativeFisherAttention, extension
        torch.set_grad_enabled(False); torch.set_num_threads(4)
        op = NativeFisherAttention(device='cuda', capacity=4,
                                   selector_backend='fused_cooperative_loop',
                                   selector_source=SELECTOR, merge_source=RETAINED)
        mods = {
            'retained': op.merge,
            'scatter_free': extension(4, source_path=args.probe, name_suffix='_lin'),
            # Second module of the SAME retained source: the A/A null arm.
            'retained_bis': extension(4, source_path=RETAINED, name_suffix='_bis'),
        }
        result['resources'] = {n: m.resources() for n, m in mods.items()}
        result['source_sha256'] = {n: m.source_sha256 for n, m in mods.items()}
        print('RESOURCES', json.dumps(result['resources']), flush=True)
        assert mods['retained'].source_sha256 == mods['retained_bis'].source_sha256, \
            'null arm must be the same source'
        save()

        payload = torch.load(CAPTURE.format(args.layer), map_location='cpu', weights_only=True)
        q, k, v = [payload[n].reshape(1, 21, 45, 80, 40, 128).cuda() for n in ('q', 'k', 'v')]
        del payload
        scale = 128 ** -.5

        def call_merge_total(module):
            """One complete call; returns summed merge time over the eight chunks."""
            op.merge = module
            marks = [torch.cuda.Event(enable_timing=True) for _ in range(17)]
            seeds = op.select(q, k, scale)
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
            return sum(marks[j].elapsed_time(marks[j + 1]) for j in range(0, 16, 2))

        for _ in range(3):
            call_merge_total(mods['retained'])
        torch.cuda.synchronize()

        def square(a, b, blocks):
            """Counterbalanced ABBA/BAAB over two arms, returning paired samples."""
            pairs = {a: [], b: []}
            for block in range(blocks):
                order = (a, b, b, a) if block % 2 == 0 else (b, a, a, b)
                for arm in order:
                    assert not errors, errors
                    pairs[arm].append(call_merge_total(mods[arm]))
            return pairs

        result['arms'] = {}
        for label, (a, b) in (('effect', ('retained', 'scatter_free')),
                              ('null', ('retained', 'retained_bis'))):
            pairs = square(a, b, args.blocks)
            n = min(len(pairs[a]), len(pairs[b]))
            deltas = [pairs[b][i] - pairs[a][i] for i in range(n)]
            mean = statistics.fmean(deltas)
            sd = statistics.stdev(deltas) if n > 1 else float('nan')
            entry = {'arm_a': a, 'arm_b': b,
                     'a_mean_ms': statistics.fmean(pairs[a]),
                     'b_mean_ms': statistics.fmean(pairs[b]),
                     'delta_mean_ms': mean, 'delta_sd_ms': sd, 'n': n,
                     't': mean / (sd / (n ** .5)) if sd and n > 1 else float('nan'),
                     'b_faster_count': sum(1 for d in deltas if d < 0),
                     'samples_a_ms': pairs[a], 'samples_b_ms': pairs[b]}
            result['arms'][label] = entry
            print(label.upper(), json.dumps({x: entry[x] for x in
                  ('a_mean_ms', 'b_mean_ms', 'delta_mean_ms', 'delta_sd_ms', 't',
                   'b_faster_count', 'n')}), flush=True)
            save()

        e, nl = result['arms']['effect'], result['arms']['null']
        result['scatter_price_ms_per_call'] = e['delta_mean_ms'] - nl['delta_mean_ms']
        print('SCATTER_PRICE_MS_PER_CALL (effect minus null offset)',
              round(result['scatter_price_ms_per_call'], 4), flush=True)
        result['valid'] = not errors
        print('SUMMARY_DONE', flush=True)
        save()
    finally:
        stop.set()
        save()


if __name__ == '__main__':
    main()
