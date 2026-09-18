"""Measure the C4 stack end to end: both retained kernels against both candidates.

The two accepted changes were measured apart - `desc_select` on the complete
call, the FP32/register-pipelined selector at its own stage inside a complete
call. Their sum is a projection, and a projection is not a number: the merge and
the selector share the same L2 and the same launch queue, and nothing so far has
run them together in one binary.

A complete-call ABBA is normally blind below about 2 ms (`abba-gate-power`). The
combined change is projected at roughly -3.25 ms, which is above that floor, so
here the complete call is the right instrument rather than a stage probe. An A/A
null pair built from second modules of the same retained sources rides in the
same square, so the window's position offset is measured rather than assumed.

Numerics are reported, never asserted: the selector arm is a declared numerics
change and its seed flips are layer-dependent, so all three captured layers are
checked and their counts recorded.
"""
import argparse, json, os, socket, statistics, sys, threading
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
    p.add_argument('--candidate-merge', type=Path, required=True)
    p.add_argument('--candidate-selector', type=Path, required=True)
    p.add_argument('--layer', action='append', default=[])
    p.add_argument('--blocks', type=int, default=15)
    p.add_argument('--allow-mismatch', action='store_true',
                   help='Record numeric differences instead of refusing to time. '
                        'Required for the FP32 selector, which is a declared '
                        'numerics change.')
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
    result = {'initial_gpu': initial, 'layers': layers,
              'candidate_merge': str(a.candidate_merge),
              'candidate_selector': str(a.candidate_selector),
              'scope': 'Complete-call paired timing of the retained C4 pair against '
                       'both candidate kernels together, with an A/A null pair in the '
                       'same square. Numerics recorded on every captured layer.'}

    def save():
        result.update(monitor_errors=errors, monitor_samples=samples)
        a.output.write_text(json.dumps(result, indent=2))

    try:
        import torch
        from kernels.best.c4.operator import NativeFisherAttention, extension
        from kernels.best.c4.fused_selector import NativeFusedFisherSelector
        torch.set_grad_enabled(False); torch.set_num_threads(4)
        op = NativeFisherAttention(device='cuda', capacity=4,
                                   selector_backend='fused_cooperative_loop',
                                   selector_source=RETAINED_SELECTOR,
                                   merge_source=RETAINED_MERGE)
        # One workspace, swapped kernels: three operators would cost 48 GB of
        # compressed K/V for no benefit, and a second workspace would change the
        # allocator state the timing is meant to hold fixed.
        merges = {'retained': op.merge,
                  'candidate': extension(4, source_path=a.candidate_merge, name_suffix='_cand'),
                  'retained_bis': extension(4, source_path=RETAINED_MERGE, name_suffix='_bis')}
        selectors = {'retained': op.native_selector,
                     'candidate': NativeFusedFisherSelector(4, device='cuda', loop_frames=True,
                                                            cooperative=True,
                                                            source_path=a.candidate_selector),
                     'retained_bis': NativeFusedFisherSelector(4, device='cuda', loop_frames=True,
                                                               cooperative=True,
                                                               source_path=RETAINED_SELECTOR)}
        result['resources'] = {n: m.resources() for n, m in merges.items()}
        result['merge_sha256'] = {n: m.source_sha256 for n, m in merges.items()}
        result['selector_sha256'] = {n: s.extension.source_sha256 for n, s in selectors.items()}
        assert result['merge_sha256']['retained'] == result['merge_sha256']['retained_bis']
        assert result['selector_sha256']['retained'] == result['selector_sha256']['retained_bis']
        print('SOURCES', json.dumps({'merge': result['merge_sha256'],
                                     'selector': result['selector_sha256']}), flush=True)
        save()

        def use(arm):
            op.merge = merges[arm]
            op.native_selector = selectors[arm]

        scale = 128 ** -.5
        result['correctness'] = {}
        result['arms'] = {}
        for layer in layers:
            payload = torch.load(CAPTURE.format(layer), map_location='cpu', weights_only=True)
            q, k, v = [payload[n].reshape(1, 21, 45, 80, 40, 128).cuda() for n in ('q', 'k', 'v')]
            del payload

            use('retained')
            reference = op(q, k, v).clone()
            reference_seeds = op.seeds.clone()
            torch.cuda.synchronize()
            row = {}
            for name in ('candidate', 'retained_bis'):
                use(name)
                produced = op(q, k, v).clone()
                seeds = op.seeds
                difference = produced.float() - reference.float()
                row[name] = {
                    'seeds_bitwise_equal': bool(torch.equal(seeds, reference_seeds)),
                    'output_bitwise_equal': bool(torch.equal(produced, reference)),
                    'seed_mismatches': int((seeds != reference_seeds).sum()),
                    'seed_total': int(reference_seeds.numel()),
                    'seed_mismatch_fraction': float((seeds != reference_seeds).sum())
                                              / reference_seeds.numel(),
                    'finite': bool(torch.isfinite(produced).all()),
                    'max_abs': float(difference.abs().max()),
                    'relative_l2': float((difference.square().sum()
                                          / reference.float().square().sum()).sqrt())}
                del produced, difference
                print('CORRECTNESS', layer, name, json.dumps(row[name]), flush=True)
            result['correctness'][layer] = row
            if not a.allow_mismatch:
                assert row['candidate']['output_bitwise_equal'], \
                    'not bitwise equal; pass --allow-mismatch to time it anyway'
            assert row['retained_bis']['output_bitwise_equal'], \
                'the null arm must reproduce the retained result exactly'
            del reference, reference_seeds
            save()

            def timed(arm):
                use(arm)
                s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                s.record(); y = op(q, k, v); e.record(); e.synchronize()
                del y
                return s.elapsed_time(e)

            for _ in range(3):
                timed('retained')
            torch.cuda.synchronize()

            def square(x, y):
                pairs = {x: [], y: []}
                for block in range(a.blocks):
                    order = (x, y, y, x) if block % 2 == 0 else (y, x, x, y)
                    for arm in order:
                        assert not errors, errors
                        pairs[arm].append(timed(arm))
                return pairs

            entry = {}
            for label, (x, y) in (('effect', ('retained', 'candidate')),
                                  ('null', ('retained', 'retained_bis'))):
                pairs = square(x, y)
                n = min(len(pairs[x]), len(pairs[y]))
                deltas = [pairs[y][i] - pairs[x][i] for i in range(n)]
                mean = statistics.fmean(deltas)
                sd = statistics.stdev(deltas) if n > 1 else float('nan')
                entry[label] = {'arm_a': x, 'arm_b': y,
                                'a_mean_ms': statistics.fmean(pairs[x]),
                                'b_mean_ms': statistics.fmean(pairs[y]),
                                'delta_mean_ms': mean, 'delta_sd_ms': sd, 'n': n,
                                't': mean / (sd / n ** .5) if sd and n > 1 else float('nan'),
                                'b_faster_count': sum(1 for d in deltas if d < 0)}
                print(label.upper(), layer, json.dumps(entry[label]), flush=True)
            entry['candidate_delta_ms'] = (entry['effect']['delta_mean_ms']
                                           - entry['null']['delta_mean_ms'])
            entry['retained_call_ms'] = entry['effect']['a_mean_ms']
            entry['candidate_call_ms'] = entry['effect']['b_mean_ms']
            result['arms'][layer] = entry
            print('LAYER', layer, 'FULL_CALL retained', round(entry['retained_call_ms'], 3),
                  'candidate', round(entry['candidate_call_ms'], 3),
                  'delta (offset corrected)', round(entry['candidate_delta_ms'], 4), flush=True)
            del q, k, v
            torch.cuda.empty_cache()
            save()

        deltas = [e['candidate_delta_ms'] for e in result['arms'].values()]
        result['summary'] = {
            'pooled_delta_ms': statistics.fmean(deltas),
            'per_layer_delta_ms': {l: e['candidate_delta_ms'] for l, e in result['arms'].items()},
            'retained_call_ms': statistics.fmean(e['retained_call_ms'] for e in result['arms'].values()),
            'candidate_call_ms': statistics.fmean(e['candidate_call_ms'] for e in result['arms'].values())}
        print('SUMMARY', json.dumps(result['summary']), flush=True)
        result['valid'] = not errors
        print('SUMMARY_DONE', flush=True)
        save()
    finally:
        stop.set(); save()


if __name__ == '__main__':
    main()
