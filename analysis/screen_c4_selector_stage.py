"""Re-measure the C4 selector regpipe with an instrument that is actually valid.

The first attempt produced two readings that disagreed, and both were unsound:

  * the selector stage was timed by replaying each arm in sequence, not
    interleaved, and a non-interleaved stage screen reports drift as effect;
  * the complete-call ABBA had n=100, which cannot resolve a sub-1 ms change.

This times the selector stage IN PLACE inside a complete call, alternates the
arms within the square, and carries an A/A null pair built from a second module
of the same source so the window's position offset is measured rather than
assumed. Seeds and complete output are gated bitwise first.
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
    p.add_argument('--layer', default='00')
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--blocks', type=int, default=30)
    p.add_argument('--allow-mismatch', action='store_true',
                   help='Report seed mismatches and output error instead of '
                        'refusing to time. The FP32 Gram-Schmidt candidate exists '
                        'precisely to find out whether the seeds move at all.')
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
    result = {'initial_gpu': initial, 'layer': args.layer, 'candidate': str(args.candidate),
              'scope': 'C4 selector stage timed in place inside a complete call, arms '
                       'interleaved, with an A/A null pair in the same square.'}

    def save():
        result.update(monitor_errors=errors, monitor_query_errors=query_errors,
                      monitor_samples=samples)
        args.output.write_text(json.dumps(result, indent=2))

    try:
        import torch
        from kernels.best.c4.operator import NativeFisherAttention
        from kernels.best.c4.fused_selector import NativeFusedFisherSelector
        torch.set_grad_enabled(False); torch.set_num_threads(4)
        op = NativeFisherAttention(device='cuda', capacity=4,
                                   selector_backend='fused_cooperative_loop',
                                   selector_source=RETAINED_SELECTOR,
                                   merge_source=RETAINED_MERGE)
        sel = {
            'retained': op.native_selector,
            'candidate': NativeFusedFisherSelector(4, device='cuda', loop_frames=True,
                                                 cooperative=True, source_path=args.candidate),
            'retained_bis': NativeFusedFisherSelector(4, device='cuda', loop_frames=True,
                                                      cooperative=True,
                                                      source_path=RETAINED_SELECTOR),
        }
        result['source_sha256'] = {n: s.extension.source_sha256 for n, s in sel.items()}
        assert result['source_sha256']['retained'] == result['source_sha256']['retained_bis'], \
            'null arm must be the same source'
        print('READY', json.dumps(result['source_sha256']), flush=True)

        payload = torch.load(CAPTURE.format(args.layer), map_location='cpu', weights_only=True)
        q, k, v = [payload[n].reshape(1, 21, 45, 80, 40, 128).cuda() for n in ('q', 'k', 'v')]
        del payload
        scale = 128 ** -.5

        op.native_selector = sel['retained']
        reference = op(q, k, v).clone()
        reference_seeds = sel['retained'](q[:, :1], k, scale).clone()
        torch.cuda.synchronize()
        result['correctness'] = {}
        for name in ('candidate', 'retained_bis'):
            op.native_selector = sel[name]
            s = sel[name](q[:, :1], k, scale).clone()
            o = op(q, k, v).clone()
            result['correctness'][name] = {
                'seeds_bitwise_equal': bool(torch.equal(s, reference_seeds)),
                'output_bitwise_equal': bool(torch.equal(o, reference)),
                'seed_mismatches': int((s != reference_seeds).sum()),
                'seed_total': int(reference_seeds.numel()),
                'seed_mismatch_fraction': float((s != reference_seeds).sum())
                                          / reference_seeds.numel(),
                'output_max_abs': float((o.float() - reference.float()).abs().max()),
                'output_relative_l2': float(((o.float() - reference.float()).square().sum()
                                            / reference.float().square().sum()).sqrt())}
            print('CORRECTNESS', name, json.dumps(result['correctness'][name]), flush=True)
            del s, o
        if not args.allow_mismatch:
            assert result['correctness']['candidate']['seeds_bitwise_equal'] and \
                   result['correctness']['candidate']['output_bitwise_equal'], \
                   'not bitwise equal; pass --allow-mismatch to measure it anyway'
        save()

        def call_selector_stage(name):
            """One complete call; returns the selector stage time inside it."""
            op.native_selector = sel[name]
            a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            seeds = op.select(q, k, scale)
            b.record()
            op.merge.pack(q, op.packed)
            out = torch.empty_like(q)
            for head in range(0, 40, op.head_chunk):
                op.merge.run(op.packed, k, v, seeds, op.mk, op.mv, op.bias, head,
                             op.head_chunk, op.tiles, scale, op.empty, 0, 1)
                op.route(q, k, v, out, head, scale, active_heads=op.head_chunk)
            torch.cuda.synchronize()
            return a.elapsed_time(b)

        for _ in range(3):
            call_selector_stage('retained')
        torch.cuda.synchronize()

        def square(a, b):
            pairs = {a: [], b: []}
            for block in range(args.blocks):
                order = (a, b, b, a) if block % 2 == 0 else (b, a, a, b)
                for arm in order:
                    assert not errors, errors
                    pairs[arm].append(call_selector_stage(arm))
            return pairs

        result['arms'] = {}
        for label, (a, b) in (('effect', ('retained', 'candidate')),
                              ('null', ('retained', 'retained_bis'))):
            pairs = square(a, b)
            n = min(len(pairs[a]), len(pairs[b]))
            deltas = [pairs[b][i] - pairs[a][i] for i in range(n)]
            mean = statistics.fmean(deltas)
            sd = statistics.stdev(deltas) if n > 1 else float('nan')
            entry = {'arm_a': a, 'arm_b': b,
                     'a_mean_ms': statistics.fmean(pairs[a]),
                     'b_mean_ms': statistics.fmean(pairs[b]),
                     'delta_mean_ms': mean, 'delta_sd_ms': sd, 'n': n,
                     't': mean / (sd / (n ** .5)) if sd and n > 1 else float('nan'),
                     'b_faster_count': sum(1 for d in deltas if d < 0)}
            result['arms'][label] = entry
            print(label.upper(), json.dumps({x: entry[x] for x in
                  ('a_mean_ms', 'b_mean_ms', 'delta_mean_ms', 'delta_sd_ms', 't',
                   'b_faster_count', 'n')}), flush=True)
            save()

        e, nl = result['arms']['effect'], result['arms']['null']
        result['candidate_delta_ms'] = e['delta_mean_ms'] - nl['delta_mean_ms']
        print('SELECTOR_DELTA_MS (effect minus null offset)',
              round(result['candidate_delta_ms'], 4), flush=True)
        result['valid'] = not errors
        print('SUMMARY_DONE', flush=True)
        save()
    finally:
        stop.set()
        save()


if __name__ == '__main__':
    main()
