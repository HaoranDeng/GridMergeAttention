"""Locate the gap between the C4 stage sum and the complete call.

Stage timings are measured by replaying one head chunk, which leaves that
chunk's inputs hot. A complete call sweeps eight different chunks. This script
times every stage in place inside a complete call, so per-chunk merge and route
costs are measured under the real cache and scheduling conditions, and compares
the total against both an ordinary call and a CUDA-graph replay.
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
    p.add_argument('--repeats', type=int, default=8)
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
                # A transient nvidia-smi timeout is not evidence of contamination;
                # only an actual foreign compute process invalidates timing.
                query_errors.append(repr(exc))
            stop.wait(3)

    threading.Thread(target=monitor, daemon=True).start()
    result = {'initial_gpu': initial, 'layer': args.layer,
              'scope': 'In-place stage timing inside a complete C4 call, against an '
                       'ordinary call and a CUDA-graph replay. No candidate is promoted.'}

    def save():
        result.update(monitor_errors=errors, monitor_query_errors=query_errors,
                      monitor_samples=samples)
        args.output.write_text(json.dumps(result, indent=2))

    try:
        import torch
        from kernels.best.c4.operator import NativeFisherAttention
        torch.set_grad_enabled(False); torch.set_num_threads(4)
        payload = torch.load(CAPTURE.format(args.layer), map_location='cpu', weights_only=True)
        q, k, v = [payload[n].reshape(1, 21, 45, 80, 40, 128).cuda() for n in ('q', 'k', 'v')]
        del payload
        op = NativeFisherAttention(device='cuda', capacity=4,
                                   selector_backend='fused_cooperative_loop',
                                   selector_source=SELECTOR, merge_source=RETAINED)
        scale = 128 ** -.5

        def instrumented():
            """One complete call with an event between every kernel."""
            marks = [torch.cuda.Event(enable_timing=True) for _ in range(2 + 2 * 8 + 1)]
            index = 0
            marks[index].record(); index += 1
            seeds = op.select(q, k, scale)
            marks[index].record(); index += 1
            op.merge.pack(q, op.packed)
            marks[index].record(); index += 1
            out = torch.empty_like(q)
            for head in range(0, 40, op.head_chunk):
                op.merge.run(op.packed, k, v, seeds, op.mk, op.mv, op.bias, head,
                             op.head_chunk, op.tiles, scale, op.empty, 0, 1)
                marks[index].record(); index += 1
                op.route(q, k, v, out, head, scale, active_heads=op.head_chunk)
                marks[index].record(); index += 1
            marks[-1].synchronize()
            spans = [marks[i].elapsed_time(marks[i + 1]) for i in range(index - 1)]
            return {'selector': spans[0], 'pack': spans[1],
                    'merge': spans[2::2], 'route': spans[3::2],
                    'total': marks[0].elapsed_time(marks[index - 1])}

        def timed(fn, repeats):
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

        plain = timed(lambda: op(q, k, v), args.repeats)
        result['plain_call_ms'] = {'mean_ms': statistics.fmean(plain), 'samples_ms': plain}
        print('PLAIN', result['plain_call_ms']['mean_ms'], flush=True)

        for _ in range(2):
            instrumented()
        torch.cuda.synchronize()
        runs = [instrumented() for _ in range(args.repeats)]
        result['in_call_stages'] = {
            'selector_ms': statistics.fmean(r['selector'] for r in runs),
            'pack_ms': statistics.fmean(r['pack'] for r in runs),
            'merge_per_chunk_ms': [statistics.fmean(r['merge'][i] for r in runs) for i in range(8)],
            'route_per_chunk_ms': [statistics.fmean(r['route'][i] for r in runs) for i in range(8)],
            'instrumented_total_ms': statistics.fmean(r['total'] for r in runs)}
        s = result['in_call_stages']
        s['merge_total_ms'] = sum(s['merge_per_chunk_ms'])
        s['route_total_ms'] = sum(s['route_per_chunk_ms'])
        s['stage_sum_ms'] = (s['selector_ms'] + s['pack_ms']
                             + s['merge_total_ms'] + s['route_total_ms'])
        print('IN_CALL', json.dumps({x: s[x] for x in
              ('selector_ms', 'pack_ms', 'merge_total_ms', 'route_total_ms',
               'stage_sum_ms', 'instrumented_total_ms')}), flush=True)
        print('MERGE_PER_CHUNK', [round(x, 3) for x in s['merge_per_chunk_ms']], flush=True)
        print('ROUTE_PER_CHUNK', [round(x, 3) for x in s['route_per_chunk_ms']], flush=True)
        save()

        # CUDA-graph replay removes every per-launch cost from the timed region.
        try:
            op(q, k, v)
            torch.cuda.synchronize()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    op(q, k, v)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = op(q, k, v)
            torch.cuda.synchronize()
            replay = timed(lambda: graph.replay(), args.repeats)
            result['graph_call_ms'] = {'mean_ms': statistics.fmean(replay), 'samples_ms': replay}
            reference = op(q, k, v)
            graph.replay(); torch.cuda.synchronize()
            result['graph_bitwise_equal'] = bool(torch.equal(captured, reference))
            print('GRAPH', result['graph_call_ms']['mean_ms'],
                  'bitwise_equal', result['graph_bitwise_equal'], flush=True)
        except Exception as exc:
            result['graph_error'] = repr(exc)
            print('GRAPH_FAILED', repr(exc), flush=True)
        result['valid'] = not errors
        save()
    finally:
        stop.set()
        save()


if __name__ == '__main__':
    main()
