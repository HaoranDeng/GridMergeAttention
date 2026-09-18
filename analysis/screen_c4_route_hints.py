"""Price L2 eviction hints on the C4 Triton route, per chunk, inside complete calls.

The route is a third of the call and streams 16.3 GB per five-head chunk at
about 88% of the achievable read bandwidth. Its one-shot compressed stream and
its L2-reusable singleton pass carry the same cache policy, so the hint
variants built by `build_c4_route_hints.py` are the first thing that tells L2
which lines to keep. A hint changes no value, so every arm is gated on bitwise
equality of the complete H40 output against the retained route.

Instrument, each choice forced by a recorded failure:
  * per-chunk route timed INSIDE a complete call (events around the eight route
    launches), because isolated replays of one hot chunk mis-state the stage;
  * arms interleaved ABBA/BAAB, because a non-interleaved stage screen drifts;
  * an A/A null arm - a byte-identical copy of the route module loaded under
    another name - in the same square, because ABBA carries a position offset.

The base stack is the promoted pair (desc_select merge, FP32 GS register-
pipelined selector), since that is what the route will ship with.
"""
import argparse, hashlib, importlib.util, json, os, socket, statistics, sys, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.launch_quality_first_videos import require_clean, gpu_processes, now

CAPTURE = '/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer{}.pt'
MERGE = ROOT / 'native_fisher/fisher_c4_desc_select.cu'
SELECTOR = ROOT / 'native_fisher/fisher_c4_selector_v9c_halving_mxredux.cu'


def load_route_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.source_sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant', type=Path, action='append', required=True,
                   help='Repeatable route module path; the arm name is the file stem.')
    p.add_argument('--null', type=Path, default=ROOT / 'native_fisher/route_c4_hint_null.py',
                   help='Byte-identical copy of the retained route for the A/A arm.')
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
              'variants': [str(v) for v in a.variant],
              'scope': 'Per-chunk route time inside complete C4 calls, promoted merge and '
                       'selector held fixed, route module swapped per arm, interleaved '
                       'ABBA/BAAB with an A/A null arm; complete-output bitwise gate.'}

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
        from kernels.best.c4 import route_reference as retained_route
        torch.set_grad_enabled(False); torch.set_num_threads(4)
        op = NativeFisherAttention(device='cuda', capacity=4, selector_backend='fused_cooperative_loop',
                                   selector_source=SELECTOR, merge_source=MERGE)
        retained_route.source_sha256 = hashlib.sha256(
            Path(retained_route.__file__).read_bytes()).hexdigest()
        routes = {'retained': retained_route,
                  'null': load_route_module(a.null, 'route_c4_null_arm')}
        for v in a.variant:
            routes[v.stem.replace('route_c4_hint_', '')] = load_route_module(v, 'route_c4_arm_' + v.stem)
        # TMA-descriptor variants build their descriptors in-kernel and need
        # Triton's global-scratch allocator installed in the launching thread
        # before their first launch; the variant module exposes that as
        # prepare_tma(). Harmless to call on more than one module.
        for name, module in routes.items():
            if hasattr(module, 'prepare_tma'):
                module.prepare_tma()
                result.setdefault('prepared_tma', []).append(name)
                print('PREPARED_TMA', name, flush=True)
        result['route_sha256'] = {n: m.source_sha256 for n, m in routes.items()}
        assert result['route_sha256']['null'] == result['route_sha256']['retained'], \
            'the null arm must be byte-identical to the retained route'
        result['merge_sha256'] = op.merge.source_sha256
        result['selector_sha256'] = op.native_selector.extension.source_sha256
        print('ROUTES', json.dumps(result['route_sha256']), flush=True)
        save()

        # Host-built TMA descriptor variants replace the two compressed-stream
        # pointer arguments with descriptors made once from the workspace tensors.
        descriptors = {}
        for name, module in routes.items():
            if hasattr(module, 'make_descriptors'):
                descriptors[name] = module.make_descriptors(op.mk, op.mv)
                result.setdefault('host_descriptors', []).append(name)
                print('HOST_DESCRIPTORS', name, flush=True)
        by_module = {id(m): n for n, m in routes.items()}

        def route_with(h, q, k, v, out, head, scale, active_heads=5):
            # operator.route(), with the kernel and its constants taken from `h`.
            mk, mv = descriptors.get(by_module.get(id(h)), (op.mk, op.mv))
            h._hard_c4_720p_route_kernel[(1, active_heads, 21 * 80)](
                q, k, v, mk, mv, op.unused, op.unused, op.bias, op.selected, op.unused, out,
                21, 21, 40, active_heads, head, 128, False, False, 45, 80, op.capacity,
                h.HARD_720_ROUTE_BLOCK_I, op.route_block, op.route_stages, h.HARD_720_INPUT_PRECISION,
                False, 0, False, 0, SINGLETON_TOP_R=1, SINGLETON_QUERY_COLUMN=True,
                SINGLETON_SECONDARY_FRAME_STRIDE=1, SINGLETON_PRIMARY_ROW_MOD4_OMIT2=False,
                QUERY_COLUMN_MAJOR=True, PACKED_SINGLETON_KV=False, SCALED_FP8_KV=False,
                COMPRESSED_FP8=False, SCALE=scale, num_warps=op.route_warps,
                num_stages=h.HARD_720_ROUTE_NUM_STAGES)

        scale = 128 ** -.5

        def call(arm, q, k, v):
            """One complete call; returns (output, summed route ms over the eight chunks)."""
            h = routes[arm]
            marks = [torch.cuda.Event(enable_timing=True) for _ in range(16)]
            seeds = op.select(q, k, scale); op.seeds = seeds
            op.merge.pack(q, op.packed)
            out = torch.empty_like(q)
            i = 0
            for head in range(0, 40, op.head_chunk):
                op.merge.run(op.packed, k, v, seeds, op.mk, op.mv, op.bias, head,
                             op.head_chunk, op.tiles, scale, op.empty, 0, 1)
                marks[i].record(); i += 1
                route_with(h, q, k, v, out, head, scale, active_heads=op.head_chunk)
                marks[i].record(); i += 1
            marks[i - 1].synchronize()
            return out, sum(marks[j].elapsed_time(marks[j + 1]) for j in range(0, 16, 2))

        result['correctness'] = {}
        result['arms'] = {}
        for layer in layers:
            payload = torch.load(CAPTURE.format(layer), map_location='cpu', weights_only=True)
            q, k, v = [payload[n].reshape(1, 21, 45, 80, 40, 128).cuda() for n in ('q', 'k', 'v')]
            del payload
            reference, _ = call('retained', q, k, v)
            reference = reference.clone()
            row = {}
            for arm in list(routes):
                if arm == 'retained':
                    continue
                try:
                    produced, _ = call(arm, q, k, v)   # first call also compiles the variant
                except Exception as exc:
                    # A variant that fails to compile or launch is dropped from the
                    # square and recorded; the other arms still get measured.
                    message = repr(exc)[:4000]
                    row[arm] = {'error': message}
                    result.setdefault('arm_errors', {})[arm] = message
                    print('ARM_ERROR', layer, arm, message[:400], flush=True)
                    del routes[arm]
                    torch.cuda.synchronize()
                    continue
                difference = produced.float() - reference.float()
                row[arm] = {'output_bitwise_equal': bool(torch.equal(produced, reference)),
                            'finite': bool(torch.isfinite(produced).all()),
                            'max_abs': float(difference.abs().max()),
                            'relative_l2': float((difference.square().sum()
                                                  / reference.float().square().sum()).sqrt())}
                del produced, difference
                print('CORRECTNESS', layer, arm, json.dumps(row[arm]), flush=True)
            result['correctness'][layer] = row
            assert row['null']['output_bitwise_equal'], 'the null arm must reproduce the retained output'
            if not a.allow_mismatch:
                for arm, c in row.items():
                    if 'error' in c:
                        continue
                    assert c['output_bitwise_equal'], '%s is not bitwise; pass --allow-mismatch' % arm
            del reference
            save()

            for _ in range(3):
                for arm in routes:
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

            entry = {}
            entry['null'] = stats(square('retained', 'null'), 'retained', 'null')
            print('NULL', layer, json.dumps({k: entry['null'][k] for k in
                  ('a_mean_ms', 'b_mean_ms', 'delta_mean_ms', 'delta_sd_ms', 't', 'b_faster_count', 'n')}),
                  flush=True)
            for arm in routes:
                if arm in ('retained', 'null'):
                    continue
                e = stats(square('retained', arm), 'retained', arm)
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
        for arm in routes:
            if arm in ('retained', 'null'):
                continue
            per = {l: result['arms'][l][arm]['delta_offset_corrected_ms'] for l in result['arms']}
            summary[arm] = {'pooled_route_delta_ms_per_call': statistics.fmean(per.values()),
                            'per_layer_ms': per,
                            'wins': sum(result['arms'][l][arm]['b_faster_count'] for l in result['arms']),
                            'n': sum(result['arms'][l][arm]['n'] for l in result['arms']),
                            'bitwise_all_layers': all(result['correctness'][l].get(arm, {}).get('output_bitwise_equal', False)
                                                      for l in result['correctness'])}
        summary['retained_route_ms_per_call'] = statistics.fmean(
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
