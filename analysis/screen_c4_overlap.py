"""Overlap the C4 merge of one head chunk with the route of the previous chunk.

Arithmetic, kernels, launch shapes and head chunking are unchanged; only the
stream schedule and the number of compressed workspaces differ, so complete H40
output must stay bitwise equal to the serial retained operator. The motivation
is bandwidth composition, not occupancy: the merge is store bound and the route
is load bound, and HBM sustains more combined read+write than either alone.

Two compressed workspaces alternate. Chunk i merges on the merge stream while
chunk i-1 routes on the route stream; events order merge-before-route on the
same buffer and route-before-reuse two chunks later.
"""
import argparse, json, os, statistics, sys, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.launch_quality_first_videos import require_clean, gpu_processes, now

CAPTURE = '/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer{}.pt'
RETAINED = ROOT / 'kernels/best/c4/fisher_c4_guarded_softmax.cu'
SELECTOR = ROOT / 'kernels/best/c4/fisher_c4_fused_selector_register_basis_loop.cu'


def build(torch, NativeFisherAttention):
    class OverlappedFisherC4Attention(NativeFisherAttention):
        """Serial operator plus a second workspace and a two-stream schedule."""

        def __init__(self, device='cuda', **kwargs):
            super().__init__(device=device, **kwargs)
            self.workspaces = [(self.mk, self.mv, self.bias),
                               (torch.empty_like(self.mk), torch.empty_like(self.mv),
                                torch.empty_like(self.bias))]
            self.merge_stream = torch.cuda.Stream(device=self.packed.device)
            self.route_stream = torch.cuda.Stream(device=self.packed.device)
            self.merge_done = [torch.cuda.Event() for _ in self.workspaces]
            self.route_done = [torch.cuda.Event() for _ in self.workspaces]
            self.inputs_ready = torch.cuda.Event()

        @torch.no_grad()
        def __call__(self, q, k, v, scale=128 ** -.5, head_count=40):
            if head_count not in (5, 40):
                raise ValueError('Only diagnostic H5 or complete H40 calls are supported')
            seeds = self.select(q, k, scale)
            self.seeds = seeds
            self.merge.pack(q, self.packed)
            out = torch.empty_like(q)
            main = torch.cuda.current_stream(device=self.packed.device)
            self.inputs_ready.record(main)
            self.merge_stream.wait_event(self.inputs_ready)
            self.route_stream.wait_event(self.inputs_ready)
            started = []
            for index, head in enumerate(range(0, head_count, self.head_chunk)):
                active = min(self.head_chunk, head_count - head)
                slot = index % len(self.workspaces)
                self.mk, self.mv, self.bias = self.workspaces[slot]
                with torch.cuda.stream(self.merge_stream):
                    if index >= len(self.workspaces):
                        self.merge_stream.wait_event(self.route_done[slot])
                    self.merge.run(self.packed, k, v, seeds, self.mk, self.mv, self.bias,
                                   head, active, self.tiles, scale, self.empty, 0, 1)
                    self.merge_done[slot].record(self.merge_stream)
                with torch.cuda.stream(self.route_stream):
                    self.route_stream.wait_event(self.merge_done[slot])
                    self.route(q, k, v, out, head, scale, active_heads=active)
                    self.route_done[slot].record(self.route_stream)
                started.append(slot)
            for slot in set(started):
                main.wait_event(self.route_done[slot])
            self.mk, self.mv, self.bias = self.workspaces[0]
            return out

    return OverlappedFisherC4Attention


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--layer', action='append', default=[])
    p.add_argument('--blocks', type=int, default=8)
    args = p.parse_args()
    layers = args.layer or ['00']
    assert not args.output.exists()
    initial = require_clean(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = initial['gpu']['uuid']
    os.environ['TORCH_EXTENSIONS_DIR'] = str(args.output.parent / 'torch_extensions')
    os.environ['TRITON_CACHE_DIR'] = str(args.output.parent / 'triton_cache')
    os.environ['OMP_NUM_THREADS'] = '4'
    args.output.parent.mkdir(parents=True, exist_ok=True)

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
    result = {'initial_gpu': initial, 'layers': layers, 'target_ms': 80,
              'scope': 'Complete H40 C4 call; two-stream merge/route overlap with two '
                       'compressed workspaces against the serial retained operator. '
                       'Kernels and arithmetic are unchanged.',
              'timing_protocol': f'{args.blocks} alternating ABBA/BAAB blocks, '
                                 '5 complete calls per method per block per layer'}

    def save():
        result.update(monitor_errors=errors, monitor_samples=samples)
        args.output.write_text(json.dumps(result, indent=2))

    try:
        import torch
        from kernels.best.c4.operator import NativeFisherAttention
        torch.set_grad_enabled(False); torch.set_num_threads(4)
        settings = dict(capacity=4, selector_backend='fused_cooperative_loop',
                        selector_source=SELECTOR, merge_source=RETAINED)
        serial = NativeFisherAttention(device='cuda', **settings)
        Overlapped = build(torch, NativeFisherAttention)
        overlapped = Overlapped(device='cuda', **settings)
        result['resources'] = serial.merge.resources()
        result['source_sha256'] = serial.merge.source_sha256
        result['workspace_bytes'] = int(serial.mk.numel() * 2 * 2 + serial.bias.numel() * 4)
        print('READY', json.dumps(result['resources']), flush=True)
        save()

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

        result['per_layer'] = {}
        for layer in layers:
            payload = torch.load(CAPTURE.format(layer), map_location='cpu', weights_only=True)
            q, k, v = [payload[n].reshape(1, 21, 45, 80, 40, 128).cuda() for n in ('q', 'k', 'v')]
            del payload
            row = {}
            reference = serial(q, k, v).clone()
            produced = overlapped(q, k, v).clone()
            torch.cuda.synchronize()
            difference = produced.float() - reference.float()
            row['correctness'] = {
                'bitwise_equal': bool(torch.equal(produced, reference)),
                'max_abs': float(difference.abs().max()),
                'relative_l2': float((difference.square().sum()
                                      / reference.float().square().sum()).sqrt()),
                'finite': bool(torch.isfinite(produced).all())}
            print('CORRECTNESS', layer, json.dumps(row['correctness']), flush=True)
            del reference, produced, difference
            if not row['correctness']['bitwise_equal']:
                row['timing_skipped'] = 'overlapped schedule is not bitwise equal'
                result['per_layer'][layer] = row; save()
                del q, k, v; torch.cuda.empty_cache(); continue

            pairs = {'serial': [], 'overlapped': []}
            operators = {'serial': serial, 'overlapped': overlapped}
            for block in range(args.blocks):
                order = ('serial', 'overlapped') if block % 2 == 0 else ('overlapped', 'serial')
                for name in order:
                    pairs[name] += timed(lambda n=name: operators[n](q, k, v))
            row['timing_ms'] = {n: {'mean_ms': statistics.fmean(s), 'samples_ms': s}
                                for n, s in pairs.items()}
            row['paired_delta_ms'] = (row['timing_ms']['overlapped']['mean_ms']
                                      - row['timing_ms']['serial']['mean_ms'])
            row['paired_wins'] = sum(int(o < s) for o, s in
                                     zip(pairs['overlapped'], pairs['serial']))
            row['paired_count'] = len(pairs['overlapped'])
            print('LAYER', layer, 'delta_ms', row['paired_delta_ms'],
                  'wins', row['paired_wins'], '/', row['paired_count'], flush=True)
            result['per_layer'][layer] = row
            save()
            del q, k, v
            torch.cuda.empty_cache()

        rows = [r for r in result['per_layer'].values() if 'paired_delta_ms' in r]
        if rows:
            result['summary'] = {
                'overlapped_mean_ms': statistics.fmean(
                    [x for r in rows for x in r['timing_ms']['overlapped']['samples_ms']]),
                'serial_mean_ms': statistics.fmean(
                    [x for r in rows for x in r['timing_ms']['serial']['samples_ms']]),
                'paired_wins': sum(r['paired_wins'] for r in rows),
                'paired_count': sum(r['paired_count'] for r in rows)}
            result['summary']['delta_ms'] = (result['summary']['overlapped_mean_ms']
                                             - result['summary']['serial_mean_ms'])
            print('SUMMARY', json.dumps(result['summary']), flush=True)
        result['valid'] = not errors
        save()
    finally:
        stop.set()
        save()


if __name__ == '__main__':
    main()
