"""Test whether a C4 route kernel can co-reside with the merge, and what it costs.

`gma-merge-route-overlap-dead-end` closed stream overlap for every GMA capacity,
but its own body records the decisive variant as untested:

    (a) SM-level co-residency - cap merge at 3 CTAs/SM *and* near 112 registers
    so route's 49 KB shared / 20 352 registers fit alongside (untested)

Merge at 4 CTAs/SM consumes 127x128x4 = 65024 of 65536 registers, so no route
CTA can be placed. Padding the merge's dynamic shared request caps residency
without touching a line of the code path, freeing both block slots and
registers. This measures two things:

  1. C4's occupancy cost curve - merge_total at 4, 3 and 2 CTAs per SM;
  2. serial versus overlapped complete calls at each padding, so the overlap is
     compared against a control with the SAME padding and the padding cost is
     not charged to the overlap.

The padding's premise is verified on-GPU, not assumed: an arm whose measured
CTAs/SM is not the intended value is recorded and never timed.
"""
import argparse, json, os, statistics, sys, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.launch_quality_first_videos import require_clean, gpu_processes, now

CAPTURE = '/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer{}.pt'
RETAINED = ROOT / 'kernels/best/c4/fisher_c4_guarded_softmax.cu'
SELECTOR = ROOT / 'kernels/best/c4/fisher_c4_fused_selector_register_basis_loop.cu'


def build_dual(torch, base):
    class Dual(base):
        """Serial and two-stream schedules over two compressed workspaces."""

        def __init__(self, device='cuda', **kw):
            super().__init__(device=device, **kw)
            self.spaces = [(self.mk, self.mv, self.bias),
                           (torch.empty_like(self.mk), torch.empty_like(self.mv),
                            torch.empty_like(self.bias))]
            self.sm = torch.cuda.Stream(device=self.packed.device)
            self.sr = torch.cuda.Stream(device=self.packed.device)
            self.merge_done = [torch.cuda.Event() for _ in self.spaces]
            self.route_done = [torch.cuda.Event() for _ in self.spaces]
            self.ready = torch.cuda.Event()

        def _prologue(self, q, k, v, scale):
            seeds = self.select(q, k, scale)
            self.merge.pack(q, self.packed)
            return seeds, torch.empty_like(q)

        @torch.no_grad()
        def serial_call(self, q, k, v, scale=128 ** -.5, timed_merge=False):
            seeds, out = self._prologue(q, k, v, scale)
            self.mk, self.mv, self.bias = self.spaces[0]
            marks = [torch.cuda.Event(enable_timing=True) for _ in range(16)] if timed_merge else None
            i = 0
            for head in range(0, 40, self.head_chunk):
                if timed_merge:
                    marks[i].record(); i += 1
                self.merge.run(self.packed, k, v, seeds, self.mk, self.mv, self.bias,
                               head, self.head_chunk, self.tiles, scale, self.empty, 0, 1)
                if timed_merge:
                    marks[i].record(); i += 1
                self.route(q, k, v, out, head, scale, active_heads=self.head_chunk)
            if timed_merge:
                marks[15].synchronize()
                return out, sum(marks[j].elapsed_time(marks[j + 1]) for j in range(0, 16, 2))
            return out, None

        @torch.no_grad()
        def overlapped_call(self, q, k, v, scale=128 ** -.5):
            seeds, out = self._prologue(q, k, v, scale)
            main = torch.cuda.current_stream(device=self.packed.device)
            self.ready.record(main)
            self.sm.wait_event(self.ready); self.sr.wait_event(self.ready)
            for index, head in enumerate(range(0, 40, self.head_chunk)):
                slot = index % len(self.spaces)
                self.mk, self.mv, self.bias = self.spaces[slot]
                with torch.cuda.stream(self.sm):
                    if index >= len(self.spaces):
                        self.sm.wait_event(self.route_done[slot])
                    self.merge.run(self.packed, k, v, seeds, self.mk, self.mv, self.bias,
                                   head, self.head_chunk, self.tiles, scale, self.empty, 0, 1)
                    self.merge_done[slot].record(self.sm)
                with torch.cuda.stream(self.sr):
                    self.sr.wait_event(self.merge_done[slot])
                    self.route(q, k, v, out, head, scale, active_heads=self.head_chunk)
                    self.route_done[slot].record(self.sr)
            for e in self.route_done:
                main.wait_event(e)
            self.mk, self.mv, self.bias = self.spaces[0]
            return out
    return Dual


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--layer', default='00')
    p.add_argument('--blocks', type=int, default=10)
    p.add_argument('--arm', action='append', default=[],
                   help='Run only these variants (repeatable). Sharding the run means '
                        'a foreign process costs one arm instead of the whole square.')
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
    result = {'initial_gpu': initial, 'layer': args.layer,
              'scope': 'C4 occupancy cost curve and merge/route co-residency. Merge '
                       'residency is capped by padding the dynamic shared request; '
                       'arithmetic and code path are untouched.'}

    def save():
        result.update(monitor_errors=errors, monitor_query_errors=query_errors,
                      monitor_samples=samples)
        args.output.write_text(json.dumps(result, indent=2))

    try:
        import torch
        from kernels.best.c4.operator import NativeFisherAttention, extension
        torch.set_grad_enabled(False); torch.set_num_threads(4)
        Dual = build_dual(torch, NativeFisherAttention)
        op = Dual(device='cuda', capacity=4, selector_backend='fused_cooperative_loop',
                  selector_source=SELECTOR, merge_source=RETAINED)
        wanted = set(args.arm) or {'cta4_retained', 'cta3_pad', 'cta2_pad'}
        result['arms_requested'] = sorted(wanted)
        sources = {'cta3_pad': (ROOT / 'native_fisher/fisher_c4_pad3.cu', '_p3', 3),
                   'cta2_pad': (ROOT / 'native_fisher/fisher_c4_pad2.cu', '_p2', 2)}
        variants = {}
        # The retained merge is always built: it is both an arm and the bitwise
        # reference every other arm is compared against.
        variants['cta4_retained'] = (op.merge, 4)
        for name, (src, suffix, want) in sources.items():
            if name in wanted:
                variants[name] = (extension(4, source_path=src, name_suffix=suffix), want)
        result['resources'] = {n: m.resources() for n, (m, _) in variants.items()}
        print('RESOURCES', json.dumps(result['resources']), flush=True)

        # The padding's whole premise is that residency drops. Verify it.
        result['occupancy_check'] = {}
        usable = {}
        for name, (mod, want) in variants.items():
            got = mod.resources()[3]
            ok = got == want
            result['occupancy_check'][name] = {'intended_ctas_per_sm': want,
                                               'measured_ctas_per_sm': got, 'ok': ok}
            print('OCCUPANCY', name, 'intended', want, 'measured', got,
                  'OK' if ok else 'PREMISE FAILED - not timed', flush=True)
            if ok and name in wanted:
                usable[name] = mod
        save()

        payload = torch.load(CAPTURE.format(args.layer), map_location='cpu', weights_only=True)
        q, k, v = [payload[n].reshape(1, 21, 45, 80, 40, 128).cuda() for n in ('q', 'k', 'v')]
        del payload
        scale = 128 ** -.5

        def timed(fn, repeats=5):
            for _ in range(2):
                fn()
            torch.cuda.synchronize()
            vals = []
            for _ in range(repeats):
                assert not errors, errors
                s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                s.record(); y = fn(); e.record(); e.synchronize()
                vals.append(s.elapsed_time(e)); del y
            return vals

        op.merge = variants['cta4_retained'][0]
        reference, _ = op.serial_call(q, k, v, scale)
        reference = reference.clone()
        torch.cuda.synchronize()

        result['arms'] = {}
        for name, mod in usable.items():
            op.merge = mod
            entry = {}
            _, merge_total = op.serial_call(q, k, v, scale, timed_merge=True)
            entry['serial_merge_total_ms'] = statistics.fmean(
                [op.serial_call(q, k, v, scale, timed_merge=True)[1] for _ in range(5)])
            produced = op.overlapped_call(q, k, v, scale).clone()
            torch.cuda.synchronize()
            entry['overlap_bitwise_equal'] = bool(torch.equal(produced, reference))
            del produced
            pairs = {'serial': [], 'overlap': []}
            for block in range(args.blocks):
                order = ('serial', 'overlap', 'overlap', 'serial') if block % 2 == 0 \
                    else ('overlap', 'serial', 'serial', 'overlap')
                for arm in order:
                    assert not errors, errors
                    pairs[arm] += timed(
                        (lambda: op.serial_call(q, k, v, scale)[0]) if arm == 'serial'
                        else (lambda: op.overlapped_call(q, k, v, scale)), repeats=1)
            n = min(len(pairs['serial']), len(pairs['overlap']))
            deltas = [pairs['overlap'][i] - pairs['serial'][i] for i in range(n)]
            mean = statistics.fmean(deltas)
            sd = statistics.stdev(deltas) if n > 1 else float('nan')
            entry.update({
                'serial_call_ms': statistics.fmean(pairs['serial']),
                'overlap_call_ms': statistics.fmean(pairs['overlap']),
                'overlap_minus_serial_ms': mean, 'sd_ms': sd, 'n': n,
                't': mean / (sd / (n ** .5)) if sd and n > 1 else float('nan'),
                'overlap_faster_count': sum(1 for d in deltas if d < 0)})
            result['arms'][name] = entry
            print('ARM', name, json.dumps({x: entry[x] for x in
                  ('serial_merge_total_ms', 'serial_call_ms', 'overlap_call_ms',
                   'overlap_minus_serial_ms', 't', 'overlap_faster_count', 'n',
                   'overlap_bitwise_equal')}), flush=True)
            save()

        if 'cta4_retained' in result['arms']:
            base = result['arms']['cta4_retained']['serial_merge_total_ms']
            result['occupancy_curve'] = {
                n: {'merge_total_ms': e['serial_merge_total_ms'],
                    'ratio_vs_4cta': e['serial_merge_total_ms'] / base}
                for n, e in result['arms'].items()}
            print('OCCUPANCY_CURVE', json.dumps(result['occupancy_curve']), flush=True)
        result['valid'] = not errors
        print('SUMMARY_DONE', flush=True)
        save()
    finally:
        stop.set()
        save()


if __name__ == '__main__':
    main()
