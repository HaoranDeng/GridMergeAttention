"""Numerical package proof for a C4 selector candidate, with a declared tolerance.

`validate_c4_selector_package.py` gates on exact seed equality. The FP32
Gram-Schmidt selector deliberately fails that gate: it flips 12 of 12 096 000
seeds. Refusing to produce a proof for it would make the video screen
impossible, and quietly loosening the old gate would make every proof that gate
ever signed ambiguous. So this is a separate instrument with the tolerance
written into the proof: the numbers a reader needs in order to disagree are
recorded whether or not they pass.

The control is the retained C4 stack, not the original Triton baseline; the
retained kernels already carry their own equivalence evidence, so comparing
against them isolates exactly one change - the selector's arithmetic.
"""
import argparse, hashlib, json, os, socket, sys, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.launch_quality_first_videos import require_clean, gpu_processes, now

CAPTURE = '/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer{}.pt'
RETAINED_MERGE = ROOT / 'kernels/best/c4/fisher_c4_guarded_softmax.cu'
RETAINED_SELECTOR = ROOT / 'kernels/best/c4/fisher_c4_fused_selector_register_basis_loop.cu'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--selector-source', type=Path, required=True)
    p.add_argument('--merge-source', type=Path, default=RETAINED_MERGE)
    p.add_argument('--control-selector-source', type=Path, default=RETAINED_SELECTOR)
    p.add_argument('--control-merge-source', type=Path, default=RETAINED_MERGE)
    p.add_argument('--allow-merge-change', action='store_true',
                   help='Permit the candidate merge to differ from the control merge. Off '
                        'by default so a selector proof cannot silently carry a merge '
                        'change; on for a whole-package proof, where the merge change '
                        'must carry its own bitwise evidence.')
    p.add_argument('--max-seed-mismatch-fraction', type=float, default=0.0,
                   help='Declared ceiling on the fraction of seed slots that may differ '
                        'from the control. 0 demands bitwise-identical selection.')
    p.add_argument('--max-relative-l2', type=float, default=5e-4,
                   help='Declared ceiling on complete-call output relative L2.')
    a = p.parse_args()
    assert not a.output.exists()
    initial = require_clean(a.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = initial['gpu']['uuid']
    # The proof lives on a filesystem both hosts mount, and torch's extension
    # builder guards each build directory with a lock file. A build killed on
    # one host leaves that lock behind and the other host then waits on it
    # forever - twenty-three silent minutes of a held GPU, in one case. Give
    # each host its own build tree so a lock can only ever block its own owner.
    host = socket.gethostname().split('.')[0]
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

    result = {
        'initial_gpu': initial,
        'scope': 'Complete H40 output and C4 seed-selection agreement against the '
                 'retained C4 stack on three captured layers, plus selector tie '
                 'edge cases. Tolerance is declared, not implied. No timings.',
        'backend': 'fused_cooperative_loop',
        'control': ('retained C4 selector and retained C4 merge'
                    if a.control_selector_source == RETAINED_SELECTOR
                    and a.control_merge_source == RETAINED_MERGE
                    else 'explicit control sources, see `sources`'),
        'sources': {'candidate_selector': str(a.selector_source),
                    'candidate_merge': str(a.merge_source),
                    'control_selector': str(a.control_selector_source),
                    'control_merge': str(a.control_merge_source)},
        'declared_tolerance': {'max_seed_mismatch_fraction': a.max_seed_mismatch_fraction,
                               'max_relative_l2': a.max_relative_l2},
        'captures': {}, 'seed_edges': {},
        'generation_validation_pending': True}

    def save():
        result.update(monitor_errors=errors, monitor_samples=samples)
        a.output.write_text(json.dumps(result, indent=2))

    try:
        import torch
        from kernels.best.c4.operator import NativeFisherAttention
        torch.set_grad_enabled(False); torch.set_num_threads(4)
        control = NativeFisherAttention(capacity=4, selector_backend='fused_cooperative_loop',
                                        selector_source=a.control_selector_source,
                                        merge_source=a.control_merge_source)
        candidate = NativeFisherAttention(capacity=4, selector_backend='fused_cooperative_loop',
                                          selector_source=a.selector_source,
                                          merge_source=a.merge_source)
        result.update(merge_source_sha256=candidate.merge.source_sha256,
                      selector_source_sha256=digest(candidate.native_selector.source_path),
                      control_merge_sha256=control.merge.source_sha256,
                      control_selector_sha256=digest(control.native_selector.source_path),
                      selector_resources=candidate.native_selector.extension.resources())
        result['merge_changed'] = result['merge_source_sha256'] != result['control_merge_sha256']
        if not a.allow_merge_change:
            assert not result['merge_changed'], \
                'merge must be identical in both arms; this instrument isolates the selector'
        print('SOURCES', json.dumps({k: result[k] for k in
              ('merge_source_sha256', 'control_merge_sha256', 'selector_source_sha256',
               'control_selector_sha256', 'merge_changed')}), flush=True)
        save()

        breaches = []
        scale = 128 ** -.5
        for layer in ('00', '19', '39'):
            assert not errors, errors
            payload = torch.load(CAPTURE.format(layer), map_location='cpu', weights_only=True)
            q, k, v = [payload[n].reshape(1, 21, 45, 80, 40, 128).cuda() for n in ('q', 'k', 'v')]
            del payload
            x = candidate(q, k, v); y = control(q, k, v)
            seed_mismatches = int((candidate.seeds != control.seeds).sum())
            seed_total = int(control.seeds.numel())
            difference = x.float() - y.float()
            d = {'raw_storage_mismatches': int((x.view(torch.int16) != y.view(torch.int16)).sum()),
                 'seed_mismatches': seed_mismatches, 'seed_total': seed_total,
                 'seed_mismatch_fraction': seed_mismatches / seed_total,
                 'seeds_bitwise_equal': seed_mismatches == 0,
                 'output_bitwise_equal': bool(torch.equal(x, y)),
                 'finite': bool(torch.isfinite(x).all()),
                 'max_abs': float(difference.abs().max()),
                 'relative_l2': float((difference.square().sum()
                                       / y.float().square().sum().clamp_min(1e-30)).sqrt())}
            del difference, x, y
            result['captures'][layer] = d
            print('FULL40', layer, json.dumps(d), flush=True)
            if not d['finite']:
                breaches.append('layer %s: non-finite output' % layer)
            if d['seed_mismatch_fraction'] > a.max_seed_mismatch_fraction:
                breaches.append('layer %s: seed mismatch fraction %.3e over declared %.3e'
                                % (layer, d['seed_mismatch_fraction'], a.max_seed_mismatch_fraction))
            if d['relative_l2'] > a.max_relative_l2:
                breaches.append('layer %s: relative L2 %.3e over declared %.3e'
                                % (layer, d['relative_l2'], a.max_relative_l2))
            save()
            if layer != '39':
                del q, k, v
                torch.cuda.empty_cache()

        first = q[:, :1]
        zero = torch.zeros_like(first)
        constant_k = k[:, :1, :1, :1].expand_as(k).contiguous()
        for name, query, key in (('zero_query_all_ties', zero, k),
                                 ('identical_probe_vectors', first, constant_k)):
            assert not errors, errors
            x = candidate.native_selector(query, key, scale)
            y = control.native_selector(query, key, scale)
            mismatches = int((x != y).sum())
            d = {'seed_mismatches': mismatches, 'seed_total': int(y.numel()),
                 'seed_mismatch_fraction': mismatches / y.numel()}
            result['seed_edges'][name] = d
            print('EDGE', name, json.dumps(d), flush=True)
            if d['seed_mismatch_fraction'] > a.max_seed_mismatch_fraction:
                breaches.append('%s: seed mismatch fraction %.3e over declared ceiling'
                                % (name, d['seed_mismatch_fraction']))
            del x, y
            save()

        result['tolerance_breaches'] = breaches
        result['within_declared_tolerance'] = not breaches
        result['valid'] = not errors and not breaches
        for line in breaches:
            print('BREACH', line, flush=True)
        print('SUMMARY_DONE valid=%s' % result['valid'], flush=True)
        save()
    except BaseException as exc:
        result.update(valid=False, error=repr(exc)); save(); raise
    finally:
        stop.set(); save()


if __name__ == '__main__':
    main()
