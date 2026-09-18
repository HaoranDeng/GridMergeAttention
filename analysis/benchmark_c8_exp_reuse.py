"""Single-clean-H200 diagnostic for C8 exp reuse, using full H40 real captures.

Reports error against r2 and slow_reference, paired complete-call timing, and
matched SDPA. This is a kernel screen, not a video-quality acceptance test.
"""
import argparse
import csv
import ctypes
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


def smi(*args):
    return subprocess.check_output(['nvidia-smi', *args], text=True, timeout=10).strip()


def processes(uuid):
    raw = smi('-i', uuid, '--query-compute-apps=pid,used_memory,process_name',
              '--format=csv,noheader,nounits')
    return [{'pid': int(row[0]), 'memory_mib': row[1].strip(),
             'name': ','.join(row[2:]).strip()}
            for row in csv.reader(io.StringIO(raw)) if row]


def require_clean(gpu):
    raw = smi('-i', str(gpu), '--query-gpu=index,uuid,name,utilization.gpu,memory.used',
              '--format=csv,noheader,nounits')
    rows = list(csv.reader(io.StringIO(raw)))
    assert len(rows) == 1
    idx, uuid, name, utilization, memory = [x.strip() for x in rows[0]]
    active = processes(uuid)
    if int(idx) != gpu or 'H200' not in name or active or int(utilization) != 0 or int(memory) > 100:
        raise RuntimeError('No exclusively idle H200 at requested index; do not wait or queue')
    return {'utc': now(), 'index': gpu, 'uuid': uuid, 'name': name,
            'memory_mib': int(memory), 'utilization': int(utilization), 'processes': active}


def compare(actual, reference, kind):
    import torch
    assert actual.shape == reference.shape and actual.dtype == reference.dtype
    a, b = actual.reshape(-1), reference.reshape(-1)
    stats = torch.zeros(2, dtype=torch.float64, device=a.device)
    peak = torch.zeros(2, dtype=torch.float32, device=a.device)
    invalid = torch.zeros((), dtype=torch.int64, device=a.device)
    for start in range(0, a.numel(), 1 << 22):
        av, bv = a[start:start+(1 << 22)].float(), b[start:start+(1 << 22)].float()
        fa, fb = torch.isfinite(av), torch.isfinite(bv)
        invalid += ((fa != fb) | (~fa & (av != bv))).sum()
        av, bv = torch.where(fa, av, 0), torch.where(fb, bv, 0)
        difference = av.double() - bv.double()
        stats[0] += difference.square().sum()
        stats[1] += bv.double().square().sum()
        peak[0] = torch.maximum(peak[0], difference.abs().max().float())
        peak[1] = torch.maximum(peak[1], bv.abs().max())
    error2, ref2 = stats.tolist()
    max_error, max_ref = peak.tolist()
    rel = math.sqrt(error2/ref2) if ref2 else (0 if error2 == 0 else math.inf)
    scaled_max = max_error/max_ref if max_ref else (0 if max_error == 0 else math.inf)
    limits = {'relative_l2': 1e-5, 'max_abs_over_reference_max': 1e-4} if kind == 'bias' else {
        'relative_l2': 1e-3, 'max_abs_over_reference_max': 2e-2}
    finite = all(math.isfinite(x) for x in (rel, scaled_max, error2, ref2, max_error, max_ref))
    return {'relative_l2': rel if math.isfinite(rel) else None,
            'max_abs': max_error, 'reference_abs_max': max_ref,
            'max_abs_over_reference_max': scaled_max if math.isfinite(scaled_max) else None,
            'invalid_special_values': int(invalid), 'thresholds': limits,
            'passed': finite and int(invalid) == 0 and rel <= limits['relative_l2']
                      and scaled_max <= limits['max_abs_over_reference_max']}


def paired(rows):
    delta = [r['control'] - r['candidate'] for r in rows]
    mean = statistics.fmean(delta)
    sem = statistics.stdev(delta) / math.sqrt(len(delta)) if len(delta) > 1 else 0
    return {'control_mean_ms': statistics.fmean(r['control'] for r in rows),
            'candidate_mean_ms': statistics.fmean(r['candidate'] for r in rows),
            'mean_saving_ms': mean, 'ci95_approx_ms': [mean-1.96*sem, mean+1.96*sem],
            'wins': sum(x > 0 for x in delta), 'blocks': len(rows), 'block_means': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--layers', default='00,19,39')
    p.add_argument('--blocks', type=int, default=8)
    p.add_argument('--sources', nargs='+', required=True)
    p.add_argument('--output', default='result.json')
    args = p.parse_args()
    assert args.blocks >= 2
    args.run_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.run_dir/args.output
    assert not output_path.exists(), 'Refusing to overwrite a result'
    initial = require_clean(args.gpu)
    tmp = args.run_dir/'runtime_tmp'
    tmp.mkdir(exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=initial['uuid'], TORCH_CUDA_ARCH_LIST='9.0a',
                      MAX_JOBS='2', OMP_NUM_THREADS='4', TMPDIR=str(tmp),
                      TORCH_EXTENSIONS_DIR=str(args.run_dir/'torch_extensions'),
                      TRITON_CACHE_DIR=str(args.run_dir/'triton_cache'))
    ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libstdc++.so.6', mode=ctypes.RTLD_GLOBAL)
    sys.path.insert(0, str(args.repo_root.resolve()))
    import torch
    from torch.utils.cpp_extension import load
    from kernels.best import create_c8
    from kernels.best.c8 import gma_c8_native as slow_package
    from torch.nn.attention import sdpa_kernel, SDPBackend
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    result = {'scope': 'H200 real Wan14B 720p captures, full H40 default Hellinger C8 r2',
              'kind': 'three-layer kernel diagnostic; no video generation or quality claim',
              'shape': [1,21,45,80,40,128], 'capacity': 8, 'singleton_count': 0,
              'initial_gpu': initial, 'blocks': args.blocks, 'layers': {}, 'candidates': {},
              'threshold_policy': 'Prespecified engineering screen, not user-specified or a video acceptance threshold',
              'acceptance_scope': 'Final output versus r2 and slow_reference, plus valid intermediate values. Intermediate closeness is diagnostic because near-tied cluster assignments may differ.',
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'baseline_commit': subprocess.check_output(['git','-C',str(args.repo_root),'rev-parse','HEAD'],text=True).strip(),
              'valid_timing': False, 'status': 'preparing'}
    stop = threading.Event()
    monitor_errors = []
    monitor_samples = []

    def save():
        result.update(monitor_errors=monitor_errors, monitor_samples=monitor_samples)
        temporary = output_path.with_suffix('.partial.json')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        temporary.replace(output_path)

    def monitor():
        while not stop.is_set():
            try:
                rows = processes(initial['uuid'])
                foreign = [row for row in rows if row['pid'] != os.getpid()]
                monitor_samples.append({'utc': now(), 'processes': rows})
                if foreign:
                    monitor_errors.append({'utc': now(), 'foreign': foreign})
            except Exception as exc:
                monitor_errors.append({'utc': now(), 'error': repr(exc)})
            stop.wait(1)

    def clean():
        if monitor_errors:
            raise RuntimeError('GPU contamination or monitor failure; all timings invalid')

    def timed(call):
        clean()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); call(); b.record(); b.synchronize()
        clean()
        return a.elapsed_time(b)

    result['prelaunch_gpu'] = require_clean(args.gpu)
    assert result['prelaunch_gpu']['uuid'] == initial['uuid']
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        op = create_c8()
        retained = op.merge
        retained_gs = op.gs
        slow_merge = slow_package._extension('gma_c8_owner_pack_v0', True)
        slow_gs = slow_package._extension('gma_c8_cuda_gs_v10')

        class SlowMerge:
            pack = staticmethod(slow_merge.pack)

            @staticmethod
            def run(*values):
                values = list(values)
                values[9] = 21
                return slow_merge.run(*values)

        result['baseline_resources'] = op.resources()
        include = Path(torch.__file__).resolve().parent.parent/'flashinfer/data/cutlass/include'
        modules = {}
        for name in args.sources:
            path = args.run_dir/'sources'/name
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            modules[name] = load(name='fisher_c8_merge_'+digest[:16], sources=[str(path)],
                                 extra_include_paths=[str(include)], extra_ldflags=['-lcuda'],
                                 extra_cuda_cflags=['-O3','--expt-relaxed-constexpr','-lineinfo','--ptxas-options=-v'],
                                 verbose=False)
            result['candidates'][name] = {'source_sha256': digest,
                'resources': dict(zip(('registers','local_bytes','shared_bytes','ctas_per_sm'), modules[name].resources()))}
        refs = [torch.empty_like(x) for x in (op.mk, op.mv, op.bias)]
        save()
        captures = Path('/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908')
        for layer in args.layers.split(','):
            clean()
            payload = torch.load(captures/f'real_full40_step0_layer{layer}.pt', map_location='cpu', weights_only=True)
            q, k, v = [payload[n].reshape(1,21,45,80,40,128).cuda() for n in ('q','k','v')]
            del payload
            actual, reference, slow_reference = [torch.empty_like(q) for _ in range(3)]
            op.merge, op.gs = retained, retained_gs
            op(q,k,v,output=reference)
            seeds = op.seeds.clone()
            op.merge, op.gs = SlowMerge, slow_gs
            op(q,k,v,output=slow_reference)
            lr = result['layers'][layer] = {
                'r2_vs_slow_reference': compare(reference,slow_reference,'output'),
                'slow_reference_seed_mismatches': int((op.seeds!=seeds).sum()), 'candidates': {}}
            op.merge, op.gs = retained, retained_gs
            op.seeds.copy_(seeds)
            retained.run(op.packed_q,k,v,seeds,op.mk,op.mv,op.bias,0,5,105,128**-.5,op.empty,0,1)
            for dst, src in zip(refs,(op.mk,op.mv,op.bias)):
                dst.copy_(src)
            torch.cuda.synchronize()
            for name, mod in modules.items():
                mod.run(op.packed_q,k,v,seeds,op.mk,op.mv,op.bias,0,5,105,128**-.5,op.empty,0,1)
                errors = {kind: compare(a,b,kind) for kind,a,b in
                          zip(('mk','mv','bias'),(op.mk,op.mv,op.bias),refs)}
                semantic_checks = {'merged_k_finite': bool(torch.isfinite(op.mk).all()),
                                   'merged_v_finite': bool(torch.isfinite(op.mv).all()),
                                   'bias_no_nan_or_negative_infinity': bool((~torch.isnan(op.bias) & ~torch.isneginf(op.bias)).all())}
                op.merge = mod
                op(q,k,v,output=actual)
                errors['output'] = compare(actual,reference,'output')
                error_slow = compare(actual,slow_reference,'output')
                row = lr['candidates'][name] = {'errors_vs_r2': errors,
                    'output_vs_slow_reference': error_slow,
                    'semantic_checks': semantic_checks,
                    'intermediate_screen_passed': all(errors[kind]['passed'] for kind in ('mk','mv','bias')),
                    'seeds_equal': bool(torch.equal(op.seeds,seeds)),
                    'screen_passed': errors['output']['passed'] and error_slow['passed'] and all(semantic_checks.values())}
                print(json.dumps({'layer':layer,'candidate':name,'screen_passed':row['screen_passed'],
                                  'errors_vs_r2':errors}),flush=True)
                save()
                if not row['screen_passed']:
                    continue

                def call(module):
                    op.merge = module
                    op(q,k,v,output=actual)

                for _ in range(3):
                    call(retained); call(mod)
                samples = []
                for block in range(args.blocks):
                    order = ('control','candidate','candidate','control') if block%2 == 0 else (
                        'candidate','control','control','candidate')
                    sample = {'control':[], 'candidate':[]}
                    for key in order:
                        sample[key].append(timed(lambda: call(retained if key=='control' else mod)))
                    samples.append({key:statistics.fmean(values) for key,values in sample.items()})
                row['complete_call'] = paired(samples)
                merge_blocks = []
                for block in range(args.blocks):
                    order = (('control',retained),('candidate',mod)) if block%2 == 0 else (
                        ('candidate',mod),('control',retained))
                    merge_blocks.append({key:timed(lambda module=module: module.run(
                        op.packed_q,k,v,seeds,op.mk,op.mv,op.bias,0,5,105,128**-.5,op.empty,0,1))
                        for key,module in order})
                row['merge_h5'] = paired(merge_blocks)
                save()
                print(json.dumps({'layer':layer,'candidate':name,'complete_call':row['complete_call'],
                                  'merge_h5':row['merge_h5']}),flush=True)
            flat = [x.reshape(1,75600,40,128).transpose(1,2) for x in (q,k,v)]
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                dense = lambda: torch.nn.functional.scaled_dot_product_attention(*flat)
                dense(); dense()
                dense_times = [timed(dense) for _ in range(6)]
            lr['sdpa_mean_ms'] = statistics.fmean(dense_times)
            lr['sdpa_samples_ms'] = dense_times
            for row in lr['candidates'].values():
                if 'complete_call' in row:
                    row['speedup_vs_sdpa'] = lr['sdpa_mean_ms']/row['complete_call']['candidate_mean_ms']
            result['status'] = 'running'
            save()
            del q,k,v,actual,reference,slow_reference,seeds,flat,dense
            torch.cuda.empty_cache()
        clean()
        result.update(status='complete',valid_timing=True)
    except BaseException as exc:
        result.update(status='failed',error=repr(exc),valid_timing=False)
        raise
    finally:
        stop.set()
        thread.join(5)
        if monitor_errors:
            result['valid_timing'] = False
        save()


if __name__ == '__main__':
    main()
