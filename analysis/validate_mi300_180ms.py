"""Three real layers, paired complete-call timings and numerical gates."""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capture', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    import torch
    from native_hip.attention import Mi300C4Attention
    from attention import hard_c4_wan21_14b_720p_triton as hard
    from attention.affine_hellinger_seed_triton import affine_hellinger_seed_rows_triton
    from attention.singleton_c4_wan21_14b_720p_triton import (
        _hard_c4_720p_temporal_shared_global_mass_attention_triton_blocks as run)
    from torch.nn.attention import sdpa_kernel, SDPBackend
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    assert torch.cuda.device_count() == 1
    result = dict(job=os.environ.get('SLURM_JOB_ID'), layers=[], monitor=[], monitor_errors=[])
    stop = threading.Event()

    def save():
        Path(args.output).write_text(json.dumps(result, indent=2))

    def monitor():
        while not stop.is_set():
            r = subprocess.run(['rocm-smi', '--showpids', '--showuse'],
                               capture_output=True, text=True, timeout=20)
            pids = [int(x) for x in re.findall(r'^\s*(\d+)\s+\S+', r.stdout, re.M)]
            result['monitor'].append(dict(time=time.time(), text=r.stdout))
            if r.returncode or any(x != os.getpid() for x in pids):
                result['monitor_errors'].append(dict(rc=r.returncode, pids=pids))
            stop.wait(3)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        original_kernel = hard._hard_c4_720p_temporal_shared_merge_kernel
        op = Mi300C4Attention()
        assert original_kernel is hard._hard_c4_720p_temporal_shared_merge_kernel
        for layer in ('00', '19', '39'):
            capture = Path(args.capture).with_name(f'real_full40_step0_layer{layer}.pt')
            payload = torch.load(capture, map_location='cpu', weights_only=True)
            q, k, v = [payload[x].reshape(1, 21, 45, 80, 40, 128).cuda() for x in ('q','k','v')]
            del payload
            out = torch.empty_like(q)

            def seeds():
                return affine_hellinger_seed_rows_triton(q[:, :1], k,
                    feature_mode='fisher_geometric', block_j=4)

            def baseline(original=False):
                s = seeds() if original else op.selector(q, k)
                chunk = 5 if original else 20
                for h in range(0, 40, chunk):
                    run(q, k, v, s, top_r=1, head_start=h, head_count=chunk,
                        query_column_selector=True, seed_rows_native_layout=True,
                        query_column_major=True, split_kv_accum=True, block_j=4,
                        merge_loop_stages=1, route_loop_stages=1,
                        merge_disable_licm=False, output=out)
                return out

            row = dict(layer=layer, seed_mismatch=int((op.selector(q,k) != seeds()).sum()))
            assert row['seed_mismatch'] == 0
            ref = baseline().clone()
            assert torch.equal(ref, baseline(True))
            y = op(q,k,v,output=out)
            d = y.float() - ref.float()
            row.update(bitwise_equal=torch.equal(y,ref), finite=bool(torch.isfinite(y).all()),
                       relative_l2=float((d.square().sum()/ref.float().square().sum()).sqrt()),
                       max_abs=float(d.abs().max()), differing_elements=int((y != ref).sum()),
                       elements=y.numel())
            # BF16 output gate for the changed accumulation layout, fixed before
            # cross-layer validation. This does not assert generated-video quality.
            assert row['finite'] and row['relative_l2'] < 1e-4, row
            del d, ref, y
            samples = {name: [] for name in ('hip_h20_j4', 'triton_h5_j4', 'candidate', 'sdpa')}
            calls = {'hip_h20_j4': baseline, 'triton_h5_j4': lambda: baseline(True),
                     'candidate': lambda: op(q,k,v,output=out)}
            for call in calls.values():
                for _ in range(2): call()
            for rep in range(12):
                names = list(calls) if rep % 2 == 0 else list(reversed(calls))
                for name in names:
                    assert not result['monitor_errors'], result['monitor_errors']
                    a,b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record(); calls[name](); b.record(); b.synchronize()
                    samples[name].append(a.elapsed_time(b))
            qf,kf,vf = [x.reshape(1,75600,40,128).transpose(1,2) for x in (q,k,v)]
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                for rep in range(14):
                    a,b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record(); torch.nn.functional.scaled_dot_product_attention(qf,kf,vf)
                    b.record(); b.synchronize()
                    if rep >= 2: samples['sdpa'].append(a.elapsed_time(b))
            row['timings'] = {n:dict(mean_ms=statistics.fmean(s), median_ms=statistics.median(s),
                                   min_ms=min(s), max_ms=max(s), samples_ms=s) for n,s in samples.items()}
            row['speedup_vs_sdpa'] = row['timings']['sdpa']['mean_ms']/row['timings']['candidate']['mean_ms']
            result['layers'].append(row); save(); print('LAYER', row, flush=True)
            # Degenerate case exercises ties and empty residual clusters.
            if layer == '39':
                q.zero_(); k.zero_()
                ref = baseline().clone(); y = op(q,k,v,output=out)
                result['zero_logits'] = dict(finite=bool(torch.isfinite(y).all()),
                                            bitwise_equal=torch.equal(y,ref))
                assert result['zero_logits']['finite'] and result['zero_logits']['bitwise_equal']
                del ref, y
            del q,k,v,out,qf,kf,vf
            torch.cuda.empty_cache()
        result['valid'] = not result['monitor_errors']
        result['completed'] = True
        assert result['valid']
    finally:
        stop.set(); thread.join(timeout=5); save()


if __name__ == '__main__':
    main()
