"""Bounded real-QKV tile tuning on one freshly clean H200; no model loading."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.launch_quality_first_videos import require_clean, gpu_processes, now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    initial = require_clean(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = initial['gpu']['uuid']
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['TRITON_CACHE_DIR'] = str(args.output.parent / 'triton_cache')
    errors, samples = [], []
    done = threading.Event()
    def monitor():
        while not done.is_set():
            try:
                active = gpu_processes(initial['gpu']['uuid'])
                samples.append(dict(utc=now(), processes=active))
                if any(row['pid'] != os.getpid() for row in active):
                    errors.append('foreign GPU process')
            except Exception as error:
                errors.append(repr(error))
            done.wait(3)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    result = dict(initial_gpu=initial, scope='real layer0 step0 BF16 QKV, one clean H200', variants=[])
    def save():
        result.update(monitor_errors=errors, monitor_samples=samples)
        args.output.write_text(json.dumps(result, indent=2))
    try:
        import torch
        from attention.affine_hellinger_seed_triton import fisher_geometric_c10_seed_rows_triton
        from analysis.fisher_c10_video_adapter import FisherC10MergeAdapter
        torch.set_grad_enabled(False)
        torch.set_num_threads(4)
        payload = torch.load('/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer00.pt', map_location='cpu', weights_only=True)
        q,k,v = [payload[name].reshape(1,21,45,80,40,128).cuda() for name in ('q','k','v')]
        del payload
        scale = 128 ** -.5
        select = lambda: fisher_geometric_c10_seed_rows_triton(q[:,:1], k, scale)
        seeds = select()
        def adapter(warps):
            a = FisherC10MergeAdapter()
            hard = a.inner.__globals__['hard_c4']
            proxy = types.SimpleNamespace(**vars(hard))
            proxy.HARD_720_MERGE_NUM_WARPS = warps
            a.inner.__globals__['hard_c4'] = proxy
            return a
        def invoke(a,j,stages,seeds,head=0,count=1,output=None):
            return a(q,k,v,seeds,scale,top_r=1,head_start=head,head_count=count,
                     selector='query_column',seed_rows_native_layout=True,
                     query_column_major=True,merge_loop_stages=stages,
                     merge_disable_licm=False,split_kv_accum=True,block_j=j,output=output)
        def timed(fn, repeats=3, warmup=1):
            for _ in range(warmup):
                y=fn(); del y
            torch.cuda.synchronize()
            values=[]
            for _ in range(repeats):
                assert not errors, errors
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record(); y=fn(); b.record(); b.synchronize(); del y
                values.append(a.elapsed_time(b))
            return dict(mean_ms=statistics.fmean(values),samples_ms=values)
        cases=[(16,4,2),(4,4,1),(4,8,1),(8,8,1)]
        baseline=None
        for j,warps,stages in cases:
            a=adapter(warps)
            output=invoke(a,j,stages,seeds)
            actual=output[:,:,:,:,0].clone();del output
            if baseline is None:baseline=actual.clone()
            difference=actual.float()-baseline.float()
            check=dict(max_abs=difference.abs().max().item(),
                       relative_l2=(difference.square().sum()/baseline.float().square().sum()).sqrt().item(),
                       bitwise_equal=torch.equal(actual,baseline),finite=torch.isfinite(actual).all().item())
            assert check['finite'] and check['max_abs'] < .02 and check['relative_l2'] < .002,check
            del actual,difference
            record=dict(block_j=j,warps=warps,loop_stages=stages,correctness=check,
                        timing=timed(lambda:invoke(a,j,stages,seeds)))
            result['variants'].append(record);save();print(json.dumps(record),flush=True)
        best=min(result['variants'],key=lambda x:x['timing']['mean_ms'])
        result['best']=best
        # Full40 includes the selector, padding and all ten four-head chunks.
        def full(a,j,stages):
            live=select()
            output=torch.empty_like(q)
            for h in range(0,40,4):invoke(a,j,stages,live,h,4,output)
            return output
        old=adapter(4);new=adapter(best['warps'])
        reference=full(old,16,2)
        candidate=full(new,best['block_j'],best['loop_stages'])
        diff=candidate.float()-reference.float()
        result['full40_correctness']=dict(max_abs=diff.abs().max().item(),
            relative_l2=(diff.square().sum()/reference.float().square().sum()).sqrt().item(),
            finite=torch.isfinite(candidate).all().item())
        assert result['full40_correctness']['finite'] and result['full40_correctness']['relative_l2']<.002
        del reference,candidate,diff
        result['full40_old']=timed(lambda:full(old,16,2))
        result['full40_new']=timed(lambda:full(new,best['block_j'],best['loop_stages']))
        from torch.nn.attention import sdpa_kernel, SDPBackend
        flat=[x.reshape(1,75600,40,128).transpose(1,2) for x in (q,k,v)]
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            result['full40_sdpa']=timed(lambda:torch.nn.functional.scaled_dot_product_attention(*flat))
        result['speedup_vs_sdpa']=result['full40_sdpa']['mean_ms']/result['full40_new']['mean_ms']
        result['speedup_vs_previous']=result['full40_old']['mean_ms']/result['full40_new']['mean_ms']
        result['valid']=not errors
        save();print(json.dumps({k:v for k,v in result.items() if k not in ('monitor_samples','variants')}),flush=True)
    except BaseException as error:
        result.update(valid=False,failure=repr(error));save();raise
    finally:
        done.set();thread.join(timeout=10);save()


if __name__=='__main__':main()
