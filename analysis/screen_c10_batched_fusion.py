"""GPU differential gate for the CPU-compiled batched fusion prototype.

Default H1 is a diagnostic without a full-call headline. --heads40 includes
current Fisher selection and pairs complete calls with the retained native op.
Never run alongside generation or its follow-up GPU metrics.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import threading
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.launch_quality_first_videos import require_clean,gpu_processes,now


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--heads',type=int,choices=(1,40),default=1)
    p.add_argument('--layer',choices=('00','19','39'),default='00')
    p.add_argument('--block-m',type=int,choices=(16,32),default=16)
    p.add_argument('--warps',type=int,choices=(4,8),default=4)
    p.add_argument('--diagnostic-timing',action='store_true',help='H1 only: record timing even if finite output misses the L2 screening gate; never claim validation')
    args=p.parse_args();assert not args.output.exists()
    if args.diagnostic_timing and args.heads!=1:p.error('Diagnostic timing is restricted to H1')
    initial=require_clean(args.gpu)
    os.environ.update(CUDA_VISIBLE_DEVICES=initial['gpu']['uuid'],OMP_NUM_THREADS='4',
                      TORCH_EXTENSIONS_DIR=str(args.output.parent/'torch_extensions'),
                      TRITON_CACHE_DIR=str(args.output.parent/'triton_compile_cache'))
    errors=[];samples=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                processes=gpu_processes(initial['gpu']['uuid']);samples.append({'utc':now(),'processes':processes})
                if any(v['pid']!=os.getpid() for v in processes):errors.append('foreign GPU process')
            except Exception as error:errors.append(repr(error))
            stop.wait(3)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    result={'initial_gpu':initial,'heads':args.heads,'layer':args.layer,'block_m':args.block_m,'warps':args.warps,
            'scope':'Complete all40 with current selector' if args.heads==40 else 'H1 merge/full-support route only; no full-call headline',
            'source_sha256':hashlib.sha256((ROOT/'native_fisher/batched_fused_route.py').read_bytes()).hexdigest()}
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        result.update(monitor_errors=errors,monitor_samples=samples)
        args.output.write_text(json.dumps(result,indent=2))
    try:
        import torch
        import triton
        from native_fisher.operator import NativeFisherAttention
        from native_fisher.batched_fused_route import batched_fused_c10,combine_batched_c10
        torch.set_grad_enabled(False);torch.set_num_threads(4)
        payload=torch.load(f'/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer{args.layer}.pt',map_location='cpu',weights_only=True)
        q,k,v=[payload[n].reshape(1,21,45,80,40,128).cuda() for n in ('q','k','v')];del payload
        op=NativeFisherAttention(capacity=10,merge_source=ROOT/'native_fisher/fisher_c10_half_owner.cu',
                                 selector_source=ROOT/'native_fisher/fisher_gs_register_distribute.cu')
        expected=op(q,k,v,head_count=40 if args.heads==40 else 5)
        seeds=op.seeds
        shape=(16,21*80*args.heads*45)
        state=(torch.empty((*shape,128),device='cuda',dtype=torch.float32),
               torch.empty(shape,device='cuda'),torch.empty(shape,device='cuda'))
        def candidate():
            current_seeds=op.select(q,k,128**-.5) if args.heads==40 else seeds
            output=torch.empty_like(q)
            batched_fused_c10[(21*40,args.heads,16*triton.cdiv(45,args.block_m))](q,k,v,current_seeds,*state,
                0,args.heads,2,16,args.block_m,num_warps=args.warps,num_stages=1)
            combine_batched_c10[(triton.cdiv(21*80*args.heads*45,4),)](*state,output,0,args.heads,16,4,num_warps=4)
            return output
        actual=candidate();a=actual[...,:args.heads,:].float();b=expected[...,:args.heads,:].float();diff=a-b
        result['correctness']={'finite':bool(torch.isfinite(a).all()),'relative_l2':float((diff.square().sum()/b.square().sum()).sqrt()),
                               'max_abs':float(diff.abs().max()),'bitwise_equal':torch.equal(a,b)}
        save();print('GATE',result['correctness'],flush=True)
        result['passed_output_gate']=result['correctness']['finite'] and result['correctness']['relative_l2']<.002
        assert result['correctness']['finite'] and (result['passed_output_gate'] or args.diagnostic_timing)
        del actual,a,b,diff,expected
        for _ in range(2):candidate()
        torch.cuda.synchronize()
        times={'candidate':[],'control':[]}
        for block in range(6):
            order=(('candidate','control','control','candidate') if block%2==0 else ('control','candidate','candidate','control')) if args.heads==40 else ('candidate',)
            for name in order:
                assert not errors,errors
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();out=candidate() if name=='candidate' else op(q,k,v);end.record();end.synchronize()
                times[name].append(start.elapsed_time(end));del out
        result['timing']={n:{'mean_ms':statistics.fmean(t),'samples_ms':t} for n,t in times.items() if t}
        if args.heads==40:
            from torch.nn.attention import sdpa_kernel,SDPBackend
            dense=[x.reshape(1,75600,40,128).transpose(1,2).contiguous() for x in (q,k,v)]
            samples_sdpa=[]
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                for i in range(7):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();out=torch.nn.functional.scaled_dot_product_attention(*dense);end.record();end.synchronize()
                    if i>=2:samples_sdpa.append(start.elapsed_time(end))
            result['sdpa']={'mean_ms':statistics.fmean(samples_sdpa),'samples_ms':samples_sdpa}
            result['speedup_vs_sdpa']=result['sdpa']['mean_ms']/result['timing']['candidate']['mean_ms']
        result['valid']=not errors;save();print(json.dumps({k:v for k,v in result.items() if k!='monitor_samples'}),flush=True)
    except BaseException as error:
        result.update(valid=False,error=repr(error));save();raise
    finally:stop.set();thread.join(5);save()


if __name__=='__main__':main()
