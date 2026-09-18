"""Bounded H1 algebraic-fusion screen; no full-call extrapolation."""
import argparse,json,os,sys,threading,statistics,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.launch_quality_first_videos import require_clean,gpu_processes,now

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fp32',action='store_true')
    p.add_argument('--split-bf16',action='store_true')
    p.add_argument('--minimal-memcheck',action='store_true',help='One block with small valid buffers; no complete output or timing claim')
    args=p.parse_args();assert not(args.fp32 and args.split_bf16);assert not args.output.exists();initial=require_clean(args.gpu)
    os.environ.update(CUDA_VISIBLE_DEVICES=initial['gpu']['uuid'],OMP_NUM_THREADS='4',MAX_JOBS='2',TORCH_EXTENSIONS_DIR=str(args.output.parent/'torch_extensions'),TRITON_CACHE_DIR=str(args.output.parent/'triton_cache'))
    errors=[];samples=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                ps=gpu_processes(initial['gpu']['uuid']);samples.append({'utc':now(),'processes':ps})
                if any(x['pid']!=os.getpid() for x in ps):errors.append('foreign GPU process')
            except Exception as e:errors.append(repr(e))
            stop.wait(2)
    thread=threading.Thread(target=monitor);thread.start()
    result={'initial_gpu':initial,'scope':'H1 full merge+SQ1 attention; excludes selector/packing; not H40','source_sha256':hashlib.sha256((ROOT/'native_fisher/reassociated_route.py').read_bytes()).hexdigest()}
    def save():
        result.update(monitor_errors=errors,monitor_samples=samples);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    try:
        import torch
        from native_fisher.operator import NativeFisherAttention
        from native_fisher.reassociated_route import reassociated_route as impl
        def reassociated_route(*a,**kw):return impl(*a,fp32=args.fp32,split_bf16=args.split_bf16,**kw)
        result['fp32_reassociation']=args.fp32
        result['split_bf16_reassociation']=args.split_bf16
        torch.set_grad_enabled(False);torch.set_num_threads(4)
        if args.minimal_memcheck:
            result['scope']='One query column,one key block; minimal address-safety smoke only'
            torch.manual_seed(0)
            q=torch.randn((45*80*40*128,),device='cuda',dtype=torch.bfloat16)
            k=torch.randn((80*40*128,),device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
            seeds=torch.arange(10,device='cuda',dtype=torch.uint8);out=torch.full_like(q,float('nan'))
            kernel=reassociated_route(q,k,v,seeds,out,groups=1,key_blocks=1)
            result['resources']={'registers':kernel.n_regs,'spills':kernel.n_spills,'shared':kernel.metadata.shared};save();print('RESOURCES',result['resources'],flush=True)
            torch.cuda.synchronize()
            result['smoke_finite']=bool(torch.isfinite(out.reshape(45,80,40,128)[:,0,0]).all())
            result['valid']=result['smoke_finite'] and not errors;save();print('MINIMAL_SMOKE',result['valid'],flush=True)
            return
        payload=torch.load('/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer00.pt',map_location='cpu',weights_only=True)
        q,k,v=[payload[n].reshape(1,21,45,80,40,128).cuda() for n in ('q','k','v')];del payload
        op=NativeFisherAttention(capacity=10,tiles=84,selector_source=ROOT/'native_fisher/fisher_gs_register_distribute.cu',merge_source=ROOT/'native_fisher/fisher_c10_long_tiles_tree.cu')
        expected=op(q,k,v,head_count=5);out=torch.empty_like(q)
        kernel=reassociated_route(q,k,v,op.seeds,out,groups=1,key_blocks=1);torch.cuda.synchronize()
        result['resources']={'registers':kernel.n_regs,'spills':kernel.n_spills,'shared':kernel.metadata.shared}
        result['smoke_finite']=bool(torch.isfinite(out[0,0,:,0,0]).all());save();print('SMOKE',result['resources'],result['smoke_finite'],flush=True)
        assert result['smoke_finite']
        reassociated_route(q,k,v,op.seeds,out);torch.cuda.synchronize()
        a=out[...,:1,:].float();b=expected[...,:1,:].float();diff=a-b
        result['correctness']={'relative_l2':float((diff.square().sum()/b.square().sum()).sqrt()),'max_abs':float(diff.abs().max()),'finite':bool(torch.isfinite(a).all()),'bitwise_equal':torch.equal(a,b)}
        result['numerical_gate_passed']=result['correctness']['finite'] and result['correctness']['relative_l2']<.002
        print('GATE',result['correctness'],flush=True);save()
        assert result['numerical_gate_passed']
        times=[]
        for _ in range(5):
            assert not errors,errors
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();reassociated_route(q,k,v,op.seeds,out);b.record();b.synchronize();times.append(a.elapsed_time(b))
        result['timing']={'mean_ms':statistics.fmean(times),'samples_ms':times};result['valid']=not errors;print('TIMING',result['timing'],flush=True);save()
    except BaseException as e:result.update(valid=False,error=repr(e));save();raise
    finally:stop.set();thread.join(5);save()
if __name__=='__main__':main()
