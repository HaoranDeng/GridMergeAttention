"""Validate shared-ring online attention before integrating merge fusion."""
import argparse, gc, hashlib, json, os, sys, threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.launch_quality_first_videos import require_clean,gpu_processes,now

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--capacities',type=int,nargs='*',choices=[4,10],default=[4,10])
    p.add_argument('--fused',action='store_true')
    p.add_argument('--fused-profile',action='store_true')
    p.add_argument('--fused-source',type=Path)
    p.add_argument('--source-dir',type=Path,default=ROOT/'native_fisher')
    args=p.parse_args();assert not args.output.exists()
    if args.fused_profile and not args.fused:p.error('--fused-profile requires --fused')
    initial=require_clean(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES']=initial['gpu']['uuid']
    os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
    os.environ['TORCH_EXTENSIONS_DIR']=str(args.output.parent/'torch_extensions')
    os.environ['TRITON_CACHE_DIR']=str(args.output.parent/'triton_cache')
    stop=threading.Event();errors=[];samples=[]
    def monitor():
        while not stop.is_set():
            try:
                processes=gpu_processes(initial['gpu']['uuid'])
                samples.append({'utc':now(),'processes':processes})
                if any(x['pid']!=os.getpid() for x in processes):errors.append('foreign GPU process')
            except Exception as e:errors.append(repr(e))
            stop.wait(3)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    result={'scope':'Shared-ring route numerical gate, including full materialized compressed streams; no fused full-call timing',
            'initial_gpu':initial,'cases':[]}
    def save():
        result.update(monitor_errors=errors,monitor_samples=samples)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2))
    try:
        import torch
        from torch.utils.cpp_extension import load
        torch.set_grad_enabled(False);torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        sources=[args.source_dir.resolve()/n for n in ('warp_mma.cuh','warp_merge.cuh','warp_route.cuh','warp_route_diagnostic.cu')]
        hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}
        key=hashlib.sha256(''.join(hashes.values()).encode()).hexdigest()[:16]
        print('BUILD_ROUTE',key,flush=True)
        ext=load(name='fisher_warp_route_'+key,sources=[str(sources[-1])],
                 extra_cuda_cflags=['-O3','-lineinfo','--ptxas-options=-v'],verbose=False)
        assert hashes=={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}
        result.update(sources=hashes,resources=ext.resources())
        def reference(q,k,v,bias,rows):
            # Same 64-key stream and BF16 probability rounding. FP32 Torch GEMM
            # is an independent arithmetic oracle, not an exact storage oracle.
            q=q.float().clone();q[:,rows:]=0
            m=torch.full(q.shape[:2],-torch.inf,device='cuda');den=torch.zeros_like(m);acc=torch.zeros_like(q)
            for start in range(0,k.shape[1],64):
                kk=k[:,start:start+64].float();vv=v[:,start:start+64].float()
                logits=((q@kk.transpose(-1,-2))*128**-.5-bias[:,None,start:start+64])*1.4426950408889634
                next_m=torch.maximum(m,logits.amax(-1));present=torch.isfinite(next_m)
                alpha=torch.where(present,torch.exp2(m-next_m),0.)
                prob=torch.where(present[:,:,None],torch.exp2(logits-next_m[:,:,None]),0.)
                acc=acc*alpha[:,:,None]+prob.to(torch.bfloat16).float()@vv
                den=den*alpha+prob.sum(-1);m=next_m
            out=torch.where(den[:,:,None]>0,acc/den[:,:,None],0.);out[:,rows:]=0
            return out,m,den
        def check(name,q,k,v,bias,rows=16):
            packed=q.reshape(-1,16,4,4,4,2).transpose(-3,-2).contiguous().reshape_as(q)
            actual,m,den=ext.run(packed,k,v,bias,rows,128**-.5)
            expected,rm,rd=reference(q,k,v,bias,rows)
            diff=actual-expected
            rel=float((diff.square().sum()/expected.square().sum().clamp_min(1e-30)).sqrt())
            row={'name':name,'tokens':k.shape[1],'query_rows':rows,'finite':bool(torch.isfinite(actual).all()),
                 'relative_l2':rel,'max_abs':float(diff.abs().max()),
                 'max_pattern_equal':torch.equal(torch.isfinite(m),torch.isfinite(rm)),
                 'zero_den_pattern_equal':torch.equal(den==0,rd==0),
                 'den_relative_l2':float(((den-rd).square().sum()/rd.square().sum().clamp_min(1e-30)).sqrt())}
            result['cases'].append(row);print(json.dumps(row),flush=True);save()
            assert row['finite'] and rel<2e-4 and row['max_pattern_equal'] and row['zero_den_pattern_equal'],row
        torch.manual_seed(123)
        for length in (1,63,64,65,129,640):
            q=torch.randn((3,16,128),device='cuda',dtype=torch.bfloat16)
            k=torch.randn((3,length,128),device='cuda',dtype=torch.bfloat16)
            v=torch.randn_like(k);bias=torch.randn((3,length),device='cuda')
            check(f'random_{length}',q,k,v,bias)
        check('query_tail_13',q,k,v,bias,13)
        mask=bias.clone();mask[:,:64]=torch.inf
        check('empty_first_ring',q,k,v,mask)
        mask=bias.clone();mask[:,64:128]=torch.inf
        check('empty_middle_ring',q,k,v,mask)
        mask=bias.clone();mask[:,::2]=torch.inf
        check('alternating_empty_keys',q,k,v,mask)
        check('all_empty',q,k,v,torch.full_like(bias,torch.inf))
        check('zero_query_ties',torch.zeros_like(q),k,v,torch.zeros_like(bias))
        del q,k,v,bias,mask
        if args.capacities:
            from native_fisher.operator import NativeFisherAttention
            payload=torch.load(args.output.parent.parent/'gma_c8_cuda_150ms_20260908/real_full40_step0_layer00.pt',map_location='cpu',weights_only=True)
            q,k,v=[payload[n].reshape(1,21,45,80,40,128).cuda() for n in ('q','k','v')];del payload
            result['native_sources']={}
            for capacity in args.capacities:
                print('NATIVE_WORKSPACE',capacity,flush=True)
                op=NativeFisherAttention(capacity=capacity)
                result['native_sources'][str(capacity)]=op.merge.source_sha256
                seeds=op.select(q,k,128**-.5);op.merge.pack(q,op.packed)
                op.merge.run(op.packed,k,v,seeds,op.mk,op.mv,op.bias,0,5,21,128**-.5,op.empty,0,1)
                result.setdefault('native_empty_fraction_h5',{})[str(capacity)]=float(torch.isposinf(op.bias).float().mean())
                positions=[(0,0,0),(10,37,2),(20,79,4)]
                group=torch.tensor([(f*80+c)*5+h for f,c,h in positions],device='cuda')
                qtile=torch.stack([q[0,f,:16,c,h] for f,c,h in positions]).contiguous()
                mk=op.mk[group].contiguous();mv=op.mv[group].contiguous();mb=op.bias[group].contiguous()
                check(f'c{capacity}_real_partition',qtile,mk[:,:64*capacity].contiguous(),mv[:,:64*capacity].contiguous(),mb[:,:64*capacity].contiguous())
                check(f'c{capacity}_real_full_compressed',qtile,mk,mv,mb)
                if args.fused:
                    from analysis.validate_fisher_warp_fused import validate_fused
                    result.setdefault('fused',{})[str(capacity)]=validate_fused(capacity,op,q,k,v,seeds,ext,errors,args.fused_profile,args.fused_source,args.source_dir)
                    save()
                del op,seeds,mk,mv,mb,qtile;gc.collect();torch.cuda.empty_cache()
        assert not errors,errors
        result['valid']=True
    except Exception as e:
        result.update(valid=False,error=repr(e));raise
    finally:
        stop.set();thread.join();save()

if __name__=='__main__':main()
