"""Isolate native-seed fusion producer resources and validate every ring snapshot."""
import argparse,gc,hashlib,json,os,sys,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.launch_quality_first_videos import require_clean,gpu_processes,now

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-dir',type=Path,default=ROOT/'native_fisher/ldmatrix_dmajor')
    p.add_argument('--capacities',type=int,nargs='+',choices=[4,10],default=[4,10])
    args=p.parse_args();assert not args.output.exists()
    initial=require_clean(args.gpu)
    os.environ.update(CUDA_VISIBLE_DEVICES=initial['gpu']['uuid'],TORCH_CUDA_ARCH_LIST='9.0a',MAX_JOBS='2',OMP_NUM_THREADS='4',TORCH_EXTENSIONS_DIR=str(args.output.parent/'torch_extensions'),TRITON_CACHE_DIR=str(args.output.parent/'triton_cache'))
    stop=threading.Event();errors=[];samples=[]
    def monitor():
        while not stop.is_set():
            try:
                ps=gpu_processes(initial['gpu']['uuid']);samples.append({'utc':now(),'processes':ps})
                if any(x['pid']!=os.getpid() for x in ps):errors.append('foreign GPU process')
            except Exception as e:errors.append(repr(e))
            stop.wait(3)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    result={'initial_gpu':initial,'scope':'Native-seed fusion producer resource diagnostic with observable K/V/bias ring snapshots; no routing or full-attention timing','capacities':{}}
    def save():
        result.update(monitor_errors=errors,monitor_samples=samples)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    try:
        import torch
        from torch.utils.cpp_extension import load
        from native_fisher.operator import NativeFisherAttention
        torch.set_grad_enabled(False);torch.set_num_threads(4)
        paths=[args.source_dir.resolve()/name for name in ('warp_mma.cuh','warp_merge.cuh','warp_route.cuh','producer_diagnostic.cu')]
        hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
        key=hashlib.sha256(''.join(hashes.values()).encode()).hexdigest()[:16]
        print('BUILD_PRODUCER',key,flush=True)
        ext=load(name='fisher_native_producer_'+key,sources=[str(paths[-1])],extra_cuda_cflags=['-O3','-lineinfo','--ptxas-options=-v'],verbose=False)
        assert hashes=={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
        result['sources']=hashes
        assert not errors,errors
        payload=torch.load(args.output.parent.parent/'gma_c8_cuda_150ms_20260908/real_full40_step0_layer00.pt',map_location='cpu',weights_only=True)
        q,k,v=[payload[n].reshape(1,21,45,80,40,128).cuda() for n in ('q','k','v')];del payload
        for cap in args.capacities:
            print('NATIVE',cap,flush=True)
            op=NativeFisherAttention(capacity=cap)
            seeds=op.select(q,k,128**-.5);op.merge.pack(q,op.packed)
            op.merge.run(op.packed,k,v,seeds,op.mk,op.mv,op.bias,0,5,21,128**-.5,op.empty,0,1)
            report={'resources':ext.resources(cap),'native_source_sha256':op.merge.source_sha256,'checks':[]}
            result['capacities'][str(cap)]=report
            print('PRODUCER_RESOURCES',cap,report['resources'],flush=True);save()
            for gs,gc_,ps,pc in ((0,2,0,2),(2092,2,7,2),(4198,2,14,1)):
                assert not errors,errors
                ok,ov,ob=ext.run(op.packed,k,v,seeds,cap,0,gs,gc_,ps,pc,128**-.5)
                for pi in range(pc):
                    part=ps+pi;start=part*64*cap;end=min((part+1)*64,945)*cap;count=end-start
                    for gi in range(gc_):
                        group=gs+gi;ah=group%5;pair=(group//5)%40;frame=group//200
                        indices=torch.tensor([(frame*80+pair*2+c)*5+ah for c in (0,1)],device='cuda')
                        row={'group':group,'partition':part,'live_tokens_per_column':count,
                             'k_equal':torch.equal(ok[pi,gi,:,:count],op.mk[indices,start:end]),
                             'v_equal':torch.equal(ov[pi,gi,:,:count],op.mv[indices,start:end]),
                             'bias_equal':torch.equal(ob[pi,gi,:,:count],op.bias[indices,start:end]),
                             'padding_zero_k':bool((ok[pi,gi,:,count:]==0).all()),
                             'padding_zero_v':bool((ov[pi,gi,:,count:]==0).all()),
                             'padding_inf_bias':bool(torch.isposinf(ob[pi,gi,:,count:]).all())}
                        report['checks'].append(row);save()
                        assert all(row[n] for n in ('k_equal','v_equal','bias_equal','padding_zero_k','padding_zero_v','padding_inf_bias')),row
                del ok,ov,ob,indices
            report['valid']=True;print('PRODUCER_GATE',cap,len(report['checks']),'passed',flush=True);save()
            del op,seeds;gc.collect();torch.cuda.empty_cache()
        assert not errors,errors
        result['valid']=True;save()
    except BaseException as e:
        result.update(valid=False,error=repr(e));save();raise
    finally:stop.set();thread.join(10);save()

if __name__=='__main__':main()
