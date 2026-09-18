"""Guarded numerical/performance screen of native WGMMA double-ring fusion."""
import argparse,hashlib,json,os,statistics,sys,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.launch_quality_first_videos import require_clean,gpu_processes,now

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--heads',type=int,choices=[1,5],default=1);p.add_argument('--static-registers',action='store_true')
    p.add_argument('--diagnostic-timing',action='store_true',help='Measure a finite candidate without promoting a failed numerical gate')
    p.add_argument('--source-directory',type=Path,default=ROOT/'native_fisher')
    args=p.parse_args();assert not args.output.exists();initial=require_clean(args.gpu)
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
    result={'initial_gpu':initial,'scope':f'Whole H{args.heads} merge plus full-support attention and partial combination; excludes selector/packing; not complete H40','heads':args.heads}
    def save():
        result.update(monitor_errors=errors,monitor_samples=samples);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    try:
        import torch
        from native_fisher.operator import NativeFisherAttention,extension
        from native_fisher.c10_ring_combine import combine
        torch.set_grad_enabled(False);torch.set_num_threads(4)
        paths=[args.source_directory/n for n in ('c10_wgmma_ring.cuh','c10_ring_consumer.cuh','c10_ring_pipeline.cu')]
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        stamp=hashlib.sha256(''.join(hashes.values()).encode()).hexdigest()[:12]
        ext=extension(10,paths[-1],name_suffix='_all'+stamp)
        assert hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        result['sources']=hashes;result['combine_sha256']=hashlib.sha256((ROOT/'native_fisher/c10_ring_combine.py').read_bytes()).hexdigest()
        result['resources']={'static':ext.resources(False),'dynamic':ext.resources(True)}
        dynamic=not args.static_registers;result['dynamic_registers']=dynamic
        resources=result['resources']['dynamic' if dynamic else 'static']
        assert resources[3]>=1 and (not dynamic or resources[0]==96),resources
        print('RESOURCES',result['resources'],flush=True);save()
        payload=torch.load('/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer00.pt',map_location='cpu',weights_only=True)
        q,k,v=[payload[n].reshape(1,21,45,80,40,128).cuda() for n in ('q','k','v')];del payload
        op=NativeFisherAttention(capacity=10,tiles=84,merge_source=ROOT/'native_fisher/fisher_c10_long_tiles_tree.cu',selector_source=ROOT/'native_fisher/fisher_gs_register_distribute.cu')
        seeds=op.select(q,k,128**-.5);op.merge.pack(q,op.packed)
        op.merge.run(op.packed,k,v,seeds,op.mk,op.mv,op.bias,0,5,84,128**-.5,op.empty,0,1)
        expected=torch.empty_like(q);op.route(q,k,v,expected,0,128**-.5)
        acc=torch.empty((21*80*args.heads,16,45,128),device='cuda');maximum=torch.empty(acc.shape[:-1],device='cuda');den=torch.empty_like(maximum)
        out=torch.empty_like(q)
        def check(a,b):
            a,b=a.float(),b.float();diff=a-b
            return {'finite':bool(torch.isfinite(a).all()),'bitwise_equal':torch.equal(a,b),'relative_l2':float((diff.square().sum()/b.square().sum()).sqrt()),'max_abs':float(diff.abs().max())}
        print('SMOKE_START',flush=True)
        ext.run(op.packed,k,v,seeds,acc,maximum,den,0,args.heads,0,1,14,1,dynamic)
        torch.cuda.synchronize()
        references=[]
        for col in range(5):
            slot=col*5
            kk=torch.cat((op.mk[slot,8960:9450],k.reshape(945,80,40,128)[896:,col,0]),0).float()
            vv=torch.cat((op.mv[slot,8960:9450],v.reshape(945,80,40,128)[896:,col,0]),0).float()
            bias=torch.cat((op.bias[slot,8960:9450],torch.zeros(49,device='cuda')))
            z=q[0,0,:,col,0].float()@kk.T*(128**-.5)-bias
            references.append(torch.softmax(z,-1)@vv)
        selected=torch.arange(5,device='cuda')*args.heads
        actual=acc[selected,14]/den[selected,14,:,None]
        result['smoke_correctness']=check(actual,torch.stack(references));print('SMOKE',result['smoke_correctness'],flush=True);save()
        assert result['smoke_correctness']['finite'] and result['smoke_correctness']['relative_l2']<.003
        del references,actual,kk,vv,z,bias
        def candidate():
            ext.run(op.packed,k,v,seeds,acc,maximum,den,0,args.heads,0,21*16*args.heads,0,16,dynamic)
            combine(acc,maximum,den,out,args.heads)
        candidate();torch.cuda.synchronize()
        result['correctness']=check(out[...,:args.heads,:],expected[...,:args.heads,:]);print('WHOLE',result['correctness'],flush=True);save()
        result['numerical_gate_passed']=result['correctness']['finite'] and result['correctness']['relative_l2']<.002
        result['diagnostic_only']=args.diagnostic_timing
        assert result['correctness']['finite'] and (result['numerical_gate_passed'] or args.diagnostic_timing)
        candidate();torch.cuda.synchronize();values=[]
        for _ in range(6):
            assert not errors,errors
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();candidate();b.record();b.synchronize();values.append(a.elapsed_time(b))
        result['timing']={'mean_ms':statistics.fmean(values),'samples_ms':values}
        result['measurement_valid']=not errors
        result['valid']=not errors and result['numerical_gate_passed'];print('TIMING',result['timing'],flush=True);save()
    except BaseException as e:result.update(valid=False,error=repr(e));save();raise
    finally:stop.set();thread.join(5);save()

if __name__=='__main__':main()
