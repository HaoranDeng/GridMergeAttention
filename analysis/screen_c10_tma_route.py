"""Paired complete H5 route screen, including unchanged SQ1."""
import argparse,json,os,sys,threading,statistics,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.launch_quality_first_videos import require_clean,gpu_processes,now
def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--route-block',type=int,choices=[64,128,256],default=64);p.add_argument('--route-stages',type=int,choices=[1,3],default=3)
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
    result={'initial_gpu':initial,'scope':'Complete H5 route only with unchanged SQ1; not full attention','source_sha256':hashlib.sha256((ROOT/'native_fisher/tma_route.py').read_bytes()).hexdigest(),'variants':[]}
    def save():
        result.update(monitor_errors=errors,monitor_samples=samples);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    try:
        import torch
        import triton
        triton.set_allocator(lambda size,alignment,stream: torch.empty((size,),device="cuda",dtype=torch.int8))
        from native_fisher.operator import NativeFisherAttention
        from native_fisher.tma_route import tma_route
        from attention import hard_c4_wan21_14b_720p_triton as h
        torch.set_grad_enabled(False);torch.set_num_threads(4)
        op=NativeFisherAttention(capacity=10,selector_source=ROOT/'native_fisher/fisher_gs_register_distribute.cu',merge_source=ROOT/'native_fisher/fisher_c10_long_tiles_tree.cu',tiles=84)
        tma_route.block=args.route_block;tma_route.stages=args.route_stages
        result['candidate_config']={'block':args.route_block,'stages':args.route_stages,'warps':4}
        control=h._hard_c4_720p_route_kernel
        for layer in ('00','19','39'):
            payload=torch.load(f'/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer{layer}.pt',map_location='cpu',weights_only=True)
            q,k,v=[payload[n].reshape(1,21,45,80,40,128).cuda() for n in ('q','k','v')];del payload
            seeds=op.select(q,k,128**-.5);op.merge.pack(q,op.packed)
            op.merge.run(op.packed,k,v,seeds,op.mk,op.mv,op.bias,0,5,84,128**-.5,op.empty,0,1)
            out=torch.empty_like(q);expected=torch.empty_like(q)
            h._hard_c4_720p_route_kernel=control;op.route(q,k,v,expected,0,128**-.5)
            h._hard_c4_720p_route_kernel=tma_route;op.route(q,k,v,out,0,128**-.5)
            row={'layer':layer,'bitwise_equal':torch.equal(out[...,:5,:],expected[...,:5,:]),'valid_cluster_fraction':float(torch.isfinite(op.bias).float().mean())}
            diff=out[...,:5,:].float()-expected[...,:5,:].float()
            row['relative_l2']=float((diff.square().sum()/expected[...,:5,:].float().square().sum()).sqrt())
            row['max_abs']=float(diff.abs().max());row['finite']=bool(torch.isfinite(out[...,:5,:]).all())
            result['variants'].append(row);save();assert row['finite'] and row['relative_l2']<.002,row
            pair={'candidate':[],'control':[]}
            for block in range(4):
                for name in (('candidate','control','control','candidate') if block%2==0 else ('control','candidate','candidate','control')):
                    assert not errors,errors
                    h._hard_c4_720p_route_kernel=tma_route if name=='candidate' else control
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record();op.route(q,k,v,out,0,128**-.5);b.record();b.synchronize();pair[name].append(a.elapsed_time(b))
            row['timing']={n:{'mean_ms':statistics.fmean(v),'samples_ms':v} for n,v in pair.items()}
            print('RESULT',json.dumps(row),flush=True);save();del q,k,v,out,expected,seeds
        h._hard_c4_720p_route_kernel=control;result['valid']=not errors;save()
    except BaseException as e:result.update(valid=False,error=repr(e));save();raise
    finally:stop.set();thread.join(5);save()
if __name__=='__main__':main()
