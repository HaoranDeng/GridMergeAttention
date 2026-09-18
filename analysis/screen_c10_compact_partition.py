"""Exact compact-producer gate and resumable full-support route diagnostic."""
import argparse,hashlib,json,os,sys,threading,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.launch_quality_first_videos import require_clean,gpu_processes,now
def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--block-m',type=int,choices=[16,32,64],default=16)
    p.add_argument('--frames',type=int,nargs='+',choices=[1,3,7,21],default=[1,3,7,21])
    p.add_argument('--route-warps',type=int,nargs='+',choices=[4,8,16],default=[4])
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
    result={'initial_gpu':initial,'scope':'Compact key-partition producer and one-head/six-column route diagnostic; not full call','producer_checks':[],'variants':[]}
    def save():
        result.update(monitor_errors=errors,monitor_samples=samples);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    try:
        import torch
        from native_fisher.operator import NativeFisherAttention,extension
        from native_fisher.compact_route import route_partition
        torch.set_grad_enabled(False);torch.set_num_threads(4)
        payload=torch.load('/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer00.pt',map_location='cpu',weights_only=True)
        q,k,v=[payload[n].reshape(1,21,45,80,40,128).cuda() for n in ('q','k','v')];del payload
        op=NativeFisherAttention(capacity=10,selector_source=ROOT/'native_fisher/fisher_gs_register_distribute.cu')
        seeds=op.select(q,k,128**-.5);op.merge.pack(q,op.packed)
        source=ROOT/'native_fisher/fisher_c10_compact_partition.cu'
        ext=extension(10,source);result['source_sha256']=ext.source_sha256;result['resources']=ext.resources();save()
        mk=torch.empty((21,6,640,128),device='cuda',dtype=torch.bfloat16);mv=torch.empty_like(mk);bias=torch.empty((21,6,640),device='cuda')
        state=(torch.empty((21,6,45,128),device='cuda'),torch.empty((21,6,45),device='cuda'),torch.empty((21,6,45),device='cuda'))
        out=torch.empty_like(q);expected=torch.empty_like(q)
        op.merge.run(op.packed,k,v,seeds,op.mk,op.mv,op.bias,0,5,21,128**-.5,op.empty,0,1)
        op.route(q,k,v,expected,0,128**-.5)
        for col in (0,42,78):
            cols=min(6,80-col)
            indices=torch.tensor([(f*80+c)*5 for f in range(21) for c in range(col,col+cols)],device='cuda')
            for begin in (0,448,896):
                count=min(64,945-begin)*10
                ext.run_compact(op.packed,k,v,seeds,mk,mv,bias,0,col,begin,1)
                record={'column':col,'key_begin':begin}
                for name,a,b in zip(('k','v','bias'),(mk,mv,bias),(op.mk,op.mv,op.bias)):
                    reference=b[indices,begin*10:begin*10+count].reshape(21,cols,count,*a.shape[3:])
                    record[name+'_exact']=torch.equal(a[:,:cols,:count],reference)
                result['producer_checks'].append(record);save();assert all(record[n+'_exact'] for n in ('k','v','bias')),record
        def group(frames,bm=None):
            bm=args.block_m if bm is None else bm
            for begin in range(0,945,64):
                ext.run_compact(op.packed,k,v,seeds,mk,mv,bias,0,0,begin,frames)
                route_partition(q,k,v,mk,mv,bias,state,out,0,0,begin,block_m=bm,num_warps=route_warps)
        for frames,route_warps in [(f,w) for f in args.frames for w in args.route_warps]:
            group(frames);a=out[0,:,:,:6,0].float();b=expected[0,:,:,:6,0].float();diff=a-b
            row={'frames_per_cta':frames,'query_block_m':args.block_m,'route_warps':route_warps,'relative_l2':float((diff.square().sum()/b.square().sum()).sqrt()),'max_abs':float(diff.abs().max()),'finite':bool(torch.isfinite(a).all())}
            result['variants'].append(row);save();assert row['finite'] and row['relative_l2']<.002,row
            torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):group(frames)
            graph.replay();torch.cuda.synchronize();times=[]
            for _ in range(7):
                assert not errors,errors
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();graph.replay();b.record();b.synchronize();times.append(a.elapsed_time(b))
            row['group_timing']={'mean_ms':statistics.fmean(times),'samples_ms':times}
            print('RESULT',json.dumps(row),flush=True);save();del graph
        result['valid']=not errors;save()
    except BaseException as e:result.update(valid=False,error=repr(e));save();raise
    finally:stop.set();thread.join(5);save()
if __name__=='__main__':main()
