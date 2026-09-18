"""Profile one warmed retained H5 merge; never report profiler timings as latency."""
import argparse,hashlib,json,os,sys,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.launch_quality_first_videos import require_clean,gpu_processes,now
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',type=Path,required=True)
args=p.parse_args();assert not args.output.exists();initial=require_clean(args.gpu)
os.environ['CUDA_VISIBLE_DEVICES']=initial['gpu']['uuid']
os.environ['TORCH_EXTENSIONS_DIR']=str(args.output.parent/'torch_extensions')
os.environ['TRITON_CACHE_DIR']=str(args.output.parent/'triton_cache')
stop=threading.Event();errors=[];samples=[]
def monitor():
    while not stop.is_set():
        try:
            ps=gpu_processes(initial['gpu']['uuid']);samples.append({'utc':now(),'processes':ps})
            if any(r['pid']!=os.getpid() for r in ps):errors.append('foreign GPU process')
        except Exception as e:errors.append(repr(e))
        stop.wait(3)
thread=threading.Thread(target=monitor,daemon=True);thread.start()
result={'initial_gpu':initial,'scope':'One warmed H5 merge under Nsight Compute; profiling evidence only, no latency claim'}
try:
    import torch
    from native_fisher.fp8_c10 import retained_control,FP8C10Attention
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    control=retained_control();op=FP8C10Attention(control,'direct',False,ROOT/'native_fisher/fisher_c10_fp8_shared_store.cu')
    result['source_sha256']=op.merge.source_sha256
    payload=torch.load('/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer00.pt',map_location='cpu',weights_only=True)
    q,k,v=[payload[n].reshape(1,21,45,80,40,128).cuda() for n in ('q','k','v')];del payload
    scale=128**-.5;seeds=op.select(q,k,scale);op.merge.pack(q,op.packed)
    for _ in range(2):op.merge_chunk(k,v,seeds,0,scale)
    torch.cuda.synchronize();assert not errors,errors
    torch.cuda.cudart().cudaProfilerStart()
    op.merge_chunk(k,v,seeds,0,scale)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    result['valid']=not errors
except BaseException as e:
    result.update(valid=False,error=repr(e));raise
finally:
    stop.set();thread.join(10)
    result.update(monitor_errors=errors,monitor_samples=samples)
    args.output.write_text(json.dumps(result,indent=2))
