"""Attribute cache-slice slowdown using H5 CUDA activity, not hardware counters."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.launch_quality_first_videos import require_clean,gpu_processes,now


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--capacity',type=int,choices=[4,10],required=True)
    p.add_argument('--frames',type=int,choices=[1,3,7,21],required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    assert not args.output.exists()
    initial=require_clean(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES']=initial['gpu']['uuid']
    os.environ['TORCH_EXTENSIONS_DIR']=str(args.output.parent/'torch_extensions')
    os.environ['TRITON_CACHE_DIR']=str(args.output.parent/'triton_cache')
    os.environ['OMP_NUM_THREADS']='4'
    stop=threading.Event()
    errors=[]
    samples=[]
    def monitor():
        while not stop.is_set():
            try:
                processes=gpu_processes(initial['gpu']['uuid'])
                samples.append({'utc':now(),'processes':processes})
                if any(row['pid']!=os.getpid() for row in processes):
                    errors.append('foreign GPU process')
            except Exception as error:
                errors.append(repr(error))
            stop.wait(3)
    thread=threading.Thread(target=monitor,daemon=True)
    thread.start()
    result={'scope':'Instrumented H5 graph call only; not complete40 timing or speedup',
            'initial_gpu':initial,'capacity':args.capacity,'frames':args.frames}
    try:
        import torch
        from native_fisher.cache_slices import CacheSlicedFisherAttention
        from native_fisher.graph import GraphFisherAttention
        from analysis.summarize_native_activity import summarize
        torch.set_grad_enabled(False)
        torch.set_num_threads(4)
        payload=torch.load('/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer00.pt',map_location='cpu',weights_only=True)
        q,k,v=[payload[n].reshape(1,21,45,80,40,128).cuda() for n in ('q','k','v')]
        del payload
        op=CacheSlicedFisherAttention(capacity=args.capacity,frames=args.frames)
        def call(qi,ki,vi,scale=128**-.5):
            return op(qi,ki,vi,scale=scale,head_count=5)
        graph=GraphFisherAttention(call,q,k,v)
        graph(q,k,v)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
            with torch.profiler.record_function('cache_slice_h5_graph'):
                output=graph(q,k,v)
                torch.cuda.synchronize()
        trace=args.output.with_suffix('.trace.json')
        profiler.export_chrome_trace(str(trace))
        result['activity']=summarize(trace)
        result['source_sha256']=op.merge.source_sha256
        result['sources']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (ROOT/'native_fisher/cache_slices.py',ROOT/'native_fisher/cache_slice_route.py')}
        result['valid']=not errors
    except BaseException as error:
        result.update(valid=False,error=repr(error))
        raise
    finally:
        stop.set()
        thread.join(10)
        result.update(monitor_errors=errors,monitor_samples=samples)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='monitor_samples'}),flush=True)


if __name__=='__main__':
    main()
