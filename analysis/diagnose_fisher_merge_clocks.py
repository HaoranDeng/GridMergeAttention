"""Sample native merge CTA stage clocks and quantify instrumentation overhead."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import threading

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.launch_quality_first_videos import require_clean,gpu_processes,now


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--capacity',type=int,choices=[4,10],required=True)
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
    monitor_samples=[]
    def monitor():
        while not stop.is_set():
            try:
                processes=gpu_processes(initial['gpu']['uuid'])
                monitor_samples.append({'utc':now(),'processes':processes})
                if any(row['pid']!=os.getpid() for row in processes):
                    errors.append('foreign GPU process')
            except Exception as error:
                errors.append(repr(error))
            stop.wait(3)
    thread=threading.Thread(target=monitor,daemon=True)
    thread.start()
    result={'initial_gpu':initial,'capacity':args.capacity,
            'scope':'H5 merge diagnostic only; stage clocks are per-CTA issue intervals, not full-call times or hardware counters',
            'sampled_frames':[]}
    def save():
        result.update(monitor_errors=errors,monitor_samples=monitor_samples)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    try:
        import numpy as np
        import torch
        from native_fisher.operator import NativeFisherAttention,extension
        torch.set_grad_enabled(False)
        torch.set_num_threads(4)
        print('LOAD',flush=True)
        payload=torch.load('/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer00.pt',map_location='cpu',weights_only=True)
        q,k,v=[payload[n].reshape(1,21,45,80,40,128).cuda() for n in ('q','k','v')]
        del payload
        op=NativeFisherAttention(capacity=args.capacity)
        control=op.merge
        source=ROOT/f'native_fisher/fisher_c{args.capacity}_clock.cu'
        print('BUILD',flush=True)
        instrumented=extension(args.capacity,source)
        result.update(source_sha256=instrumented.source_sha256,control_source_sha256=control.source_sha256,
                      resources=instrumented.resources(),control_resources=control.resources())
        print('RESOURCES',result['resources'],result['control_resources'],flush=True)
        seeds=op.select(q,k,128**-.5)
        control.pack(q,op.packed)
        groups=5 if args.capacity==4 else 14
        clocks=torch.zeros((groups,5,945,16),device='cuda',dtype=torch.int64)
        out=torch.empty_like(q)
        def run(module,trace,frame=0):
            module.run(op.packed,k,v,seeds,op.mk,op.mv,op.bias,0,5,21,128**-.5,trace,frame,1)
        def timed(fn):
            for _ in range(2):
                fn()
            values=[]
            for _ in range(5):
                assert not errors,errors
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                end.synchronize()
                values.append(start.elapsed_time(end))
            return {'mean_ms':statistics.fmean(values),'samples_ms':values}
        run(control,op.empty)
        op.route(q,k,v,out,0,128**-.5)
        expected=out[...,:5,:].clone()
        result['control_merge']=timed(lambda:run(control,op.empty))
        result['instrumented_disabled']=timed(lambda:run(instrumented,op.empty))
        sequence=list(range(15))
        names=['query_load_and_score_mma','normalization','shared_owner_or_marker','sq1_weights',
               'k_merge_mma','k_residual_scale','k_staging_wait','k_stage_output','k_tma_enqueue',
               'v_merge_mma','v_residual_scale','v_staging_wait','v_stage_output','v_tma_enqueue']
        if args.capacity==10:
            sequence.insert(3,15)
            names[2:3]=['previous_v_staging_wait','shared_owner']
        for frame in (0,10,20):
            clocks.zero_()
            row={'frame':frame,'timing':timed(lambda:run(instrumented,clocks,frame))}
            op.route(q,k,v,out,0,128**-.5)
            actual=out[...,:5,:]
            row['h5_output_equal']=torch.equal(actual,expected)
            row['finite']=bool(torch.isfinite(actual).all())
            assert row['h5_output_equal'] and row['finite'],row
            raw=clocks.cpu().numpy()
            bulk=raw if args.capacity==4 else raw[:13]
            values=bulk.reshape(-1,16)[:,sequence]
            assert (values>0).all(),'Missing timestamp'
            differences=np.diff(values,axis=1)
            assert (differences>=0).all(),'Nonmonotonic within-CTA clocks'
            means=differences.mean(axis=0)
            row['cta_count']=int(len(values))
            row['sampled_cycle_span_mean']=float(means.sum())
            row['stage_cycles']={name:{'mean':float(means[index]),
                                      'p50':float(np.percentile(differences[:,index],50)),
                                      'p90':float(np.percentile(differences[:,index],90)),
                                      'fraction_of_sampled_span':float(means[index]/means.sum())}
                                  for index,name in enumerate(names)}
            archive=args.output.with_name(args.output.stem+f'_frame{frame}.npz')
            np.savez_compressed(archive,timestamps=raw,sequence=np.array(sequence))
            row['clock_archive']=str(archive)
            result['sampled_frames'].append(row)
            save()
            print('FRAME',json.dumps(row),flush=True)
        result['control_merge_after']=timed(lambda:run(control,op.empty))
        result['valid']=not errors
        result['tail_scope']='C10 final two-column global-store tile excluded from CTA clock samples; included in H5 event timings'
        save()
        print('DONE',json.dumps({key:value for key,value in result.items() if key not in ('monitor_samples','sampled_frames')}),flush=True)
    except BaseException as error:
        result.update(valid=False,error=repr(error))
        save()
        raise
    finally:
        stop.set()
        thread.join(10)
        save()


if __name__=='__main__':
    main()
