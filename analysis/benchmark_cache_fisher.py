"""Exclusive-H200 complete native40 tests of small merge/route slices."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.launch_quality_first_videos import require_clean,gpu_processes,now


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--capacity',type=int,choices=[4,10],required=True)
    parser.add_argument('--frames',type=int,nargs='+',default=[21,7,3,1])
    parser.add_argument('--layer',default='00')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert all(n in (1,3,7,21) for n in args.frames)
    assert not args.output.exists()
    initial=require_clean(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES']=initial['gpu']['uuid']
    os.environ['TORCH_EXTENSIONS_DIR']=str(args.output.parent/'torch_extensions')
    os.environ['TRITON_CACHE_DIR']=str(args.output.parent/'triton_cache')
    os.environ['OMP_NUM_THREADS']='4'
    args.output.parent.mkdir(parents=True,exist_ok=True)
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
    result={'initial_gpu':initial,'capacity':args.capacity,'layer':args.layer,
            'scope':'Complete native40 BF16 720p: Fisher selector, packing, every SQ1 merge and full route',
            'graph_scope':'Graph replay includes three input copies and independent output clone; selection recomputed each replay',
            'target_ms':80 if args.capacity==4 else 180,'variants':[]}
    def save():
        result.update(monitor_errors=errors,monitor_samples=samples)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    try:
        print('LOAD',flush=True)
        import torch
        from native_fisher.operator import NativeFisherAttention,extension
        from native_fisher.cache_slices import CacheSlicedFisherAttention
        from native_fisher.graph import GraphFisherAttention
        from attention.affine_hellinger_seed_triton import affine_hellinger_seed_rows_triton
        torch.set_grad_enabled(False)
        torch.set_num_threads(4)
        capture=Path('/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908')/f'real_full40_step0_layer{args.layer}.pt'
        payload=torch.load(capture,map_location='cpu',weights_only=True)
        q,k,v=[payload[n].reshape(1,21,45,80,40,128).cuda() for n in ('q','k','v')]
        del payload
        print('BUILD',flush=True)
        op=CacheSlicedFisherAttention(capacity=args.capacity,frames=args.frames[0])
        control=extension(args.capacity)
        candidate=op.merge
        result['source_sha256']=candidate.source_sha256
        result['control_source_sha256']=control.source_sha256
        result['resources']=candidate.resources()
        result['sources']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/'native_fisher/cache_slices.py',ROOT/'native_fisher/cache_slice_route.py',
                      ROOT/'native_fisher/graph.py',ROOT/'analysis/benchmark_cache_fisher.py']}
        def native(qi=q,ki=k,vi=v,scale=128**-.5):
            op.merge=control
            try:
                return NativeFisherAttention.__call__(op,qi,ki,vi,scale=scale)
            finally:
                op.merge=candidate
        seeds=op.select(q,k,128**-.5)
        reference_seeds=affine_hellinger_seed_rows_triton(q[:,:1],k,scale=128**-.5,
            feature_mode='fisher_geometric',num_seeds=args.capacity,block_j=4)
        result['seed_mismatch_count']=int((seeds!=reference_seeds).sum())
        assert result['seed_mismatch_count']==0
        del seeds,reference_seeds
        def check(actual,expected):
            difference=actual.float()-expected.float()
            return {'bitwise_equal':torch.equal(actual,expected),'finite':bool(torch.isfinite(actual).all()),
                'max_abs':float(difference.abs().max()),
                'relative_l2':float((difference.square().sum()/expected.float().square().sum()).sqrt())}
        def timed(fn,repeats=5):
            for _ in range(2):
                fn()
            values=[]
            for _ in range(repeats):
                assert not errors,errors
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                output=fn()
                end.record()
                end.synchronize()
                values.append(start.elapsed_time(end))
                del output
            return {'mean_ms':statistics.fmean(values),'samples_ms':values}
        expected=native()
        result['native_control']=timed(native)
        print('CONTROL',result['native_control'],flush=True)
        for frames in args.frames:
            op.frames=frames
            row={'frames':frames,'head_span':1,'columns_per_tile':op.columns,
                 'merge_route_pairs':40*((80+op.columns-1)//op.columns)*(21//frames),
                 'logical_max_kv_slice_bytes':2*frames*op.columns*945*args.capacity*128*2}
            result['variants'].append(row)
            print('CHECK',frames,flush=True)
            actual=op(q,k,v)
            row['uncaptured_correctness']=check(actual,expected)
            del actual
            assert row['uncaptured_correctness']['bitwise_equal'],row
            row['uncaptured']=timed(lambda:op(q,k,v),repeats=3)
            print('CAPTURE',frames,flush=True)
            begin=time.monotonic()
            graph=GraphFisherAttention(op,q,k,v)
            row['capture_and_warmup_seconds']=time.monotonic()-begin
            actual=graph(q,k,v)
            row['graph_correctness']=check(actual,expected)
            del actual
            assert row['graph_correctness']['bitwise_equal'],row
            row['graph']=timed(lambda:graph(q,k,v))
            changed=q*.875
            changed_expected=native(changed,k,v)
            actual=graph(changed,k,v)
            row['changed_query_correctness']=check(actual,changed_expected)
            assert row['changed_query_correctness']['bitwise_equal'],row
            del actual,changed,changed_expected
            actual=graph(q,k,v)
            row['restored_query_equal']=torch.equal(actual,expected)
            assert row['restored_query_equal']
            del actual,graph
            gc.collect()
            torch.cuda.empty_cache()
            row['native_control_after']=timed(native)
            save()
            print('RESULT',json.dumps(row),flush=True)
        # Match replay overhead with a separate captured native-control run.
        native_graph=GraphFisherAttention(native,q,k,v)
        result['native_graph_control']=timed(lambda:native_graph(q,k,v))
        del native_graph,expected
        from torch.nn.attention import sdpa_kernel,SDPBackend
        flat=[x.reshape(1,75600,40,128).transpose(1,2) for x in (q,k,v)]
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            result['sdpa']=timed(lambda:torch.nn.functional.scaled_dot_product_attention(*flat))
        for row in result['variants']:
            row['speedup_vs_sdpa']=result['sdpa']['mean_ms']/row['graph']['mean_ms']
        result['valid']=not errors
        save()
        print('DONE',json.dumps({k:v for k,v in result.items() if k!='monitor_samples'}),flush=True)
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
