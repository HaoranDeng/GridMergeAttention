"""Count zero-contribution compressed tokens on every head of real captures.

This is an IO-opportunity diagnostic, not an attention timing measurement.
The retained merge/selector and its invalid-cluster bias convention are intact.
"""
import argparse,hashlib,json,os,sys,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.launch_quality_first_videos import require_clean,gpu_processes,now

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();assert not args.output.exists()
    initial=require_clean(args.gpu)
    os.environ.update(CUDA_VISIBLE_DEVICES=initial['gpu']['uuid'],OMP_NUM_THREADS='4',
        TORCH_EXTENSIONS_DIR=str(args.output.parent/'torch_extensions'),TRITON_CACHE_DIR=str(args.output.parent/'triton_cache'))
    errors=[];samples=[];stop=threading.Event()
    result={'valid':False,'scope':'All40 heads of real step0 layers00/19/39; empty-cluster IO diagnostic only',
            'initial_gpu':initial,'layers':[],'source_files':{}}
    def save():
        result.update(monitor_errors=errors,monitor_samples=samples)
        args.output.write_text(json.dumps(result,indent=2))
    def monitor():
        while not stop.is_set():
            try:
                processes=gpu_processes(initial['gpu']['uuid']);samples.append({'utc':now(),'processes':processes})
                if any(x['pid']!=os.getpid() for x in processes):errors.append('foreign GPU process')
            except Exception as e:errors.append(repr(e))
            stop.wait(3)
    worker=threading.Thread(target=monitor,daemon=True);worker.start()
    try:
        import torch
        from native_fisher.fp8_c10 import retained_control,FP8C10Attention
        torch.set_grad_enabled(False);torch.set_num_threads(4)
        op=FP8C10Attention(retained_control(),'direct',source=ROOT/'native_fisher/fisher_c10_fp8_local_scores.cu')
        for path in (Path(__file__),ROOT/'native_fisher/fp8_c10.py',ROOT/'native_fisher/fisher_c10_fp8_local_scores.cu',ROOT/'native_fisher/operator.py',ROOT/'native_fisher/selector.py',ROOT/'native_fisher/fisher_gs_register_distribute.cu'):
            result['source_files'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        for layer in ('00','19','39'):
            path='/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer'+layer+'.pt'
            payload=torch.load(path,map_location='cpu',weights_only=True)
            q,k,v=[payload[n].reshape(1,21,45,80,40,128).cuda() for n in ('q','k','v')];del payload
            seeds=op.select(q,k,128**-.5);op.merge.pack(q,op.packed)
            row={'layer':layer,'capture':path,'chunks':[]}
            for head in range(0,40,5):
                assert not errors,errors
                op.merge_chunk(k,v,seeds,head,128**-.5)
                finite=torch.isfinite(op.bias)
                assert bool((finite|torch.isposinf(op.bias)).all()),'Unexpected bias convention'
                counts=finite.reshape(8400,945,10).sum(-1)
                hist=torch.bincount(counts.flatten(),minlength=11).cpu().tolist()
                empty=int((~finite).sum());total=finite.numel()
                padded=torch.nn.functional.pad(finite,(0,22),value=False)
                empty_tiles=int((~padded.reshape(8400,148,64).any(-1)).sum())
                chunk={'head_start':head,'total_tokens':total,'empty_tokens':empty,'empty_fraction':empty/total,
                    'nonempty_cluster_count_histogram':hist,'fully_empty_64_token_tiles':empty_tiles}
                row['chunks'].append(chunk)
                del finite,counts,padded
            row['total_tokens']=sum(v['total_tokens'] for v in row['chunks'])
            row['empty_tokens']=sum(v['empty_tokens'] for v in row['chunks'])
            row['empty_fraction']=row['empty_tokens']/row['total_tokens']
            row['avoidable_fp8_kv_read_GB']=row['empty_tokens']*256/1e9
            row['ideal_read_saving_ms_at_3_36_TB_s']=row['empty_tokens']*256/3.36e9
            result['layers'].append(row);save()
            print('EMPTY',layer,row['empty_fraction'],'ideal_read_saving_ms',row['ideal_read_saving_ms_at_3_36_TB_s'],flush=True)
            del q,k,v,seeds
        assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest for path,digest in result['source_files'].items())
        result['valid']=not errors
    except BaseException as e:
        result['error']=repr(e);raise
    finally:
        stop.set();worker.join(5);result['valid']=result['valid'] and not errors;save()

if __name__=='__main__':main()
