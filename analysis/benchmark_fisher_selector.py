"""Fisher seed comparison and selector timing; single freshly exclusive GPU."""
import argparse,os,json,sys,threading,statistics,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.launch_quality_first_videos import require_clean,gpu_processes,now

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--production-sweep',action='store_true');p.add_argument('--layout',choices=['candidate','probe4','probe8','probe16'],default='candidate');p.add_argument('--layer',default='00');args=p.parse_args()
    assert not args.output.exists()
    initial=require_clean(args.gpu);os.environ.update(CUDA_VISIBLE_DEVICES=initial['gpu']['uuid'],OMP_NUM_THREADS='4',
        TORCH_EXTENSIONS_DIR=str(args.output.parent/'torch_extensions'),TRITON_CACHE_DIR=str(args.output.parent/'triton_cache'))
    stop=threading.Event();errors=[];samples=[];result={'gpu':initial,'variants':[],'layout':args.layout,'layer':args.layer}
    def monitor():
        while not stop.is_set():
            try:
                ps=gpu_processes(initial['gpu']['uuid']);samples.append({'utc':now(),'processes':ps})
                if any(v['pid']!=os.getpid() for v in ps):errors.append('foreign GPU process')
            except Exception as e:errors.append(repr(e))
            stop.wait(3)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    def save():
        result.update(monitor_errors=errors,monitor_samples=samples);args.output.write_text(json.dumps(result,indent=2))
    try:
        import torch
        from native_fisher.selector import NativeFisherSelector, _logits
        from attention.affine_hellinger_seed_triton import affine_hellinger_seed_rows_triton
        torch.set_num_threads(4);torch.set_grad_enabled(False)
        payload=torch.load(f'/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer{args.layer}.pt',map_location='cpu',weights_only=True)
        q=payload['q'].reshape(1,21,45,80,40,128)[:,:1].cuda();k=payload['k'].reshape(1,21,45,80,40,128).cuda();del payload
        def timed(fn):
            for _ in range(2):fn()
            values=[]
            for _ in range(5):
                assert not errors,errors
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();y=fn();b.record();b.synchronize();values.append(a.elapsed_time(b))
            return {'mean_ms':statistics.fmean(values),'samples_ms':values}
        for c in [4,8,10]:
            op=NativeFisherSelector(c,layout=args.layout);native=op(q,k).clone()
            result['source_sha256']=hashlib.sha256(op.source_path.read_bytes()).hexdigest()
            original=lambda:affine_hellinger_seed_rows_triton(q,k,feature_mode='fisher_geometric',num_seeds=c)
            ref=original();mismatch=native!=ref
            row={'capacity':c,'arithmetic':op.arithmetic,'seed_indices':ref.numel(),'mismatch_count':int(mismatch.sum()),
                 'per_slot_mismatches':mismatch.reshape(-1,c).sum(0).tolist(),'native':timed(lambda:op(q,k)),
                 'reference':timed(original)}
            row['stages']={
                'logits':timed(lambda:_logits[(45*20,40)](q,k,op.candidates,op.logits,num_warps=4,num_stages=1)),
                'gs':timed(lambda:op.gs()),
            }
            result['variants'].append(row);save();print(json.dumps(row),flush=True)
            if row['mismatch_count']:
                positions=mismatch.nonzero()[:10].cpu().tolist();row['first_mismatch_positions']=positions;save()
            else:
                zeros=torch.zeros_like(q);a=op(zeros,k).clone();b=affine_hellinger_seed_rows_triton(zeros,k,feature_mode='fisher_geometric',num_seeds=c)
                row['zero_q_ties_exact']=torch.equal(a,b);save();print('ZERO_Q',c,row['zero_q_ties_exact'],flush=True);del zeros,a,b
            if args.production_sweep:
                row['production_sweep']=[]
                zeros=torch.zeros_like(q)
                zero_ref=affine_hellinger_seed_rows_triton(zeros,k,feature_mode='fisher_geometric',num_seeds=c)
                for block_j in [1,2,4,8]:
                    for warps in [4,8]:
                        candidate=lambda x=q:affine_hellinger_seed_rows_triton(x,k,feature_mode='fisher_geometric',num_seeds=c,block_j=block_j,num_warps=warps)
                        item={'block_j':block_j,'warps':warps}
                        try:
                            actual=candidate();item['mismatch_count']=int((actual!=ref).sum());del actual
                            actual=candidate(zeros);item['zero_q_ties_exact']=torch.equal(actual,zero_ref);del actual
                            item['timing']=timed(candidate)
                        except Exception as e:item['error']=repr(e)
                        row['production_sweep'].append(item);save();print('PRODUCTION',c,json.dumps(item),flush=True)
                del zeros,zero_ref
            del op,native,ref,mismatch
        result['valid']=not errors and all(
            (v['arithmetic']=='fp32' or v['mismatch_count']==0) and v.get('zero_q_ties_exact')
            for v in result['variants']);save()
    except BaseException as e:result.update(valid=False,error=repr(e));save();raise
    finally:stop.set();thread.join(10);save()

if __name__=='__main__':main()
