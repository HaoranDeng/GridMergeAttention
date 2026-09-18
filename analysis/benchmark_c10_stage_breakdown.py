"""Sum phase events inside actual all40 calls, with matched unsplit control."""
import argparse,json,os,sys,threading,statistics,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.launch_quality_first_videos import require_clean,gpu_processes,now
def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',type=Path,required=True)
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
    result={'initial_gpu':initial,'scope':'Actual all40 calls with per-phase CUDA events, no H5 multiplication','layers':[]}
    def save():
        result.update(monitor_errors=errors,monitor_samples=samples);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    try:
        import torch
        from native_fisher.operator import NativeFisherAttention
        torch.set_grad_enabled(False);torch.set_num_threads(4)
        op=NativeFisherAttention(capacity=10,selector_source=ROOT/'native_fisher/fisher_gs_register_distribute.cu',merge_source=ROOT/'native_fisher/fisher_c10_half_owner.cu')
        result['merge_sha256']=op.merge.source_sha256
        result['selector_sha256']=hashlib.sha256(op.native_selector.source_path.read_bytes()).hexdigest()
        for layer in ('00','19','39'):
            payload=torch.load(f'/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer{layer}.pt',map_location='cpu',weights_only=True)
            q,k,v=[payload[n].reshape(1,21,45,80,40,128).cuda() for n in ('q','k','v')];del payload
            out=torch.empty_like(q)
            def instrument():
                ev=[torch.cuda.Event(enable_timing=True) for _ in range(19)]
                ev[0].record();seeds=op.select(q,k,128**-.5);ev[1].record()
                op.merge.pack(q,op.packed);ev[2].record()
                for i,head in enumerate(range(0,40,5)):
                    op.merge.run(op.packed,k,v,seeds,op.mk,op.mv,op.bias,head,5,21,128**-.5,op.empty,0,1);ev[3+2*i].record()
                    op.route(q,k,v,out,head,128**-.5);ev[4+2*i].record()
                ev[-1].synchronize()
                merges=[ev[2+2*i].elapsed_time(ev[3+2*i]) for i in range(8)]
                routes=[ev[3+2*i].elapsed_time(ev[4+2*i]) for i in range(8)]
                return {'selector_ms':ev[0].elapsed_time(ev[1]),'pack_ms':ev[1].elapsed_time(ev[2]),'merge_ms':sum(merges),'route_ms':sum(routes),'total_ms':ev[0].elapsed_time(ev[-1]),'merge_by_h5_ms':merges,'route_by_h5_ms':routes}
            ref=op(q,k,v);instrument();exact=torch.equal(out,ref);del ref
            assert exact
            op(q,k,v);instrument();torch.cuda.synchronize()
            row={'layer':layer,'bitwise_equal':exact,'instrumented':[],'native_call_ms':[]};result['layers'].append(row)
            for block in range(6):
                for name in (('instrumented','native') if block%2==0 else ('native','instrumented')):
                    assert not errors,errors
                    if name=='instrumented':row['instrumented'].append(instrument())
                    else:
                        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        a.record();op(q,k,v);b.record();b.synchronize();row['native_call_ms'].append(a.elapsed_time(b))
            row['means']={name:statistics.fmean(x[name] for x in row['instrumented']) for name in ('selector_ms','pack_ms','merge_ms','route_ms','total_ms')}
            row['means']['native_call_ms']=statistics.fmean(row['native_call_ms'])
            print('RESULT',layer,json.dumps(row['means']),flush=True);save();del q,k,v,out
        result['means']={name:statistics.fmean(row['means'][name] for row in result['layers']) for name in result['layers'][0]['means']}
        result['valid']=not errors;save()
    except BaseException as e:result.update(valid=False,error=repr(e));save();raise
    finally:stop.set();thread.join(5);save()
if __name__=='__main__':main()
