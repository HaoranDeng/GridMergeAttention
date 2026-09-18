"""Isolate compressed CUDA, SQ1, and state-combination numerical errors."""
import argparse,json,os,sys,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.launch_quality_first_videos import require_clean,gpu_processes,now

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();assert not args.output.exists()
    initial=require_clean(args.gpu);os.environ['CUDA_VISIBLE_DEVICES']=initial['gpu']['uuid']
    os.environ['TORCH_EXTENSIONS_DIR']=str(args.output.parent/'torch_extensions')
    os.environ['TRITON_CACHE_DIR']=str(args.output.parent/'triton_cache')
    stop=threading.Event();errors=[];samples=[];result={'initial_gpu':initial}
    def monitor():
        while not stop.is_set():
            try:
                ps=gpu_processes(initial['gpu']['uuid']);samples.append({'utc':now(),'processes':ps})
                if any(r['pid']!=os.getpid() for r in ps):errors.append('foreign GPU process')
            except Exception as e:errors.append(repr(e))
            stop.wait(3)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    def save():
        result.update(monitor_errors=errors,monitor_samples=samples);args.output.write_text(json.dumps(result,indent=2))
    try:
        import torch
        from native_fisher.fp8_c10 import retained_control,FP8C10Attention
        from native_fisher.fp8_flashinfer_full import FlashinferFullRoute,_combine
        torch.set_num_threads(4);torch.set_grad_enabled(False)
        def check(a,b):
            a=a.float();b=b.float();d=a-b
            return {'relative_l2':float((d.square().sum()/b.square().sum().clamp_min(1e-30)).sqrt()),'max_abs':float(d.abs().max()),'mean_abs':float(d.abs().mean()),'finite':bool(torch.isfinite(a).all())}
        control=retained_control();op=FP8C10Attention(control,'direct');fi=FlashinferFullRoute()
        payload=torch.load('/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer00.pt',map_location='cpu',weights_only=True)
        q,k,v=[payload[n].reshape(1,21,45,80,40,128).cuda() for n in ('q','k','v')];del payload
        scale=128**-.5;head=0;seeds=op.select(q,k,scale);op.merge.pack(q,op.packed);op.merge_chunk(k,v,seeds,head,scale)
        packed_q=q[0,:,:,:,head:head+5,:].permute(0,2,3,1,4).reshape(-1,1,128)
        a,la=fi.compressed(packed_q,op.mk,op.mv,op.bias,scale)
        ta=torch.empty_like(a,dtype=torch.float32);tla=torch.empty_like(la)
        fi.state(op,q,k,v,ta,tla,head,scale,compressed=True)
        fi.state(op,q,k,v,fi.singleton_out,fi.singleton_lse,head,scale)
        result['compressed_output']=check(a,ta);result['compressed_log2_lse']=check(la,tla)
        result['manual_samples']=[]
        for group,row in [(0,0),(1,20),(7,44),(400,3),(8399,30)]:
            qvec=packed_q[group*45+row,0].float()
            logits=op.mk[group].float()@qvec*scale-op.bias[group]
            ref=torch.softmax(logits,0)@op.mv[group].float()
            reflse=torch.logsumexp(logits,0)/__import__('math').log(2)
            i=group*45+row
            result['manual_samples'].append({'group':group,'row':row,'cuda_output':check(a[i,0],ref),'triton_output':check(ta[i,0],ref),'cuda_lse':float(la[i]),'triton_lse':float(tla[i]),'manual_log2_lse':float(reflse),'bias_min':float(op.bias[group].min()),'bias_max':float(op.bias[group].max())})
        expected=torch.empty_like(q);combined=torch.empty_like(q)
        op.route(q,k,v,expected,head,scale)
        _combine[(8400*45,)](ta,tla,fi.singleton_out,fi.singleton_lse,combined,head,num_warps=4)
        result['triton_split_combine']=check(combined[...,head:head+5,:],expected[...,head:head+5,:])
        _combine[(8400*45,)](a,la,fi.singleton_out,fi.singleton_lse,combined,head,num_warps=4)
        result['cuda_split_combine']=check(combined[...,head:head+5,:],expected[...,head:head+5,:])
        torch.cuda.synchronize();result['valid']=not errors;save();print(json.dumps({k:v for k,v in result.items() if not k.startswith('monitor')},indent=2),flush=True)
    except BaseException as e:
        result.update(valid=False,error=repr(e));save();raise
    finally:
        stop.set();thread.join(10);save()

if __name__=='__main__':main()
