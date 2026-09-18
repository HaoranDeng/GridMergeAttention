"""Exercise the actual video dispatch and audit on captured full40 QKV."""
import json,os,sys,threading
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.launch_quality_first_videos import require_clean,gpu_processes,now

def main():
    output=Path(sys.argv[2]);assert not output.exists()
    state=require_clean(int(sys.argv[1]))
    os.environ['CUDA_VISIBLE_DEVICES']=state['gpu']['uuid']
    os.environ['OMP_NUM_THREADS']='4'
    os.environ['TRITON_CACHE_DIR']=str(output.parent/'triton_cache')
    errors=[];samples=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                active=gpu_processes(state['gpu']['uuid']);samples.append(dict(utc=now(),processes=active))
                if any(p['pid']!=os.getpid() for p in active):errors.append('foreign GPU process')
            except Exception as e:errors.append(repr(e))
            stop.wait(3)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    result=dict(valid=False,gpu=state)
    try:
        import torch
        from attention import token_merging as token
        from attention.affine_hellinger_seed_triton import fisher_geometric_c10_seed_rows_triton as select
        from analysis.fisher_c10_video_adapter import FisherC10MergeAdapter
        from analysis.quality_first_mode_contract import quality_first_token_merging_args
        from analysis.run_quality_first_videos import DispatchAudit,MERGE_SYMBOL
        torch.set_grad_enabled(False);torch.set_num_threads(4)
        payload=torch.load('/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908/real_full40_step0_layer00.pt',map_location='cpu',weights_only=True)
        q,k,v=[payload[n].reshape(1,75600,40,128).cuda() for n in ('q','k','v')];del payload
        method='fisher_geometric_c10_rf0'
        config=quality_first_token_merging_args(method)
        audit=DispatchAudit(method,output.parent/'unused_dispatch_audit.json')
        audit.active_layer=0
        config={k:v for k,v in config.items() if k!='enable'}
        args=(q,k,v,config.pop('f_tied'),config.pop('h_reduce'),config.pop('w_reduce'),45,80)
        audit.enter_core(args,config)
        a=FisherC10MergeAdapter()
        def selector(*args,**kwargs):
            audit.enter_selector(args,kwargs)
            result=select(*args,**kwargs)
            audit.leave_selector(result);return result
        def merge(*args,**kwargs):
            audit.enter_merge(args,kwargs)
            return a(*args,**kwargs)
        with patch.object(token,'entropy_mass_seed_rows_triton',selector),patch.object(token,MERGE_SYMBOL,merge):
            actual=token.token_merging_attn(*args,**config)
        assert (audit.core_calls,audit.selector_calls,audit.merge_calls)==(1,1,10)
        assert torch.isfinite(actual).all()
        qb,kb,vb=[x.reshape(1,21,45,80,40,128) for x in (q,k,v)]
        seeds=select(qb[:,:1],kb,128**-.5)
        reference=torch.empty_like(qb)
        for head in range(0,40,4):
            a(qb,kb,vb,seeds,128**-.5,top_r=1,head_start=head,head_count=4,
              selector='query_column',seed_rows_native_layout=True,query_column_major=True,
              merge_loop_stages=1,merge_disable_licm=False,split_kv_accum=True,output=reference)
        torch.cuda.synchronize()
        assert torch.equal(actual,reference.reshape_as(actual))
        assert not errors,errors
        result.update(valid=True,seed_mode=config['seed_mode'],bitwise_equal=True,
            core_calls=audit.core_calls,selector_calls=audit.selector_calls,merge_calls=audit.merge_calls,
            scope='actual production token_merging_attn plus video DispatchAudit; all40 captured QKV')
    except BaseException as e:
        result['failure']=repr(e);raise
    finally:
        stop.set();thread.join(timeout=10)
        result.update(monitor_errors=errors,monitor_samples=samples)
        output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='monitor_samples'}),flush=True)

if __name__=='__main__':main()
