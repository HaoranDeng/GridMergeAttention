"""Exact-output launch tuning; all changes are local to this benchmark."""
import argparse,json,statistics,sys,os,threading,subprocess,time,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--capture',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    import torch
    from native_hip.selector import HipFisherSelector
    from attention.affine_hellinger_seed_triton import affine_hellinger_seed_rows_triton
    from attention import hard_c4_wan21_14b_720p_triton as hard
    from attention.singleton_c4_wan21_14b_720p_triton import _hard_c4_720p_temporal_shared_global_mass_attention_triton_blocks as run
    assert torch.cuda.device_count()==1
    torch.set_grad_enabled(False);torch.set_num_threads(4)
    result={'job':os.environ.get('SLURM_JOB_ID'),'capture':args.capture,'sweep':[],'monitor':[],'monitor_errors':[]}
    def save():Path(args.output).write_text(json.dumps(result,indent=2))
    stop=threading.Event()
    def monitor():
        while not stop.is_set():
            r=subprocess.run(['rocm-smi','--showpids','--showuse','--showmeminfo','vram'],capture_output=True,text=True)
            result['monitor'].append({'time':time.time(),'returncode':r.returncode,'text':r.stdout})
            pids=[int(pid) for pid in re.findall(r'^\s*(\d+)\s+\S+',r.stdout,re.M)]
            if r.returncode or any(pid!=os.getpid() for pid in pids):
                result['monitor_errors'].append({'returncode':r.returncode,'pids':pids})
            stop.wait(3)
    th=threading.Thread(target=monitor,daemon=True);th.start()
    try:
        payload=torch.load(args.capture,map_location='cpu',weights_only=True)
        q,k,v=[payload[n].reshape(1,21,45,80,40,128).cuda() for n in ('q','k','v')];del payload
        op=HipFisherSelector();seeds=op(q,k);out=torch.empty_like(q)
        cfg={'heads':5,'j':4,'merge_warps':4,'route_warps':4,'route_block':64,'merge_stages':1}
        def reference_seeds():
            return affine_hellinger_seed_rows_triton(q[:,:1],k,feature_mode='fisher_geometric',block_j=4)
        def full(include_selector=False,profile=False,reference_selector=False):
            hard.HARD_720_MERGE_NUM_WARPS=cfg['merge_warps'];hard.HARD_720_ROUTE_NUM_WARPS=cfg['route_warps'];hard.HARD_720_ROUTE_BLOCK_TOKENS=cfg['route_block']
            s=reference_seeds() if reference_selector else (op(q,k) if include_selector else seeds)
            times={'merge_ms':0.,'route_ms':0.}
            for h in range(0,40,cfg['heads']):
                t={} if profile else None
                run(q,k,v,s,top_r=1,head_start=h,head_count=cfg['heads'],query_column_selector=True,seed_rows_native_layout=True,
                    query_column_major=True,split_kv_accum=True,block_j=cfg['j'],merge_loop_stages=cfg['merge_stages'],
                    route_loop_stages=1,merge_disable_licm=False,output=out,timings=t)
                if profile:
                    for key in times:times[key]+=t[key]
            return times if profile else out
        ref=full().clone()
        configs=[dict(cfg)]
        configs += [dict(cfg,heads=h) for h in (4,8,10,20)]
        configs += [dict(cfg,j=j) for j in (8,16)]
        configs += [dict(cfg,merge_warps=8),dict(cfg,route_warps=8),dict(cfg,route_block=128),dict(cfg,merge_stages=2)]
        for config in configs:
            cfg=config
            print('CONFIG',cfg,flush=True)
            try:
                y=full();d=y.float()-ref.float()
                row={'config':dict(cfg),'exact':torch.equal(y,ref),'relative_l2':float((d.square().sum()/ref.float().square().sum()).sqrt())}
                del d
                row['stages']=full(profile=True)
                full(True);torch.cuda.synchronize();samples=[]
                for _ in range(4):
                    assert not result['monitor_errors'],result['monitor_errors']
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record();full(True);b.record();b.synchronize();samples.append(a.elapsed_time(b))
                row.update(mean_ms=statistics.fmean(samples),samples_ms=samples)
            except Exception as e:
                row={'config':dict(cfg),'error':repr(e)}
            result['sweep'].append(row);save();print('RESULT',row,flush=True)
        result['best']=min([r for r in result['sweep'] if r.get('exact')],key=lambda r:r['mean_ms'])
        result['valid']=not result['monitor_errors']
        print('BEST',result['best'],flush=True)
        assert result['valid'],result['monitor_errors']
        # Validate the chosen configuration on all three real captures. Keep
        # the original selector and original launch settings as the control.
        result['validation']=[]
        best_cfg=dict(result['best']['config']);base_cfg=configs[0]
        del ref
        for layer in ('00','19','39'):
            capture=Path(args.capture).with_name(f'real_full40_step0_layer{layer}.pt')
            if not capture.exists():
                raise FileNotFoundError(f'Missing cross-layer capture: {capture}')
            del q,k,v,out
            torch.cuda.empty_cache()
            payload=torch.load(capture,map_location='cpu',weights_only=True)
            q,k,v=[payload[n].reshape(1,21,45,80,40,128).cuda() for n in ('q','k','v')];del payload
            out=torch.empty_like(q);seeds=op(q,k)
            gate={'layer':layer,'seed_mismatch':int((seeds!=reference_seeds()).sum())}
            assert gate['seed_mismatch']==0,gate
            cfg=base_cfg;b=full(reference_selector=True).clone()
            cfg=best_cfg;a=full(True)
            gate['bitwise_equal']=torch.equal(a,b)
            assert gate['bitwise_equal'],gate
            del b
            samples={'reference':[],'candidate':[],'sdpa':[]}
            for pair in range(8):
                for name in (('reference','candidate') if pair%2==0 else ('candidate','reference')):
                    assert not result['monitor_errors'],result['monitor_errors']
                    cfg=base_cfg if name=='reference' else best_cfg
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();full(True,reference_selector=name=='reference');end.record();end.synchronize()
                    samples[name].append(start.elapsed_time(end))
            from torch.nn.attention import sdpa_kernel,SDPBackend
            qf,kf,vf=[x.reshape(1,75600,40,128).transpose(1,2) for x in (q,k,v)]
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                for rep in range(10):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();torch.nn.functional.scaled_dot_product_attention(qf,kf,vf);end.record();end.synchronize()
                    if rep>=2:samples['sdpa'].append(start.elapsed_time(end))
            del qf,kf,vf,a
            gate['timings']={name:{'mean_ms':statistics.fmean(s),'samples_ms':s} for name,s in samples.items()}
            gate['speedup_vs_sdpa']=gate['timings']['sdpa']['mean_ms']/gate['timings']['candidate']['mean_ms']
            result['validation'].append(gate);save();print('VALIDATED_LAYER',gate,flush=True)
        result['valid']=not result['monitor_errors']
        result['completed']=True
    finally:
        stop.set();th.join(timeout=5);save()

if __name__=='__main__':main()
