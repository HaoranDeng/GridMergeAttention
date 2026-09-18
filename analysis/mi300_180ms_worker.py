"""One exclusive GPU worker for bounded, file-fed C4 launch experiments."""
import argparse,json,os,re,sys,threading,time,subprocess,statistics,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

class Launch:
    def __init__(self,kernel,opts=None,transform=None):
        self.kernel=kernel;self.opts=opts or {};self.transform=transform;self.compiled=None
    def __getitem__(self,grid):
        def run(*args,**kw):
            if self.transform:args,grid2,kw=self.transform(args,grid,kw)
            else:grid2=grid
            kw.update(self.opts)
            self.compiled=self.kernel[grid2](*args,**kw)
            return self.compiled
        return run

def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',required=True);p.add_argument('--output',required=True)
    p.add_argument('--queue',required=True);p.add_argument('--idle-seconds',type=int,default=900)
    args=p.parse_args();import torch
    from native_hip.selector import HipFisherSelector
    from attention import hard_c4_wan21_14b_720p_triton as hard
    from attention.singleton_c4_wan21_14b_720p_triton import _hard_c4_720p_temporal_shared_global_mass_attention_triton_blocks as run
    assert torch.cuda.device_count()==1
    torch.set_grad_enabled(False);torch.set_num_threads(4)
    root=Path(args.output).parent;root.mkdir(parents=True,exist_ok=True)
    result={'job':os.environ.get('SLURM_JOB_ID'),'trials':[],'monitor':[],'monitor_errors':[]}
    stop=threading.Event()
    def monitor():
        while not stop.is_set():
            r=subprocess.run(['rocm-smi','--showpids','--showuse'],capture_output=True,text=True,timeout=20)
            pids=[int(x) for x in re.findall(r'^\s*(\d+)\s+\S+',r.stdout,re.M)]
            result['monitor'].append({'time':time.time(),'text':r.stdout})
            if r.returncode or any(x!=os.getpid() for x in pids):result['monitor_errors'].append({'rc':r.returncode,'pids':pids})
            stop.wait(3)
    th=threading.Thread(target=monitor,daemon=True);th.start()
    def save():Path(args.output).write_text(json.dumps(result,indent=2))
    try:
        data=torch.load(args.capture,map_location='cpu',weights_only=True)
        q,k,v=[data[x].reshape(1,21,45,80,40,128).cuda() for x in ('q','k','v')];del data
        op=HipFisherSelector();seeds=op(q,k);out=torch.empty_like(q)
        original_merge=hard._hard_c4_720p_temporal_shared_merge_kernel
        original_route=hard._hard_c4_720p_route_kernel
        cfg={'j':4,'heads':20}
        packed=None
        def full(select=True,profile=False):
            if cfg.get('pack_q'):
                packed.copy_(q.permute(0,4,1,2,3,5))
            s=op(q,k) if select else seeds
            times={'merge_ms':0.,'route_ms':0.}
            for h in range(0,40,cfg.get('heads',20)):
                t={} if profile else None
                run(q,k,v,s,top_r=1,head_start=h,head_count=cfg.get('heads',20),query_column_selector=True,
                    seed_rows_native_layout=True,query_column_major=True,split_kv_accum=True,block_j=cfg.get('j',4),
                    merge_loop_stages=cfg.get('stages',1),route_loop_stages=cfg.get('route_stages',1),merge_disable_licm=cfg.get('licm',False),output=out,timings=t)
                if profile:
                    for n in times:times[n]+=t[n]
            return times if profile else out
        ref=full().clone();torch.cuda.synchronize()
        print('READY',flush=True);seen=set();last=time.monotonic()
        while time.monotonic()-last<args.idle_seconds:
            queue=json.loads(Path(args.queue).read_text())
            todo=[c for c in queue if c['id'] not in seen]
            if not todo:
                if (Path(args.queue).parent/'STOP').exists():break
                time.sleep(2);continue
            for cfg in todo:
                last=time.monotonic();seen.add(cfg['id']);print('TRIAL',cfg,flush=True)
                row={'config':cfg}
                try:
                    assert not result['monitor_errors'],result['monitor_errors']
                    hard.HARD_720_MERGE_NUM_WARPS=cfg.get('warps',4)
                    hard.HARD_720_ROUTE_NUM_WARPS=cfg.get('route_warps',4)
                    hard.HARD_720_ROUTE_BLOCK_TOKENS=cfg.get('route_block',64)
                    kernel=original_merge;transform=None
                    if cfg.get('custom'):
                        from native_hip.merge_tiled import merge_kernel
                        kernel=merge_kernel
                        if cfg.get('pack_q') and packed is None:packed=torch.empty((1,40,21,45,80,128),device=q.device,dtype=q.dtype)
                        def transform(a,g,kw):
                            kw.update(PACK_Q=cfg.get('pack_q',False),COL_SPLITS=cfg.get('col_splits',1),FRAME_SPLITS=cfg.get('frame_splits',1))
                            if cfg.get('pack_q'):a=(packed,)+a[1:]
                            return a,(g[0],g[1],cfg.get('col_splits',1)*cfg.get('frame_splits',1)),kw
                    mp=Launch(kernel,cfg.get('merge_opts'),transform);rp=Launch(original_route,cfg.get('route_opts'))
                    hard._hard_c4_720p_temporal_shared_merge_kernel=mp;hard._hard_c4_720p_route_kernel=rp
                    y=full();d=y.float()-ref.float()
                    row.update(exact=torch.equal(y,ref),relative_l2=float((d.square().sum()/ref.float().square().sum()).sqrt()),max_abs=float(d.abs().max()),finite=bool(torch.isfinite(y).all()))
                    del d
                    row['stages']=full(profile=True)
                    full();torch.cuda.synchronize();samples=[]
                    for _ in range(4):
                        assert not result['monitor_errors']
                        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        a.record();full();b.record();b.synchronize();samples.append(a.elapsed_time(b))
                    row.update(mean_ms=statistics.fmean(samples),samples_ms=samples)
                    for name,proxy in [('merge',mp),('route',rp)]:
                        ck=proxy.compiled
                        row[name+'_resources']={n:getattr(ck,n,None) for n in ('n_regs','n_spills')}
                        row[name+'_resources']['metadata']=str(ck.metadata)
                    # Preserve compiled artifacts for register/spill analysis.
                    if cfg.get('dump'):
                        for name,proxy in [('merge',mp),('route',rp)]:
                            for ext in ('amdgcn','ttgir'):
                                (root/f"{cfg['id']}.{name}.{ext}").write_text(proxy.compiled.asm.get(ext,''))
                except Exception:
                    row['error']=traceback.format_exc();print(row['error'],flush=True)
                result['trials'].append(row);save();print('RESULT',row,flush=True)
        result['completed']=True;result['valid']=not result['monitor_errors']
    finally:
        stop.set();th.join(timeout=5);save()

if __name__=='__main__':main()
