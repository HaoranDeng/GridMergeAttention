"""Why is Q80K90Even slower inside I2V generation than in isolation? Same process, same data.

Runs the frozen I2V snapshot's generate.py (as q80k90_even_i2v_timing.py) for the first
--steps denoising steps of prompt 0, with the adapter's kernel factory replaced by a
subclass of Q80K90Even that records CUDA events between its stages (layout, select, and
each head chunk's merge and route) on every call. The QKV of a few calls is cloned on
the GPU. Generation is then aborted and the same kernel instance is re-timed on the
cloned QKV in the same process: (a) back-to-back, (b) after flushing L2, (c) with a
power load (bf16 GEMMs) before every call. GPU clock and power are sampled through
NVML the whole time. Writes diag.json.

python q80k90_even_i2v_diag.py --plan .../experiment_plan.json --kernels ~/workspace/gma_q80k90_even_v2 --out DIR --capacity 8
"""
import argparse,hashlib,importlib.util,json,os,statistics,sys,threading,time
from pathlib import Path

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class Sampler(threading.Thread):
    """NVML SM clock / power / throttle samples every 20 ms, tagged with a phase label."""
    def __init__(self,index):
        super().__init__(daemon=True)
        import pynvml;pynvml.nvmlInit();self.nv=pynvml;self.h=pynvml.nvmlDeviceGetHandleByIndex(index)
        self.phase='idle';self.rows=[];self.stop=False
    def run(self):
        nv=self.nv
        while not self.stop:
            try:
                self.rows.append((self.phase,nv.nvmlDeviceGetClockInfo(self.h,nv.NVML_CLOCK_SM),nv.nvmlDeviceGetPowerUsage(self.h)/1000.,
                                  nv.nvmlDeviceGetCurrentClocksThrottleReasons(self.h)))
            except Exception:pass
            time.sleep(.02)
    def summary(self):
        out={}
        for ph in sorted({r[0] for r in self.rows}):
            rs=[r for r in self.rows if r[0]==ph]
            out[ph]=dict(samples=len(rs),sm_mhz=statistics.mean(r[1] for r in rs),sm_mhz_min=min(r[1] for r in rs),
                         power_w=statistics.mean(r[2] for r in rs),power_capped_frac=sum(1 for r in rs if r[3]&4)/len(rs))
        return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--kernels',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--capacity',type=int,default=8)
    ap.add_argument('--steps',type=int,default=10);ap.add_argument('--nvml-index',type=int,required=True)
    a=ap.parse_args()
    assert 'TORCH_EXTENSIONS_DIR' in os.environ
    plan=json.loads(a.plan.read_text());snap=Path(plan['snapshot']);a.out.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_grad_enabled(False);torch.empty(1,device='cuda');torch.cuda.synchronize()
    even=module('q80k90_even_kernel',a.kernels/'analysis/q80k90_even_kernel.py')
    cap=a.capacity
    sampler=Sampler(a.nvml_index);sampler.start()
    class Timed(even.Q80K90Even):
        record=True
        def __call__(self,q,k,v,return_debug=False):
            t,h=self.prepare(q)
            (q,qs),(k,ks),(v,vs)=even.input_strides(q),even.input_strides(k),even.input_strides(v)
            ev=[torch.cuda.Event(enable_timing=True) for _ in range(3+2*(h//self.hc))]
            ev[0].record();qm,qms=self.merge_query(q,qs,t,h)  # the layout stage (empty with MERGE_PSI_K)
            ev[1].record();even.native_select(t,h,self.cap,self.select_flags).run(q,k,self.table,self.seeds,*qs,*ks)
            ev[2].record()
            out=torch.empty((1,t*3600,h,128),device=q.device,dtype=q.dtype);i=3
            for hs in range(0,h,self.hc):
                ah=min(self.hc,h-hs)
                self.merge_module(t,h,ah).run(qm,k,v,self.seeds,self.mk,self.mv,self.bias,hs,ah,self.tiles,*qms,*ks,*vs);ev[i].record()
                even.native_route(t,h,ah,self.cap,self.route_flags).run(q,self.mk,self.mv,self.bias,out,hs,ah,*qs);ev[i+1].record();i+=2
            if self.record:self.events.append(ev)
            return out
    def stages(ev):
        el=[ev[j].elapsed_time(ev[j+1]) for j in range(len(ev)-1)]
        # el: layout, select, then (merge, route) per head chunk
        return dict(layout=el[0],select=el[1],merge=sum(el[2::2]),route=sum(el[3::2]),total=ev[0].elapsed_time(ev[-1]))
    core=Timed(cap);core.events=[]
    # build, warm
    g=torch.Generator(device='cuda').manual_seed(0)
    x=[torch.randn(1,75600,40,128,device='cuda',dtype=torch.bfloat16,generator=g) for _ in range(3)]
    for _ in range(2):core(*x)
    torch.cuda.synchronize();core.events=[];del x
    sys.path[:0]=[str(snap)];os.chdir(snap)
    from attention import q80k90_wan as adapter
    adapter._make=lambda capacity:core
    original=adapter.q80k90_attention
    keep=[];KEEP={0:'step0_layer0',20:'step0_layer20',5*80+20:'step5_layer20',9*80+39:'step9_layer39',9*80+40+20:'step9_uncond_layer20'}
    wrap=[];count=dict(n=0)
    class Stop(Exception):pass
    def attention(q,k,v,grid_size,capacity=8):
        assert capacity==cap
        n=count['n']
        if n>=a.steps*80:raise Stop()
        b=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
        b.record();out=original(q,k,v,grid_size,capacity);e.record()
        wrap.append((b,e))
        if n in KEEP:keep.append((KEEP[n],q.detach().clone(),k.detach().clone(),v.detach().clone(),len(core.events)-1))
        count['n']=n+1
        return out
    adapter.q80k90_attention=attention
    gen=module('frozen_i2v_generate',snap/'generate.py')
    gen.save_video_atomic=lambda *x,**y:None
    sys.argv=['generate.py','--config',plan['configs'][str(cap)],'--checkpoint','pretrained','--prompt_path',plan['prompt_path'],
              '--image_list',plan['image_list'],'--image_dir',plan['image_dir'],'--output_dir',str(a.out/'unused'),
              '--num_videos','1','--num_samples','1','--seed','0']
    sampler.phase='pipeline'
    try:gen.main()
    except Stop:pass
    torch.cuda.synchronize();sampler.phase='idle'
    assert count['n']>=a.steps*80,count
    pipe_rows=[stages(ev) for ev in core.events[:a.steps*80]]
    wrapper_ms=[b.elapsed_time(e) for b,e in wrap[:a.steps*80]]
    # isolated re-timing on the cloned QKV, same kernel instance, same process
    def run_iso(q,k,v,label,prep=None,reps=6):
        rows=[]
        for r in range(reps):
            if prep:prep()
            core.events=[];core(q,k,v);torch.cuda.synchronize();rows.append(stages(core.events[0]))
        return rows
    flush=torch.empty(256*2**20,dtype=torch.uint8,device='cuda')
    A=torch.randn(8192,8192,device='cuda',dtype=torch.bfloat16);B=torch.randn(8192,8192,device='cuda',dtype=torch.bfloat16)
    def l2_flush():flush.zero_()
    def power_load():
        for _ in range(30):A@B
    cases=[]
    for label,q,k,v,idx in keep:
        pipe=pipe_rows[idx]
        sampler.phase='isolated';iso=run_iso(q,k,v,label)
        sampler.phase='isolated_l2_flush';l2=run_iso(q,k,v,label,l2_flush)
        sampler.phase='isolated_power_load';pw=run_iso(q,k,v,label,power_load)
        sampler.phase='idle'
        med=lambda rows,key:statistics.median(r[key] for r in rows[1:])
        case=dict(label=label,in_pipeline=pipe,in_pipeline_wrapper_ms=wrapper_ms[idx],
                  isolated={k2:med(iso,k2) for k2 in pipe},isolated_l2_flush={k2:med(l2,k2) for k2 in pipe},isolated_power_load={k2:med(pw,k2) for k2 in pipe})
        cases.append(case);print(json.dumps(case),flush=True)
    sampler.stop=True;time.sleep(.1)
    pm=lambda key:statistics.mean(r[key] for r in pipe_rows)
    result=dict(capacity=cap,steps=a.steps,calls=len(pipe_rows),
                in_pipeline_mean={k2:pm(k2) for k2 in pipe_rows[0]},in_pipeline_wrapper_mean_ms=statistics.mean(wrapper_ms),
                in_pipeline_by_step=[{k2:statistics.mean(r[k2] for r in pipe_rows[s*80:(s+1)*80]) for k2 in ('total','merge','route','select')} for s in range(a.steps)],
                cases=cases,nvml=sampler.summary(),host=os.uname().nodename,gpu=torch.cuda.get_device_name())
    (a.out/'diag.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k2:result[k2] for k2 in ('in_pipeline_mean','in_pipeline_wrapper_mean_ms','nvml')},indent=1))

if __name__=='__main__':main()
