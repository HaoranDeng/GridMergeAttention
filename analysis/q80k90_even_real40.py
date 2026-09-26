"""All-eight-layer full40-head real-QKV benchmark: Q80K90 even capacities vs the v4 Q80K90 kernels.

Same protocol as kernels/experimental/q80k90_c4_c12/benchmark_family_real40.py and
analysis/q90k80_even_real40.py: fresh frozen StepDistill dense trajectories
(3 prompts x 4 steps), layers 0, 5, ..., 35, all 40 heads, BF16 [1, 75600, 40, 128].
Per case and capacity the new operator and the production v4 operator
(recommended.make_q80k90) are warmed once on that case's QKV (their outputs
compared with each other and with dense), then the new operator, v4 and dense
Flash SDPA (and with --with-q90 the Q90K80 even operator of the same capacity)
are timed three times each, the order rotating by one and reversing on every
other repeat. Only one capacity's workspaces are live at a time.

python q80k90_even_real40.py --repo ~/workspace/GridMergeAttention --kernels ~/workspace/gma_q80k90_even_opt \
    --root /local2/.../q80k90_even_real40 --capacities 4 6 8 10
"""
import argparse,ctypes,gc,hashlib,importlib.util,json,math,os,statistics,subprocess,sys
from pathlib import Path
from unittest.mock import patch
import torch

LAYERS=[0,5,10,15,20,25,30,35]
def save(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def ci95(xs):return 1.985*statistics.stdev(xs)/math.sqrt(len(xs)) if len(xs)>2 else float('nan')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True,help='model code (generate.py, configs, prompts)')
    p.add_argument('--kernels',type=Path,required=True,help='directory with analysis/q80k90_even_* and kernels/experimental/q80k90_c4_c12')
    p.add_argument('--checkpoint',type=Path,default=Path.home()/'wan_models/Wan2.1-T2V-14B-StepDistill-CfgDistill/distill_model.pt')
    p.add_argument('--root',type=Path,required=True);p.add_argument('--capacities',type=int,nargs='+',required=True)
    p.add_argument('--prompts',type=int,default=3);p.add_argument('--hc',type=int,default=8)
    p.add_argument('--with-q90',action='store_true',help='also time the Q90K80 even-capacity operator of the same capacity (analysis/q90k80_even_kernel.py)')
    a=p.parse_args()
    assert 'TORCH_EXTENSIONS_DIR' in os.environ,'set a private TORCH_EXTENSIONS_DIR'
    os.environ['WAN_MODEL_ROOT']=str(a.checkpoint.parent.parent);os.environ['WAN_SDPA_BACKEND']='flash'
    os.environ.setdefault('MAX_JOBS','8')
    ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libstdc++.so.6',mode=ctypes.RTLD_GLOBAL)
    sys.path.insert(0,str(a.repo))
    a.root.mkdir(parents=True,exist_ok=True)
    even=module('q80k90_even_kernel_real40',a.kernels/'analysis/q80k90_even_kernel.py')
    v4dir=a.kernels/'kernels/experimental/q80k90_c4_c12'
    rec=module('q80k90_recommended_real40',v4dir/'recommended.py')
    q90=module('q90k80_even_kernel_real40',a.kernels/'analysis/q90k80_even_kernel.py') if a.with_q90 else None
    print('real40: building kernels',flush=True)
    for cap in a.capacities:
        even.native_merge(21,40,a.hc,cap);even.native_route(21,40,a.hc,cap);even.native_select(21,40,cap)
        v4=rec.make_q80k90(cap);v4.native(21,40,min(v4.hc,40));del v4
        if q90:q90.native_merge(21,40,a.hc,cap);q90.native_route(21,40,a.hc,cap);q90.native_select(21,40,cap)
    if q90:q90.native_layout(21,40)
    even.native_layout(21,40)
    print('real40: loading Python model dependencies',flush=True)
    from omegaconf import OmegaConf
    from generate import build_pipeline,load_generator_checkpoint,set_seed
    from utils.prompt_contract import load_nonempty_prompts,validate_canonical_prompt_contract
    from analysis.capture_rf0_layer33_diagnostic import validate_config,ACTUAL_TIMESTEPS
    import wan.modules.model as wm
    torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    prompts_path=a.repo/'prompts/MovieGenVideoBench_extended.txt';prompts=load_nonempty_prompts(str(prompts_path));validate_canonical_prompt_contract(str(prompts_path),prompts)
    subprocess.run([sys.executable,str(a.repo/'analysis/verify_prompt_contract.py'),'--prompt_path',str(prompts_path),'--num_videos',str(a.prompts),'--seed','0','--num_samples','1','--manifest_out',str(a.root/'prompt_manifest.json')],check=True,capture_output=True)
    conf=OmegaConf.load(a.repo/'configs/dense_wan_14b_720p_fewstep.yaml');validate_config(OmegaConf.to_container(conf,resolve=True))
    print('real40: constructing model',flush=True)
    pipeline=build_pipeline(conf,torch.device('cuda'))
    print('real40: loading checkpoint',flush=True)
    assert load_generator_checkpoint(pipeline,str(a.checkpoint),True)=='raw_state_dict'
    pipeline.to(dtype=torch.bfloat16);pipeline.text_encoder.to('cuda');pipeline.generator.to('cuda');pipeline.eval()
    assert len(pipeline.generator.model.blocks)==40 and tuple(pipeline.denoising_step_list.tolist())==ACTUAL_TIMESTEPS
    sources={'analysis/'+n:sha(a.kernels/'analysis'/n) for n in ('q80k90_even_kernel.py','q80k90_even_layout.cu','q80k90_even_select.cu','q80k90_even_merge.cu','q80k90_even_route.cu')}
    sources.update({'v4/'+n:sha(v4dir/n) for n in ('recommended.py','validated_h200_configs.json','q80k90_kernel.py','q80k90_merge.cu')})
    if q90:sources.update({'analysis/'+n:sha(a.kernels/'analysis'/n) for n in ('q90k80_even_kernel.py','q90k80_even_layout.cu','q90k80_even_select.cu','q90k80_even_merge.cu','q90k80_even_route.cu')})
    result=dict(complete=False,layers=LAYERS,prompts=a.prompts,steps=4,heads=list(range(40)),shape=[1,75600,40,128],dtype='bfloat16',
                gpu=torch.cuda.get_device_name(),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),host=os.uname().nodename,head_chunk=a.hc,
                source_sha256=sources,benchmark_sha256=sha(__file__),rows=[])
    state=dict(layer=None,prompt=None,step=None,call=0)
    forward0=wm.WanSelfAttention.forward;attention0=wm.flash_attention
    def forward(m,*args,**kw):
        assert state['layer'] is None and not m.enable_token_merging and m.attention_backend=='sdpa'
        state['layer']=m.layer_idx
        try:return forward0(m,*args,**kw)
        finally:state['layer']=None
    def measure(fn):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();out=fn();end.record();end.synchronize();return start.elapsed_time(end),out
    def rel(x,y):return ((x.float()-y.float()).norm()/y.float().norm()).item()
    def attention(q,k,v,*args,**kw):
        dense=attention0(q,k,v,*args,**kw)
        layer=state['layer']
        if layer is None:return dense
        assert layer==state['call']%40
        state['call']+=1
        if layer not in LAYERS:return dense
        assert q.shape==(1,75600,40,128)
        for cap in a.capacities:
            new=even.Q80K90Even(cap,head_chunk=a.hc);old=rec.make_q80k90(cap)
            o_new=new(q,k,v);o_old=old(q,k,v);torch.cuda.synchronize()
            assert torch.isfinite(o_new).all() and torch.isfinite(o_old).all()
            info=dict(rel_l2_new_vs_v4=rel(o_new,o_old),rel_l2_new_to_dense=rel(o_new,dense),rel_l2_v4_to_dense=rel(o_old,dense))
            del o_new,o_old
            arms=[('new',lambda:new(q,k,v)),('v4',lambda:old(q,k,v)),('dense',lambda:attention0(q,k,v,*args,**kw))]
            if q90:
                other=q90.Q90K80Even(cap,head_chunk=a.hc);o=other(q,k,v);torch.cuda.synchronize()
                info.update(rel_l2_q90_to_dense=rel(o,dense));del o
                arms.insert(2,('q90',lambda:other(q,k,v)))
            times={n:[] for n,_ in arms}
            for repeat in range(3):
                order=arms[repeat:]+arms[:repeat]
                if repeat%2:order=order[::-1]
                for name,fn in order:
                    ms,out=measure(fn);times[name].append(ms);del out
            row=dict(capacity=cap,prompt=state['prompt'],step=state['step'],layer=layer,ms=times,mean_ms={n:statistics.mean(t) for n,t in times.items()},**info)
            result['rows'].append(row);save(a.root/'real40.json',result);print(json.dumps(row),flush=True)
            del new,old,arms
            if q90:del other
            gc.collect();torch.cuda.empty_cache()
        return dense
    with torch.inference_mode(),patch.object(wm.WanSelfAttention,'forward',forward),patch.object(wm,'flash_attention',attention):
        for prompt in range(a.prompts):
            state.update(prompt=prompt,call=0);set_seed(prompt)
            noise=torch.randn(list(conf.image_or_video_shape),device='cuda',dtype=torch.bfloat16);conditional=pipeline.text_encoder(text_prompts=[prompts[prompt]]);current=noise
            for step,timestep in enumerate(pipeline.denoising_step_list.to('cuda')):
                state['step']=step
                _,predicted=pipeline.generator(noisy_image_or_video=current,conditional_dict=conditional,timestep=timestep.expand(noise.shape[:2]))
                assert state['call']==(step+1)*40
                if step<3:
                    nxt=pipeline.denoising_step_list[step+1].to('cuda').expand(noise.shape[:2])
                    current=pipeline.scheduler.add_noise(predicted.flatten(0,1),torch.randn_like(predicted.flatten(0,1)),nxt.flatten(0,1)).unflatten(0,noise.shape[:2])
            del noise,current,predicted,conditional
    result['capacities']={}
    for cap in a.capacities:
        rows=[r for r in result['rows'] if r['capacity']==cap]
        names=('new','v4','dense')+(('q90',) if q90 else ())
        mean={n:statistics.mean(t for r in rows for t in r['ms'][n]) for n in names}
        diff=[r['mean_ms']['new']-r['mean_ms']['v4'] for r in rows]
        result['capacities'][str(cap)]=dict(cases=len(rows),new_mean_ms=mean['new'],v4_mean_ms=mean['v4'],dense_mean_ms=mean['dense'],
             new_speedup_vs_dense=mean['dense']/mean['new'],v4_speedup_vs_dense=mean['dense']/mean['v4'],
             new_minus_v4_ms=statistics.mean(diff),new_minus_v4_ci95_ms=ci95(diff),new_faster_cases=sum(x<0 for x in diff),
             new_median_case_ms=statistics.median(r['mean_ms']['new'] for r in rows),
             mean_rel_l2_new_vs_v4=statistics.mean(r['rel_l2_new_vs_v4'] for r in rows),max_rel_l2_new_vs_v4=max(r['rel_l2_new_vs_v4'] for r in rows),
             mean_rel_l2_new_to_dense=statistics.mean(r['rel_l2_new_to_dense'] for r in rows),mean_rel_l2_v4_to_dense=statistics.mean(r['rel_l2_v4_to_dense'] for r in rows),
             new_faster_than_dense_all_cases=all(r['mean_ms']['new']<r['mean_ms']['dense'] for r in rows))
        if q90:
            d90=[r['mean_ms']['new']-r['mean_ms']['q90'] for r in rows]
            result['capacities'][str(cap)].update(q90_mean_ms=mean['q90'],q90_speedup_vs_dense=mean['dense']/mean['q90'],new_minus_q90_ms=statistics.mean(d90),
                 new_minus_q90_ci95_ms=ci95(d90),new_faster_than_q90_cases=sum(x<0 for x in d90),mean_rel_l2_q90_to_dense=statistics.mean(r['rel_l2_q90_to_dense'] for r in rows))
    result['complete']=all(v['cases']==a.prompts*4*len(LAYERS) for v in result['capacities'].values())
    save(a.root/'real40.json',result)
    print(json.dumps({k:result[k] for k in ('complete','capacities')}),flush=True)
if __name__=='__main__':main()
