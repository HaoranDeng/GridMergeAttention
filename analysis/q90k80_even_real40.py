"""All-eight-layer full40-head real-QKV benchmark for the Q90K80 even capacities.

Mirrors kernels/experimental/q80k90_c4_c12/benchmark_family_real40.py: fresh
frozen StepDistill dense trajectories (3 prompts x 4 steps), layers 0, 5, ...,
35, 40 real heads; per case each capacity's operator and dense Flash SDPA are
timed three times in alternating order on the same QKV. One capacity's
workspace is live at a time.

python q90k80_even_real40.py --repo ~/workspace/GridMergeAttention --kernels ~/workspace/gma_q90k80_even_opt \
    --root /local2/.../q90k80_even_real40 --capacities 4 6 8 10 [--hc 8]
"""
import argparse,gc,hashlib,json,os,statistics,subprocess,sys,time,ctypes
from pathlib import Path
from unittest.mock import patch
import torch

LAYERS=[0,5,10,15,20,25,30,35]
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--kernels',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=Path.home()/'wan_models/Wan2.1-T2V-14B-StepDistill-CfgDistill/distill_model.pt')
    p.add_argument('--root',type=Path,required=True);p.add_argument('--capacities',type=int,nargs='+',required=True)
    p.add_argument('--prompts',type=int,default=3);p.add_argument('--hc',type=int,default=8)
    a=p.parse_args()
    os.environ['WAN_MODEL_ROOT']=str(a.checkpoint.parent.parent);os.environ['WAN_SDPA_BACKEND']='flash'
    os.environ.setdefault('TORCH_EXTENSIONS_DIR',str(a.kernels/('.torch_ext_'+os.uname().nodename)));os.environ.setdefault('MAX_JOBS','8')
    ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libstdc++.so.6',mode=ctypes.RTLD_GLOBAL)
    sys.path.insert(0,str(a.kernels));sys.path.insert(0,str(a.repo))
    from analysis.q90k80_even_kernel import Q90K80Even,native_merge,native_route,native_select,native_layout
    print('real40: building kernels',flush=True)
    for cap in a.capacities:
        native_merge(21,40,a.hc,cap);native_route(21,40,a.hc,cap);native_select(21,40,cap)
    native_layout(21,40)
    print('real40: loading Python model dependencies',flush=True)
    from omegaconf import OmegaConf
    from generate import build_pipeline,load_generator_checkpoint,set_seed
    from utils.prompt_contract import load_nonempty_prompts,validate_canonical_prompt_contract
    from analysis.capture_rf0_layer33_diagnostic import validate_config,ACTUAL_TIMESTEPS
    import wan.modules.model as wm
    torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    a.root.mkdir(exist_ok=True,parents=True)
    prompts_path=a.repo/'prompts/MovieGenVideoBench_extended.txt';prompts=load_nonempty_prompts(str(prompts_path));validate_canonical_prompt_contract(str(prompts_path),prompts)
    subprocess.run([sys.executable,str(a.repo/'analysis/verify_prompt_contract.py'),'--prompt_path',str(prompts_path),'--num_videos',str(a.prompts),'--seed','0','--num_samples','1','--manifest_out',str(a.root/'prompt_manifest.json')],check=True,capture_output=True)
    conf=OmegaConf.load(a.repo/'configs/dense_wan_14b_720p_fewstep.yaml');validate_config(OmegaConf.to_container(conf,resolve=True))
    print('real40: constructing model',flush=True)
    pipeline=build_pipeline(conf,torch.device('cuda'))
    print('real40: loading checkpoint',flush=True)
    assert load_generator_checkpoint(pipeline,str(a.checkpoint),True)=='raw_state_dict'
    pipeline.to(dtype=torch.bfloat16);pipeline.text_encoder.to('cuda');pipeline.generator.to('cuda');pipeline.eval()
    assert len(pipeline.generator.model.blocks)==40 and tuple(pipeline.denoising_step_list.tolist())==ACTUAL_TIMESTEPS
    kdir=a.kernels/'analysis'
    sources={n:hashlib.sha256((kdir/n).read_bytes()).hexdigest() for n in ('q90k80_even_kernel.py','q90k80_even_merge.cu','q90k80_even_route.cu','q90k80_even_select.cu','q90k80_even_layout.cu')}
    state=dict(layer=None,prompt=None,step=None,call=0)
    result=dict(complete=False,passed=False,layers=LAYERS,prompts=a.prompts,steps=4,heads=list(range(40)),shape=[1,75600,40,128],dtype='bfloat16',
                gpu=torch.cuda.get_device_name(),head_chunk=a.hc,rows=[],source_sha256=sources,benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    forward0=wm.WanSelfAttention.forward;attention0=wm.flash_attention
    def forward(m,*args,**kw):
        assert state['layer'] is None and not m.enable_token_merging and m.attention_backend=='sdpa'
        state['layer']=m.layer_idx
        try:return forward0(m,*args,**kw)
        finally:state['layer']=None
    def measure(fn):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();out=fn();end.record();end.synchronize();return start.elapsed_time(end),out
    def attention(q,k,v,*args,**kw):
        dense=attention0(q,k,v,*args,**kw)
        layer=state['layer']
        if layer is None:return dense
        assert layer==state['call']%40
        state['call']+=1
        if layer not in LAYERS:return dense
        assert q.shape==(1,75600,40,128)
        for cap in a.capacities:
            core=Q90K80Even(cap,merge_backend='cuda',head_chunk=a.hc)
            out=core(q,k,v);torch.cuda.synchronize()
            rel=((out.float()-dense.float()).norm()/dense.float().norm()).item();del out
            times={'method':[],'dense':[]}
            for repeat in range(3):
                methods=[('method',lambda:core(q,k,v)),('dense',lambda:attention0(q,k,v,*args,**kw))]
                if repeat%2:methods.reverse()
                for name,fn in methods:
                    ms,out=measure(fn);times[name].append(ms);assert torch.isfinite(out).all();del out
            row=dict(capacity=cap,prompt=state['prompt'],step=state['step'],layer=layer,method_ms=times['method'],dense_ms=times['dense'],
                     mean_attention_ms=statistics.mean(times['method']),dense_mean_attention_ms=statistics.mean(times['dense']),rel_l2_to_dense=rel)
            row['speedup_vs_sdpa']=row['dense_mean_attention_ms']/row['mean_attention_ms']
            result['rows'].append(row);save(a.root/'real40.json',result);print(json.dumps(row),flush=True)
            del core;gc.collect();torch.cuda.empty_cache()
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
        method=statistics.mean(t for r in rows for t in r['method_ms'])
        dense=statistics.mean(t for r in rows for t in r['dense_ms'])
        result['capacities'][str(cap)]=dict(cases=len(rows),mean_attention_ms=method,dense_mean_attention_ms=dense,speedup_vs_sdpa=dense/method,
             mean_rel_l2_to_dense=statistics.mean(r['rel_l2_to_dense'] for r in rows),
             faster_than_dense_all_cases=all(r['mean_attention_ms']<r['dense_mean_attention_ms'] for r in rows))
    result['complete']=all(v['cases']==a.prompts*4*8 for v in result['capacities'].values())
    result['passed']=result['complete'] and all(v['faster_than_dense_all_cases'] for v in result['capacities'].values())
    save(a.root/'real40.json',result)
    print(json.dumps({k:result[k] for k in ('complete','passed','capacities')}),flush=True)
if __name__=='__main__':main()
