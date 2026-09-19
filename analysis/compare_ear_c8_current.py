"""Pure SVG-EAR p=.85 on the identical fixed-eight-layer C8 replay captures."""
import argparse
import ctypes
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics
import sys
from unittest.mock import patch

from analysis.benchmark_c8_current import LAYERS, HEADS, SAMPLES, save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--svg',type=Path,required=True)
    p.add_argument('--captures',type=Path,required=True)
    p.add_argument('--c8-results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert not a.output.exists()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libstdc++.so.6',mode=ctypes.RTLD_GLOBAL)
    sys.path.insert(0,str(a.svg))
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from analysis.benchmark_wan14b_720p_methods import install_flashinfer_workspace_compat,install_rocm_import_stubs
    install_rocm_import_stubs();install_flashinfer_workspace_compat()
    from svg.models.wan import attention as svg_attention
    EAR=svg_attention.WanAttn_EARAttn_Processor
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    c8=json.loads(a.c8_results.read_text())
    assert c8['complete'] and c8['layers']==list(LAYERS) and c8['heads']==list(HEADS)
    pairs={(r['prompt'],r['step'],r['layer'],r['head']):r for r in c8['rows']}
    for name,value in dict(context_length=0,num_frame=21,frame_size=3600,num_layers=40,
        num_q_centroids=300,num_k_centroids=1000,top_p_kmeans=.85,min_kc_ratio=.05,
        kmeans_iter_init=50,kmeans_iter_step=2,first_layers_fp=0,first_times_fp=1001,
        zero_step_kmeans_init=False,logging_file=None).items():
        setattr(EAR,name,value)
    report={'complete':False,'method':'SVG-EAR top_p=.85','layers':list(LAYERS),'heads':list(HEADS),
        'samples':SAMPLES,'dense_first_layers':0,'dense_first_steps':0,'top_p':.85,'min_kc_ratio':.05,
        'num_q_centroids':300,'num_k_centroids':1000,'kmeans_iter_init':50,'kmeans_iter_step':2,
        'seed_policy':'EAR initialization seed 0 per prompt, logical H40 random centroid indices selected at heads 0/10/20/30',
        'state_policy':'one processor per layer, reset each prompt, advance once per denoising step',
        'quality_scope':'same BF16 QKV and saved Flash SDPA targets as c8_current, 384 cases, 27 sampled queries/head',
        'timing_scope':'H4 diagnostic only; three warmed paired complete EAR/SDPA calls per capture with centroid and RNG state restored; not comparable to H40 C8 timing',
        'rows':[],'timings':[],'density':[],
        'source_sha256':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in
            (Path(__file__),Path(svg_attention.__file__),a.svg/'svg/kmeans_utils.py',a.c8_results)},
        'psnr':None,'ssim':None,'lpips':None,'vbench':None}
    indices=torch.tensor([(t*45+h)*80+w for t,h,w in itertools.product(*SAMPLES)],device='cuda')
    selected_heads=torch.tensor(HEADS,device='cuda')
    original_randint=torch.randint
    init_draws=[]
    def logical_randint(low,high,size,**kw):
        assert size in ((4,300),(4,1000)),size
        result=original_randint(low,high,(40,size[1]),**kw)
        init_draws.append(size[1])
        return result.index_select(0,selected_heads)
    original_map=svg_attention.identify_dynamic_map_estimated
    density_maps=[]
    capture_density=False
    def observed(*args,**kwargs):
        assert kwargs['top_p']==.85 and kwargs['min_kc_ratio']==.05 and kwargs['gamma']==1
        mask=original_map(*args,**kwargs)
        if capture_density:density_maps.append((mask,args[3],args[4]))
        return mask
    def snapshot(proc):
        return (proc.centroids_init,None if proc.q_centroids is None else proc.q_centroids.clone(),
                None if proc.k_centroids is None else proc.k_centroids.clone(),torch.cuda.get_rng_state())
    def restore(proc,state):
        proc.centroids_init=state[0]
        proc.q_centroids=None if state[1] is None else state[1].clone()
        proc.k_centroids=None if state[2] is None else state[2].clone()
        torch.cuda.set_rng_state(state[3])
    def timed(fn):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();value=fn();end.record();end.synchronize()
        return value,start.elapsed_time(end)
    with patch.object(torch,'randint',logical_randint),patch.object(svg_attention,'identify_dynamic_map_estimated',observed),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for prompt in range(3):
            torch.manual_seed(0);torch.cuda.manual_seed_all(0)
            processors={layer:EAR(layer_idx=layer) for layer in LAYERS}
            for proc in processors.values():proc.centroids_init=False;proc.q_centroids=None;proc.k_centroids=None
            for step in range(4):
                for layer in LAYERS:
                    path=a.captures/f'p{prompt:03d}_s{step}_l{layer:02d}.pt'
                    blob=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
                    meta=blob['metadata'];assert (meta['prompt'],meta['step'],meta['layer'],meta['heads'])==(prompt,step,layer,list(HEADS))
                    q,k,v=[blob[n].permute(0,2,1,3).cuda().contiguous() for n in ('q','k','v')]
                    target=blob['dense_output'][0,indices.cpu()].cuda().float()
                    proc=processors[layer];pre=snapshot(proc)
                    timestep=torch.tensor([meta['actual_timestep']],device='cuda')
                    def ear():return proc.attention_core_logic(q,k,v,timestep)
                    def dense():return torch.nn.functional.scaled_dot_product_attention(q,k,v)
                    capture_density=True;prediction=ear();capture_density=False
                    post=snapshot(proc)
                    sample=prediction[0,:,indices].permute(1,0,2).float()
                    assert bool(sample.isfinite().all())
                    for slot,head in enumerate(HEADS):
                        ref=target[:,slot];error=sample[:,slot]-ref;pair=pairs[prompt,step,layer,head]
                        assert abs(float(ref.square().sum())/pair['cases']['current']['reference_squared']-1)<1e-5
                        report['rows'].append({'prompt':prompt,'step':step,'layer':layer,'head':head,
                            'relative_l2':float(error.norm()/ref.norm()),'squared_error':float(error.square().sum()),
                            'reference_squared':float(ref.square().sum()),'c8_rf0':pair['cases']['rf0']['relative_l2'],
                            'c8_current':pair['cases']['current']['relative_l2']})
                    assert len(density_maps)==1
                    mask,qs,ks=density_maps.pop();area=qs.double().unsqueeze(-1)*ks.double().unsqueeze(-2)
                    assert bool((area.sum((-2,-1))==75600**2).all())
                    report['density'].append({'prompt':prompt,'step':step,'layer':layer,
                        'retained_area':float((mask*area).sum()),'total_area':float(area.sum())})
                    del mask,qs,ks,area,prediction,sample,target
                    dense()
                    times=[]
                    for repeat in range(3):
                        row={}
                        for name,fn in [('ear',ear),('dense',dense)][::1 if repeat%2==0 else -1]:
                            if name=='ear':restore(proc,pre)
                            out,ms=timed(fn);row[name]=ms;del out
                        times.append(row)
                    restore(proc,post)
                    report['timings'].append({'prompt':prompt,'step':step,'layer':layer,'paired_ms':times})
                    rr=report['rows'];tt=[x for call in report['timings'] for x in call['paired_ms']]
                    em=statistics.mean(x['ear'] for x in tt);dm=statistics.mean(x['dense'] for x in tt)
                    report['summary']={'cases':len(rr),'mean_relative_l2':statistics.mean(x['relative_l2'] for x in rr),
                        'pooled_relative_l2':(sum(x['squared_error'] for x in rr)/sum(x['reference_squared'] for x in rr))**.5,
                        'ear_wins_vs_current':sum(x['relative_l2']<x['c8_current'] for x in rr),
                        'realized_density':sum(x['retained_area'] for x in report['density'])/sum(x['total_area'] for x in report['density']),
                        'mean_attention_ms':em,'mean_sdpa_attention_ms':dm,'speedup_vs_sdpa':dm/em,'timing_heads':4}
                    save(a.output,report)
                    print(json.dumps({'prompt':prompt,'step':step,'layer':layer,'summary':report['summary']}),flush=True)
                    del blob,q,k,v
    assert len(report['rows'])==384 and len(report['timings'])==96
    report['complete']=True;save(a.output,report)


if __name__=='__main__':main()
