"""Twenty canonical 50-step videos with current-frame Q45K80 Hellinger C8."""
import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from analysis.benchmark_c8_current import save
from utils.prompt_contract import (build_prompt_manifest,ensure_output_prompt_manifest,
    validate_canonical_prompt_contract,assert_matching_prompt_manifests)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--model',default='/home/denghaoran/wan_models/Wan2.1-T2V-14B-Diffusers')
    parser.add_argument('--svg',default='/local2/denghaoran/GridMergeAttention_worktrees/svg_official_20260906ch')
    a=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    assert not (a.output/'runtime.json').exists(), 'Fresh output required'
    a.output.mkdir(parents=True,exist_ok=True)
    assert not list(a.output.glob('*.mp4')), 'Refusing to overwrite videos'
    evidence=json.loads(a.evidence.read_text())
    assert evidence['timing']['speed_gate_passed']
    assert evidence['gpu_guards']['timing']['valid'] and evidence['gpu_guards']['quality']['valid']
    assert evidence['numerics']['current']['cases']==384
    package=root/'kernels/experimental/c8_current'
    for name,digest in evidence['cuda_source_sha256'].items():
        assert hashlib.sha256((package/name).read_bytes()).hexdigest()==digest,name
    prompt_path=root/'prompts/MovieGenVideoBench_extended.txt'
    with (a.output/'prompt_check.log').open('w') as log:
        subprocess.run([sys.executable,'-m','analysis.verify_prompt_contract','--prompt_path',str(prompt_path),
                        '--num_videos','20','--seed','0','--manifest_out',str(a.output/'prompt_preflight_manifest.json')],
                       cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    prompts=[line.strip() for line in prompt_path.read_text().splitlines() if line.strip()]
    validate_canonical_prompt_contract(prompt_path,prompts)
    manifest=build_prompt_manifest(prompt_path,prompts,selected_count=20,seed=0,
                                   num_samples=1,vbench_filenames=False,allow_noncanonical=False)
    ensure_output_prompt_manifest(a.output,manifest)
    assert_matching_prompt_manifests(a.reference,a.output,prompt_indices=range(20))
    assert all((a.reference/f'{i:03d}-00.mp4').is_file() for i in range(20))
    source_paths=list(package.glob('*.cu'))+list(package.glob('*.py'))+[
        Path(__file__),root/'kernels/best/c8_r2/gma_c8_candidate_table.json',
        root/'analysis/benchmark_wan14b_720p_methods.py',root/'analysis/evaluate_video_quality.py']
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    protocol={'method':'Q45K80 C8 Hellinger current-frame; no RF0 reuse','model':a.model,
        'steps':50,'height':720,'width':1280,'frames':81,'fps':16,'seed':0,'num_videos':20,
        'guidance_scale':5.,'scheduler':'UniPC flow_shift=5','all_layers':40,'capacity':8,
        'singleton_routes':0,'expected_calls_per_video':4000,'dense_self_attention_calls':0,
        'seed_policy':'Each query frame independently selects C32/g45 Hellinger alpha=.2, A8U8 probes in every call',
        'source_sha256':hashes,'svg_source':a.svg,'reference':str(a.reference),
        'reference_note':'Historical canonical same-protocol dense videos; not a matched runtime denominator',
        'speed_gate':evidence['timing'],'vbench':None}
    runtime={'protocol':protocol,'videos':[],'generation_complete':False,'quality_complete':False,
             'quality_failures':[],'status':'loading model'}
    save(a.output/'runtime.json',runtime)
    print(json.dumps({'event':'loading_model','output':str(a.output)}),flush=True)
    ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libstdc++.so.6',mode=ctypes.RTLD_GLOBAL)
    import numpy as np
    import torch
    sys.path.insert(0,a.svg)
    from diffusers import AutoencoderKLWan,WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video
    from analysis.benchmark_wan14b_720p_methods import AttentionTimer,AttentionErrorStats,NEGATIVE_PROMPT,configure_processors
    from kernels.experimental.c8_current import create
    protocol['negative_prompt']=NEGATIVE_PROMPT
    protocol['wan_forward']='svg';protocol['qk_norm']='triton';protocol['rope']='svg'
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.set_num_threads(4)
    torch.backends.cuda.preferred_linalg_library(backend='magma')
    vae=AutoencoderKLWan.from_pretrained(a.model,subfolder='vae',torch_dtype=torch.float32,local_files_only=True)
    pipe=WanPipeline.from_pretrained(a.model,vae=vae,torch_dtype=torch.bfloat16,local_files_only=True)
    pipe.scheduler=UniPCMultistepScheduler(prediction_type='flow_prediction',use_flow_sigmas=True,
                                          num_train_timesteps=1000,flow_shift=5.)
    pipe.to('cuda')
    timer,full_timer=AttentionTimer(),AttentionTimer()
    configure_processors(pipe,'dense',timer,full_timer,AttentionErrorStats(0),a.svg,50,720,1280,81,
                         gma_output_layout='head_major')
    core=create()
    counts=[0]*40
    assert len(pipe.transformer.blocks)==40
    for layer,block in enumerate(pipe.transformer.blocks):
        processor=block.attn1.processor
        def invoke(q,k,v,layer=layer):
            assert q.shape==(1,40,75600,128)
            tokens=[x.transpose(1,2).contiguous() for x in (q,k,v)]
            result=core(*tokens,policy='current')
            counts[layer]+=1
            return result.transpose(1,2).contiguous()
        processor.use_gma=True
        processor.gma_attention=invoke
    runtime['status']='generating';save(a.output/'runtime.json',runtime)
    print(json.dumps({'event':'model_ready','layers':40,'policy':'current'}),flush=True)

    def evaluate(count):
        from analysis.evaluate_video_quality import main as evaluate_main
        subset=build_prompt_manifest(prompt_path,prompts,selected_count=count,seed=0,
                                     num_samples=1,vbench_filenames=False,allow_noncanonical=False)
        paths=[]
        for name,source in [('reference',a.reference),('method',a.output)]:
            target=a.output/f'evaluation_{count}'/name
            ensure_output_prompt_manifest(target,subset)
            assert_matching_prompt_manifests(source,target,prompt_indices=range(count))
            for index in range(count):
                src=source/f'{index:03d}-00.mp4';dst=target/src.name
                assert src.is_file()
                if not dst.exists():dst.symlink_to(src.resolve())
                assert dst.resolve()==src.resolve()
            paths.append(target)
        old=sys.argv
        try:
            sys.argv=['evaluate_video_quality','--reference_dir',str(paths[0]),'--method_dir',str(paths[1]),
                '--method_name','q45k80c8_current','--output',str(a.output/f'quality_{count}.json'),
                '--max_videos',str(count),'--expected_videos',str(count),'--expected_frames','81','--lpips_batch_size','4']
            evaluate_main()
        finally:
            sys.argv=old;gc.collect();torch.cuda.empty_cache()

    with torch.inference_mode():
        for index in range(20):
            before=counts.copy();timer.reset();full_timer.reset();torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize();started=time.perf_counter()
            def progress(pipeline,step,timestep,kwargs):
                assert bool(torch.isfinite(kwargs['latents']).all()),'Nonfinite generated latents'
                save(a.output/'progress.json',{'video_index':index,'completed_videos':len(runtime['videos']),
                    'step':step+1,'steps':50,'elapsed_video_seconds':time.perf_counter()-started,
                    'layer_calls_this_video':[x-y for x,y in zip(counts,before)],'updated_unix':time.time()})
                return kwargs
            frames=pipe(prompt=prompts[index],negative_prompt=NEGATIVE_PROMPT,height=720,width=1280,
                num_frames=81,guidance_scale=5.,num_inference_steps=50,
                generator=torch.Generator(device='cuda').manual_seed(0),callback_on_step_end=progress,
                callback_on_step_end_tensor_inputs=['latents']).frames[0]
            torch.cuda.synchronize();seconds=time.perf_counter()-started
            ms,n=timer.finish();full_ms,full_n=full_timer.finish()
            layer_calls=[x-y for x,y in zip(counts,before)]
            assert n==full_n==4000 and layer_calls==[100]*40
            assert len(frames)==81
            destination=a.output/f'{index:03d}-00.mp4';assert not destination.exists()
            temporary=a.output/f'{index:03d}-00.partial.mp4'
            export_to_video(frames,str(temporary),fps=16);temporary.replace(destination)
            record={'index':index,'video':destination.name,'prompt':prompts[index],'seed':0,'attention_calls':n,
                'mean_attention_ms':ms/n,'mean_full_attention_ms':full_ms/full_n,'inference_seconds':seconds,
                'layer_calls':layer_calls,'dense_self_attention_calls':0,'max_memory_gib':torch.cuda.max_memory_allocated()/2**30,
                'speedup_vs_sdpa_approximate':evidence['timing']['mean_attention_ms']['dense']/(ms/n),
                'speedup_note':'Approximate: independent H40 replay timing, different evolving QKV and layout adapters',
                'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}
            runtime['videos'].append(record);save(a.output/'runtime.json',runtime)
            print(json.dumps({'event':'video_complete',**record}),flush=True)
            del frames
            if index in (0,4):
                try:evaluate(index+1)
                except Exception as exc:
                    runtime['quality_failures'].append({'count':index+1,'error':repr(exc)})
                    save(a.output/'runtime.json',runtime)
    runtime['generation_complete']=True;runtime['status']='evaluating';save(a.output/'runtime.json',runtime)
    del pipe,vae,core,processor,invoke
    gc.collect();torch.cuda.empty_cache()
    evaluate(20)
    for relative,digest in hashes.items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest,relative
    runtime['quality_complete']=True;runtime['status']='complete';save(a.output/'runtime.json',runtime)
    print(json.dumps({'event':'complete','videos':20}),flush=True)


if __name__=='__main__':main()
