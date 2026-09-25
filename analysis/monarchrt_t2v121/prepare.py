"""Freeze the historical T2V Diffusers protocol; preparation does not use a GPU."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from common import ROOT,read,save,sha

BASE=Path('/local2/denghaoran/GridMergeAttention')
SOURCE=BASE/'worktrees/hellinger_c8_kernel_opt_20260831'
OLD=BASE/'experiments/wan14b_720p_50step_monarchrt_h2w2_3videos_20260901'
I2V=BASE/'experiments/monarchrt_i2v20_sparkler_20260925'
DENSE=BASE/'experiments/wan14b_720p_50step_hellinger_p16_c32g45_a0p2_20260829/dense_reference_exact_prompt_20260816'
PYTHON='/home/denghaoran/miniconda3/envs/monarch_rt/bin/python'

assert not (ROOT/'experiment_plan.json').exists(),'Already prepared; retain frozen inputs'
snapshot=ROOT/'snapshot';assert not snapshot.exists()
for folder in ('attention','analysis','utils'):
    for src in (SOURCE/folder).rglob('*.py'):
        dst=snapshot/src.relative_to(SOURCE);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
prompt=snapshot/'prompts/MovieGenVideoBench_extended.txt';prompt.parent.mkdir(parents=True)
shutil.copy2(SOURCE/'prompts/MovieGenVideoBench_extended.txt',prompt)
score=snapshot/'analysis/score_videos_diffusers.py'
shutil.copy2(I2V/'snapshot/analysis/score_videos_diffusers.py',score)
video=ROOT/'video20';video.mkdir()
command=[PYTHON,str(snapshot/'analysis/verify_prompt_contract.py'),'--prompt_path',str(prompt),
         '--num_videos','20','--seed','0','--num_samples','1','--manifest_out',str(video/'prompt_manifest.json')]
subprocess.run(command,check=True,stdout=subprocess.DEVNULL,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
manifest=read(video/'prompt_manifest.json');old=read(OLD/'monarch_rt_h2w2_all/prompt_manifest.json');dense=read(DENSE/'prompt_manifest.json')
assert manifest['prompt_file_sha256']==old['prompt_file_sha256']==dense['prompt_file_sha256']=='50293c2717c8f2df5773154d4680c74e636ecb08eb4fe13cdce7357d0ab9d121'
assert manifest['prompts']==old['prompts'][:20]==dense['prompts'][:20]
assert manifest['seed']==old['seed']==dense['seed']==0
inputs=[]
for row in manifest['prompts']:
    i=row['index'];path=DENSE/f'{i:03d}.mp4';assert path.is_file(),path
    inputs.append(dict(index=i,seed=0,prompt=row['text'],prompt_sha256=row['sha256'],dense_video=str(path),dense_sha256=sha(path)))
runtime=read(OLD/'monarch_rt_h2w2_all/runtime.json')
assert len(runtime['videos'])==20 and all(r['seed']==0 for r in runtime['videos'])
model=Path(runtime['protocol']['model']);assert (model/'model_index.json').is_file()
svg=Path('/home/denghaoran/workspace/Sparse-VideoGen');assert svg.is_dir()
svg_hashes={str(p):sha(p) for folder in ('svg','svg_inference') for p in (svg/folder).rglob('*.py')}
assert svg_hashes,'Expected historical Sparse-VideoGen source package'
files={str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*') if p.is_file() and p.suffix in ('.py','.txt')}
plan=dict(prepared_at=time.time(),status='prepared_not_launched_no_idle_H200',python=PYTHON,source=str(SOURCE),
    snapshot=str(snapshot),source_sha256=files,source_revision=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
    source_dirty=subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','-uno'],text=True).splitlines(),
    svg_root=str(svg),svg_sha256=svg_hashes,model=str(model),prompt_path=str(prompt),inputs=inputs,
    method=dict(f_tied=1,h_reduce=2,w_reduce=1,q_init='ith',head_chunk=1,query_outer_chunk=21,num_merge_tokens=1,backend='monarch_rt'),
    protocol=dict(model='Wan2.1-T2V-14B-Diffusers',width=1280,height=720,frames=81,steps=50,flow_shift=5,guidance_scale=5,
        seed='constant 0 per video, matching historical T2V (not the I2V index-based policy)',videos=20,attention_calls_per_video=4000,pure_all_layers=True),
    gate=dict(layers=[0,5,10,15,20,25,30,35],cfg_branches=2,reference='FP32 same-operator, TF32 disabled',engineering_relative_l2_limit=.02),
    tflops_per_call=9.5056625664,full_50step_pflops=38.0226502656,
    historical_max_memory_gib=runtime['summary']['max_memory_gib'],historical_runtime=str(OLD/'monarch_rt_h2w2_all/runtime.json'))
save(ROOT/'experiment_plan.json',plan)
for target,source in [('vbench_model_cache',I2V/'vbench_model_cache'),('vbench_clip_cache',I2V/'vbench_clip_cache')]:
    assert source.is_dir();(ROOT/target).symlink_to(source.resolve(),target_is_directory=True)
save(ROOT/'preparation_receipt.json',dict(prepared=True,inputs_verified=20,dense_verified=20,source_files=len(files),gpu_work_launched=False,time=time.time()))
subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_eval_support.py'))],check=True)
print('Prepared T2V121: 20 matched prompts/Dense hashes, fixed seed0, frozen historical source; no GPU launched')
