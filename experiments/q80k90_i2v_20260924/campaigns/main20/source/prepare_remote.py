"""Freeze the existing I2V campaign's working source and input identities."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/local2/denghaoran/GridMergeAttention/experiments/q80k90_i2v_c4_c6_c8_c10_20260924')
CAMPAIGN = Path('/xuanwu-tank/east/zrl/exp-gridmergeattn')
ORIGINAL = CAMPAIGN/'ours/GridMergeAttention'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def main():
    import yaml
    assert not (ROOT/'experiment_plan.json').exists(), 'Already prepared; inspect before reusing'
    snap = ROOT/'snapshot'
    snap.mkdir(parents=True, exist_ok=True)
    (ROOT/'source').mkdir(exist_ok=True)
    suffixes = {'.py','.cu','.cuh','.cpp','.cc','.h','.hpp','.yaml','.yml','.json','.txt'}
    for folder in ('attention','analysis','configs','kernels','pipeline','utils','wan'):
        for directory, dirs, files in os.walk(ORIGINAL/folder, followlinks=False):
            dirs[:] = [d for d in dirs if d not in ('.git','__pycache__','.cache','build','out')]
            for name in files:
                p = Path(directory)/name
                if p.suffix not in suffixes or name.endswith('_environment.json'): continue
                if p.stat().st_size > 20*1024*1024: continue
                dst = snap/p.relative_to(ORIGINAL)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
    shutil.copy2(ORIGINAL/'generate.py', snap/'generate.py')
    (snap/'prompts').mkdir(exist_ok=True)
    for name in ('vbench_i2v_20.txt','vbench_i2v_20.images.txt'):
        shutil.copy2(ORIGINAL/'prompts'/name, snap/'prompts'/name)
    prompt_path = snap/'prompts/vbench_i2v_20.txt'
    image_list = snap/'prompts/vbench_i2v_20.images.txt'
    prompts = [x.strip() for x in prompt_path.read_text().splitlines() if x.strip()]
    names = [x.strip() for x in image_list.read_text().splitlines() if x.strip()]
    image_dir = Path('/xuanwu-tank/east/zrl/datasets/vbench_i2v_20_720p')
    dense = CAMPAIGN/'h200_run/videos_p50i/p50i_dense'
    assert len(prompts) == len(names) == 20 and 'astronaut floating in space' in prompts[0]
    inputs = []
    for i, (prompt, name) in enumerate(zip(prompts, names)):
        image = image_dir/name
        video = dense/f'{i:03d}-00.mp4'
        assert image.is_file() and video.is_file()
        inputs.append(dict(index=i, seed=i, prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                           image=str(image), image_sha256=sha(image), dense_video=str(video), dense_sha256=sha(video)))
    base = yaml.safe_load((snap/'configs/base50i_q80k90_c8_unipc.yaml').read_text())
    dense_cfg = yaml.safe_load((snap/'configs/base50i_dense_unipc.yaml').read_text())
    stripped = dict(base); stripped['token_merging_args'] = {'enable':False}
    assert stripped == dense_cfg, 'Existing I2V C8 and Dense protocols differ beyond attention'
    configs = {}
    for c in (4,6,8,10):
        cfg = dict(base, token_merging_args={'enable':False,'attention_backend':f'q80k90_c{c}_all'})
        p = snap/f'configs/q80k90_i2v_C{c}.yaml'
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))
        configs[str(c)] = str(p)
    kernel = snap/'kernels/experimental/q80k90_c4_c12'
    selected = json.loads((kernel/'validated_h200_configs.json').read_text())
    assert all(selected[str(c)]['implementation'] == 'v4' for c in (4,6,8,10))
    critical = ['generate.py','wan/modules/model.py','attention/q80k90_wan.py','attention/runtime_timing.py',
                'analysis/score_videos_diffusers.py','kernels/experimental/q80k90_c4_c12/recommended.py',
                'kernels/experimental/q80k90_c4_c12/q80k90_kernel.py','kernels/experimental/q80k90_c4_c12/q80k90_merge.cu',
                'kernels/experimental/q80k90_c4_c12/validated_h200_configs.json']
    hashes = {rel:sha(snap/rel) for rel in critical}
    assert all(sha(ORIGINAL/rel) == value for rel,value in hashes.items())
    for folder in ('pipeline','utils','wan'):
        for p in (snap/folder).rglob('*.py'): hashes[str(p.relative_to(snap))] = sha(p)
    plan = dict(created_at=time.time(), status='prepared_not_launched', root=str(ROOT), snapshot=str(snap),
                source_repo=str(ORIGINAL), source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ORIGINAL,text=True).strip(),
                protocol=dict(model='Wan2.1-I2V-14B-720P',model_root='/xuanwu-tank/east/zrl/wan_models',
                    inputs='VBench I2V first20; same existing campaign',videos_per_capacity=20,frames=81,height=720,width=1280,
                    steps=50,sampler='FlowUniPC',flow_shift=5.,guidance_scale=5.,seed_policy='base0 + prompt_index',
                    attention_calls_per_video=4000,all_layers=40,q=80,k=90,capacities=[4,6,8,10],pure_all_layers=True),
                python='/home/zrlu/miniforge3/envs/monarch_rt/bin/python',
                evaluation_python='/home/denghaoran/miniconda3/envs/monarch_rt/bin/python',configs=configs,
                prompt_path=str(prompt_path),image_list=str(image_list),image_dir=str(image_dir),dense_reference=str(dense),
                input_manifest=inputs,source_sha256=hashes,selected_kernels=selected,
                launch_capacities=[4,6,10],gpu=dict(host='scai7',index=0,uuid='GPU-4eca131f-0ba3-759d-bcb5-0418d6b7c68e',model='NVIDIA H200'),
                existing_c8=dict(host='scai6',gpu=4,pid=518609,root=str(CAMPAIGN/'h200_run/videos_p50i/p50i_q80k90_c8'),
                    runtime=str(CAMPAIGN/'h200_run/runtime_p50i/p50i_q80k90_c8.json'),do_not_restart=True),
                evaluation=dict(source='analysis/score_videos_diffusers.py',frames='all81',resize=False,device='cpu',lpips='alex',
                    note='Same metric implementation as existing I2V campaign; CPU evaluation recorded separately.'),
                timing_note='Record scai7 GPU0 separately from existing C8 scai6 GPU4; no pooled timings.',
                user_protocol_authorization='User explicitly requested same settings as other I2V experiments; I2V VBench/seed-index protocol overrides generic T2V prompt defaults.')
    (ROOT/'experiment_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'prepared':str(ROOT),'source_files':len(list(snap.rglob('*.*'))),'inputs':len(inputs),'capacities_to_launch':plan['launch_capacities'],'existing_c8_reused':True}))

if __name__ == '__main__': main()
