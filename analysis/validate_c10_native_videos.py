"""Frozen-source, single-H200 native C10 video validation (50 steps, 20 videos).

Kernel choice changes only the self-attention implementation. Every layer and
call uses native Fisher C10 RF0, Uniform16, SQ1 and full compressed attention.
Control and candidate run consecutively for each prompt on the same GPU.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def worker(run):
    import torch
    import random
    import numpy as np
    from native_fisher.operator import NativeFisherAttention, extension
    meta = json.loads((run/'manifest.json').read_text())
    for name, sha in meta['sources'].items():
        assert digest(ROOT/name) == sha, name
    op = NativeFisherAttention(capacity=10, merge_source=ROOT/meta['merge'],
                              selector_source=ROOT/meta['selector'])
    merges = {'candidate':op.merge, 'control':extension(10,ROOT/'native_fisher/fisher_c10_merge.cu')}
    from analysis import benchmark_wan14b_720p_methods as bench
    original_configure = bench.configure_processors
    counts, videos = Counter(), []
    current = {}

    class Audit:
        def begin_video(self,index):
            assert not counts

        def finish_video(self,core_calls,full_calls):
            assert core_calls == full_calls == 4000
            assert counts == Counter({i:100 for i in range(40)}), counts
            row = {**current,'total_calls':core_calls,'native_calls_by_layer':dict(counts),
                   'dense_calls':0,'expected_selector_calls':core_calls,
                   'expected_merge_calls':8*core_calls,'expected_route_calls':8*core_calls}
            videos.append(row)
            counts.clear()
            write(run/'dispatch_audit.json',{'complete':False,'videos':videos})
            return row

        def summary(self):
            return {'method':'native_fisher_geometric_c10_rf0_'+meta['variant'],
                    'pure_all_layer':True,'expected_calls_per_video':4000,'videos':videos}

    def configure(pipe,*args,**kwargs):
        # Reuse the established full-model preprocessing/projection path; every
        # processor's complete attention core is replaced before inference.
        assert args[0] == 'gma' and len(pipe.transformer.blocks) == 40
        original_configure(pipe,*args,**kwargs)
        for layer, block in enumerate(pipe.transformer.blocks):
            processor = block.attn1.processor
            assert processor.use_gma
            def attention(q,k,v,layer=layer):
                assert all(x.shape == (1,40,75600,128) and x.dtype == torch.bfloat16 for x in (q,k,v))
                counts[layer] += 1
                result = op(*(x.transpose(1,2).contiguous().reshape(1,21,45,80,40,128) for x in (q,k,v)))
                return result.reshape(1,75600,40,128).transpose(1,2)
            processor.gma_attention = attention
        return Audit()

    sys.path.insert(0,meta['svg_root'])
    from diffusers import AutoencoderKLWan, WanPipeline, UniPCMultistepScheduler
    from diffusers.utils import export_to_video
    from utils.prompt_contract import (load_nonempty_prompts, validate_canonical_prompt_contract,
                                      build_prompt_manifest, ensure_output_prompt_manifest)
    prompt_path = ROOT/'prompts/MovieGenVideoBench_extended.txt'
    prompts = load_nonempty_prompts(prompt_path)
    validate_canonical_prompt_contract(prompt_path,prompts)
    manifest = build_prompt_manifest(prompt_path,prompts,selected_count=meta['num_videos'],seed=0,
                                     num_samples=1,vbench_filenames=False,allow_noncanonical=False)
    for name in merges:
        ensure_output_prompt_manifest(run/name,manifest)
    torch.set_num_threads(4)
    torch.backends.cuda.preferred_linalg_library(backend='magma')
    vae = AutoencoderKLWan.from_pretrained(meta['model'],subfolder='vae',torch_dtype=torch.float32,local_files_only=True)
    pipe = WanPipeline.from_pretrained(meta['model'],vae=vae,torch_dtype=torch.bfloat16,local_files_only=True)
    pipe.scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction',use_flow_sigmas=True,
                                           num_train_timesteps=1000,flow_shift=5.0)
    pipe.to('cuda')
    timer, full_timer = bench.AttentionTimer(), bench.AttentionTimer()
    audit = configure(pipe,'gma',timer,full_timer,bench.AttentionErrorStats(0),meta['svg_root'],50,720,1280,81,
                      gma_output_layout='head_major',gma_head_chunk=5)
    records = []
    for index in range(meta['num_videos']):
        for name in ('control','candidate'):
            op.merge = merges[name]
            current.clear(); current.update(index=index,variant=name)
            random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
            audit.begin_video(index); timer.reset(); full_timer.reset()
            torch.cuda.synchronize(); started=time.perf_counter()
            frames = pipe(prompt=prompts[index],negative_prompt=bench.NEGATIVE_PROMPT,height=720,width=1280,
                          num_frames=81,guidance_scale=5.0,num_inference_steps=50,
                          generator=torch.Generator(device='cuda').manual_seed(0)).frames[0]
            torch.cuda.synchronize(); seconds=time.perf_counter()-started
            core_ms,core_calls = timer.finish(); full_ms,full_calls = full_timer.finish()
            audit.finish_video(core_calls,full_calls)
            export_to_video(frames,str(run/name/f'{index:03d}-00.mp4'),fps=16)
            records.append({**current,'attention_calls':core_calls,'attention_total_ms':core_ms,
                            'mean_attention_ms':core_ms/core_calls,'full_attention_calls':full_calls,
                            'full_attention_total_ms':full_ms,'inference_seconds':seconds})
            write(run/'runtime.json',{'protocol':{k:meta[k] for k in ('model','steps','guidance_scale','scheduler')},
                                      'videos':records,'matched_sdpa_ms':None})
            print(json.dumps(records[-1]),flush=True)
            del frames
    assert len(videos) == 2*meta['num_videos']
    for name in merges:
        assert sorted(p.name for p in (run/name).glob('*.mp4')) == [f'{i:03d}-00.mp4' for i in range(meta['num_videos'])]
    for name, sha in meta['sources'].items():
        assert digest(ROOT/name) == sha, name
    write(run/'dispatch_audit.json', {'complete':True,'videos':videos})


def launch(args):
    from analysis.launch_quality_first_videos import require_clean, gpu_processes, child_environment
    from analysis.launch_official_hybrid_50step import MODEL, SVG_ROOT, SVG_COMMIT
    initial = require_clean(args.gpu)
    assert (MODEL/'model_index.json').is_file()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SVG_ROOT,text=True).strip() == SVG_COMMIT
    assets = {str(p.relative_to(MODEL)):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
              for p in MODEL.rglob('*') if p.is_file()}
    run = args.run_dir.resolve()
    run.mkdir(parents=True,exist_ok=False)
    frozen = run/'source'
    names = subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    names += [str(p.relative_to(ROOT)).replace(os.sep,'/') for p in (ROOT/'native_fisher').glob('*') if p.is_file()]
    names += ['analysis/validate_c10_native_videos.py', 'prompts/MovieGenVideoBench_extended.txt']
    extensions = {'.py','.yaml','.yml','.json','.jinja','.jinja2','.cu','.cuh','.h'}
    selected = sorted({n for n in names if n and (Path(n).suffix in extensions or n.startswith('prompts/'))
                       and not n.startswith(('experiments/','outputs/','videos/'))})
    sources = {}
    for name in selected:
        target = frozen/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
        sources[name] = digest(target)
    meta = {'variant':args.variant,'num_videos':args.num_videos,'steps':50,'expected_calls_per_video':4000,
            'model':str(MODEL.resolve()),'model_file_metadata':assets,'svg_root':str(SVG_ROOT),'svg_commit':SVG_COMMIT,
            'scheduler':'UniPC flow shift5','guidance_scale':5.0,
            'merge':f'native_fisher/fisher_c10_{args.variant}.cu',
            'selector':'native_fisher/fisher_gs_register_distribute.cu','sources':sources,
            'timing_valid':False,
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    assert (frozen/meta['merge']).is_file()
    write(run/'manifest.json',meta)
    subprocess.run([sys.executable,str(frozen/'analysis/verify_prompt_contract.py'),
                    '--prompt_path',str(frozen/'prompts/MovieGenVideoBench_extended.txt'),
                    '--num_videos',str(args.num_videos),'--seed','0',
                    '--manifest_out',str(run/'prompt_preflight.json')],check=True,stdout=subprocess.DEVNULL)
    fresh = require_clean(args.gpu,expected_uuid=initial['gpu']['uuid'])
    env = child_environment(run,initial['gpu']['uuid'])
    env['TORCH_EXTENSIONS_DIR'] = str(run.parent/'torch_extensions')
    samples, errors = [], []
    with (run/'generation.log').open('x') as log:
        child = subprocess.Popen([sys.executable,str(frozen/'analysis/validate_c10_native_videos.py'),
                                  '--worker','--run-dir',str(run)],cwd=frozen,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                processes = gpu_processes(initial['gpu']['uuid'])
                samples.append({'time':time.time(),'processes':processes})
                if any(p['pid'] != child.pid for p in processes):
                    errors.append('foreign GPU process; own child stopped')
                    child.terminate()
                    break
                time.sleep(3)
            code = child.wait(timeout=30)
        except BaseException:
            child.terminate()
            child.wait(timeout=30)
            raise
        finally:
            write(run/'gpu_purity.json',{'initial':initial,'fresh':fresh,'samples':samples,'errors':errors})
    (run/'generation.exitcode').write_text(str(code)+'\n')
    assert code == 0 and not errors, (code,errors)
    meta['timing_valid'] = True
    write(run/'manifest.json',meta)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--gpu',type=int)
    p.add_argument('--variant',choices=['merge','half_owner','balanced_sums','probability_owner'],default='half_owner')
    p.add_argument('--num-videos',type=int,default=20)
    p.add_argument('--worker',action='store_true')
    args = p.parse_args()
    if args.worker:
        worker(args.run_dir.resolve())
    else:
        assert args.gpu is not None and 1 <= args.num_videos <= 20
        launch(args)
