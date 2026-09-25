"""Finish one existing 20-pair C10 generation's metrics, without restarting it.

CPU fidelity runs as pairs become complete. GPU VBench starts only after the
identified generation terminates successfully and its single device is clean.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        return None


def write(path,data):
    temp=Path(path).with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2))
    temp.replace(path)


def identity(path):
    path=Path(path)
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}


def process_identity(pid):
    try:
        # Field 22 is process start time; pid alone is insufficient after exit.
        return (Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19],
                Path(f'/proc/{pid}/cmdline').read_bytes())
    except FileNotFoundError:
        return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--generation-parent',type=int,required=True)
    p.add_argument('--generation-worker',type=int,required=True)
    args=p.parse_args()
    run=args.run_dir.resolve(); result=run/'followup_metrics'
    result.mkdir(exist_ok=False)
    pins={pid:process_identity(pid) for pid in (args.generation_parent,args.generation_worker)}
    assert all(value is not None and str(run).encode() in value[1] for value in pins.values())
    assert read(run/'manifest.json')['num_videos']==20
    deadline=time.monotonic()+30*3600
    status={'valid':False,'phase':'cpu_fidelity','generation_pids':list(pins),'gpu':args.gpu,
            'scope':'20 canonical pairs; 81 full-resolution frames; six custom-input VBench dimensions'}
    write(result/'status.json',status)
    def wait_ready(predicate):
        while not predicate():
            exitfile=run/'generation.exitcode'
            if exitfile.exists() and exitfile.read_text().strip()!='0':
                raise RuntimeError('Existing generation failed; no restart or GPU evaluation')
            if not any(process_identity(pid)==pin for pid,pin in pins.items()):
                if not predicate():
                    raise RuntimeError('Generation terminated without the required completed output')
            if time.monotonic()>deadline:
                raise TimeoutError('Existing generation exceeded the 30-hour follow-up bound')
            time.sleep(30)
    try:
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
        os.environ['TORCH_HOME']='/local2/denghaoran/GridMergeAttention/vbench_home/.cache/torch'
        import torch
        import lpips
        from analysis.evaluate_video_quality import evaluate_pair,aggregate
        from utils.prompt_contract import assert_matching_prompt_manifests
        torch.set_num_threads(4)
        contract=assert_matching_prompt_manifests(run/'control',run/'candidate')
        metric=lpips.LPIPS(net='alex').cpu().eval()
        rows=[]
        def ready(index):
            runtime=read(run/'runtime.json')
            return runtime and all(any(v['index']==index and v['variant']==name and v['attention_calls']==4000
                                       for v in runtime['videos']) and (run/name/f'{index:03d}-00.mp4').is_file()
                                   for name in ('control','candidate'))
        for index in range(20):
            wait_ready(lambda:ready(index))
            paths=[run/name/f'{index:03d}-00.mp4' for name in ('control','candidate')]
            before=[identity(path) for path in paths]
            row=evaluate_pair(*paths,metric,2,torch.device('cpu'),expected_frames=81)
            assert before==[identity(path) for path in paths]
            row['inputs']=before; rows.append(row)
            write(result/'quality.json',{'valid':False,'prompt_contract':contract,'summary':aggregate(rows),
                                         'videos':rows,'generation_complete':False,
                                         'timing_note':'CPU fidelity runs alongside later generation; no matched SDPA timing'})
            print(json.dumps({'completed_pairs':len(rows),'metrics':row}),flush=True)
        wait_ready(lambda:(run/'generation.exitcode').exists() and (read(run/'manifest.json') or {}).get('timing_valid',False))
        assert (run/'generation.exitcode').read_text().strip()=='0'
        assert read(run/'dispatch_audit.json')['complete']
        purity=read(run/'gpu_purity.json')
        assert purity and not purity['errors']
        wait_ready(lambda:not any(process_identity(pid)==pin for pid,pin in pins.items()))
        from analysis.evaluate_video_quality import prepare_comparison
        prepare_comparison(run/'control',run/'candidate',expected_videos=20)
        quality=read(result/'quality.json');quality.update(valid=True,generation_complete=True)
        write(result/'quality.json',quality)
        del metric
        status['phase']='vbench';write(result/'status.json',status)
        from analysis.launch_quality_first_videos import require_clean,gpu_processes,child_environment
        from analysis.run_four_hybrid_comparison import DIMENSIONS,VBENCH,CACHE
        for name in ('control','candidate'):
            output=result/f'vbench_{name}';output.mkdir()
            subprocess.run([sys.executable,str(ROOT/'analysis/prepare_vbench_prompt_map.py'),
                            '--prompt_path',str(ROOT/'prompts/MovieGenVideoBench_extended.txt'),
                            '--video_dir',str(run/name),'--output',str(output/'prompts.json')],check=True)
            initial=require_clean(args.gpu)
            assert initial['gpu']['uuid']==purity['initial']['gpu']['uuid']
            env=child_environment(output,initial['gpu']['uuid'])
            env.update(PYTHONPATH=str(ROOT),VBENCH_CACHE_DIR=str(CACHE/'vbench'),XDG_CACHE_HOME=str(CACHE),
                       TORCH_HOME=str(CACHE/'torch'),HF_HOME=str(CACHE/'huggingface'))
            command=[sys.executable,'-u',str(VBENCH/'evaluate.py'),'--videos_path',str(run/name),
                     '--dimension',*DIMENSIONS,'--mode','custom_input','--load_ckpt_from_local','True',
                     '--prompt_file',str(output/'prompts.json'),'--output_path',str(output)]
            errors=[];samples=[]
            with (output/'evaluation.log').open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                try:
                    while child.poll() is None:
                        processes=gpu_processes(initial['gpu']['uuid'])
                        samples.append({'time':time.time(),'processes':processes})
                        if any(v['pid']!=child.pid for v in processes):
                            errors.append('foreign GPU process');child.terminate();break
                        time.sleep(5)
                    code=child.wait(timeout=30)
                finally:
                    if child.poll() is None:child.terminate();child.wait(timeout=30)
                    write(output/'gpu_purity.json',{'initial':initial,'samples':samples,'errors':errors})
            assert code==0 and not errors,(code,errors)
        status.update(valid=True,phase='complete',vbench_dimensions=list(DIMENSIONS),official_vbench_total=False)
        write(result/'status.json',status)
    except BaseException as error:
        status.update(valid=False,error=repr(error));write(result/'status.json',status)
        raise


if __name__=='__main__':main()
