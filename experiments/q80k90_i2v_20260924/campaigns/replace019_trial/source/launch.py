"""Launch one owned process group only after two clean H200 checks."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from worker import read,save

ROOT=Path(__file__).resolve().parents[1]

def state(gpu=0):
    fields=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
    procs=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()))
    return dict(uuid=fields[0].strip(),model=fields[1].strip(),memory_mib=int(fields[2]),utilization=int(fields[3]),
                pids=[int(row[1]) for row in procs if row and row[0].strip()==fields[0].strip()])

def clean(s,gpu=0):
    assert s['uuid']=={0:'GPU-4eca131f-0ba3-759d-bcb5-0418d6b7c68e',7:'GPU-7aa670f7-95a6-1e16-c28f-80664a449ad8'}[gpu] and s['model']=='NVIDIA H200'
    assert not s['pids'] and s['memory_mib']<100 and s['utilization']==0,s

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[0,7],default=0);a=p.parse_args()
    control=ROOT if a.gpu==0 else ROOT/'gpu7';control.mkdir(exist_ok=True)
    lock=(control/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (control/'launch.json').exists(),'Existing launch; inspect instead of duplicating'
    plan=read(ROOT/'experiment_plan.json')
    first=state(a.gpu);clean(first,a.gpu);time.sleep(1);clean(state(a.gpu),a.gpu)
    save(control/'preflight.json',dict(time=time.time(),**first))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=first['uuid'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
             PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1',MAX_JOBS='4',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
             WAN_MODEL_ROOT=plan['protocol']['model_root'],GRIDMERGE_SDPA_BACKEND='flash_attn2',
             SPARSE_VIDEO_GEN_ROOT=plan['source_svg_root'],
             FLASHINFER_WORKSPACE_BASE='/xuanwu-tank/east/zrl/exp-gridmergeattn/.cache/environments/monarch-rt/flashinfer_workspace',
             SVG_TOP_P='0.8',SVG_NUM_Q_CENTROIDS='300',SVG_NUM_K_CENTROIDS='1000',SVG_MIN_KC_RATIO='0.10',SVG_KMEANS_INIT='50',SVG_KMEANS_STEP='2',
             TORCH_EXTENSIONS_DIR=str(control/'torch_extensions'),TRITON_CACHE_DIR=str(control/'triton_cache'),
             PATH='/usr/local/cuda/bin:'+os.environ['PATH'])
    child=None;failure=None
    def interrupted(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        with (control/'supervisor.log').open('w') as log,(control/'monitor.jsonl').open('w') as monitor:
            child=subprocess.Popen([plan['python'],str(ROOT/'source/run_campaign.py'),'--gpu',str(a.gpu)],cwd=ROOT,env=env,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(control/'launch.json',dict(supervisor=os.getpid(),child=child.pid,time=time.time(),gpu=a.gpu,gpu_state=first,borrowed=a.gpu==0))
            start=time.monotonic()
            while True:
                s=state(a.gpu);foreign=[]
                for pid in s['pids']:
                    try:
                        if os.getsid(pid)!=child.pid:foreign.append(pid)
                    except ProcessLookupError:pass
                monitor.write(json.dumps(dict(time=time.time(),foreign_pids=foreign,**s))+'\n');monitor.flush()
                if foreign:raise RuntimeError(f'Foreign GPU process: {foreign}; stop only owned group')
                if child.poll() is not None:
                    assert child.returncode==0,f'Owned worker exited {child.returncode}'
                    break
                if time.monotonic()-start>12*3600:raise TimeoutError('12-hour bounded batch deadline')
                time.sleep(5)
    except BaseException as e:failure=repr(e)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        save(control/'completion.json',dict(valid=failure is None,failure=failure,exit_code=None if child is None else child.returncode,
                                         time=time.time(),gpu=a.gpu,borrowed=a.gpu==0,restoration_required=a.gpu==0))
    if failure:raise RuntimeError(failure)

if __name__=='__main__':main()
