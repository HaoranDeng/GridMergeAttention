"""Exclusive per-slot guard for the user's three-card replacement campaign."""
import argparse,csv,fcntl,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
from worker import ROOT,read,save

def state(gpu):
    fields=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
    rows=csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines())
    return dict(uuid=fields[0].strip(),model=fields[1].strip(),memory_mib=int(fields[2]),utilization=int(fields[3]),pids=[int(x[1]) for x in rows if x and x[0].strip()==fields[0].strip()])

def clean(s,gpu):
    host=socket.gethostname().split('.')[0];slot=f'{host}_gpu{gpu}';wanted=read(ROOT/'experiment_plan.json')['gpus'][slot]
    assert s['uuid']==wanted['uuid'] and s['model']=='NVIDIA H200'
    assert not s['pids'] and s['memory_mib']<100 and s['utilization']==0,s

def main():
    p=argparse.ArgumentParser();p.add_argument('--slot',required=True);a=p.parse_args();plan=read(ROOT/'experiment_plan.json')
    g=plan['gpus'][a.slot];assert socket.gethostname().split('.')[0]==g['host'];gpu=g['index'];control=ROOT/a.slot;control.mkdir(exist_ok=True)
    lock=(control/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (control/'launch.json').exists(),'Existing slot launch, diagnose before retry'
    first=state(gpu);clean(first,gpu);time.sleep(1);clean(state(gpu),gpu)
    save(control/'preflight.json',dict(time=time.time(),**first))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=g['uuid'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1',MAX_JOBS='4',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        WAN_MODEL_ROOT=plan['protocol']['model_root'],GRIDMERGE_SDPA_BACKEND='flash_attn2',SPARSE_VIDEO_GEN_ROOT=plan['source_svg_root'],
        FLASHINFER_WORKSPACE_BASE='/xuanwu-tank/east/zrl/exp-gridmergeattn/.cache/environments/monarch-rt/flashinfer_workspace',
        SVG_NUM_Q_CENTROIDS='300',SVG_NUM_K_CENTROIDS='1000',SVG_MIN_KC_RATIO='0.10',SVG_KMEANS_INIT='50',SVG_KMEANS_STEP='2',
        TORCH_EXTENSIONS_DIR=str(ROOT/'gpu2/torch_extensions') if a.slot=='scai6_gpu2' else str(control/'torch_extensions'),
        TRITON_CACHE_DIR=str(ROOT/'gpu2/triton_cache') if a.slot=='scai6_gpu2' else str(control/'triton_cache'),PATH='/usr/local/cuda/bin:'+os.environ['PATH'])
    child=None;failure=None
    def stop(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    try:
        with (control/'supervisor.log').open('w') as log,(control/'monitor.jsonl').open('w') as monitor:
            child=subprocess.Popen([plan['python'],str(ROOT/'source/campaign_multi.py'),'--slot',a.slot],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(control/'launch.json',dict(supervisor=os.getpid(),child=child.pid,slot=a.slot,gpu=gpu,time=time.time(),gpu_state=first))
            started=time.monotonic()
            while True:
                s=state(gpu);foreign=[]
                for pid in s['pids']:
                    try:
                        if os.getsid(pid)!=child.pid:foreign.append(pid)
                    except ProcessLookupError:pass
                monitor.write(json.dumps(dict(time=time.time(),foreign_pids=foreign,**s))+'\n');monitor.flush()
                assert not foreign,('Foreign compute process',foreign)
                if child.poll() is not None:
                    assert child.returncode==0,child.returncode;break
                assert time.monotonic()-started<12*3600,'Bounded campaign deadline';time.sleep(5)
    except BaseException as e:failure=repr(e)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        save(control/'completion.json',dict(valid=failure is None,failure=failure,slot=a.slot,gpu=gpu,time=time.time(),restoration_required=a.slot!='scai6_gpu2'))
    if failure:raise RuntimeError(failure)

if __name__=='__main__':main()
