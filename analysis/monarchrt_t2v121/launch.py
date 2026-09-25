"""One explicitly chosen idle H200 only; no availability queue or scheduler."""
import argparse
import csv
import fcntl
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from common import ROOT,read,save

def state(gpu):
    row=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20).strip().split(',')
    rows=csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=20).splitlines())
    return dict(uuid=row[0].strip(),model=row[1].strip(),memory_mib=int(row[2]),utilization=int(row[3]),pids=[int(r[1]) for r in rows if r and r[0].strip()==row[0].strip()])

def clean(gpu,uuid=None):
    s=state(gpu);assert 'H200' in s['model'] and not s['pids'] and s['memory_mib']<=16 and s['utilization']==0,s
    if uuid:assert uuid==s['uuid']
    # Historical paused generation queues can still reserve an otherwise idle card.
    for owner in Path('/xuanwu-tank/east/zrl/exp-gridmergeattn/h200_run/.queue').glob('lock_*/owner'):
        fields=owner.read_text().split()
        if len(fields)>2 and fields[0]==socket.gethostname().split('.')[0] and fields[2]==f'gpu{gpu}':
            assert not Path('/proc',fields[1]).exists(),str(owner)
    return s

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True,type=int);p.add_argument('--attempt',type=int,default=1)
    p.add_argument('--foreground',action='store_true');a=p.parse_args()
    host=socket.gethostname().split('.')[0];assert host in ('scai6','scai7')
    plan=read(ROOT/'experiment_plan.json');slot=f'{host}_gpu{a.gpu}_attempt{a.attempt}';control=ROOT/slot
    first=clean(a.gpu);time.sleep(1);clean(a.gpu,first['uuid'])
    if not a.foreground:
        control.mkdir(exist_ok=True);assert not (control/'dispatch.json').exists()
        with (control/'supervisor.log').open('w') as log:
            child=subprocess.Popen([plan['python'],str(Path(__file__).resolve()),'--gpu',str(a.gpu),'--attempt',str(a.attempt),'--foreground'],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        receipt=dict(host=host,pid=child.pid,slot=slot,time=time.time());save(control/'dispatch.json',receipt);print(receipt);return
    lock=(ROOT/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (control/'launch.json').exists();first.update(host=host,index=a.gpu,time=time.time());save(control/'preflight.json',first)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=first['uuid'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',
        TRITON_CACHE_DIR=str(control/'triton_cache'),TORCH_EXTENSIONS_DIR=str(control/'torch_extensions'),MAX_JOBS='4',
        FLASHINFER_WORKSPACE_BASE='/xuanwu-tank/east/zrl/exp-gridmergeattn/.cache/environments/monarch-rt/flashinfer_workspace',
        PATH='/usr/local/cuda/bin:'+os.environ['PATH'],PYTHONPATH=plan['snapshot']+':'+plan['svg_root'],
        XDG_CACHE_HOME='/xuanwu-tank/east/zrl/exp-gridmergeattn/.cache/environments/vbench',VBENCH_CACHE_DIR=str(ROOT/'vbench_model_cache'),
        MASTER_ADDR='127.0.0.1',MASTER_PORT=str(29730+a.gpu),WORLD_SIZE='1',RANK='0',LOCAL_RANK='0')
    save(control/'launch.json',dict(pid=os.getpid(),gpu=first,time=time.time()))
    with (ROOT/'quality.log').open('a') as log:
        cpu=subprocess.Popen([plan['python'],str(ROOT/'source/evaluate.py')],cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=''),
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    save(ROOT/'quality_launch.json',dict(pid=cpu.pid,host=host,time=time.time()))
    child=None;error=None
    def stop(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    try:
        for phase in ('gate','generate','vbench'):
            if phase in ('gate','generate') and (ROOT/'generation_complete.json').exists():continue
            if phase=='vbench' and (ROOT/'vbench/summary.json').exists():continue
            clean(a.gpu,first['uuid'])
            command=[plan['python'],str(ROOT/'source/worker.py'),'--slot',slot]
            if phase=='gate':command.append('--gate')
            if phase=='vbench':command=['/home/zrlu/miniforge3/envs/vbench/bin/python',str(ROOT/'source/vbench.py')]
            with (control/f'{phase}.log').open('w') as log,(control/'gpu_monitor.jsonl').open('a') as monitor:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                save(control/'active.json',dict(phase=phase,pid=child.pid,time=time.time()));started=time.monotonic()
                while child.poll() is None:
                    s=state(a.gpu);foreign=[]
                    for pid in s['pids']:
                        try:
                            if os.getsid(pid)!=child.pid:foreign.append(pid)
                        except ProcessLookupError:pass
                    monitor.write(__import__('json').dumps(dict(time=time.time(),foreign_pids=foreign,**s))+'\n');monitor.flush()
                    assert not foreign,('Foreign GPU process; invalidate this attempt',foreign)
                    assert time.monotonic()-started<24*3600,'Bounded24-hour phase';time.sleep(5)
                assert child.returncode==0,(phase,child.returncode);child=None
    except BaseException as exc:error=repr(exc)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        save(control/'completion.json',dict(valid=error is None,error=error,time=time.time()))
    if error:raise RuntimeError(error)

if __name__=='__main__':main()
