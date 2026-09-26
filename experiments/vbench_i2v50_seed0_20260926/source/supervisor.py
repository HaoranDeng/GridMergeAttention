"""One GPU, sequential Dense50 -> C4 gate -> C4_50, with exclusive-use monitoring."""
import csv
import fcntl
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback
from worker import ROOT, read, save


def state(gpu):
    fields=subprocess.check_output(['nvidia-smi','-i',gpu['uuid'],'--query-gpu=uuid,name,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'],text=True).strip().split(',')
    procs=csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines())
    return dict(uuid=fields[0].strip(),model=fields[1].strip(),memory_mib=int(fields[2]),utilization=int(fields[3]),
                pids=[int(r[1]) for r in procs if r and r[0].strip()==gpu['uuid']])


def clean(s,gpu):
    assert s['uuid']==gpu['uuid'] and s['model']=='NVIDIA H200'
    assert not s['pids'] and s['memory_mib']<100 and s['utilization']==0,s


def stop_owned(proc):
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)


def main():
    lock=(ROOT/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'launch.json').exists(),'Launch already exists; inspect before resume'
    plan=read(ROOT/'experiment_plan.json');gpu=plan['gpu']
    assert socket.gethostname().split('.')[0]==gpu['host']
    first=state(gpu);clean(first,gpu);time.sleep(2);second=state(gpu);clean(second,gpu)
    save(ROOT/'preflight.json',dict(time=time.time(),checks=[first,second]))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
             PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1',MAX_JOBS='4',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
             WAN_MODEL_ROOT=plan['protocol']['model_root'],GRIDMERGE_SDPA_BACKEND='flash_attn2',
             SPARSE_VIDEO_GEN_ROOT='/xuanwu-tank/east/zrl/exp-gridmergeattn/third_party/Sparse-VideoGen',
             FLASHINFER_WORKSPACE_BASE='/xuanwu-tank/east/zrl/exp-gridmergeattn/.cache/environments/monarch-rt/flashinfer_workspace',
             TORCH_EXTENSIONS_DIR=str(ROOT/'torch_extensions'),TRITON_CACHE_DIR=str(ROOT/'triton_cache'),
             PATH='/usr/local/cuda/bin:'+os.environ['PATH'])
    save(ROOT/'launch.json',dict(supervisor=os.getpid(),time=time.time(),gpu=gpu,max_gpus=1,order=['Dense','gate','C4']))
    child=evaluator=None
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for stage in ('Dense','gate','C4'):
            clean(state(gpu),gpu)
            if stage=='C4':
                assert read(ROOT/'Dense/generation_complete.json')['complete'] and read(ROOT/'native_gate.json')['passed']
                with (ROOT/'quality.log').open('w') as qlog:
                    evaluator=subprocess.Popen([plan['evaluation_python'],str(ROOT/'source/worker.py'),'quality'],
                        env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=qlog,stderr=subprocess.STDOUT,start_new_session=True)
                save(ROOT/'quality_process.json',dict(pid=evaluator.pid,device='cpu',time=time.time()))
            with (ROOT/f'{stage}.log').open('w') as log,(ROOT/'monitor.jsonl').open('a') as monitor:
                child=subprocess.Popen([plan['python'],str(ROOT/'source/worker.py'),stage],cwd=ROOT,env=env,
                                       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                save(ROOT/'stage.json',dict(stage=stage,pid=child.pid,gpu=gpu,time=time.time()))
                started=time.monotonic()
                while True:
                    s=state(gpu);foreign=[]
                    for pid in s['pids']:
                        try:
                            if os.getsid(pid)!=child.pid:
                                foreign.append(pid)
                        except ProcessLookupError:
                            pass
                    import json
                    monitor.write(json.dumps(dict(time=time.time(),stage=stage,foreign_pids=foreign,**s))+'\n');monitor.flush()
                    if foreign:
                        save(ROOT/'timing_invalid.json',dict(reason='foreign GPU process',pids=foreign,time=time.time()))
                        raise RuntimeError(f'Foreign GPU processes {foreign}; stopping only owned work')
                    if child.poll() is not None:
                        assert child.returncode==0,f'{stage} exit {child.returncode}'
                        break
                    if evaluator is not None and evaluator.poll() not in (None,0):
                        raise RuntimeError(f'CPU evaluator exit {evaluator.returncode}')
                    assert time.monotonic()-started<120*3600,'Stage exceeded 120 hours'
                    time.sleep(10)
            time.sleep(3)
        save(ROOT/'generation_complete.json',dict(complete=True,Dense=50,C4=50,gpu=gpu,time=time.time()))
        save(ROOT/'stage.json',dict(stage='quality_finishing',time=time.time()))
        assert evaluator.wait(timeout=12*3600)==0
        save(ROOT/'completion.json',dict(complete=True,generation_videos=100,quality_pairs=50,time=time.time()))
        save(ROOT/'stage.json',dict(stage='complete',time=time.time()))
    except BaseException:
        save(ROOT/'failure.json',dict(error=traceback.format_exc(),time=time.time()))
        stop_owned(child);stop_owned(evaluator)
        raise


if __name__=='__main__':
    main()
