"""Run C4 on one additional H200 while the original Dense job continues."""
import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
import traceback
from worker import ROOT,read,save,sha,verify
from supervisor import clean,state,stop_owned

CONTROL=ROOT/'parallel'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    CONTROL.mkdir(exist_ok=True)
    lock=(CONTROL/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (CONTROL/'launch.json').exists(),'Existing parallel launch'
    plan=read(ROOT/'experiment_plan.json');assignment=read(ROOT/'parallel_plan.json');gpu=assignment['gpu']
    assert assignment['original_plan_sha256']==sha(ROOT/'experiment_plan.json')
    assert socket.gethostname().split('.')[0]==gpu['host']
    assert gpu['uuid']!=plan['gpu']['uuid'] and assignment['max_gpus']==2
    verify(plan)
    if a.resume:
        recovery=read(ROOT/'resume_plan.json');gate=read(ROOT/'native_gate.json')
        assert recovery['original_plan_sha256']==sha(ROOT/'experiment_plan.json')
        assert recovery['native_gate_sha256']==sha(ROOT/'native_gate.json')
        assert gate['complete'] and gate['passed'] and gate['gpu']==gpu
    first=state(gpu);clean(first,gpu);time.sleep(2);second=state(gpu);clean(second,gpu)
    save(CONTROL/'preflight.json',dict(time=time.time(),checks=[first,second]))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
             PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1',MAX_JOBS='4',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
             WAN_MODEL_ROOT=plan['protocol']['model_root'],GRIDMERGE_SDPA_BACKEND='flash_attn2',
             SPARSE_VIDEO_GEN_ROOT='/xuanwu-tank/east/zrl/exp-gridmergeattn/third_party/Sparse-VideoGen',
             FLASHINFER_WORKSPACE_BASE='/xuanwu-tank/east/zrl/exp-gridmergeattn/.cache/environments/monarch-rt/flashinfer_workspace',
             TORCH_EXTENSIONS_DIR=str(CONTROL/'torch_extensions'),TRITON_CACHE_DIR=str(CONTROL/'triton_cache'),
             PATH='/usr/local/cuda/bin:'+os.environ['PATH'])
    order=['C4'] if a.resume else ['gate','C4']
    save(CONTROL/'launch.json',dict(supervisor=os.getpid(),time=time.time(),gpu=gpu,order=order,resume=a.resume))
    child=None
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for stage in order:
            clean(state(gpu),gpu)
            with (CONTROL/f'{stage}.log').open('w') as log,(CONTROL/'monitor.jsonl').open('a') as monitor:
                command=[plan['python'],str(ROOT/'source/worker.py'),stage,'--parallel']
                if a.resume:command.append('--resume')
                child=subprocess.Popen(command,cwd=ROOT,env=env,
                                       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                save(CONTROL/'stage.json',dict(stage=stage,pid=child.pid,gpu=gpu,time=time.time()))
                started=time.monotonic()
                while True:
                    s=state(gpu);foreign=[]
                    for pid in s['pids']:
                        try:
                            if os.getsid(pid)!=child.pid:foreign.append(pid)
                        except ProcessLookupError:
                            pass
                    monitor.write(json.dumps(dict(time=time.time(),stage=stage,foreign_pids=foreign,**s))+'\n');monitor.flush()
                    if foreign:
                        save(CONTROL/'timing_invalid.json',dict(reason='foreign GPU process',pids=foreign,time=time.time()))
                        raise RuntimeError(f'Foreign GPU processes {foreign}; stop only owned C4 work')
                    if child.poll() is not None:
                        assert child.returncode==0,f'{stage} exit {child.returncode}'
                        break
                    assert time.monotonic()-started<120*3600,'C4 stage exceeded 120 hours'
                    time.sleep(10)
            time.sleep(3)
        save(CONTROL/'completion.json',dict(complete=True,videos=50,gpu=gpu,time=time.time()))
        save(CONTROL/'stage.json',dict(stage='C4_complete_GPU_released',gpu=gpu,time=time.time()))
    except BaseException:
        save(CONTROL/'failure.json',dict(error=traceback.format_exc(),time=time.time()))
        stop_owned(child)
        # Dense and its supervisor are independent and must continue unchanged.
        raise
    # The original supervisor owns the CPU evaluator and combined completion.
    # Amend its legacy single-GPU completion metadata after both arms finish.
    started=time.monotonic()
    while not (ROOT/'generation_complete.json').exists():
        if (ROOT/'failure.json').exists():
            return
        if time.monotonic()-started>120*3600:
            return
        time.sleep(30)
    result=read(ROOT/'generation_complete.json')
    assert result['complete'] and result['Dense']==result['C4']==50
    result.pop('gpu',None)
    result.update(gpu_by_method={'Dense':plan['gpu'],'C4':gpu},execution='parallel on two H200 GPUs')
    save(ROOT/'generation_complete.json',result)


if __name__=='__main__':
    main()
