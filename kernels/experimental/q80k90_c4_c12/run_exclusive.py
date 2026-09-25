"""Launch immediately on a clean GPU, monitor it, never wait or queue."""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def state(gpu):
    fields=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
    rows=csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines())
    return dict(uuid=fields[0].strip(),model=fields[1].strip(),memory=int(fields[2]),utilization=int(fields[3]),
                pids=[int(x[1]) for x in rows if x and x[0].strip()==fields[0].strip()])


def clean(snapshot):
    assert not snapshot['pids'] and snapshot['memory']<100 and snapshot['utilization']==0,snapshot


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--max-seconds',type=int,default=10800)
    p.add_argument('--output',type=Path,required=True);p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();assert a.command
    first=state(a.gpu);clean(first)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'preflight.json').write_text(json.dumps(first,indent=2)+'\n')
    cmd=a.command[1:] if a.command[0]=='--' else a.command
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=first['uuid'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
             PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1',LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6')
    child=None;failure=None
    def interrupted(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        clean(state(a.gpu))
        with (a.output/'worker.log').open('w') as log,(a.output/'monitor.jsonl').open('w') as monitor:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            (a.output/'process.json').write_text(json.dumps(dict(parent=os.getpid(),child=child.pid,command=cmd),indent=2)+'\n')
            started=time.monotonic()
            while True:
                if time.monotonic()-started>a.max_seconds:raise TimeoutError('Bounded optimization deadline; restore original job')
                snapshot=state(a.gpu)
                monitor.write(json.dumps(dict(time=time.time(),**snapshot))+'\n');monitor.flush()
                if set(snapshot['pids'])-{child.pid}:raise RuntimeError('Foreign GPU process: invalidate this run')
                if child.poll() is not None:break
                time.sleep(2)
            assert child.returncode==0,f'Worker exited {child.returncode}'
    except BaseException as exc:
        failure=repr(exc)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        (a.output/'completion.json').write_text(json.dumps(dict(valid=failure is None,failure=failure,
              exit_code=None if child is None else child.returncode),indent=2)+'\n')
    if failure:raise RuntimeError(failure)


if __name__=='__main__':main()
