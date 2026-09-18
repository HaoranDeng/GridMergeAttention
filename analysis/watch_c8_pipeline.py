"""User-authorized scai7 idle-H200 queue for bounded pipeline experiments.

One child / GPU at a time. Never touches other users' processes. Contaminated
attempts are invalid and retried on a freshly clean GPU. Job definitions are
read from a JSON queue so new CPU-compiled rounds can be submitted safely.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    temp=path.with_suffix('.writing.json')
    temp.write_text(json.dumps(value,indent=2))
    temp.replace(path)


def inventory():
    def smi(*a):
        return subprocess.check_output(['nvidia-smi',*a,'--format=csv,noheader,nounits'],text=True,timeout=15)
    gpu=list(csv.reader(smi('--query-gpu=index,name,uuid,utilization.gpu,memory.used').splitlines()))
    proc=list(csv.reader(smi('--query-compute-apps=gpu_uuid,pid').splitlines()))
    busy={r[0].strip() for r in proc if r}
    return [{'index':int(r[0]),'name':r[1].strip(),'uuid':r[2].strip(),
             'utilization':int(r[3]),'memory_mib':int(r[4]),'has_process':r[2].strip() in busy}
            for r in gpu]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--poll-seconds',type=int,default=30)
    a=p.parse_args()
    if socket.gethostname().split('.')[0]!='scai7':
        raise RuntimeError('This user-authorized queue runs only on scai7')
    a.run_dir.mkdir(parents=True,exist_ok=True)
    lock=a.run_dir/'watcher.lock'
    try:
        fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError:
        raise RuntimeError('Watcher lock exists; inspect the recorded PID before restarting')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    status={'pid':os.getpid(),'host':socket.gethostname(),'started_utc':now(),
            'target_complete_call_ms':140,'completed_jobs':{},'attempts':[]}
    state_path=a.run_dir/'watcher_state.json'
    if state_path.exists():
        old=json.loads(state_path.read_text())
        status['completed_jobs']=old.get('completed_jobs',{})
        status['attempts']=old.get('attempts',[])
    child=None
    def terminate(signum, frame):
        raise SystemExit(f'Watcher received signal {signum}')
    signal.signal(signal.SIGTERM,terminate)
    signal.signal(signal.SIGINT,terminate)
    try:
        while True:
            status['updated_utc']=now()
            if (a.run_dir/'watcher.stop').exists():
                status['status']='stopped';atomic(state_path,status)
                return
            queue=json.loads((a.run_dir/'queue.json').read_text())
            status['target_complete_call_ms']=queue.get('target_complete_call_ms',140)
            jobs=[j for j in queue['jobs'] if j['id'] not in status['completed_jobs']]
            if not jobs:
                status['status']='waiting_for_new_compiled_round'
                atomic(state_path,status)
                time.sleep(a.poll_seconds)
                continue
            job=jobs[0]
            compilation=json.loads((a.run_dir/job['compile_result']).read_text())
            if not all(compilation['candidates'].get(n,{}).get('compiled') for n in job['sources']):
                status['status']='compilation_not_ready';atomic(state_path,status)
                time.sleep(a.poll_seconds);continue
            devices=inventory()
            status['inventory']=devices
            idle=[g for g in devices if 'H200' in g['name'] and not g['has_process'] and
                  g['utilization']==0 and g['memory_mib']<=16]
            if not idle:
                status['status']='waiting_for_clean_H200';atomic(state_path,status)
                time.sleep(a.poll_seconds);continue
            # Prefer the user's GPU4, otherwise use any exclusively idle H200.
            idle.sort(key=lambda g:(g['index']!=4,g['index']))
            gpu=idle[0]
            time.sleep(3)
            check=next(g for g in inventory() if g['uuid']==gpu['uuid'])
            if check['has_process'] or check['utilization'] or check['memory_mib']>16:
                continue
            number=len(status['attempts'])+1
            name=f"{job['id']}_attempt{number:03d}_gpu{gpu['index']}"
            result=a.run_dir/f'{name}.json'
            log=a.run_dir/f'{name}.log'
            command=[sys.executable,'-u','analysis/benchmark_c8_shallow_pipeline.py',
                     '--gpu',str(gpu['index']),'--blocks',str(job.get('blocks',16)),
                     '--layers',job.get('layers','00,19,39'),'--output',str(result),
                     '--sources',*job['sources']]
            record={'job':job['id'],'gpu':gpu,'started_utc':now(),
                    'result':str(result),'log':str(log)}
            status['attempts'].append(record)
            env=dict(os.environ,CUDA_HOME='/usr/local/cuda')
            env['PATH']='/usr/local/cuda/bin:'+env['PATH']
            with log.open('w') as stream:
                child=subprocess.Popen(command,cwd=a.run_dir/'source',env=env,
                                       stdout=stream,stderr=subprocess.STDOUT)
                record['pid']=child.pid
                status['status']='running';status['active_job']=record
                atomic(state_path,status)
                deadline=time.monotonic()+job.get('timeout_seconds',900)
                while child.poll() is None:
                    status['updated_utc']=now();atomic(state_path,status)
                    if time.monotonic()>deadline:
                        record['timeout']=True
                        child.terminate()
                        try: child.wait(timeout=20)
                        except subprocess.TimeoutExpired: child.kill();child.wait()
                        break
                    time.sleep(10)
                record['exit_code']=child.returncode
                child=None
            record['finished_utc']=now()
            r=json.loads(result.read_text()) if result.exists() else {}
            if r.get('status')=='complete' and r.get('valid_timing'):
                status['completed_jobs'][job['id']]={'result':str(result),'finished_utc':now()}
                status['status']='round_complete'
            elif r.get('monitor_errors') or 'GPU is not exclusively clean' in log.read_text()[-4000:]:
                record['invalid_reason']='contention; retry only on a clean GPU'
                status['status']='waiting_for_clean_H200'
            else:
                record['invalid_reason']=r.get('error','worker failed; inspect log')
                status['status']='needs_attention'
                atomic(state_path,status)
                return
            status.pop('active_job',None);atomic(state_path,status)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=20)
            except subprocess.TimeoutExpired: child.kill();child.wait()
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    main()
