"""Explicitly authorized expansion from one GPU to two; preserve active Dense."""
import os
from pathlib import Path
import socket
import subprocess
import time
import json
from worker import ROOT,read,save,sha,verify
from supervisor import clean,state


def main():
    plan=read(ROOT/'experiment_plan.json');verify(plan)
    assert not (ROOT/'parallel_plan.json').exists(),'Parallel assignment already exists'
    assert not (ROOT/'failure.json').exists()
    stage=read(ROOT/'stage.json');assert stage['stage']=='Dense'
    pid=stage['pid'];os.kill(pid,0)
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    assert str(ROOT/'source/worker.py').encode() in cmd and b'Dense' in cmd
    assert os.getsid(pid)==pid
    assert pid in state(plan['gpu'])['pids'],'Original Dense worker must still own its GPU'
    gpu=dict(host='scai6',index=4,uuid='GPU-923ad71f-2808-6bf7-701e-dc79b7aaa38f',model='NVIDIA H200')
    assert socket.gethostname().split('.')[0]==gpu['host']
    first=state(gpu);clean(first,gpu);time.sleep(2);clean(state(gpu),gpu)
    assert not list((ROOT/'C4/videos').glob('*.mp4')) and not (ROOT/'native_gate.json').exists()
    control=ROOT/'parallel';control.mkdir(exist_ok=True)
    fd=os.open(control/'dispatch.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644);os.close(fd)
    assignment=dict(created_at=time.time(),authorization='User requested one additional GPU on 2026-09-26',
                    max_gpus=2,methods=['gate','C4'],gpu=gpu,dense_gpu=plan['gpu'],dense_pid=pid,
                    original_plan_sha256=sha(ROOT/'experiment_plan.json'),
                    source_sha256={p.name:sha(p) for p in (ROOT/'source').glob('*.py')},
                    quality_owner='Original Dense supervisor, CPU only after Dense50 completes',
                    timing_note='C4 native gate measures C4 and SDPA on GPU4. Dense video timings are from GPU2; keep GPU identities separate.')
    save(ROOT/'parallel_plan.json',assignment)
    with (control/'supervisor.log').open('w') as log:
        proc=subprocess.Popen([plan['evaluation_python'],str(ROOT/'source/parallel_supervisor.py')],cwd=ROOT,
                              env=dict(os.environ,PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,
                              stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(control/'dispatch.json',dict(pid=proc.pid,time=time.time(),gpu=gpu))
    # Source verification can take tens of seconds on the shared filesystem.
    for _ in range(30):
        assert proc.poll() is None,(control/'supervisor.log').read_text()[-3000:]
        if (control/'stage.json').exists():break
        time.sleep(1)
    print(json.dumps(dict(dispatched=True,supervisor=proc.pid,dense_pid_unchanged=pid,
                         stage=read(control/'stage.json') if (control/'stage.json').exists() else 'verifying_sources')))


if __name__=='__main__':
    main()
