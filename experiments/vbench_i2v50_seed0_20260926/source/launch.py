"""Detach exactly one supervised campaign after explicit user authorization."""
import json
import os
import subprocess
import time
from worker import ROOT, read, save, verify
from supervisor import clean, state


def main():
    plan=read(ROOT/'experiment_plan.json')
    verify(plan)
    clean(state(plan['gpu']),plan['gpu'])
    fd=os.open(ROOT/'dispatch.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
    with os.fdopen(fd,'w') as f:
        json.dump(dict(time=time.time(),status='dispatching'),f)
    with (ROOT/'supervisor.log').open('w') as log:
        proc=subprocess.Popen([plan['evaluation_python'],str(ROOT/'source/supervisor.py')],cwd=ROOT,
             env=dict(os.environ,PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,
             stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(ROOT/'dispatch.json',dict(time=time.time(),pid=proc.pid,status='dispatched',gpu=plan['gpu']))
    time.sleep(6)
    assert proc.poll() is None, (ROOT/'supervisor.log').read_text()[-3000:]
    print(json.dumps(dict(dispatched=True,supervisor=proc.pid,stage=read(ROOT/'stage.json'))))


if __name__=='__main__':
    main()
