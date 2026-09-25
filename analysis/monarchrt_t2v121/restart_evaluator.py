"""Restart only this campaign's absent CPU evaluator; preserve its last receipt."""
import os
from pathlib import Path
import socket
import subprocess
import time
from common import ROOT,read,save

old=read(ROOT/'quality_launch.json');assert old['host']==socket.gethostname().split('.')[0]
proc=Path('/proc',str(old['pid']))
if proc.exists():
    assert proc.stat().st_uid==os.getuid()
    try:state=next(line for line in (proc/'status').read_text().splitlines() if line.startswith('State:'))
    except FileNotFoundError:state='State: Z (already gone)'
    assert state.split()[1]=='Z','Do not duplicate a live evaluator'
save(ROOT/'history'/f'quality_launch_{old["pid"]}.json',old)
plan=read(ROOT/'experiment_plan.json');env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
with (ROOT/'quality.log').open('a') as log:
    child=subprocess.Popen([plan['python'],str(ROOT/'source/evaluate.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
save(ROOT/'quality_launch.json',dict(host=old['host'],pid=child.pid,time=time.time(),restarted_after=old['pid']))
print(f'Restarted CPU evaluator {child.pid}')
