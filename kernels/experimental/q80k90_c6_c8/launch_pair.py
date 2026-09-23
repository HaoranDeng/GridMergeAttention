"""Explicit manual launch after the user supplies two clean GPU indices."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from run_exclusive import state,clean

p=argparse.ArgumentParser()
p.add_argument('--c8-gpu',type=int,required=True)
p.add_argument('--c6-gpu',type=int,required=True)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--phase',choices=('invariants','tune','full'),default='full')
a=p.parse_args();assert a.c8_gpu!=a.c6_gpu
# Preflight both before starting either. Each guard independently rechecks its
# own card immediately before model/kernel work begins.
for gpu in (a.c8_gpu,a.c6_gpu):clean(state(gpu))
a.root.mkdir(parents=True,exist_ok=False)
here=Path(__file__).resolve().parent
started=[]
for cap,gpu in ((8,a.c8_gpu),(6,a.c6_gpu)):
    out=a.root/f'C{cap}'
    command=[sys.executable,'-u',str(here/'run_exclusive.py'),'--gpu',str(gpu),'--output',str(out),'--',
             sys.executable,'-u',str(here/'run_gpu.py'),'--capacity',str(cap),'--root',str(out),'--phase',a.phase]
    with (a.root/f'launch_c{cap}.log').open('w') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    started.append(dict(capacity=cap,gpu=gpu,parent_pid=proc.pid,output=str(out),command=command))
(a.root/'launch.json').write_text(json.dumps(started,indent=2)+'\n')
print(json.dumps(started))
