"""Sequential generation on one exclusively idle GPU; never starts C8 twice."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from worker import read,save

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,choices=[0,7],default=0);a=parser.parse_args()
    control=ROOT if a.gpu==0 else ROOT/'gpu7';control.mkdir(exist_ok=True)
    priority=[4,6,10] if a.gpu==0 else [10,6,4]
    claims=ROOT/'claims';claims.mkdir(exist_ok=True)
    def claim():
        for cap in priority:
            path=claims/f'C{cap}.json'
            try:fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
            except FileExistsError:continue
            with os.fdopen(fd,'w') as f:json.dump(dict(capacity=cap,gpu=a.gpu,host='scai7',pid=os.getpid(),time=time.time()),f)
            return cap
    def run(stage,args):
        save(control/'stage.json',dict(stage=stage,pid=os.getpid(),gpu=a.gpu,time=time.time()))
        with (control/f'{stage}.log').open('w') as log:
            child=subprocess.Popen([sys.executable,str(ROOT/'source/worker.py'),'--gpu',str(a.gpu),*args],stdout=log,stderr=subprocess.STDOUT)
            save(control/'stage.json',dict(stage=stage,pid=os.getpid(),child=child.pid,gpu=a.gpu,time=time.time()))
            code=child.wait()
        if code:
            save(control/'failure.json',dict(stage=stage,exit_code=code,gpu=a.gpu,time=time.time()))
            raise RuntimeError(f'{stage} failed with {code}; preserve outputs and claim for diagnosis')
    cap=claim();done=[]
    if cap is not None:
        run('native_gate',['--gate'])
        while cap is not None:
            run(f'C{cap}',['--capacity',str(cap)]);done.append(cap);cap=claim()
    save(control/'batch_complete.json',dict(complete=True,gpu=a.gpu,capacities=done,videos=20*len(done),time=time.time()))
    if all((ROOT/f'C{c}/generation_complete.json').exists() for c in (4,6,10)):
        save(ROOT/'generation_complete.json',dict(complete=True,capacities=[4,6,10],videos=60,time=time.time()))
    save(control/'stage.json',dict(stage='generation_complete',gpu=a.gpu,time=time.time()))

if __name__=='__main__':main()
