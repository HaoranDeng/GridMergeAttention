"""A gate followed by one disjoint ten-video C6 shard."""
import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from worker import read, save

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[2,5],required=True);a=p.parse_args()
    assert socket.gethostname().split('.')[0]=='scai6'
    control=ROOT/f'scai6_gpu{a.gpu}'
    claim=read(ROOT/'claims/C6.json')
    assert claim['host']=='scai6' and a.gpu in claim['gpus']
    for stage,args in [('native_gate',['--gate']),('C6',['--capacity','6'])]:
        with (control/f'{stage}.log').open('w') as log:
            child=subprocess.Popen([sys.executable,str(ROOT/'source/worker_scai6.py'),'--gpu',str(a.gpu),*args],stdout=log,stderr=subprocess.STDOUT)
            save(control/'stage.json',dict(stage=stage,pid=os.getpid(),child=child.pid,host='scai6',gpu=a.gpu,time=time.time()))
            rc=child.wait()
        if rc:
            save(control/'failure.json',dict(stage=stage,exit_code=rc,time=time.time()))
            raise RuntimeError(f'{stage} failed; preserve shard and claim')
    save(control/'stage.json',dict(stage='generation_complete',host='scai6',gpu=a.gpu,time=time.time()))

if __name__=='__main__':main()
