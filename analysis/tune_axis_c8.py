"""Bounded full-call tuning within an already running exclusive-GPU guard.

This script never waits for, reserves, or chooses a GPU and never launches video
generation. A separate full numerical and four-capture speed receipt is required.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from analysis.quick_axis_c4_qkv import save
from analysis.axis_c8_speed_gate import DEFAULTS


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--configs',nargs='+',default=['4:21:64:4:2','4:63:64:4:2',
                                                '4:126:64:4:2','4:63:128:8:2',
                                                '5:63:64:4:2','8:63:64:4:2'])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for index,config in enumerate(a.configs):
        hc,tiles,tokens,warps,stages=config.split(':')
        environment=dict(DEFAULTS,AXIS_C8_HEAD_CHUNK=hc,AXIS_C8_CUDA_TILES=tiles,
                         AXIS_C8_ROUTE_TOKENS=tokens,AXIS_C8_ROUTE_WARPS=warps,AXIS_C8_ROUTE_STAGES=stages)
        output=a.output/f'trial_{index:02d}.json'
        with (a.output/f'trial_{index:02d}.log').open('w') as log:
            result=subprocess.run([sys.executable,'-m','analysis.benchmark_axis_c8','--single',
                                   '--repeats','5','--output',str(output),'--profile'],
                                  env=dict(os.environ,**environment),stdout=log,stderr=subprocess.STDOUT)
        row={'config':config,'environment':environment,'returncode':result.returncode,'output':str(output)}
        if result.returncode==0:
            report=json.loads(output.read_text())
            row.update(mean_attention_ms=report['mean_attention_ms'],speedup_vs_sdpa=report['speedup_vs_sdpa'])
        results.append(row);save(a.output/'trials.json',{'trials':results})
        print(json.dumps(row),flush=True)
    valid=[row for row in results if row['returncode']==0]
    if not valid:raise RuntimeError('No valid C8 tuning configuration')
    best=min(valid,key=lambda row:row['mean_attention_ms'])
    save(a.output/'best.json',best)
    print(json.dumps({'best':best}),flush=True)


if __name__=='__main__':main()
