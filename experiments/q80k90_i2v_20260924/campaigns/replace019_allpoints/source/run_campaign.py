"""Sequential, exclusively monitored all-point replacement campaign."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from worker import ROOT,read,save

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[2],required=True);a=p.parse_args()
    plan=read(ROOT/'experiment_plan.json');control=ROOT/'gpu2'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    assert not (ROOT/'evaluation_launch.json').exists()
    with (ROOT/'quality.log').open('w') as log:
        evaluator=subprocess.Popen([plan['evaluation_python'],str(ROOT/'source/evaluate.py')],cwd=ROOT,env=env,
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    save(ROOT/'evaluation_launch.json',dict(pid=evaluator.pid,time=time.time(),device='cpu'))
    def run(name,args,setting=None):
        env=dict(os.environ)
        if setting:
            if 'top_p' in setting:env['SVG_TOP_P']=str(setting['top_p'])
            if 'sparsity' in setting:env['SVG_SPARSITY']=str(setting['sparsity'])
        with (control/(name+'.log')).open('w') as log:
            child=subprocess.Popen([plan['python'],str(ROOT/'source'/args[0]),*args[1:]],cwd=plan['snapshot'],env=env,stdout=log,stderr=subprocess.STDOUT)
            save(control/'stage.json',dict(stage=name,pid=os.getpid(),worker_pid=child.pid,gpu=2,time=time.time()))
            rc=child.wait();assert rc==0,(name,rc)
    try:
        run('native_gate',['worker.py','--gate','--gpu','2','--capacity','6'])
        for method in plan['new_methods']:
            claim=ROOT/'claims'/f'{method}.json';claim.parent.mkdir(exist_ok=True)
            with claim.open('x') as f:
                import json;json.dump(dict(pid=os.getpid(),host='scai6',gpu=2,method=method,time=time.time()),f)
            setting=plan['settings'][method]
            args=['worker.py','--capacity',str(setting['capacity']),'--gpu','2'] if setting['family']=='gca' else ['baseline_worker.py','--method',method]
            run(method,args,setting)
        save(ROOT/'generation_complete.json',dict(complete=True,new_videos=15,reused_method_videos=3,methods=plan['methods'],time=time.time()))
        save(control/'stage.json',dict(stage='generation_complete',gpu=2,time=time.time()))
    except BaseException:
        save(ROOT/'generation_failure.json',dict(error=traceback.format_exc(),time=time.time()));raise

if __name__=='__main__':main()
