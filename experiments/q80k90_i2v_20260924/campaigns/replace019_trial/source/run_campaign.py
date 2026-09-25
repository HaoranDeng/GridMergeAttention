"""Four single-input runs on the already exclusively leased GPU0."""
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from worker import ROOT,read,save

def main():
    plan=read(ROOT/'experiment_plan.json');failure=None
    work=[('native_gate',['worker.py','--gpu','0','--capacity','4','--gate']),
          ('Dense',['baseline_worker.py','--method','Dense']),('C4',['worker.py','--capacity','4','--gpu','0']),
          ('C8',['worker.py','--capacity','8','--gpu','0']),('EAR_p08',['baseline_worker.py','--method','EAR_p08'])]
    try:
        for name,args in work:
            save(ROOT/'stage.json',dict(stage=name,status='running',pid=os.getpid(),time=time.time()))
            with (ROOT/(name+'.log')).open('w') as log:
                child=subprocess.Popen([plan['python'],str(ROOT/'source'/args[0]),*args[1:]],cwd=plan['snapshot'],stdout=log,stderr=subprocess.STDOUT)
                save(ROOT/'stage.json',dict(stage=name,status='running',pid=os.getpid(),worker_pid=child.pid,time=time.time()))
                rc=child.wait()
            assert rc==0,(name,rc)
        save(ROOT/'generation_complete.json',dict(complete=True,methods=plan['methods'],videos=4,time=time.time()))
    except BaseException:
        failure=traceback.format_exc();save(ROOT/'generation_failure.json',dict(error=failure,time=time.time()))
        raise
    finally:
        save(ROOT/'stage.json',dict(stage='generation_finished',status='failed' if failure else 'complete',time=time.time()))
        if (ROOT/'Dense/generation_complete.json').exists():
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            with (ROOT/'quality.log').open('a') as log:
                child=subprocess.Popen([plan['evaluation_python'],str(ROOT/'source/evaluate.py')],cwd=ROOT,env=env,
                     stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            save(ROOT/'evaluation_launch.json',dict(pid=child.pid,time=time.time(),device='cpu'))

if __name__=='__main__':main()
