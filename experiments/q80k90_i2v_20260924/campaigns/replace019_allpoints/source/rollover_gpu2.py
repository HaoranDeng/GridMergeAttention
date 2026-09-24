"""Finish the active GPU2 unit, then replace its single-card dispatcher safely."""
import os,signal,subprocess,time,traceback
from pathlib import Path
from worker import ROOT,read,save,sha

def alive(pid):
    p=Path('/proc',str(pid))
    try:return p.exists() and (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False

def owned(pid,needle):
    p=Path('/proc',str(pid));assert p.stat().st_uid==os.getuid()
    assert needle.encode() in (p/'cmdline').read_bytes()

def main():
    assert not (ROOT/'rollover_gpu2.json').exists()
    old=ROOT/'gpu2';launch=read(old/'launch.json');controller=launch['child'];guard=launch['supervisor']
    owned(controller,'run_campaign.py');owned(guard,'launch.py')
    os.kill(controller,signal.SIGSTOP)
    stage=read(old/'stage.json');worker=stage['worker_pid']
    save(ROOT/'rollover_gpu2.json',dict(phase='waiting_current_unit',pid=os.getpid(),controller=controller,guard=guard,worker=worker,stage=stage['stage'],time=time.time()))
    try:
        started=time.monotonic()
        while alive(worker):
            # A zombie child has completed; its paused parent cannot reap it yet.
            stat=Path('/proc',str(worker),'stat')
            if stat.exists() and stat.read_text().rsplit(')',1)[1].split()[0]=='Z':break
            assert time.monotonic()-started<3600,'Current unit did not finish within one hour'
            time.sleep(5)
        gate=read(old/'native_gate.json');assert gate['passed'] and gate['complete']
        if stage['stage']!='native_gate':assert read(ROOT/stage['stage']/'generation_complete.json')['complete']
        os.kill(controller,signal.SIGKILL)
        for _ in range(20):
            if not alive(guard):break
            time.sleep(1)
        assert not alive(guard),'Old guard has not exited'
        history=ROOT/'recovery_history/gpu2_multicard_handoff';history.mkdir(parents=True,exist_ok=True)
        for name in ['launch.json','stage.json','completion.json','native_gate.json']:
            p=old/name
            if p.exists():(history/name).write_bytes(p.read_bytes())
        evaluator=read(ROOT/'evaluation_launch.json')['pid']
        if alive(evaluator):
            owned(evaluator,str(ROOT/'source/evaluate.py'));os.kill(evaluator,signal.SIGTERM)
            for _ in range(10):
                if not alive(evaluator):break
                time.sleep(1)
            assert not alive(evaluator)
        for name in ['evaluation_launch.json','evaluation_waiting.json']:
            p=ROOT/name
            if p.exists():(history/name).write_bytes(p.read_bytes());p.unlink()
        control=ROOT/'scai6_gpu2';control.mkdir(exist_ok=True);save(control/'native_gate.json',gate)
        plan=read(ROOT/'experiment_plan.json')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        with (ROOT/'quality_multi.log').open('a') as log:
            ev=subprocess.Popen([plan['evaluation_python'],str(ROOT/'source/evaluate_multi.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(ROOT/'evaluation_launch.json',dict(pid=ev.pid,host='scai6',device='cpu',time=time.time()))
        with (control/'launcher.log').open('a') as log:
            new=subprocess.Popen([plan['evaluation_python'],str(ROOT/'source/launch_multi.py'),'--slot','scai6_gpu2'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(ROOT/'rollover_gpu2.json',dict(phase='complete',new_guard=new.pid,evaluator=ev.pid,preserved_gate_sha256=sha(control/'native_gate.json'),time=time.time()))
    except BaseException:
        save(ROOT/'rollover_gpu2_failure.json',dict(error=traceback.format_exc(),time=time.time()))
        try:os.kill(controller,signal.SIGKILL)
        except ProcessLookupError:pass
        raise

if __name__=='__main__':main()
