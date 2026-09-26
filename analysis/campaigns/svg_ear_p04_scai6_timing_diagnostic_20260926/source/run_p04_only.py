"""Preserve the running p=.4 worker and cancel every p=.5 case by user request."""
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback
from common import ROOT,read,save
from gpu_checks import inventory,clean
from run import telemetry


def alive(pid):
    try:
        state=(Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()[0]
        return state not in ('Z','X')
    except FileNotFoundError:return False


def verify(pid,needle):
    path=Path('/proc')/str(pid)
    assert path.stat().st_uid==os.getuid()
    cmd=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert str(ROOT/'source') in cmd and needle in cmd,(pid,cmd)


def stop_group(pid):
    assert os.getsid(pid)==pid
    os.killpg(pid,signal.SIGTERM)
    os.killpg(pid,signal.SIGCONT)
    until=time.monotonic()+30
    while alive(pid) and time.monotonic()<until:time.sleep(.3)
    if alive(pid):os.killpg(pid,signal.SIGKILL)


def main():
    assert socket.gethostname().split('.')[0]=='scai6'
    lock=(ROOT/'p04_only.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'p04_only_launch.json').exists(),'Already handed off'
    plan=read(ROOT/'experiment_plan.json')
    old=read(ROOT/'launch.json')['pid']
    p04=read(ROOT/'gpu2_p04/launch.json')['pid']
    p05=read(ROOT/'gpu4_p05/launch.json')['pid']
    verify(old,'/run.py');verify(p04,'--case gpu2_p04');verify(p05,'--case gpu4_p05')
    assert os.getsid(p04)==p04 and alive(p04)
    save(ROOT/'p04_only_handoff_started.json',dict(time=time.time(),new_supervisor=os.getpid(),old_supervisor=old,preserved_p04=p04,canceled_p05=p05))
    # Freeze the old supervisor before ending p=.5, so its failure handler
    # cannot terminate the retained p=.4 worker or launch queued p=.5 work.
    os.kill(old,signal.SIGSTOP)
    stop_group(p05)
    assert not alive(p05)
    verify(old,'/run.py');os.kill(old,signal.SIGKILL)
    until=time.monotonic()+10
    while alive(old) and time.monotonic()<until:time.sleep(.2)
    assert not alive(old) and alive(p04)
    save(ROOT/'gpu4_p05/canceled.json',dict(time=time.time(),reason='User explicitly canceled p=.5',pid=p05,excluded_from_results=True))
    save(ROOT/'gpu2_p05/canceled.json',dict(time=time.time(),reason='User explicitly canceled queued p=.5',never_started=True,excluded_from_results=True))
    save(ROOT/'scope_update.json',dict(time=time.time(),user_request='p=.4 only on both idle scai6 cards',
        active_cases=['gpu2_p04','gpu4_p04'],canceled_cases=['gpu4_p05','gpu2_p05'],
        preserved_gpu2_worker=p04,old_supervisor_exited=True,p05_worker_exited=True,
        samples_per_gpu=5,warmup_index=0,steady_indices=[1,2,3,4],status='p04_only'))
    gpu=plan['gpus']['4'];state,=[g for g in inventory() if g['uuid']==gpu['uuid']]
    assert clean(state),state
    time.sleep(2);state,=[g for g in inventory() if g['uuid']==gpu['uuid']];assert clean(state),state
    folder=ROOT/'gpu4_p04';folder.mkdir(exist_ok=True)
    save(folder/'preflight.json',dict(time=time.time(),**state))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],SVG_TOP_P='0.4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1',MAX_JOBS='4',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        WAN_MODEL_ROOT=plan['protocol']['model_root'],GRIDMERGE_SDPA_BACKEND='flash_attn2',SPARSE_VIDEO_GEN_ROOT=plan['source_svg_root'],
        SVG_NUM_Q_CENTROIDS='300',SVG_NUM_K_CENTROIDS='1000',SVG_MIN_KC_RATIO='0.10',SVG_KMEANS_INIT='50',SVG_KMEANS_STEP='2',
        FLASHINFER_WORKSPACE_BASE='/xuanwu-tank/east/zrl/exp-gridmergeattn/.cache/environments/monarch-rt/flashinfer_workspace',
        TORCH_EXTENSIONS_DIR=str(ROOT/'gpu4/torch_extensions'),TRITON_CACHE_DIR=str(ROOT/'gpu4/triton_cache'),
        PATH='/usr/local/cuda/bin:'+os.environ['PATH'])
    with (folder/'worker.log').open('x') as log:
        child=subprocess.Popen([plan['python'],str(ROOT/'source/worker.py'),'--case','gpu4_p04'],cwd=ROOT,
            env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(folder/'launch.json',dict(pid=child.pid,case='gpu4_p04',top_p=.4,gpu=gpu,time=time.time(),phase='p04_only'))
    jobs={'gpu2_p04':p04,'gpu4_p04':child.pid}
    save(ROOT/'p04_only_launch.json',dict(pid=os.getpid(),jobs=jobs,time=time.time(),host='scai6'))
    failure=None;started=time.monotonic()
    try:
        with (ROOT/'monitor_p04_only.jsonl').open('x') as monitor:
            while True:
                states=inventory();foreign=[];finished=[]
                for name,pid in jobs.items():
                    if not alive(pid):
                        assert (ROOT/name/'generation_complete.json').exists(),f'{name} exited without completion'
                        assert read(ROOT/name/'generation_complete.json')['videos']==5
                        finished.append(name);continue
                    gpu=plan['cases'][name]['gpu'];state,=[g for g in states if g['uuid']==gpu['uuid']]
                    for current in state['pids']:
                        try:
                            if os.getsid(current)!=pid:foreign.append(dict(gpu=gpu['index'],pid=current))
                        except ProcessLookupError:pass
                row=dict(time=time.time(),phase='p04_only',foreign=foreign,finished=finished,
                    telemetry=telemetry(),load_average=os.getloadavg())
                monitor.write(json.dumps(row)+'\n');monitor.flush();save(ROOT/'latest_monitor.json',row)
                assert not foreign,foreign
                save(ROOT/'stage_p04_only.json',dict(phase='complete' if len(finished)==2 else 'p04_only',
                    jobs=jobs,finished=finished,time=time.time()))
                if len(finished)==2:break
                assert time.monotonic()-started<8*3600,'8-hour deadline'
                time.sleep(5)
    except BaseException:
        failure=traceback.format_exc();save(ROOT/'failure_p04_only.json',dict(error=failure,time=time.time()))
    finally:
        for pid in jobs.values():
            if alive(pid):verify(pid,'--case gpu');stop_group(pid)
        save(ROOT/'completion_p04_only.json',dict(valid=failure is None,failure=failure,time=time.time(),
            active_cases=list(jobs),canceled_p05_excluded=True,restoration_required=False))
    if failure:raise RuntimeError(failure)


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda signum,frame:(_ for _ in ()).throw(KeyboardInterrupt(f'signal {signum}')))
    try:main()
    except BaseException:
        save(ROOT/'handoff_failure.json',dict(error=traceback.format_exc(),time=time.time()))
        raise
