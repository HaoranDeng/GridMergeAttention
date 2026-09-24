"""Own C6 once, run two guarded shards, publish committed results for CPU scoring."""
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from worker import read, save, sha
from launch_scai6 import state, clean

ROOT=Path(__file__).resolve().parents[1]
GPUS=(2,5)

def publish():
    target=ROOT/'C6/video20';target.mkdir(parents=True,exist_ok=True)
    records=[];proofs=[];summaries={};config=None
    for gpu in GPUS:
        shard=ROOT/f'C6/shards/scai6_gpu{gpu}/video20'
        if not (shard/'runtime.json').exists():continue
        runtime=read(shard/'runtime.json');purity=read(shard/'purity_records.json')
        byname={p['video']:p for p in purity};config=runtime.get('config')
        for row in runtime['records']:
            i=row['prompt_index'];name=row['output']
            assert i in range({2:0,5:1}[gpu],20,2) and name==f'{i:03d}-00.mp4'
            proof=byname[name];assert proof['attention_calls']==4000 and proof['layer_calls']==[100]*40
            assert proof['dense_self_attention_calls']==0 and proof['full_50_step_pflops']==84.678967296
            src=shard/name;dst=target/name
            if not dst.exists():
                assert sha(src)==proof['video_sha256']
                os.link(src,dst)
            else:assert os.path.samefile(src,dst), 'Unexpected duplicate C6 output'
            records.append(dict(row,gpu=proof['gpu']));proofs.append(proof)
        summaries[f'scai6_gpu{gpu}']=dict(host='scai6',gpu=gpu,**runtime['summary'])
    records.sort(key=lambda r:r['prompt_index']);proofs.sort(key=lambda p:p['video'])
    assert len({r['prompt_index'] for r in records})==len(records)
    # The CPU scorer observes runtime last: every visible row has a committed
    # video and its matching purity record. Timings are never averaged across GPUs.
    save(target/'purity_records.json',proofs)
    save(target/'runtime.json',dict(config=config,checkpoint='pretrained',records=records,
         summary=dict(videos=len(records),attention_calls_per_video=4000,by_gpu=summaries,timing_aggregation='not_pooled_across_gpus')))
    if len(records)==20 and all((ROOT/f'scai6_gpu{g}/generation_complete.json').exists() for g in GPUS):
        assert [r['prompt_index'] for r in records]==list(range(20))
        save(ROOT/'C6/generation_complete.json',dict(complete=True,videos=20,host='scai6',gpus=list(GPUS),time=time.time()))
    return len(records)

def main():
    assert socket.gethostname().split('.')[0]=='scai6'
    control=ROOT/'scai6';control.mkdir(exist_ok=True)
    lock=(control/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (control/'launch.json').exists(), 'Already dispatched'
    base=Path('/xuanwu-tank/east/zrl/exp-gridmergeattn/h200_run')
    for gpu in GPUS:
        assert not (ROOT/f'scai6_gpu{gpu}/launch.json').exists()
        clean(state(gpu),gpu)
        # Shared claims must be diagnosed rather than removed.
        assert not (base/f'.campaign_state/claim_gpu{gpu}').exists()
    for owner in (base/'.queue').glob('lock_*/owner'):
        contents=owner.read_text()
        assert not (contents.startswith('scai6 ') and any(f'gpu{g}' in contents for g in GPUS)), contents
    time.sleep(2)
    for gpu in GPUS:clean(state(gpu),gpu)
    claim=ROOT/'claims/C6.json'
    fd=os.open(claim,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
    with os.fdopen(fd,'w') as f:
        json.dump(dict(capacity=6,host='scai6',gpu=list(GPUS),gpus=list(GPUS),pid=os.getpid(),time=time.time(),
                       shards={'2':list(range(0,20,2)),'5':list(range(1,20,2))}),f)
    children={};failure=None
    def interrupted(sig,frame):raise KeyboardInterrupt(f'signal {sig}')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for gpu in GPUS:
            directory=ROOT/f'scai6_gpu{gpu}';directory.mkdir(exist_ok=True)
            with (directory/'launcher.log').open('w') as log:
                children[gpu]=subprocess.Popen([sys.executable,str(ROOT/'source/launch_scai6.py'),'--gpu',str(gpu)],cwd=ROOT,
                                                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(control/'launch.json',dict(supervisor=os.getpid(),children={str(g):p.pid for g,p in children.items()},time=time.time()))
        deadline=time.monotonic()+72*3600
        while time.monotonic()<deadline:
            count=publish()
            states={str(g):p.poll() for g,p in children.items()}
            save(control/'supervisor_status.json',dict(pid=os.getpid(),host='scai6',gpus=list(GPUS),generated=count,expected=20,
                                                      child_exit_codes=states,time=time.time()))
            if all(p.poll() is not None for p in children.values()):
                assert all(p.returncode==0 for p in children.values()), states
                assert count==20 and (ROOT/'C6/generation_complete.json').exists()
                break
            time.sleep(5)
        else:raise TimeoutError('72-hour deadline')
    except BaseException as e:failure=repr(e)
    finally:
        for child in children.values():
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    # The card guard owns its generation session. Save the failure
                    # and leave identity evidence for explicit recovery if it cannot exit.
                    failure=f'{failure}; guard {child.pid} did not exit after TERM'
        save(control/'completion.json',dict(valid=failure is None,failure=failure,time=time.time(),restoration_required=False))
    if failure:raise RuntimeError(failure)

if __name__=='__main__':main()
