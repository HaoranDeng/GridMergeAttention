"""Audited recovery on the same clean GPU; preserve all interruption evidence."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from worker import ROOT,read,save,sha,verify,validate_resume
from supervisor import clean,state


def main():
    lock=(ROOT/'recovery.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'resume_plan.json').exists(),'Recovery already dispatched; inspect before retry'
    original=read(ROOT/'experiment_plan.json');assignment=read(ROOT/'parallel_plan.json')
    assert assignment['original_plan_sha256']==sha(ROOT/'experiment_plan.json')
    plan=dict(original,gpu=assignment['gpu']);gpu=plan['gpu'];verify(plan)
    control=ROOT/'parallel'
    assert (control/'failure.json').exists() and not (ROOT/'failure.json').exists()
    for pid in (read(control/'launch.json')['supervisor'],read(control/'stage.json')['pid']):
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError(f'Previous C4 process still exists: {pid}')
    stage=read(ROOT/'stage.json');assert stage['stage']=='Dense'
    os.kill(stage['pid'],0)
    assert stage['pid'] in state(original['gpu'])['pids']
    first=state(gpu);clean(first,gpu);time.sleep(2);second=state(gpu);clean(second,gpu)
    purity=validate_resume(plan,ROOT/'C4/videos')
    gate=read(ROOT/'native_gate.json')
    assert gate['passed'] and gate['complete'] and gate['gpu']==gpu
    # Retain only videos saved before the last clean sample preceding contention.
    last_clean=None;first_foreign=None
    with (control/'monitor.jsonl').open() as f:
        for line in f:
            row=json.loads(line)
            if row['foreign_pids']:
                first_foreign=row;break
            last_clean=row
    assert first_foreign is not None and last_clean is not None
    assert all(row['saved_at']<last_clean['time'] for row in purity)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive=ROOT/f'parallel_interrupted_{stamp}'
    assert not archive.exists()
    control.rename(archive);control.mkdir()
    # Cached kernels remain byte-for-byte identical and are reused on this H200.
    for name in ('torch_extensions','triton_cache'):
        if (archive/name).exists():(control/name).symlink_to(archive/name,target_is_directory=True)
    recovery=dict(created_at=time.time(),reason='Foreign process ended; same GPU is exclusively idle again',
                  archive=str(archive),gpu=gpu,dense_pid_unchanged=stage['pid'],verified_existing_videos=len(purity),
                  next_index=len(purity),preserved_video_hashes={r['video']:r['video_sha256'] for r in purity},
                  original_plan_sha256=sha(ROOT/'experiment_plan.json'),native_gate_sha256=sha(ROOT/'native_gate.json'),
                  last_clean_before_contention=last_clean,first_foreign_check=first_foreign,
                  fresh_preflight=[first,second],source_sha256={p.name:sha(p) for p in (ROOT/'source').glob('*.py')},
                  timing_scope='Completed prefix was saved before the last clean check. Interrupted unsaved work has no runtime result.')
    save(ROOT/'resume_plan.json',recovery)
    with (control/'supervisor.log').open('w') as log:
        proc=subprocess.Popen([plan['evaluation_python'],str(ROOT/'source/parallel_supervisor.py'),'--resume'],cwd=ROOT,
             env=dict(os.environ,PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(control/'dispatch.json',dict(pid=proc.pid,time=time.time(),gpu=gpu,resume=True))
    for _ in range(30):
        assert proc.poll() is None,(control/'supervisor.log').read_text()[-3000:]
        if (control/'stage.json').exists():break
        time.sleep(1)
    print(json.dumps(dict(resumed=True,next_index=len(purity),supervisor=proc.pid,dense_pid_unchanged=stage['pid'],
                         stage=read(control/'stage.json') if (control/'stage.json').exists() else 'verifying_sources')))


if __name__=='__main__':
    main()
