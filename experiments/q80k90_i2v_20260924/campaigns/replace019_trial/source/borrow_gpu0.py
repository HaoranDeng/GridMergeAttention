"""Borrow only the user's GPU0 queue, then resume and verify its original work."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from worker import read,save,sha
from launch import state,clean
ROOT=Path(__file__).resolve().parents[1]
PRIVATE=ROOT/'borrow_private'
BASE=Path('/xuanwu-tank/east/zrl/exp-gridmergeattn')

def proc(pid,environment=False):
    p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split()
    d=dict(pid=pid,uid=p.stat().st_uid,ppid=int(s[1]),pgid=int(s[2]),sid=int(s[3]),start_ticks=int(s[19]),
           cwd=os.readlink(p/'cwd'),argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x])
    if environment:d['environment']=dict(x.decode().split('=',1) for x in (p/'environ').read_bytes().split(b'\0') if b'=' in x)
    return d

def verify_process(saved):
    p=proc(saved['pid']);assert p['uid']==os.getuid() and p['start_ticks']==saved['start_ticks'] and p['argv']==saved['argv']

def owner(path):return (path/'owner').read_text().split()

def lease(path,pid,expected):
    old=owner(path);assert old[:3]==['scai7',str(expected),'gpu0'],old
    t=path/'owner.i2v-tmp';t.write_text(f'scai7 {pid} gpu0 {datetime.datetime.now().astimezone().isoformat()}\n');os.replace(t,path/'owner')

def prepare():
    assert not PRIVATE.exists(),'Existing handoff must be inspected, never overwritten'
    s=state();assert len(s['pids'])==1 and s['uuid']=='GPU-4eca131f-0ba3-759d-bcb5-0418d6b7c68e',s
    g=proc(s['pids'][0],True);r=proc(g['ppid'],True);w=proc(r['ppid'],True)
    assert all(p['uid']==os.getuid() for p in (g,r,w))
    assert w['argv']==['bash','h200_run/125_worker.sh'] and w['cwd']==str(BASE)
    assert r['argv']==['bash','h200_run/116_paper_i2v.sh','50','q80k90_c8']
    assert all(p['pgid']==w['pid'] for p in (g,r,w)) and 'generate.py' in g['argv']
    assert r['environment']['CUDA_VISIBLE_DEVICES']=='0' and r['environment']['RESUME']=='1'
    args=g['argv'];value=lambda key:args[args.index(key)+1]
    out=Path(value('--output_dir'));rt=Path(value('--runtime_json'));tag=out.name
    assert out.parent==BASE/'h200_run/videos_p50i' and tag=='p50i_q80k90_c8'
    tag='n50_'+tag
    assert r['environment'].get('NSET')=='50'
    qdir=Path(w['environment'].get('QUEUE_DIR',str(BASE/'h200_run/.queue')));lock=qdir/f'lock_{tag}'
    assert owner(lock)[:3]==['scai7',str(w['pid']),'gpu0']
    cfg=Path(g['cwd'])/value('--config');assert cfg.is_file()
    PRIVATE.mkdir(mode=0o700);PRIVATE.chmod(0o700)
    d=dict(gpu=0,tag=tag,output=str(out),runtime=str(rt),queue_dir=str(qdir),lock=str(lock),config=str(cfg),config_sha256=sha(cfg),
           phase='prepared',preflight=s,prepared_at=time.time(),expected_videos=int(r['environment'].get('NSET','20')),
           completed=[dict(name=p.name,size=p.stat().st_size,sha256=sha(p)) for p in sorted(out.glob('*.mp4'))])
    for role,p in [('generation',g),('runner',r),('worker',w)]:
        env=p.pop('environment');path=PRIVATE/f'gpu0_{role}_environment.json';save(path,env);path.chmod(0o600);d[role]=p
    if rt.exists():save(PRIVATE/'runtime_before.json',read(rt))
    claim=BASE/'h200_run/.campaign_state/claim_gpu0';d['claim']=str(claim)
    d['claim_owned']=claim.exists() and claim.read_text().split()[0]==str(w['pid'])
    d['input_files']={str(Path(g['cwd'])/value(k)):sha(Path(g['cwd'])/value(k)) for k in ['--prompt_path','--image_list']}
    names=Path(value('--image_list')).read_text().splitlines()
    d['input_files'].update({str(Path(value('--image_dir'))/n):sha(Path(value('--image_dir'))/n) for n in names})
    save(PRIVATE/'handoff.json',d)
    save(ROOT/'borrow_status.json',dict(phase='prepared',gpu=0,tag=tag,completed_preserved=len(d['completed']),restoration_required=True))
    print(json.dumps(dict(prepared=True,gpu=0,tag=tag,original_pid=g['pid'],preserved=len(d['completed']))))

def preserved(d):
    return all(sha(Path(p))==h for p,h in d.get('input_files',{}).items()) and all((Path(d['output'])/p['name']).is_file() and sha(Path(d['output'])/p['name'])==p['sha256'] for p in d['completed'])

def verify_restoration():
    d=read(PRIVATE/'handoff.json');s=read(ROOT/'restoration.json') if (ROOT/'restoration.json').exists() else {}
    result=dict(verified=False,time=time.time(),gpu=0,tag=d['tag'],preserved=preserved(d),config_unchanged=sha(Path(d['config']))==d['config_sha256'])
    old=d['generation']['argv']
    for pid in state()['pids']:
        try:p=proc(pid,True)
        except (FileNotFoundError,PermissionError):continue
        if p['uid']!=os.getuid() or p['sid']!=s.get('pid') or 'generate.py' not in p['argv']:continue
        keys=['--config','--checkpoint','--prompt_path','--output_dir','--runtime_json','--num_videos','--num_samples','--seed','--image_list','--image_dir']
        matched=all(k in p['argv'] and p['argv'][p['argv'].index(k)+1]==old[old.index(k)+1] for k in keys)
        env=read(PRIVATE/'gpu0_generation_environment.json')
        matched=matched and all(p['environment'].get(k)==env.get(k) for k in ['SVG_SPARSITY','SVG_TOP_P','SVG2_TOP_P','SVG_EAR_TOP_P','SVG_KMEANS_INIT','SVG_KMEANS_STEP','NSET']) and p['environment'].get('CUDA_VISIBLE_DEVICES')=='0'
        result.update(actual_generation_pid=pid,arguments_match=matched,skip_existing='--skip_existing' in p['argv'])
        result['verified']=all(result.get(k) for k in ('preserved','config_unchanged','arguments_match','skip_existing'))
    if not result['verified'] and s.get('phase')=='returned_to_original_queue' and s.get('original_returncode')==0:
        rt=read(Path(d['runtime']));n=len(list(Path(d['output']).glob('*.mp4')))
        result.update(original_completed=n>=d['expected_videos'],completed_videos=n)
        result['verified']=result['preserved'] and result['config_unchanged'] and result['original_completed']
    save(ROOT/'restoration_verified.json',result);return result

def resume():
    d=read(PRIVATE/'handoff.json');lock=Path(d['lock'])
    lease(lock,os.getpid(),d['supervisor_pid'])
    if d['claim_owned']:Path(d['claim']).write_text(f'{os.getpid()} restoring gpu0\n')
    assert preserved(d) and sha(Path(d['config']))==d['config_sha256']
    env=read(PRIVATE/'gpu0_runner_environment.json');env['RESUME']='1';assert env['CUDA_VISIBLE_DEVICES']=='0'
    save(ROOT/'restoration.json',dict(phase='restarting_original',pid=os.getpid(),gpu=0,tag=d['tag'],time=time.time()))
    with (ROOT/'original_gpu0_resumed.log').open('a') as log:
        child=subprocess.Popen(d['runner']['argv'],cwd=d['runner']['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT)
        save(ROOT/'restoration.json',dict(phase='original_running',pid=os.getpid(),child_pid=child.pid,gpu=0,tag=d['tag'],time=time.time()))
        rc=child.wait()
    q=Path(d['queue_dir'])
    if rc==0:
        assert len(list(Path(d['output']).glob('*.mp4')))>=d['expected_videos']
        (q/f"done_{d['tag']}").touch();(q/f"fail_{d['tag']}").unlink(missing_ok=True)
    else:
        f=q/f"fail_{d['tag']}";f.write_text(str((int(f.read_text()) if f.exists() else 0)+1)+'\n')
    assert owner(lock)[:3]==['scai7',str(os.getpid()),'gpu0']
    assert set(p.name for p in lock.iterdir())=={'owner'}
    (lock/'owner').unlink();lock.rmdir()
    c=Path(d['claim'])
    if d['claim_owned'] and c.exists() and c.read_text().split()[0]==str(os.getpid()):c.unlink()
    save(ROOT/'restoration.json',dict(phase='returned_to_original_queue',pid=os.getpid(),original_returncode=rc,gpu=0,tag=d['tag'],time=time.time()))
    verify_restoration()
    w=d['worker'];os.chdir(w['cwd']);os.execvpe(w['argv'][0],w['argv'],read(PRIVATE/'gpu0_worker_environment.json'))

def run():
    lockfile=(ROOT/'borrow.lock').open('w');fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    d=read(PRIVATE/'handoff.json');assert d['phase']=='prepared' and not (ROOT/'launch.json').exists()
    for role in ('worker','runner','generation'):verify_process(d[role])
    assert state()['pids']==[d['generation']['pid']]
    d['supervisor_pid']=os.getpid();save(PRIVATE/'handoff.json',d)
    borrowed=False;stopped=False;guard=None;failure=None
    def interrupted(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        members=[]
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():continue
            try:member=proc(int(entry.name))
            except (FileNotFoundError,ProcessLookupError,PermissionError):continue
            if member['pgid']==d['worker']['pid']:
                assert member['uid']==os.getuid() and member['sid']==d['worker']['pid']
                members.append(member['pid'])
        assert set(d[r]['pid'] for r in ['worker','runner','generation']).issubset(members)
        os.killpg(d['worker']['pid'],signal.SIGSTOP);stopped=True
        d['completed']=[dict(name=p.name,size=p.stat().st_size,sha256=sha(p)) for p in sorted(Path(d['output']).glob('*.mp4'))]
        save(PRIVATE/'handoff.json',d)
        lease(Path(d['lock']),os.getpid(),d['worker']['pid']);borrowed=True
        if d['claim_owned']:Path(d['claim']).write_text(f'{os.getpid()} borrowed gpu0\n')
        try:os.killpg(d['worker']['pid'],signal.SIGKILL)
        except ProcessLookupError:pass
        d['phase']='original_stopped';save(PRIVATE/'handoff.json',d)
        for _ in range(60):
            s=state()
            if not s['pids'] and s['memory_mib']<100 and s['utilization']==0:break
            assert not set(s['pids'])-{d['generation']['pid']},s
            time.sleep(1)
        clean(state())
        with (ROOT/'launcher.log').open('a') as log:
            guard=subprocess.Popen([sys.executable,str(ROOT/'source/launch.py'),'--gpu','0'],stdout=log,stderr=subprocess.STDOUT)
            save(ROOT/'borrow_status.json',dict(phase='i2v_running',pid=os.getpid(),guard_pid=guard.pid,gpu=0,tag=d['tag'],completed_preserved=len(d['completed']),restoration_required=True,time=time.time()))
            rc=guard.wait()
        if rc:raise RuntimeError(f'I2V guard failed: {rc}')
    except BaseException as e:failure=repr(e)
    finally:
        if guard is not None and guard.poll() is None:
            guard.terminate();guard.wait(timeout=60)
        if stopped and not borrowed:os.killpg(d['worker']['pid'],signal.SIGCONT)
        if borrowed:
            with (ROOT/'restorer.log').open('a') as log:
                r=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'resume'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            save(ROOT/'borrow_status.json',dict(phase='restoring_original',gpu=0,tag=d['tag'],restorer_pid=r.pid,failure=failure,time=time.time()))
            for _ in range(180):
                result=verify_restoration()
                if result['verified']:break
                if r.poll() is not None:break
                time.sleep(5)
            save(ROOT/'borrow_status.json',dict(phase='original_restore_verified' if result['verified'] else 'restoration_needs_attention',gpu=0,tag=d['tag'],restorer_pid=r.pid,failure=failure,time=time.time()))
    if failure:raise RuntimeError(failure)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','resume','verify']);a=p.parse_args()
    if a.action=='verify':print(json.dumps(verify_restoration()))
    else:globals()[a.action]()
