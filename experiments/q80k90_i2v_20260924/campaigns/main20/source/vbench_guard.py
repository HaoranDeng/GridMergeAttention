"""Exclusive, bounded six-dimension I2V evaluation; no generator/kernel changes."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from worker import read,save,sha

MAIN=Path(__file__).resolve().parents[1]
ROOT=MAIN/'vbench20_20260924'
BASE=Path('/xuanwu-tank/east/zrl/exp-gridmergeattn')
VB=BASE/'third_party/VBench'
PYTHON='/home/zrlu/miniforge3/envs/vbench/bin/python'
DIMS=['subject_consistency','background_consistency','motion_smoothness','dynamic_degree','aesthetic_quality','imaging_quality']
UUIDS={2:'GPU-244e87b8-5a10-aadb-858a-f0eea028dbd1',5:'GPU-60b9a1cc-32ad-7b4d-e8eb-7fc32ed5ba3c'}
NORM={'subject_consistency':(.1462,1.),'background_consistency':(.2615,1.),'motion_smoothness':(.706,.9975),
      'dynamic_degree':(0.,1.),'aesthetic_quality':(0.,1.),'imaging_quality':(0.,1.)}

def state(gpu):
    row=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
    rows=csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines())
    return dict(uuid=row[0].strip(),model=row[1].strip(),memory_mib=int(row[2]),utilization=int(row[3]),
                pids=[int(r[1]) for r in rows if r and r[0].strip()==row[0].strip()])

def clean(gpu):
    s=state(gpu);assert s['uuid']==UUIDS[gpu] and s['model']=='NVIDIA H200'
    assert not s['pids'] and s['memory_mib']<100 and s['utilization']==0,s
    return s

def prepare():
    ROOT.mkdir(exist_ok=True);assert not (ROOT/'plan.json').exists()
    main=read(MAIN/'experiment_plan.json')
    paths={'C4':MAIN/'C4/video20','C6':MAIN/'C6/video20','C8':Path(main['existing_c8']['root']),
           'SVG_EAR_p08':BASE/'h200_run/videos_p50i/p50i_svg_ear_p08'}
    for cap in [4,6,8]:assert len(read(MAIN/f'C{cap}/quality_cpu/quality.json')['videos'])==20
    jobs={}
    for method,src in paths.items():
        out=ROOT/method;(out/'videos').mkdir(parents=True)
        records=[]
        for i in range(20):
            p=src/f'{i:03d}-00.mp4';assert p.is_file() and p.stat().st_size>0
            (out/'videos'/p.name).symlink_to(p)
            records.append(dict(index=i,path=str(p),sha256=sha(p)))
        jobs[method]=dict(source=str(src),videos=records)
    files=[VB/'evaluate.py',VB/'vbench/__init__.py',VB/'vbench/utils.py',VB/'vbench/distributed.py',
           *[VB/f'vbench/{d}.py' for d in DIMS],BASE/'h200_run/137_vbench_table.sh',BASE/'h200_run/make_vbench_table.py']
    save(ROOT/'plan.json',dict(protocol='Existing six-dimension custom_input I2V protocol; first20, full videos, longer imaging preprocessing.',
         dimensions=DIMS,python=PYTHON,assignments={'2':['C4','C6','C8','SVG_EAR_p08']},jobs=jobs,
         source_sha256={str(p):sha(p) for p in files},normalization=NORM,dynamic_degree_weight=.5,total_weight=5.5,
         note='Six-dimension quality composite, not a leaderboard score. CLIP download directory changes storage only.',
         input_manifest=main['input_manifest'],created=time.time()))
    print(json.dumps({'prepared':True,'methods':list(jobs),'videos_per_method':20}))

def worker(method):
    """Run the existing evaluator with only checkpoint-cache location redirected."""
    import runpy
    sys.path.insert(0,str(VB));os.chdir(VB)
    import clip
    original=clip.load
    def load(*args,**kwargs):
        kwargs.setdefault('download_root',str(ROOT/'clip_cache'))
        return original(*args,**kwargs)
    clip.load=load
    sys.argv=['evaluate.py','--videos_path',str(ROOT/method/'videos'),'--output_path',str(ROOT/method/'out'),
              '--mode','custom_input','--dimension',*DIMS]
    runpy.run_path(str(VB/'evaluate.py'),run_name='__main__')

def run(gpu):
    assert socket.gethostname().split('.')[0]=='scai6'
    control=ROOT/f'gpu{gpu}';control.mkdir(exist_ok=True)
    lock=(control/'guard.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (control/'launch.json').exists(),'Existing VBench run: inspect, never duplicate'
    plan=read(ROOT/'plan.json')
    for p,h in plan['source_sha256'].items():assert sha(Path(p))==h,p
    s=clean(gpu);time.sleep(1);clean(gpu)
    # Check that an old generation queue does not own this otherwise empty card.
    for p in (BASE/'h200_run/.queue').glob('lock_*/owner'):
        fields=p.read_text().split()
        if fields[:1]==['scai6'] and len(fields)>2 and fields[2]==f'gpu{gpu}':
            assert not Path('/proc',fields[1]).exists(),str(p)
    save(control/'launch.json',dict(pid=os.getpid(),gpu=gpu,host='scai6',preflight=s,time=time.time()))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUIDS[gpu],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
             XDG_CACHE_HOME=str(BASE/'.cache/environments/vbench'),VBENCH_CACHE_DIR=str(ROOT/'model_cache'),
             MASTER_ADDR='127.0.0.1',MASTER_PORT=str(29620+gpu),WORLD_SIZE='1',RANK='0',LOCAL_RANK='0',PYTHONUNBUFFERED='1')
    child=None;failure=None
    def interrupted(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for method in plan['assignments'][str(gpu)]:
            for row in plan['jobs'][method]['videos']:assert sha(Path(row['path']))==row['sha256']
            clean(gpu)
            out=ROOT/method
            assert not list((out/'out').glob('*_eval_results.json'))
            with (out/'evaluate.log').open('w') as log,(control/'monitor.jsonl').open('a') as monitor:
                child=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'worker','--method',method],cwd=VB,env=env,
                                       stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                save(control/'stage.json',dict(method=method,pid=os.getpid(),worker_pid=child.pid,gpu=gpu,time=time.time()))
                start=time.monotonic()
                while True:
                    s=state(gpu);foreign=[]
                    for pid in s['pids']:
                        try:
                            if os.getsid(pid)!=child.pid:foreign.append(pid)
                        except ProcessLookupError:pass
                    monitor.write(json.dumps(dict(time=time.time(),foreign_pids=foreign,**s))+'\n');monitor.flush()
                    assert not foreign,('Foreign GPU process; stop only own evaluator',foreign)
                    if child.poll() is not None:
                        assert child.returncode==0,(method,child.returncode)
                        break
                    assert time.monotonic()-start<6*3600,'Bounded six-hour evaluator deadline'
                    time.sleep(5)
            result,=list((out/'out').glob('*_eval_results.json'));data=read(result)
            assert set(data)==set(DIMS),list(data)
            raw={d:float(data[d][0]) for d in DIMS}
            assert all(len(data[d][1])==20 for d in DIMS)
            norm={d:(raw[d]-NORM[d][0])/(NORM[d][1]-NORM[d][0]) for d in DIMS}
            composite=sum(norm[d]*(.5 if d=='dynamic_degree' else 1) for d in DIMS)/5.5
            for row in plan['jobs'][method]['videos']:assert sha(Path(row['path']))==row['sha256']
            save(out/'summary.json',dict(complete=True,method=method,videos=20,gpu=gpu,host='scai6',raw=raw,normalized=norm,
                 six_dimension_quality_score=composite,source=str(result),source_sha256=sha(result),time=time.time()))
    except BaseException:
        failure=traceback.format_exc();save(control/'failure.json',dict(error=failure,time=time.time()))
        raise
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        save(control/'completion.json',dict(valid=failure is None,failure=failure,gpu=gpu,time=time.time(),restoration_required=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','worker']);p.add_argument('--gpu',type=int,choices=[2]);p.add_argument('--method',choices=['C4','C6','C8','SVG_EAR_p08']);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='run':run(a.gpu)
    else:worker(a.method)
