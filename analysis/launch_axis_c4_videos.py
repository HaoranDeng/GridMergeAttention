"""Launch twenty videos once this method's full operator beats dense."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from analysis.quick_axis_c4_qkv import gpu_state, save, sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--axes',required=True,choices=['tw','wh','th'])
    p.add_argument('--gpu',required=True,type=int)
    p.add_argument('--speed-gate',required=True,type=Path)
    p.add_argument('--validation',required=True,type=Path)
    p.add_argument('--attempt',required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    base=Path('/local2/denghaoran/GridMergeAttention/experiments/axis_video_c4_20260918')
    result=base/f'q{a.axes[0]}_k{a.axes[1]}_c4_current_{a.attempt}_20'
    guard=base/f'run_{a.axes}_{a.attempt}'
    validation=json.loads(a.validation.read_text())
    assert validation['passed'] and validation['axes']==a.axes
    from analysis.axis_c4_speed_gate import verify_gate
    gate=verify_gate(a.speed_gate,root,a.axes)
    state=gpu_state(a.gpu)
    if state['pids'] or state['memory_mib']>100 or state['utilization']:
        raise RuntimeError(f'GPU busy; not launching or queuing: {state}')
    launch=base/f'launch_{a.axes}_{a.attempt}'; launch.mkdir(exist_ok=False)
    with (launch/'prompt_check.log').open('w') as log:
        subprocess.run([sys.executable,'-m','analysis.verify_prompt_contract',
                        '--prompt_path',str(root/'prompts/MovieGenVideoBench_extended.txt'),
                        '--num_videos','20','--seed','0','--manifest_out',str(launch/'prompt_manifest.json')],
                       cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    command=[sys.executable,'-m','analysis.guard_axis_job','--gpu',str(a.gpu),'--output',str(guard),'--',
             sys.executable,'-m','analysis.run_axis_c4_videos','--axes',a.axes,'--output',str(result),
             '--validation',str(a.validation),'--reference',str(base/'dense_first20'),
             '--speed-gate',str(a.speed_gate)]
    env=dict(os.environ,**gate['methods'][a.axes]['environment'])
    with (launch/'supervisor.log').open('w') as log:
        child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,
                               stderr=subprocess.STDOUT,start_new_session=True)
    save(launch/'launch.json',{'pid':child.pid,'axes':a.axes,'gpu':state,'command':command,
         'source':str(root),'source_sha256':{str(f.relative_to(root)):sha(f) for f in
          [root/'analysis/axis_c4_speed_gate.py',root/'analysis/run_axis_c4_videos.py',
           root/'kernels/best/c4/route_reference.py',root/'analysis/benchmark_wan14b_720p_methods.py']}})
    with (base/'active_video_jobs.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        registry=base/'active_video_jobs.json'
        active=json.loads(registry.read_text()) if registry.exists() else {}
        active[a.axes]={'output':str(result),'guard':str(guard),'launch':str(launch),'expected_videos':20}
        save(registry,active)
    print(json.dumps({'pid':child.pid,'guard':str(guard),'output':str(result)}))


if __name__=='__main__':main()
