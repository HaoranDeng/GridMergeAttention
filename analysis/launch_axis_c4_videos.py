"""Launch one finite five-video job on a currently idle GPU, never queue it."""
import argparse
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
    p.add_argument('--block-j',required=True,type=int)
    p.add_argument('--validation',required=True,type=Path)
    p.add_argument('--attempt',default='v1')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    base=Path('/local2/denghaoran/GridMergeAttention/experiments/axis_video_c4_20260918')
    result=base/f'q{a.axes[0]}_k{a.axes[1]}_c4_current'
    guard=base/f'run_{a.axes}_{a.attempt}'
    validation=json.loads(a.validation.read_text())
    assert validation['passed'] and validation['axes']==a.axes
    state=gpu_state(a.gpu)
    if state['pids'] or state['memory_mib']>100 or state['utilization']:
        raise RuntimeError(f'GPU busy; not launching or queuing: {state}')
    launch=base/f'launch_{a.axes}_{a.attempt}'; launch.mkdir(exist_ok=False)
    with (launch/'prompt_check.log').open('w') as log:
        subprocess.run([sys.executable,'-m','analysis.verify_prompt_contract',
                        '--prompt_path',str(root/'prompts/MovieGenVideoBench_extended.txt'),
                        '--num_videos','5','--seed','0','--manifest_out',str(launch/'prompt_manifest.json')],
                       cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    command=[sys.executable,'-m','analysis.guard_axis_job','--gpu',str(a.gpu),'--output',str(guard),'--',
             sys.executable,'-m','analysis.run_axis_c4_videos','--axes',a.axes,'--output',str(result),
             '--validation',str(a.validation),'--reference',str(base/'dense_first5')]
    env=dict(os.environ,AXIS_C4_BLOCK_J=str(a.block_j),AXIS_C4_FISHER_FP64='0',AXIS_C4_PRECOMPUTE='0')
    with (launch/'supervisor.log').open('w') as log:
        child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,
                               stderr=subprocess.STDOUT,start_new_session=True)
    save(launch/'launch.json',{'pid':child.pid,'axes':a.axes,'gpu':state,'command':command,
         'source':str(root),'source_sha256':{str(f.relative_to(root)):sha(f) for f in
          [root/'attention/axis_current_c4_triton.py',root/'analysis/run_axis_c4_videos.py',
           root/'kernels/best/c4/route_reference.py',root/'analysis/benchmark_wan14b_720p_methods.py']}})
    print(json.dumps({'pid':child.pid,'guard':str(guard),'output':str(result)}))


if __name__=='__main__':main()
