"""Launch only on a currently clean H200, without a waiting queue."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from analysis.quick_axis_c4_qkv import gpu_state,save
from analysis.axis_c8_speed_gate import verify_gate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--attempt',required=True)
    p.add_argument('--speed-gate',type=Path,required=True)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=Path('/local2/denghaoran/GridMergeAttention/experiments/axis_video_c4_20260918/dense_first20'))
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    gate=verify_gate(a.speed_gate)
    validation=json.loads(a.validation.read_text())
    assert validation['passed'] and validation['source_sha256']==gate['source_sha256']
    assert validation['environment']==gate['environment']
    state=gpu_state(a.gpu)
    if 'H200' not in state['name'] or state['pids'] or state['memory_mib']>100 or state['utilization']:
        raise RuntimeError(f'GPU not clean: no launch or queue. {state}')
    launch=a.base/f'launch_{a.attempt}';launch.mkdir(parents=True,exist_ok=False)
    with (launch/'prompt_check.log').open('w') as log:
        subprocess.run([sys.executable,'-m','analysis.verify_prompt_contract','--prompt_path',
                        str(root/'prompts/MovieGenVideoBench_extended.txt'),'--num_videos','20',
                        '--seed','0','--manifest_out',str(launch/'prompt_manifest.json')],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    guard=a.base/f'run_{a.attempt}';output=a.base/f'qw_kh_c8_current_{a.attempt}_20'
    command=[sys.executable,'-m','analysis.guard_axis_job','--gpu',str(a.gpu),'--output',str(guard),'--',
             sys.executable,'-m','analysis.run_axis_c8_videos','--axes','wh','--output',str(output),
             '--validation',str(a.validation),'--reference',str(a.reference),'--speed-gate',str(a.speed_gate)]
    with (launch/'supervisor.log').open('w') as log:
        child=subprocess.Popen(command,cwd=root,env=dict(os.environ,**gate['environment']),stdin=subprocess.DEVNULL,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt={'pid':child.pid,'gpu':state,'source':str(root),'command':command,'guard':str(guard),
             'output':str(output),'source_sha256':gate['source_sha256'],'expected_videos':20}
    save(launch/'launch.json',receipt);save(a.base/'active_video_job.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
