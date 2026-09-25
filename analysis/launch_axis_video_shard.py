"""Launch one authorized shard or migrate a legacy worker at its next boundary."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from analysis.quick_axis_c4_qkv import save,gpu_state
from analysis.axis_c4_speed_gate import verify_gate


def main():
    p=argparse.ArgumentParser();p.add_argument('--axes',choices=['tw','wh'],required=True)
    p.add_argument('--host',choices=['scai6','scai7'],required=True)
    p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    assert socket.gethostname().split('.')[0]==a.host
    root=Path(__file__).resolve().parents[1]
    base=Path('/local2/denghaoran/GridMergeAttention/experiments/axis_video_c4_20260918')
    cfg_path=base/f'{a.axes}_parallel_config.json';cfg=json.loads(cfg_path.read_text())
    validation='trimmed_validate_tw.json' if a.axes=='tw' else 'validate_native_fast_wh.json'
    gate=base/f'speed_gate_{a.axes}.json';verify_gate(gate,root,a.axes)
    subprocess.run([sys.executable,'-m','analysis.verify_prompt_contract','--prompt_path',str(root/'prompts/MovieGenVideoBench_extended.txt'),
                    '--num_videos','20','--seed','0','--manifest_out',str(base/f'{a.axes}_{a.host}_prompt_preflight.json')],cwd=root,check=True,stdout=subprocess.DEVNULL)
    indices=cfg[a.host+'_indices'];assert indices
    out=base/f'{a.axes}_{a.host}_shard20';guard=base/f'run_{a.axes}_{a.host}_shard20'
    assert not out.exists() and not guard.exists()
    command=[sys.executable,'-m','analysis.guard_axis_job','--gpu',str(a.gpu),'--output',str(guard),'--',sys.executable,
             '-m','analysis.run_axis_c4_video_shard','--axes',a.axes,'--output',str(out),'--validation',str(base/validation),
             '--reference',str(base/'dense_first20'),'--speed-gate',str(gate),'--aggregate-config',str(cfg_path),
             '--indices',*map(str,indices)]
    if a.host=='scai7':
        old_guard=base/('run_tw_cuda_trim_v1' if a.axes=='tw' else 'run_wh_cuda_fast_v1')
        command=[sys.executable,'-m','analysis.migrate_axis_video_worker','--legacy-output',cfg['sources'][0],
                 '--legacy-guard',str(old_guard),'--stop-after-count',str(cfg['legacy_stop_count']),'--gpu',str(a.gpu),
                 '--receipt',str(base/f'{a.axes}_migration.json'),'--',*command]
    else:
        state=gpu_state(a.gpu)
        assert not state['pids'] and state['memory_mib']<=100 and not state['utilization'],state
    log=base/f'{a.axes}_{a.host}_dispatch.log'
    with log.open('x') as handle:
        child=subprocess.Popen(command,cwd=root,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    receipt={'pid':child.pid,'host':a.host,'gpu':a.gpu,'indices':indices,'guard':str(guard),'output':str(out),'source':str(root),'command':command}
    save(base/f'{a.axes}_{a.host}_dispatch.json',receipt)
    print(json.dumps({key:receipt[key] for key in ('pid','host','gpu','indices','output')}),flush=True)


if __name__=='__main__':main()
