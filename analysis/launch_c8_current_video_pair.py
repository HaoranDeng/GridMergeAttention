"""Explicitly authorized two-H200 run, prompts 0-9 and 10-19, no queue."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from analysis.quick_axis_c4_qkv import gpu_state
from analysis.benchmark_c8_current import save


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',type=int,nargs=2,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert len(set(a.gpus))==2
    states=[gpu_state(gpu) for gpu in a.gpus]
    for state in states:
        if 'H200' not in state['name'] or state['pids'] or state['memory_mib']>100 or state['utilization']:
            raise RuntimeError(f'Both GPUs must be clean; no launch or queue: {state}')
    a.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    shards=[]
    for gpu,start in zip(a.gpus,(0,10)):
        shard=a.output/f'gpu{gpu}_{start:03d}_{start+9:03d}'
        subprocess.run([sys.executable,'-m','analysis.launch_c8_current_videos','--gpu',str(gpu),
                        '--start',str(start),'--stop',str(start+10),'--output',str(shard)],cwd=root,check=True)
        shards.append(str(shard))
    save(a.output/'launch.json',{'gpus':a.gpus,'preflight':states,'shards':shards,
                                'authorization':'User explicitly requested two GPUs','prompt_indices':list(range(20))})
    with (a.output/'collector.log').open('x') as log:
        child=subprocess.Popen([sys.executable,'-u','-m','analysis.collect_c8_current_video_pair',
            '--output',str(a.output)],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True)
    save(a.output/'collector.json',{'pid':child.pid})
    print(json.dumps({'output':str(a.output),'shards':shards,'collector_pid':child.pid}),flush=True)
