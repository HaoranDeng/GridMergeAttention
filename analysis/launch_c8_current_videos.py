"""Launch a guarded single-H200 20-video current-frame C8 job, never queue."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from analysis.quick_axis_c4_qkv import gpu_state
from analysis.benchmark_c8_current import save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--stop',type=int,default=20)
    a=p.parse_args()
    assert 0<=a.start<a.stop<=20
    state=gpu_state(a.gpu)
    if 'H200' not in state['name'] or state['pids'] or state['memory_mib']>100 or state['utilization']:
        raise RuntimeError(f'No clean H200; no launch or queue: {state}')
    a.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    reference=Path('/local2/denghaoran/GridMergeAttention/experiments/axis_video_c4_20260918/dense_first20')
    command=[sys.executable,'-m','analysis.guard_axis_job','--gpu',str(a.gpu),'--output',str(a.output/'guard'),'--',
        sys.executable,'-u','-m','analysis.run_c8_current_videos','--output',str(a.output/'videos'),
        '--reference',str(reference),'--evidence',str(root/'kernels/experimental/c8_current/evidence.json'),
        '--start',str(a.start),'--stop',str(a.stop)]
    env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ['PATH'],TORCH_CUDA_ARCH_LIST='9.0a',MAX_JOBS='2',
        PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        TORCH_EXTENSIONS_DIR='/local2/denghaoran/GridMergeAttention/experiments/c8_current_20260919/extensions',
        FLASHINFER_WORKSPACE_BASE=str(a.output/'flashinfer'),TORCHINDUCTOR_CACHE_DIR=str(a.output/'inductor'),
        TMPDIR=str(a.output/'tmp'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    (a.output/'tmp').mkdir()
    with (a.output/'supervisor.log').open('x') as log:
        child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,
                               stderr=subprocess.STDOUT,start_new_session=True)
    receipt={'pid':child.pid,'gpu':state,'command':command,'output':str(a.output),'source':str(root),
             'prompt_indices':list(range(a.start,a.stop))}
    save(a.output/'launch.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
