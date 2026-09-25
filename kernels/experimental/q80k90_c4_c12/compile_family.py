"""Compile each new ownership/tile path on CPU before interrupting any GPU job."""
import json,os,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'
os.environ['MAX_JOBS']='2'
os.environ['PATH']='/usr/local/cuda/bin:'+os.environ['PATH']
import torch
from q80k90_kernel import native_merge
rows=[]
for cap,tile in ((4,4),(4,8),(6,8),(8,8),(10,16),(12,16)):
    start=time.time()
    native_merge(cap,21,40,4,seed_tile=tile,fast_math=True,direct_store=True,
                 min_blocks=4,owner_mode=1,vector_store=True,sum_groups=2)
    assert not torch.cuda.is_initialized()
    rows.append(dict(capacity=cap,seed_tile=tile,compiled=True,seconds=time.time()-start))
    Path('offline_compile.json').write_text(json.dumps(dict(complete=False,gpu_used=False,rows=rows),indent=2)+'\n')
    print(json.dumps(rows[-1]),flush=True)
Path('offline_compile.json').write_text(json.dumps(dict(complete=True,gpu_used=False,rows=rows),indent=2)+'\n')
