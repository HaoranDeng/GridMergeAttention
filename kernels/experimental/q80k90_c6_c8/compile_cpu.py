"""Compile sm_90a binaries and inspect resources without creating a GPU context."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'
os.environ['MAX_JOBS']='2'
os.environ['PATH']='/usr/local/cuda/bin:'+os.environ['PATH']
os.environ.setdefault('CUDA_HOME','/usr/local/cuda')
os.environ.setdefault('LD_PRELOAD','/usr/lib/x86_64-linux-gnu/libstdc++.so.6')

import torch
from q80k90_kernel import native_merge

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
assert not torch.cuda.is_initialized()
report=dict(complete=False,gpu_used=False,rows=[],source_sha256={
    n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
    for n in ('q80k90_kernel.py','q80k90_merge.cu')})

for cap in (8,6):
    for name,fast,direct,blocks in [('conservative',False,False,2),
                                    ('fast_blocks2',True,False,2),
                                    ('fast_blocks3',True,False,3),
                                    ('direct_blocks3',True,True,3),
                                    ('direct_blocks4',True,True,4)]:
        start=time.time()
        try:
            mod=native_merge(cap,21,40,4,8,fast,direct,blocks)
            resources=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',mod.__file__],text=True)
            (a.output/f'c{cap}_{name}_resources.txt').write_text(resources)
            match=re.search(r'Function .*merge_kernel.*?\n\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)',resources)
            assert match,resources[-1000:]
            reg,stack,static_shared,local=map(int,match.groups())
            row=dict(capacity=cap,name=name,compiled=True,fast_math=fast,direct_store=direct,min_blocks=blocks,
                     registers=reg,stack_bytes=stack,static_shared_bytes=static_shared,local_bytes=local,
                     dynamic_shared_bytes=(96*128*2+(0 if direct else 64*128))*2,
                     binary=mod.__file__,seconds=time.time()-start)
        except Exception as exc:
            row=dict(capacity=cap,name=name,compiled=False,error=repr(exc),seconds=time.time()-start)
        assert not torch.cuda.is_initialized(),'Offline build created a CUDA context'
        report['rows'].append(row)
        (a.output/'compile.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(row),flush=True)
report['complete']=True
report['all_compiled']=all(x['compiled'] for x in report['rows'])
(a.output/'compile.json').write_text(json.dumps(report,indent=2)+'\n')
assert report['all_compiled'],'Inspect failed CPU compilation variants'
