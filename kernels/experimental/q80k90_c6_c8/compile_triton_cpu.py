"""Offline sm_90 compilation of Triton paths; never loads a CUDA kernel."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('TRITON_CACHE_DIR','/local2/denghaoran/GridMergeAttention/q80c6_c8_v2_triton')
import torch
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from q80k90_kernel import _pack_inputs,_select,_merge,_route

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
cases=[('pack',_pack_inputs,{n:'*bf16' for n in ('Q','K','V','QP','KP','VP','Packed')},dict(T=21,H=40,B=1024),dict(num_warps=4))]
for cap in (6,8):
    for fp64,groups in [(True,4),(False,1),(False,2),(False,4),(False,8)]:
        cases.append((f'c{cap}_select_{"fp64" if fp64 else "fp32"}_g{groups}',_select,
                      dict(Q='*bf16',K='*bf16',Table='*i32',Seeds='*u8'),
                      dict(T=21,H=40,BG=groups,CAP=cap,FP64=fp64),dict(num_warps=4,enable_fp_fusion=False)))
    for m,n in [(128,64),(128,128),(64,64),(32,64)]:
        cases.append((f'c{cap}_route_m{m}_n{n}',_route,
                      dict(Q='*bf16',MK='*bf16',MV='*bf16',Bias='*fp32',O='*bf16',HS='i32'),
                      dict(T=21,H=40,AH=4,BM=m,BN=n,CAP=cap),dict(num_warps=4,num_stages=2)))
    cases.append((f'c{cap}_triton_merge',_merge,
                  dict(Q='*bf16',K='*bf16',V='*bf16',Seeds='*u8',MK='*bf16',MV='*bf16',Bias='*fp32',HS='i32'),
                  dict(T=2,H=5,AH=3,BG=4,KG_LOOP=1,CAP=cap),dict(num_warps=4,num_stages=2)))
report=dict(complete=False,gpu_used=False,source_sha256=hashlib.sha256(Path(__file__).with_name('q80k90_kernel.py').read_bytes()).hexdigest(),rows=[])
for name,fn,signature,constexprs,options in cases:
    start=time.time()
    try:
        result=triton.compile(ASTSource(fn,signature,constexprs),target=GPUTarget('cuda',90,32),options=options)
        binary=a.output/f'{name}.cubin';binary.write_bytes(result.asm['cubin'])
        resources=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(binary)],text=True)
        (a.output/f'{name}_resources.txt').write_text(resources)
        row=dict(name=name,compiled=True,seconds=time.time()-start,shared_bytes=result.metadata.shared)
    except Exception as exc:
        row=dict(name=name,compiled=False,seconds=time.time()-start,error=repr(exc))
    assert not torch.cuda.is_initialized(),'Offline Triton build initialized CUDA'
    report['rows'].append(row)
    (a.output/'compile_triton.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(row),flush=True)
report['complete']=True;report['all_compiled']=all(x['compiled'] for x in report['rows'])
(a.output/'compile_triton.json').write_text(json.dumps(report,indent=2)+'\n')
assert report['all_compiled'],'Inspect failed offline Triton variants'
