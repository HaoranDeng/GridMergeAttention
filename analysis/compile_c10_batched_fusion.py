"""CPU-only compilation feasibility screen; never launches a CUDA kernel."""
import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
import hashlib
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from native_fisher.batched_fused_route import batched_fused_c10,combine_batched_c10

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--configs',default='2x32x8,2x16x4,4x16x16',help='Columns x query rows x warps, comma separated')
args=p.parse_args()
assert not args.output.exists();args.output.parent.mkdir(parents=True,exist_ok=True)
rows=[]
for columns,bm,warps in (tuple(map(int,v.split('x'))) for v in args.configs.split(',')):
    row={'warps':warps,'columns':columns,'block_m':bm,'parts':16};rows.append(row)
    tag=f'c{columns}_m{bm}_w{warps}'
    try:
        signature={n:'*bf16' for n in ('Q','K','V')}
        signature.update(Seeds='*u8',Acc='*fp32',Max='*fp32',Den='*fp32')
        source=ASTSource(batched_fused_c10,signature=signature,constexprs={'HEAD_START':0,'HEADS':1,'COLS':columns,'PARTS':16,'BM':bm})
        kernel=triton.compile(source,target=GPUTarget('cuda',90,32),options={'num_warps':warps,'num_stages':1})
        row.update(compiled=True,metadata=kernel.metadata._asdict())
        (args.output.parent/f'batched_fusion_{tag}.ptx').write_text(kernel.asm['ptx'])
        binary=args.output.parent/f'batched_fusion_{tag}.cubin'
        binary.write_bytes(kernel.asm['cubin'])
        usage=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(binary)],capture_output=True,text=True)
        row['resource_usage']=usage.stdout+usage.stderr
    except Exception as error:
        row.update(compiled=False,error=str(error))
    args.output.write_text(json.dumps({'scope':'CPU compilation only; no CUDA device/context or timing',
                                     'source_sha256':hashlib.sha256((ROOT/'native_fisher/batched_fused_route.py').read_bytes()).hexdigest(),
                                     'variants':rows},indent=2,default=str))
    print(json.dumps({k:v for k,v in row.items() if k!='metadata'},default=str),flush=True)

source=ASTSource(combine_batched_c10,signature={'Acc':'*fp32','Max':'*fp32','Den':'*fp32','Out':'*bf16'},
                 constexprs={'HEAD_START':0,'HEADS':1,'PARTS':16,'BQ':4})
kernel=triton.compile(source,target=GPUTarget('cuda',90,32),options={'num_warps':4,'num_stages':1})
binary=args.output.parent/'batched_fusion_combine.cubin';binary.write_bytes(kernel.asm['cubin'])
usage=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(binary)],capture_output=True,text=True)
report=json.loads(args.output.read_text());report['combine']={'compiled':True,'resource_usage':usage.stdout+usage.stderr}
args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report['combine']),flush=True)
