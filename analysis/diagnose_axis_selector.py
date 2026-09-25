"""Diagnostic-only selector timings. These bypasses are never video candidates."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import torch
from torch.utils.cpp_extension import load
from attention.axis_current_c4_cuda import _compiled, _workspace


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    o,r,l,h,ah=80,21,45,40,2
    q=torch.randn((1,o,r,l,h,128),device='cuda',dtype=torch.bfloat16)
    k=torch.randn_like(q)
    candidates,anchors,seeds,*_=_workspace(o,r,l,h,ah,'cuda:0')
    real,_=_compiled(o,r,l,h,ah)
    source=Path(__file__).resolve().parents[1]/'attention/axis_current_c4_selector.cu'
    text=source.read_text()
    start=text.index(' if(column<L){',text.index('__global__'))
    end=text.index('\n  }',start)+len('\n  }')
    bypass=''' if(column<L){
 float checksum=0.f;
 #pragma unroll
 for(int c=0;c<4;++c){
   #pragma unroll
   for(int p=0;p<4;++p)checksum+=local[c][p];
 }
 checksum+=__shfl_xor_sync(0xffffffff,checksum,16);
 checksum+=__shfl_xor_sync(0xffffffff,checksum,8);
 checksum+=__shfl_xor_sync(0xffffffff,checksum,4);
 checksum+=__shfl_xor_sync(0xffffffff,checksum,2);
 checksum+=__shfl_xor_sync(0xffffffff,checksum,1);
 if(lane==0)*reinterpret_cast<unsigned*>(seeds+int64_t(group)*4)=__float_as_uint(checksum);
 }'''
    diagnostic=text[:start]+bypass+text[end:]
    diagnostic=diagnostic.replace('"../kernels/best/c4/warp_mma.cuh"', '"'+str(source.parents[1]/'kernels/best/c4/warp_mma.cuh')+'"')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    cu=a.output.with_suffix('.cu');cu.write_text(diagnostic)
    flags=['-O3','--use_fast_math','--expt-relaxed-constexpr','-lineinfo',
           '-DAXIS_OUTER=80','-DAXIS_ROWS=21','-DAXIS_COLS=45','-DAXIS_HEADS=40',
           '-DAXIS_ACTIVE_HEADS=2','-DAXIS_CHOLESKY=4',
           '-DAXIS_SELECTOR_WARPS='+os.environ.get('AXIS_C4_SELECTOR_WARPS','2')]
    digest=hashlib.sha256(diagnostic.encode()+' '.join(flags).encode()).hexdigest()[:16]
    probe=load(name='axis_selector_diagnostic_'+digest,sources=[str(cu)],extra_cuda_cflags=flags,verbose=False)
    result={'diagnostic_only':True,'not_a_gma_candidate':True,'scope':'selector all40 heads','results':{}}
    for name,module in [('complete_selector',real),('qk_without_fisher',probe)]:
        def invoke():
            for hs in range(0,h,ah):module.run(q,k,candidates,anchors,seeds,hs,128**-.5)
        invoke();torch.cuda.synchronize()
        samples=[]
        for _ in range(5):
            s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            s.record();invoke();e.record();e.synchronize();samples.append(s.elapsed_time(e))
        result['results'][name]={'samples_ms':samples,'median_ms':statistics.median(samples),'resources':module.resources()}
        print(json.dumps({name:result['results'][name]}),flush=True)
    a.output.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
