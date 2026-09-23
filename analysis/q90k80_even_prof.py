"""clock64 phase profile of a merge kernel (even-capacity family, or the C14 copy).

python q90k80_even_prof.py --cap 8 [--tiles N] [--flags "..."]
python q90k80_even_prof.py --c14
Warp 0 lane 0 of every CTA stamps clock64 between phases; the sums are divided
by the number of (kb, head, tile) items to give CTA cycles per tile per phase.
"""
import os,sys,hashlib,ctypes,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent;WS=HERE.parent;sys.path.insert(0,str(WS))
HOST=os.uname().nodename
os.environ.setdefault('TORCH_CUDA_ARCH_LIST','9.0a');os.environ.setdefault('TORCH_EXTENSIONS_DIR',str(WS/('.torch_ext_'+HOST)));os.environ.setdefault('MAX_JOBS','8')
ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libstdc++.so.6',mode=ctypes.RTLD_GLOBAL)
import torch
from torch.utils.cpp_extension import load
ap=argparse.ArgumentParser();ap.add_argument('--cap',type=int,default=8);ap.add_argument('--c14',action='store_true');ap.add_argument('--tiles',type=int,default=0);ap.add_argument('--flags',default='')
a=ap.parse_args()
T,H,AH=21,40,4
CUTLASS=Path(torch.__file__).resolve().parent.parent/'flashinfer/data/cutlass/include'
common=['-O3','--expt-relaxed-constexpr','-lineinfo',f'-DAXIS_OUTER={T}','-DAXIS_ROWS=45','-DAXIS_COLS=80',f'-DAXIS_HEADS={H}',f'-DAXIS_ACTIVE_HEADS={AH}','-DAXIS_FAST_MATH=1','-DMERGE_FUSED_ARGMAX=1']
if a.c14:
    src=HERE/'variants_tmp/c14_merge_prof.cu';cap=14;fl=common
    from analysis.q90k80c14_kernel import Q90K80C14,native_select
    op=Q90K80C14(merge_backend='cuda');tiles=a.tiles or 35;total_tiles=210
else:
    src=HERE/'q90k80_even_merge.cu';cap=a.cap
    fl=common+[f'-DAXIS_CAP={cap}','-DMERGE_PACKED=1','-DMERGE_MINBLOCKS=3']
    from analysis.q90k80_even_kernel import Q90K80Even,native_select as ns_even,merge_tiles
    op=Q90K80Even(cap,merge_backend='cuda');total_tiles=merge_tiles(T,cap,True);tiles=a.tiles or max(1,(total_tiles+5)//6)
fl+=a.flags.split()
def build(more):
    f=fl+more
    return load(name='evprof_'+hashlib.sha256(src.read_bytes()+' '.join(f).encode()).hexdigest()[:16],sources=[str(src)],extra_include_paths=[str(CUTLASS)],extra_cuda_cflags=f,extra_ldflags=['-lcuda'],verbose=False)
m0=build([]);m1=build(['-DMERGE_PROF=1'])
print('[resources] base',m0.resources(),'prof',m1.resources(),flush=True)
torch.manual_seed(0);dev='cuda';N=T*3600
q=torch.randn(1,N,H,128,device=dev,dtype=torch.bfloat16)*0.05
k=torch.randn(1,N,H,128,device=dev,dtype=torch.bfloat16)*0.05
v=torch.randn(1,N,H,128,device=dev,dtype=torch.bfloat16)*0.05
t,h=op.prepare(q)
qp=q.view(t,45,40,2,h,128).permute(4,0,2,3,1,5).contiguous()
kp=k.view(t*45,80,h,128).permute(2,0,1,3).contiguous()
vp=v.view(t*45,80,h,128).permute(2,0,1,3).contiguous()
if a.c14:native_select(t,h).run(qp,kp,op.table,op.seeds)
else:ns_even(t,h,cap).run(qp,kp,op.table,op.seeds)
packed=torch.empty_like(qp);m0.pack(qp,packed);torch.cuda.synchronize()
def timeit(fn,reps=10):
    fn();torch.cuda.synchronize();s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);ts=[]
    for _ in range(reps):
        s.record();fn();e.record();torch.cuda.synchronize();ts.append(s.elapsed_time(e))
    ts.sort();return ts[len(ts)//2]
t0=timeit(lambda:m0.run(packed,kp,vp,op.seeds,op.mk,op.mv,op.bias,0,AH,tiles))
prof=torch.zeros(8,dtype=torch.int64,device=dev)
t1=timeit(lambda:m1.run_prof(packed,kp,vp,op.seeds,op.mk,op.mv,op.bias,0,AH,tiles,prof))
prof.zero_();m1.run_prof(packed,kp,vp,op.seeds,op.mk,op.mv,op.bias,0,AH,tiles,prof);torch.cuda.synchronize()
p=prof.cpu().tolist();n=945*AH*total_tiles
nm=['score wait','gather issue','scalar block','merge k0 wait','stage+TMA k0','merge k1 wait','stage+TMA k1','loop top']
tot=sum(p)
print(f'[time] cap={cap} base {t0:.2f} ms  prof-build {t1:.2f} ms per head chunk (tiles/CTA={tiles}, tiles={total_tiles})')
print(f'[cta cycles per tile] total={tot/n:.0f}: '+' | '.join(f'{k}={p[i]/n:.0f}' for i,k in enumerate(nm)),flush=True)
