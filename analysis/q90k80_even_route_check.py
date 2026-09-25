"""Route accuracy against an FP64 softmax on the same merged K/V.

python q90k80_even_route_check.py --cap 8 [--flags "-DROUTE_LAZY=1"] [--groups 64]
Runs select + merge for head chunk 0 once, then the baseline native route and
the route built with --flags on the same mk/mv/bias, and compares both with
an FP64 softmax(q mk^T / sqrt(128) + bias) mv over a random sample of groups
of that head chunk. Prints the relative L2 of each route to FP64 and of the
two routes to each other.
"""
import os,sys,argparse,ctypes
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
HOST=os.uname().nodename
os.environ.setdefault('TORCH_CUDA_ARCH_LIST','9.0a');os.environ.setdefault('MAX_JOBS','8')
os.environ.setdefault('TORCH_EXTENSIONS_DIR',str(HERE.parent/('.torch_ext_'+HOST)))
ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libstdc++.so.6',mode=ctypes.RTLD_GLOBAL)
import torch
from analysis.q90k80_even_kernel import Q90K80Even,native_merge,native_route,native_select,merge_tiles

ap=argparse.ArgumentParser()
ap.add_argument('--cap',type=int,required=True);ap.add_argument('--hc',type=int,default=8)
ap.add_argument('--flags',default='-DROUTE_LAZY=1');ap.add_argument('--groups',type=int,default=64)
ap.add_argument('--scale',type=float,default=0.05);ap.add_argument('--seed',type=int,default=0)
a=ap.parse_args()
T,H,AH,cap=21,40,a.hc,a.cap
tiles=max(1,(merge_tiles(T,cap,1)+3)//4)
mmod=native_merge(T,H,AH,cap,True,1,3,())
r0=native_route(T,H,AH,cap,())
r1=native_route(T,H,AH,cap,tuple(a.flags.split()))
smod=native_select(T,H,cap)
torch.manual_seed(a.seed);dev='cuda';N=T*3600
q=torch.randn(1,N,H,128,device=dev,dtype=torch.bfloat16)*a.scale
k=torch.randn(1,N,H,128,device=dev,dtype=torch.bfloat16)*a.scale
v=torch.randn(1,N,H,128,device=dev,dtype=torch.bfloat16)*a.scale
op=Q90K80Even(cap,merge_backend='cuda',native_tiles=tiles,head_chunk=AH)
t,h=op.prepare(q)
qp=q.view(t,45,40,2,h,128).permute(4,0,2,3,1,5).contiguous()
kp=k.view(t*45,80,h,128).permute(2,0,1,3).contiguous()
vp=v.view(t*45,80,h,128).permute(2,0,1,3).contiguous()
packed=torch.empty_like(qp);mmod.pack(qp,packed)
smod.run(qp,kp,op.table,op.seeds)
mmod.run(packed,kp,vp,op.seeds,op.mk,op.mv,op.bias,0,AH,tiles)
o0=torch.zeros_like(qp);o1=torch.zeros_like(qp)
r0.run(qp,op.mk,op.mv,op.bias,o0,0,AH);r1.run(qp,op.mk,op.mv,op.bias,o1,0,AH);torch.cuda.synchronize()
G=t*40;KB=t*45
mk=op.mk.view(AH,KB,G,cap,128);mv=op.mv.view(AH,KB,G,cap,128);bias=op.bias.view(AH,G,KB*cap)
Q=qp.view(h,G,90,128);O0=o0.view(h,G,90,128);O1=o1.view(h,G,90,128)
gen=torch.Generator(device='cpu').manual_seed(1)
groups=torch.randperm(G,generator=gen)[:a.groups].tolist()
e0=e1=e01=ref2=0.0;nonfinite=0
for lh in range(AH):
    for g in groups:
        K=mk[lh,:,g].reshape(-1,128).double();V=mv[lh,:,g].reshape(-1,128).double();B=bias[lh,g].double()
        x=Q[lh,g].double()
        logits=x@K.T/128**0.5+B
        ref=torch.softmax(logits,dim=-1)@V
        a0=O0[lh,g].double();a1=O1[lh,g].double()
        nonfinite+=int((~torch.isfinite(a1)).sum().item())
        e0+=((a0-ref)**2).sum().item();e1+=((a1-ref)**2).sum().item();e01+=((a0-a1)**2).sum().item();ref2+=(ref**2).sum().item()
print(f'[route-check] cap={cap} flags="{a.flags}" groups={len(groups)}x{AH} heads  '
      f'baseline_vs_fp64={(e0/ref2)**0.5:.4e}  variant_vs_fp64={(e1/ref2)**0.5:.4e}  baseline_vs_variant={(e01/ref2)**0.5:.4e}  '
      f'variant_nonfinite={nonfinite}  full_chunk_finite={bool(torch.isfinite(o1).all().item())}',flush=True)
