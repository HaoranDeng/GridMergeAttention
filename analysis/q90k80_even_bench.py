"""Stage benchmark and correctness harness for the Q90K80 even-capacity kernels.

python q90k80_even_bench.py --cap 8 [--tiles N] [--packed 1] [--fused 1] [--minblocks 3]
    [--hc 4] [--reps 30] [--ref] [--ref-hc 4] [--tag s] [--out results.jsonl]
    [--merge-flags "-DX=1"] [--route-flags "..."] [--seed 0]
--ref runs the Triton select/merge/route path on the same QKV and reports the
relative L2 of the native output against it, the seed agreement and the bias
agreement (last head chunk).
"""
import os,sys,time,json,argparse,ctypes
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
HOST=os.uname().nodename
os.environ.setdefault('TORCH_CUDA_ARCH_LIST','9.0a');os.environ.setdefault('MAX_JOBS','8')
os.environ.setdefault('TORCH_EXTENSIONS_DIR',str(HERE.parent/('.torch_ext_'+HOST)))
ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libstdc++.so.6',mode=ctypes.RTLD_GLOBAL)
import torch
from analysis.q90k80_even_kernel import Q90K80Even,native_merge,native_route,native_select,native_layout,merge_tiles,groups_per_tile

ap=argparse.ArgumentParser()
ap.add_argument('--cap',type=int,required=True)
ap.add_argument('--tiles',type=int,default=0)
ap.add_argument('--packed',type=int,default=1);ap.add_argument('--fused',type=int,default=1);ap.add_argument('--minblocks',type=int,default=3)
ap.add_argument('--hc',type=int,default=4);ap.add_argument('--reps',type=int,default=30)
ap.add_argument('--ref',action='store_true');ap.add_argument('--ref-hc',type=int,default=0)
ap.add_argument('--merge-flags',default='');ap.add_argument('--route-flags',default='')
ap.add_argument('--tag',default='');ap.add_argument('--out',default='');ap.add_argument('--seed',type=int,default=0)
ap.add_argument('--scale',type=float,default=0.05)
ap.add_argument('--stages-only',action='store_true');ap.add_argument('--full-only',action='store_true')
args=ap.parse_args()

T,H=21,40;cap=args.cap;AH=args.hc
mflags=tuple(args.merge_flags.split());rflags=tuple(args.route_flags.split())
tiles=args.tiles if args.tiles>0 else max(1,(merge_tiles(T,cap,args.packed)+5)//6)
t0=time.time()
mmod=native_merge(T,H,AH,cap,bool(args.fused),args.packed,args.minblocks,mflags)
rmod=native_route(T,H,AH,cap,rflags)
smod=native_select(T,H,cap)
print(f'[build] cap={cap} packed={args.packed} fused={args.fused} minblocks={args.minblocks} hc={AH} tiles/CTA={tiles} '
      f'{time.time()-t0:.1f}s merge(regs,local,smem,ctas/sm,tiles,gpt)={mmod.resources()} route(regs,local,smem,ctas/sm,chunks,kpc)={rmod.resources()}',flush=True)

torch.manual_seed(args.seed);dev='cuda';N=T*3600
q=torch.randn(1,N,H,128,device=dev,dtype=torch.bfloat16)*args.scale
k=torch.randn(1,N,H,128,device=dev,dtype=torch.bfloat16)*args.scale
v=torch.randn(1,N,H,128,device=dev,dtype=torch.bfloat16)*args.scale
op=Q90K80Even(cap,merge_backend='cuda',native_tiles=tiles,head_chunk=AH,merge_fused_argmax=bool(args.fused),merge_packed=args.packed,
              merge_minblocks=args.minblocks,merge_flags=mflags,route_flags=rflags)
t,h=op.prepare(q)
res=op(q,k,v);torch.cuda.synchronize()
finite=bool(torch.isfinite(res).all().item())
print(f'[native] finite={finite} out_norm={res.float().norm().item():.4f}',flush=True)
result={'tag':args.tag,'cap':cap,'args':vars(args),'tiles':tiles,'gpu':torch.cuda.get_device_name(0),'finite':finite,
        'merge_resources':mmod.resources(),'route_resources':rmod.resources()}
if args.ref:
    ref=Q90K80Even(cap,merge_backend='triton',head_chunk=args.ref_hc or AH)
    t1=time.time();ro=ref(q,k,v);torch.cuda.synchronize();tref=time.time()-t1
    rel=((res.float()-ro.float()).norm()/ro.float().norm()).item();mx=(res.float()-ro.float()).abs().max().item()
    seed_changed=(ref.seeds!=op.seeds).float().mean().item()
    # bias of the last head chunk (both operators keep only that chunk)
    rb=ref.bias.reshape(-1);ab=op.bias.reshape(-1)
    fin=torch.isfinite(rb)&torch.isfinite(ab)
    bias_rel=((rb[fin]-ab[fin]).norm()/rb[fin].norm()).item();empty_mismatch=(torch.isfinite(rb)!=torch.isfinite(ab)).float().mean().item()
    empty_frac=(~torch.isfinite(ab)).float().mean().item()
    print(f'[ref triton {tref:.1f}s] rel_l2={rel:.3e} max_abs={mx:.3e} seed_changed_frac={seed_changed:.3e} bias_rel(last chunk)={bias_rel:.3e} '
          f'empty_mismatch_frac={empty_mismatch:.3e} empty_frac={empty_frac:.4f} ref_finite={bool(torch.isfinite(ro).all().item())}',flush=True)
    result.update(rel_l2=rel,max_abs=mx,seed_changed_frac=seed_changed,bias_rel=bias_rel,empty_mismatch_frac=empty_mismatch,empty_frac=empty_frac)
    del ref,ro;torch.cuda.empty_cache()

qp=q.view(t,45,40,2,h,128).permute(4,0,2,3,1,5).contiguous()
kp=k.view(t*45,80,h,128).permute(2,0,1,3).contiguous()
vp=v.view(t*45,80,h,128).permute(2,0,1,3).contiguous()
out=torch.empty_like(qp);packed=torch.empty_like(qp)
def select_only():smod.run(qp,kp,op.table,op.seeds)
def pack_only():mmod.pack(qp,packed)
def merge_only():
    for hs in range(0,h,AH):mmod.run(packed,kp,vp,op.seeds,op.mk,op.mv,op.bias,hs,AH,tiles)
def route_only():
    for hs in range(0,h,AH):rmod.run(qp,op.mk,op.mv,op.bias,out,hs,AH)
lmod=native_layout(T,H)
qp2=torch.empty_like(qp);pk2=torch.empty_like(qp);kp2=torch.empty_like(kp);vp2=torch.empty_like(kp);res2=torch.empty_like(q)
def layout_only():
    q.view(t,45,40,2,h,128).permute(4,0,2,3,1,5).contiguous();k.view(t*45,80,h,128).permute(2,0,1,3).contiguous();v.view(t*45,80,h,128).permute(2,0,1,3).contiguous()
    out.view(h,t,40,2,45,128).permute(1,4,2,3,0,5).reshape(1,t*3600,h,128).contiguous()
def fused_layout_only():
    lmod.forward(q,k,v,qp2,pk2,kp2,vp2);lmod.backward(out,res2)
lmod.forward(q,k,v,qp2,pk2,kp2,vp2);mmod.pack(qp,packed);lmod.backward(out,res2);torch.cuda.synchronize()
ref_res=out.view(h,t,40,2,45,128).permute(1,4,2,3,0,5).reshape(1,t*3600,h,128).contiguous()
print(f'[layout] fused qp={torch.equal(qp,qp2)} packed={torch.equal(packed,pk2)} kp={torch.equal(kp,kp2)} vp={torch.equal(vp,vp2)} backward={torch.equal(ref_res,res2)}',flush=True)
def full_call():return op(q,k,v)

def timeit(fn,reps):
    fn();torch.cuda.synchronize();ts=[]
    s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        s.record();fn();e.record();torch.cuda.synchronize();ts.append(s.elapsed_time(e))
    ts.sort();n=len(ts)
    return {'mean':sum(ts)/n,'median':ts[n//2],'min':ts[0],'max':ts[-1],'n':n}
if not args.full_only:
    for name,fn in [('select',select_only),('pack',pack_only),('merge',merge_only),('route',route_only),('layout',layout_only),('fused_layout',fused_layout_only)]:
        r=timeit(fn,max(5,args.reps//2));result[name]=r
        print(f'[stage] {name:7s} mean={r["mean"]:.3f} median={r["median"]:.3f} min={r["min"]:.3f} ms',flush=True)
if not args.stages_only:
    r=timeit(full_call,args.reps);result['full']=r
    print(f'[full] cap={cap} mean={r["mean"]:.3f} median={r["median"]:.3f} min={r["min"]:.3f} max={r["max"]:.3f} ms (n={r["n"]})',flush=True)
# determinism: two consecutive calls identical
a=op(q,k,v);b=op(q,k,v);torch.cuda.synchronize()
same=bool(torch.equal(a,b));result['consecutive_identical']=same
print(f'[determinism] consecutive calls identical={same}',flush=True)
if args.out:
    with open(args.out,'a') as f:f.write(json.dumps(result)+'\n')
