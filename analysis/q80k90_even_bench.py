"""Stage benchmark and correctness harness for the Q80K90 even-capacity kernels.

python q80k90_even_bench.py --cap 8 [--tiles N] [--packed 1] [--minblocks 3]
    [--hc 8] [--reps 30] [--ref] [--v4] [--tag s] [--out results.jsonl]
    [--merge-flags "-DX=1"] [--route-flags "..."] [--seed 0]
--ref runs the v4 Triton select/merge/route path (kernels/experimental/
q80k90_c4_c12, FP32 selector as in production, plus the FP64 selector for the
seed agreement) on the same QKV and reports the relative L2 of the native
output against it, the seed agreement and the bias agreement (last head chunk;
both operators use the [ah, 945, 840, CAP] bias layout).
--v4 also times the production v4 operator (recommended.make_q80k90) on the same
QKV, interleaved with the new operator, and reports its output difference.
"""
import os,sys,time,json,argparse,ctypes,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
HOST=os.uname().nodename
os.environ.setdefault('TORCH_CUDA_ARCH_LIST','9.0a');os.environ.setdefault('MAX_JOBS','8')
os.environ.setdefault('TORCH_EXTENSIONS_DIR',str(HERE.parent/('.torch_ext_'+HOST)))
ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libstdc++.so.6',mode=ctypes.RTLD_GLOBAL)
import torch
from analysis.q80k90_even_kernel import Q80K90Even,native_merge,native_route,native_select,native_layout,merge_tiles,candidate_table,input_strides

V4_DIR=HERE.parent/'kernels/experimental/q80k90_c4_c12'
def v4_module(name):
    spec=importlib.util.spec_from_file_location(name,V4_DIR/f'{name}.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

ap=argparse.ArgumentParser()
ap.add_argument('--cap',type=int,required=True)
ap.add_argument('--tiles',type=int,default=0)
ap.add_argument('--packed',type=int,default=1);ap.add_argument('--minblocks',type=int,default=3)
ap.add_argument('--hc',type=int,default=8);ap.add_argument('--reps',type=int,default=30)
ap.add_argument('--ref',action='store_true');ap.add_argument('--v4',action='store_true')
ap.add_argument('--merge-flags',default='');ap.add_argument('--route-flags',default='');ap.add_argument('--select-flags',default='')
ap.add_argument('--psi',type=int,default=1,help='merge_psi: 1 mv direct (default), 2 mk and mv direct, 0 staged')
ap.add_argument('--tag',default='');ap.add_argument('--out',default='');ap.add_argument('--seed',type=int,default=0)
ap.add_argument('--scale',type=float,default=1.0,help='std of the random Q/K/V entries')
ap.add_argument('--stages-only',action='store_true');ap.add_argument('--full-only',action='store_true')
ap.add_argument('--head-major',action='store_true',help='pass Q/K/V as transposed views of head-major [1, H, N, 128] tensors (as attention processors hold them)')
args=ap.parse_args()

T,H=21,40;cap=args.cap;AH=args.hc
mflags=tuple(args.merge_flags.split());rflags=tuple(args.route_flags.split());sflags=tuple(args.select_flags.split())
tiles=args.tiles if args.tiles>0 else max(1,(merge_tiles(T,cap,args.packed)+3)//4)
t0=time.time()
mmod=native_merge(T,H,AH,cap,args.packed,args.minblocks,mflags,args.psi>=2,args.psi>=1)
rmod=native_route(T,H,AH,cap,rflags)
smod=native_select(T,H,cap,sflags)
lmod=native_layout(T,H)
print(f'[build] cap={cap} packed={args.packed} minblocks={args.minblocks} hc={AH} tiles/CTA={tiles} {time.time()-t0:.1f}s '
      f'merge(regs,local,smem,ctas/sm,tiles,gpt)={mmod.resources()} route(regs,local,smem,ctas/sm,chunks,kpc)={rmod.resources()} '
      f'select(regs,local,smem,ctas/sm)={smod.resources()}',flush=True)

torch.manual_seed(args.seed);dev='cuda';N=T*3600
q=torch.randn(1,N,H,128,device=dev,dtype=torch.bfloat16)*args.scale
k=torch.randn(1,N,H,128,device=dev,dtype=torch.bfloat16)*args.scale
v=torch.randn(1,N,H,128,device=dev,dtype=torch.bfloat16)*args.scale
if args.head_major:
    token_major=(q,k,v)
    q,k,v=(x.transpose(1,2).contiguous().transpose(1,2) for x in token_major);del token_major
op=Q80K90Even(cap,head_chunk=AH,native_tiles=tiles,merge_packed=args.packed,merge_minblocks=args.minblocks,
              merge_flags=mflags,route_flags=rflags,select_flags=sflags,merge_psi=args.psi)
t,h=op.prepare(q)
res=op(q,k,v);torch.cuda.synchronize()
finite=bool(torch.isfinite(res).all().item())
print(f'[native] finite={finite} out_norm={res.float().norm().item():.4f}',flush=True)
result={'tag':args.tag,'cap':cap,'args':vars(args),'tiles':tiles,'gpu':torch.cuda.get_device_name(0),'finite':finite,
        'merge_resources':mmod.resources(),'route_resources':rmod.resources(),'select_resources':smod.resources()}

def rel(a,b):return ((a.float()-b.float()).norm()/b.float().norm()).item()
if args.ref:
    kmod=v4_module('q80k90_kernel')
    assert kmod.candidate_table()==candidate_table(),'candidate table differs from v4'
    ref=kmod.Q80K90(cap,merge_backend='triton',route_backend='triton',selector_precision='fp32',head_chunk=AH)
    t1=time.time();ro=ref(q,k,v);torch.cuda.synchronize();tref=time.time()-t1
    r=rel(res,ro);mx=(res.float()-ro.float()).abs().max().item()
    seed_changed=(ref.seeds!=op.seeds).float().mean().item()
    rb=ref.bias.reshape(-1);ab=op.bias.reshape(-1)
    fin=torch.isfinite(rb)&torch.isfinite(ab)
    bias_rel=((rb[fin]-ab[fin]).norm()/rb[fin].norm()).item();empty_mismatch=(torch.isfinite(rb)!=torch.isfinite(ab)).float().mean().item()
    empty_frac=(~torch.isfinite(ab)).float().mean().item()
    # merged K/V of the last head chunk: v4 [ah][qt][45][kb][c][128] vs [ah][kb][group][c][128]
    rmk=ref.mk.reshape(AH,T*45,T*40,cap,128).permute(0,2,1,3,4)
    num=den=0.
    for i in range(AH):  # per head: the float copies of a whole chunk do not fit next to both workspaces
        x=op.mk[i].float();y=rmk[i].float();num+=float((x-y).pow(2).sum());den+=float(y.pow(2).sum());del x,y
    mk_rel=(num/den)**.5;del rmk
    print(f'[ref v4-triton fp32 {tref:.1f}s] rel_l2={r:.3e} max_abs={mx:.3e} seed_changed_frac={seed_changed:.3e} bias_rel(last chunk)={bias_rel:.3e} '
          f'mk_rel(last chunk)={mk_rel:.3e} empty_mismatch_frac={empty_mismatch:.3e} empty_frac={empty_frac:.4f} ref_finite={bool(torch.isfinite(ro).all().item())}',flush=True)
    result.update(rel_l2=r,max_abs=mx,seed_changed_frac=seed_changed,bias_rel=bias_rel,mk_rel=mk_rel,empty_mismatch_frac=empty_mismatch,empty_frac=empty_frac)
    # FP64 selector seeds (the v4 Triton kernel with FP64=True) on the same QKV.
    import triton as tr
    qp=q.reshape(t,45,80,h,128).permute(3,0,1,2,4).contiguous()
    kp=k.reshape(t,45,40,2,h,128).permute(4,0,2,3,1,5).reshape(h,t*40,90,128).contiguous()
    seeds64=torch.empty_like(op.seeds)
    kmod._select[(t*40,tr.cdiv(45,2),h)](qp,kp,ref.table,seeds64,t,h,2,cap,True,num_warps=4,enable_fp_fusion=False)
    torch.cuda.synchronize()
    s64=(seeds64!=op.seeds).float().mean().item();s32_64=(seeds64!=ref.seeds).float().mean().item()
    print(f'[ref seeds] native_vs_fp64={s64:.3e} v4fp32_vs_fp64={s32_64:.3e}',flush=True)
    result.update(seed_changed_vs_fp64=s64,v4_fp32_seed_changed_vs_fp64=s32_64)
    del ref,ro,seeds64,qp,kp;torch.cuda.empty_cache()

v4op=None
if args.v4:
    rec=v4_module('recommended')
    v4op=rec.make_q80k90(cap)
    o4=v4op(q,k,v);torch.cuda.synchronize()
    r4=rel(res,o4);print(f'[v4 native] rel_l2(new vs v4)={r4:.3e}',flush=True);result['rel_l2_vs_v4']=r4;del o4

packed=torch.empty((h,N,128),device=dev,dtype=torch.bfloat16)
out=torch.empty((1,N,H,128),device=dev,dtype=torch.bfloat16)
(_,qs),(_,ks),(_,vs)=input_strides(q),input_strides(k),input_strides(v)
qm,qms=op.merge_query(q,qs,t,h)
def layout_only():lmod.forward(q,packed,*qs)
def select_only():smod.run(q,k,op.table,op.seeds,*qs,*ks)
def merge_only():
    for hs in range(0,h,AH):mmod.run(qm,k,v,op.seeds,op.mk,op.mv,op.bias,hs,AH,tiles,*qms,*ks,*vs)
def route_only():
    for hs in range(0,h,AH):rmod.run(q,op.mk,op.mv,op.bias,out,hs,AH,*qs)
layout_only();torch.cuda.synchronize()
if op.packed_q is not None:print(f'[layout] packed matches operator copy={torch.equal(packed,op.packed_q)}',flush=True)
def full_call():return op(q,k,v)

def timeit(fn,reps):
    fn();torch.cuda.synchronize();ts=[]
    s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        s.record();fn();e.record();torch.cuda.synchronize();ts.append(s.elapsed_time(e))
    ts.sort();n=len(ts)
    return {'mean':sum(ts)/n,'median':ts[n//2],'min':ts[0],'max':ts[-1],'n':n}
if not args.full_only:
    for name,fn in [('layout',layout_only),('select',select_only),('merge',merge_only),('route',route_only)]:
        r=timeit(fn,max(5,args.reps//2));result[name]=r
        print(f'[stage] {name:7s} mean={r["mean"]:.3f} median={r["median"]:.3f} min={r["min"]:.3f} ms',flush=True)
if not args.stages_only:
    if v4op is None:
        r=timeit(full_call,args.reps);result['full']=r
        print(f'[full] cap={cap} mean={r["mean"]:.3f} median={r["median"]:.3f} min={r["min"]:.3f} max={r["max"]:.3f} ms (n={r["n"]})',flush=True)
    else:
        # interleaved pairs: new, v4, v4, new, ...
        full_call();v4op(q,k,v);torch.cuda.synchronize()
        s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);tn=[];t4=[]
        for i in range(args.reps):
            for name in (('new','v4') if i%2==0 else ('v4','new')):
                s.record();(full_call() if name=='new' else v4op(q,k,v));e.record();torch.cuda.synchronize()
                (tn if name=='new' else t4).append(s.elapsed_time(e))
        for name,ts in (('full',tn),('full_v4',t4)):
            ts=sorted(ts);n=len(ts);r={'mean':sum(ts)/n,'median':ts[n//2],'min':ts[0],'max':ts[-1],'n':n};result[name]=r
            print(f'[{name}] cap={cap} mean={r["mean"]:.3f} median={r["median"]:.3f} min={r["min"]:.3f} max={r["max"]:.3f} ms (n={n})',flush=True)
a=op(q,k,v);b=op(q,k,v);torch.cuda.synchronize()
same=bool(torch.equal(a,b));result['consecutive_identical']=same
print(f'[determinism] consecutive calls identical={same}',flush=True)
if args.out:
    with open(args.out,'a') as f:f.write(json.dumps(result)+'\n')
