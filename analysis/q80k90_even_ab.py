"""Interleaved A/B timing of Q80K90 even-capacity stage variants on one QKV.

python q80k90_even_ab.py --cap 8 --stage merge --variants "" " -DMERGE_SUM_WAYS=2" [--tiles 30 15] [--rounds 12]
python q80k90_even_ab.py --cap 8 --stage select --variants "" " -DSEL_MINBLOCKS=4"
python q80k90_even_ab.py --cap 8 --stage full --tiles 30 20
(Flags start with a space so argparse does not read them as options.)

One operator provides the workspace, seeds and packed queries; every variant
is a (flags, tiles) pair of the chosen stage writing into that shared
workspace. Each variant's result is compared with the first variant's (seeds
for select, head 0 of the last head chunk's merged K and bias for merge, the
output for route and full), then each round times every variant once in a
rotated order. Reports per-variant median/mean and the paired difference
against the first variant.
"""
import os,sys,json,argparse,ctypes,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
os.environ.setdefault('TORCH_CUDA_ARCH_LIST','9.0a');os.environ.setdefault('MAX_JOBS','8')
os.environ.setdefault('TORCH_EXTENSIONS_DIR',str(HERE.parent/('.torch_ext_'+os.uname().nodename)))
ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libstdc++.so.6',mode=ctypes.RTLD_GLOBAL)
import torch
from analysis.q80k90_even_kernel import Q80K90Even,native_merge,native_route,native_select,merge_tiles,input_strides

ap=argparse.ArgumentParser()
ap.add_argument('--cap',type=int,required=True);ap.add_argument('--stage',choices=['merge','route','select','full'],required=True)
ap.add_argument('--variants',nargs='+',default=['']);ap.add_argument('--tiles',type=int,nargs='*',default=[])
ap.add_argument('--hc',type=int,default=8);ap.add_argument('--rounds',type=int,default=12);ap.add_argument('--seed',type=int,default=0)
ap.add_argument('--scale',type=float,default=1.0);ap.add_argument('--out',default='');ap.add_argument('--tag',default='')
a=ap.parse_args()
T,H,cap,AH=21,40,a.cap,a.hc
tiles=a.tiles or [max(1,(merge_tiles(T,cap)+3)//4)]
if a.stage=='full':assert a.variants==[''],'full-call variants differ only in tiles (other flags need their own workspace)'
variants=[(f,t) for f in a.variants for t in tiles]
print(f'[variants] {variants}',flush=True)
torch.manual_seed(a.seed);N=T*3600
q,k,v=(torch.randn(1,N,H,128,device='cuda',dtype=torch.bfloat16)*a.scale for _ in range(3))
op=Q80K90Even(cap,head_chunk=AH,native_tiles=tiles[0])
op(q,k,v);torch.cuda.synchronize()
out=torch.empty_like(q)
_,S=input_strides(q)  # q, k, v are token-major contiguous here
def make(flags,nt):
    fl=tuple(flags.split())
    if a.stage=='select':
        m=native_select(T,H,cap,fl);return m,(lambda:m.run(q,k,op.table,op.seeds,*S,*S))
    if a.stage=='merge':
        m=native_merge(T,H,AH,cap,op.packed,op.minblocks,fl)
        def f():
            for hs in range(0,H,AH):m.run(op.packed_q,k,v,op.seeds,op.mk,op.mv,op.bias,hs,AH,nt,*S,*S)
        return m,f
    if a.stage=='route':
        m=native_route(T,H,AH,cap,fl)
        def g():
            for hs in range(0,H,AH):m.run(q,op.mk,op.mv,op.bias,out,hs,AH,*S)
        return m,g
    def h():
        op.tiles=nt;return op(q,k,v)
    return None,h
runs=[];ref=None
for flags,nt in variants:
    m,fn=make(flags,nt)
    r=fn();torch.cuda.synchronize()
    if a.stage=='select':snap=op.seeds.clone()
    elif a.stage=='merge':snap=(op.mk[0].clone(),op.bias[0].clone())
    elif a.stage=='route':snap=out.clone()
    else:snap=r.clone()
    if ref is None:ref=snap;check=None
    elif a.stage=='merge':
        check=dict(mk0_equal=bool(torch.equal(snap[0],ref[0])),bias0_equal=bool(torch.equal(snap[1],ref[1])),
                   mk0_rel=((snap[0].float()-ref[0].float()).norm()/ref[0].float().norm()).item())
    elif a.stage=='select':check=dict(seeds_equal=bool(torch.equal(snap,ref)),seed_changed_frac=(snap!=ref).float().mean().item())
    else:check=dict(equal=bool(torch.equal(snap,ref)),rel=((snap.float()-ref.float()).norm()/ref.float().norm()).item())
    if check is not None or ref is not snap:del snap
    res=m.resources() if m is not None else None
    runs.append(dict(flags=flags,tiles=nt,fn=fn,resources=res,check=check))
    print(f'[build] flags="{flags.strip()}" tiles={nt} resources={res} check={check}',flush=True)
del ref;torch.cuda.empty_cache()
# Restore the first variant's state so every variant sees the same inputs.
runs[0]['fn']();torch.cuda.synchronize()
s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
times=[[] for _ in runs]
for r in range(a.rounds):
    order=list(range(len(runs)));order=order[r%len(order):]+order[:r%len(order)]
    if r%2:order.reverse()
    for i in order:
        s.record();runs[i]['fn']();e.record();e.synchronize();times[i].append(s.elapsed_time(e))
rows=[]
for o,ts in zip(runs,times):
    d=[x-y for x,y in zip(ts,times[0])]
    row=dict(tag=a.tag,cap=cap,stage=a.stage,flags=o['flags'].strip(),tiles=o['tiles'],median=statistics.median(ts),mean=statistics.mean(ts),min=min(ts),
             delta_vs_first_mean=statistics.mean(d),delta_vs_first_median=statistics.median(d),faster_rounds=sum(x<0 for x in d),rounds=len(d),
             resources=o['resources'],check=o['check'])
    rows.append(row)
    print(f'[ab] {a.stage} flags="{row["flags"]}" tiles={o["tiles"]}: median {row["median"]:.3f} mean {row["mean"]:.3f} min {row["min"]:.3f} ms; '
          f'vs first {row["delta_vs_first_mean"]:+.3f} (median {row["delta_vs_first_median"]:+.3f}), faster {row["faster_rounds"]}/{row["rounds"]}',flush=True)
if a.out:
    with open(a.out,'a') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
