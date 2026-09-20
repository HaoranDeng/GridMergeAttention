"""Q90K80C14, RF0 two-intersection Fisher, exact 14 representatives.

Inference-only full compressed attention for THW 21x45x80, D128.
No dense exceptions. Padding to 16 in the tensor-core tile is never stored
as an extra representative. Empty representatives have zero KV and -inf bias.
"""
import os
import ctypes,functools,hashlib
from pathlib import Path
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib

@functools.lru_cache(maxsize=8)
def native_merge(t,h,ah):
    from torch.utils.cpp_extension import load
    ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libstdc++.so.6',mode=ctypes.RTLD_GLOBAL)
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST','9.0a');os.environ.setdefault('MAX_JOBS','2')
    os.environ.setdefault('TORCH_EXTENSIONS_DIR','/local2/denghaoran/GridMergeAttention/q90c14_cuda_extensions_20260920')
    source=Path(__file__).with_name('q90k80c14_merge.cu')
    flags=['-O3','--expt-relaxed-constexpr','-lineinfo',f'-DAXIS_OUTER={t}','-DAXIS_ROWS=45','-DAXIS_COLS=80',f'-DAXIS_HEADS={h}',f'-DAXIS_ACTIVE_HEADS={ah}']
    digest=hashlib.sha256(source.read_bytes()+' '.join(flags).encode()).hexdigest()[:16]
    cutlass=Path(torch.__file__).resolve().parent.parent/'flashinfer/data/cutlass/include'
    return load(name='q90c14_merge_'+digest,sources=[str(source)],extra_include_paths=[str(cutlass)],extra_cuda_cflags=flags,verbose=False)


def candidate_table():
    tables=[]
    for anchor in range(45):
        pool={anchor}|set(range(max(0,anchor-11),min(45,anchor+12)))|{round(i*44/10) for i in range(11)}
        while len(pool)<32:
            pool.add(max(set(range(45))-pool,key=lambda x:(min(abs(x-y) for y in pool),-x)))
        chosen={anchor}
        while len(chosen)<16:
            chosen.add(max(pool-chosen,key=lambda x:(min(abs(x-y) for y in chosen),-x)))
        first=sorted(chosen)
        tables.append(first+[x+45 for x in first])
    return tables


@tr.jit
def _select(Q,K,Table,Seeds,T:tl.constexpr,H:tl.constexpr,BG:tl.constexpr):
    # Four query groups share the same K probes; all indices come from frame 0.
    kg=tl.program_id(0);gb=tl.program_id(1);head=tl.program_id(2)
    row=kg%45
    r=tl.arange(0,BG*32);c=r%32;g=gb*BG+r//32;d=tl.arange(0,128);p=tl.arange(0,16)
    candidates=tl.load(Table+row*32+c)
    qs=tl.load(Q+((head*T*40+g[:,None])*90+candidates[:,None])*128+d[None,:])
    ks=tl.load(K+((head*T*45+kg)*80+p[None,:]*5)*128+d[:,None])
    z=(tl.dot(qs,ks)*0.08838834764831845).to(tl.float64).reshape(BG,32,16)
    cids=candidates.reshape(BG,32)
    za=tl.sum(tl.where((cids==row)[:,:,None],z,0.),1)
    ex=lib.exp(za-tl.max(za,1)[:,None]);rho=ex/tl.sum(ex,1)[:,None]
    delta=z-za[:,None,:]
    residual=(delta-tl.sum(delta*rho[:,None,:],2)[:,:,None])*lib.sqrt(rho)[:,None,:]
    used=cids==row
    gg=gb*BG+tl.arange(0,BG)
    tl.store(Seeds+((head*40+gg)*T*45+kg)*14,row)
    for slot in tl.static_range(1,14):
        scores=tl.sum(residual*residual,2)
        scores=tl.where(used,-float('inf'),scores)
        picked=tl.argmax(scores,1,tie_break_left=True)
        if slot==1:
            picked=tl.sum(tl.where(cids==row+45,tl.arange(0,32)[None,:],0),1)
        chosen=tl.sum(tl.where(tl.arange(0,32)[None,:]==picked[:,None],cids,0),1)
        tl.store(Seeds+((head*40+gg)*T*45+kg)*14+slot,chosen)
        used=used|(tl.arange(0,32)[None,:]==picked[:,None])
        if slot<13:
            direction=tl.sum(tl.where((tl.arange(0,32)[None,:]==picked[:,None])[:,:,None],residual,0.),1)
            basis=direction/lib.sqrt(tl.sum(direction*direction,1)+1.e-12)[:,None]
            residual=residual-tl.sum(residual*basis[:,None,:],2)[:,:,None]*basis[:,None,:]


@tr.jit(do_not_specialize=['HS'])
def _merge(Q,K,V,Seeds,MK,MV,Bias,T:tl.constexpr,H:tl.constexpr,AH:tl.constexpr,HS,
           BG:tl.constexpr,KG_LOOP:tl.constexpr):
    ktile=tl.program_id(0);gframe=tl.program_id(1);lh=tl.program_id(2);head=HS+lh
    gb=gframe%tl.cdiv(40,BG);qt=gframe//tl.cdiv(40,BG)
    r=tl.arange(0,BG*16);g=gb*BG+r//16;c=r%16;d=tl.arange(0,128);j=tl.arange(0,128)
    for it in range(KG_LOOP):
        kg=ktile*KG_LOOP+it
        if kg<T*45:
            seed=tl.load(Seeds+((head*40+g)*T*45+kg)*14+c,c<14,0).to(tl.int32)
            qs=tl.load(Q+(((head*T+qt)*40+g[:,None])*90+seed[:,None])*128+d[None,:],(c<14)[:,None],0)
            kk=tl.load(K+((head*T*45+kg)*80+j[None,:])*128+d[:,None],j[None,:]<80,0)
            z=tl.dot(qs,kk)*0.08838834764831845
            z=tl.where(j[None,:]<80,z,-float('inf'))
            mx=tl.max(z,1)
            den=tl.sum(tl.exp(z-mx[:,None]),1)
            normalized=z-mx[:,None]-tl.log(den)[:,None]
            normalized=tl.where((c<14)[:,None],normalized,-float('inf')).reshape(BG,16,128)
            owner=tl.argmax(normalized,1,tie_break_left=True)
            assigned=(tl.arange(0,16)[None,:,None]==owner[:,None,:])&(j[None,None,:]<80)&(tl.arange(0,16)[None,:,None]<14)
            assigned=assigned.reshape(BG*16,128)
            selected=tl.where(assigned,z,-float('inf'))
            active=tl.sum(assigned.to(tl.int32),1)>0
            sm=tl.where(active,tl.max(selected,1),0.)
            weight=tl.exp(selected-sm[:,None]);wd=tl.sum(weight,1)
            weight=weight/tl.maximum(wd[:,None],1.e-30)
            entropy=-tl.sum(weight*tl.log(tl.maximum(weight,1.e-30)),1)
            kv=tl.trans(kk)
            vv=tl.load(V+((head*T*45+kg)*80+j[:,None])*128+d[None,:],j[:,None]<80,0)
            mk=tl.dot(weight.to(tl.bfloat16),kv)
            mv=tl.dot(weight.to(tl.bfloat16),vv)
            off=(((lh*T+qt)*40+g)*T*45+kg)*14+c
            tl.store(MK+off[:,None].to(tl.int64)*128+d[None,:],mk,(c<14)[:,None])
            tl.store(MV+off[:,None].to(tl.int64)*128+d[None,:],mv,(c<14)[:,None])
            tl.store(Bias+off,tl.where(active,entropy,-float('inf')),c<14)


@tr.jit(do_not_specialize=['HS'])
def _route(Q,MK,MV,Bias,O,T:tl.constexpr,H:tl.constexpr,AH:tl.constexpr,HS,
           BM:tl.constexpr,BN:tl.constexpr):
    group=tl.program_id(0);lh=tl.program_id(1);mb=tl.program_id(2)
    head=HS+lh
    i=mb*BM+tl.arange(0,BM);d=tl.arange(0,128);n=tl.arange(0,BN)
    q=tl.load(Q+((head*T*40+group)*90+i[:,None])*128+d[None,:],i[:,None]<90,0)
    base=((lh*T*40+group)*T*45*14).to(tl.int64)
    acc=tl.full((BM,128),0.,tl.float32);maximum=tl.full((BM,),-float('inf'),tl.float32);denom=tl.full((BM,),0.,tl.float32)
    for start in range(tl.cdiv(T*45*14,BN)):
        ids=start*BN+n
        k=tl.load(MK+(base+ids[None,:])*128+d[:,None],ids[None,:]<T*45*14,0)
        bias=tl.load(Bias+base+ids,ids<T*45*14,-float('inf'))
        score=tl.dot(q,k)*0.12751743082459866+bias[None,:]*1.4426950408889634
        newmax=tl.maximum(maximum,tl.max(score,1))
        p=tl.exp2(score-newmax[:,None]);alpha=tl.exp2(maximum-newmax)
        value=tl.load(MV+(base+ids[:,None])*128+d[None,:],ids[:,None]<T*45*14,0)
        acc=acc*alpha[:,None]+tl.dot(p.to(tl.bfloat16),value)
        denom=denom*alpha+tl.sum(p,1);maximum=newmax
    tl.store(O+((head*T*40+group)*90+i[:,None])*128+d[None,:],acc/denom[:,None],i[:,None]<90)


class Q90K80C14:
    def __init__(self,head_chunk=4,merge_groups=4,merge_loop=1,route_m=64,route_n=64,merge_warps=4,route_warps=4,merge_backend='triton',native_tiles=105):
        self.hc=head_chunk;self.bg=merge_groups;self.kl=merge_loop;self.bm=route_m;self.bn=route_n;self.mw=merge_warps;self.rw=route_warps
        assert 40%merge_groups==0 and merge_groups in (1,2,4,8)
        self.cache=None
        assert merge_backend in ('triton','cuda')
        self.backend=merge_backend;self.nt=native_tiles

    def prepare(self,q):
        b,n,h,d=q.shape;t=n//3600
        assert b==1 and n==t*3600 and d==128 and q.dtype==torch.bfloat16 and q.is_cuda
        key=(t,h,q.device)
        if self.cache!=key:
            ah=min(self.hc,h)
            self.table=torch.tensor(candidate_table(),device=q.device,dtype=torch.int32)
            self.seeds=torch.empty((h,40,t*45,14),device=q.device,dtype=torch.uint8)
            shape=(ah,t,40,t*45,14,128)
            self.mk=torch.empty(shape,device=q.device,dtype=torch.bfloat16);self.mv=torch.empty_like(self.mk)
            self.bias=torch.empty(shape[:-1],device=q.device,dtype=torch.float32)
            self.cache=key
        return t,h

    @torch.no_grad()
    def __call__(self,q,k,v,return_debug=False):
        t,h=self.prepare(q)
        assert k.shape==v.shape==q.shape and k.dtype==v.dtype==q.dtype
        qp=q.view(t,45,40,2,h,128).permute(4,0,2,3,1,5).contiguous()
        kp=k.view(t*45,80,h,128).permute(2,0,1,3).contiguous()
        vp=v.view(t*45,80,h,128).permute(2,0,1,3).contiguous()
        out=torch.empty_like(qp)
        _select[(t*45,10,h)](qp,kp,self.table,self.seeds,t,h,4,num_warps=4,enable_fp_fusion=False)
        if self.backend=='cuda':
            native=native_merge(t,h,min(h,self.hc));packed=torch.empty_like(qp);native.pack(qp,packed)
        for hs in range(0,h,self.hc):
            ah=min(self.hc,h-hs)
            if self.backend=='cuda':native_merge(t,h,ah).run(packed,kp,vp,self.seeds,self.mk,self.mv,self.bias,hs,ah,self.nt)
            else:_merge[(tr.cdiv(t*45,self.kl),t*tr.cdiv(40,self.bg),ah)](qp,kp,vp,self.seeds,self.mk,self.mv,self.bias,t,h,ah,hs,self.bg,self.kl,num_warps=self.mw,num_stages=2)
            _route[(t*40,ah,tr.cdiv(90,self.bm))](qp,self.mk,self.mv,self.bias,out,t,h,ah,hs,self.bm,self.bn,num_warps=self.rw,num_stages=2)
        result=out.view(h,t,40,2,45,128).permute(1,4,2,3,0,5).reshape(1,t*3600,h,128).contiguous()
        return (result,self.seeds) if return_debug else result
