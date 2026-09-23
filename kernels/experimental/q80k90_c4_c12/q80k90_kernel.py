"""Experimental Q80/K90 exact C4/C6/C8/C10/C12 optimization.

The shared CUDA merge, FP64/FP32 selectors, and layout fusion are independently
configurable. Candidate paths retain exact capacity and full compressed
attention; only configurations recorded in GPU evidence are validated.
"""
import ctypes
import functools
import hashlib
import os
from pathlib import Path
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib

@functools.lru_cache(maxsize=64)
def native_merge(capacity,t,h,ah,seed_tile=8,fast_math=False,direct_store=False,min_blocks=2,owner_mode=0,fused_kv=False,warp_groups=1,carveout=-1,vector_store=False,shared_q=False,hoist_seeds=False,configure_once=False,reuse_exp=False,sum_groups=1):
    from torch.utils.cpp_extension import load
    ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libstdc++.so.6',mode=ctypes.RTLD_GLOBAL)
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST','9.0a')
    os.environ.setdefault('MAX_JOBS','2')
    os.environ.setdefault('TORCH_EXTENSIONS_DIR','/local2/denghaoran/GridMergeAttention/q80c4_c12_v4_extensions')
    source=Path(__file__).with_name('q80k90_merge.cu')
    flags=['-O3','--expt-relaxed-constexpr','-lineinfo','--ptxas-options=-v',
           f'-DAXIS_CAPACITY={capacity}',f'-DAXIS_SEED_TILE={seed_tile}',
           f'-DAXIS_FAST_MATH={int(fast_math)}',f'-DAXIS_DIRECT_STORE={int(direct_store)}',
           f'-DAXIS_MIN_BLOCKS={min_blocks}',f'-DAXIS_OUTER={t}',
           f'-DAXIS_OWNER_MODE={owner_mode}',
           f'-DAXIS_FUSED_KV={int(fused_kv)}',
           f'-DAXIS_WARP_GROUPS={warp_groups}',f'-DAXIS_CARVEOUT={carveout}',
           f'-DAXIS_VECTOR_STORE={int(vector_store)}',
           f'-DAXIS_SHARED_Q={int(shared_q)}',
           f'-DAXIS_HOIST_SEEDS={int(hoist_seeds)}',
           f'-DAXIS_CONFIGURE_ONCE={int(configure_once)}',
           f'-DAXIS_REUSE_EXP={int(reuse_exp)}',
           f'-DAXIS_SUM_GROUPS={sum_groups}',
           '-DAXIS_ROWS=40','-DAXIS_COLS=90',f'-DAXIS_HEADS={h}',f'-DAXIS_ACTIVE_HEADS={ah}']
    digest=hashlib.sha256(source.read_bytes()+' '.join(flags).encode()).hexdigest()[:16]
    cutlass=Path(torch.__file__).resolve().parent.parent/'flashinfer/data/cutlass/include'
    return load(name='q80c4c12_v4_'+digest,sources=[str(source)],extra_include_paths=[str(cutlass)],extra_cuda_cflags=flags,verbose=False)

def candidate_table():
    def original_pool(anchor):
        chosen={anchor}
        chosen.update(range(max(0,anchor-11),min(80,anchor+12)))
        chosen.update(round(i*79/10) for i in range(11))
        while len(chosen)<32:
            chosen.add(max(set(range(80))-chosen,key=lambda x:(min(abs(x-y) for y in chosen),-x)))
        return chosen
    tables=[]
    for group in range(40):
        first=2*group; second=first+1
        chosen={first,second};pool=original_pool(first)|chosen
        if len(pool)<=32:chosen=pool.copy()
        while len(chosen)<32:
            available=pool-chosen
            if not available:available=set(range(80))-chosen
            chosen.add(max(available,key=lambda x:(min(abs(x-y) for y in chosen),-x)))
        tables.append(sorted(chosen))
    return tables



@tr.jit
def _pack_inputs(Q,K,V,QP,KP,VP,Packed,T:tl.constexpr,H:tl.constexpr,B:tl.constexpr):
    x=tl.program_id(0)*B+tl.arange(0,B)
    total=H*T*3600*128
    d=x%128;token=x//128;head=token//(T*3600)
    pos=token%(T*3600)
    src=((pos*H+head)*128+d).to(tl.int64)
    q=tl.load(Q+src,x<total,0)
    tl.store(QP+x,q,x<total)
    # Involution: transpose the 4x4 uint32 words in each 32-BF16 slice.
    pd=(d//32)*32+((d%8)//2)*8+((d%32)//8)*2+d%2
    tl.store(Packed+(token*128+pd).to(tl.int64),q,x<total)
    frame=pos//3600;within=pos%3600;pair=within//90;j=within%90
    original=((frame*45+j%45)*80+pair*2+j//45)
    kvsrc=((original*H+head)*128+d).to(tl.int64)
    tl.store(KP+x,tl.load(K+kvsrc,x<total,0),x<total)
    tl.store(VP+x,tl.load(V+kvsrc,x<total,0),x<total)

@tr.jit
def _select(Q,K,Table,Seeds,T:tl.constexpr,H:tl.constexpr,BG:tl.constexpr,CAP:tl.constexpr,FP64:tl.constexpr):
    # Query rows share K probes; all seed indices come from frame 0.
    kg=tl.program_id(0);gb=tl.program_id(1);head=tl.program_id(2)
    row=kg%40
    r=tl.arange(0,BG*32);c=r%32;g=gb*BG+r//32;d=tl.arange(0,128);p=tl.arange(0,16)
    candidates=tl.load(Table+row*32+c)
    qs=tl.load(Q+((head*T*45+g[:,None])*80+candidates[:,None])*128+d[None,:],g[:,None]<45,0)
    ks=tl.load(K+((head*T*40+kg)*90+p[None,:]*90//16)*128+d[:,None])
    z=(tl.dot(qs,ks)*0.08838834764831845).reshape(BG,32,16)
    if FP64:
        z=z.to(tl.float64)
    cids=candidates.reshape(BG,32)
    za=tl.sum(tl.where((cids==2*row)[:,:,None],z,0.),1)
    if FP64:
        ex=lib.exp(za-tl.max(za,1)[:,None])
    else:
        ex=tl.exp(za-tl.max(za,1)[:,None])
    rho=ex/tl.sum(ex,1)[:,None]
    delta=z-za[:,None,:]
    residual=(delta-tl.sum(delta*rho[:,None,:],2)[:,:,None])*lib.sqrt(rho)[:,None,:]
    used=cids==2*row
    gg=gb*BG+tl.arange(0,BG)
    tl.store(Seeds+((head*45+gg)*T*40+kg)*CAP,2*row,gg<45)
    for slot in tl.static_range(1,CAP):
        scores=tl.sum(residual*residual,2)
        scores=tl.where(used,-float('inf'),scores)
        picked=tl.argmax(scores,1,tie_break_left=True)
        if slot==1:
            picked=tl.sum(tl.where(cids==2*row+1,tl.arange(0,32)[None,:],0),1)
        chosen=tl.sum(tl.where(tl.arange(0,32)[None,:]==picked[:,None],cids,0),1)
        tl.store(Seeds+((head*45+gg)*T*40+kg)*CAP+slot,chosen,gg<45)
        used=used|(tl.arange(0,32)[None,:]==picked[:,None])
        if slot<CAP-1:
            direction=tl.sum(tl.where((tl.arange(0,32)[None,:]==picked[:,None])[:,:,None],residual,0.),1)
            basis=direction/lib.sqrt(tl.sum(direction*direction,1)+1.e-12)[:,None]
            residual=residual-tl.sum(residual*basis[:,None,:],2)[:,:,None]*basis[:,None,:]


@tr.jit(do_not_specialize=['HS'])
def _merge(Q,K,V,Seeds,MK,MV,Bias,T:tl.constexpr,H:tl.constexpr,AH:tl.constexpr,HS,
           BG:tl.constexpr,KG_LOOP:tl.constexpr,CAP:tl.constexpr):
    ktile=tl.program_id(0);gframe=tl.program_id(1);lh=tl.program_id(2);head=HS+lh
    gb=gframe%tl.cdiv(45,BG);qt=gframe//tl.cdiv(45,BG)
    r=tl.arange(0,BG*16);g=gb*BG+r//16;c=r%16;d=tl.arange(0,128);j=tl.arange(0,128)
    for it in range(KG_LOOP):
        kg=ktile*KG_LOOP+it
        if kg<T*40:
            seed=tl.load(Seeds+((head*45+g)*T*40+kg)*CAP+c,(c<CAP)&(g<45),0).to(tl.int32)
            qs=tl.load(Q+(((head*T+qt)*45+g[:,None])*80+seed[:,None])*128+d[None,:],((c<CAP)&(g<45))[:,None],0)
            kk=tl.load(K+((head*T*40+kg)*90+j[None,:])*128+d[:,None],j[None,:]<90,0)
            z=tl.dot(qs,kk)*0.08838834764831845
            z=tl.where(j[None,:]<90,z,-float('inf'))
            mx=tl.max(z,1)
            den=tl.sum(tl.exp(z-mx[:,None]),1)
            normalized=z-mx[:,None]-tl.log(den)[:,None]
            normalized=tl.where((c<CAP)[:,None],normalized,-float('inf')).reshape(BG,16,128)
            owner=tl.argmax(normalized,1,tie_break_left=True)
            assigned=(tl.arange(0,16)[None,:,None]==owner[:,None,:])&(j[None,None,:]<90)&(tl.arange(0,16)[None,:,None]<CAP)
            assigned=assigned.reshape(BG*16,128)
            selected=tl.where(assigned,z,-float('inf'))
            active=tl.sum(assigned.to(tl.int32),1)>0
            sm=tl.where(active,tl.max(selected,1),0.)
            weight=tl.exp(selected-sm[:,None]);wd=tl.sum(weight,1)
            weight=weight/tl.maximum(wd[:,None],1.e-30)
            entropy=-tl.sum(weight*tl.log(tl.maximum(weight,1.e-30)),1)
            kv=tl.trans(kk)
            vv=tl.load(V+((head*T*40+kg)*90+j[:,None])*128+d[None,:],j[:,None]<90,0)
            mk=tl.dot(weight.to(tl.bfloat16),kv)
            mv=tl.dot(weight.to(tl.bfloat16),vv)
            off=(((lh*T+qt)*45+g)*T*40+kg)*CAP+c
            tl.store(MK+off[:,None].to(tl.int64)*128+d[None,:],mk,((c<CAP)&(g<45))[:,None])
            tl.store(MV+off[:,None].to(tl.int64)*128+d[None,:],mv,((c<CAP)&(g<45))[:,None])
            tl.store(Bias+off,tl.where(active,entropy,-float('inf')),(c<CAP)&(g<45))


@tr.jit(do_not_specialize=['HS'])
def _route(Q,MK,MV,Bias,O,T:tl.constexpr,H:tl.constexpr,AH:tl.constexpr,HS,
           BM:tl.constexpr,BN:tl.constexpr,CAP:tl.constexpr):
    group=tl.program_id(0);lh=tl.program_id(1);mb=tl.program_id(2)
    head=HS+lh
    i=mb*BM+tl.arange(0,BM);d=tl.arange(0,128);n=tl.arange(0,BN)
    q=tl.load(Q+((head*T*45+group)*80+i[:,None])*128+d[None,:],i[:,None]<80,0)
    base=((lh*T*45+group)*T*40*CAP).to(tl.int64)
    acc=tl.full((BM,128),0.,tl.float32);maximum=tl.full((BM,),-float('inf'),tl.float32);denom=tl.full((BM,),0.,tl.float32)
    for start in range(tl.cdiv(T*40*CAP,BN)):
        ids=start*BN+n
        k=tl.load(MK+(base+ids[None,:])*128+d[:,None],ids[None,:]<T*40*CAP,0)
        bias=tl.load(Bias+base+ids,ids<T*40*CAP,-float('inf'))
        score=tl.dot(q,k)*0.12751743082459866+bias[None,:]*1.4426950408889634
        newmax=tl.maximum(maximum,tl.max(score,1))
        p=tl.exp2(score-newmax[:,None]);alpha=tl.exp2(maximum-newmax)
        value=tl.load(MV+(base+ids[:,None])*128+d[None,:],ids[:,None]<T*40*CAP,0)
        acc=acc*alpha[:,None]+tl.dot(p.to(tl.bfloat16),value)
        denom=denom*alpha+tl.sum(p,1);maximum=newmax
    tl.store(O+(((group*80+i[:,None])*H+head)*128+d[None,:]).to(tl.int64),acc/denom[:,None],i[:,None]<80)


@tr.jit
def _route_tile(Q,MK,MV,Bias,O,T:tl.constexpr,H:tl.constexpr,AH:tl.constexpr,HS,
                group,lh,START:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,CAP:tl.constexpr):
    head=HS+lh
    i=START+tl.arange(0,BM);d=tl.arange(0,128);n=tl.arange(0,BN)
    q=tl.load(Q+((head*T*45+group)*80+i[:,None])*128+d[None,:],i[:,None]<80,0)
    base=((lh*T*45+group)*T*40*CAP).to(tl.int64)
    acc=tl.full((BM,128),0.,tl.float32);maximum=tl.full((BM,),-float('inf'),tl.float32);denom=tl.full((BM,),0.,tl.float32)
    for start in range(tl.cdiv(T*40*CAP,BN)):
        ids=start*BN+n
        k=tl.load(MK+(base+ids[None,:])*128+d[:,None],ids[None,:]<T*40*CAP,0)
        bias=tl.load(Bias+base+ids,ids<T*40*CAP,-float('inf'))
        score=tl.dot(q,k)*0.12751743082459866+bias[None,:]*1.4426950408889634
        newmax=tl.maximum(maximum,tl.max(score,1))
        p=tl.exp2(score-newmax[:,None]);alpha=tl.exp2(maximum-newmax)
        value=tl.load(MV+(base+ids[:,None])*128+d[None,:],ids[:,None]<T*40*CAP,0)
        acc=acc*alpha[:,None]+tl.dot(p.to(tl.bfloat16),value)
        denom=denom*alpha+tl.sum(p,1);maximum=newmax
    tl.store(O+(((group*80+i[:,None])*H+head)*128+d[None,:]).to(tl.int64),acc/denom[:,None],i[:,None]<80)


@tr.jit(do_not_specialize=['HS'])
def _route_exact80(Q,MK,MV,Bias,O,T:tl.constexpr,H:tl.constexpr,AH:tl.constexpr,HS,
                   BN:tl.constexpr,CAP:tl.constexpr):
    pair=tl.program_id(0);group=pair//2;lh=tl.program_id(1)
    if pair%2==0:
        _route_tile(Q,MK,MV,Bias,O,T,H,AH,HS,group,lh,0,64,BN,CAP)
    else:
        _route_tile(Q,MK,MV,Bias,O,T,H,AH,HS,group,lh,64,16,BN,CAP)


@tr.jit(do_not_specialize=['HS'])
def _augment_route_q(Q,A,T:tl.constexpr,AH:tl.constexpr,HS,B:tl.constexpr):
    x=tl.program_id(0)*B+tl.arange(0,B)
    d=x%136;row=x//136
    val=tl.load(Q+((HS*T*45*80+row)*128+d).to(tl.int64),(x<AH*T*45*80*136)&(d<128),0)
    val=tl.where((d==128)|(d==129),16.,val)
    tl.store(A+x,val,x<AH*T*45*80*136)


@tr.jit
def _augment_route_kv(K,V,Bias,AK,AV,ROWS:tl.constexpr,B:tl.constexpr):
    x=tl.program_id(0).to(tl.int64)*B+tl.arange(0,B);d=x%136;row=x//136
    valid=x<ROWS*136
    k=tl.load(K+(row*128+d).to(tl.int64),valid&(d<128),0)
    v=tl.load(V+(row*128+d).to(tl.int64),valid&(d<128),0)
    bias=tl.load(Bias+row,valid&(d>=128)&(d<130),0)*0.7071067811865476
    high=bias.to(tl.bfloat16).to(tl.float32)
    low=tl.where(bias==float('-inf'),0.,bias-high)
    k=tl.where(d==128,high,tl.where(d==129,low,k))
    tl.store(AK+x,k,valid);tl.store(AV+x,v,valid)


@tr.jit(do_not_specialize=['HS'])
def _scatter_route(A,O,T:tl.constexpr,H:tl.constexpr,AH:tl.constexpr,HS,D:tl.constexpr,B:tl.constexpr):
    x=tl.program_id(0)*B+tl.arange(0,B);d=x%128;row=x//128
    head=HS+row//(T*45*80);token=row%(T*45*80)
    val=tl.load(A+(row*D+d).to(tl.int64),x<AH*T*45*80*128,0)
    tl.store(O+((token*H+head)*128+d).to(tl.int64),val,x<AH*T*45*80*128)

class Q80K90:
    def __init__(self,capacity,head_chunk=4,merge_groups=4,merge_loop=1,
                 route_m=128,route_n=64,merge_warps=4,route_warps=4,
                 merge_backend='cuda',native_tiles=63,seed_tile=None,
                 selector_precision='fp64',selector_groups=4,
                 fast_math=False,direct_store=False,min_blocks=2,fused_layout=True,owner_mode=0,fused_kv=False,
                 route_backend='triton',warp_groups=1,carveout=-1,vector_store=False,shared_q=False,hoist_seeds=False,
                 pipeline=False,configure_once=False,reuse_exp=False,route_stages=2,sum_groups=1):
        assert capacity in (4,6,8,10,12)
        if seed_tile is None:seed_tile=8 if capacity<=8 else 16
        assert head_chunk>0 and native_tiles>0 and merge_loop>0
        assert merge_groups in (1,2,4,8) and selector_groups in (1,2,4,8)
        assert route_m in (32,64,128) and route_n in (32,64,128)
        assert merge_warps in (4,8) and route_warps in (4,8)
        assert seed_tile in (4,8,16) and capacity<=seed_tile and min_blocks in (1,2,3,4)
        assert warp_groups in (1,2,3,4) and carveout in (-1,0,25,50,75,100)
        assert not vector_store or direct_store
        assert not hoist_seeds or not shared_q
        assert owner_mode in (0,1,2) and (owner_mode!=2 or seed_tile<=8)
        assert route_backend in ('triton','exact80','flash_augmented','cudnn_bias','efficient_bias','flashinfer_fa3')
        assert selector_precision in ('fp64','fp32') and merge_backend in ('triton','cuda')
        self.cap=capacity;self.hc=head_chunk;self.bg=merge_groups;self.kl=merge_loop
        self.bm=route_m;self.bn=route_n;self.mw=merge_warps;self.rw=route_warps
        self.backend=merge_backend;self.nt=native_tiles;self.st=seed_tile
        self.fp64=selector_precision=='fp64';self.sg=selector_groups
        self.fm=fast_math;self.ds=direct_store;self.mb=min_blocks;self.fl=fused_layout
        self.om=owner_mode;self.fkv=fused_kv
        self.rb=route_backend
        self.wg=warp_groups;self.co=carveout
        self.vs=vector_store
        self.sq=shared_q
        self.hoist=hoist_seeds
        self.pipe=pipeline
        self.once=configure_once
        self.rexp=reuse_exp
        assert sum_groups in (1,2,4,8)
        self.sumg=sum_groups
        assert route_stages in (1,2,3,4,5)
        self.rs=route_stages
        self.cache=None

    def native(self,t,h,ah):
        return native_merge(self.cap,t,h,ah,self.st,self.fm,self.ds,self.mb,self.om,self.fkv,self.wg,self.co,self.vs,self.sq,self.hoist,self.once,self.rexp,self.sumg)

    def prepare(self,q):
        b,n,h,d=q.shape;t=n//3600
        assert b==1 and t>0 and h>0 and n==t*3600 and d==128
        assert q.dtype==torch.bfloat16 and q.is_cuda
        # The workspace is reusable on the stream that constructed it. Separate
        # instances are required for concurrent calls, including other streams.
        key=(t,h,q.device,torch.cuda.current_stream(q.device).cuda_stream)
        if self.cache!=key:
            ah=min(self.hc,h)
            self.table=torch.tensor(candidate_table(),device=q.device,dtype=torch.int32)
            self.seeds=torch.empty((h,45,t*40,self.cap),device=q.device,dtype=torch.uint8)
            shape=(ah,t,45,t*40,self.cap,128)
            self.mk=torch.empty(shape,device=q.device,dtype=torch.bfloat16)
            self.mv=torch.empty_like(self.mk)
            self.bias=torch.empty(shape[:-1],device=q.device,dtype=torch.float32)
            self.fi_routes={}
            if self.pipe:
                self.banks=[(self.mk,self.mv,self.bias),
                            (torch.empty_like(self.mk),torch.empty_like(self.mv),torch.empty_like(self.bias))]
                self.merge_stream=torch.cuda.Stream(device=q.device)
                self.merged_events=[torch.cuda.Event(),torch.cuda.Event()]
                self.free_events=[torch.cuda.Event(),torch.cuda.Event()]
            if self.rb=='flash_augmented':
                self.rq=torch.empty((ah*t*45,1,80,136),device=q.device,dtype=q.dtype)
                self.rk=torch.empty((ah*t*45,1,t*40*self.cap,136),device=q.device,dtype=q.dtype)
                self.rv=torch.empty_like(self.rk)
            if self.fl:
                self.qp=torch.empty((h,t,45,80,128),device=q.device,dtype=q.dtype)
                self.kp=torch.empty((h,t*40,90,128),device=q.device,dtype=q.dtype)
                self.vp=torch.empty_like(self.kp);self.packed=torch.empty_like(self.qp)
            self.cache=key
        return t,h

    def route(self,qp,out,t,h,ah,hs):
        if self.rb=='triton':
            try:
                _route[(t*45,ah,tr.cdiv(80,self.bm))](qp,self.mk,self.mv,self.bias,out,t,h,ah,hs,self.bm,self.bn,self.cap,num_warps=self.rw,num_stages=self.rs)
            except Exception as exc:
                if type(exc).__name__ in ('OutOfResources','CompilationError'):
                    raise AssertionError(f'Invalid route configuration: {exc}') from exc
                raise
            return
        if self.rb=='exact80':
            _route_exact80[(t*45*2,ah)](qp,self.mk,self.mv,self.bias,out,t,h,ah,hs,self.bn,self.cap,num_warps=self.rw,num_stages=self.rs)
            return
        from torch.nn.attention import sdpa_kernel,SDPBackend
        groups=ah*t*45;length=t*40*self.cap
        if self.rb=='flashinfer_fa3':
            from q80_flashinfer_route import EntropyRoute
            if ah not in self.fi_routes:
                try:
                    self.fi_routes[ah]=EntropyRoute(groups,length,qp.device)
                except (RuntimeError,ImportError) as exc:
                    if 'illegal memory' in str(exc).lower() or 'device-side' in str(exc).lower():raise
                    raise AssertionError(f'FlashInfer route initialization failed: {exc}') from exc
            result=self.fi_routes[ah](qp[hs:hs+ah].reshape(groups*80,1,128),
                self.mk[:ah].reshape(groups*length,1,128),self.mv[:ah].reshape(groups*length,1,128),
                self.bias[:ah].reshape(-1))
            _scatter_route[(tr.cdiv(groups*80*128,1024),)](result,out,t,h,ah,hs,128,1024)
            return
        if self.rb=='flash_augmented':
            # Two BF16 terms encode the FP32 entropy bias. Q's exact factor
            # 16, combined with the original scale, decodes the sum. This is
            # the same full compressed attention, with no additional routes.
            _augment_route_q[(tr.cdiv(groups*80*136,1024),)](qp,self.rq,t,ah,hs,1024)
            _augment_route_kv[(tr.cdiv(groups*length*136,1024),)](
                self.mk,self.mv,self.bias,self.rk,self.rv,groups*length,1024)
            q=self.rq[:groups];k=self.rk[:groups];v=self.rv[:groups]
            mask=None;dim=136;backend=SDPBackend.FLASH_ATTENTION
        else:
            q=qp[hs:hs+ah].reshape(groups,1,80,128)
            k=self.mk[:ah].reshape(groups,1,length,128)
            v=self.mv[:ah].reshape(groups,1,length,128)
            # Backend bias dtype must match Q. Finite -1e30 still underflows
            # to an exact zero softmax weight for an empty representative.
            mask=self.bias[:ah].reshape(groups,1,1,length).clamp_min(-1.e30).to(q.dtype)
            dim=128
            backend=SDPBackend.CUDNN_ATTENTION if self.rb=='cudnn_bias' else SDPBackend.EFFICIENT_ATTENTION
        try:
            with sdpa_kernel(backend):
                result=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask,scale=128**-.5).contiguous()
        except (RuntimeError,NotImplementedError) as exc:
            raise AssertionError(f'{self.rb} unavailable: {exc}') from exc
        _scatter_route[(tr.cdiv(groups*80*128,1024),)](result,out,t,h,ah,hs,dim,1024)

    @torch.no_grad()
    def __call__(self,q,k,v,return_debug=False):
        assert k.shape==v.shape==q.shape and k.dtype==v.dtype==q.dtype
        assert k.device==v.device==q.device
        with torch.cuda.device(q.device):
            t,h=self.prepare(q)
            if self.fl:
                q,k,v=[x.contiguous() for x in (q,k,v)]
                qp,kp,vp,packed=self.qp,self.kp,self.vp,self.packed
                _pack_inputs[(tr.cdiv(q.numel(),1024),)](q,k,v,qp,kp,vp,packed,t,h,1024)
            else:
                qp=q.reshape(t,45,80,h,128).permute(3,0,1,2,4).contiguous()
                kp=k.reshape(t,45,40,2,h,128).permute(4,0,2,3,1,5).reshape(h,t*40,90,128).contiguous()
                vp=v.reshape(t,45,40,2,h,128).permute(4,0,2,3,1,5).reshape(h,t*40,90,128).contiguous()
                if self.backend=='cuda':
                    packed=torch.empty_like(qp);self.native(t,h,min(h,self.hc)).pack(qp,packed)
            out=torch.empty((1,t*3600,h,128),device=q.device,dtype=q.dtype)
            _select[(t*40,tr.cdiv(45,self.sg),h)](qp,kp,self.table,self.seeds,t,h,self.sg,self.cap,self.fp64,num_warps=4,enable_fp_fusion=False)
            if self.pipe:
                assert self.backend=='cuda','Pipeline currently supports native merge only'
                caller=torch.cuda.current_stream(q.device)
                self.merge_stream.wait_stream(caller)
                for chunk,hs in enumerate(range(0,h,self.hc)):
                    slot=chunk%2;ah=min(self.hc,h-hs)
                    self.mk,self.mv,self.bias=self.banks[slot]
                    with torch.cuda.stream(self.merge_stream):
                        if chunk>=2:self.merge_stream.wait_event(self.free_events[slot])
                        self.native(t,h,ah).run(qp if self.sq else packed,kp,vp,self.seeds,self.mk,self.mv,self.bias,hs,ah,t if self.hoist else self.nt)
                        self.merged_events[slot].record()
                    caller.wait_event(self.merged_events[slot])
                    self.route(qp,out,t,h,ah,hs)
                    self.free_events[slot].record(caller)
                return (out,self.seeds) if return_debug else out
            for hs in range(0,h,self.hc):
                ah=min(self.hc,h-hs)
                if self.backend=='cuda':
                    self.native(t,h,ah).run(qp if self.sq else packed,kp,vp,self.seeds,self.mk,self.mv,self.bias,hs,ah,t if self.hoist else self.nt)
                else:
                    _merge[(tr.cdiv(t*40,self.kl),t*tr.cdiv(45,self.bg),ah)](qp,kp,vp,self.seeds,self.mk,self.mv,self.bias,t,h,ah,hs,self.bg,self.kl,self.cap,num_warps=self.mw,num_stages=2)
                self.route(qp,out,t,h,ah,hs)
            return (out,self.seeds) if return_debug else out
