"""Resume full-support online attention over compact C10 key partitions."""
import triton
import triton.language as tl

@triton.jit
def compact_route(Q,K,V,MK,MV,Bias,Acc,Max,Den,Out,head,column,key_begin,
                  FIRST:tl.constexpr,LAST:tl.constexpr,COLS:tl.constexpr,BM:tl.constexpr=16,BN:tl.constexpr=64):
    rows=tl.program_id(0)*BM+tl.arange(0,BM);group=tl.program_id(1)
    frame=group//COLS;local_col=group%COLS;col=column+local_col
    d=tl.arange(0,128);n=tl.arange(0,BN)
    qaddr=(((frame*45+rows[:,None])*80+col)*40+head)*128+d[None,:]
    query=tl.load(Q+qaddr,rows[:,None]<45,0)
    state=(frame*6+local_col)*45+rows
    if FIRST:
        acc=tl.full((BM,128),0,tl.float32);m=tl.full((BM,),-float('inf'),tl.float32);den=tl.full((BM,),0,tl.float32)
    else:
        acc=tl.load(Acc+state[:,None]*128+d[None,:],rows[:,None]<45,0)
        m=tl.load(Max+state,rows<45,-float('inf'));den=tl.load(Den+state,rows<45,0)
    base=((frame*6+local_col)*640).to(tl.int64)
    count=tl.minimum(64,945-key_begin)
    for start in range(0,640,BN):
        token=start+n;valid=token<count*10
        addr=(base+token[:,None])*128+d[None,:]
        kk=tl.load(MK+addr,valid[:,None],0);vv=tl.load(MV+addr,valid[:,None],0)
        bias=tl.load(Bias+base+token,valid,float('inf'))
        z=(tl.dot(query,tl.trans(kk))*(128**-.5)-bias[None,:])*1.4426950408889634
        nm=tl.maximum(m,tl.max(z,1));safe=tl.where(nm==-float('inf'),0.,nm)
        p=tl.exp2(z-safe[:,None]);old=tl.exp2(m-safe)
        acc=acc*old[:,None]+tl.dot(p.to(tl.bfloat16),vv)
        den=den*old+tl.sum(p,1);m=nm
    for start in range(0,64,BN):
        token=key_begin+start+n;valid=token<945
        addr=((token[:,None]*80+col)*40+head)*128+d[None,:]
        kk=tl.load(K+addr,valid[:,None],0);vv=tl.load(V+addr,valid[:,None],0)
        z=tl.dot(query,tl.trans(kk))*(128**-.5*1.4426950408889634)
        z=tl.where(valid[None,:],z,-float('inf'))
        nm=tl.maximum(m,tl.max(z,1));safe=tl.where(nm==-float('inf'),0.,nm)
        p=tl.exp2(z-safe[:,None]);old=tl.exp2(m-safe)
        acc=acc*old[:,None]+tl.dot(p.to(tl.bfloat16),vv)
        den=den*old+tl.sum(p,1);m=nm
    if LAST:
        tl.store(Out+qaddr,acc/den[:,None],rows[:,None]<45)
    else:
        tl.store(Acc+state[:,None]*128+d[None,:],acc,rows[:,None]<45)
        tl.store(Max+state,m,rows<45);tl.store(Den+state,den,rows<45)

def route_partition(q,k,v,mk,mv,bias,state,out,head,column,key_begin,block_m=16,num_warps=4):
    cols=min(6,80-column)
    compact_route[(triton.cdiv(45,block_m),21*cols)](q,k,v,mk,mv,bias,*state,out,head,column,key_begin,key_begin==0,key_begin>=896,cols,block_m,num_warps=num_warps,num_stages=3)
