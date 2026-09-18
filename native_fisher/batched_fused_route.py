"""Unvalidated C10 fusion: four query columns share each producer matrix tile.

Every key block remains in support. Partitions retain FP32 online-softmax state
and are combined exactly in real arithmetic. Normal FP rounding needs validation.
"""
import triton
import triton.language as tl


@triton.jit
def batched_fused_c10(Q,K,V,Seeds,Acc,Max,Den,HEAD_START:tl.constexpr,HEADS:tl.constexpr,
                      COLS:tl.constexpr=4,PARTS:tl.constexpr=16,BM:tl.constexpr=64):
    group=tl.program_id(0);local_head=tl.program_id(1)
    part=tl.program_id(2)//triton.cdiv(45,BM);row_tile=tl.program_id(2)%triton.cdiv(45,BM)
    frame=group//(80//COLS);first_col=(group%(80//COLS))*COLS
    head=HEAD_START+local_head
    cols=first_col+tl.arange(0,COLS)
    rows=row_tile*BM+tl.arange(0,BM);d=tl.arange(0,128);n=tl.arange(0,128)
    seed_flat=tl.arange(0,COLS*16);seed=seed_flat%16;seed_col=first_col+seed_flat//16
    acc=tl.full((COLS,BM,128),0,tl.float32)
    maximum=tl.full((COLS,BM),-float('inf'),tl.float32)
    denominator=tl.full((COLS,BM),0,tl.float32)
    key_begin=part*triton.cdiv(945,PARTS)
    key_end=tl.minimum(key_begin+triton.cdiv(945,PARTS),945)
    for kb in range(key_begin,key_end):
        sr=tl.load(Seeds+((kb*80+seed_col)*40+head)*10+seed,seed<10,0).to(tl.int32)
        sq=tl.load(Q+(((frame*45+sr[:,None])*80+seed_col[:,None])*40+head)*128+d[None,:])
        address=((kb*80+n[:,None])*40+head)*128+d[None,:]
        kk=tl.load(K+address,n[:,None]<80,0,eviction_policy='evict_last')
        vv=tl.load(V+address,n[:,None]<80,0,eviction_policy='evict_last')
        logits=tl.dot(sq,tl.trans(kk))*(128**-.5*1.4426950408889634)
        logits=tl.reshape(logits,(COLS,16,128))
        logits=tl.where(n[None,None,:]<80,logits,-float('inf'))
        mx=tl.max(logits,2)
        sm=tl.sum(tl.exp2(logits-mx[:,:,None]),2)
        norm=logits-(mx+tl.log2(sm))[:,:,None]
        seeds=tl.arange(0,16)
        norm=tl.where(seeds[None,:,None]<10,norm,-float('inf'))
        winner,owner=tl.max(norm,1,return_indices=True,return_indices_tie_break_left=True)
        prob=tl.where((seeds[None,:,None]==owner[:,None,:]) & (seeds[None,:,None]<10)
                      & (n[None,None,:]<80),tl.exp2(winner)[:,None,:],0.)
        total=tl.sum(prob,2);safe=tl.where(total>0,total,1.)
        entropy=tl.sum(prob*tl.where(n[None,None,:]<80,winner[:,None,:],0.),2)
        weights=prob/safe[:,:,None]
        singleton=tl.sum(tl.where(n[None,None,:]==cols[:,None,None],weights,0.),2)
        residual=1.-singleton
        valid=(total>0)&(residual>1.e-6)&(seeds[None,:]<10)
        safe_residual=tl.where(valid,residual,1.)
        bias=(entropy/safe-tl.log2(safe))*0.6931471805599453
        selected_entropy=singleton*tl.log2(tl.maximum(singleton,1.e-20))*0.6931471805599453
        bias=(bias-selected_entropy)/safe_residual-tl.log2(safe_residual)*0.6931471805599453
        weights=tl.reshape(tl.where(n[None,None,:]==cols[:,None,None],0.,weights),(COLS*16,128)).to(tl.bfloat16)
        mk=tl.reshape(tl.dot(weights,kk),(COLS,16,128))/safe_residual[:,:,None]
        mv=tl.reshape(tl.dot(weights,vv),(COLS,16,128))/safe_residual[:,:,None]
        mk=mk.to(tl.bfloat16);mv=mv.to(tl.bfloat16)
        singleton_k=tl.load(K+((kb*80+cols[:,None])*40+head)*128+d[None,:])
        singleton_v=tl.load(V+((kb*80+cols[:,None])*40+head)*128+d[None,:])
        mk=tl.where(seeds[None,:,None]==10,singleton_k[:,None,:],mk)
        mv=tl.where(seeds[None,:,None]==10,singleton_v[:,None,:],mv)
        bias=tl.where(seeds[None,:]==10,0.,tl.where(valid,bias,float('inf')))
        qa=(((frame*45+rows[None,:,None])*80+cols[:,None,None])*40+head)*128+d[None,None,:]
        query=tl.load(Q+qa,rows[None,:,None]<45,0,eviction_policy='evict_last')
        score=(tl.dot(query,tl.permute(mk,(0,2,1)))*(128**-.5)-bias[:,None,:])*1.4426950408889634
        nm=tl.maximum(maximum,tl.max(score,2))
        p=tl.exp2(score-nm[:,:,None]);old=tl.exp2(maximum-nm)
        acc=acc*old[:,:,None]+tl.dot(p.to(tl.bfloat16),mv)
        denominator=denominator*old+tl.sum(p,2);maximum=nm
    state=(((part*21*80*HEADS+(frame*80+cols[:,None])*HEADS+local_head)*45)+rows[None,:]).to(tl.int64)
    tl.store(Acc+state[:,:,None]*128+d[None,None,:],acc,rows[None,:,None]<45)
    tl.store(Max+state,maximum,rows[None,:]<45)
    tl.store(Den+state,denominator,rows[None,:]<45)


@triton.jit
def combine_batched_c10(Acc,Max,Den,Out,HEAD_START:tl.constexpr,HEADS:tl.constexpr,
                         PARTS:tl.constexpr,BQ:tl.constexpr=4):
    query=tl.program_id(0)*BQ+tl.arange(0,BQ)
    parts=tl.arange(0,PARTS);d=tl.arange(0,128)
    valid=query<21*80*HEADS*45
    state=parts[None,:].to(tl.int64)*(21*80*HEADS*45)+query[:,None]
    maximum=tl.load(Max+state,valid[:,None],-float('inf'))
    m=tl.max(maximum,1);scale=tl.exp2(maximum-m[:,None])
    den=tl.sum(tl.load(Den+state,valid[:,None],0)*scale,1)
    acc=tl.load(Acc+state[:,:,None]*128+d[None,None,:],valid[:,None,None],0)
    out=tl.sum(acc*scale[:,:,None],1)/den[:,None]
    row=query%45;group=query//45;head=HEAD_START+group%HEADS
    col=(group//HEADS)%80;frame=group//(HEADS*80)
    address=(((frame*45+row)*80+col)*40+head).to(tl.int64)*128
    tl.store(Out+address[:,None]+d[None,:],out,valid[:,None])
