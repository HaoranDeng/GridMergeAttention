"""Unpromoted algebraic fusion: Q(K'W)= (QK')W; no merged K/V storage.

Keeps exact supplied Fisher seeds and full ten-cluster/SQ1 support. Moves
BF16 rounding boundaries, so numerical and generation validation are required.
"""
import triton
import triton.language as tl

@triton.jit
def reassociated_c10(Q,K,V,Seeds,Out,HEAD:tl.constexpr,KEY_BLOCKS:tl.constexpr=945):
    group=tl.program_id(0);frame=group//80;col=group%80
    rows=tl.arange(0,64);d=tl.arange(0,128);s=tl.arange(0,16);n=tl.arange(0,128)
    qa=(((frame*45+rows[:,None])*80+col)*40+HEAD)*128+d[None,:]
    query=tl.load(Q+qa,rows[:,None]<45,0)
    acc=tl.full((64,128),0,tl.float32)
    maximum=tl.full((64,),-float('inf'),tl.float32)
    denominator=tl.full((64,),0,tl.float32)
    for kb in range(KEY_BLOCKS):
        seed_rows=tl.load(Seeds+((kb*80+col)*40+HEAD)*10+s,s<10,0).to(tl.int32)
        ka=((kb*80+n[:,None])*40+HEAD)*128+d[None,:]
        kk=tl.load(K+ka,n[:,None]<80,0);vv=tl.load(V+ka,n[:,None]<80,0)
        raw_qk=tl.dot(query,tl.trans(kk))
        logits=tl.gather(raw_qk,tl.broadcast_to(seed_rows[:,None],(16,128)),0)*(128**-.5*1.4426950408889634)
        logits=tl.where(n[None,:]<80,logits,-float('inf'))
        norm=tl.max(logits,1);mass=tl.sum(tl.exp2(logits-norm[:,None]),1)
        normalized=logits-(norm+tl.log2(mass))[:,None]
        normalized=tl.where(s[:,None]<10,normalized,-float('inf'))
        winner,owner=tl.max(normalized,0,return_indices=True,return_indices_tie_break_left=True)
        probability=tl.where((s[:,None]==owner[None,:])&(n[None,:]<80)&(s[:,None]<10),tl.exp2(winner)[None,:],0.)
        total=tl.sum(probability,1);safe=tl.where(total>0,total,1.)
        entropy=tl.sum(probability*tl.where(n[None,:]<80,winner[None,:],0.),1)
        weights=probability/safe[:,None]
        selected=tl.sum(tl.where(n[None,:]==col,weights,0.),1)
        residual=1.-selected;valid=(total>0)&(residual>1.e-6)&(s<10)
        safe_residual=tl.where(valid,residual,1.)
        bias=(entropy/safe-tl.log2(safe))*0.6931471805599453
        selected_entropy=selected*tl.log2(tl.maximum(selected,1.e-20))*0.6931471805599453
        bias=(bias-selected_entropy)/safe_residual-tl.log2(safe_residual)*0.6931471805599453
        # Keep the native BF16 bulk weights, then move centroid division and
        # rounding into weights. Slot10 supplies the original SQ1 vector.
        bulk=tl.where(n[None,:]==col,0.,weights).to(tl.bfloat16)
        mixed=(bulk.to(tl.float32)/safe_residual[:,None]).to(tl.bfloat16)
        mixed=tl.where(s[:,None]==10,(n[None,:]==col).to(tl.bfloat16),mixed)
        mixed=tl.where((s[:,None]<10)|(s[:,None]==10),mixed,0.)
        bias=tl.where(s==10,0.,tl.where(valid,bias,float('inf')))
        score=tl.dot(raw_qk,tl.trans(mixed.to(tl.float32)),input_precision='tf32x3')
        score=(score*(128**-.5)-bias[None,:])*1.4426950408889634
        new_max=tl.maximum(maximum,tl.max(score,1))
        p=tl.exp2(score-new_max[:,None]);old=tl.exp2(maximum-new_max)
        raw_p=tl.dot(p.to(tl.bfloat16),mixed.to(tl.bfloat16))
        acc=acc*old[:,None]+tl.dot(raw_p.to(tl.bfloat16),vv)
        denominator=denominator*old+tl.sum(p,1);maximum=new_max
    tl.store(Out+qa,acc/denominator[:,None],rows[:,None]<45)

def reassociated_route(q,k,v,seeds,out,head=0,groups=1680,key_blocks=945):
    return reassociated_c10[(groups,)](q,k,v,seeds,out,head,key_blocks,num_warps=8,num_stages=1)
