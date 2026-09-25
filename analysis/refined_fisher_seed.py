"""Two fixed Fisher-GS refinements, with four real queries and RF0."""
REFINED_METHODS=('fisher_maxvol1','fisher_antipodal')
BREGMAN_METHODS=('kl_geometric_gs','kl_anchor_gs')
MEDOID_METHODS=('query_kl_medoid','query_fisher_medoid')


def query_medoid_refine(z,x,chosen,method):
    """One fixed-query-cluster update, keeping the actual anchor fixed."""
    import torch
    z=z.double();batch=torch.arange(z.shape[0],device=z.device)
    ids=torch.arange(z.shape[1],device=z.device)[None]
    if method=='query_kl_medoid':
        lp=torch.log_softmax(z,-1);p=lp.exp()
        centers=p[batch[:,None],chosen]
        center_lp=lp[batch[:,None],chosen]
        d=(centers*center_lp).sum(-1)[:,None]-torch.bmm(lp,centers.transpose(1,2))
    elif method=='query_fisher_medoid':
        centers=x[batch[:,None],chosen]
        d=(x[:,:,None]-centers[:,None]).square().sum(-1)
    else:raise ValueError('unknown query medoid')
    assignment=d.argmin(-1)
    initial=chosen.clone();out=chosen.clone()
    for slot in range(1,4):
        members=assignment==slot;count=members.sum(-1)
        if method=='query_kl_medoid':
            mean=(z*members[:,:,None]).sum(1)/count.clamp_min(1)[:,None]
            lr=torch.log_softmax(mean,-1)
            cost=(p*(lp-lr[:,None])).sum(-1)
        else:
            mean=(x*members[:,:,None]).sum(1)/count.clamp_min(1)[:,None]
            cost=(x-mean[:,None]).square().sum(-1)
        allowed=members | (ids==initial[:,slot,None])
        # Reserve other original seeds, and already-updated seeds. This also
        # keeps empty/tied clusters from creating repeated real row indices.
        for other in range(4):
            if other!=slot:
                allowed &= (ids!=initial[:,other,None]) & (ids!=out[:,other,None])
        score=cost.masked_fill(~allowed,float('inf'))
        best=score.argmin(-1)
        improve=cost[batch,out[:,slot]]-cost[batch,best]
        out[:,slot]=torch.where((count>0)&(improve>1e-12*(1+cost[batch,out[:,slot]].abs())),best,out[:,slot])
    return out


def bregman_features(z,anchor,method):
    """Signed separable reverse-KL embedding, then real-anchor translation."""
    import torch
    if method not in BREGMAN_METHODS: raise ValueError('unknown KL embedding')
    z=z.double();batch=torch.arange(z.shape[0],device=z.device)
    logp=torch.log_softmax(z,-1);p=logp.exp()
    logrho=torch.log_softmax(z.mean(1),-1) if method=='kl_geometric_gs' else logp[batch,anchor]
    rho=logrho.exp();delta=logp-logrho[:,None]
    # Each term is the scalar generalized KL D(rho_j || p_ij).
    # Use probabilities separately to avoid exp(delta) overflow.
    terms=(-rho[:,None]*delta-rho[:,None]+p).clamp_min(0)
    raw=delta.sign()*(2*terms).sqrt()
    return raw-raw[batch,anchor,None],raw,logp,logrho


def fisher_features(z,anchor):
    import torch
    z=z.double(); batch=torch.arange(z.shape[0],device=z.device)
    rho=torch.softmax(z.mean(1),-1)
    delta=z-z[batch,anchor,None]
    return (delta-(delta*rho[:,None]).sum(-1,keepdim=True))*rho[:,None].sqrt()


def project_off(residual,basis):
    basis=basis/(basis.square().sum(-1,keepdim=True)+1e-12).sqrt()
    return residual-(residual*basis[:,None]).sum(-1,keepdim=True)*basis[:,None]


def refined_indices(z,anchor,method):
    import torch
    if method not in REFINED_METHODS+BREGMAN_METHODS+MEDOID_METHODS: raise ValueError('unknown refinement')
    anchor=torch.as_tensor(anchor,device=z.device,dtype=torch.long)
    if z.ndim!=3 or z.shape[1]<4 or anchor.shape!=(z.shape[0],):
        raise ValueError('invalid bank shape')
    if not torch.isfinite(z).all() or ((anchor<0)|(anchor>=z.shape[1])).any():
        raise ValueError('invalid bank or anchor')
    x=bregman_features(z,anchor,method)[0] if method in BREGMAN_METHODS else fisher_features(z,anchor)
    r=x.clone()
    batch=torch.arange(z.shape[0],device=z.device)
    used=torch.arange(z.shape[1],device=z.device)[None]==anchor[:,None]
    picks=[anchor]
    for step in range(1,4):
        norm=r.square().sum(-1).masked_fill(used,-float('inf'))
        pick=norm.argmax(-1)
        if method=='fisher_antipodal' and step==2:
            direction=x[batch,picks[1]]
            signed=(x*direction[:,None]).sum(-1)
            opposite=(-signed).masked_fill(used,-float('inf'))
            pick=torch.where(opposite.max(-1).values>0,opposite.argmax(-1),pick)
        picks.append(pick);used[batch,pick]=True
        r=project_off(r,r[batch,pick])
    chosen=torch.stack(picks,-1)
    if method in MEDOID_METHODS:return query_medoid_refine(z,x,chosen,method)
    if method=='fisher_maxvol1':
        # One coordinate-exchange pass. For fixed other two directions,
        # determinant is their squared area times this squared residual.
        for slot in range(1,4):
            remaining=[j for j in range(1,4) if j!=slot]
            residual=x.clone()
            forbidden=torch.arange(z.shape[1],device=z.device)[None]==anchor[:,None]
            for j in remaining:
                forbidden[batch,chosen[:,j]]=True
                residual=project_off(residual,residual[batch,chosen[:,j]])
            scores=residual.square().sum(-1).masked_fill(forbidden,-float('inf'))
            best=scores.argmax(-1)
            # Preserve the existing pivot in a tie or pure roundoff difference.
            gain=scores[batch,best]-scores[batch,chosen[:,slot]]
            chosen[:,slot]=torch.where(gain>1e-12*(1+scores[batch,best]),best,chosen[:,slot])
    return chosen


def select_refined(q,k,*,bregman=False,medoids=False):
    import torch
    from attention.adaptive_seed_pytorch import multiscale_candidate_table
    table,anchors=multiscale_candidate_table(45,32,q.device,geometry_count=45)
    heads=q.shape[2];total=21*45*80*heads
    methods=MEDOID_METHODS if medoids else BREGMAN_METHODS if bregman else REFINED_METHODS
    result={m:torch.empty((total,4),device=q.device,dtype=torch.uint8) for m in methods}
    for start in range(0,total,2048):
        ids=torch.arange(start,min(start+2048,total),device=q.device)
        h,c=ids%heads,(ids//heads)%80;r,f=(ids//(heads*80))%45,ids//(heads*80*45)
        real=table[r].long()
        qq=q[0,real*80+c[:,None],h[:,None]].float()
        kk=k[0,f[:,None]*3600+r[:,None]*80+torch.arange(0,80,5,device=q.device),h[:,None]].float()
        z=torch.bmm(qq,kk.transpose(1,2)).double()*(128**-.5)
        for m in methods:
            slots=refined_indices(z,anchors[r],m)
            result[m][start:start+len(ids)]=real.gather(1,slots).to(torch.uint8)
    return {m:v.reshape(1,1,21,45,80,heads,4) for m,v in result.items()}
