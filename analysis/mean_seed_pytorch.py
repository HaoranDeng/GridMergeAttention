"""One analytic mean seed plus three real seeds; fixed RF0 selection."""
from __future__ import annotations

MEAN_METHODS = ('mean_fisher32', 'mean_anchor_fisher32', 'mean_hellinger32')


def mean_seed_indices(logits, mean_logits, anchor, method):
    import torch
    if method not in MEAN_METHODS:
        raise ValueError('unsupported mean-seed rule')
    z, center = logits.double(), mean_logits.double()
    if z.ndim != 3 or center.shape != (z.shape[0], z.shape[2]) or z.shape[1] < 3:
        raise ValueError('expected [banks,candidates,probes] and [banks,probes]')
    if not torch.isfinite(z).all() or not torch.isfinite(center).all():
        raise ValueError('finite logits required')
    anchor = torch.as_tensor(anchor,device=z.device,dtype=torch.long)
    batch = torch.arange(z.shape[0],device=z.device)
    if anchor.shape != (z.shape[0],) or ((anchor<0)|(anchor>=z.shape[1])).any():
        raise ValueError('invalid anchor')
    rho = torch.softmax(center,-1)
    if method == 'mean_hellinger32':
        residual = torch.softmax(z,-1).sqrt()-rho[:,None].sqrt()
    else:
        delta = z-center[:,None]
        residual = (delta-(delta*rho[:,None]).sum(-1,keepdim=True))*rho[:,None].sqrt()
    used = torch.zeros(z.shape[:2],device=z.device,dtype=torch.bool)
    chosen = []
    for step in range(3):
        pick = anchor if method=='mean_anchor_fisher32' and step==0 else (
            residual.square().sum(-1).masked_fill(used,-float('inf')).argmax(-1))
        chosen.append(pick)
        used[batch,pick] = True
        basis = residual[batch,pick]
        basis = basis/(basis.square().sum(-1,keepdim=True)+1e-12).sqrt()
        residual -= (residual*basis[:,None]).sum(-1,keepdim=True)*basis[:,None]
    return torch.stack(chosen,-1)


def select_means(q,k):
    import torch
    from attention.adaptive_seed_pytorch import multiscale_candidate_table
    table,anchors=multiscale_candidate_table(45,32,q.device,geometry_count=45)
    heads=q.shape[2]; total=21*45*80*heads
    # BF16 vector is exactly the one loaded by the actual merge. No rescaling.
    means=q.reshape(1,21,45,80,heads,128).float().mean(2).to(q.dtype).contiguous()
    outputs={name:torch.empty((total,4),device=q.device,dtype=torch.uint8)
             for name in MEAN_METHODS}
    for start in range(0,total,2048):
        ids=torch.arange(start,min(start+2048,total),device=q.device)
        head,col=ids%heads,(ids//heads)%80
        row,frame=(ids//(heads*80))%45,ids//(heads*80*45)
        real=table[row].long()
        qq=q[0,real*80+col[:,None],head[:,None]].float()
        kk=k[0,frame[:,None]*3600+row[:,None]*80+
             torch.arange(0,80,5,device=q.device),head[:,None]].float()
        z=torch.bmm(qq,kk.transpose(1,2)).double()*(128**-.5)
        mq=means[0,0,col,head].float()[:,None]
        mz=torch.bmm(mq,kk.transpose(1,2)).double()[:,0]*(128**-.5)
        for name in MEAN_METHODS:
            slots=mean_seed_indices(z,mz,anchors[row],name)
            # Slot zero's row is a placeholder: merge replaces its vector.
            rows=torch.cat((row[:,None],real.gather(1,slots)),-1)
            outputs[name][start:start+len(ids)]=rows.to(torch.uint8)
    return {name:value.reshape(1,1,21,45,80,heads,4)
            for name,value in outputs.items()},means


class MeanSeedKernel:
    """Explicit isolated adapter; no production dispatcher enables this."""
    def __init__(self,kernel,mean_queries):
        self.kernel,self.mean_queries=kernel,mean_queries

    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            if kwargs.get('FP8_SCORE_DOT') or kwargs.get('USE_TEMPORAL_ROW_MAP'):
                raise ValueError('mean screen requires BF16 score and no row map')
            return self.kernel[grid](*args,**kwargs,MEAN_SEED_ZERO=True,
                                     mean_seed_ptr=self.mean_queries)
        return launch
