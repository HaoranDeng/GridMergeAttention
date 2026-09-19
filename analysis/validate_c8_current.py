"""GPU semantic checks against an independent sampled PyTorch merge/route."""
import torch


def reference(q,k,v,seeds,frame,column,rows=(0,22,44)):
    query=q[0,frame,:,column,0].float()
    kb=k[0,:,:,:,0].reshape(945,80,128).float()
    vb=v[0,:,:,:,0].reshape_as(kb).float()
    selected=seeds[0,frame,:,:,column,0].reshape(945,8).long()
    z=torch.bmm(query[selected],kb.transpose(1,2))*128**-.5
    owner=z.log_softmax(-1).argmax(1)
    mask=owner[:,None]==torch.arange(8,device=q.device)[None,:,None]
    active=mask.any(-1)
    z=z.masked_fill(~mask,-float('inf'))
    maximum=torch.where(active,z.amax(-1),0.)
    weights=(z-maximum[...,None]).exp()
    weights/=weights.sum(-1,keepdim=True).clamp_min(1e-30)
    entropy=-(weights*weights.clamp_min(1e-30).log()).sum(-1)
    w=weights.to(torch.bfloat16).float()
    mk=torch.bmm(w,kb).to(torch.bfloat16).float()
    mv=torch.bmm(w,vb).to(torch.bfloat16).float()
    logits=query[list(rows)]@mk.flatten(0,1).T*128**-.5+entropy.flatten()[None]
    logits.masked_fill_(~active.flatten()[None],-float('inf'))
    return logits.softmax(-1)@mv.flatten(0,1)


def validate(op,q,k,v):
    q,k,v=[op._check(x) for x in (q,k,v)]
    assert op.heads==1
    seeds=op.select(q,k).clone()
    assert bool((seeds<45).all())
    # Every query frame owns its selection, including adaptive probe selection.
    perm=torch.arange(20,-1,-1,device=q.device)
    reordered=op.select(q[:,perm].contiguous(),k).clone()
    assert torch.equal(reordered,seeds[:,perm])
    prediction=op.run_with_seeds(q,k,v,seeds)
    errors=[]
    for frame,column in ((0,0),(10,40),(20,79)):
        expected=reference(q,k,v,seeds,frame,column)
        value=prediction[0,frame,[0,22,44],column,0].float()
        errors.append(float((value-expected).norm()/expected.norm()))
    assert max(errors)<.015, errors
    # All score ties assign every key to slot zero. Seven empty slots per block
    # must carry +infinity bias, zero merged KV, and finite final output.
    zero=torch.zeros_like(q)
    zseeds=op.select(zero,k).clone()
    out=op.run_with_seeds(zero,k,v,zseeds)
    active=torch.isfinite(op.bias).reshape(-1,8)
    assert bool(active[:,0].all()) and not bool(active[:,1:].any())
    assert bool((op.mk.reshape(-1,8,128)[:,1:]==0).all())
    assert bool((op.mv.reshape(-1,8,128)[:,1:]==0).all())
    assert bool(out.isfinite().all())
    return {'frame_permutation_equivariance':True,'exact_capacity':8,
            'empty_slots_zero_kv_and_infinite_bias':True,'finite_empty_case':True,
            'reference_relative_l2':errors,'reference_gate':.015,
            'gate_note':'Engineering test threshold for BF16 merge/route, not a user-specified universal tolerance'}
