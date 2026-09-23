"""Eight-layer block and paired-query extensions with RF0 seed reuse.

Sampled mathematical reference only, not a production r2 latency benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import itertools
import functools
import json
import math
import os
import re
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import sys
import time

LAYERS=(0,5,10,15,20,25,30,35)
HEADS=(0,10,20,30)
GRID=(21,45,80)
SAMPLES=((0,10,20),(0,22,44),(0,40,79))
METHODS=((80,8,'q45_k80_c8_center'),(40,4,'q45_k40_c4_center'),
         (160,8,'q45_k160_c8_intersections'),(320,16,'q45_k320_c16_center'),
         (320,16,'q45_k320_c16_intersections'),(45,8,'q80_k45_c8_center'),
         (90,8,'q80_k90_c8_center'),(90,8,'q80_k90_c8_intersections'))
COMPARISONS=((0,1),(2,4),(3,4),(5,6),(6,7),(0,7),(2,7))
CHUNK_RECIPES=((2,1,4),(2,2,4),(2,2,8),(2,3,8),(2,3,12),(2,4,8),(2,4,16),
               (3,1,4),(3,2,8),(3,3,12),(3,4,16),(4,2,8),(4,4,16),
               (6,2,12),(6,4,24),(8,4,32),(4,8,32),(2,8,16))
REFINE_RECIPES=(('wh',2,3,16),('wh',2,4,24),('wh',3,2,16),('wh',3,3,24),
                ('wh',4,2,16),('hw',2,2,12),('hw',2,3,16),('hw',3,2,16))


def load_reference(path):
    spec=importlib.util.spec_from_file_location('rf0_reference',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def partition(k,v,block_length,boundary,probe_count=16):
    import torch
    t,h,w,d=k.shape
    n=t*h*w
    if boundary=='global':
        starts=torch.arange(0,n,block_length,device=k.device)
        lengths=(n-starts).clamp(max=block_length)
    elif boundary=='frame':
        offsets=torch.arange(0,h*w,block_length,device=k.device)
        starts=(torch.arange(t,device=k.device)[:,None]*(h*w)+offsets).flatten()
        lengths=(h*w-offsets).clamp(max=block_length).repeat(t)
    else:raise ValueError(boundary)
    positions=starts[:,None]+torch.arange(block_length,device=k.device)
    valid=torch.arange(block_length,device=k.device)[None]<lengths[:,None]
    positions=positions.clamp(max=n-1)
    kb=k.reshape(n,d)[positions].masked_fill(~valid[...,None],0).contiguous()
    vb=v.reshape(n,d)[positions].masked_fill(~valid[...,None],0).contiguous()
    centers=starts+(lengths-1)//2
    anchors=(centers//w)%h
    probe_count=min(probe_count,block_length)
    probes=torch.arange(probe_count,device=k.device)[None]*lengths[:,None]//probe_count
    kp=kb[torch.arange(len(kb),device=k.device)[:,None],probes]
    return kb,vb,valid,kp,anchors,starts,lengths


def merge_route(query,kb,vb,valid,seeds,sampled):
    import torch
    capacity=seeds.shape[-1]
    z=torch.bmm(query[seeds],kb.transpose(-1,-2))*query.shape[-1]**-.5
    z=z.masked_fill(~valid[:,None],-float('inf'))
    owner=z.log_softmax(-1).argmax(-2)
    assigned=(owner[:,None]==torch.arange(capacity,device=query.device)[None,:,None])&valid[:,None]
    active=assigned.any(-1)
    selected=z.masked_fill(~assigned,-float('inf'))
    maximum=torch.where(active[...,None],selected.amax(-1,keepdim=True),0.)
    weights=(selected-maximum).exp()
    weights/=weights.sum(-1,keepdim=True).clamp_min(1e-30)
    entropy=-(weights*weights.clamp_min(1e-30).log()).sum(-1)
    merge_weights=weights.to(torch.bfloat16).float()
    mk=torch.bmm(merge_weights,kb).to(torch.bfloat16).float()
    mv=torch.bmm(merge_weights,vb).to(torch.bfloat16).float()
    logits=query[sampled]@mk.flatten(0,1).T*query.shape[-1]**-.5
    logits+=entropy.flatten()[None]
    logits.masked_fill_(~active.flatten()[None],-float('inf'))
    output=logits.softmax(-1)@mv.flatten(0,1)
    assert bool(torch.isfinite(output).all())
    assert not bool(assigned.masked_select(~valid[:,None].expand_as(assigned)).any())
    return output,int((~active).sum()),active.numel()


def intersection_rows(starts,lengths,width,height,query_w,center_rows):
    """Actual intersection rows, primary first then remaining token order.

Preserve the old primary when it intersects the block. Otherwise use the
nearest actual intersection. Invalid tail prefix slots are -1.
"""
    import torch
    first=starts+(query_w-starts.remainder(width)).remainder(width)
    count=int((lengths.max()+width-1)//width)
    positions=first[:,None]+torch.arange(count,device=starts.device)[None]*width
    valid=positions<(starts+lengths)[:,None]
    if not bool(valid[:,0].all()):raise ValueError('Every tested block must intersect the Q column')
    rows=(positions//width)%height
    center=starts+(lengths-1)//2
    matches=(rows==center_rows[:,None])&valid
    nearest=(positions-center[:,None]).abs().masked_fill(~valid,2**60).argmin(-1)
    primary=torch.where(matches.any(-1),matches.long().argmax(-1),nearest)
    order=torch.arange(count,device=starts.device)[None].expand_as(positions).clone()
    order.masked_fill_(~valid,count+1)
    order[torch.arange(len(starts),device=starts.device),primary]=-1
    indices=order.argsort(-1)
    prefix=rows.gather(1,indices).masked_fill(~valid.gather(1,indices),-1)
    for i in range(count):
        for j in range(i):
            assert not bool(((prefix[:,i]>=0)&(prefix[:,i]==prefix[:,j])).any()),'Repeated RF0 row in prefix'
    return prefix


def select_intersections(query,kp,candidates,anchors,prefix,capacity,allow_intersection_subset=False,seed_policy='intersections'):
    """Force intersections, or use Fisher inside them when they exceed capacity."""
    import torch
    z=torch.einsum('id,gpd->gip',query,kp)*query.shape[-1]**-.5
    z=z.gather(1,candidates[...,None].expand(-1,-1,kp.shape[1])).double()
    groups=torch.arange(len(kp),device=query.device)
    za=z[groups,anchors]
    rho=za.softmax(-1)
    delta=z-za[:,None]
    residual=(delta-(delta*rho[:,None]).sum(-1,keepdim=True))*rho[:,None].sqrt()
    used=torch.zeros(candidates.shape,dtype=torch.bool,device=query.device)
    used[groups,anchors]=True
    selected=[candidates[groups,anchors]]
    assert capacity<=candidates.shape[-1]
    mixed=seed_policy in ('anchor_fisher','quarter_fisher')
    assert seed_policy in ('intersections','anchor_fisher','quarter_fisher')
    assert prefix.shape[-1]<=capacity or allow_intersection_subset or mixed
    forced=prefix>=0
    over=forced.sum(-1)>capacity
    assert allow_intersection_subset or mixed or not bool(over.any())
    # Invert the pool once; avoid a G x intersections x candidates tensor.
    inverse=torch.full((len(kp),query.shape[0]),-1,dtype=torch.long,device=query.device)
    inverse.scatter_(1,candidates,torch.arange(candidates.shape[1],device=query.device)[None].expand_as(candidates))
    prefix_slots=inverse.gather(1,prefix.clamp_min(0))
    assert bool(((prefix_slots>=0)|~forced).all()),'Intersection missing from candidate pool'
    intersection_counts=torch.zeros_like(candidates)
    intersection_counts.scatter_add_(1,prefix_slots.clamp_min(0),forced.long())
    intersection_candidates=intersection_counts>0
    eligible=~over[:,None]|intersection_candidates
    junction_quota=torch.minimum(forced.sum(-1),torch.full_like(forced.sum(-1),1 if seed_policy=='anchor_fisher' else max(1,(capacity+3)//4)))
    for slot in range(1,capacity):
        allowed=(~(slot<junction_quota)[:,None]|intersection_candidates) if mixed else eligible
        picked=residual.square().sum(-1).masked_fill(used|~allowed,-float('inf')).argmax(-1)
        if not mixed and slot<prefix.shape[-1]:picked=torch.where(forced[:,slot]&~over,prefix_slots[:,slot],picked)
        assert not bool(used[groups,picked].any()),'Duplicate seed'
        used[groups,picked]=True
        selected.append(candidates[groups,picked])
        if slot<capacity-1:
            direction=residual[groups,picked]
            basis=direction/(direction.square().sum(-1,keepdim=True)+1e-12).sqrt()
            residual-=(residual*basis[:,None]).sum(-1,keepdim=True)*basis[:,None]
    return torch.stack(selected,-1)


def evaluate(q,k,v,block_length,boundary,reference,samples=SAMPLES,poison_padding=False,capacity=8,seed_policy='center'):
    import torch
    if capacity not in (4,8,16):raise ValueError('Only exact C4, C8 or C16 supported')
    kb,vb,valid,kp,anchor_rows,starts,lengths=partition(k,v,block_length,boundary)
    table,anchor_slots=reference.candidate_table(q.shape[1])
    candidates=torch.tensor(table,device=q.device)[anchor_rows]
    anchors=torch.tensor(anchor_slots,device=q.device)[anchor_rows]
    if poison_padding:
        kb=kb.masked_fill(~valid[...,None],1e5)
        vb=vb.masked_fill(~valid[...,None],-1e5)
    rows,banks,audits={}, {}, []
    probes=torch.arange(kp.shape[1],device=q.device)
    for w in samples[2]:
        if seed_policy=='center':
            seeds=reference.select_fisher(q[0,:,w],kp,candidates,anchors,probes,capacity)
            seed_details={'forced_anchors_per_block':1}
        elif seed_policy=='intersections':
            prefix=intersection_rows(starts,lengths,k.shape[2],k.shape[1],w,anchor_rows)
            primary=prefix[:,0]
            plane_candidates=torch.tensor(table,device=q.device)[primary]
            plane_anchors=torch.tensor(anchor_slots,device=q.device)[primary]
            seeds=select_intersections(q[0,:,w],kp,plane_candidates,plane_anchors,prefix,capacity)
            assert torch.equal(seeds[:,0],primary)
            forced=prefix>=0
            assert torch.equal(seeds[:,:prefix.shape[-1]][forced],prefix[forced])
            counts=forced.sum(-1)
            seed_details={'anchor_count_histogram':{str(i):int((counts==i).sum()) for i in range(1,prefix.shape[-1]+1)},
                          'primary_changed_from_center':int((primary!=anchor_rows).sum()),
                          'forced_intersections_verified':True}
        else:raise ValueError(seed_policy)
        assert seeds.shape==(len(kb),capacity)
        original=seeds.clone()
        bank=seeds.to(torch.uint8).cpu().contiguous()
        banks[str(w)]=bank
        empty,total=0,0
        for t in samples[0]:
            output,e,n=merge_route(q[t,:,w],kb,vb,valid,seeds,list(samples[1]))
            assert torch.equal(seeds,original)
            empty+=e;total+=n
            for slot,h in enumerate(samples[1]):rows[(t,h,w)]=output[slot]
        audits.append({'query_w':w,'selected_at_t':0,'reused_at_t':list(samples[0]),
                       'selector_calls':1,'indices_unchanged':True,
                       'index_sha256':hashlib.sha256(bank.numpy().tobytes()).hexdigest(),
                       'empty_slots':empty,'total_slots':total,**seed_details})
    geometry={'key_groups':len(kb),'block_length':block_length,'boundary':boundary,
              'last_group_valid_tokens':int(lengths[-1]),'masked_tokens':int((~valid).sum()),
              'capacity':capacity,'merged_representatives_per_query':len(kb)*capacity,
              'compressed_token_density':len(kb)*capacity/k.numel()*k.shape[-1],
              'seed_policy':seed_policy,
              'anchor':'physical H of lower-center valid key token' if seed_policy=='center' else 'actual Q-column intersections with K block; keep center first if valid, then all remaining intersections in key token order; RF0 H'}
    return torch.stack([rows[c] for c in itertools.product(*samples)]),banks,audits,geometry


def selftest(reference):
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(64)
    q,k,v=[torch.randn(3,9,12,8) for _ in range(3)]
    samples=((0,1,2),(0,4,8),(0,5,11))
    checks=[]
    for capacity in (4,8):
        original,banks,_=reference.evaluate_rf0(q,k,v,1,2,capacity,samples)
        value,flat_banks,_,_=evaluate(q,k,v,12,'global',reference,samples,capacity=capacity)
        torch.testing.assert_close(value,original,rtol=1e-5,atol=1e-6)
        assert all(torch.equal(banks[key],flat_banks[key]) for key in banks)
    for block_length,capacity in ((64,4),(128,8)):
        boundary='global'
        value,banks,audits,geometry=evaluate(q,k,v,block_length,boundary,reference,samples,capacity=capacity)
        poisoned,poison_banks,_,_=evaluate(q,k,v,block_length,boundary,reference,samples,True,capacity)
        torch.testing.assert_close(value,poisoned,rtol=1e-6,atol=1e-7)
        assert all(torch.equal(banks[key],poison_banks[key]) for key in banks)
        changed=q.clone();changed[1:]*=-2
        altered,changed_banks,_,_=evaluate(changed,k,v,block_length,boundary,reference,samples,capacity=capacity)
        assert all(torch.equal(banks[key],changed_banks[key]) for key in banks)
        assert not torch.equal(value,altered)
        ones,_,empty_audit,_=evaluate(torch.zeros_like(q),k,torch.ones_like(v),block_length,boundary,reference,samples,capacity=capacity)
        torch.testing.assert_close(ones,torch.ones_like(ones),rtol=1e-6,atol=1e-6)
        assert all(x['empty_slots']*capacity==x['total_slots']*(capacity-1) for x in empty_audit)
        assert all(x['selector_calls']==1 and x['indices_unchanged'] for x in audits)
        kb,_,valid,_,_,starts,lengths=partition(k,v,block_length,boundary)
        positions=starts[:,None]+torch.arange(block_length)[None]
        assert torch.equal(positions[valid],torch.arange(k.numel()//k.shape[-1]))
        assert bool((starts//(9*12)!=(starts+lengths-1)//(9*12)).any())
        assert all(bank.shape[-1]==capacity for bank in banks.values())
        checks.append({'boundary':boundary,'padding_invariance':True,'rf0_reuse':True,'zero_query_empty_slots':True,**geometry})
    q,k,v=[torch.randn(3,9,80,8) for _ in range(3)]
    samples=((0,1,2),(0,4,8),(0,40,79))
    for block_length in (128,160):
        value,banks,audit,g=evaluate(q,k,v,block_length,'global',reference,samples,capacity=8,seed_policy='intersections')
        poisoned,pbanks,_,_=evaluate(q,k,v,block_length,'global',reference,samples,True,8,'intersections')
        torch.testing.assert_close(value,poisoned,rtol=1e-6,atol=1e-7)
        assert all(torch.equal(banks[key],pbanks[key]) for key in banks)
        changed=q.clone();changed[1:]*=-2
        altered,changed_banks,_,_=evaluate(changed,k,v,block_length,'global',reference,samples,capacity=8,seed_policy='intersections')
        assert all(torch.equal(banks[key],changed_banks[key]) for key in banks)
        assert not torch.equal(value,altered)
        ones,_,empty,_=evaluate(torch.zeros_like(q),k,torch.ones_like(v),block_length,'global',reference,samples,capacity=8,seed_policy='intersections')
        torch.testing.assert_close(ones,torch.ones_like(ones),rtol=1e-6,atol=1e-6)
        assert all(x['empty_slots']*8==x['total_slots']*7 for x in empty)
        assert all(x['forced_intersections_verified'] for x in audit)
        checks.append({'block_length':block_length,'seed_policy':'intersections','padding_invariance':True,
                       'rf0_reuse':True,'empty_slots':True,'forced_intersections':True,'capacity':8})
    # With one row per block, the forced selector must reduce to original Fisher.
    original,_,_,_=evaluate(q,k,v,80,'global',reference,samples,capacity=8)
    intersection,_,_,_=evaluate(q,k,v,80,'global',reference,samples,capacity=8,seed_policy='intersections')
    torch.testing.assert_close(original,intersection,rtol=1e-5,atol=1e-6)
    q,k,v=[torch.randn(5,45,80,8) for _ in range(3)]
    samples=((0,2,4),(0,22,44),(0,40,79))
    for policy in ('center','intersections'):
        output,banks,audits,g=evaluate(q,k,v,320,'global',reference,samples,capacity=16,seed_policy=policy)
        poisoned,pbanks,_,_=evaluate(q,k,v,320,'global',reference,samples,True,16,policy)
        torch.testing.assert_close(output,poisoned,rtol=1e-6,atol=1e-7)
        assert all(torch.equal(banks[w],pbanks[w]) and banks[w].shape[-1]==16 for w in banks)
        assert g['last_group_valid_tokens']==80 and g['masked_tokens']==240
        changed=q.clone();changed[1:]*=-2
        altered,abanks,_,_=evaluate(changed,k,v,320,'global',reference,samples,capacity=16,seed_policy=policy)
        assert all(torch.equal(banks[w],abanks[w]) for w in banks) and not torch.equal(output,altered)
        ones,_,empty,_=evaluate(torch.zeros_like(q),k,torch.ones_like(v),320,'global',reference,samples,capacity=16,seed_policy=policy)
        torch.testing.assert_close(ones,torch.ones_like(ones),rtol=1e-6,atol=1e-6)
        assert all(x['empty_slots']*16==x['total_slots']*15 for x in empty)
        if policy=='intersections':
            assert all(x['anchor_count_histogram']=={'1':1,'2':0,'3':0,'4':56} for x in audits)
        checks.append({'block_length':320,'capacity':16,'seed_policy':policy,'padding_invariance':True,
                       'rf0_reuse':True,'empty_slots':True,'tail80':True})
    original,_,_=reference.evaluate_rf0(q,k,v,1,2,16,samples)
    aligned,_,_,_=evaluate(q,k,v,80,'global',reference,samples,capacity=16,seed_policy='intersections')
    torch.testing.assert_close(original,aligned,rtol=1e-5,atol=1e-6)
    # K40 is exactly the two contiguous W halves of every original key row.
    kb,_,valid,_,_,starts,lengths=partition(k,v,40,'global')
    expected=k.reshape(-1,80,8).reshape(-1,2,40,8).reshape_as(kb)
    assert torch.equal(kb,expected) and bool(valid.all()) and bool((lengths==40).all())
    k40,_,_,_=evaluate(q,k,v,40,'global',reference,samples,capacity=4)
    assert bool(torch.isfinite(k40).all())
    checks.append({'block_length':40,'capacity':4,'exact_w_reduce2_partition':True})
    # The Q80 wrapper must preserve the original physical sampled queries.
    for capacity in (4,8):
        original,_,_=reference.evaluate_rf0(q,k,v,2,1,capacity,samples)
        rotated,_,_,g=evaluate_method(q,k,v,45,capacity,f'q80_k45_c{capacity}_center',reference,samples)
        torch.testing.assert_close(original,rotated,rtol=1e-5,atol=1e-6)
        checks.append({'method':f'q80_k45_c{capacity}_center','original_reference_agreement':True,
                       'physical_sample_order':True})
    q90,banks,audits,g=evaluate_method(q,k,v,90,8,'q80_k90_c8_intersections',reference,samples)
    assert bool(torch.isfinite(q90).all()) and g['query_group_size']==80 and g['key_axis']=='H'
    assert all(a['anchor_count_histogram']=={'1':0,'2':200} for a in audits)
    for bank in banks.values():
        assert bool((bank[:,0]%2==0).all()) and torch.equal(bank[:,1],bank[:,0]+1)
    changed=q.clone();changed[1:]*=-2
    altered,abanks,_,_=evaluate_method(changed,k,v,90,8,'q80_k90_c8_intersections',reference,samples)
    assert all(torch.equal(banks[w],abanks[w]) for w in banks) and not torch.equal(q90,altered)
    ones,_,empty,_=evaluate_method(torch.zeros_like(q),k,torch.ones_like(v),90,8,'q80_k90_c8_intersections',reference,samples)
    torch.testing.assert_close(ones,torch.ones_like(ones),rtol=1e-6,atol=1e-6)
    assert all(a['empty_slots']*8==a['total_slots']*7 for a in empty)
    checks.append({'method':'q80_k90_c8_intersections','physical_sample_order':True,
                   'two_adjacent_H45_columns':True,'rf0_reuse':True,'empty_slots':True,
                   'q80_k45_original_reference_agreement':True})
    return {'passed':True,'width_aligned_matches_original_rf0':True,'checks':checks}


def evaluate_method(q,k,v,block_length,capacity,name,reference,samples=SAMPLES):
    """Return all layouts in identical physical (T,H,W) sampled-query order."""
    if name.startswith('chunk_'):
        layout,a,b,c=parse_chunk_name(name)
        assert c==capacity and block_length==b*(q.shape[2] if layout=='hw' else q.shape[1])
        policy=chunk_seed_policy(name)
        return evaluate_chunks(q,k,v,layout,a,b,c,reference,samples,allow_intersection_subset=policy in ('junction_subset','probe64','probe128'),seed_policy=policy if policy in ('anchor_fisher','quarter_fisher') else 'intersections',probe_count=chunk_probe_count(name))
    if name.startswith('q160_'):
        return evaluate_paired_queries(q,k,v,capacity,reference,samples,64 if 'candidates64' in name else 32)
    policy=name.rsplit('_',1)[-1]
    rotated=name.startswith('q80_')
    if rotated:
        import torch
        transformed=[x.permute(0,2,1,3).contiguous() for x in (q,k,v)]
        value,banks,audits,geometry=evaluate(*transformed,block_length,'global',reference,
            (samples[0],samples[2],samples[1]),capacity=capacity,seed_policy=policy)
        value=value.reshape(len(samples[0]),len(samples[2]),len(samples[1]),q.shape[-1]).permute(0,2,1,3).reshape(-1,q.shape[-1])
        for audit in audits:audit['query_h']=audit.pop('query_w')
    else:
        value,banks,audits,geometry=evaluate(q,k,v,block_length,'global',reference,samples,capacity=capacity,seed_policy=policy)
    geometry.update(query_axis='W' if rotated else 'H',key_axis='H' if rotated else 'W',
        query_group_size=q.shape[2] if rotated else q.shape[1],
        key_flatten_order='TWH' if rotated else 'THW',physical_sample_order='THW')
    return value,banks,audits,geometry


def paired_candidates(width,row_count,reference,candidate_budget):
    """Fixed spatial candidates, with equal budget per valid query row.

    C32 uses 16 of each row's original C32 via farthest-point spatial sampling,
    starting with both intersections. C64 keeps the full per-row C32 union.
    An odd final single row keeps its original C32 and no padded candidates.
    """
    table,_=reference.candidate_table(width)
    rows=[];anchors=[];prefixes=[]
    for left in range(0,width,2):
        intersections=[left,left+1]
        source=table[left]
        assert set(intersections)<=set(source)
        if row_count==1 or candidate_budget==64:
            chosen=set(source)
        else:
            chosen=set(intersections)
            while len(chosen)<16:
                chosen.add(max((x for x in source if x not in chosen),
                               key=lambda x:(min(abs(x-y) for y in chosen),-x)))
        candidates=sorted(x+row*width for row in range(row_count) for x in chosen)
        prefix=[x+row*width for x in intersections for row in range(row_count)]
        rows.append(candidates);anchors.append(candidates.index(left));prefixes.append(prefix)
    return rows,anchors,prefixes


def evaluate_paired_queries(q,k,v,capacity,reference,samples=SAMPLES,candidate_budget=32):
    import torch
    t_count,height,width,dim=q.shape
    assert width==80 and height%2==1 and capacity==16
    transformed=[x.permute(0,2,1,3).contiguous() for x in (k,v)]
    kb,vb,valid,kp,_,starts,lengths=partition(*transformed,2*height,'global')
    assert bool(valid.all()) and len(kb)==t_count*width//2
    result,banks,audits={},{},[]
    for first_h in sorted({h//2*2 for h in samples[1]}):
        count=min(2,height-first_h)
        cand,anchor,prefix=paired_candidates(width,count,reference,candidate_budget)
        candidates=torch.tensor(cand,device=q.device).repeat(t_count,1)
        anchors=torch.tensor(anchor,device=q.device).repeat(t_count)
        forced=torch.tensor(prefix,device=q.device).repeat(t_count,1)
        query=q[0,first_h:first_h+count].reshape(count*width,dim)
        seeds=select_intersections(query,kp,candidates,anchors,forced,capacity)
        assert seeds.shape==(len(kb),16) and torch.equal(seeds[:,:2*count],forced)
        assert bool((seeds<count*width).all())
        original=seeds.clone()
        bank=seeds.to(torch.uint8).cpu().contiguous();banks[str(first_h)]=bank
        selected=[(h,w) for h in samples[1] if first_h<=h<first_h+count for w in samples[2]]
        indices=[(h-first_h)*width+w for h,w in selected]
        empty,total=0,0
        for t in samples[0]:
            current=q[t,first_h:first_h+count].reshape(count*width,dim)
            values,e,n=merge_route(current,kb,vb,valid,seeds,indices)
            assert torch.equal(seeds,original)
            empty+=e;total+=n
            for i,(h,w) in enumerate(selected):result[(t,h,w)]=values[i]
        audits.append({'query_h_start':first_h,'valid_query_tokens':count*width,
            'candidate_count':len(cand[0]),'forced_intersections_per_block':2*count,
            'forced_intersections_verified':True,'selected_at_t':0,'reused_at_t':list(samples[0]),
            'selector_calls':1,'indices_unchanged':True,'invalid_query_seeds':0,
            'index_sha256':hashlib.sha256(bank.numpy().tobytes()).hexdigest(),
            'empty_slots':empty,'total_slots':total})
    geometry={'query_axis':'two adjacent H rows, each W80','query_group_size':160,
        'query_groups_per_frame':(height+1)//2,'last_query_group_valid_tokens':width,
        'invalid_query_padding_evaluated':False,'key_axis':'H','key_flatten_order':'TWH',
        'block_length':2*height,'key_groups':len(kb),'last_group_valid_tokens':2*height,'masked_tokens':0,
        'capacity':16,'merged_representatives_per_query':len(kb)*16,
        'compressed_token_density':16/(2*height),'seed_policy':'intersections',
        'candidate_budget':candidate_budget,'tail_candidate_count':32,
        'candidate_policy':'per-row original C32 union' if candidate_budget==64 else '16 spatially farthest candidates per row from original C32, seeded by both intersections',
        'boundary':'global','physical_sample_order':'THW'}
    return torch.stack([result[c] for c in itertools.product(*samples)]),banks,audits,geometry


def selftest_paired_queries(reference):
    import torch
    torch.manual_seed(16090)
    samples=((0,1,2),(0,1,4),(0,40,79))
    q,k,v=[torch.randn(3,5,80,8) for _ in range(3)]
    controls,_,_,_=evaluate_method(q,k,v,10,16,'q80_k90_c16_intersections',reference,samples)
    for budget in (32,64):
        value,banks,audits,g=evaluate_paired_queries(q,k,v,16,reference,samples,budget)
        assert [a['valid_query_tokens'] for a in audits]==[160,80]
        assert [a['forced_intersections_per_block'] for a in audits]==[4,2]
        assert [a['candidate_count'] for a in audits]==[budget,32]
        torch.testing.assert_close(value.reshape(3,3,3,8)[:,2],controls.reshape(3,3,3,8)[:,2],rtol=1e-5,atol=1e-6)
        # Independent original no-padding merge verifies grouping and sample ordering.
        kb=k.permute(0,2,1,3).reshape(-1,10,8)
        vb=v.permute(0,2,1,3).reshape_as(kb)
        expected={}
        for start,bank in banks.items():
            start=int(start);count=min(2,5-start)
            coords=[(h,w) for h in samples[1] if start<=h<start+count for w in samples[2]]
            indices=[(h-start)*80+w for h,w in coords]
            for t in samples[0]:
                out,_,_=reference.merge_route(q[t,start:start+count].reshape(-1,8),kb,vb,bank.long(),indices)
                for i,(h,w) in enumerate(coords):expected[(t,h,w)]=out[i]
        torch.testing.assert_close(value,torch.stack([expected[c] for c in itertools.product(*samples)]),rtol=1e-5,atol=1e-6)
        changed=q.clone();changed[1:]*=-2
        altered,abanks,_,_=evaluate_paired_queries(changed,k,v,16,reference,samples,budget)
        assert all(torch.equal(banks[x],abanks[x]) for x in banks) and not torch.equal(value,altered)
        ones,_,empty,_=evaluate_paired_queries(torch.zeros_like(q),k,torch.ones_like(v),16,reference,samples,budget)
        torch.testing.assert_close(ones,torch.ones_like(ones),rtol=1e-6,atol=1e-6)
        assert all(a['empty_slots']*16==a['total_slots']*15 for a in empty)
    return {'passed':True,'candidate_budgets':[32,64],'exact_capacity':16,
            'four_intersections':True,'tail80_matches_q80_c16':True,'independent_merge_agreement':True,
            'physical_sample_order':True,'rf0_reuse':True,'empty_clusters':True}


def parse_chunk_name(name):
    match=re.fullmatch(r'chunk_(hw|wh)_q(\d+)_k(\d+)_c(\d+)(?:_(?:junction_subset|anchor_fisher|quarter_fisher|probe64|probe128))?',name)
    if not match:raise ValueError(name)
    layout,a,b,c=match.groups()
    return layout,int(a),int(b),int(c)


def chunk_seed_policy(name):
    return next((p for p in ('junction_subset','anchor_fisher','quarter_fisher','probe64','probe128') if name.endswith('_'+p)),'intersections')


def chunk_probe_count(name):
    return 64 if name.endswith('_probe64') else 128 if name.endswith('_probe128') else 16


def chunk_flops(t,h,w,layout,a,b,c,heads=40,dim=128,probe_count=16):
    length,planes=(h,w) if layout=='hw' else (w,h)
    n=t*h*w;kg=math.ceil(t*length/b);k=b*planes;qg=t*math.ceil(planes/a)
    core=6*heads*dim*c*qg*kg*k+4*heads*dim*n*c*kg
    selector=2*heads*dim*probe_count*kg*h*w
    return {'core_flops':core,'rf0_selector_qk_flops':selector,'total_major_matmul_flops':core+selector,
            'total_tflops':(core+selector)/1e12,'dense_ratio':(core+selector)/(4*heads*dim*n*n)}


@functools.lru_cache(maxsize=None)
def chunk_metadata(t,length,planes,a,b,first_plane,reference,device,candidate_budget=32):
    """Balanced candidate pool, all intersections first; no query crosses a frame."""
    import torch
    count=min(a,planes-first_plane)
    source_table,_=reference.candidate_table(length)
    all_candidates=[];anchor_slots=[];prefixes=[];forced_counts=[];primary_rows=[]
    for start in range(0,t*length,b):
        valid=min(b,t*length-start)
        spatial_rows=[(start+j)%length for j in range(valid)]
        primary=(start+(valid-1)//2)%length
        forced=[row+p*length for row in spatial_rows for p in range(count)]
        forced=[primary]+[x for x in forced if x!=primary]
        assert len(set(forced))==count*valid
        candidates=[]
        for p in range(count):
            budget=candidate_budget//count+(p<candidate_budget%count)
            chosen=set(spatial_rows)
            pool=set(source_table[primary])|chosen
            if count==1 and len(pool)<=budget:chosen=pool.copy()
            while len(chosen)<budget:
                available=pool-chosen
                if not available:available=set(range(length))-chosen
                chosen.add(max(available,key=lambda x:(min(abs(x-y) for y in chosen),-x)))
            assert len(chosen)==budget and set(spatial_rows)<=chosen
            candidates.extend(x+p*length for x in sorted(chosen))
        assert len(candidates)==candidate_budget and set(forced)<=set(candidates)
        all_candidates.append(candidates);anchor_slots.append(candidates.index(primary))
        forced_counts.append(len(forced));primary_rows.append(primary)
        prefixes.append(forced+[-1]*(a*b-len(forced)))
    return (torch.tensor(all_candidates,device=device),torch.tensor(anchor_slots,device=device),
            torch.tensor(prefixes,device=device),forced_counts,primary_rows)


def evaluate_chunks(q,k,v,layout,a,b,c,reference,samples=SAMPLES,poison_padding=False,allow_intersection_subset=False,seed_policy='intersections',probe_count=16):
    import torch
    t,h,w,d=q.shape
    mixed=seed_policy in ('anchor_fisher','quarter_fisher')
    assert 1<=c<=h*w and isinstance(c,int) and (a*b<=c or allow_intersection_subset or mixed)
    candidate_budget=min(a*(h if layout=='hw' else w),h*w,max(32,1<<(max(c,a*b)-1).bit_length()))
    assert layout in ('hw','wh')
    canonical=[x.permute(0,2,1,3).contiguous() for x in (q,k,v)] if layout=='hw' else [q,k,v]
    cq,ck,cv=canonical;_,planes,length,_=cq.shape
    kb,vb,valid,kp,_,starts,lengths=partition(ck.permute(0,2,1,3).contiguous(),
                                          cv.permute(0,2,1,3).contiguous(),b*planes,'global',probe_count)
    if poison_padding:
        kb=kb.masked_fill(~valid[...,None],1e5);vb=vb.masked_fill(~valid[...,None],-1e5)
    plane_samples=samples[2] if layout=='hw' else samples[1]
    long_samples=samples[1] if layout=='hw' else samples[2]
    result,banks,audits={},{},[]
    for first_plane in sorted({p//a*a for p in plane_samples}):
        count=min(a,planes-first_plane)
        candidates,anchors,prefix,counts,_=chunk_metadata(t,length,planes,a,b,first_plane,reference,str(q.device),candidate_budget)
        query=cq[0,first_plane:first_plane+count].reshape(count*length,d)
        seeds=select_intersections(query,kp,candidates,anchors,prefix,c,allow_intersection_subset,seed_policy)
        forced=prefix>=0
        over=forced.sum(-1)>c
        assert seeds.shape==(len(kb),c)
        required=forced&~over[:,None]&~torch.full_like(forced,mixed)
        selected_prefix=seeds[:,torch.arange(prefix.shape[1],device=q.device).clamp(max=c-1)]
        assert torch.equal(selected_prefix[required],prefix[required])
        intersection_mask=torch.zeros((len(kb),count*length),dtype=torch.long,device=q.device)
        intersection_mask.scatter_add_(1,prefix.clamp_min(0),forced.long())
        subset_membership=intersection_mask.gather(1,seeds)>0
        if not mixed:assert bool(subset_membership[over].all())
        quota=torch.minimum(forced.sum(-1),torch.full_like(forced.sum(-1),1 if seed_policy=='anchor_fisher' else max(1,(c+3)//4)))
        if mixed:
            assert torch.equal(seeds[:,0],prefix[:,0])
            assert bool((subset_membership|~(torch.arange(c,device=q.device)[None,:]<quota[:,None])).all())
        assert bool((seeds<count*length).all())
        assert all(torch.unique(row).numel()==c for row in seeds[:1])
        original=seeds.clone()
        bank=seeds.to(torch.uint8 if count*length<=256 else torch.int16).cpu().contiguous()
        assert torch.equal(bank.long(),seeds.cpu()),'Seed serialization overflow'
        banks[str(first_plane)]=bank
        selected=[(p,l) for p in plane_samples if first_plane<=p<first_plane+count for l in long_samples]
        indices=[(p-first_plane)*length+l for p,l in selected]
        empty,total=0,0
        for ti in samples[0]:
            current=cq[ti,first_plane:first_plane+count].reshape(count*length,d)
            output,e,n=merge_route(current,kb,vb,valid,seeds,indices)
            assert torch.equal(seeds,original)
            empty+=e;total+=n
            for i,(p,l) in enumerate(selected):result[(ti,l,p) if layout=='hw' else (ti,p,l)]=output[i]
        audits.append({'query_plane_start':first_plane,'valid_query_tokens':count*length,
            'valid_query_chunks':count,'candidate_count':candidate_budget,'selected_at_t':0,'reused_at_t':list(samples[0]),
            'selector_calls':1,'indices_unchanged':True,'forced_intersections_verified':not mixed and not bool(over.any()),
            'seed_intersection_policy_verified':True,'intersection_subset_groups':0 if mixed else int(over.sum()),
            'subset_seeds_are_intersections':not mixed,
            'seed_policy_mode':seed_policy,'selected_intersection_slots':int(subset_membership.sum()),
            'selected_nonintersection_slots':int((~subset_membership).sum()),
            'forced_junction_quota_histogram':{str(n):int((quota==n).sum()) for n in quota.unique().tolist()} if mixed else None,
            'forced_intersection_histogram':{str(n):counts.count(n) for n in sorted(set(counts))},
            'seed_storage_dtype':str(bank.dtype),'invalid_query_seeds':0,
            'index_sha256':hashlib.sha256(bank.numpy().tobytes()).hexdigest(),
            'empty_slots':empty,'total_slots':total})
    geometry={'layout':layout,'q_chunks':a,'k_chunks':b,'capacity':c,'query_group_size':a*length,
        'block_length':b*planes,'query_axis':'H columns' if layout=='hw' else 'W rows',
        'key_axis':'W' if layout=='hw' else 'H','query_groups_per_frame':math.ceil(planes/a),
        'last_query_group_valid_tokens':((planes-1)%a+1)*length,
        'key_flatten_order':'THW' if layout=='hw' else 'TWH','boundary':'global',
        'key_groups':len(kb),'last_group_valid_tokens':int(lengths[-1]),'masked_tokens':int((~valid).sum()),
        'merged_representatives_per_query':len(kb)*c,'compressed_token_density':len(kb)*c/(t*h*w),
        'candidate_budget':candidate_budget,'seed_policy':('one anchor then unrestricted Fisher' if seed_policy=='anchor_fisher' else 'first min(ceil(C/4), intersections) from intersections by Fisher, then unrestricted Fisher') if mixed else 'all intersections, or Fisher subset when count exceeds C' if allow_intersection_subset else 'all intersections then Fisher',
        'seed_policy_mode':seed_policy,'probe_count':probe_count,
        'allow_intersection_subset':allow_intersection_subset,
        'candidate_policy':'balanced per query chunk from original C32/g45; intersections protected; spatial farthest fill',
        'physical_sample_order':'THW','invalid_query_padding_evaluated':False,
        'flops':chunk_flops(t,h,w,layout,a,b,c,probe_count=probe_count)}
    return torch.stack([result[x] for x in itertools.product(*samples)]),banks,audits,geometry


def selftest_chunks(reference):
    import torch
    torch.manual_seed(20260920)
    q,k,v=[torch.randn(3,45,80,8) for _ in range(3)]
    samples=((0,1,2),(0,22,44),(0,40,79))
    checks=[]
    if any(name=='chunk_hw_q1_k1_c10' for _,_,name in METHODS):
        expected,old_banks,_=reference.evaluate_rf0(q,k,v,1,2,10,samples)
        value,banks,audits,_=evaluate_chunks(q,k,v,'hw',1,1,10,reference,samples)
        torch.testing.assert_close(value,expected,rtol=1e-5,atol=1e-6)
        assert all(torch.equal(bank,old_banks[x]) and bank.shape[-1]==10 for x,bank in banks.items())
        changed=q.clone();changed[1:]*=-2
        altered,abanks,_,_=evaluate_chunks(changed,k,v,'hw',1,1,10,reference,samples)
        assert all(torch.equal(bank,abanks[x]) for x,bank in banks.items()) and not torch.equal(value,altered)
        ones,_,empty,_=evaluate_chunks(torch.zeros_like(q),k,torch.ones_like(v),'hw',1,1,10,reference,samples)
        torch.testing.assert_close(ones,torch.ones_like(ones),rtol=1e-6,atol=1e-6)
        assert all(x['empty_slots']*10==x['total_slots']*9 for x in empty)
        checks.append({'control':'original_rf0_c10','operator_and_seed_agreement':True,'exact_capacity':10,'rf0_reuse':True,'empty_clusters':True})
    for layout,a,b,c,old in [('hw',1,1,8,'q45_k80_c8_center'),('hw',1,4,16,'q45_k320_c16_intersections'),
                            ('wh',1,2,8,'q80_k90_c8_intersections'),('wh',2,2,16,'q160_k90_c16_intersections')]:
        expected,old_banks,_,_=evaluate_method(q,k,v,b*(80 if layout=='hw' else 45),c,old,reference,samples)
        value,banks,_,_=evaluate_chunks(q,k,v,layout,a,b,c,reference,samples)
        torch.testing.assert_close(value,expected,rtol=1e-5,atol=1e-6)
        assert all(torch.equal(bank,old_banks[x]) for x,bank in banks.items())
        checks.append({'control':old,'operator_and_seed_agreement':True})
    for recipe in [('hw',3,3,12),('wh',3,3,12),('hw',6,4,24),('wh',8,4,32),('hw',4,8,32),
                         ('wh',2,2,17),('wh',4,4,34),('hw',8,8,64),('hw',12,6,73),('hw',12,7,76,True),('hw',16,9,104,True),('wh',9,16,104,True)]:
        layout,a,b,c=recipe[:4];subset=len(recipe)==5
        value,banks,audits,g=evaluate_chunks(q,k,v,layout,a,b,c,reference,samples,allow_intersection_subset=subset)
        poisoned,pbanks,_,_=evaluate_chunks(q,k,v,layout,a,b,c,reference,samples,True,subset)
        torch.testing.assert_close(value,poisoned,rtol=1e-6,atol=1e-7)
        assert all(torch.equal(bank,pbanks[x]) for x,bank in banks.items())
        changed=q.clone();changed[1:]*=-2
        altered,abanks,_,_=evaluate_chunks(changed,k,v,layout,a,b,c,reference,samples,allow_intersection_subset=subset)
        assert all(torch.equal(bank,abanks[x]) for x,bank in banks.items()) and not torch.equal(value,altered)
        ones,_,empty,_=evaluate_chunks(torch.zeros_like(q),k,torch.ones_like(v),layout,a,b,c,reference,samples,allow_intersection_subset=subset)
        torch.testing.assert_close(ones,torch.ones_like(ones),rtol=1e-6,atol=1e-6)
        assert all(x['empty_slots']*c==x['total_slots']*(c-1) for x in empty)
        assert all(x['seed_intersection_policy_verified'] and not x['invalid_query_seeds'] for x in audits)
        checks.append({'layout':layout,'q_chunks':a,'k_chunks':b,'capacity':c,'padding_invariance':True,
            'rf0_reuse':True,'all_intersections':not subset,'intersection_subset_checked':subset,'exact_capacity':True,'empty_clusters':True,
            'query_index_overflow_checked':True})
    query=torch.randn(128,8);kp=torch.randn(2,16,8);pool=torch.arange(128).repeat(2,1);prefix=torch.arange(84).repeat(2,1);anchors=torch.zeros(2,dtype=torch.long)
    selected=select_intersections(query,kp,pool,anchors,prefix,76,True)
    restricted=select_intersections(query,kp,prefix,anchors,prefix[:,:1],76)
    assert torch.equal(selected,restricted),'Fisher subset must match selection on an intersections-only candidate pool'
    anchored=select_intersections(query,kp,pool,anchors,prefix,76,seed_policy='anchor_fisher')
    unrestricted=select_intersections(query,kp,pool,anchors,prefix[:,:1],76)
    assert torch.equal(anchored,unrestricted),'Anchor-only must equal ordinary Fisher over the same candidate pool'
    assert bool((anchored>=84).any()),'Mixed selector must permit non-intersection candidates'
    for policy in ('anchor_fisher','quarter_fisher'):
        for layout,a,b,c in [('hw',16,9,116),('wh',9,16,116),('hw',12,7,88),('hw',12,3,56)]:
            value,banks,audits,_=evaluate_chunks(q,k,v,layout,a,b,c,reference,samples,seed_policy=policy)
            poisoned,pbanks,_,_=evaluate_chunks(q,k,v,layout,a,b,c,reference,samples,True,seed_policy=policy)
            torch.testing.assert_close(value,poisoned,rtol=1e-6,atol=1e-7)
            assert all(torch.equal(bank,pbanks[x]) for x,bank in banks.items())
            changed=q.clone();changed[1:]*=-2
            altered,abanks,_,_=evaluate_chunks(changed,k,v,layout,a,b,c,reference,samples,seed_policy=policy)
            assert all(torch.equal(bank,abanks[x]) for x,bank in banks.items()) and not torch.equal(value,altered)
            ones,_,empty,_=evaluate_chunks(torch.zeros_like(q),k,torch.ones_like(v),layout,a,b,c,reference,samples,seed_policy=policy)
            torch.testing.assert_close(ones,torch.ones_like(ones),rtol=1e-6,atol=1e-6)
            assert all(x['empty_slots']*c==x['total_slots']*(c-1) for x in empty)
            assert sum(x['selected_nonintersection_slots'] for x in audits)>0
            checks.append({'layout':layout,'q_chunks':a,'k_chunks':b,'capacity':c,'seed_policy':policy,
                'padding_invariance':True,'rf0_reuse':True,'exact_capacity':True,'empty_clusters':True,'nonintersection_selection_verified':True})
    for _,c,name in METHODS:
        if not name.startswith('chunk_') or c<=128:continue
        layout,a,b,_=parse_chunk_name(name)
        assert a*(45 if layout=='hw' else 80)==b*(80 if layout=='hw' else 45)==3600
        probes=chunk_probe_count(name)
        value,banks,audits,g=evaluate_chunks(q,k,v,layout,a,b,c,reference,samples,allow_intersection_subset=True,probe_count=probes)
        assert g['candidate_budget']==3600 and all(not x['selected_nonintersection_slots'] for x in audits)
        changed=q.clone();changed[1:]*=-2
        altered,abanks,_,_=evaluate_chunks(changed,k,v,layout,a,b,c,reference,samples,allow_intersection_subset=True,probe_count=probes)
        assert all(torch.equal(bank,abanks[x]) for x,bank in banks.items()) and not torch.equal(value,altered)
        assert all(bank.shape[-1]==c and all(row.unique().numel()==c for row in bank) for bank in banks.values())
        ones,_,empty,_=evaluate_chunks(torch.zeros_like(q),k,torch.ones_like(v),layout,a,b,c,reference,samples,allow_intersection_subset=True,probe_count=probes)
        torch.testing.assert_close(ones,torch.ones_like(ones),rtol=1e-6,atol=1e-6)
        assert all(x['empty_slots']*c==x['total_slots']*(c-1) for x in empty)
        checks.append({'layout':layout,'capacity':c,'candidate_budget':3600,'probe_count':probes,'full_frame_all_queries_are_intersections':True,
            'rf0_reuse':True,'exact_capacity':True,'empty_clusters':True,'query_index_overflow_checked':True})
    return {'passed':True,'checks':checks}


def summarize(r):
    names=tuple(x[2] for x in METHODS)
    summary={}
    for name in names:
        rows=[x for x in r['rows'] if x['method']==name]
        if not rows:continue
        summary[name]={'cases':len(rows),'mean_relative_l2':statistics.mean(x['relative_l2'] for x in rows),
          'pooled_relative_l2':(sum(x['squared_error'] for x in rows)/sum(x['reference_squared'] for x in rows))**.5,
          'layer_means':{str(l):statistics.mean(x['relative_l2'] for x in rows if x['layer']==l)
                         for l in LAYERS if any(x['layer']==l for x in rows)}}
    pairs={}
    for row in r['rows']:pairs.setdefault((row['prompt'],row['step'],row['layer'],row['head']),{})[row['method']]=row
    summary['paired']={}
    for i,j in COMPARISONS:
        control,candidate=names[i],names[j]
        complete=[x for x in pairs.values() if control in x and candidate in x]
        summary['paired'][candidate+'_vs_'+control]={
            'cases':len(complete),
            'candidate_wins':sum(p[candidate]['relative_l2']<p[control]['relative_l2'] for p in complete),
            'control_wins':sum(p[control]['relative_l2']<p[candidate]['relative_l2'] for p in complete),
            'ties':sum(p[control]['relative_l2']==p[candidate]['relative_l2'] for p in complete)}
    r['summary']=summary


def worker(a):
    if os.environ.get('Q45_POW2_PARENT')!=str(os.getppid()):raise RuntimeError('Monitored parent required')
    import torch
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.init()
    native_bf16=torch.cuda.is_bf16_supported()
    if not native_bf16 and not a.allow_bf16_rounding_emulation:
        raise RuntimeError('Native BF16 GPU required unless the explicit mathematical-reference emulation check is enabled')
    precision={'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
        'device_capability':list(torch.cuda.get_device_capability()),'native_bf16':native_bf16,
        'bf16_rounding_emulation':not native_bf16,'matmul_dtype':'float32','fisher_dtype':'float64'}
    if not native_bf16 or a.compatibility_only:
        # Reference uses BF16 storage/rounding only, never BF16 GEMM. Check
        # conversion on both devices and FP32 GEMM before loading real QKV.
        values=torch.linspace(-100,100,20003,dtype=torch.float32)+0.000123
        assert torch.equal(values.bfloat16().float(),values.cuda().bfloat16().float().cpu())
        generator=torch.Generator().manual_seed(1609032)
        aa=torch.randn(4,17,128,generator=generator);bb=torch.randn(4,128,19,generator=generator)
        expected=torch.bmm(aa,bb);actual=torch.bmm(aa.cuda(),bb.cuda()).cpu()
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-5)
        precision.update(bf16_cpu_gpu_rounding_identical=True,fp32_bmm_checked=True,
                         fp32_bmm_max_abs_difference=float((actual-expected).abs().max()))
    if a.compatibility_only:
        save(a.output/'results.json',{'complete':True,'compatibility_only':True,'precision':precision,
             'gpu_model':torch.cuda.get_device_name(),'server':socket.gethostname()})
        return
    reference=load_reference(a.reference_script)
    old=json.loads((a.input/'results.json').read_text())
    assert old['complete'] and old['layers']==list(LAYERS) and old['capacity']==8
    assert json.loads((a.input/'completion.json').read_text())['valid']
    assert sha(a.reference_script)==old['script_sha256']
    old_rows={(x['prompt'],x['step'],x['layer'],x['head'],x['axes']):x for x in old['rows']}
    r={'complete':False,'suite':a.suite,'refine':a.refine,'max_tflops':a.max_tflops,'methods':[{'name':n,'key_block_length':k,'capacity':c,'seed_policy':chunk_seed_policy(n) if n.startswith('chunk_') else n.rsplit('_',1)[-1]} for k,c,n in METHODS],
       'layers':list(LAYERS),'heads':list(HEADS),'grid':list(GRID),'samples':SAMPLES,
       'server':socket.gethostname(),'gpu_model':torch.cuda.get_device_name(),'gpu_index':a.gpu,'precision':precision,
       'prompts':3,'steps':4,'expected_paired_cases':384,'query_groups':'Q45=physical H; Q80=physical W; Q160=two adjacent W80 rows, final unpaired row valid80',
       'key_layout':a.boundary,'selector':('FP64 anchor-Fisher Uniform16; Q80 original C32/g45; Q160 per-row 16 spatially sampled from C32 or full per-row C32 union; all intersections retained' if a.suite=='q160' else 'FP64 anchor-Fisher C32/g45 Uniform16; forced prefix depends on seed policy, candidate table unchanged'),
       'anchor_policy':'center control vs all actual Q-column/K-block intersections; center retained first when valid, remaining intersections forced in token order, then Fisher; exact method capacity',
       'rf0_policy':'select actual query row indices once at t=0 per query w and key block; reuse across t',
       'padding_policy':'invalid tail keys masked in normalization, ownership, merge and routing; no extra route',
       'baseline_input':str(a.input),'baseline_results_sha256':sha(a.input/'results.json'),
       'script_sha256':sha(__file__),'reference_script_sha256':sha(a.reference_script),
       'selftest':selftest(reference),'paired_query_selftest':selftest_paired_queries(reference) if a.suite=='q160' else None,
       'chunk_selftest':selftest_chunks(reference) if a.suite=='chunks' else None,
       'rows':[],'captures':[],'geometry':{},
       'mean_attention_ms':None,'speedup_vs_sdpa':None,'psnr':None,'ssim':None,'lpips':None,'vbench':None,
       'timing_note':'Sampled mathematical reference; not complete-call latency or native C8 r2 benchmark'}
    if a.suite=='chunks':
        r['query_groups']='Adjacent a complete spatial query chunks within each frame; final smaller group stays valid; key chunks globally flattened and may cross frames'
        r['selector']='FP64 Fisher, Uniform16; candidate budget=max(32,next_power_of_two(max(C,q_chunks*k_chunks))), balanced across valid query chunks. All intersections protected unless an explicitly named junction_subset method has more intersections than C, in which case Fisher selects exclusively among intersections. Spatial farthest fill from original per-chunk C32/g45.'
        r['chunk_plan']=[{'name':n,'layout':parse_chunk_name(n)[0],'q_chunks':parse_chunk_name(n)[1],
                         'k_chunks':parse_chunk_name(n)[2],'capacity':c,
                         'flops':chunk_flops(*GRID,*parse_chunk_name(n),probe_count=chunk_probe_count(n))} for _,c,n in METHODS]
    save(a.output/'results.json',r)
    shutil.copyfile(a.input/'prompt_manifest.json',a.output/'prompt_manifest.json')
    coords=list(itertools.product(*SAMPLES))
    for capture in old['captures']:
        path=Path(capture['path'])
        if not path.exists():raise FileNotFoundError(path)
        assert sha(path)==capture['sha256'],'QKV source changed'
        payload=torch.load(str(path),map_location='cpu',weights_only=True,mmap=True)
        identity={key:capture[key] for key in ('prompt','step','layer','call')}
        assert payload['metadata']['heads']==list(HEADS)
        assert all(payload['metadata'][key]==value for key,value in identity.items())
        assert all(tuple(payload[name].shape)==(1,75600,4,128) for name in ('q','k','v','dense_output'))
        q,k,v=[payload[name].to('cuda') for name in ('q','k','v')]
        seed_payload={}
        for slot,head in enumerate(HEADS):
            qh,kh,vh=[x[0,:,slot].float().reshape(*GRID,128) for x in (q,k,v)]
            queries=torch.stack([qh[c] for c in coords])
            target=(queries@kh.reshape(-1,128).T*128**-.5).softmax(-1)@vh.reshape(-1,128)
            key=(identity['prompt'],identity['step'],identity['layer'],head)
            prior=old_rows[(*key,'hw')]
            reference_squared=float(target.square().sum())
            assert abs(reference_squared/prior['reference_squared']-1)<1e-5
            for block_length,capacity,name in METHODS:
                policy=chunk_seed_policy(name) if name.startswith('chunk_') else name.rsplit('_',1)[-1]
                prediction,banks,audits,geometry=evaluate_method(qh,kh,vh,block_length,capacity,name,reference)
                r['geometry'][name]=geometry
                delta=prediction-target
                row={**identity,'head':head,'method':name,'capacity':capacity,'block_length':block_length,'seed_policy':policy,
                     'relative_l2':float(delta.norm()/target.norm()),
                     'squared_error':float(delta.square().sum()),'reference_squared':reference_squared,
                     'seed_audit':audits,'empty_fraction':sum(x['empty_slots'] for x in audits)/sum(x['total_slots'] for x in audits)}
                if name.startswith('chunk_'):
                    layout,ac,bc,cc=parse_chunk_name(name)
                    planes=GRID[2] if layout=='hw' else GRID[1]
                    tail=torch.tensor([((coord[2] if layout=='hw' else coord[1])//ac*ac+ac)>planes for coord in coords],device=q.device)
                    row['query_group_segments']={label:{'query_count':int(mask.sum()),
                        'squared_error':float(delta[mask].square().sum()),'reference_squared':float(target[mask].square().sum())}
                        for label,mask in [('full',~tail),('tail',tail)]}
                if name in ('q45_k80_c8_center','q80_k45_c8_center'):
                    axis='hw' if name.startswith('q45') else 'wh'
                    matching_prior=old_rows[(*key,axis)]
                    row['baseline_replay_rel_l2_difference']=abs(row['relative_l2']-matching_prior['relative_l2'])
                    prior_hashes={x['plane']:x['index_sha256'] for x in matching_prior['seed_audit']}
                    row['baseline_seed_hash_matches']=all(x['index_sha256']==prior_hashes[x['query_w' if axis=='hw' else 'query_h']] for x in audits)
                r['rows'].append(row)
                seed_payload[f'{name}_h{head}']=banks
            del qh,kh,vh,queries,target,prediction,delta
        stem=f"p{identity['prompt']:03d}_s{identity['step']}_l{identity['layer']:02d}"
        seed_path=a.output/'seed_indices'/(stem+'.pt')
        with seed_path.open('xb') as stream:torch.save({'metadata':identity,'banks':seed_payload},stream)
        r['captures'].append({**identity,'qkv_path':str(path),'qkv_sha256':capture['sha256'],
                              'seed_indices_path':str(seed_path),'seed_indices_sha256':sha(seed_path)})
        summarize(r);save(a.output/'results.json',r)
        print(json.dumps({'capture':stem,'captures':len(r['captures']),
              'mean_relative_l2':{n:r['summary'][n]['mean_relative_l2'] for _,_,n in METHODS}}),flush=True)
        del payload,q,k,v,seed_payload
    expected={(p,s,l,h,m) for p in range(3) for s in range(4) for l in LAYERS for h in HEADS for _,_,m in METHODS}
    assert len(r['rows'])==len(expected)==384*len(METHODS)
    assert {(x['prompt'],x['step'],x['layer'],x['head'],x['method']) for x in r['rows']}==expected
    assert len(r['captures'])==96
    assert sha(a.reference_script)==r['reference_script_sha256']
    summarize(r);r['complete']=True;save(a.output/'results.json',r)
    print(json.dumps({'complete':True,'summary':r['summary']}),flush=True)


def parent(a):
    reference=load_reference(a.reference_script)
    def clean():
        state=reference.gpu_state(a.gpu)
        if state['pids'] or state['memory_mib']>100 or state['utilization']:
            raise RuntimeError(f'No exclusively idle GPU; no launch or queue: {state}')
        return state
    initial=clean()
    a.output.mkdir(parents=True,exist_ok=False)
    save(a.output/'preflight.json',initial)
    frozen=a.output/Path(__file__).name;shutil.copyfile(__file__,frozen)
    frozen_reference=a.output/'rf0_reference.py';shutil.copyfile(a.reference_script,frozen_reference)
    (a.output/'seed_indices').mkdir()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),Q45_POW2_PARENT=str(os.getpid()),OMP_NUM_THREADS='4',
             MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,'-u',str(frozen),'--worker','--reference-script',str(frozen_reference),
             '--input',str(a.input),'--output',str(a.output),'--gpu',str(a.gpu),'--boundary',a.boundary,
             '--suite',a.suite,'--max-tflops',str(a.max_tflops)]
    if a.allow_bf16_rounding_emulation:command.append('--allow-bf16-rounding-emulation')
    if a.compatibility_only:command.append('--compatibility-only')
    if a.refine:command.append('--refine')
    if a.recipes:
        frozen_recipes=a.output/'recipes.json';shutil.copyfile(a.recipes,frozen_recipes)
        command.extend(['--recipes',str(frozen_recipes)])
    clean();child=None;failure=None
    try:
        with (a.output/'worker.log').open('x') as log,(a.output/'gpu_monitor.jsonl').open('x') as monitor:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            save(a.output/'process.json',{'parent_pid':os.getpid(),'child_pid':child.pid,'command':command})
            while True:
                state=reference.gpu_state(a.gpu)
                monitor.write(json.dumps({'time':time.time(),**state})+'\n');monitor.flush()
                if set(state['pids'])-{child.pid}:raise RuntimeError('Foreign GPU process; comparison invalid')
                if child.poll() is not None:break
                time.sleep(2)
            if child.returncode:raise RuntimeError(f'Worker exit {child.returncode}')
            assert json.loads((a.output/'results.json').read_text())['complete']
    except BaseException as exc:failure=str(exc)
    finally:
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=20)
        save(a.output/'completion.json',{'valid':failure is None,'failure':failure,
                                       'worker_exit_code':None if child is None else child.returncode})
    if failure:raise RuntimeError(failure)


def main():
    global METHODS,COMPARISONS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-script',type=Path,required=True)
    p.add_argument('--input',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--boundary',choices=('global',),default='global')
    p.add_argument('--suite',choices=('block_extensions','q80_equal_count','q160','chunks'),default='block_extensions')
    p.add_argument('--gpu',type=int)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--allow-bf16-rounding-emulation',action='store_true',help='Mathematical reference only; validate BF16 rounding and use FP32 matmul on legacy GPUs')
    p.add_argument('--compatibility-only',action='store_true')
    p.add_argument('--refine',action='store_true',help='Eight intermediate-capacity chunk recipes and original C8 replay')
    p.add_argument('--recipes',type=Path,help='Explicit budget-checked chunk recipes; original C8 is prepended as replay control')
    p.add_argument('--max-tflops',type=float,default=43.47592704,help='Exclusive candidate major-matmul FLOPs limit; freeze the measured baseline used')
    p.add_argument('--selftest',action='store_true')
    a=p.parse_args()
    if a.suite=='q80_equal_count':
        METHODS=((45,4,'q80_k45_c4_center'),(90,8,'q80_k90_c8_center'),
                 (90,8,'q80_k90_c8_intersections'))
        COMPARISONS=((0,1),(0,2),(1,2))
    elif a.suite=='q160':
        METHODS=((90,8,'q80_k90_c8_intersections'),(90,16,'q80_k90_c16_intersections'),
                 (90,16,'q160_k90_c16_intersections'),(90,16,'q160_k90_c16_candidates64_intersections'))
        COMPARISONS=((0,2),(1,2),(2,3),(0,1))
    elif a.suite=='chunks':
        recipes=[('hw',1,1,8),('wh',1,2,8),('wh',2,2,16),('hw',1,4,16)]
        recipes += [(layout,a,b,c) for layout in ('hw','wh') for a,b,c in CHUNK_RECIPES]
        if a.refine:recipes=[('hw',1,1,8),*REFINE_RECIPES]
        if a.recipes:
            selected=json.loads(a.recipes.read_text())
            assert selected and len({tuple(x) for x in selected})==len(selected)
            for recipe in selected:
                assert len(recipe) in (4,5)
                layout,qa,kb,c=recipe[:4];variant=recipe[4] if len(recipe)==5 else None
                assert variant in (None,'junction_subset','anchor_fisher','quarter_fisher','probe64','probe128')
                assert layout in ('hw','wh') and all(isinstance(x,int) and x>=1 for x in (qa,kb,c))
                probes=64 if variant=='probe64' else 128 if variant=='probe128' else 16
                assert 1<=c<=GRID[1]*GRID[2] and (qa*kb<=c or variant is not None) and chunk_flops(*GRID,layout,qa,kb,c,probe_count=probes)['total_tflops']<a.max_tflops
                assert ('hw',1,1,8)!=(layout,qa,kb,c)
            recipes=[('hw',1,1,8),*selected]
        METHODS=tuple((recipe[2]*(80 if recipe[0]=='hw' else 45),recipe[3],
                       f'chunk_{recipe[0]}_q{recipe[1]}_k{recipe[2]}_c{recipe[3]}'+('_'+recipe[4] if len(recipe)==5 else '')) for recipe in recipes)
        COMPARISONS=tuple((0,i) for i in range(1,len(METHODS)))
    if a.selftest:
        reference=load_reference(a.reference_script)
        checks=selftest(reference)
        if a.suite=='q160':checks['paired_queries']=selftest_paired_queries(reference)
        if a.suite=='chunks':checks['chunks']=selftest_chunks(reference)
        print(json.dumps(checks));return
    if a.input is None or a.output is None:p.error('--input and --output required')
    if a.gpu is None:p.error('--gpu must explicitly identify an exclusively idle GPU')
    worker(a) if a.worker else parent(a)


if __name__=='__main__':main()
