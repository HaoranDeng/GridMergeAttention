"""CPU precision diagnostic on frozen QKV, not GPU validation or ranking."""
import argparse
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from q80k90_kernel import candidate_table

p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,default=Path('/local2/denghaoran/GridMergeAttention/experiments/rf0_spatial_c8_8layers_20260919_v1'))
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();torch.set_num_threads(2);torch.set_grad_enabled(False)
tables=torch.tensor(candidate_table())


def select(z,cids,anchor,capacity,dtype):
    z=z.to(dtype);aidx=int((cids==anchor).nonzero()[0]);bidx=int((cids==anchor+1).nonzero()[0])
    za=z[aidx];rho=(za-za.max()).exp();rho/=rho.sum()
    delta=z-za;residual=(delta-(delta*rho).sum(1)[:,None])*rho.sqrt()
    used=torch.zeros(32,dtype=torch.bool);used[aidx]=True;picks=[anchor]
    for slot in range(1,capacity):
        score=(residual*residual).sum(1).masked_fill(used,-float('inf'))
        index=bidx if slot==1 else int(score.argmax())
        picks.append(int(cids[index]));used[index]=True
        if slot<capacity-1:
            direction=residual[index];basis=direction/(direction.square().sum()+1e-12).sqrt()
            residual=residual-(residual*basis).sum(1)[:,None]*basis
    assert len(set(picks))==capacity and picks[:2]==[anchor,anchor+1]
    return picks


result=dict(complete=False,gpu_used=False,prompt=0,step=0,layers=[0,5,10,15,20,25,30,35],
            heads=[0,10,20,30],query_groups=[0,22,44],key_groups=[0,40,420,839],
            scope='CPU FP32 vs FP64 selector arithmetic on sampled chunks; no GPU execution, full numerical gate still required',rows=[])
for layer in result['layers']:
    capture=torch.load(a.data/'captures'/f'p000_s0_l{layer:02d}.pt',map_location='cpu',weights_only=False)
    for cap in (6,8):
        changed=cases_changed=cases=0
        for head in range(4):
            q=capture['q'][0,:,head].reshape(21,45,80,128)
            k=capture['k'][0,:,head].reshape(21,45,40,2,128).permute(0,2,3,1,4).reshape(840,90,128)
            for g in result['query_groups']:
                for kg in result['key_groups']:
                    anchor=2*(kg%40);cids=tables[kg%40]
                    z=(q[0,g,cids].float()@k[kg,torch.arange(16)*90//16].float().T)*(128**-.5)
                    left=select(z,cids,anchor,cap,torch.float64)
                    right=select(z,cids,anchor,cap,torch.float32)
                    mismatch=sum(x!=y for x,y in zip(left,right))
                    changed+=mismatch;cases_changed+=bool(mismatch);cases+=1
        row=dict(layer=layer,capacity=cap,cases=cases,changed_cases=cases_changed,
                 changed_seed_slots=changed,total_seed_slots=cases*cap)
        result['rows'].append(row);print(json.dumps(row),flush=True)
    del capture,q,k
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
result['summary']={str(cap):dict(cases=sum(r['cases'] for r in result['rows'] if r['capacity']==cap),
                    changed_cases=sum(r['changed_cases'] for r in result['rows'] if r['capacity']==cap),
                    changed_seed_fraction=sum(r['changed_seed_slots'] for r in result['rows'] if r['capacity']==cap)/sum(r['total_seed_slots'] for r in result['rows'] if r['capacity']==cap))
                   for cap in (6,8)}
result['complete']=True
assert not torch.cuda.is_initialized()
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['summary']),flush=True)
