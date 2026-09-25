"""CPU literal-operator checks independent of the CUDA implementation."""
import json
import torch
from attention.axis_current_c8_reference import prepare,line_attention,select_fisher


def main():
    torch.set_num_threads(1);torch.manual_seed(719)
    q,k,v=[torch.randn((2,9,12,8),dtype=torch.float64) for _ in range(3)]
    kb,vb,candidates,slots,probes=prepare(k,v,2,1)
    query=q[1,8]
    seeds=select_fisher(query,kb,candidates,slots,probes,8**-.5)
    assert seeds.shape==(24,8)
    assert all(len(set(row.tolist()))==8 for row in seeds)
    keys,values,biases=[],[],[]
    for group in range(len(kb)):
        logits=query[seeds[group]]@kb[group].T*(8**-.5)
        owners=logits.log_softmax(-1).argmax(0)
        for cluster in range(8):
            selected=owners==cluster
            if selected.any():
                weights=logits[cluster,selected].softmax(0)
                keys.append(weights@kb[group,selected]);values.append(weights@vb[group,selected])
                biases.append(-(weights*weights.log()).sum())
    expected=(query@torch.stack(keys).T*(8**-.5)+torch.stack(biases)).softmax(-1)@torch.stack(values)
    actual,_=line_attention(query,query,kb,vb,candidates,slots,probes,list(range(12)),quantize=False)
    assert torch.allclose(actual,expected,atol=1e-12,rtol=1e-12)
    zero=torch.zeros_like(query)
    out,counts=line_attention(zero,zero,kb,vb,candidates,slots,probes,list(range(12)),quantize=False)
    assert counts['empty']==counts['slots']*7//8
    assert torch.allclose(out,v.mean((0,1,2)).expand_as(out),atol=1e-12,rtol=1e-12)
    constant=torch.full_like(vb,2.5)
    out,_=line_attention(query,query,kb,constant,candidates,slots,probes,list(range(12)),quantize=False)
    assert torch.allclose(out,torch.full_like(out,2.5),atol=1e-12,rtol=1e-12)
    print(json.dumps({'passed':True,'checks':['eight distinct seeds','literal scalar C8 operator',
                                            'seven empty slots for zero Q','dense uniform identity','constant V']}))


if __name__=='__main__':main()
