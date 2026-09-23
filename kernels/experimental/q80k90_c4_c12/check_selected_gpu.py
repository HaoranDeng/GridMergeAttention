"""All-output and tail-head checks of the exact selected configuration."""
import gc
import torch
from q80k90_kernel import Q80K90


def check(capacity,config):
    torch.manual_seed(9045)
    heads=config.get('head_chunk',4)+1
    inputs=[torch.randn(1,7200,heads,128,device='cuda',dtype=torch.bfloat16)*.25 for _ in range(3)]
    selected=Q80K90(capacity=capacity,**config)
    actual,seeds=selected(*inputs,return_debug=True)
    seeds=seeds.clone()
    independent=dict(config,merge_backend='triton',route_backend='triton',route_m=128,route_n=64,
        route_warps=4,route_stages=2,pipeline=False)
    reference=Q80K90(capacity=capacity,**independent)
    expected=reference(*inputs)
    error=float((actual.float()-expected.float()).norm()/expected.float().norm())
    assert torch.isfinite(actual).all() and error<.02,error
    assert torch.equal(seeds,reference.seeds)
    modified=inputs[0].clone();modified[:,3600:]*=-2
    selected(modified,inputs[1],inputs[2])
    assert torch.equal(seeds,selected.seeds)
    sorted_seeds=seeds.sort(-1).values
    assert (sorted_seeds[...,1:]>sorted_seeds[...,:-1]).all()
    anchors=(torch.arange(80,device='cuda')%40)*2
    assert torch.equal(seeds[...,0],anchors[None,None,:].expand_as(seeds[...,0]))
    assert torch.equal(seeds[...,1],(anchors+1)[None,None,:].expand_as(seeds[...,1]))
    ones=selected(torch.zeros_like(inputs[0]),inputs[1],torch.ones_like(inputs[2]))
    assert torch.allclose(ones,torch.ones_like(ones),atol=.01,rtol=0)
    # hc+1 heads leave one active head in the last workspace bank.
    assert torch.isfinite(selected.bias[:1,...,0]).all()
    assert torch.isneginf(selected.bias[:1,...,1:]).all()
    assert not torch.count_nonzero(selected.mk[:1,...,1:,:])
    assert not torch.count_nonzero(selected.mv[:1,...,1:,:])
    result=dict(passed=True,shape=list(inputs[0].shape),all_output_positions_checked=True,
        partial_head_chunk_covered=True,relative_l2_vs_independent_triton=error,
        rf0_seed_reuse=True,exact_distinct_capacity=capacity,two_intersection_anchors=True,
        empty_cluster_zero_kv_negative_infinity_bias=True,constant_value_preserved=True)
    del inputs,selected,reference,actual,expected,seeds,modified,sorted_seeds,ones
    gc.collect();torch.cuda.empty_cache()
    return result
