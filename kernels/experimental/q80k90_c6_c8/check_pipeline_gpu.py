"""GPU data-hazard checks for the double-buffered implementation."""
import gc
import torch
from q80k90_kernel import Q80K90


def check(capacity,config,first_capture):
    if not config.get('pipeline',False):return {'applicable':False}
    serial_config=dict(config,pipeline=False)
    torch.manual_seed(9045)
    # Three chunks exercise buffer reuse, plus a partial final head chunk.
    heads=2*config['head_chunk']+1
    a=[torch.randn(1,7200,heads,128,device='cuda',dtype=torch.bfloat16)*.25 for _ in range(3)]
    b=[-a[0],a[1].flip(1).contiguous(),a[2]*.5]
    piped=Q80K90(capacity,**config);serial=Q80K90(capacity,**serial_config)
    out_a=piped(*a);out_b=piped(*b)
    expected_a=serial(*a);expected_b=serial(*b)
    assert torch.equal(out_a,expected_a),'First returned output changed or pipeline ordering mismatch'
    assert torch.equal(out_b,expected_b),'Reused bank ordering mismatch'
    ones=piped(torch.zeros_like(a[0]),a[1],torch.ones_like(a[2]))
    assert torch.allclose(ones,torch.ones_like(ones),atol=.01,rtol=0)
    assert torch.isfinite(piped.bias[:1,...,0]).all()
    assert torch.isneginf(piped.bias[:1,...,1:]).all()
    assert not torch.count_nonzero(piped.mk[:1,...,1:,:])
    assert not torch.count_nonzero(piped.mv[:1,...,1:,:])
    del a,b,piped,serial,out_a,out_b,expected_a,expected_b,ones
    gc.collect();torch.cuda.empty_cache()
    payload=torch.load(first_capture,map_location='cpu',weights_only=False)
    full=[payload[name].to('cuda').repeat(1,1,10,1).contiguous() for name in ('q','k','v')]
    piped=Q80K90(capacity,**config);serial=Q80K90(capacity,**serial_config)
    actual=piped(*full);expected=serial(*full)
    assert torch.equal(actual,expected),'Full 40-head pipeline must match sequential dispatch'
    del payload,full,piped,serial,actual,expected
    gc.collect();torch.cuda.empty_cache()
    return dict(applicable=True,passed=True,partial_heads=heads,buffer_reuse=True,
                consecutive_distinct_calls_bitwise_equal=True,returned_outputs_independent=True,
                empty_cluster_zero_kv_negative_infinity_bias=True,full40_all_tokens_bitwise_equal=True)
