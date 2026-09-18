"""Isolated C10 extension of the original geometric-Fisher RF0 operator.

Ten distinct seeds use sixteen physical slots. Six trailing duplicate seeds
lose all ownership ties to the earlier real seed and therefore have zero mass.
The route scans padded empty cells: timings include this unoptimized overhead.
No model is loaded and no GPU is selected by importing this module.
"""
from __future__ import annotations

import types


class FisherC10MergeAdapter:
    def __init__(self, *, scaled_fp8_kv=False):
        if type(scaled_fp8_kv) is not bool:
            raise ValueError("scaled_fp8_kv must be bool")
        self.scaled_fp8_kv = scaled_fp8_kv
        from attention import singleton_c4_wan21_14b_720p_triton as module
        original = module._hard_c4_720p_temporal_shared_global_mass_attention_triton_blocks
        namespace = dict(original.__globals__)
        namespace["SINGLETON_C"] = 16
        self.inner = types.FunctionType(original.__code__, namespace,
                                       original.__name__, original.__defaults__, original.__closure__)
        self.inner.__kwdefaults__ = original.__kwdefaults__
        self.calls = 0

    def __call__(self, q, k, v, seeds, scale, **kwargs):
        if seeds.shape[-1] != 10 or kwargs.pop("selector", None) != "query_column":
            raise ValueError("C10 adapter requires ten real seeds and query-column SQ1")
        if kwargs.get("top_r") != 1 or not kwargs.get("seed_rows_native_layout"):
            raise ValueError("C10 requires one SQ1 and the original native RF0 layout")
        for key in ("direct_bulk_normalization", "fp8_score_dot", "split_merge",
                    "compressed_fp8_kv", "scaled_fp8_kv", "materialize_singleton"):
            if kwargs.get(key, False):
                raise ValueError("C10 changes only capacity, not " + key)
        if self.scaled_fp8_kv:
            kwargs["scaled_fp8_kv"] = True
        self.calls += 1
        # Triton tiles are power-of-two. Duplicate the final real seed into
        # six trailing slots. Identical logits plus leftmost argmax assign
        # zero keys to those slots, yielding zero vectors and +inf bias.
        # The independent C10 full-support reference below checks semantics.
        import torch
        padded = torch.cat((seeds, seeds[..., -1:].expand(*seeds.shape[:-1], 6)), -1)
        # C16 physical tiles need a smaller column block to avoid spilling
        # registers. Callers can explicitly reproduce the old J16 baseline.
        kwargs.setdefault("block_j", 4)
        return self.inner(q,k,v,padded,scale=scale,query_column_selector=True,
                          bias_only=False,defer_correction=False,**kwargs)


def validate_native():
    """Check nine pivots, C4-prefix parity, C10 SQ1 output and empty cells."""
    import torch
    from attention.affine_hellinger_seed_triton import affine_hellinger_seed_rows_triton
    from attention.adaptive_seed_pytorch import multiscale_candidate_table
    torch.manual_seed(817)
    q=torch.randn((1,21,45,80,1,128),device='cuda',dtype=torch.bfloat16)
    k=torch.randn_like(q)
    v=torch.randn_like(k)
    scale=128**-.5
    select=lambda c:affine_hellinger_seed_rows_triton(q[:,:1],k,feature_mode='fisher_geometric',num_seeds=c)
    c4,c10=select(4),select(10)
    assert torch.equal(c4,c10[...,:4]), 'C4 prefix changed'
    assert bool((c10.sort(-1).values[...,1:]!=c10.sort(-1).values[...,:-1]).all()), 'duplicate C10 rows'
    candidates,anchors=multiscale_candidate_table(45,32,q.device,geometry_count=45)
    seed_checks=0
    for row in (0,23,44):
        for col in (0,40,79):
            z=q[0,0,candidates[row].long(),col,0].float()@k[0,0,row,::5,0].float().T
            z=z.double()*scale
            rho=z.mean(0).softmax(0)
            delta=z-z[anchors[row].long()]
            residual=(delta-(delta*rho).sum(-1,keepdim=True))*rho.sqrt()
            picks=[int(anchors[row])]
            for _ in range(9):
                score=(residual*residual).sum(-1); score[picks]=-torch.inf
                picked=int(score.argmax()); picks.append(picked)
                basis=residual[picked]/((residual[picked]**2).sum()+1e-12).sqrt()
                residual=residual-(residual@basis)[:,None]*basis
            assert torch.equal(c10[0,0,0,row,col,0].long(),candidates[row,picks].long()), 'C10 GS mismatch'
            seed_checks+=10
    options=dict(top_r=1,head_start=0,head_count=1,selector='query_column',
                 seed_rows_native_layout=True,query_column_major=True,
                 merge_loop_stages=1,merge_disable_licm=False,split_kv_accum=True)
    adapter=FisherC10MergeAdapter()
    out=adapter(q,k,v,c10,scale,**options)
    assert bool(torch.isfinite(out).all()), 'nonfinite random C10 output'
    # Independent full-support grouped attention on three current queries.
    max_error=0.
    for col in (0,40,79):
        rows=c10[0,0,:,:,col,0].long().reshape(945,10)
        qr=q[0,10,rows,col,0].float()
        keys=k[0,:,:,:,0].float().reshape(945,80,128)
        values=v[0,:,:,:,0].float().reshape(945,80,128)
        logits=torch.bmm(qr,keys.transpose(1,2))*scale
        owner=logits.log_softmax(-1).argmax(1)
        mask=owner[:,None,:]==torch.arange(10,device=q.device)[None,:,None]
        mask[:,:,col]=False
        valid=mask.any(-1)
        weights=logits.masked_fill(~mask,-torch.inf).softmax(-1).nan_to_num(0.)
        mk=torch.bmm(weights,keys).reshape(-1,128)
        mv=torch.bmm(weights,values).reshape(-1,128)
        bias=(weights*weights.clamp_min(1e-38).log()).sum(-1).masked_fill(~valid,torch.inf).flatten()
        mk=torch.cat((mk,keys[:,col])); mv=torch.cat((mv,values[:,col]))
        bias=torch.cat((bias,torch.zeros(945,device=q.device)))
        z=q[0,10,23,col,0].float()@mk.T*scale-bias
        expected=z.softmax(-1)@mv
        error=float((out[0,10,23,col,0].float()-expected).abs().max())
        max_error=max(max_error,error)
    assert max_error<.02, f'C10 full compressed attention mismatch: {max_error}'
    q.zero_()
    tied=select(10)
    empty_output=adapter(q,k,v,tied,scale,**options)
    expected=v.float().mean((1,2,3),keepdim=True)
    empty_error=float((empty_output.float()-expected).abs().max())
    assert empty_error<.002, f'C10 ties/empty-cell mismatch: {empty_error}'
    from attention import singleton_c4_wan21_14b_720p_triton as module
    assert module.SINGLETON_C==4, 'production capacity changed'
    proof=dict(valid=True,c=10,physical_capacity=16,merge_block_j=4,merge_loop_stages=1,merge_num_warps=4,
               padding="six trailing duplicate seeds; leftmost ownership ties",seed_indices_checked=seed_checks,c4_prefix_exact=True,
               nine_pivots_checked=True,full_key_groups_per_query=945,
               full_route_max_abs=max_error,empty_cell_max_abs=empty_error,
               production_capacity_unchanged=True,sq1_per_key_group=1)
    del q,k,v,c4,c10,tied,out,empty_output
    torch.cuda.empty_cache()
    return proof
