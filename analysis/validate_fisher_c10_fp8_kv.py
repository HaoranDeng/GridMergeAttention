"""Pre-model FP8 storage/scale/route check; this is not video quality evidence."""
from __future__ import annotations


class KernelCapture:
    def __init__(self, kernel):
        self.kernel = kernel
        self.args = None
        self.kwargs = None

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.args, self.kwargs = args, kwargs
            return self.kernel[grid](*args, **kwargs)
        return launch


def validate_fp8():
    import types
    import torch
    from analysis.fisher_c10_video_adapter import FisherC10MergeAdapter
    from attention.affine_hellinger_seed_triton import fisher_geometric_c10_seed_rows_triton
    from attention import hard_c4_wan21_14b_720p_triton as hard

    torch.manual_seed(819)
    shape = (1, 21, 45, 80, 1, 128)
    q = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    scale = 128 ** -.5
    seeds = fisher_geometric_c10_seed_rows_triton(q[:, :1], k, scale)
    options = dict(top_r=1, head_start=0, head_count=1, selector='query_column',
                   seed_rows_native_layout=True, query_column_major=True,
                   merge_loop_stages=1, merge_disable_licm=False, split_kv_accum=True)
    baseline = FisherC10MergeAdapter()(q, k, v, seeds, scale, **options)
    adapter = FisherC10MergeAdapter(scaled_fp8_kv=True)
    merge = KernelCapture(hard._hard_c4_720p_temporal_shared_merge_kernel)
    route = KernelCapture(hard._hard_c4_720p_route_kernel)
    proxy = types.SimpleNamespace(**vars(hard))
    proxy._hard_c4_720p_temporal_shared_merge_kernel = merge
    proxy._hard_c4_720p_route_kernel = route
    adapter.inner.__globals__['hard_c4'] = proxy
    output = adapter(q, k, v, seeds, scale, **options)
    torch.cuda.synchronize()
    mk, mv, ks, vs, bias = merge.args[5:10]
    assert mk.dtype == mv.dtype == torch.float8_e4m3fn
    assert ks.dtype == vs.dtype == bias.dtype == torch.float32
    assert ks.numel() == vs.numel() == bias.numel() == mk.shape[0]
    assert merge.kwargs['SCALED_FP8_KV'] is True and route.kwargs['SCALED_FP8_KV'] is True
    assert route.kwargs['COMPRESSED_FP8'] is False  # BF16 route arithmetic after decode.
    assert merge.kwargs['FP8_SCORE_DOT'] is False
    assert merge.kwargs['FP8_MERGE_K_DOT'] is False and merge.kwargs['FP8_MERGE_V_DOT'] is False
    assert all(a.data_ptr() == b.data_ptr() for a, b in zip((mk,mv,ks,vs,bias),route.args[3:8]))
    assert bool(torch.isfinite(output).all())
    delta = output.float() - baseline.float()
    rel_l2 = float(delta.norm() / baseline.float().norm())
    assert rel_l2 < .15, f'gross FP8 error: {rel_l2}'
    samples = []
    # All 945 key groups, all 16 physical slots and original SQ1 in each sample.
    for qt, col, row in ((0,0,0), (10,40,23), (20,79,44)):
        begin = (qt * 80 + col) * 945 * 16
        sl = slice(begin, begin + 945 * 16)
        key = (mk[sl].float() * ks[sl,None]).to(q.dtype).float()
        value = (mv[sl].float() * vs[sl,None]).to(q.dtype).float()
        b = bias[sl]
        assert bool(torch.isfinite(key).all() & torch.isfinite(value).all())
        assert bool((ks[sl] > 0).all() & (vs[sl] > 0).all())
        assert bool((key.reshape(945,16,128)[:,10:] == 0).all())
        assert bool((value.reshape(945,16,128)[:,10:] == 0).all())
        assert bool(torch.isposinf(b.reshape(945,16)[:,10:]).all())
        key = torch.cat((key, k[0,:,:,col,0].reshape(945,128).float()))
        value = torch.cat((value, v[0,:,:,col,0].reshape(945,128).float()))
        b = torch.cat((b, torch.zeros(945,device=q.device)))
        expected = (q[0,qt,row,col,0].float() @ key.T * scale - b).softmax(-1) @ value
        error = float((expected-output[0,qt,row,col,0].float()).abs().max())
        assert error < .002, f'FP8 full decoded route mismatch: {error}'
        samples.append(dict(query_time=qt,column=col,row=row,max_abs=error,padded_slots_empty=True))
    receipt = dict(valid=True, scope='synthetic H1 full720p pre-model gate, not video PSNR',
                   format='float8_e4m3fn', scale='per-vector absmax/448 floor 1e-8',
                   source_dtype='FP32 merged vectors', decoded_dtype='bfloat16',
                   actual_storage_verified=True, route_consumes_same_buffers=True,
                   relative_l2_vs_bf16=rel_l2, max_abs_vs_bf16=float(delta.abs().max()),
                   full_decoded_route_samples=samples, singleton_dtype='bfloat16',
                   kv_bytes=mk.numel()*2, scale_bytes=ks.numel()*8,
                   bf16_kv_bytes=mk.numel()*4, unchanged_bias_bytes=bias.numel()*4)
    # Capture objects retain the actual allocation solely during this gate.
    merge.args = route.args = None
    del q,k,v,seeds,baseline,output,delta,mk,mv,ks,vs,bias,adapter
    torch.cuda.empty_cache()
    return receipt
