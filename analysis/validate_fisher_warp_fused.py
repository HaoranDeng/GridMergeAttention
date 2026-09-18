"""Validate fused compressed partitions against the materialized route."""
import hashlib, statistics
from pathlib import Path

def validate_fused(capacity,op,q,k,v,seeds,route,monitor_errors,profile=False,source_path=None,source_dir=None):
    root=Path(__file__).resolve().parents[1]
    directory=Path(source_dir).resolve() if source_dir is not None else root/'native_fisher'
    paths=[directory/n for n in ('warp_mma.cuh','warp_merge.cuh','warp_route.cuh','warp_fused.cu')]
    if source_path is not None:paths[-1]=Path(source_path).resolve()
    # Quoted CUDA includes resolve beside the .cu file. Hashing headers from
    # another directory would falsely attribute the compiled implementation.
    if paths[-1].parent.resolve()!=directory.resolve():
        raise ValueError('Fused CUDA source must reside in the selected source directory so recorded headers match quoted includes')
    import torch
    from torch.utils.cpp_extension import load
    hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
    key=hashlib.sha256(''.join(hashes.values()).encode()).hexdigest()[:16]
    print('BUILD_FUSED',capacity,key,flush=True)
    ext=load(name='fisher_warp_fused_'+key,sources=[str(paths[-1])],
             extra_cuda_cflags=['-O3','-lineinfo','--ptxas-options=-v'],verbose=False)
    assert hashes=={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
    report={'scope':'Selected fused compressed partitions against materialized native merge plus validated warp route; no SQ1 contribution or complete attention yet',
            'sources':hashes,'resources':ext.resources(capacity),'checks':[]}
    print('FUSED_RESOURCES',report['resources'],flush=True)
    partial=torch.empty((8400,16,48,128),device='cuda',dtype=torch.float32)
    maxima=torch.empty((8400,16,48),device='cuda');den=torch.empty_like(maxima)
    for group,partition in ((0,0),(2092,7),(4199,14)):
        ah=group%5;column_pair=(group//5)%40;frame=group//200
        ext.run(op.packed,k,v,seeds,partial,maxima,den,capacity,0,group,1,partition,1,128**-.5)
        begin=partition*64*capacity;end=min((partition+1)*64*capacity,945*capacity)
        groups=torch.tensor([(frame*80+column_pair*2+c)*5+ah for c in (0,1)],device='cuda')
        mk=op.mk[groups,begin:end].contiguous();mv=op.mv[groups,begin:end].contiguous();mb=op.bias[groups,begin:end].contiguous()
        for row_start in (0,16,32):
            rows=min(16,45-row_start)
            qt=torch.zeros((2,16,128),device='cuda',dtype=torch.bfloat16)
            for col in (0,1):qt[col,:rows]=q[0,frame,row_start:row_start+rows,column_pair*2+col,ah]
            packed=qt.reshape(2,16,4,4,4,2).transpose(-3,-2).contiguous().reshape_as(qt)
            expected,rm,rd=route.run(packed,mk,mv,mb,rows,128**-.5)
            a=partial[groups,partition,row_start:row_start+16]
            d=den[groups,partition,row_start:row_start+16]
            m=maxima[groups,partition,row_start:row_start+16]
            actual=torch.where(d[:,:,None]>0,a/d[:,:,None],0.)
            diff=actual-expected
            record={'group':group,'partition':partition,'row_start':row_start,'valid_rows':rows,
                    'relative_l2':float((diff.square().sum()/expected.square().sum().clamp_min(1e-30)).sqrt()),
                    'max_abs':float(diff.abs().max()),'finite':bool(torch.isfinite(actual).all()),
                    'storage_equal':torch.equal(actual.view(torch.int32),expected.view(torch.int32)),
                    'max_pattern_equal':torch.equal(torch.isfinite(m[:,:rows]),torch.isfinite(rm[:,:rows])),
                    'zero_den_pattern_equal':torch.equal(d[:,:rows]==0,rd[:,:rows]==0)}
            report['checks'].append(record);print('FUSED_CHECK',record,flush=True)
            assert record['finite'] and record['relative_l2']<2e-5 and record['max_pattern_equal'] and record['zero_den_pattern_equal'],record
    if profile:
        assert not monitor_errors,monitor_errors
        print('FUSED_H5_PROFILE',capacity,flush=True)
        ext.run(op.packed,k,v,seeds,partial,maxima,den,capacity,0,0,4200,0,15,128**-.5)
        torch.cuda.synchronize();values=[]
        for _ in range(3):
            assert not monitor_errors,monitor_errors
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();ext.run(op.packed,k,v,seeds,partial,maxima,den,capacity,0,0,4200,0,15,128**-.5);b.record();b.synchronize()
            values.append(a.elapsed_time(b))
        report['h5_kernel_only']={'mean_ms':statistics.fmean(values),'samples_ms':values,
            'scope':'All H5 compressed merge plus route partitions and FP32 partial stores; excludes selector, Q packing, SQ1, partial combination, other35 heads'}
        print('FUSED_H5_TIMING',report['h5_kernel_only'],flush=True)
    assert not monitor_errors,monitor_errors
    report['valid']=True
    return report
