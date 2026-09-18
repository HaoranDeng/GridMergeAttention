"""Compare warp SQ1 merge tiles to the retained native full H5 workspace."""
import hashlib
from pathlib import Path

def validate_merge(capacity, torch, q_cpu, k_cpu, v_cpu,source_dir=None):
    from torch.utils.cpp_extension import load
    from native_fisher.operator import NativeFisherAttention
    root = Path(__file__).resolve().parents[1]
    directory = Path(source_dir).resolve() if source_dir is not None else root/'native_fisher'
    paths = [directory/n for n in ('warp_mma.cuh','warp_merge.cuh','warp_merge_diagnostic.cu')]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    key = hashlib.sha256(''.join(hashes.values()).encode()).hexdigest()[:16]
    print('BUILD_MERGE', key, flush=True)
    ext = load(name='fisher_warp_merge_'+key, sources=[str(paths[-1])],
               extra_cuda_cflags=['-O3','-lineinfo','--ptxas-options=-v'],verbose=False)
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    q,k,v = [x.unsqueeze(0).contiguous().cuda() for x in (q_cpu,k_cpu,v_cpu)]
    op = NativeFisherAttention(capacity=capacity)
    seeds = op.select(q,k,128**-.5)
    seed_cpu = seeds.reshape(945,80,40,capacity).cpu().long()
    op.merge.pack(q,op.packed)
    op.merge.run(op.packed,k,v,seeds,op.mk,op.mv,op.bias,0,5,21,128**-.5,op.empty,0,1)
    torch.cuda.synchronize()
    packed_queries=[];keys=[];values=[];columns=[]
    tile_indices=[]
    # Consecutive key blocks cross frame/row boundaries; include first and last
    # key block, both column tails, middle frames, and five native head lanes.
    for frame,column,head,start in [(0,0,0,0),(0,1,1,0),(10,37,2,430),(20,78,3,881),(20,79,4,881)]:
        for kb in range(start,start+64):
            rows=seed_cpu[kb,column,head]
            sq=torch.zeros((16,128),dtype=torch.bfloat16)
            sq[:capacity]=q_cpu[frame,rows,column,head]
            packed_queries.append(sq);keys.append(k_cpu[kb//45,kb%45,:,head]);values.append(v_cpu[kb//45,kb%45,:,head]);columns.append(column)
            tile_indices.append((frame,column,head,kb))
    qa,ka,va=[torch.stack(x).contiguous().cuda() for x in (packed_queries,keys,values)]
    packed=torch.empty_like(qa);op.merge.pack(qa,packed)
    ci=torch.tensor(columns,device='cuda',dtype=torch.int32)
    # Gather all reference slices in one indexed operation.
    group=torch.tensor([(f*80+c)*5+h for f,c,h,b in tile_indices],device='cuda')
    key_index=torch.tensor([b for f,c,h,b in tile_indices],device='cuda')[:,None]*capacity+torch.arange(capacity,device='cuda')[None,:]
    expected=[x[group[:,None],key_index].clone() for x in (op.mk,op.mv,op.bias)]
    report={'capacity':capacity,'tiles':len(tile_indices),'sources':hashes,'native_merge_sha256':op.merge.source_sha256,
            'scope':'320 real tiles and the same tiles with zero-query ties against native H5 merge; not full-attention correctness or performance', 'cases':{}}
    def compare(actual,expected):
        comparisons={}
        for name,a,b in zip(('k','v','bias'),actual,expected):
            finite=torch.isfinite(a)&torch.isfinite(b)
            delta=(a[finite].float()-b[finite].float())
            storage_dtype=torch.int16 if a.dtype==torch.bfloat16 else torch.int32
            comparisons[name]={'bitwise_equal':torch.equal(a.view(storage_dtype),b.view(storage_dtype)),
                'finite_pattern_equal':torch.equal(torch.isfinite(a),torch.isfinite(b)),
                'mismatch_elements':int((a.view(storage_dtype)!=b.view(storage_dtype)).sum()),
                'max_abs_finite':float(delta.abs().max()) if delta.numel() else 0.,
                'relative_l2_finite':float((delta.square().sum()/b[finite].float().square().sum().clamp_min(1e-30)).sqrt())}
        return comparisons
    report['cases']['real']=compare(ext.run(packed,ka,va,ci,capacity,128**-.5),expected)
    # Zero Q makes all normalized seed scores tie and all but the first cluster
    # empty. Keep the same real K/V to exercise singleton removal and +inf bias.
    op.packed.zero_();packed.zero_()
    op.merge.run(op.packed,k,v,seeds,op.mk,op.mv,op.bias,0,5,21,128**-.5,op.empty,0,1)
    expected=[x[group[:,None],key_index].clone() for x in (op.mk,op.mv,op.bias)]
    report['cases']['zero_query_ties']=compare(ext.run(packed,ka,va,ci,capacity,128**-.5),expected)
    print('MERGE_GATE',report,flush=True)
    # This is intentionally an exact gate: a failure is evidence to investigate,
    # not permission to silently relax the arithmetic or replace the operator.
    report['exact_gate_passed']=all(x['bitwise_equal'] for case in report['cases'].values() for x in case.values())
    return report
