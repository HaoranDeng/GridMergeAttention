"""Paired MI300 selector and complete C4 attention validation on saved QKV."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--selector-only',action='store_true')
    p.add_argument('--pairs',type=int,default=8)
    args=p.parse_args()
    import torch
    if torch.cuda.device_count()!=1:
        raise RuntimeError('Require exactly one Slurm-isolated visible MI300 GPU')
    from native_hip.selector import HipFisherSelector
    from attention.affine_hellinger_seed_triton import affine_hellinger_seed_rows_triton
    from attention.singleton_c4_wan21_14b_720p_triton import _hard_c4_720p_temporal_shared_global_mass_attention_triton_blocks as merge_route
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    result={'scope':'real saved QKV replay; no generated-video quality claim','job_id':os.environ.get('SLURM_JOB_ID'),
            'torch':torch.__version__,'hip':torch.version.hip,'capture':str(args.capture),
            'gpu':str(torch.cuda.get_device_properties(0)),'selector_sweep':[]}
    digest=hashlib.sha256()
    with args.capture.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    result['capture_sha256']=digest.hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result,indent=2))
    def timed(fn,n=5):
        fn();fn();torch.cuda.synchronize()
        samples=[]
        for _ in range(n):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();fn();end.record();end.synchronize();samples.append(start.elapsed_time(end))
        return {'mean_ms':statistics.fmean(samples),'samples_ms':samples}
    def compare(a,b):
        d=a.float()-b.float()
        return {'bitwise_equal':torch.equal(a,b),'finite':bool(torch.isfinite(a).all()),
                'relative_l2':float((d.square().sum()/b.float().square().sum().clamp_min(1.e-30)).sqrt()),'max_abs':float(d.abs().max())}
    print('LOAD',args.capture,flush=True)
    payload=torch.load(args.capture,map_location='cpu',weights_only=True)
    q,k,v=[payload[x].reshape(1,21,45,80,40,128).cuda() for x in ('q','k','v')]
    del payload
    print('BUILD_HIP',flush=True)
    op=HipFisherSelector()
    result['hip_source_sha256']=op.source_sha256
    def reference_seeds():
        return affine_hellinger_seed_rows_triton(q[:,:1],k,feature_mode='fisher_geometric',block_j=4)
    print('REFERENCE_SELECTOR',flush=True)
    ref=reference_seeds()
    result['reference_selector']=timed(reference_seeds)
    print('REFERENCE_SELECTOR_MS',result['reference_selector'],flush=True)
    op(q,k)
    # Every layout uses exactly the same materialized BF16-QK/FP32 logits.
    control=None
    for wave,probes in [(32,4),(32,8),(32,16),(64,2),(64,4),(64,8),(64,16)]:
        for threads in (128,256):
            op.wave,op.probes,op.threads=wave,probes,threads
            seeds=op.gs()
            if control is None:control=seeds.clone()
            row={'wave':wave,'probes':probes,'threads':threads,
                 'mismatch_vs_triton':int((seeds!=ref).sum()),'mismatch_vs_hip_control':int((seeds!=control).sum()),
                 'gs':timed(op.gs)}
            result['selector_sweep'].append(row);save();print('SWEEP',row,flush=True)
    valid=[r for r in result['selector_sweep'] if r['mismatch_vs_hip_control']==0 and r['mismatch_vs_triton']==0]
    if not valid:
        result['valid']=False;save();raise RuntimeError('No exact selector candidate; inspect seeds before full timing')
    best=min(valid,key=lambda r:r['gs']['mean_ms'])
    op.wave,op.probes,op.threads=best['wave'],best['probes'],best['threads']
    result['selected']=best
    result['hip_selector']=timed(lambda:op(q,k))
    # Tie gate checks every output group and every implementation.
    op.logits.zero_();tie_ref=None
    for wave,probes in [(32,4),(32,8),(32,16),(64,2),(64,4),(64,8),(64,16)]:
        op.wave,op.probes=wave,probes
        z=op.gs()
        if tie_ref is None:
            table=op.candidates.cpu();anchors=op.anchors.cpu()
            expected=torch.stack([table[r,torch.tensor([int(anchors[r])]+[j for j in range(32) if j!=int(anchors[r])][:3])] for r in range(45)]).cuda().to(torch.uint8)
            tie_ref=expected[None,:,None,None,:].expand(21,45,80,40,4).reshape_as(z)
        assert torch.equal(z,tie_ref),(wave,probes,'tie mismatch')
    result['zero_ties_exact']=True
    op.wave,op.probes,op.threads=best['wave'],best['probes'],best['threads']
    save();print('SELECTED',result['selected'],'HIP_SELECTOR',result['hip_selector'],flush=True)
    if args.selector_only:
        result['valid']=True;save();return
    out=torch.empty_like(q)
    def full(selector):
        seeds=selector()
        for head in range(0,40,5):
            merge_route(q,k,v,seeds,top_r=1,head_start=head,head_count=5,query_column_selector=True,
                seed_rows_native_layout=True,query_column_major=True,split_kv_accum=True,block_j=4,
                merge_loop_stages=1,route_loop_stages=1,merge_disable_licm=False,output=out)
        return out
    print('FULL_CALL_COMPILE',flush=True)
    b=full(reference_seeds).clone();a=full(lambda:op(q,k))
    result['full40_correctness']=compare(a,b);save();print('FULL40_CHECK',result['full40_correctness'],flush=True)
    assert result['full40_correctness']['bitwise_equal']
    del b
    full(reference_seeds);full(lambda:op(q,k));torch.cuda.synchronize()
    samples={'triton':[],'hip':[]}
    for pair in range(args.pairs):
        for name in (('triton','hip') if pair%2==0 else ('hip','triton')):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();full(reference_seeds if name=='triton' else lambda:op(q,k));end.record();end.synchronize()
            samples[name].append(start.elapsed_time(end))
        print('PAIR',pair,{k:v[-1] for k,v in samples.items()},flush=True)
    result['complete']={name:{'mean_ms':statistics.fmean(s),'samples_ms':s} for name,s in samples.items()}
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel,SDPBackend
    qf,kf,vf=[x.reshape(1,75600,40,128).transpose(1,2) for x in (q,k,v)]
    def sdpa():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):return F.scaled_dot_product_attention(qf,kf,vf)
    result['sdpa']=timed(sdpa,n=args.pairs)
    result['speedup_vs_sdpa']=result['sdpa']['mean_ms']/result['complete']['hip']['mean_ms']
    result['speedup_vs_triton']=result['complete']['triton']['mean_ms']/result['complete']['hip']['mean_ms']
    result['valid']=True;save();print('FINAL',json.dumps(result),flush=True)


if __name__=='__main__':main()
