"""Numerical and complete-call timing checks, using the fixed eight-layer QKV."""
import argparse,hashlib,importlib.util,itertools,json,os,statistics,time
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from q80k90c10_kernel import Q80K90C10,candidate_table

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
def save(path,r):path.write_text(json.dumps(r,indent=2)+'\n')
def event(fn):
    s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    s.record();out=fn();e.record();e.synchronize();return s.elapsed_time(e),out
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--operator',type=Path,required=True)
    p.add_argument('--phase',choices=('smoke','tune','tune2','validate'),default='smoke');p.add_argument('--config',type=Path)
    a=p.parse_args();a.root.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    reference=load('frozen_ref',a.reference);mathref=load('frozen_op',a.operator)
    r=dict(complete=False,phase=a.phase,layers=[0,5,10,15,20,25,30,35],gpu=torch.cuda.get_device_name(),torch=torch.__version__,triton=__import__('triton').__version__,source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('q80k90_kernel.py','q80k90_merge.cu') if Path(__file__).with_name(n).exists()},numerical=[],timing=[])
    config=json.loads(a.config.read_text()) if a.config else {}
    if config.get('pipeline',False):
        from check_pipeline_gpu import check
        r['pipeline_checks']=check(10,config,a.data/'captures/p000_s0_l00.pt')
        save(a.root/f'{a.phase}.json',r)
    if a.phase=='validate':
        from check_selected_gpu import check as check_selected
        r['selected_configuration_checks']=check_selected(10,config)
        save(a.root/f'{a.phase}.json',r)
    core=Q80K90C10(**config)
    # The candidate pool must match the frozen Q80K90C10 scan operator.
    table=mathref.chunk_metadata(21,80,45,1,2,0,reference,'cpu',32)[0]
    assert torch.equal(torch.tensor(candidate_table()),table[:40])
    coords=list(itertools.product((0,10,20),(0,22,44),(0,40,79)))
    indices=torch.tensor([(t*45+h)*80+w for t,h,w in coords],device='cuda')
    captures=[a.data/'captures'/f'p{p:03d}_s{s}_l{l:02d}.pt' for p in range(3) for s in range(4) for l in r['layers']]
    if a.phase=='smoke':captures=captures[:1]
    if a.phase.startswith('tune'):captures=captures[:1]
    try:
        for ci,path in enumerate(captures):
            payload=torch.load(path,map_location='cpu',weights_only=False)
            tensors=[payload[x].to('cuda').contiguous() for x in ('q','k','v')]
            assert tensors[0].shape==(1,75600,4,128),tensors[0].shape
            if not a.phase.startswith('tune'):
                value,seeds=core(*tensors,return_debug=True)
                assert torch.isfinite(value).all()
                for head in range(4):
                    q,k,v=[x[0,:,head].reshape(21,45,80,128).float() for x in tensors]
                    expected,banks,_,_=mathref.evaluate_chunks(q,k,v,'wh',1,2,10,reference)
                    actual=value.index_select(1,indices)[0,:,head].float()
                    err=float((actual-expected).norm()/expected.norm())
                    bank=torch.cat([seeds[head,int(key)].flatten().cpu() for key in sorted(banks,key=int)])
                    old=torch.cat([banks[key].flatten() for key in sorted(banks,key=int)])
                    # Match the QKC scan target exactly: explicit FP32 sampled
                    # dense attention, not the saved BF16 FlashAttention output.
                    queries=q.reshape(-1,128).index_select(0,indices)
                    dense=(queries@k.reshape(-1,128).T*128**-.5).softmax(-1)@v.reshape(-1,128)
                    row=dict(capture=path.name,head=[0,10,20,30][head],relative_l2_vs_operator=err,max_abs_vs_operator=float((actual-expected).abs().max()),seed_changed_fraction=float((bank!=old).float().mean()),mean_relative_l2_to_dense=float((actual-dense).norm()/dense.norm()),reference_relative_l2_to_dense=float((expected-dense).norm()/dense.norm()),squared_error=float((actual-dense).square().sum()),reference_squared=float(dense.square().sum()))
                    r['numerical'].append(row);print(json.dumps(row),flush=True)
                del value,seeds,q,k,v,expected,actual,bank,old,dense,queries
                if ci==0 and a.phase=='validate':
                    initial=core.seeds.clone()
                    changed=tensors[0].clone();changed[:,3600:]*=-2
                    core(changed,tensors[1],tensors[2]);assert torch.equal(core.seeds,initial),'RF0 indexes changed with later frames'
                    sorted_seeds=core.seeds.sort(-1).values;assert (sorted_seeds[...,1:]>sorted_seeds[...,:-1]).all()
                    anchors=(torch.arange(840,device='cuda')%40)*2
                    assert torch.equal(core.seeds[...,0],anchors[None,None,:].expand_as(core.seeds[...,0]))
                    assert torch.equal(core.seeds[...,1],(anchors+1)[None,None,:].expand_as(core.seeds[...,1]))
                    ones=core(torch.zeros_like(tensors[0]),tensors[1],torch.ones_like(tensors[2]))
                    assert torch.allclose(ones,torch.ones_like(ones),atol=.01,rtol=0)
                    assert torch.isfinite(core.bias[...,0]).all() and torch.isneginf(core.bias[...,1:]).all()
                    r['invariants']=dict(rf0_reused=True,exact_distinct_capacity=10,two_intersections_forced=True,empty_cluster_bias_negative_infinity=True,constant_value_preserved=True)
                    del initial,changed,sorted_seeds,ones
            if a.phase in ('smoke','tune','tune2'):
                # Shape test: repeat four captured real heads ten times. Explicitly
                # not a claim of 40 independent captured heads or all-layer speed.
                full=[x.repeat(1,1,10,1).contiguous() for x in tensors]
                configs=[config] if a.phase=='smoke' else [dict(head_chunk=4,merge_groups=g,merge_loop=loop,route_m=m,route_n=n,merge_warps=4,route_warps=4) for g,loop,m,n in [(4,1,64,64),(4,4,64,64),(4,15,64,64),(4,1,64,128),(4,1,128,64),(4,1,128,128),(2,1,64,64),(8,1,64,64)]]
                if a.phase=='tune2':
                    configs=[dict(head_chunk=4,merge_groups=g,merge_loop=loop,route_m=m,route_n=n,merge_warps=mw,route_warps=rw) for g,loop,m,n,mw,rw in [(4,45,128,64,4,4),(4,105,128,64,4,4),(4,15,128,64,8,4),(4,15,64,64,4,8),(4,15,128,32,4,4),(4,45,128,64,4,8)]]
                def dense():return torch.nn.functional.scaled_dot_product_attention(*(x.transpose(1,2) for x in full)).transpose(1,2)
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    dense();torch.cuda.synchronize()
                    for cfg in configs:
                        del core;torch.cuda.empty_cache();core=Q80K90C10(**cfg)
                        core(*full);torch.cuda.synchronize()
                        ts={'kernel':[],'dense':[]}
                        for repeat in range(3):
                            for name,fn in ([('dense',dense),('kernel',lambda:core(*full))] if repeat%2==0 else [('kernel',lambda:core(*full)),('dense',dense)]):
                                ms,out=event(fn);ts[name].append(ms);assert torch.isfinite(out).all();del out
                        row=dict(config=cfg,shape=[1,75600,40,128],input_kind='four real captured heads repeated 10x; preliminary shape timing',samples_ms=ts,mean_attention_ms=statistics.mean(ts['kernel']),dense_mean_attention_ms=statistics.mean(ts['dense']),speedup_vs_sdpa=statistics.mean(ts['dense'])/statistics.mean(ts['kernel']))
                        r['timing'].append(row);save(a.root/f'{a.phase}.json',r);print(json.dumps(row),flush=True)
                if a.phase=='smoke':
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                        core(*full);torch.cuda.synchronize()
                    r['profile']=[dict(name=e.key,ms=e.device_time_total/1000,count=e.count) for e in prof.key_averages() if e.device_time_total>100]
                del full
            del tensors,payload;save(a.root/f'{a.phase}.json',r)
            print(json.dumps(dict(captures=ci+1,total=len(captures))),flush=True)
        r['complete']=True
        r['dense_target']='explicit FP32 sampled dense attention, same as QKC scan'
        if r['numerical']:
            r['numerical_summary']=dict(cases=len(r['numerical']),mean_relative_l2_vs_operator=statistics.mean(x['relative_l2_vs_operator'] for x in r['numerical']),max_relative_l2_vs_operator=max(x['relative_l2_vs_operator'] for x in r['numerical']),mean_l2_to_dense=statistics.mean(x['mean_relative_l2_to_dense'] for x in r['numerical']),reference_mean_l2_to_dense=statistics.mean(x['reference_relative_l2_to_dense'] for x in r['numerical']),mean_seed_changed_fraction=statistics.mean(x['seed_changed_fraction'] for x in r['numerical']))
            r['numerical_summary']['pooled_l2_to_dense']=(sum(x['squared_error'] for x in r['numerical'])/sum(x['reference_squared'] for x in r['numerical']))**.5
    finally:save(a.root/f'{a.phase}.json',r)
if __name__=='__main__':main()
