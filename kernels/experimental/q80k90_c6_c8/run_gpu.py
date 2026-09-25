"""After GPU authorization: invariants -> tune -> frozen numerical -> real40.

One process throughout, so the exclusive-GPU guard can attribute every CUDA
allocation. This file does not wait for a card or launch video generation.
"""
import argparse
import gc
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import statistics
import sys
import time


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--capacity',type=int,choices=(6,8),required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('/local2/denghaoran/GridMergeAttention/experiments/rf0_spatial_c8_8layers_20260919_v1'))
    p.add_argument('--repo',type=Path,default=Path('/local2/denghaoran/GridMergeAttention/axis_wh_c8_source_20260918'))
    p.add_argument('--checkpoint',type=Path,default=Path('/xuanwu-tank/east/zrl/MonarchRT/wan_models/Wan2.1-T2V-14B-StepDistill-CfgDistill/distill_model.pt'))
    p.add_argument('--phase',choices=('invariants','tune','full'),default='full')
    a=p.parse_args();a.root.mkdir(exist_ok=True,parents=True)
    os.environ['PATH']='/usr/local/cuda/bin:'+os.environ['PATH']
    os.environ.setdefault('TRITON_CACHE_DIR','/local2/denghaoran/GridMergeAttention/q80c6_c8_v2_triton')
    os.environ.setdefault('FLASHINFER_WORKSPACE_BASE',str(a.root/'flashinfer'))
    here=Path(__file__).parent
    sys.path.insert(0,str(a.repo))
    import generate  # Preserve the model environment's library loading order.
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from q80k90_kernel import Q80K90,_pack_inputs,native_merge
    import triton
    torch.set_grad_enabled(False);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    hashes={n:hashlib.sha256((here/n).read_bytes()).hexdigest()
            for n in ('q80k90_kernel.py','q80k90_merge.cu','run_gpu.py','compare_block_extensions_rf0.py',f'benchmark_q80k90c{a.capacity}.py',f'benchmark_q80_real40_c{a.capacity}.py')}
    write(a.root/'manifest.json',dict(capacity=a.capacity,target_ms=100 if a.capacity==8 else 120,
          gpu=torch.cuda.get_device_name(),layers=[0,5,10,15,20,25,30,35],source_sha256=hashes))
    def stage(name):
        write(a.root/'stage.json',dict(stage=name,time=time.time(),pid=os.getpid()))
        print(json.dumps(dict(stage=name)),flush=True)
    def make(config):return Q80K90(capacity=a.capacity,**config)
    base=dict(merge_backend='cuda',head_chunk=4,route_m=128,route_n=64,native_tiles=63,
              seed_tile=8,selector_precision='fp64',selector_groups=4,fast_math=False,
              direct_store=False,min_blocks=2,fused_layout=True)
    stage('invariants')
    torch.manual_seed(9045)
    small=[torch.randn(1,7200,5,128,device='cuda',dtype=torch.bfloat16)*.25 for _ in range(3)]
    core=make(dict(base,head_chunk=3))
    initial,seeds=core(*small,return_debug=True);seeds=seeds.clone()
    assert torch.isfinite(initial).all()
    changed=small[0].clone();changed[:,3600:]*=-2
    core(changed,small[1],small[2]);assert torch.equal(seeds,core.seeds)
    sorted_seeds=seeds.sort(-1).values
    assert (sorted_seeds[...,1:]>sorted_seeds[...,:-1]).all()
    anchors=torch.arange(80,device='cuda')%40*2
    assert torch.equal(seeds[...,0],anchors[None,None,:].expand_as(seeds[...,0]))
    assert torch.equal(seeds[...,1],(anchors+1)[None,None,:].expand_as(seeds[...,1]))
    # Fused layout is a lossless transformation, so compare it bit-for-bit
    # against torch permutations and the previously tested native Q pack.
    core(*small)
    expected_q=small[0].reshape(2,45,80,5,128).permute(3,0,1,2,4).contiguous()
    expected_k=small[1].reshape(2,45,40,2,5,128).permute(4,0,2,3,1,5).reshape(5,80,90,128).contiguous()
    expected_v=small[2].reshape(2,45,40,2,5,128).permute(4,0,2,3,1,5).reshape(5,80,90,128).contiguous()
    expected_packed=torch.empty_like(expected_q)
    core.native(2,5,3).pack(expected_q,expected_packed)
    assert all(torch.equal(x,y) for x,y in [(core.qp,expected_q),(core.kp,expected_k),(core.vp,expected_v),(core.packed,expected_packed)])
    plain=make(dict(base,head_chunk=3,fused_layout=False))
    assert torch.equal(initial,plain(*small))
    # Independent Triton arithmetic path catches CUDA tile/index mistakes.
    triton_core=make(dict(base,head_chunk=3,merge_backend='triton'))
    triton_out=triton_core(*small)
    implementation_error=float((initial.float()-triton_out.float()).norm()/triton_out.float().norm())
    assert implementation_error<.02,implementation_error
    modes=[dict(fast_math=False,direct_store=False,min_blocks=2),
           dict(fast_math=True,direct_store=False,min_blocks=2),
           dict(fast_math=True,direct_store=False,min_blocks=3),
           dict(fast_math=True,direct_store=True,min_blocks=3),
           dict(fast_math=True,direct_store=True,min_blocks=4)]
    checks=[]
    for mode in modes:
        candidate=make(dict(base,head_chunk=3,**mode))
        output=candidate(*small)
        error=float((output.float()-initial.float()).norm()/initial.float().norm())
        assert torch.isfinite(output).all() and error<.02,(mode,error)
        ones=candidate(torch.zeros_like(small[0]),small[1],torch.ones_like(small[2]))
        assert torch.allclose(ones,torch.ones_like(ones),atol=.01,rtol=0)
        # Last head chunk has two heads; ignore unused workspace capacity.
        assert torch.isfinite(candidate.bias[:2,...,0]).all()
        assert torch.isneginf(candidate.bias[:2,...,1:]).all()
        assert not torch.count_nonzero(candidate.mk[:2,...,1:,:])
        assert not torch.count_nonzero(candidate.mv[:2,...,1:,:])
        checks.append(dict(config=mode,relative_l2_vs_conservative=error))
        del candidate,output,ones
    write(a.root/'invariants.json',dict(passed=True,rf0_reused=True,exact_capacity=a.capacity,
          distinct_seeds=True,two_intersections=True,fused_layout_bitwise_equal=True,
          output_layout_bitwise_equal=True,empty_cluster_zero_kv_negative_infinity_bias=True,
          constant_value_preserved=True,partial_head_chunk_covered=True,
          relative_l2_vs_triton=implementation_error,modes=checks))
    del small,core,plain,triton_core,initial,seeds,changed,sorted_seeds,expected_q,expected_k,expected_v,expected_packed,triton_out
    gc.collect();torch.cuda.empty_cache()
    if a.phase=='invariants':stage('complete');return

    stage('shape_tuning')
    payload=torch.load(a.data/'captures/p000_s0_l00.pt',map_location='cpu',weights_only=False)
    full=[payload[n].to('cuda').repeat(1,1,10,1).contiguous() for n in ('q','k','v')]
    indices=torch.tensor([(t*45+h)*80+w for t,h,w in itertools.product((0,10,20),(0,22,44),(0,40,79))],device='cuda')
    configs=[dict(base,fused_layout=False),base]
    # Orthogonal, attributable changes first; then test their combination.
    configs += [dict(base,**mode) for mode in modes[1:]]
    configs += [dict(base,selector_precision='fp32',selector_groups=g) for g in (1,2,4,8)]
    configs += [dict(base,fast_math=True,direct_store=direct,min_blocks=blocks,
                    selector_precision='fp32',route_n=n)
                for direct,blocks,n in [(False,2,64),(False,3,128),(True,3,64),(True,4,128)]]
    def dense():return torch.nn.functional.scaled_dot_product_attention(*(x.transpose(1,2) for x in full)).transpose(1,2)
    def event(fn):
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        s.record();out=fn();e.record();e.synchronize();return s.elapsed_time(e),out
    report=dict(complete=False,input_kind='four captured real heads repeated ten times; preliminary shape timing',
                source_sha256=hashes,rows=[])
    reference=reference_seeds=None
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        dense();torch.cuda.synchronize()
        for config in configs:
            core=make(config)
            try:
                core(*full);out=core(*full)
                assert torch.isfinite(out).all()
                sample=out.index_select(1,indices).float();del out
                if reference is None:reference=sample.clone();reference_seeds=core.seeds.clone()
                error=float((sample-reference).norm()/reference.norm())
                seed_change=float((core.seeds!=reference_seeds).float().mean())
                assert error<.02 and seed_change<.005,(error,seed_change)
                times=dict(method=[],dense=[])
                for repeat in range(7):
                    funcs=[('method',lambda:core(*full)),('dense',dense)]
                    if repeat%2:funcs.reverse()
                    for name,fn in funcs:
                        ms,out=event(fn);times[name].append(ms);del out
                row=dict(valid=True,config=config,samples_ms=times,relative_l2_vs_conservative=error,
                         seed_changed_fraction=seed_change,mean_attention_ms=statistics.mean(times['method']),
                         dense_mean_attention_ms=statistics.mean(times['dense']))
                row['speedup_vs_sdpa']=row['dense_mean_attention_ms']/row['mean_attention_ms']
            except AssertionError as exc:
                # Keep numerical/tuning failures visible; never promote them.
                # CUDA/compilation errors propagate and terminate the worker.
                row=dict(valid=False,config=config,error=repr(exc))
            report['rows'].append(row);write(a.root/'tune.json',report);print(json.dumps(row),flush=True)
            del core;gc.collect();torch.cuda.empty_cache()
    valid=[x for x in report['rows'] if x['valid']]
    assert valid,'No candidate passed the tuning numerical screen'
    selected=min(valid,key=lambda x:x['mean_attention_ms'])
    report.update(complete=True,best=selected)
    write(a.root/'tune.json',report)
    config_path=a.root/'selected_config.json';write(config_path,selected['config'])
    del payload,full,reference,reference_seeds;gc.collect();torch.cuda.empty_cache()
    if a.phase=='tune':stage('complete');return

    stage('frozen_eight_layer_numerical_96_captures')
    numerical=importlib.import_module(f'benchmark_q80k90c{a.capacity}')
    sys.argv=['numerical','--root',str(a.root/'numerical'),'--data',str(a.data),
              '--reference',str(a.data/'compare_spatial_rf0_eight_layers_20260919.py'),
              '--operator',str(here/'compare_block_extensions_rf0.py'),'--phase','validate','--config',str(config_path)]
    numerical.main()
    numerical_result=json.loads((a.root/'numerical/validate.json').read_text());s=numerical_result['numerical_summary']
    assert numerical_result['complete'] and s['cases']==384
    assert numerical_result['layers']==[0,5,10,15,20,25,30,35]
    approved=(s['max_relative_l2_vs_operator']<.02 and s['mean_seed_changed_fraction']<.005
              and s['mean_l2_to_dense']<=s['reference_mean_l2_to_dense']+.0005)
    write(a.root/'numerical_gate.json',dict(approved=approved,summary=s,
          invariants=numerical_result['invariants'],thresholds_owner='implementation review, not user universal tolerance',
          engineering_thresholds=dict(max_relative_l2_vs_operator=.02,max_mean_seed_changed_fraction=.005,
                                      max_absolute_mean_l2_regression=.0005)))
    assert approved,'Frozen numerical validation needs review; real40 timing not started'
    gc.collect();torch.cuda.empty_cache()
    stage('real40_eight_layer_speed_gate')
    speed=importlib.import_module(f'benchmark_q80_real40_c{a.capacity}')
    sys.argv=['real40','--repo',str(a.repo),'--checkpoint',str(a.checkpoint),'--root',str(a.root/'real40'),
              '--config',str(config_path),'--prompts','3']
    speed.main()
    real=json.loads((a.root/'real40/real40.json').read_text())
    target=100 if a.capacity==8 else 120
    write(a.root/'result.json',dict(complete=True,capacity=a.capacity,target_ms=target,
          mean_attention_ms=real['mean_attention_ms'],dense_mean_attention_ms=real['dense_mean_attention_ms'],
          speedup_vs_sdpa=real['speedup_vs_sdpa'],target_met=real['mean_attention_ms']<=target,
          numerical_passed=True,faster_than_dense_all_cases=real['passed'],config=selected['config'],
          protocol='3 prompts x 4 frozen timesteps x 8 fixed layers x 40 real heads',
          video_quality='pending; no videos generated',source_sha256=hashes))
    stage('complete')


if __name__=='__main__':main()
