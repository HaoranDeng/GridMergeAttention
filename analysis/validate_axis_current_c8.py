"""Numerical/semantic checks for directed W80/H45 exact C8; no video inspection."""
import argparse
import gc
import itertools
import json
from pathlib import Path

import torch
from attention.axis_current_c8_cuda import AxisCurrentC8, merged_attention, _workspace
from attention.axis_current_c8_reference import line_attention, select_fisher, evaluate
from analysis.quick_axis_c4_qkv import candidate_table, SAMPLES, save
from analysis.axis_c8_speed_gate import source_hashes, environment

CAPTURES = Path('/local2/denghaoran/GridMergeAttention/experiments/full40_bf16_dense_capture_20260905bb/captures')


def error(a, b):
    delta = a.float()-b.float()
    return {'relative_l2':float(delta.norm()/b.float().norm().clamp_min(1.e-30)),
            'max_absolute':float(delta.abs().max()), 'rms':float(delta.square().mean().sqrt())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--real-cases',type=int,default=6)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(719);torch.cuda.manual_seed_all(719)
    # These are diagnostic thresholds chosen for this implementation, not user tolerances.
    report={'passed':False,'axes':'wh','capacity':8,'implementation':'r2',
            'source_sha256':source_hashes(),'environment':environment(),
            'diagnostic_limits':{'independent_reference_relative_l2':0.012,
                                 'r2_vs_slow_reference_relative_l2':0.003},'cases':[]}
    try:
        r,l,o=80,45,2
        k,v=[torch.randn((1,o,r,l,1,128),device='cuda',dtype=torch.bfloat16) for _ in range(2)]
        q0=torch.randn_like(k)
        table,anchors=candidate_table(r)
        candidates=torch.tensor(table,device='cuda').repeat(o,1)
        anchors=torch.tensor(anchors,device='cuda').repeat(o)
        probes=torch.arange(16,device='cuda')*l//16
        for kind in ('random','zero_query','constant_value'):
            q=torch.zeros_like(q0) if kind=='zero_query' else q0
            value=torch.full_like(v,2.5) if kind=='constant_value' else v
            out,mk,mv,bias,seeds=merged_attention(q,k,value,audit=True)
            assert torch.isfinite(out).all()
            assert seeds.shape[-1]==8 and mk.shape[-2]==mv.shape[-2]==8
            empty=torch.isposinf(bias)
            assert not bool(torch.isnan(bias).any())
            assert bool((mk[empty]==0).all()) and bool((mv[empty]==0).all())
            row={'kind':kind,'empty_slots':int(empty.sum()),'slots':bias.numel(),'independent_errors':[]}
            if kind=='zero_query':
                assert int(empty.sum())==bias.numel()*7//8
            if kind=='constant_value':
                row['constant_preservation']=error(out,torch.full_like(out,2.5))
                assert row['constant_preservation']['max_absolute']<0.04
            kb,vb=[x[0,...,0,:].float().reshape(o*r,l,128) for x in (k,value)]
            # Include the final partial 8-column tile and two distinct query frames.
            for frame,column in ((0,0),(0,l-1),(1,l//2),(1,l-1)):
                query=q[0,frame,:,column,0].float()
                reference,_=line_attention(query,query,kb,vb,candidates,anchors,probes,list(range(r)))
                err=error(out[0,frame,:,column,0],reference)
                refseeds=select_fisher(query,kb,candidates,anchors,probes,128**-.5)
                err['seed_mismatched_groups']=int((seeds[0,frame,column,0].reshape(-1,8)!=refseeds).any(-1).sum())
                row['independent_errors'].append(err)
                assert err['relative_l2']<report['diagnostic_limits']['independent_reference_relative_l2'],err
            exact=merged_attention(q,k,value,implementation='slow_reference')
            row['r2_vs_slow_reference']=error(out,exact)
            assert row['r2_vs_slow_reference']['relative_l2']<0.003,row
            report['cases'].append(row);save(a.output,report)
            print(json.dumps(row),flush=True)
        del k,v,q,q0,value,out,mk,mv,bias,seeds,exact,kb,vb,reference,query,refseeds
        _workspace.cache_clear();gc.collect();torch.cuda.empty_cache()
        files=[CAPTURES/f'prompt{prompt:03d}_{suffix}.pt' for prompt in range(3)
               for suffix in ('call000_layer0','call033_layer33')]
        report['real_qkv']=[]
        for index,path in enumerate(files[:a.real_cases]):
            payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            head=(0,10,20,30,0,30)[index]
            tensors=[payload[name][0,:,head].to('cuda').reshape(1,1,75600,128) for name in ('q','k','v')]
            core=AxisCurrentC8()
            out=core(*tensors)
            exact=AxisCurrentC8(implementation='slow_reference')(*tensors)
            pair=error(out,exact)
            physical=[x[0,0].reshape(21,45,80,128).float() for x in tensors]
            reference,_=evaluate(*physical,2,1,'current')
            sampled=torch.stack([out[0,0].reshape(21,45,80,128)[c] for c in itertools.product(*SAMPLES)])
            independent=error(sampled,reference)
            row={'capture':str(path),'head':head,'r2_vs_slow_reference':pair,'independent_reference':independent}
            assert torch.isfinite(out).all() and pair['relative_l2']<0.003,row
            assert independent['relative_l2']<0.012,row
            report['real_qkv'].append(row);save(a.output,report);print(json.dumps(row),flush=True)
            del payload,tensors,core,out,exact,physical,reference,sampled
            _workspace.cache_clear();gc.collect();torch.cuda.empty_cache()
        assert len(report['real_qkv'])==6,'Full gate requires all six real prompt/layer cases'
        report['passed']=True
    finally:
        save(a.output,report)
    print(json.dumps({'passed':report['passed'],'output':str(a.output)}),flush=True)


if __name__=='__main__':main()
