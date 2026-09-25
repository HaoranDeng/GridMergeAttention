"""Native121 validation/generation using the frozen historical T2V pipeline."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types
from common import ROOT,read,save,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--gate',action='store_true');p.add_argument('--slot',required=True);a=p.parse_args()
    plan=read(ROOT/'experiment_plan.json');snapshot=Path(plan['snapshot']);control=ROOT/a.slot
    for rel,h in plan['source_sha256'].items():assert sha(snapshot/rel)==h,rel
    for path,h in plan['svg_sha256'].items():assert sha(path)==h,path
    out=ROOT/('gate_probe' if a.gate else 'video20');out.mkdir(exist_ok=True)
    subprocess.run([plan['python'],str(snapshot/'analysis/verify_prompt_contract.py'),'--prompt_path',plan['prompt_path'],
        '--num_videos','20','--seed','0','--num_samples','1','--manifest_out',str(out/'prompt_manifest.json')],check=True,stdout=subprocess.DEVNULL)
    assert read(out/'prompt_manifest.json')['prompts']==read(ROOT/'video20/prompt_manifest.json')['prompts']
    sys.path.insert(0,str(snapshot));os.chdir(snapshot)
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    import attention.token_merging as tm
    torch.set_grad_enabled(False);torch.set_num_threads(4)
    native=tm._causal_one_token_attention;count=0
    gate=dict(complete=False,passed=False,rows=[],gpu=read(control/'preflight.json'),scope=plan['gate'])
    gate_path=control/'gate.json'
    class GateComplete(Exception):pass
    def operator(q,k,v,**kw):
        nonlocal count
        assert tuple(q.shape)==(1,75600,40,128) and (kw['h'],kw['w'])==(45,80)
        layer=count%40;branch=count//40;count+=1
        args=dict(f_tied=1,h_reduce=2,w_reduce=1,h=45,w=80,sm_scale=kw['sm_scale'],q_init='ith',
                  query_outer_chunk=kw['query_outer_chunk'],backend='monarch_rt',head_chunk=kw['head_chunk'])
        assert args['query_outer_chunk']==21 and args['head_chunk']==1
        def method():return native(q,k,v,**args)
        if not a.gate:return method()
        def dense():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)
        dense_out=dense()
        if layer in plan['gate']['layers']:
            result=method();assert bool(torch.isfinite(result).all())
            old=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
            try:
                ref=native(q[:,:,:1].float().contiguous(),k[:,:,:1].float().contiguous(),v[:,:,:1].float().contiguous(),
                    **dict(args,backend='sdpa',query_outer_chunk=1))
            finally:torch.backends.cuda.matmul.allow_tf32=old
            rel=float((result[:,:,:1].float()-ref).norm()/ref.norm());del result,ref
            row=dict(layer=layer,cfg_branch=branch,relative_l2=rel,method_ms=[],sdpa_ms=[])
            gate['rows'].append(row);save(gate_path,gate)
            assert rel<plan['gate']['engineering_relative_l2_limit'],row
            for repeat in range(3):
                for name,fn in [('method_ms',method),('sdpa_ms',dense)][::1 if repeat%2==0 else -1]:
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();value=fn();end.record();end.synchronize();row[name].append(start.elapsed_time(end));del value
            save(gate_path,gate);print(f'121 gate L{layer} CFG{branch}: FP32 relative L2={rel:.6f}',flush=True)
        if count==80:raise GateComplete()
        return dense_out
    tm._hard_c4_attention=operator
    spec=importlib.util.spec_from_file_location('frozen_t2v_benchmark',snapshot/'analysis/benchmark_wan14b_720p_methods.py')
    bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
    def dumps(payload,*args,**kw):
        nonlocal count
        if isinstance(payload,dict) and 'protocol' in payload and 'videos' in payload:
            assert not a.gate and count==4000,count
            assert all(r['attention_calls']==r['full_attention_calls']==4000 and r['seed']==0 for r in payload['videos'])
            protocol=payload['protocol'];protocol.pop('gma_seed_mode',None)
            protocol.update(method='MonarchRT121 f1/h2/w1',monarch_rt=plan['method'],method_classification='external MonarchRT baseline',
                pure_all_layer_gma=False,pure_all_layer=True,attention_schedule=dict(dense_first_steps=0,dense_first_layers=0,transformer_layers=40,remaining_attention='MonarchRT121'),
                gpu=read(control/'preflight.json'),tflops_per_call=plan['tflops_per_call'],full_50step_pflops=plan['full_50step_pflops'])
            count=0
        return json.dumps(payload,*args,**kw)
    bench.json=types.SimpleNamespace(loads=json.loads,dumps=dumps)
    sys.argv=['benchmark','--backend','gma_hellinger_p16_all','--model',plan['model'],'--sparse_video_gen_root',plan['svg_root'],
        '--prompt_path',plan['prompt_path'],'--output_dir',str(out),'--start','0','--count','1' if a.gate else '20','--seed','0',
        '--height','720','--width','1280','--num_frames','81','--num_steps','50','--wan_forward_backend','svg','--qk_norm_backend','triton',
        '--rope_backend','svg','--gma_preprocess_backend','fused','--gma_output_layout','head_major','--gma_head_chunk','1']
    if a.gate:
        try:bench.main()
        except GateComplete:
            assert len(gate['rows'])==16 and count==80
            gate.update(complete=True,passed=True,time=time.time(),
                mean_method_ms=statistics.mean(x for row in gate['rows'] for x in row['method_ms']),
                mean_sdpa_ms=statistics.mean(x for row in gate['rows'] for x in row['sdpa_ms']))
            save(gate_path,gate)
        else:raise AssertionError('First-step gate did not intercept80 calls')
    else:
        assert read(gate_path)['passed'];bench.main()
        runtime=read(out/'runtime.json');assert len(runtime['videos'])==20 and count==0
        save(ROOT/'generation_complete.json',dict(complete=True,videos=20,slot=a.slot,time=time.time()))

if __name__=='__main__':main()
