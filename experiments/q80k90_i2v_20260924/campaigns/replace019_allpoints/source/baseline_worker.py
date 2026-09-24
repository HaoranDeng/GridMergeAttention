"""Run one unchanged native external baseline on new019 with full-call audits."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import time
from worker import ROOT,read,save,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',required=True);a=p.parse_args()
    plan=read(ROOT/'experiment_plan.json');setting=plan['settings'][a.method];kind=setting['family']
    assert kind in ['ear','svg2','svg'] and read(ROOT/'gpu2/native_gate.json')['passed']
    assert os.environ['CUDA_VISIBLE_DEVICES']==plan['gpu']['uuid']
    snap=Path(plan['snapshot']);out=ROOT/a.method/'video20'
    for rel,h in plan['source_sha256'].items():assert sha(snap/rel)==h,rel
    for rel,h in plan['svg_sha256'].items():assert sha(Path(plan['source_svg_root'])/rel)==h,rel
    for path,h in plan['config_sha256'].items():assert sha(Path(path))==h,path
    for row in plan['input_manifest']:assert sha(Path(row['image']))==row['image_sha256']
    out.mkdir(parents=True,exist_ok=True);assert not list(out.glob('*.mp4'))
    save(out/'input_manifest.json',dict(protocol=plan['protocol'],inputs=plan['input_manifest']))
    if kind in ['ear','svg2']:
        assert float(os.environ['SVG_TOP_P'])==setting['top_p']
        os.environ['SVG_DENSITY_LOG']=str(out/'density.jsonl');assert not (out/'density.jsonl').exists()
    else:assert float(os.environ['SVG_SPARSITY'])==setting['sparsity']
    os.chdir(snap);sys.path.insert(0,str(snap))
    import torch
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    from wan.modules.model import WanSelfAttention
    from attention import svg_family_wan as family
    layers=[0]*40;iterations=[];old_forward=WanSelfAttention.forward
    def forward(self,x,grid_size,rope_cache):
        assert tuple(x.shape)==(1,75600,5120) and tuple(map(int,grid_size))==(21,45,80)
        assert not self.enable_token_merging and self.attention_backend=={'ear':'ear_all','svg2':'svg2_all','svg':'svg_all'}[kind]
        layers[self.layer_idx]+=1
        return old_forward(self,x,grid_size,rope_cache)
    WanSelfAttention.forward=forward
    if kind in ['ear','svg2']:
        cls=family._load_svg_modules()[kind];original=cls.kmeans_clustering
        def cluster(self,q,k,layer_idx):
            initial=not self.centroids_init;result=original(self,q,k,layer_idx)
            iterations.append(dict(layer=layer_idx,step=(layers[layer_idx]-1)//2,branch=(layers[layer_idx]-1)%2,
                initialization=initial,q_iterations=int(result[3]),k_iterations=int(result[7])))
            return result
        cls.kmeans_clustering=cluster
    spec=importlib.util.spec_from_file_location('frozen_generate',snap/'generate.py');gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
    original_save=gen.save_video_atomic
    def save_video(path,video,**kwargs):
        assert Path(path).name=='019-00.mp4' and tuple(video.shape)==(81,720,1280,3) and layers==[100]*40
        if kind=='svg':
            actual=upper=setting['static_full50_flops'];scope=setting['static_flops_scope']
            # The same static mask and grid have no prompt-dependent arithmetic.
            call_rows=[dict(layer=l,step=s,branch=b,major_matmul_flops=actual/4000) for s in range(50) for b in range(2) for l in range(40)]
        else:
            import json
            densities=[json.loads(x) for x in (out/'density.jsonl').read_text().splitlines()]
            assert len(densities)==len(iterations)==4000 and sum(x['initialization'] for x in iterations)==40
            for row,d in zip(iterations,densities):
                assert row['layer']==d['layer'];row['density']=float(d['avg_density'])
                limit=50 if row['initialization'] else 2
                assert 0<=row['density']<=1 and 0<row['q_iterations']<=limit and 0<row['k_iterations']<=limit
                sparse=4*75600**2*40*128*row['density'];comp=4*75600*1000*40*128 if kind=='ear' else 0
                row['major_matmul_flops']=sparse+2*75600*40*128*(300*row['q_iterations']+1000*row['k_iterations'])+comp
                row['scheduled_upper_flops']=sparse+2*75600*40*128*1300*limit+comp
            call_rows=iterations;actual=sum(x['major_matmul_flops'] for x in call_rows);upper=sum(x['scheduled_upper_flops'] for x in call_rows)
            scope='Actual4000-call density and returned Q/K iterations; matched scheduled upper bound retained separately.'
        save(out/'flops50_019.json',dict(index=19,seed=19,calls=4000,full_50_step_flops=actual,scheduled_upper_full_50_step_flops=upper,rows=call_rows,scope=scope))
        original_save(path,video,**kwargs)
        save(out/'purity_records.json',[dict(video=Path(path).name,index=19,seed=19,attention_calls=4000,layer_calls=layers,dense_self_attention_calls=0,
             video_sha256=sha(Path(path)),full_50_step_pflops=actual/1e15,scheduled_upper_full_50_step_pflops=upper/1e15,gpu=plan['gpu'],saved_at=time.time())])
    gen.save_video_atomic=save_video
    sys.argv=['generate.py','--config',setting['config'],'--checkpoint','pretrained','--prompt_path',plan['prompt_path'],
        '--image_list',plan['image_list'],'--image_dir',plan['image_dir'],'--output_dir',str(out),'--runtime_json',str(out/'runtime.json'),
        '--num_videos','20','--num_samples','1','--seed','0','--num_prompt_shards','20','--prompt_shard_id','19']
    gen.main();row,=read(out/'runtime.json')['records']
    assert row['prompt_index']==19 and row['attention_calls']==row['full_attention_calls']==4000
    save(ROOT/a.method/'generation_complete.json',dict(complete=True,videos=1,index=19,seed=19,time=time.time()))

if __name__=='__main__':main()
