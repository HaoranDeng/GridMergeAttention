"""Unchanged native Dense/SVG-EAR inference, audited for replacement input019."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import time
from worker import ROOT, read, save, sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['Dense','EAR_p08'],required=True);a=p.parse_args()
    plan=read(ROOT/'experiment_plan.json');snap=Path(plan['snapshot']);out=ROOT/a.method/'video20'
    assert os.environ['CUDA_VISIBLE_DEVICES']==plan['gpu']['uuid']
    for rel,h in plan['source_sha256'].items():assert sha(snap/rel)==h,rel
    for rel,h in plan['svg_sha256'].items():assert sha(Path(plan['source_svg_root'])/rel)==h,rel
    for row in plan['input_manifest']:assert sha(Path(row['image']))==row['image_sha256']
    out.mkdir(parents=True,exist_ok=True);assert not list(out.glob('*.mp4'))
    save(out/'input_manifest.json',dict(protocol=plan['protocol'],inputs=plan['input_manifest']))
    if a.method=='EAR_p08':
        assert os.environ['SVG_TOP_P']=='0.8'
        os.environ['SVG_DENSITY_LOG']=str(out/'density.jsonl')
        assert not (out/'density.jsonl').exists()
    os.chdir(snap);sys.path.insert(0,str(snap))
    import torch
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    from wan.modules.model import WanSelfAttention
    layers=[0]*40;iterations=[]
    old_forward=WanSelfAttention.forward
    def forward(self,x,grid_size,rope_cache):
        assert tuple(x.shape)==(1,75600,5120) and tuple(map(int,grid_size))==(21,45,80)
        assert not self.enable_token_merging
        assert self.attention_backend==('sdpa' if a.method=='Dense' else 'ear_all')
        layers[self.layer_idx]+=1
        return old_forward(self,x,grid_size,rope_cache)
    WanSelfAttention.forward=forward
    if a.method=='EAR_p08':
        from attention import svg_family_wan as family
        ear=family._load_svg_modules()['ear'];original_cluster=ear.kmeans_clustering
        def cluster(self,q,k,layer_idx):
            initial=not self.centroids_init
            result=original_cluster(self,q,k,layer_idx)
            iterations.append(dict(layer=layer_idx,step=(layers[layer_idx]-1)//2,
                                   branch=(layers[layer_idx]-1)%2,initialization=initial,
                                   q_iterations=result[3],k_iterations=result[7]))
            return result
        ear.kmeans_clustering=cluster
    spec=importlib.util.spec_from_file_location('frozen_generate',snap/'generate.py')
    gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen);old_save=gen.save_video_atomic
    def save_video(path,video,**kwargs):
        assert Path(path).name=='019-00.mp4' and tuple(video.shape)==(81,720,1280,3)
        assert layers==[100]*40,layers
        flops=4*75600**2*40*128*4000
        if a.method=='EAR_p08':
            import json
            densities=[json.loads(s) for s in (out/'density.jsonl').read_text().splitlines()]
            assert len(densities)==len(iterations)==4000
            assert sum(r['initialization'] for r in iterations)==40
            for row,d in zip(iterations,densities):
                assert row['layer']==d['layer']
                row['density']=float(d['avg_density'])
                for key in ['q_iterations','k_iterations']:row[key]=int(row[key])
                assert 0<=row['density']<=1 and 0<row['q_iterations']<=50 and 0<row['k_iterations']<=50
                row['major_matmul_flops']=4*75600**2*40*128*row['density']+2*75600*40*128*(300*row['q_iterations']+1000*row['k_iterations'])+4*75600*1000*40*128
            flops=sum(r['major_matmul_flops'] for r in iterations)
            save(out/'flops50_019.json',dict(index=19,seed=19,calls=4000,full_50_step_flops=flops,rows=iterations,
                 scope='Actual 4000-call density and returned Q/K iterations; major matmuls, 2 FLOPs per multiply-add.'))
        old_save(path,video,**kwargs)
        save(out/'purity_records.json',[dict(video=Path(path).name,index=19,seed=19,attention_calls=4000,layer_calls=layers,
             dense_self_attention_calls=4000 if a.method=='Dense' else 0,video_sha256=sha(Path(path)),
             full_50_step_pflops=flops/1e15,gpu=plan['gpu'],saved_at=time.time())])
    gen.save_video_atomic=save_video
    sys.argv=['generate.py','--config',plan['configs'][a.method],'--checkpoint','pretrained',
              '--prompt_path',plan['prompt_path'],'--image_list',plan['image_list'],'--image_dir',plan['image_dir'],
              '--output_dir',str(out),'--runtime_json',str(out/'runtime.json'),'--num_videos','20',
              '--num_samples','1','--seed','0','--num_prompt_shards','20','--prompt_shard_id','19']
    gen.main()
    records=read(out/'runtime.json')['records'];assert len(records)==1
    assert records[0]['prompt_index']==19 and records[0]['attention_calls']==records[0]['full_attention_calls']==4000
    save(ROOT/a.method/'generation_complete.json',dict(complete=True,videos=1,index=19,seed=19,time=time.time()))

if __name__=='__main__':main()
