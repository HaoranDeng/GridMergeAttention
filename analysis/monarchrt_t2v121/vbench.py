"""Existing six-dimension custom-input VBench, using verified generated videos."""
import os
from pathlib import Path
import runpy
import sys
import time
from common import ROOT,read,save,sha

VB=Path('/xuanwu-tank/east/zrl/exp-gridmergeattn/third_party/VBench')
DIMS=['subject_consistency','background_consistency','motion_smoothness','dynamic_degree','aesthetic_quality','imaging_quality']
NORM={'subject_consistency':(.1462,1.),'background_consistency':(.2615,1.),'motion_smoothness':(.706,.9975),
      'dynamic_degree':(0.,1.),'aesthetic_quality':(0.,1.),'imaging_quality':(0.,1.)}

assert read(ROOT/'generation_complete.json')['complete']
out=ROOT/'vbench';videos=out/'videos';videos.mkdir(parents=True,exist_ok=True)
assert not list((out/'out').glob('*_eval_results.json')),'Existing result; inspect instead of replacing'
proofs={}
for r in read(ROOT/'video20/runtime.json')['videos']:
    path=ROOT/'video20'/r['video'];proofs[str(path)]=sha(path);target=videos/path.name
    if target.exists():assert target.resolve()==path.resolve()
    else:target.symlink_to(path)
assert len(proofs)==20
sources=[VB/'evaluate.py',VB/'vbench/__init__.py',VB/'vbench/utils.py',VB/'vbench/distributed.py',*[VB/f'vbench/{d}.py' for d in DIMS]]
hashes={str(p):sha(p) for p in sources}
save(out/'protocol.json',dict(dimensions=DIMS,normalization=NORM,mode='custom_input',videos_sha256=proofs,source_sha256=hashes,
    note='Existing six-dimension custom-input quality composite, not an official leaderboard score'))
sys.path.insert(0,str(VB));os.chdir(VB)
import clip
old=clip.load
def load(*a,**kw):kw.setdefault('download_root',str(ROOT/'vbench_clip_cache'));return old(*a,**kw)
clip.load=load
sys.argv=['evaluate.py','--videos_path',str(videos),'--output_path',str(out/'out'),'--mode','custom_input','--dimension',*DIMS]
runpy.run_path(str(VB/'evaluate.py'),run_name='__main__')
result,=list((out/'out').glob('*_eval_results.json'));data=read(result)
assert set(data)==set(DIMS) and all(len(data[d][1])==20 for d in DIMS)
raw={d:float(data[d][0]) for d in DIMS};norm={d:(raw[d]-NORM[d][0])/(NORM[d][1]-NORM[d][0]) for d in DIMS}
for path,h in {**proofs,**hashes}.items():assert sha(path)==h
save(out/'summary.json',dict(complete=True,raw=raw,normalized=norm,six_dimension_quality_score=sum(norm[d]*(.5 if d=='dynamic_degree' else 1.) for d in DIMS)/5.5,time=time.time()))
