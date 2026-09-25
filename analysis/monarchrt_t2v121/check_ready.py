"""CPU-only integrity/dependency check; does not claim numerical GPU validation."""
import importlib.util
import os
from pathlib import Path
import sys
import time
from common import ROOT,read,save,sha

assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
plan=read(ROOT/'experiment_plan.json');snapshot=Path(plan['snapshot'])
for rel,h in plan['source_sha256'].items():assert sha(snapshot/rel)==h,rel
for path,h in plan['svg_sha256'].items():assert sha(path)==h,path
sys.path.insert(0,str(snapshot))
import torch
import attention.token_merging as tm
assert not torch.cuda.is_available()
assert callable(tm._hard_c4_attention) and callable(tm._causal_one_token_attention)
import diffusers,lpips
assert hasattr(diffusers,'WanPipeline')
spec=importlib.util.spec_from_file_location('ready_benchmark',snapshot/'analysis/benchmark_wan14b_720p_methods.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert callable(module.main) and callable(module.summarize)
for path,h in read(ROOT/'evaluation_support.json')['files'].items():assert sha(path)==h,path
sys.path.insert(0,str(ROOT/'evaluation_support'))
spec=importlib.util.spec_from_file_location('ready_scores',snapshot/'analysis/score_videos_diffusers.py')
scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
assert callable(scorer.score_pair) and callable(scorer.aggregate)
manifest=read(ROOT/'video20/prompt_manifest.json')
assert manifest['seed']==0 and manifest['selected_prompt_count']==20
assert len(plan['inputs'])==20 and plan['method']['w_reduce']==1 and plan['method']['h_reduce']==2
receipt=dict(passed=True,cpu_only=True,gpu_validation='not exercised by this CPU-only check',torch=torch.__version__,cuda=torch.version.cuda,
             diffusers=diffusers.__version__,source_files=len(plan['source_sha256']),time=time.time())
save(ROOT/'cpu_readiness.json',receipt);print(receipt)
