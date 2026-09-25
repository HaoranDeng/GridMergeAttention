"""Freeze the scorer's helper import separately from the historical generator."""
from pathlib import Path
import shutil
from common import ROOT,save,sha

original=Path('/local2/denghaoran/GridMergeAttention/experiments/monarchrt_i2v20_sparkler_20260925/snapshot/analysis/probe_results.py')
target=ROOT/'evaluation_support/analysis/probe_results.py';target.parent.mkdir(parents=True,exist_ok=True)
if target.exists():assert sha(target)==sha(original)
else:shutil.copy2(original,target)
save(ROOT/'evaluation_support.json',dict(source=str(original),files={str(target):sha(target)}))
print('Frozen metric helper dependency; generator source unchanged')
