"""Compile customized FP8 compressed route on CPU; never initialize CUDA."""
import os,sys,json
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['FLASHINFER_CUDA_ARCH_LIST']='9.0a'
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from native_fisher.fp8_flashinfer_route import compile_cpu,BASE,jit_args
import torch
result={'scope':'CPU build only, compressed-only FP8 route, no timing or quality claim','jit_name':jit_args()[0]}
try:
    module=compile_cpu();assert not torch.cuda.is_initialized()
    result.update(valid=True,gpu_context_initialized=False)
except BaseException as e:
    result.update(valid=False,error=repr(e));raise
finally:
    (Path(BASE)/'flashinfer_route_compile.json').write_text(json.dumps(result,indent=2))
print(result,flush=True)
