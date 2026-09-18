"""CPU-only compilation and static inspection; no CUDA context."""
import os,sys,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']=''
ART=Path('/local2/denghaoran/GridMergeAttention/experiments/gma_c10_fp8_150ms_20260910/scai7')
ART.mkdir(parents=True,exist_ok=True)
os.environ['TORCH_EXTENSIONS_DIR']=str(ART/'torch_extensions')
import torch
from native_fisher.operator import extension
from native_fisher.fp8_c10 import selector_extension
for name in sys.argv[1:]:
    module=selector_extension() if name=='selector' else extension(10,ROOT/'native_fisher'/(name+'.cu'))
    assert not torch.cuda.is_initialized()
    resources=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',module.__file__],text=True)
    (ART/(name+'_compile.json')).write_text(json.dumps({'source_sha256':module.source_sha256,'resources':resources,'gpu_context_initialized':False},indent=2))
    print(name,resources,flush=True)
