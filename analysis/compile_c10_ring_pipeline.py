"""CPU-only compilation: no runtime or numerical claims."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
out=Path(sys.argv[1]);assert not out.exists()
os.environ.update(CUDA_VISIBLE_DEVICES='-1',MAX_JOBS='2',TORCH_EXTENSIONS_DIR=str(out.parent/'torch_extensions'))
from native_fisher.operator import extension
source_dir=Path(sys.argv[2]) if len(sys.argv)>2 else ROOT/'native_fisher'
paths=[source_dir/n for n in ('c10_wgmma_ring.cuh','c10_ring_consumer.cuh','c10_ring_pipeline.cu')]
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
stamp=hashlib.sha256(''.join(hashes.values()).encode()).hexdigest()[:12]
result={'scope':'CPU compilation only; no GPU context or performance claim','sources':hashes}
try:
    ext=extension(10,paths[-1],name_suffix='_all'+stamp)
    assert hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result.update(compiled=True,module=ext.__file__)
    result['resource_dump']=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',ext.__file__],text=True)
    print(result['resource_dump'],flush=True)
except BaseException as e:result.update(compiled=False,error=repr(e));raise
finally:out.write_text(json.dumps(result,indent=2))
