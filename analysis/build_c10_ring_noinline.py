"""Isolate producer register allocation; preserve all arithmetic and barriers."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'native_fisher';target=source/'ring_noinline';target.mkdir(exist_ok=True)
for name in ('c10_wgmma_ring.cuh','c10_ring_consumer.cuh','c10_ring_pipeline.cu'):
    code=(source/name).read_text()
    if name=='c10_wgmma_ring.cuh':
        before='__device__ __forceinline__ void produce('
        assert code.count(before)==1
        code=code.replace(before,'__device__ __noinline__ void produce(')
    (target/name).write_text(code)
print(target)
