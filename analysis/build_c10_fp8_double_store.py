"""Use FP8's smaller outputs to fit two TMA staging buffers at four CTAs."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'native_fisher'
s=(root/'fisher_c10_fp8_shared_store.cu').read_text()
old='''      if(tid==0) asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");
      __syncthreads();
      uint8_t* fp8temp=reinterpret_cast<uint8_t*>(temp);'''
assert s.count(old)==1
s=s.replace(old,'''      // Previous tile's stores finished before owner scratch was reused.
      // K and V now occupy disjoint buffers, so V need not wait for K TMA.
      uint8_t* fp8temp=reinterpret_cast<uint8_t*>(temp)+kind*64*128;''')
assert s.count('uint32_t p=__cvta_generic_to_shared(temp);')==1
s=s.replace('uint32_t p=__cvta_generic_to_shared(temp);','uint32_t p=__cvta_generic_to_shared(fp8temp);')
assert s.count('(80*128*2)*2+(64*40+2*6*40)*4')==2
s=s.replace('(80*128*2)*2+(64*40+2*6*40)*4','(80*128*2)*2+2*64*128')
# Weight preparation touches only registers. The end-of-owner-loop barrier
# already completed every scratch read before either output can reuse it.
s=s.replace('    __syncthreads(); // All weights now reside in registers; shared scores may be reused.\n','')
(root/'fisher_c10_fp8_double_store.cu').write_text(s)
