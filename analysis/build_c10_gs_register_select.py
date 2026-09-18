"""Prevent dynamic-index lowering of fixed-register seed gathers."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=(root/'native_fisher/fisher_gs_distributed.cu').read_text()
helper='''
// Opaque selects stop CUDA's array-index recognition from moving the entire
// FP64 feature table into thread-local memory for two runtime gathers.
__device__ __forceinline__ double register_select(double previous,double value,bool take) {
  double out;
  asm("{ .reg .pred p; setp.ne.u32 p,%3,0; selp.f64 %0,%2,%1,p; }"
      : "=d"(out) : "d"(previous),"d"(value),"r"(int(take)));
  return out;
}
'''
base=base.replace('template<int N>',helper+'\ntemplate<int N>',1)
for old,new in [
    ('if(c==anchor/CS)value=z[c][p];','value=register_select(value,z[c][p],c==anchor/CS);'),
    ('if(c==picked/CS)value=res[c][p];','value=register_select(value,res[c][p],c==picked/CS);')]:
    assert old in base,old;base=base.replace(old,new)
(root/'native_fisher/fisher_gs_register_select.cu').write_text(base)
