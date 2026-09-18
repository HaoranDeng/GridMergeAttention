"""Swizzle adjacent score columns to distribute WGMMA stores across banks."""
from pathlib import Path
import re
root=Path(__file__).resolve().parents[1]/'native_fisher'
s=(root/'fisher_c10_fp8_shared_store.cu').read_text()
pos=s.index('__device__ __forceinline__ uint32_t load_query')
s=s[:pos]+'''// Each score-store instruction otherwise touches only even or odd banks.
// Row bit 2 distinguishes the duplicate four-row groups within a warp.
__device__ __forceinline__ int normoff(int row,int ki) {
  return row*40+(ki^((row>>2)&1));
}
'''+s[pos:]
assert s.count('normalized[row*40+ki]')==1
s=s.replace('normalized[row*40+ki]','normalized[normoff(row,ki)]')
s,count=re.subn(r'normalized\[\(column\*10\+(\d)\)\*40\+ki\]',r'normalized[normoff(column*10+\1,ki)]',s)
assert count==10
(root/'fisher_c10_fp8_score_swizzle.cu').write_text(s)
