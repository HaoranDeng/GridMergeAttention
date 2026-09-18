"""Prefetch next-frame selected Q after current QK consumes its fragments.

The next frame uses the same RF0 seed indices and query columns. One lane
per four-lane row group requests the two128-byte cache lines of each row.
Frame20 is excluded; every requested address stays within native BF16 Q.
No additional shared buffer, global workspace, arithmetic or output is added.
"""
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'native_fisher'
s=(root/'fisher_c10_fp8_local_scores.cu').read_text()
marker='    if(stop_after==1) continue;'
assert s.count(marker)==1
s=s.replace(marker,marker+r'''
    if(qo+1<Q && (lane&3)==0) {
      constexpr int frame_stride=R*C*H*128;
      asm volatile("prefetch.global.L1 [%0];"::"l"(q+qi0+frame_stride):"memory");
      asm volatile("prefetch.global.L1 [%0];"::"l"(q+qi0+frame_stride+64):"memory");
      asm volatile("prefetch.global.L1 [%0];"::"l"(q+qi1+frame_stride):"memory");
      asm volatile("prefetch.global.L1 [%0];"::"l"(q+qi1+frame_stride+64):"memory");
    }
''')
(root/'fisher_c10_fp8_query_prefetch.cu').write_text(s)
