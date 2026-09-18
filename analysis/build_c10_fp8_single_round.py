"""Round FP32 merge accumulators directly to FP8, replacing the BF16 boundary."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'native_fisher/fisher_c10_fp8_direct.cu').read_text()
needle='      #pragma unroll\n      for(int i=0;i<64;++i) acc[i]=__bfloat162float(__float2bfloat16_rn(acc[i]));'
assert s.count(needle)==1
s=s.replace(needle,'      // One rounding step: FP32 accumulation directly to E4M3 storage.')
(root/'native_fisher/fisher_c10_fp8_single_round.cu').write_text(s)
