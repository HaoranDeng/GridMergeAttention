"""Reduce FP8 epilogue shuffles using native halfword shared stores."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'native_fisher'
s=(root/'fisher_c10_fp8_single_round.cu').read_text()
a=s.index('      #pragma unroll\n      for(int g=0;g<4;++g) {',s.index('uint8_t* fp8temp='))
b=s.index('      asm volatile("fence.proxy.async.shared::cta;',a)
replacement='''      #pragma unroll
      for(int i=0;i<64;i+=2) {
        int half=(i&3)>>1,row=warp*16+seed+8*half;
        int d=2*(lane&3)+8*(i>>2),offset=row*128+d;
        float2 value=make_float2(acc[i],acc[i+1]);
        uint16_t bits=__nv_cvt_float2_to_fp8x2(value,__NV_SATFINITE,__NV_E4M3);
        *reinterpret_cast<uint16_t*>(fp8temp+(offset^(((offset>>7)&7)<<4)))=bits;
      }
'''
out=s[:a]+replacement+s[b:]
(root/'fisher_c10_fp8_shared_store.cu').write_text(out)
(root/'fisher_c10_fp8_single_round_regs.cu').write_text(s.replace('__launch_bounds__(128,4)','__launch_bounds__(128,3)'))
