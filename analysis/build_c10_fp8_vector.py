"""FP8 register transpose and 128-bit global stores, no output shared staging."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
for mode in ('direct','scaled'):
    s=(root/f'native_fisher/fisher_c10_fp8_{mode}.cu').read_text()
    start=s.index('      if(cb==13) {')
    end=s.index('\n    }\n  }\n  if(tma_store==1',start)
    s=s[:start]+'''
      #pragma unroll
      for(int g=0;g<4;++g) {
        uint32_t packed[4];
        #pragma unroll
        for(int j=0;j<4;++j) {
          int half=j&1,ix=g*16+(j>>1)*8+half*2;
          float2 lo=make_float2(acc[ix]*inv_quant[half],acc[ix+1]*inv_quant[half]);
          float2 hi=make_float2(acc[ix+4]*inv_quant[half],acc[ix+5]*inv_quant[half]);
          uint32_t pair0=__nv_cvt_float2_to_fp8x2(lo,__NV_SATFINITE,__NV_E4M3);
          uint32_t pair1=__nv_cvt_float2_to_fp8x2(hi,__NV_SATFINITE,__NV_E4M3);
          uint32_t pair=pair0|(pair1<<16);
          int src=(lane&~3)+2*(lane&1);
          uint32_t a=__shfl_sync(0xffffffff,pair,src),b=__shfl_sync(0xffffffff,pair,src+1);
          int shift=(lane&2)*8;
          packed[j]=((a>>shift)&65535)|((b>>shift)<<16);
        }
        uint4 value=fragment_transpose(make_uint4(packed[0],packed[1],packed[2],packed[3]),lane);
        int row=warp*16+seed+8*(lane&1),col=cb*6+col_for_row(row);
        int d=g*32+16*((lane>>1)&1);
        int64_t oi=(((int64_t(qo)*C+col)*AH+head)*(KO*R*10)+kb*10+seed_for_row(row))*128+d;
        uint8_t* dst=kind==0?mk:mv;
        if(row<60 && col<80) {
          asm volatile("st.global.L1::evict_first.v4.b32 [%0],{%1,%2,%3,%4};"::"l"(dst+oi),"r"(value.x),"r"(value.y),"r"(value.z),"r"(value.w):"memory");
        }
      }
'''+s[end:]
    s=s.replace('if(tid==0)asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");','')
    s=s.replace('if(tma_store==1 && tid==0) asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");','')
    # Descriptors are unused in this variant.
    s=s.replace(' if(tma_store==1) {\n   uint64_t dims', ' if(false) {\n   uint64_t dims')
    path=root/f'native_fisher/fisher_c10_fp8_vector_{mode}.cu'
    s='\n'.join(line.rstrip() for line in s.splitlines())+'\n'
    path.write_text('// Experimental FP8 vector-store merge.\n'+s)
    print(path.name)
