"""Smaller output accumulators and ownership strips with resident original KV.

Unlike the rejected N32 streamed-V experiment, both full BF16 K and V stay
resident across all query tiles. Five16-key ownership strips preserve the
original first64/last16 mass and entropy reduction order. N64 output tiles
need4KiB FP8 staging; the smaller scratch and registers target five CTAs.
"""
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'native_fisher'
s=(root/'fisher_c10_fp8_local_scores.cu').read_text()
s=s.replace('__launch_bounds__(128,4)','__launch_bounds__(128,5)')
a=s.index('    float* normalized=reinterpret_cast<float*>(temp);')
b=s.index('    uint32_t packed_weights[20];',a)
t=s[a:b]
t=t.replace('normalized+64*40','normalized+64*16')
t=t.replace('chunk<2','chunk<5').replace('i<20','i<8')
t=t.replace('row*40+ki','row*16+ki').replace('chunk*20','chunk*8')
t=t.replace('index<6*40','index<6*16').replace('index/40,ki=index%40','index/16,ki=index%16')
t=t.replace(')*40+ki',')*16+ki')
t=t.replace('pair<5','pair<2').replace('min(col_for_row(row),5)*40','min(col_for_row(row),5)*16')
t=t.replace('else if(chunk==0 || pair<3)','else if(chunk<4)').replace('else if(pair==3)','else if(pair==0)')
s=s[:a]+t+s[b:]
a=s.index('    #pragma unroll\n    for(int kind=0;kind<2;++kind)')
b=s.index('  if(tma_store==1 && tid==0)',a)
body=r'''
    #pragma unroll
    for(int kind=0;kind<2;++kind) {
      #pragma unroll 1
      for(int feature=0;feature<128;feature+=64) {
        asm volatile("wgmma.fence.sync.aligned;":::"memory");
        float acc[32];
        using Merge=MMA_64x64x16_F32BF16BF16_RS<Major::K,Major::MN>;
        #pragma unroll
        for(int ki=0;ki<80;ki+=16) {
          auto bd=desc((kind==0?sk:sv)+boff(ki,feature),10240,1024);
          int wi=(ki/8)*2;
          Merge::fma(packed_weights[wi],packed_weights[wi+1],packed_weights[wi+2],packed_weights[wi+3],bd,E32(acc),ki==0?ScaleOut::Zero:ScaleOut::One);
        }
        asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
        #pragma unroll
        for(int i=0;i<32;++i)acc[i]=__fmul_rn(acc[i],residual_scales[(i&3)>>1]);
        if(cb==13) {
          uint8_t* dst=kind==0?mk:mv;
          #pragma unroll
          for(int i=0;i<32;i+=2) {
            int half=(i&3)>>1,row=warp*16+seed+8*half;
            int d=feature+2*(lane&3)+8*(i>>2),col=cb*6+col_for_row(row);
            int64_t oi=(((int64_t(qo)*C+col)*AH+head)*(KO*R*10)+kb*10+seed_for_row(row))*128+d;
            uint16_t bits=__nv_cvt_float2_to_fp8x2(make_float2(acc[i],acc[i+1]),__NV_SATFINITE,__NV_E4M3);
            if(row<60 && col<80)*reinterpret_cast<uint16_t*>(dst+oi)=bits;
          }
          continue;
        }
        if(tid==0)asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");
        __syncthreads();
        uint8_t* fp8temp=reinterpret_cast<uint8_t*>(temp);
        #pragma unroll
        for(int i=0;i<32;i+=2) {
          int half=(i&3)>>1,row=warp*16+seed+8*half;
          int d=2*(lane&3)+8*(i>>2),offset=row*64+d;
          uint16_t bits=__nv_cvt_float2_to_fp8x2(make_float2(acc[i],acc[i+1]),__NV_SATFINITE,__NV_E4M3);
          // Byte-address Swizzle<2,4,3>: source bits7..8, destination4..5.
          *reinterpret_cast<uint16_t*>(fp8temp+(offset^(((offset>>7)&3)<<4)))=bits;
        }
        asm volatile("fence.proxy.async.shared::cta;":::"memory");
        __syncthreads();
        if(tid==0) {
          uint32_t p=__cvta_generic_to_shared(temp);
          const CUtensorMap* map=kind==0?&map_k:&map_v;
          asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%0, {%1,%2,%3,%4,%5}], [%6];"::"l"(map),"r"(feature),"r"(kb*10),"r"(head),"r"(cb*6),"r"(qo),"r"(p):"memory");
          asm volatile("cp.async.bulk.commit_group;":::"memory");
        }
      }
    }
  }
'''
s=s[:a]+body+s[b:]
s=s.replace('(64*40+6*40)*4','(64*16+6*16)*4')
s=s.replace('box[5]={128,10,1,6,1}','box[5]={64,10,1,6,1}')
s=s.replace('CU_TENSOR_MAP_SWIZZLE_128B','CU_TENSOR_MAP_SWIZZLE_64B')
(root/'fisher_c10_fp8_n64_owner16.cu').write_text(s)
