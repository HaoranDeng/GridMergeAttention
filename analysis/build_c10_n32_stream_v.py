"""Reduce live merge accumulators and stream V in 32-feature shared tiles.

Same BF16 inputs/weights, FP32 K16 accumulation order, SQ1 division and output
layout. V staging is reloaded per feature/query tile, trading cache traffic for
five-CTA residency. Experimental until differential and timing gates pass.
"""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'native_fisher/fisher_c10_half_owner.cu').read_text()
s=s.replace('#define E32(a)', '#define E16(a) E8(a,0),E8(a,8)\n#define E32(a)',1)
s=s.replace('__launch_bounds__(128,4)', '__launch_bounds__(128,5)')
s=s.replace('bf16* temp=sv+80*128;', 'bf16* temp=sv+80*32;')
s=s.replace('    *reinterpret_cast<uint4*>(sv+boff(ki,d))=*reinterpret_cast<const uint4*>(v+src);','')
s=s.replace('(80*128*2)*2+64*128*2', '80*128*2+80*32*2+64*128*2')
pos=s.index('__device__ __forceinline__ int aoff')
s=s[:pos]+'''__device__ __forceinline__ uint64_t desc64(const bf16* p,int lead,int stride) {
  uint64_t a=__cvta_generic_to_shared(p);
  return ((a&0x3ffff)>>4)|(uint64_t(lead>>4)<<16)|(uint64_t(stride>>4)<<32)|(uint64_t(2)<<62);
}
__device__ __forceinline__ int v32off(int k,int n) {
  // CuTe Swizzle<2,4,3>: source bits remain byte bits 7..8.
  int x=k*32+n;return x^(((x>>6)&3)<<3);
}
'''+s[pos:]
start=s.index('    #pragma unroll\n    for(int kind=0;kind<2;++kind)')
end=s.index('  if(tma_store==1 && tid==0)',start)
s=s[:start]+'''    #pragma unroll
    for(int kind=0;kind<2;++kind) {
      if(tid==0) asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");
      __syncthreads();
      #pragma unroll
      for(int feature=0;feature<128;feature+=32) {
        if(kind==1) {
          for(int vec=tid;vec<80*4;vec+=128) {
            int ki=vec/4,d=(vec%4)*8;
            int src=((kb*80+ki)*H+gh)*128+feature+d;
            *reinterpret_cast<uint4*>(sv+v32off(ki,d))=*reinterpret_cast<const uint4*>(v+src);
          }
          __syncthreads();
          asm volatile("fence.proxy.async.shared::cta;":::"memory");
        }
        asm volatile("wgmma.fence.sync.aligned;":::"memory");
        float acc[16];
        using Merge=MMA_64x32x16_F32BF16BF16_RS<Major::K,Major::MN>;
        #pragma unroll
        for(int ki=0;ki<80;ki+=16) {
          uint64_t bd=kind==0?desc(sk+boff(ki,feature),10240,1024):desc64(sv+v32off(ki,0),5120,512);
          int wi=(ki/8)*2;
          Merge::fma(packed_weights[wi],packed_weights[wi+1],packed_weights[wi+2],packed_weights[wi+3],bd,E16(acc),ki==0?ScaleOut::Zero:ScaleOut::One);
        }
        asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
        #pragma unroll
        for(int i=0;i<16;++i)acc[i]=__fmul_rn(acc[i],residual_scales[(i&3)>>1]);
        if(cb==13) {
          bf16* dst=kind==0?mk:mv;
          #pragma unroll
          for(int i=0;i<16;i+=2) {
            int row=warp*16+seed+8*((i&3)>>1);
            int d=feature+2*(lane&3)+8*(i>>2),col=cb*6+col_for_row(row);
            int64_t oi=(((int64_t(qo)*C+col)*AH+head)*(KO*R*10)+kb*10+seed_for_row(row))*128+d;
            union {bf16 values[2];uint32_t bits;} pair;
            pair.values[0]=__float2bfloat16_rn(acc[i]);pair.values[1]=__float2bfloat16_rn(acc[i+1]);
            if(row<60 && col<80)*reinterpret_cast<uint32_t*>(dst+oi)=pair.bits;
          }
        } else {
          #pragma unroll
          for(int g=0;g<2;++g) {
            uint32_t packed[4];
            #pragma unroll
            for(int j=0;j<4;++j) {
              __nv_bfloat162 value=__floats2bfloat162_rn(acc[g*8+j*2],acc[g*8+j*2+1]);
              packed[j]=*reinterpret_cast<uint32_t*>(&value);
            }
            int row=warp*16+(lane&7)+8*((lane>>3)&1);
            int d=feature+g*16+8*(lane>>4),offset=row*128+d;
            uint32_t sp=__cvta_generic_to_shared(temp+(offset^(((offset>>6)&7)<<3)));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0],{%1,%2,%3,%4};"::"r"(sp),"r"(packed[0]),"r"(packed[1]),"r"(packed[2]),"r"(packed[3]):"memory");
          }
        }
        // All consumers must finish this V feature tile before reuse.
        if(kind==1)__syncthreads();
      }
      if(cb!=13) {
        asm volatile("fence.proxy.async.shared::cta;":::"memory");
        __syncthreads();
        if(tid==0) {
          uint32_t p=__cvta_generic_to_shared(temp);
          const CUtensorMap* map=kind==0?&map_k:&map_v;
          asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%0, {%1,%2,%3,%4,%5}], [%6];"::"l"(map),"r"(0),"r"(kb*20),"r"(head),"r"(cb*6),"r"(qo),"r"(p):"memory");
          asm volatile("cp.async.bulk.commit_group;":::"memory");
        }
      }
    }
  }
'''+s[end:]
p=ROOT/'native_fisher/fisher_c10_n32_stream_v.cu';p.write_text(s);print(p)
