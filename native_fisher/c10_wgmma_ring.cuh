#pragma once
// Derived from C4 SQ1 prototype and validated C8 CUDA. Native ten-seed storage, no padded route tokens.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cuda.h>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/arch/mma_sm90_gmma_ext.hpp>
#include <cstdint>

namespace c10_ring {
using bf16 = __nv_bfloat16;
using namespace cute::SM90::GMMA;

#define E8(a,o) a[o],a[o+1],a[o+2],a[o+3],a[o+4],a[o+5],a[o+6],a[o+7]
#define E32(a) E8(a,0),E8(a,8),E8(a,16),E8(a,24)
#define E40(a) E32(a),E8(a,32)
#define E64(a) E32(a),E8(a,32),E8(a,40),E8(a,48),E8(a,56)

__device__ __forceinline__ uint64_t desc(const bf16* p,int lead,int stride) {
  uint64_t s=__cvta_generic_to_shared(p);
  return ((s&0x3ffff)>>4) | (uint64_t(lead>>4)<<16) | (uint64_t(stride>>4)<<32) | (uint64_t(1)<<62);
}
__device__ __forceinline__ int aoff(int row,int k) {
  int offset=row*64+(k&63)+(k>>6)*4096;
  return offset^(((offset>>6)&7)<<3);
}
__device__ __forceinline__ int boff(int k,int n) {
  int offset=k*64+(n&63)+(n>>6)*5120;
  return offset^(((offset>>6)&7)<<3);
}
__device__ __forceinline__ int col_for_row(int row) {return row/10;}
__device__ __forceinline__ int seed_for_row(int row) {return row%10;}
__device__ __forceinline__ int storeoff(int row,int d) {return row*128+(d^((row&7)*8));}
__device__ __forceinline__ uint32_t load_query(const bf16* p,uint64_t policy) {
  uint32_t value;
  asm volatile("ld.global.L1::evict_last.L2::cache_hint.b32 %0,[%1],%2;":"=r"(value):"l"(p),"l"(policy));
  return value;
}
__device__ __forceinline__ uint4 query_vector(const bf16* p,uint64_t policy,int lane) {
  uint4 a;
  asm volatile("ld.global.L1::evict_last.L2::cache_hint.v4.b32 {%0,%1,%2,%3},[%4],%5;":"=r"(a.x),"=r"(a.y),"=r"(a.z),"=r"(a.w):"l"(p),"l"(policy));
  return a;
}
__device__ __forceinline__ uint4 fragment_transpose(uint4 a,int lane) {
  uint32_t t0=__shfl_xor_sync(0xffffffff,(lane&1)?a.x:a.y,1);
  uint32_t t1=__shfl_xor_sync(0xffffffff,(lane&1)?a.z:a.w,1);
  uint4 b={(lane&1)?t0:a.x,(lane&1)?a.y:t0,(lane&1)?t1:a.z,(lane&1)?a.w:t1};
  t0=__shfl_xor_sync(0xffffffff,(lane&2)?b.x:b.z,2);
  t1=__shfl_xor_sync(0xffffffff,(lane&2)?b.y:b.w,2);
  return {(lane&2)?t0:b.x,(lane&2)?t1:b.y,(lane&2)?b.z:t0,(lane&2)?b.w:t1};
}
__device__ __forceinline__ void store_output(bf16* p,uint4 value,uint64_t policy) {
  asm volatile("st.global.L1::evict_first.L2::cache_hint.v4.b32 [%0],{%1,%2,%3,%4},%5;"::"l"(p),"r"(value.x),"r"(value.y),"r"(value.z),"r"(value.w),"l"(policy):"memory");
}
__device__ __forceinline__ float ex2(float x) {
  float y; asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y;
}
__device__ __forceinline__ float lg2(float x) {
  // Call sites supply positive normal floats: sums of FTZ exp2 values
  // (or one), singleton weights clamped to 1e-20, and residuals >1e-6.
  // Explicit SFU
  // avoids libdevice's subnormal path; its small rounding difference is
  // checked in complete real-QKV calls, not assumed bitwise equivalent.
  float y; asm("lg2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y;
}
__device__ __forceinline__ float sum4(float x) {
  x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,2,4));
  return __fadd_rn(x,__shfl_xor_sync(0xffffffff,x,1,4));
}
__device__ __forceinline__ float max4(float x) {
  x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,1,4));
  return fmaxf(x,__shfl_xor_sync(0xffffffff,x,2,4));
}
__device__ __forceinline__ float tree80(float (&p)[10]) {
  float main=__fadd_rn(__fadd_rn(__fadd_rn(p[0],p[1]),__fadd_rn(p[2],p[3])),
                       __fadd_rn(__fadd_rn(p[4],p[5]),__fadd_rn(p[6],p[7])));
  return __fadd_rn(sum4(main),sum4(__fadd_rn(p[8],p[9])));
}
__device__ __forceinline__ float reduce80(float (&p)[20]) {
  float main=p[0];
  #pragma unroll
  for(int i=1;i<16;++i) main=__fadd_rn(main,p[i]);
  float tail=p[16];
  #pragma unroll
  for(int i=17;i<20;++i) tail=__fadd_rn(tail,p[i]);
  return __fadd_rn(sum4(main),sum4(tail));
}

__device__ __forceinline__ float reciprocal(float x){float y;asm("div.full.f32 %0,0f3F800000,%1;":"=f"(y):"f"(x));return y;}
__device__ __forceinline__ float dot80(float (&p)[20],float (&x)[20]){
 float a=__fmaf_rn(p[0],x[0],__fmul_rn(p[1],x[1]));
 #pragma unroll
 for(int i=2;i<16;++i)a=__fmaf_rn(p[i],x[i],a);
 float b=__fmaf_rn(p[16],x[16],__fmul_rn(p[17],x[17]));
 #pragma unroll
 for(int i=18;i<20;++i)b=__fmaf_rn(p[i],x[i],b);
 return __fadd_rn(sum4(a),sum4(b));
}
// K/V stay resident across query tiles. All score, winner, and cluster
// reductions remain in registers. Query, weights, and store staging alias
// the same shared buffer after their respective WGMMA operations finish.

constexpr int COLUMNS=5,RING=16;
constexpr int SHARED_BYTES=(2*80*128+2*COLUMNS*RING*128)*2+COLUMNS*RING*4;
__device__ __forceinline__ void producer_barrier(){asm volatile("bar.sync 1, 128;":::"memory");}
__device__ __forceinline__ int ring_offset(int row,int d){return row*128+(d^((row&7)*8));}
__device__ __forceinline__ void load_raw(const bf16* k,const bf16* v,bf16* sk,bf16* sv,int kb,int gh){
  for(int vec=threadIdx.x;vec<80*16;vec+=128){
    int ki=vec/16,d=(vec%16)*8;
    int64_t src=((int64_t(kb)*80+ki)*40+gh)*128+d;
    *reinterpret_cast<uint4*>(sk+boff(ki,d))=*reinterpret_cast<const uint4*>(k+src);
    *reinterpret_cast<uint4*>(sv+boff(ki,d))=*reinterpret_cast<const uint4*>(v+src);
  }
  producer_barrier();
  asm volatile("fence.proxy.async.shared::cta;":::"memory");
}
__device__ __forceinline__ void produce(const bf16* q,const uint8_t* seeds,
    const bf16* sk,const bf16* sv,bf16* mk,bf16* mv,float* bias,
    int qo,int column_base,int kb,int gh,float scale){
  constexpr int KO=21,R=45,C=80,H=40;
  int tid=threadIdx.x,lane=tid&31,warp=tid>>5,seed=lane>>2;
  int ko=kb/45,kr=kb%45,rf=0;
  bf16* temp=mk;
  uint64_t cache_last;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0,1.0;":"=l"(cache_last));
    int row0=warp*16+seed,col0=column_base+col_for_row(row0);
    int si0=((((rf*KO+ko)*R+kr)*C+min(col0,79))*H+gh)*10+seed_for_row(row0);
    int col1=column_base+col_for_row(row0+8);
    int si1=((((rf*KO+ko)*R+kr)*C+min(col1,79))*H+gh)*10+seed_for_row(row0+8);
    int sr0=seeds[si0],sr1=seeds[si1];
    int qi0=(((qo*R+sr0)*C+min(col0,79))*H+gh)*128+(lane&3)*8;
    int qi1=(((qo*R+sr1)*C+min(col1,79))*H+gh)*128+(lane&3)*8;
    uint32_t qreg[32];
    #pragma unroll
    for(int d=0;d<128;d+=32) {
      uint4 a=query_vector(q+qi0+d,cache_last,lane),b=query_vector(q+qi1+d,cache_last,lane);
      int i=d/4;
      qreg[i]=a.x;qreg[i+1]=b.x;qreg[i+2]=a.y;qreg[i+3]=b.y;
      qreg[i+4]=a.z;qreg[i+5]=b.z;qreg[i+6]=a.w;qreg[i+7]=b.w;
    }
    #pragma unroll
    for(int i=0;i<32;++i) asm volatile("" : "+r"(qreg[i]) :: "memory");
    float s[40];
    asm volatile("wgmma.fence.sync.aligned;":::"memory");
    using Score=MMA_64x80x16_F32BF16BF16_RS<Major::K,Major::K>;
    #pragma unroll
    for(int d=0;d<128;d+=16) {
      int i=d/4;
      Score::fma(qreg[i],qreg[i+1],qreg[i+2],qreg[i+3],desc(sk+boff(0,d),16,1024),E40(s),d==0?ScaleOut::Zero:ScaleOut::One);
    }
    asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
    float norm_max[2],norm_sum[2];
    #pragma unroll
    for(int half=0;half<2;++half) {
      float mx=-CUDART_INF_F;
      #pragma unroll
      for(int i=0;i<10;++i) mx=fmaxf(mx,fmaxf(__fmul_rn(s[4*i+2*half],scale),__fmul_rn(s[4*i+2*half+1],scale)));
      mx=max4(mx);
      float bs[20];
      #pragma unroll
      for(int i=0;i<20;++i)
        bs[i]=ex2(__fmaf_rn(s[4*(i/2)+2*half+(i&1)],scale,-mx));
      norm_max[half]=mx;norm_sum[half]=reduce80(bs);
    }
    // Each four-lane row group has the same reduced sums. Compute its two
    // independent logarithms in alternating lanes, then broadcast each.
    float norm_log=lg2((lane&1)?norm_sum[1]:norm_sum[0]);
    #pragma unroll
    for(int half=0;half<2;++half) {
      float norm=__fadd_rn(norm_max[half],__shfl_sync(0xffffffff,norm_log,(lane&~1)+half));
      #pragma unroll
      for(int i=0;i<20;++i) {
        int ix=(i/2)*4+half*2+(i&1);
        s[ix]=__fadd_rn(__fmul_rn(s[ix],scale),-norm);
      }
    }
    // Sixty live rows represent six columns with ten seeds each. Four
    // unused tensor-core rows never contribute a seed or output token.
    // Normalize each seed exactly as before, then reduce ten scores per
    // key in shared memory without rounding the score or changing ties.
    // Forty-key owner tiles fit below the existing 16 KiB output staging
    // footprint. Four CTAs can now share an SM without changing live C10.
    float* normalized=reinterpret_cast<float*>(temp);
    float* selected_winners=normalized+64*40;
    int* selected_owners=reinterpret_cast<int*>(selected_winners+6*40);
    float mass_main[2],mass_tail[2],entropy_main[2],entropy_tail[2];
    producer_barrier();
    #pragma unroll
    for(int chunk=0;chunk<2;++chunk) {
      #pragma unroll
      for(int i=0;i<20;++i) {
        int row=warp*16+seed+8*((i&3)>>1);
        int ki=2*(lane&3)+8*(i>>2)+(i&1);
        normalized[row*40+ki]=s[chunk*20+i];
      }
      producer_barrier();
      for(int index=tid;index<6*40;index+=128) {
        int column=index/40,ki=index%40;
        float z0=normalized[(column*10+0)*40+ki];
        float b0=normalized[(column*10+1)*40+ki];
        int i0=0;
        if(b0>z0){z0=b0;i0=1;}
        float z1=normalized[(column*10+2)*40+ki];
        float b1=normalized[(column*10+3)*40+ki];
        int i1=2;
        if(b1>z1){z1=b1;i1=3;}
        float z2=normalized[(column*10+4)*40+ki];
        float b2=normalized[(column*10+5)*40+ki];
        int i2=4;
        if(b2>z2){z2=b2;i2=5;}
        float z3=normalized[(column*10+6)*40+ki];
        float b3=normalized[(column*10+7)*40+ki];
        int i3=6;
        if(b3>z3){z3=b3;i3=7;}
        float z4=normalized[(column*10+8)*40+ki];
        float b4=normalized[(column*10+9)*40+ki];
        int i4=8;
        if(b4>z4){z4=b4;i4=9;}
        if(z1>z0){z0=z1;i0=i1;}
        if(z3>z2){z2=z3;i2=i3;}
        if(z2>z0){z0=z2;i0=i2;}
        if(z4>z0){z0=z4;i0=i4;}
        float best=z0;int owner=i0;
        selected_winners[index]=best;selected_owners[index]=owner;
      }
      producer_barrier();
      #pragma unroll
      for(int half=0;half<2;++half) {
        int row=warp*16+seed+8*half,owner=seed_for_row(row);
        #pragma unroll
        for(int pair=0;pair<5;++pair) {
          int index=min(col_for_row(row),5)*40+pair*8+2*(lane&3);
          float x0=selected_winners[index],x1=selected_winners[index+1];
          float p0=selected_owners[index]==owner?ex2(x0):0.f;
          float p1=selected_owners[index+1]==owner?ex2(x1):0.f;
          s[chunk*20+pair*4+half*2]=p0;
          s[chunk*20+pair*4+half*2+1]=p1;
          if(chunk==0 && pair==0) {
            mass_main[half]=__fadd_rn(p0,p1);
            entropy_main[half]=__fmaf_rn(p0,x0,__fmul_rn(p1,x1));
          } else if(chunk==0 || pair<3) {
            mass_main[half]=__fadd_rn(__fadd_rn(mass_main[half],p0),p1);
            entropy_main[half]=__fmaf_rn(p1,x1,__fmaf_rn(p0,x0,entropy_main[half]));
          } else if(pair==3) {
            mass_tail[half]=__fadd_rn(p0,p1);
            entropy_tail[half]=__fmaf_rn(p0,x0,__fmul_rn(p1,x1));
          } else {
            mass_tail[half]=__fadd_rn(__fadd_rn(mass_tail[half],p0),p1);
            entropy_tail[half]=__fmaf_rn(p1,x1,__fmaf_rn(p0,x0,entropy_tail[half]));
          }
        }
      }
      producer_barrier();
    }
    uint32_t packed_weights[20];
    float residual_scales[2];
    #pragma unroll
    for(int half=0;half<2;++half) {
      float probabilities[20];
      int row=warp*16+seed+8*half,col=column_base+col_for_row(row);
      int owner=seed_for_row(row);
      #pragma unroll
      for(int i=0;i<20;++i)probabilities[i]=s[(i/2)*4+half*2+(i&1)];
      float sum=__fadd_rn(sum4(mass_main[half]),sum4(mass_tail[half]));
      float safe=sum>0?sum:1.f;
      float inv=reciprocal(safe);
      float expected=__fadd_rn(sum4(entropy_main[half]),sum4(entropy_tail[half]));
      float selected=0.f;
      #pragma unroll
      for(int i=0;i<20;++i) {
        probabilities[i]=__fmul_rn(probabilities[i],inv);
        int ki=(i/2)*8+2*(lane&3)+(i&1);
        if(ki==col)selected=probabilities[i];
      }
      float selected_weight=sum4(selected);
      float residual=__fadd_rn(1.f,-selected_weight);
      bool valid=sum>0 && residual>1.e-6f;
      float safe_residual=valid?residual:1.f;
      residual_scales[half]=reciprocal(safe_residual);
      float log_argument=(lane&3)==0?safe:(lane&3)==1?fmaxf(selected_weight,1.e-20f):safe_residual;
      float logarithm=lg2(log_argument);
      float log_sum=__shfl_sync(0xffffffff,logarithm,lane&~3);
      float log_selected=__shfl_sync(0xffffffff,logarithm,(lane&~3)+1);
      float log_residual=__shfl_sync(0xffffffff,logarithm,(lane&~3)+2);
      float correction=__fmul_rn(__fadd_rn(__fmul_rn(expected,inv),-log_sum),0.6931471805599453f);
      float selected_entropy=__fmul_rn(__fmul_rn(selected_weight,log_selected),0.6931471805599453f);
      correction=__fadd_rn(__fmul_rn(__fadd_rn(correction,-selected_entropy),residual_scales[half]),-__fmul_rn(log_residual,0.6931471805599453f));
      int out=col_for_row(row)*16+owner;
      if((lane&3)==0 && row<50 && col<80)bias[out]=valid?correction:CUDART_INF_F;
      #pragma unroll
      for(int i=0;i<10;++i) {
        int ki=(i/1)*8+2*(lane&3);
        union { bf16 values[2]; uint32_t bits; } pair;
        pair.values[0]=__float2bfloat16_rn(ki==col?0.f:probabilities[2*i]);
        pair.values[1]=__float2bfloat16_rn(ki+1==col?0.f:probabilities[2*i+1]);
        packed_weights[2*i+half]=pair.bits;
      }
    }
    producer_barrier(); // All weights now reside in registers; shared scores may be reused.

  // Owner scratch aliases MK. Reinitialize only the six non-cluster slots
  // after every producer lane has finished using scratch and packed weights.
  for(int vec=tid;vec<COLUMNS*6*16;vec+=128){
    int col=vec/(6*16),row=10+(vec/16)%6,d=(vec%16)*8;
    uint4 x={0,0,0,0},y=x;
    if(row==10){
      x=*reinterpret_cast<const uint4*>(sk+boff(column_base+col,d));
      y=*reinterpret_cast<const uint4*>(sv+boff(column_base+col,d));
    }
    *reinterpret_cast<uint4*>(mk+ring_offset(col*16+row,d))=x;
    *reinterpret_cast<uint4*>(mv+ring_offset(col*16+row,d))=y;
  }
  if(tid<COLUMNS*6){int col=tid/6,row=10+tid%6;bias[col*16+row]=row==10?0.f:CUDART_INF_F;}
  #pragma unroll
  for(int kind=0;kind<2;++kind){
    asm volatile("wgmma.fence.sync.aligned;":::"memory");
    float acc[64];
    using Merge=MMA_64x128x16_F32BF16BF16_RS<Major::K,Major::MN>;
    #pragma unroll
    for(int ki=0;ki<80;ki+=16){
      uint64_t bd=desc((kind==0?sk:sv)+boff(ki,0),10240,1024);
      int wi=(ki/8)*2;
      Merge::fma(packed_weights[wi],packed_weights[wi+1],packed_weights[wi+2],packed_weights[wi+3],bd,E64(acc),ki==0?ScaleOut::Zero:ScaleOut::One);
    }
    asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
    bf16* dst=kind==0?mk:mv;
    #pragma unroll
    for(int i=0;i<64;i+=2){
      int row=warp*16+seed+8*((i&3)>>1),d=2*(lane&3)+8*(i>>2);
      __nv_bfloat162 pair=__floats2bfloat162_rn(__fmul_rn(acc[i],residual_scales[(i&3)>>1]),__fmul_rn(acc[i+1],residual_scales[(i&3)>>1]));
      if(row<50)*reinterpret_cast<__nv_bfloat162*>(dst+ring_offset(col_for_row(row)*16+seed_for_row(row),d))=pair;
    }
  }
}
} // namespace c10_ring
