#pragma once
#include "warp_mma.cuh"
#include <math_constants.h>

namespace fisher_warp {
#ifndef FISHER_WEIGHT_QUALIFIER
#define FISHER_WEIGHT_QUALIFIER __forceinline__
#endif
__device__ __forceinline__ float exp2_fast(float x) {
  float y; asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y;
}
template<int CAP> __device__ __forceinline__ float log2_native(float x) {
  if constexpr (CAP==4) return log2f(x);
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
__device__ __forceinline__ float reduce80(float (&p)[20]) {
  float main=p[0];
  #pragma unroll
  for(int i=1;i<16;++i)main=__fadd_rn(main,p[i]);
  float tail=p[16];
  #pragma unroll
  for(int i=17;i<20;++i)tail=__fadd_rn(tail,p[i]);
  return __fadd_rn(sum4(main),sum4(tail));
}
__device__ __forceinline__ float reciprocal(float x) {
  float y;asm("div.full.f32 %0,0f3F800000,%1;":"=f"(y):"f"(x));return y;
}
__device__ __forceinline__ float dot80(float (&p)[20],float (&x)[20]) {
  float a=__fmaf_rn(p[0],x[0],__fmul_rn(p[1],x[1]));
  #pragma unroll
  for(int i=2;i<16;++i)a=__fmaf_rn(p[i],x[i],a);
  float b=__fmaf_rn(p[16],x[16],__fmul_rn(p[17],x[17]));
  #pragma unroll
  for(int i=18;i<20;++i)b=__fmaf_rn(p[i],x[i],b);
  return __fadd_rn(sum4(a),sum4(b));
}

// One warp owns one column, sixteen physical score rows, CAP live seeds.
// Input Q rows are gathered and packed for this gate; the fused kernel will
// load the same fragments directly using seed-row indices.
struct GatheredSeedQuery {
  const bf16* pointer;
  __device__ __forceinline__ uint32_t operator()(int row,int offset) const {
    return *reinterpret_cast<const uint32_t*>(pointer+row*128+offset);
  }
};
template<int CAP,typename QueryLoader>
__device__ FISHER_WEIGHT_QUALIFIER void make_weights_loaded(QueryLoader query,
    const bf16* raw_k, float scale, int column, uint32_t (&weights)[20],
    float (&residual_scales)[2], float* bias) {
  int lane=threadIdx.x&31, seed=lane>>2, t=lane&3;
  float s[40];
  #pragma unroll
  for(int n=0;n<80;n+=8) {
    float c[4]={};
    #pragma unroll
    for(int d=0;d<128;d+=16) {
      int offset=(d/32)*32+t*8+(d%32)/4;
      uint32_t a[4],b[2];
      a[0]=query(seed,offset);
      a[1]=query(seed+8,offset);
      a[2]=query(seed,offset+2);
      a[3]=query(seed+8,offset+2);
      score_b(raw_k,n,d,lane,b);mma(a,b,c);
    }
    #pragma unroll
    for(int i=0;i<4;++i)s[n/2+i]=c[i];
  }
  float norm_max[2],norm_sum[2];
  #pragma unroll
  for(int half=0;half<2;++half) {
    float mx=-CUDART_INF_F;
    #pragma unroll
    for(int i=0;i<10;++i)mx=fmaxf(mx,fmaxf(__fmul_rn(s[4*i+2*half],scale),__fmul_rn(s[4*i+2*half+1],scale)));
    mx=max4(mx);
    float bs[20];
    #pragma unroll
    for(int i=0;i<20;++i)bs[i]=exp2_fast(__fmaf_rn(s[4*(i/2)+2*half+(i&1)],scale,-mx));
    norm_max[half]=mx;norm_sum[half]=reduce80(bs);
  }
  float norm_log=log2_native<CAP>((lane&1)?norm_sum[1]:norm_sum[0]);
  #pragma unroll
  for(int half=0;half<2;++half) {
    float norm=__fadd_rn(norm_max[half],__shfl_sync(0xffffffff,norm_log,(lane&~1)+half));
    #pragma unroll
    for(int i=0;i<20;++i) {
      int ix=(i/2)*4+half*2+(i&1);
      s[ix]=__fadd_rn(__fmul_rn(s[ix],scale),-norm);
    }
  }
  float winners[20];int owners[20];
  #pragma unroll
  for(int i=0;i<20;++i) {
    int ix=(i/2)*4+(i&1);
    float a=seed<CAP?s[ix]:-CUDART_INF_F;
    float b=seed+8<CAP?s[ix+2]:-CUDART_INF_F;
    float best=fmaxf(a,b);
    best=fmaxf(best,__shfl_xor_sync(0xffffffff,best,4));
    best=fmaxf(best,__shfl_xor_sync(0xffffffff,best,8));
    best=fmaxf(best,__shfl_xor_sync(0xffffffff,best,16));
    int owner=min(seed<CAP && a==best?seed:32,seed+8<CAP && b==best?seed+8:32);
    owner=min(owner,__shfl_xor_sync(0xffffffff,owner,4));
    owner=min(owner,__shfl_xor_sync(0xffffffff,owner,8));
    owner=min(owner,__shfl_xor_sync(0xffffffff,owner,16));
    winners[i]=best;owners[i]=owner;
  }
  #pragma unroll
  for(int half=0;half<2;++half) {
    int row=seed+8*half;
    float probabilities[20];
    #pragma unroll
    for(int i=0;i<20;++i)probabilities[i]=owners[i]==row?exp2_fast(winners[i]):0.f;
    float sum=reduce80(probabilities),safe=sum>0?sum:1.f;
    float inv=reciprocal(safe),expected=dot80(probabilities,winners),selected=0.f;
    #pragma unroll
    for(int i=0;i<20;++i) {
      probabilities[i]=__fmul_rn(probabilities[i],inv);
      if((i/2)*8+2*t+(i&1)==column)selected=probabilities[i];
    }
    float selected_weight=sum4(selected),residual=__fadd_rn(1.f,-selected_weight);
    bool valid=sum>0 && residual>1.e-6f;
    float safe_residual=valid?residual:1.f;
    residual_scales[half]=reciprocal(safe_residual);
    float argument=t==0?safe:t==1?fmaxf(selected_weight,1.e-20f):safe_residual;
    float logarithm=log2_native<CAP>(argument);
    float log_sum=__shfl_sync(0xffffffff,logarithm,lane&~3);
    float log_selected=__shfl_sync(0xffffffff,logarithm,(lane&~3)+1);
    float log_residual=__shfl_sync(0xffffffff,logarithm,(lane&~3)+2);
    float correction=__fmul_rn(__fadd_rn(__fmul_rn(expected,inv),-log_sum),0.6931471805599453f);
    float selected_entropy=__fmul_rn(__fmul_rn(selected_weight,log_selected),0.6931471805599453f);
    correction=__fadd_rn(__fmul_rn(__fadd_rn(correction,-selected_entropy),residual_scales[half]),-__fmul_rn(log_residual,0.6931471805599453f));
    if(t==0 && row<CAP)bias[row]=valid?correction:CUDART_INF_F;
    #pragma unroll
    for(int i=0;i<10;++i) {
      int ki=i*8+2*t;
      weights[2*i+half]=pair(__float2bfloat16_rn(ki==column?0.f:probabilities[2*i]),
                            __float2bfloat16_rn(ki+1==column?0.f:probabilities[2*i+1]));
    }
  }
}

template<int CAP>
__device__ __forceinline__ void make_weights(const bf16* packed_seed_q,
    const bf16* raw_k,float scale,int column,uint32_t (&weights)[20],
    float (&residual_scales)[2],float* bias) {
  make_weights_loaded<CAP>(GatheredSeedQuery{packed_seed_q},raw_k,scale,column,weights,residual_scales,bias);
}

// The fused path holds K accumulators until every producer has finished
// reading raw K, then safely reuses that shared storage for ring overflow.
__device__ __forceinline__ void merge_accumulators(const uint32_t (&weights)[20],
    const bf16* raw,float (&acc)[64]) {
  int lane=threadIdx.x&31;
  #pragma unroll
  for(int d=0;d<128;d+=8) {
    float c[4]={};
    #pragma unroll
    for(int ki=0;ki<80;ki+=16) {
      int wi=ki/4;
      uint32_t a[4]={weights[wi],weights[wi+1],weights[wi+2],weights[wi+3]},b[2];
      merge_b(raw,ki,d,lane,b);mma(a,b,c);
    }
    #pragma unroll
    for(int i=0;i<4;++i)acc[d/2+i]=c[i];
  }
}

template<int CAP>
__device__ __forceinline__ void merge_values(const uint32_t (&weights)[20],
    const float (&residual_scales)[2], const bf16* raw, bf16* out) {
  int lane=threadIdx.x&31,seed=lane>>2,t=lane&3;
  #pragma unroll
  for(int d=0;d<128;d+=8) {
    float c[4]={};
    #pragma unroll
    for(int ki=0;ki<80;ki+=16) {
      int wi=ki/4;
      uint32_t a[4]={weights[wi],weights[wi+1],weights[wi+2],weights[wi+3]},b[2];
      merge_b(raw,ki,d,lane,b);mma(a,b,c);
    }
    #pragma unroll
    for(int i=0;i<4;++i) {
      int row=seed+8*(i>>1);
      if(row<CAP)out[row*128+d+2*t+(i&1)]=__float2bfloat16_rn(__fmul_rn(c[i],residual_scales[i>>1]));
    }
  }
}
} // namespace fisher_warp
