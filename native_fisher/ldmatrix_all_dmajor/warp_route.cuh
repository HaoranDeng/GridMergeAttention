#pragma once
#include "warp_merge.cuh"

namespace fisher_warp {
#ifndef FISHER_ROUTE_QUALIFIER
#define FISHER_ROUTE_QUALIFIER __forceinline__
#endif
__device__ __forceinline__ int ring_offset(int row,int d) {
  return row*128+(d^((row&7)*8));
}

struct RouteState {
  float acc[64];
  float maximum[2];
  float sum[2];
  __device__ __forceinline__ void clear() {
    #pragma unroll
    for(int i=0;i<64;++i)acc[i]=0.f;
    maximum[0]=maximum[1]=-CUDART_INF_F;sum[0]=sum[1]=0.f;
  }
};

__device__ __forceinline__ float sum64(float (&x)[16]) {
  // Balanced local tree, then the four lanes that hold one query row.
  #pragma unroll
  for(int stride=8;stride>0;stride/=2) {
    #pragma unroll
    for(int i=0;i<stride;++i)x[i]=__fadd_rn(x[i],x[i+stride]);
  }
  return sum4(x[0]);
}

// Consume one already-padded ring. Invalid K/V rows must contain zero and
// bias +inf. Q uses the retained lossless 32-dimension packing permutation.
// row_stride permits direct reads of one column in native [frame,row,col,H,D].
__device__ FISHER_ROUTE_QUALIFIER void consume_ring(const bf16* packed_q,
    int64_t row_stride,int valid_rows,const bf16* rk,const bf16* rv,
    const float* bias,float scale,RouteState& state) {
  int lane=threadIdx.x&31,g=lane>>2,t=lane&3;
  // Accumulate all eight key panels one feature panel at a time. Each score
  // keeps the same K-dimension MMA order, but only four Q words stay live.
  float scores[32]={};
  #pragma unroll 1
  for(int d=0;d<128;d+=16) {
    int offset=(d/32)*32+t*8+(d%32)/4;
    uint32_t a[4];
    a[0]=g<valid_rows?*reinterpret_cast<const uint32_t*>(packed_q+g*row_stride+offset):0;
    a[1]=g+8<valid_rows?*reinterpret_cast<const uint32_t*>(packed_q+(g+8)*row_stride+offset):0;
    a[2]=g<valid_rows?*reinterpret_cast<const uint32_t*>(packed_q+g*row_stride+offset+2):0;
    a[3]=g+8<valid_rows?*reinterpret_cast<const uint32_t*>(packed_q+(g+8)*row_stride+offset+2):0;
    #pragma unroll
    for(int n=0;n<64;n+=8) {
      float c[4]={scores[n/2],scores[n/2+1],scores[n/2+2],scores[n/2+3]};
      uint32_t b[2];
      uint32_t address=__cvta_generic_to_shared(rk+ring_offset(n+(lane&7),d+8*((lane>>3)&1)));
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
        :"=r"(b[0]),"=r"(b[1]):"r"(address):"memory");
      mma(a,b,c);
      #pragma unroll
      for(int i=0;i<4;++i)scores[n/2+i]=c[i];
    }
  }
  #pragma unroll
  for(int n=0;n<64;n+=8) {
    #pragma unroll
    for(int i=0;i<4;++i) {
      float correction=bias[n+2*t+(i&1)];
      scores[n/2+i]=__fmul_rn(__fadd_rn(__fmul_rn(scores[n/2+i],scale),-correction),1.4426950408889634f);
    }
  }
  uint32_t probability_words[16];float old_scale[2];
  #pragma unroll
  for(int half=0;half<2;++half) {
    float mx=-CUDART_INF_F;
    #pragma unroll
    for(int i=0;i<16;++i)mx=fmaxf(mx,scores[(i/2)*4+half*2+(i&1)]);
    mx=fmaxf(state.maximum[half],max4(mx));
    bool nonempty=mx!=-CUDART_INF_F;
    old_scale[half]=nonempty?exp2_fast(state.maximum[half]-mx):0.f;
    float p[16];
    #pragma unroll
    for(int i=0;i<16;++i)p[i]=nonempty?exp2_fast(scores[(i/2)*4+half*2+(i&1)]-mx):0.f;
    #pragma unroll
    for(int i=0;i<8;++i)
      probability_words[2*i+half]=pair(__float2bfloat16_rn(p[2*i]),__float2bfloat16_rn(p[2*i+1]));
    state.sum[half]=__fadd_rn(__fmul_rn(state.sum[half],old_scale[half]),sum64(p));
    state.maximum[half]=mx;
  }
  #pragma unroll
  for(int d=0;d<128;d+=8) {
    float c[4]={};
    #pragma unroll
    for(int k=0;k<64;k+=16) {
      int key=k+2*t,feature=d+g,ix=k/4;
      uint32_t a[4]={probability_words[ix],probability_words[ix+1],probability_words[ix+2],probability_words[ix+3]};
      uint32_t b[2];
      uint32_t address=__cvta_generic_to_shared(rv+ring_offset(k+(lane&15),d));
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
        :"=r"(b[0]),"=r"(b[1]):"r"(address):"memory");
      mma(a,b,c);
    }
    #pragma unroll
    for(int i=0;i<4;++i)state.acc[d/2+i]=__fadd_rn(__fmul_rn(state.acc[d/2+i],old_scale[i>>1]),c[i]);
  }
}
} // namespace fisher_warp
