#pragma once
#include "c10_wgmma_ring.cuh"

namespace c10_ring {
struct RouteState {
  float acc[64],maximum[2],denominator[2];
  __device__ __forceinline__ void clear(){
    #pragma unroll
    for(int i=0;i<64;++i)acc[i]=0.f;
    maximum[0]=maximum[1]=-CUDART_INF_F;
    denominator[0]=denominator[1]=0.f;
  }
};
__device__ __forceinline__ void warp_mma(const uint32_t (&a)[4],const uint32_t (&b)[2],float (&c)[4]){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
__device__ __forceinline__ uint32_t pack_prob(float a,float b){
  __nv_bfloat162 value=__floats2bfloat162_rn(a,b);return *reinterpret_cast<uint32_t*>(&value);
}
// One warp covers sixteen query rows and all sixteen ring slots. The Q
// fragments use the retained lossless packing; only four words stay live.
__device__ __forceinline__ void consume(const bf16* q,int valid_rows,const bf16* mk,const bf16* mv,
    const float* bias,RouteState& state){
  constexpr int64_t ROW_STRIDE=80*40*128;
  constexpr float SCALE=0.08838834764831845f,LOG2E=1.4426950408889634f;
  int lane=threadIdx.x&31,g=lane>>2,t=lane&3;
  float scores[8]={};
  #pragma unroll 1
  for(int d=0;d<128;d+=16){
    int off=(d/32)*32+t*8+(d%32)/4;
    uint32_t a[4]={g<valid_rows?*reinterpret_cast<const uint32_t*>(q+g*ROW_STRIDE+off):0,
      g+8<valid_rows?*reinterpret_cast<const uint32_t*>(q+(g+8)*ROW_STRIDE+off):0,
      g<valid_rows?*reinterpret_cast<const uint32_t*>(q+g*ROW_STRIDE+off+2):0,
      g+8<valid_rows?*reinterpret_cast<const uint32_t*>(q+(g+8)*ROW_STRIDE+off+2):0};
    #pragma unroll
    for(int n=0;n<16;n+=8){
      uint32_t b[2];
      uint32_t address=__cvta_generic_to_shared(mk+ring_offset(n+(lane&7),d+8*((lane>>3)&1)));
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];":"=r"(b[0]),"=r"(b[1]):"r"(address):"memory");
      float c[4]={scores[n/2],scores[n/2+1],scores[n/2+2],scores[n/2+3]};
      warp_mma(a,b,c);
      #pragma unroll
      for(int i=0;i<4;++i)scores[n/2+i]=c[i];
    }
  }
  #pragma unroll
  for(int n=0;n<16;n+=8){
    #pragma unroll
    for(int i=0;i<4;++i)scores[n/2+i]=__fmul_rn(__fadd_rn(__fmul_rn(scores[n/2+i],SCALE),-bias[n+2*t+(i&1)]),LOG2E);
  }
  uint32_t probability[4];float old[2];
  #pragma unroll
  for(int half=0;half<2;++half){
    float p[4];
    float m=-CUDART_INF_F;
    #pragma unroll
    for(int i=0;i<4;++i)m=fmaxf(m,scores[(i/2)*4+half*2+(i&1)]);
    m=fmaxf(state.maximum[half],max4(m));
    bool live=m!=-CUDART_INF_F;
    old[half]=live?ex2(state.maximum[half]-m):0.f;
    #pragma unroll
    for(int i=0;i<4;++i)p[i]=live?ex2(scores[(i/2)*4+half*2+(i&1)]-m):0.f;
    probability[half]=pack_prob(p[0],p[1]);probability[2+half]=pack_prob(p[2],p[3]);
    float sum=sum4(__fadd_rn(__fadd_rn(p[0],p[1]),__fadd_rn(p[2],p[3])));
    state.denominator[half]=__fadd_rn(__fmul_rn(state.denominator[half],old[half]),sum);
    state.maximum[half]=m;
  }
  #pragma unroll
  for(int d=0;d<128;d+=8){
    uint32_t b[2];float c[4]={};
    uint32_t address=__cvta_generic_to_shared(mv+ring_offset(lane&15,d));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];":"=r"(b[0]),"=r"(b[1]):"r"(address):"memory");
    warp_mma(probability,b,c);
    #pragma unroll
    for(int i=0;i<4;++i)state.acc[d/2+i]=__fadd_rn(__fmul_rn(state.acc[d/2+i],old[i>>1]),c[i]);
  }
}
} // namespace c10_ring
