#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <math_constants.h>
#include <cstdint>
using bf16=__nv_bfloat16;
using namespace cute::SM90::GMMA;
#define E8(a) a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]
__device__ __forceinline__ int sw(int x){return x^(((x>>6)&7)<<3);}
__device__ __forceinline__ int qa(int r,int d){return sw(r*64+(d&63)+(d>>6)*8192);}
__device__ __forceinline__ int ka(int r,int d){return sw(r*64+(d&63)+(d>>6)*1024);}
__device__ __forceinline__ uint64_t desc(const bf16* p){
 auto s=__cvta_generic_to_shared(p);
 return ((s&0x3ffff)>>4)|(uint64_t(1)<<16)|(uint64_t(64)<<32)|(uint64_t(1)<<62);
}
__device__ __forceinline__ float ex2(float x){float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
__device__ __forceinline__ float lg2(float x){return log2f(x);}
__device__ __forceinline__ float sqroot(float x){float y;asm("sqrt.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
__device__ __forceinline__ float reciprocal(float x){float y;asm("div.full.f32 %0,0f3F800000,%1;":"=f"(y):"f"(x));return y;}
__device__ __forceinline__ float invsqrt(float x){float y;asm("rsqrt.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
__device__ __forceinline__ float sum16(float (&x)[16]){
 float a[4];
 #pragma unroll
 for(int i=0;i<4;++i)a[i]=__fadd_rn(__fadd_rn(__fadd_rn(x[2*i],x[2*i+1]),x[8+2*i]),x[9+2*i]);
 return __fadd_rn(__fadd_rn(a[0],a[2]),__fadd_rn(a[1],a[3]));
}
__device__ __forceinline__ float warp_sum(float x){
 #pragma unroll
 for(int d=8;d>=1;d/=2)x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,d));
 return __fadd_rn(x,__shfl_xor_sync(0xffffffff,x,16));
}
__device__ __forceinline__ float dot16(float (&x)[16],float (&y)[16]){
 float a[4];
 #pragma unroll
 for(int i=0;i<4;++i){
  a[i]=__fmul_rn(x[2*i+1],y[2*i+1]);
  a[i]=__fmaf_rn(x[2*i],y[2*i],a[i]);
  a[i]=__fmaf_rn(x[8+2*i],y[8+2*i],a[i]);
  a[i]=__fmaf_rn(x[9+2*i],y[9+2*i],a[i]);
 }
 return __fadd_rn(__fadd_rn(a[0],a[2]),__fadd_rn(a[1],a[3]));
}
__device__ __forceinline__ float variance_sum(float x){
 int lane=threadIdx.x&31;
 float lo=__shfl_sync(0xffffffff,x,lane&~8),hi=__shfl_sync(0xffffffff,x,lane|8);
 x=__fmaf_rn(lo,lo,__fmul_rn(hi,hi));
 #pragma unroll
 for(int d=4;d;d/=2)x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,d));
 return __fadd_rn(x,__shfl_xor_sync(0xffffffff,x,16));
}
__global__ __launch_bounds__(128,6) void gs_kernel(const bf16* __restrict__ q,const bf16* __restrict__ k,
 const int* __restrict__ probes,const int* __restrict__ candidates,const int* __restrict__ anchors,
 uint8_t* __restrict__ output,float alpha,float scale){
 extern __shared__ __align__(128) unsigned char raw[];
 bf16* sq=reinterpret_cast<bf16*>(raw);bf16* sk=sq+128*128;
 float* scores=reinterpret_cast<float*>(sk);
 int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
 int kr=blockIdx.x/20,group=blockIdx.x%20,head=blockIdx.y;
 for(int vec=tid;vec<128*16;vec+=128){
  int row=vec/16,d=(vec%16)*8,col=group*4+row/32;
  int candidate=candidates[kr*32+(row&31)];
  int64_t qi=((int64_t(candidate)*80+col)*40+head)*128+d;
  *reinterpret_cast<uint4*>(sq+qa(row,d))=*reinterpret_cast<const uint4*>(q+qi);
 }
 int candidate_row=candidates[kr*32+lane],anchor=anchors[kr];
 for(int ko=0;ko<21;++ko){
  for(int vec=tid;vec<16*16;vec+=128){
   int p=vec/16,d=(vec%16)*8;
   int probe=probes[(((ko*45+kr)*20+group)*40+head)*16+p];
   int64_t ki=(((int64_t(ko)*45+kr)*80+probe)*40+head)*128+d;
   *reinterpret_cast<uint4*>(sk+ka(p,d))=*reinterpret_cast<const uint4*>(k+ki);
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;":::"memory");
  asm volatile("wgmma.fence.sync.aligned;":::"memory");
  float a[8],b[8];
  using M=MMA_64x16x16_F32BF16BF16_SS<Major::K,Major::K>;
  #pragma unroll
  for(int d=0;d<128;d+=16){
   M::fma(desc(sq+qa(0,d)),desc(sk+ka(0,d)),E8(a),d==0?ScaleOut::Zero:ScaleOut::One);
   M::fma(desc(sq+qa(64,d)),desc(sk+ka(0,d)),E8(b),d==0?ScaleOut::Zero:ScaleOut::One);
  }
  asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
  #pragma unroll
  for(int i=0;i<8;++i){
   int row=warp*16+(lane>>2)+8*((i&3)>>1),p=(lane&3)*2+8*(i>>2)+(i&1);
   scores[(row>>5)*512+p*32+(row&31)]=a[i];
  }
  __syncthreads();
  float residual[16],work[16];float mx=-CUDART_INF_F;
  if(warp<2){
   #pragma unroll
   for(int p=0;p<16;++p)residual[p]=scores[warp*512+p*32+lane];
  }
  __syncthreads();
  #pragma unroll
  for(int i=0;i<8;++i){
   int row=warp*16+(lane>>2)+8*((i&3)>>1),p=(lane&3)*2+8*(i>>2)+(i&1);
   scores[(row>>5)*512+p*32+(row&31)]=b[i];
  }
  __syncthreads();
  if(warp>=2){
   #pragma unroll
   for(int p=0;p<16;++p)residual[p]=scores[(warp-2)*512+p*32+lane];
  }
  #pragma unroll
  for(int p=0;p<16;++p)mx=fmaxf(mx,__fmul_rn(residual[p],scale));
  #pragma unroll
  for(int p=0;p<16;++p)work[p]=ex2(__fmaf_rn(residual[p],scale,-mx));
  float denominator=sum16(work),inv=reciprocal(denominator),mass=__fadd_rn(mx,lg2(denominator));
  #pragma unroll
  for(int p=0;p<16;++p)residual[p]=sqroot(__fmul_rn(work[p],inv));
  float mean_profile=__fmul_rn(sum16(residual),0.0625f);
  #pragma unroll
  for(int p=0;p<16;++p)residual[p]=__fadd_rn(residual[p],-mean_profile);
  float mean=__fmul_rn(warp_sum(mass),0.03125f),deviation=__fadd_rn(mass,-mean);
  float variance=__fmul_rn(variance_sum(deviation),0.03125f);
  float z=__fmul_rn(deviation,invsqrt(__fadd_rn(variance,1.e-6f)));
  float weight=ex2(__fmul_rn(alpha,z));
  bool used=lane==anchor;int pick=anchor;
  #pragma unroll
  for(int p=0;p<16;++p)work[p]=__fmul_rn(residual[p],residual[p]);
  float norm2=dot16(residual,residual);
  int64_t oi=(((int64_t(ko)*45+kr)*80+group*4+warp)*40+head)*8;
  if(lane==0)output[oi]=candidates[kr*32+anchor];
  #pragma unroll
  for(int step=1;step<8;++step){
   float basis[16];
   #pragma unroll
   for(int p=0;p<16;++p)basis[p]=__shfl_sync(0xffffffff,residual[p],pick);
   float norm=invsqrt(__fadd_rn(__shfl_sync(0xffffffff,norm2,pick),1.e-12f));
   #pragma unroll
   for(int p=0;p<16;++p){basis[p]=__fmul_rn(basis[p],norm);work[p]=__fmul_rn(residual[p],basis[p]);}
   float projection=dot16(residual,basis);
   #pragma unroll
   for(int p=0;p<16;++p){residual[p]=__fmaf_rn(-projection,basis[p],residual[p]);work[p]=__fmul_rn(residual[p],residual[p]);}
   norm2=dot16(residual,residual);
   float best=used?-1.f:__fmul_rn(norm2,weight);pick=lane;
   if(__any_sync(0xffffffff,isnan(best))) {
    // Retain the existing comparison behavior for unordered values.
    #pragma unroll
    for(int d=16;d>=1;d/=2){
     float other=__shfl_xor_sync(0xffffffff,best,d);int op=__shfl_xor_sync(0xffffffff,pick,d);
     if(other>best || (other==best && op<pick)){best=other;pick=op;}
    }
   } else {
    // Unused candidates have a nonnegative squared norm times exp2 weight.
    // Exclude used candidates explicitly when the maximum is zero.
    uint32_t key=used?0u:__float_as_uint(best);
    uint32_t maximum=__reduce_max_sync(0xffffffff,key);
    pick=__ffs(__ballot_sync(0xffffffff,!used && key==maximum))-1;
   }
   used|=lane==pick;
   int selected=__shfl_sync(0xffffffff,candidate_row,pick);
   if(lane==0)output[oi+step]=selected;
  }
  __syncthreads();
 }
}
void run(torch::Tensor q,torch::Tensor k,torch::Tensor probes,torch::Tensor candidates,torch::Tensor anchors,torch::Tensor output,double alpha,double scale){
 TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.is_contiguous() && k.is_contiguous(),"Require CUDA contiguous BF16 inputs");
 TORCH_CHECK(q.sizes()==torch::IntArrayRef({1,1,45,80,40,128}) && k.sizes()==torch::IntArrayRef({1,21,45,80,40,128}),"Require native40 RF0 shape");
 TORCH_CHECK(candidates.scalar_type()==at::kInt && anchors.scalar_type()==at::kInt && probes.scalar_type()==at::kInt && output.scalar_type()==at::kByte,"Require int32 indices and uint8 seeds");
 TORCH_CHECK(candidates.numel()==45*32 && output.numel()==21*45*80*40*8,"Require 32 candidates and 8 seeds");
 constexpr int bytes=128*128*2+16*128*2;
 C10_CUDA_CHECK(cudaFuncSetAttribute(gs_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
 gs_kernel<<<dim3(45*20,40),128,bytes,at::cuda::getCurrentCUDAStream()>>>(
  (const bf16*)q.data_ptr(),(const bf16*)k.data_ptr(),probes.data_ptr<int>(),candidates.data_ptr<int>(),anchors.data_ptr<int>(),output.data_ptr<uint8_t>(),float(2*alpha*1.4426950408889634),float(scale*1.4426950408889634));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}
