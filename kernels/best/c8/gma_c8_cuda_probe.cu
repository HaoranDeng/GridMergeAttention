#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cuda.h>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/arch/mma_sm90_gmma_ext.hpp>
#include <cstdint>

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
__device__ __forceinline__ int col_for_row(int row) {return (row>>4)*2+(row&1);}
__device__ __forceinline__ int seed_for_row(int row) {return (row&15)>>1;}
__device__ __forceinline__ int storeoff(int row,int d) {return row*128+(d^((row&7)*8));}
__device__ __forceinline__ uint32_t load_query(const bf16* p,uint64_t policy) {
  uint32_t value;
  asm volatile("ld.global.L1::evict_last.L2::cache_hint.b32 %0,[%1],%2;":"=r"(value):"l"(p),"l"(policy));
  return value;
}
__device__ __forceinline__ void store_output(bf16* p,uint4 value,uint64_t policy) {
  asm volatile("st.global.L1::evict_first.L2::cache_hint.v4.b32 [%0],{%1,%2,%3,%4},%5;"::"l"(p),"r"(value.x),"r"(value.y),"r"(value.z),"r"(value.w),"l"(policy):"memory");
}
__device__ __forceinline__ float ex2(float x) {
  float y; asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y;
}
__device__ __forceinline__ float lg2(float x) {
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


__device__ __forceinline__ float warp_max(float v){
 #pragma unroll
 for(int d=16;d;d/=2)v=fmaxf(v,__shfl_xor_sync(0xffffffff,v,d));
 return v;
}
__device__ __forceinline__ float warp_sum(float v){
 #pragma unroll
 for(int d=16;d;d/=2)v=__fadd_rn(v,__shfl_xor_sync(0xffffffff,v,d));
 return v;
}
__global__ __launch_bounds__(128,5) void probe_kernel(const bf16* q,const bf16* k,int* output,float scale){
 extern __shared__ __align__(128) unsigned char raw[];
 bf16* sk=(bf16*)raw;bf16* aq=sk+80*128;float* score=(float*)aq;
 int tid=threadIdx.x,lane=tid&31,warp=tid>>5,head=blockIdx.y,kb=blockIdx.x;
 int kr=kb%45,ko=kb/45;
 for(int vec=tid;vec<80*16;vec+=128){
  int row=vec/16,d=(vec%16)*8;
  *reinterpret_cast<uint4*>(sk+boff(row,d))=*reinterpret_cast<const uint4*>(k+((int64_t(kb)*80+row)*40+head)*128+d);
 }
 for(int vec=tid;vec<64*16;vec+=128){
  int row=vec/16,d=(vec%16)*8,group=row/2;
  int qr=min(44,max(0,kr+((row&1)?1:-1)));uint4 x{};
  if(group<20)x=*reinterpret_cast<const uint4*>(q+((int64_t(qr)*80+group*4)*40+head)*128+d);
  *reinterpret_cast<uint4*>(aq+aoff(row,d))=x;
 }
 __syncthreads();asm volatile("fence.proxy.async.shared::cta;":::"memory");
 asm volatile("wgmma.fence.sync.aligned;":::"memory");
 float acc[40];using Score=MMA_64x80x16_F32BF16BF16_SS<Major::K,Major::K>;
 #pragma unroll
 for(int d=0;d<128;d+=16)Score::fma(desc(aq+aoff(0,d),16,1024),desc(sk+boff(0,d),16,1024),E40(acc),d==0?ScaleOut::Zero:ScaleOut::One);
 asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
 asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
 __syncthreads();
 #pragma unroll
 for(int i=0;i<40;++i){
  int row=warp*16+(lane>>2)+8*((i&3)>>1),col=2*(lane&3)+8*(i>>2)+(i&1);
  score[row*80+col]=__fmul_rn(acc[i],scale);
 }
 __syncthreads();
 for(int group=warp;group<20;group+=4){
  float z[2][3],norm[2],work[3];
  #pragma unroll
  for(int a=0;a<2;++a){
   float mx=-CUDART_INF_F;
   #pragma unroll
   for(int j=0;j<3;++j){int col=lane+j*32;z[a][j]=col<80?score[(group*2+a)*80+col]:-CUDART_INF_F;mx=fmaxf(mx,z[a][j]);}
   mx=warp_max(mx);float sum=0;
   #pragma unroll
   for(int j=0;j<3;++j)sum=__fadd_rn(sum,ex2(__fadd_rn(z[a][j],-mx)));
   norm[a]=__fadd_rn(mx,lg2(warp_sum(sum)));
  }
  #pragma unroll
  for(int j=0;j<3;++j){int col=lane+j*32;work[j]=col<80&&col%10!=0?fmaxf(__fadd_rn(z[0][j],-norm[0]),__fadd_rn(z[1][j],-norm[1])):-CUDART_INF_F;}
  int64_t oi=((int64_t(kb)*20+group)*40+head)*16;
  if(lane<8)output[oi+8+lane]=lane*10;
  #pragma unroll
  for(int rank=0;rank<8;++rank){
   float best=-CUDART_INF_F;int pick=INT_MAX;
   #pragma unroll
   for(int j=0;j<3;++j){int col=lane+j*32;if(work[j]>best||(work[j]==best&&col<pick)){best=work[j];pick=col;}}
   #pragma unroll
   for(int d=16;d;d/=2){float other=__shfl_xor_sync(0xffffffff,best,d);int op=__shfl_xor_sync(0xffffffff,pick,d);if(other>best||(other==best&&op<pick)){best=other;pick=op;}}
   if(lane==0)output[oi+rank]=pick;
   #pragma unroll
   for(int j=0;j<3;++j)if(lane+j*32==pick)work[j]=-CUDART_INF_F;
  }
 }
}
void run(torch::Tensor q,torch::Tensor k,torch::Tensor output,double scale){
 TORCH_CHECK(q.is_cuda()&&q.scalar_type()==at::kBFloat16&&q.is_contiguous()&&k.is_contiguous(),"Require contiguous CUDA BF16");
 TORCH_CHECK(q.sizes()==torch::IntArrayRef({1,1,45,80,40,128})&&k.sizes()==torch::IntArrayRef({1,21,45,80,40,128}),"Require native40 RF0 inputs");
 TORCH_CHECK(output.scalar_type()==at::kInt&&output.numel()==21*45*20*40*16,"Invalid probe buffer");
 constexpr int bytes=80*128*2+64*80*4;
 C10_CUDA_CHECK(cudaFuncSetAttribute(probe_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
 probe_kernel<<<dim3(945,40),128,bytes,at::cuda::getCurrentCUDAStream()>>>((const bf16*)q.data_ptr(),(const bf16*)k.data_ptr(),output.data_ptr<int>(),float(scale*1.4426950408889634));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}
