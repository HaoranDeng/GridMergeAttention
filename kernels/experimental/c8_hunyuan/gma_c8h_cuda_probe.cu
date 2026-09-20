// GMA-C8 "Hunyuan" port of kernels/best/c8_r2/gma_c8_cuda_probe.cu.
//
// Shape change: Wan [1,21,45,80,40,128] -> Hunyuan [1,33,30,45,24,128].
//   temporal KO 21->33, rows(H) R 45->30, real cols(W) 80->45, heads 40->24.
//
// LOCKED DECISION (see task): the key/column axis scored by the WGMMA is
// padded 45->64 (not 48: the vendored cute header has no MMA_64x48x16
// template) and columns >=45 are masked to -inf before the softmax that
// ranks probe columns. This changes:
//   - sk (key staging) buffer: 80*128 bf16 -> 64*128 bf16
//   - Score WGMMA: MMA_64x80x16 -> MMA_64x64x16, E40(acc) -> E32(acc)
//   - the per-group softmax loop that used 3 lane-chunks of 32 to cover 80
//     columns now uses exactly 2 chunks of 32 to cover the padded 64 (no
//     partial last chunk -- 64 = 2*32 exactly, so no separate "chunk<3
//     partial" masking is needed, only the col<45 real-data mask).
//
// The probe/select "group" axis (this file's own W/4 anchor-sampling stride
// and the b1h_gs_regpipe_v0.cu selector's dense W/4 column blocking) shares
// one index space across both kernels (see gma_c8h_r0.py). Old W=80 gave
// W/4=20 groups (exact). New real W=45 gives ceil(45/4)=12 groups: the
// selector's last group spans real columns 44,45,46,47 (only 44 real; see
// that file for its masking), while THIS file's sparse anchor sampling at
// stride 4 lands only on 0,4,...,44 -- all real, no masking needed there.
//
// Uniform probe columns: the original hardcoded "lane*10" (a clean stride
// for W=80, spacing 10) approximated the Python reference's
// round(linspace(0, W-1, 8)). W=45 has no clean integer stride, so this port
// uses a small lookup table UNIFORM8 = round(linspace(0,44,8)) computed
// once, ahead of time, in Python and hardcoded here -- an engineering choice
// (not a locked decision), documented rather than guessed silently.
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

// Shape constants (Hunyuan 13B).
constexpr int KO = 33;   // temporal
constexpr int R = 30;    // rows (H)
constexpr int W = 45;    // real cols (W)
constexpr int WPAD = 64; // key axis padded for the WGMMA (45 -> 64, not 48)
constexpr int HEADS = 24;
constexpr int CG = 12;   // ceil(W/4): shared with the selector's group axis

__device__ __constant__ int UNIFORM8[8] = {0, 6, 13, 19, 25, 31, 38, 44};

__device__ __forceinline__ uint64_t desc(const bf16* p,int lead,int stride) {
  uint64_t s=__cvta_generic_to_shared(p);
  return ((s&0x3ffff)>>4) | (uint64_t(lead>>4)<<16) | (uint64_t(stride>>4)<<32) | (uint64_t(1)<<62);
}
__device__ __forceinline__ int aoff(int row,int k) {
  int offset=row*64+(k&63)+(k>>6)*4096;
  return offset^(((offset>>6)&7)<<3);
}
__device__ __forceinline__ int boff(int k,int n) {
  // 4096 = WPAD(64)*64: the padded key buffer's own row count, not the old
  // W=80 buffer's 5120 (=80*64).
  int offset=k*64+(n&63)+(n>>6)*4096;
  return offset^(((offset>>6)&7)<<3);
}
__device__ __forceinline__ float ex2(float x) {
  float y; asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y;
}
__device__ __forceinline__ float lg2(float x) {
  float y; asm("lg2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y;
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
 bf16* sk=(bf16*)raw;bf16* aq=sk+WPAD*128;float* score=(float*)aq;
 int tid=threadIdx.x,lane=tid&31,warp=tid>>5,head=blockIdx.y,kb=blockIdx.x;
 int kr=kb%R,ko=kb/R;
 for(int vec=tid;vec<WPAD*16;vec+=128){
  int row=vec/16,d=(vec%16)*8;uint4 x{};
  if(row<W)x=*reinterpret_cast<const uint4*>(k+((int64_t(kb)*W+row)*HEADS+head)*128+d);
  *reinterpret_cast<uint4*>(sk+boff(row,d))=x;
 }
 for(int vec=tid;vec<64*16;vec+=128){
  int row=vec/16,d=(vec%16)*8,group=row/2;
  int qr=min(R-1,max(0,kr+((row&1)?1:-1)));uint4 x{};
  if(group<CG)x=*reinterpret_cast<const uint4*>(q+((int64_t(qr)*W+group*4)*HEADS+head)*128+d);
  *reinterpret_cast<uint4*>(aq+aoff(row,d))=x;
 }
 __syncthreads();asm volatile("fence.proxy.async.shared::cta;":::"memory");
 asm volatile("wgmma.fence.sync.aligned;":::"memory");
 float acc[32];using Score=MMA_64x64x16_F32BF16BF16_SS<Major::K,Major::K>;
 #pragma unroll
 for(int d=0;d<128;d+=16)Score::fma(desc(aq+aoff(0,d),16,1024),desc(sk+boff(0,d),16,1024),E32(acc),d==0?ScaleOut::Zero:ScaleOut::One);
 asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
 asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
 __syncthreads();
 #pragma unroll
 for(int i=0;i<32;++i){
  int row=warp*16+(lane>>2)+8*((i&3)>>1),col=2*(lane&3)+8*(i>>2)+(i&1);
  score[row*WPAD+col]=__fmul_rn(acc[i],scale);
 }
 __syncthreads();
 for(int group=warp;group<CG;group+=4){
  float z[2][2],norm[2],work[2];
  #pragma unroll
  for(int a=0;a<2;++a){
   float mx=-CUDART_INF_F;
   #pragma unroll
   for(int j=0;j<2;++j){int col=lane+j*32;z[a][j]=col<W?score[(group*2+a)*WPAD+col]:-CUDART_INF_F;mx=fmaxf(mx,z[a][j]);}
   mx=warp_max(mx);float sum=0;
   #pragma unroll
   for(int j=0;j<2;++j)sum=__fadd_rn(sum,ex2(__fadd_rn(z[a][j],-mx)));
   norm[a]=__fadd_rn(mx,lg2(warp_sum(sum)));
  }
  #pragma unroll
  for(int j=0;j<2;++j){
   int col=lane+j*32;
   bool is_uniform=false;
   #pragma unroll
   for(int u=0;u<8;++u)is_uniform|=(col==UNIFORM8[u]);
   work[j]=(col<W&&!is_uniform)?fmaxf(__fadd_rn(z[0][j],-norm[0]),__fadd_rn(z[1][j],-norm[1])):-CUDART_INF_F;
  }
  int64_t oi=((int64_t(kb)*CG+group)*HEADS+head)*16;
  if(lane<8)output[oi+8+lane]=UNIFORM8[lane];
  #pragma unroll
  for(int rank=0;rank<8;++rank){
   float best=-CUDART_INF_F;int pick=INT_MAX;
   #pragma unroll
   for(int j=0;j<2;++j){int col=lane+j*32;if(work[j]>best||(work[j]==best&&col<pick)){best=work[j];pick=col;}}
   #pragma unroll
   for(int d=16;d;d/=2){float other=__shfl_xor_sync(0xffffffff,best,d);int op=__shfl_xor_sync(0xffffffff,pick,d);if(other>best||(other==best&&op<pick)){best=other;pick=op;}}
   if(lane==0)output[oi+rank]=pick;
   #pragma unroll
   for(int j=0;j<2;++j)if(lane+j*32==pick)work[j]=-CUDART_INF_F;
  }
 }
}
void run(torch::Tensor q,torch::Tensor k,torch::Tensor output,double scale){
 TORCH_CHECK(q.is_cuda()&&q.scalar_type()==at::kBFloat16&&q.is_contiguous()&&k.is_contiguous(),"Require contiguous CUDA BF16");
 TORCH_CHECK(q.sizes()==torch::IntArrayRef({1,1,R,W,HEADS,128})&&k.sizes()==torch::IntArrayRef({1,KO,R,W,HEADS,128}),"Require native24 RF0 inputs");
 TORCH_CHECK(output.scalar_type()==at::kInt&&output.numel()==int64_t(KO)*R*CG*HEADS*16,"Invalid probe buffer");
 constexpr int bytes=WPAD*128*2+64*WPAD*4;
 C10_CUDA_CHECK(cudaFuncSetAttribute(probe_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
 probe_kernel<<<dim3(KO*R,HEADS),128,bytes,at::cuda::getCurrentCUDAStream()>>>((const bf16*)q.data_ptr(),(const bf16*)k.data_ptr(),output.data_ptr<int>(),float(scale*1.4426950408889634));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}
