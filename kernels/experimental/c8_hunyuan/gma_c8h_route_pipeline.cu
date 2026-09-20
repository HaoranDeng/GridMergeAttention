// GMA-C8 "Hunyuan" port of kernels/best/c8_r2/gma_c8_route_pipeline.cu.
// Shape: Wan [1,21,45,80,40,128] -> Hunyuan [1,33,30,45,24,128].
//
// This kernel's own compute (paging through the merged K/V pool in
// MMA_64x64x16 tiles) is UNCHANGED -- it already operated at N=64 on the
// *merged* key pool, which has nothing to do with the real-W=45 padding
// used elsewhere in this port. What DOES change here:
//   - H-axis mask row<45 -> row<30 (R), and the "45"/"80"/"40" address
//     strides -> R(30)/W(45)/HEADS(24) for the (qo,row,col,head) <-> flat Q
//     and output tensor addressing.
//   - KEYPOOL 7560=KO*R*8 -> 7920=33*30*8, and the per-CTA tile loop count
//     119=ceil(7560/64) -> 124=ceil(7920/64).
//   - The grid/block decomposition. The Wan source could use blockIdx.x
//     directly as the flat (qo,col,head) index into mk/mv because real W=80
//     was already a multiple of 8 (no merge-side output padding was ever
//     needed there). Real W=45 is NOT a multiple of 8, so
//     gma_c8h_merge_v0.cu writes mk/mv/bias at a PADDED column stride
//     C_OUT=48 (see that file's header). This kernel's grid still launches
//     one block per REAL (qo, col, head) triple -- col in [0,45), so the
//     final output write addresses the real (unpadded) Q/output tensors
//     unchanged -- but every read of mk/mv/bias (the TMA tile loads and the
//     per-key-group bias slice) must use the PADDED index
//     kv_index = (qo*C_OUT + col)*AH + head, matching how merge wrote it,
//     not blockIdx.x itself.
#include <cooperative_groups.h>
#include <cuda_runtime.h>
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
#define E64(a) E32(a),E8(a,32),E8(a,40),E8(a,48),E8(a,56)

constexpr int QO = 33;
constexpr int R = 30;     // H
constexpr int W = 45;     // real cols
constexpr int HEADS = 24;
constexpr int C_OUT = 48; // padded merge output column stride
constexpr int KEYPOOL = 7920;      // KO*R*8 = 33*30*8
constexpr int KEYPOOL_TILES = 124; // ceil(7920/64)

__device__ __forceinline__ uint64_t desc(const bf16* p,int lead,int stride) {
  uint64_t s=__cvta_generic_to_shared(p);
  return ((s&0x3ffff)>>4) | (uint64_t(lead>>4)<<16) | (uint64_t(stride>>4)<<32) | (uint64_t(1)<<62);
}
__device__ __forceinline__ int aoff(int row,int k) {
  int offset=row*64+(k&63)+(k>>6)*4096;
  return offset^(((offset>>6)&7)<<3);
}
__device__ __forceinline__ float ex2(float x) {
  float y; asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y;
}
__device__ __forceinline__ float max4(float x) {
  x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,1,4));
  return fmaxf(x,__shfl_xor_sync(0xffffffff,x,2,4));
}
__device__ __forceinline__ float sum4(float x) {
  x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,2,4));
  return __fadd_rn(x,__shfl_xor_sync(0xffffffff,x,1,4));
}
__device__ __forceinline__ float sum64(float (&p)[16]) {
 float value=p[0];
 #pragma unroll
 for(int i=1;i<16;++i)value=__fadd_rn(value,p[i]);
 return sum4(value);
}
__device__ __forceinline__ void load_tile(bf16* dst,const CUtensorMap* map,uint32_t bar,int start,int group) {
 uint32_t sp=__cvta_generic_to_shared(dst);
 asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3,%4}],[%5];"::"r"(sp),"l"(map),"r"(0),"r"(start),"r"(group),"r"(bar):"memory");
}
__device__ __forceinline__ void wait_tile(uint32_t bar,int phase) {
 uint32_t ready;
 do {
  asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p,[%1],%2; selp.u32 %0,1,0,p; }":"=r"(ready):"r"(bar),"r"(phase):"memory");
 } while(!ready);
}

__global__ __launch_bounds__(128,4) void route_kernel(
 const bf16* __restrict__ q,const float* __restrict__ bias,bf16* __restrict__ output,int HS,int AH,float sm_scale,
 const __grid_constant__ CUtensorMap map_k,const __grid_constant__ CUtensorMap map_k_hi,
 const __grid_constant__ CUtensorMap map_v,const __grid_constant__ CUtensorMap map_v_hi) {
 constexpr int stop_stage=0;
 extern __shared__ __align__(128) unsigned char raw[];
 bf16* route_q=reinterpret_cast<bf16*>(raw);bf16* sk=route_q+64*128;bf16* sv=sk+64*128;
 float* route_bias=reinterpret_cast<float*>(sv+64*128);
 uint64_t* barrier=reinterpret_cast<uint64_t*>(route_bias+64);
 uint32_t bar=__cvta_generic_to_shared(barrier),bar_v=__cvta_generic_to_shared(barrier+1);
 int tid=threadIdx.x,lane=tid&31,warp=tid>>5,seed=lane>>2;
 int block=blockIdx.x,head=block%AH,gh=HS+head;
 int qo=block/(W*AH),col=(block/AH)%W;
 // Padded index into mk/mv/bias, matching gma_c8h_merge_v0.cu's C_OUT=48
 // output stride. block itself (real-W based) must NOT be used for these.
 int kv_index=(qo*C_OUT+col)*AH+head;
 if(tid==0) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(bar):"memory");
  asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(bar_v):"memory");
  asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
 }
 for(int vec=tid;vec<64*16;vec+=128) {
  int row=vec/16,d=(vec%16)*8;uint4 value{};
  if(row<R) {
   int64_t qi=(((int64_t(qo)*R+row)*W+col)*HEADS+gh)*128+d;
   value=*reinterpret_cast<const uint4*>(q+qi);
  }
  *reinterpret_cast<uint4*>(route_q+aoff(row,d))=value;
 }
 __syncthreads();
 if(tid==0) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],16384;"::"r"(bar):"memory");
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],16384;"::"r"(bar_v):"memory");
  load_tile(sk,&map_k,bar,0,kv_index);load_tile(sk+4096,&map_k_hi,bar,0,kv_index);
  load_tile(sv,&map_v,bar_v,0,kv_index);load_tile(sv+4096,&map_v_hi,bar_v,0,kv_index);
 }
 float result[64];
 #pragma unroll
 for(int i=0;i<64;++i)result[i]=0.f;
 float running_max[2]={-CUDART_INF_F,-CUDART_INF_F},running_sum[2]={0.f,0.f};
 for(int key_group=0;key_group<KEYPOOL_TILES;++key_group) {
  int start=key_group*64;
  if(tid<64)route_bias[tid]=start+tid<KEYPOOL?bias[int64_t(kv_index)*KEYPOOL+start+tid]:CUDART_INF_F;
  wait_tile(bar,key_group&1);
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;":::"memory");
  asm volatile("wgmma.fence.sync.aligned;":::"memory");
  float logits[32];using QK=MMA_64x64x16_F32BF16BF16_SS<Major::K,Major::K>;
  #pragma unroll
  for(int d=0;d<128;d+=16)
   QK::fma(desc(route_q+aoff(0,d),16,1024),desc(sk+aoff(0,d),16,1024),E32(logits),d==0?ScaleOut::Zero:ScaleOut::One);
  asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
  if(tid==0 && key_group+1<KEYPOOL_TILES) {
   asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],16384;"::"r"(bar):"memory");
   load_tile(sk,&map_k,bar,start+64,kv_index);load_tile(sk+4096,&map_k_hi,bar,start+64,kv_index);
  }
  #pragma unroll
  for(int i=0;i<32;++i){
   int ki=2*(lane&3)+8*(i>>2)+(i&1);
   logits[i]=__fmul_rn(__fmaf_rn(logits[i],sm_scale,-route_bias[ki]),1.4426950408889634f);
  }
  uint32_t probabilities[16];
  #pragma unroll
  for(int half=0;half<2;++half){
   float p[16],mx=-CUDART_INF_F;
   #pragma unroll
   for(int i=0;i<16;++i){p[i]=logits[4*(i/2)+2*half+(i&1)];mx=fmaxf(mx,p[i]);}
   mx=fmaxf(running_max[half],max4(mx));float alpha=ex2(__fadd_rn(running_max[half],-mx));
   #pragma unroll
   for(int i=0;i<16;++i)p[i]=ex2(__fadd_rn(p[i],-mx));
   running_sum[half]=__fmaf_rn(running_sum[half],alpha,sum64(p));running_max[half]=mx;
   #pragma unroll
   for(int i=0;i<64;++i)if(((i&3)>>1)==half)result[i]=__fmul_rn(result[i],alpha);
   #pragma unroll
   for(int i=0;i<8;++i){
    union{__nv_bfloat162 value;uint32_t bits;} pair;pair.value=__floats2bfloat162_rn(p[2*i],p[2*i+1]);
    probabilities[2*i+half]=pair.bits;
   }
  }
  if(stop_stage==3)continue;
  wait_tile(bar_v,key_group&1);
  #pragma unroll
  for(int i=0;i<64;++i)asm volatile("" : "+f"(result[i]) :: "memory");
  #pragma unroll
  for(int i=0;i<16;++i)asm volatile("" : "+r"(probabilities[i]) :: "memory");
  asm volatile("wgmma.fence.sync.aligned;":::"memory");
  using PV=MMA_64x128x16_F32BF16BF16_RS<Major::K,Major::MN>;
  #pragma unroll
  for(int ki=0;ki<64;ki+=16){
   int pi=(ki/8)*2;
   PV::fma(probabilities[pi],probabilities[pi+1],probabilities[pi+2],probabilities[pi+3],
      desc(sv+aoff(ki,0),8192,1024),E64(result),ScaleOut::One);
  }
  asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
  if(tid==0 && key_group+1<KEYPOOL_TILES) {
   asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],16384;"::"r"(bar_v):"memory");
   load_tile(sv,&map_v,bar_v,start+64,kv_index);load_tile(sv+4096,&map_v_hi,bar_v,start+64,kv_index);
  }
 }
 #pragma unroll
 for(int i=0;i<64;i+=2){
  int half=(i&3)>>1,row=warp*16+seed+8*half,d=2*(lane&3)+8*(i>>2);
  if(row<R){
   int64_t oi=(((int64_t(qo)*R+row)*W+col)*HEADS+gh)*128+d;
   *reinterpret_cast<__nv_bfloat162*>(output+oi)=__floats2bfloat162_rn(
     __fdividef(result[i],running_sum[half]),__fdividef(result[i+1],running_sum[half]));
  }
 }
}

void run(torch::Tensor q,torch::Tensor mk,torch::Tensor mv,torch::Tensor bias,torch::Tensor output,int hs,int ah,double scale) {
 TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.sizes()==torch::IntArrayRef({1,QO,R,W,HEADS,128}),"Require full native24 BF16 Q");
 TORCH_CHECK(q.is_contiguous() && mk.is_contiguous() && mv.is_contiguous() && bias.is_contiguous() && output.is_contiguous(),"Require contiguous tensors");
 TORCH_CHECK(mk.numel()==int64_t(QO)*C_OUT*ah*KEYPOOL*128 && mv.numel()==mk.numel() && bias.numel()==mk.numel()/128,"Invalid native45 C8 workspace");
 TORCH_CHECK(output.numel()==q.numel() && hs>=0 && ah>0 && hs+ah<=HEADS,"Invalid output or head range");
 CUtensorMap map_k{},map_k_hi{},map_v{},map_v_hi{};
 uint64_t dims[3]={64,uint64_t(KEYPOOL),uint64_t(QO*C_OUT*ah)},strides[2]={256,uint64_t(KEYPOOL)*256};
 uint32_t box[3]={64,64,1},element_strides[3]={1,1,1};
 auto encode=[&](CUtensorMap& map,torch::Tensor x,int half) {
  auto status=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,reinterpret_cast<char*>(x.data_ptr())+half*128,dims,strides,box,element_strides,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(status==CUDA_SUCCESS,"Route TMA encode failed: ",int(status));
 };
 encode(map_k,mk,0);encode(map_k_hi,mk,1);encode(map_v,mv,0);encode(map_v_hi,mv,1);
 constexpr int bytes=3*64*128*2+64*4+16;
 C10_CUDA_CHECK(cudaFuncSetAttribute(route_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
 route_kernel<<<QO*W*ah,128,bytes,at::cuda::getCurrentCUDAStream()>>>(
  (const bf16*)q.data_ptr(),bias.data_ptr<float>(),(bf16*)output.data_ptr(),hs,ah,float(scale),map_k,map_k_hi,map_v,map_v_hi);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("run",&run);}
