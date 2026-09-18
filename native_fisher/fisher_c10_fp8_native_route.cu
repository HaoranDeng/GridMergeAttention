// Experimental native FP8 storage route, full C10 plus SQ1.
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
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



__device__ __forceinline__ float sum64(float (&p)[16]) {
 float value=p[0];
 #pragma unroll
 for(int i=1;i<16;++i)value=__fadd_rn(value,p[i]);
 return sum4(value);
}
__device__ __forceinline__ void load_tile(uint8_t* dst,const CUtensorMap* map,uint32_t bar,int start,int group) {
 uint32_t sp=__cvta_generic_to_shared(dst);
 asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3,%4}],[%5];"::"r"(sp),"l"(map),"r"(0),"r"(start),"r"(group),"r"(bar):"memory");
}
__device__ __forceinline__ void wait_tile(uint32_t bar,int phase) {
 uint32_t ready;
 do {
  asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p,[%1],%2; selp.u32 %0,1,0,p; }":"=r"(ready):"r"(bar),"r"(phase):"memory");
 } while(!ready);
}


__device__ __forceinline__ void decode_tile(const uint8_t* src,bf16* dst,int tid) {
 for(int vec=tid;vec<64*64;vec+=128) {
  int row=vec/64,d=(vec%64)*2,offset=row*128+d;
  __nv_fp8x2_e4m3 value;
  value.__x=*reinterpret_cast<const uint16_t*>(src+(offset^(((offset>>7)&7)<<4)));
  float2 f=static_cast<float2>(value);
  *reinterpret_cast<__nv_bfloat162*>(dst+aoff(row,d))=__floats2bfloat162_rn(f.x,f.y);
 }
}
__global__ void decode_encodings_kernel(const uint8_t* src,bf16* dst) {
 int i=threadIdx.x;
 __nv_fp8x2_e4m3 value;value.__x=*reinterpret_cast<const uint16_t*>(src+2*i);
 float2 f=static_cast<float2>(value);
 *reinterpret_cast<__nv_bfloat162*>(dst+2*i)=__floats2bfloat162_rn(f.x,f.y);
}
__global__ __launch_bounds__(128,4) void route_kernel(
 const bf16* __restrict__ q,const bf16* __restrict__ raw_k,const bf16* __restrict__ raw_v,
 const float* __restrict__ bias,bf16* __restrict__ output,int HS,int AH,float sm_scale,float raw_scale,
 const __grid_constant__ CUtensorMap map_k,const __grid_constant__ CUtensorMap map_v) {
 extern __shared__ __align__(128) unsigned char raw[];
 bf16* route_q=reinterpret_cast<bf16*>(raw);bf16* sk=route_q+64*128;bf16* sv=sk+64*128;
 uint8_t* staging=reinterpret_cast<uint8_t*>(sv+64*128);
 float* route_bias=reinterpret_cast<float*>(staging+8192);
 uint64_t* barrier=reinterpret_cast<uint64_t*>(route_bias+64);
 uint32_t bar=__cvta_generic_to_shared(barrier),bar_v=__cvta_generic_to_shared(barrier+1);
 int tid=threadIdx.x,lane=tid&31,warp=tid>>5,seed=lane>>2;
 // Adjacent query frames share the same original singleton column in L2.
 int head=blockIdx.x%AH,col=(blockIdx.x/AH)/21,qo=(blockIdx.x/AH)%21,gh=HS+head;
 int group=(qo*80+col)*AH+head;
 if(tid==0) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(bar):"memory");
  asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(bar_v):"memory");
  asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
 }
 for(int vec=tid;vec<64*16;vec+=128) {
  int row=vec/16,d=(vec%16)*8;uint4 value{};
  if(row<45) {
   int64_t qi=(((int64_t(qo)*45+row)*80+col)*40+gh)*128+d;
   value=*reinterpret_cast<const uint4*>(q+qi);
  }
  *reinterpret_cast<uint4*>(route_q+aoff(row,d))=value;
 }

 __syncthreads();
 if(tid==0) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],8192;"::"r"(bar):"memory");
  load_tile(staging,&map_k,bar,0,group);
 }
 float result[64];
 #pragma unroll
 for(int i=0;i<64;++i)result[i]=0.f;
 float running_max[2]={-CUDART_INF_F,-CUDART_INF_F},running_sum[2]={0.f,0.f};
 for(int key_group=0;key_group<163;++key_group) {
  bool singleton=key_group>=148;
  int start=(singleton?key_group-148:key_group)*64;
  if(tid<64)route_bias[tid]=singleton?0.f:(start+tid<9450?bias[int64_t(group)*9450+start+tid]:CUDART_INF_F);
  if(!singleton) {
   wait_tile(bar,key_group&1);
   __syncthreads();
   decode_tile(staging,sk,tid);
   __syncthreads();
   if(tid==0) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],8192;"::"r"(bar_v):"memory");
    load_tile(staging,&map_v,bar_v,start,group);
   }
  } else {
   for(int vec=tid;vec<64*16;vec+=128) {
    int row=vec/16,d=(vec%16)*8;uint4 key{},value{};
    if(start+row<945) {
     int64_t offset=(((int64_t(start+row)*80+col)*40+gh)*128)+d;
     key=*reinterpret_cast<const uint4*>(raw_k+offset);
     value=*reinterpret_cast<const uint4*>(raw_v+offset);
    }
    *reinterpret_cast<uint4*>(sk+aoff(row,d))=key;
    *reinterpret_cast<uint4*>(sv+aoff(row,d))=value;
   }
   __syncthreads();
  }
  asm volatile("fence.proxy.async.shared::cta;":::"memory");
  asm volatile("wgmma.fence.sync.aligned;":::"memory");
  float logits[32];using QK=MMA_64x64x16_F32BF16BF16_SS<Major::K,Major::K>;
  #pragma unroll
  for(int d=0;d<128;d+=16)
   QK::fma(desc(route_q+aoff(0,d),16,1024),desc(sk+aoff(0,d),16,1024),E32(logits),d==0?ScaleOut::Zero:ScaleOut::One);
  asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
  #pragma unroll
  for(int i=0;i<32;++i){
   int ki=2*(lane&3)+8*(i>>2)+(i&1);
   if(singleton)logits[i]=start+ki<945?__fmul_rn(logits[i],raw_scale):-CUDART_INF_F;
   else logits[i]=__fmul_rn(__fmaf_rn(logits[i],sm_scale,-route_bias[ki]),1.4426950408889634f);
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

  if(!singleton) {
   wait_tile(bar_v,key_group&1);
   __syncthreads();
   decode_tile(staging,sv,tid);
   __syncthreads();
   if(tid==0 && key_group+1<148) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],8192;"::"r"(bar):"memory");
    load_tile(staging,&map_k,bar,start+64,group);
   }
   asm volatile("fence.proxy.async.shared::cta;":::"memory");
  }
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
  __syncthreads();
 }
 #pragma unroll
 for(int i=0;i<64;i+=2){
  int half=(i&3)>>1,row=warp*16+seed+8*half,d=2*(lane&3)+8*(i>>2);
  if(row<45){
   int64_t oi=(((int64_t(qo)*45+row)*80+col)*40+gh)*128+d;
   *reinterpret_cast<__nv_bfloat162*>(output+oi)=__floats2bfloat162_rn(
     __fdividef(result[i],running_sum[half]),__fdividef(result[i+1],running_sum[half]));
  }
 }
}

void run(torch::Tensor q,torch::Tensor raw_k,torch::Tensor raw_v,torch::Tensor mk,torch::Tensor mv,torch::Tensor bias,torch::Tensor output,int hs,int ah,double scale) {
 TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.sizes()==torch::IntArrayRef({1,21,45,80,40,128}),"Require native H40 BF16 Q");
 TORCH_CHECK(hs>=0 && ah==5 && hs+ah<=40,"Require valid five-head chunk");
 for(auto x:{q,raw_k,raw_v,output})TORCH_CHECK(x.device()==q.device() && x.sizes()==q.sizes() && x.scalar_type()==at::kBFloat16 && x.is_contiguous(),"Require same-device native contiguous BF16 tensors");
 for(auto x:{mk,mv})TORCH_CHECK(x.device()==q.device() && x.scalar_type()==at::ScalarType::Float8_e4m3fn && x.is_contiguous() && x.sizes()==torch::IntArrayRef({21*80*ah,9450,128}),"Require C10 E4M3 workspace");
 TORCH_CHECK(bias.device()==q.device() && bias.scalar_type()==at::kFloat && bias.is_contiguous() && bias.sizes()==mk.sizes().slice(0,2),"Require FP32 bias");
 TORCH_CHECK(output.data_ptr()!=q.data_ptr() && output.data_ptr()!=raw_k.data_ptr() && output.data_ptr()!=raw_v.data_ptr(),"Output must be out of place");
 CUtensorMap map_k{},map_v{};
 uint64_t dims[3]={128,9450,uint64_t(21*80*ah)},strides[2]={128,9450*128};
 uint32_t box[3]={128,64,1},elements[3]={1,1,1};
 auto encode=[&](CUtensorMap& map,torch::Tensor x) {
  auto status=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_UINT8,3,x.data_ptr(),dims,strides,box,elements,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(status==CUDA_SUCCESS,"FP8 route TMA encode failed: ",int(status));
 };
 encode(map_k,mk);encode(map_v,mv);
 constexpr int bytes=3*64*128*2+8192+64*4+16;
 C10_CUDA_CHECK(cudaFuncSetAttribute(route_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
 route_kernel<<<21*80*ah,128,bytes,at::cuda::getCurrentCUDAStream()>>>(
  (const bf16*)q.data_ptr(),(const bf16*)raw_k.data_ptr(),(const bf16*)raw_v.data_ptr(),bias.data_ptr<float>(),(bf16*)output.data_ptr(),hs,ah,float(scale),float(scale*1.4426950408889634),map_k,map_v);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void decode(torch::Tensor src,torch::Tensor dst) {
 TORCH_CHECK(src.is_cuda() && dst.device()==src.device() && src.is_contiguous() && dst.is_contiguous() && src.numel()==256 && dst.numel()==256 && src.scalar_type()==at::kByte && dst.scalar_type()==at::kBFloat16,"Require256 CUDA uint8 to BF16 elements");
 decode_encodings_kernel<<<1,128,0,at::cuda::getCurrentCUDAStream()>>>(src.data_ptr<uint8_t>(),(bf16*)dst.data_ptr());
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<int> resources() {
 cudaFuncAttributes attr{};C10_CUDA_CHECK(cudaFuncGetAttributes(&attr,route_kernel));
 constexpr int bytes=3*64*128*2+8192+64*4+16;
 C10_CUDA_CHECK(cudaFuncSetAttribute(route_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
 int active=0;C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,route_kernel,128,bytes));
 return {attr.numRegs,int(attr.localSizeBytes),bytes,active};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);m.def("decode",&decode);m.def("resources",&resources);}
