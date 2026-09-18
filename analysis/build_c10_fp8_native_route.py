"""Native C10 route: one FP8 TMA staging tile, BF16 WGMMA and full SQ1.

The checked-in BF16 reference is copied from the prior C4 experiment,
SHA256 794d1a64d46b053979fb12f3c0985d12a9faf7c5ddb3f2181c1c3550d3251532.
K and V alternate ownership of an8KiB FP8 tile. Collective conversion ends
before the next TMA overwrites it. WGMMA consumes separate BF16 K/V tiles.
Only SRAM contains expanded BF16 KV; global intermediates remain E4M3.
"""
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'native_fisher'
base = (root / 'fp8_route_bf16_reference.cu').read_text()
helpers = base[base.index('#include'):base.index('__device__ __forceinline__ void load_raw_tile')]
helpers = helpers.replace('#include <cuda_bf16.h>', '#include <cuda_bf16.h>\n#include <cuda_fp8.h>')
helpers = helpers.replace('void load_tile(bf16* dst', 'void load_tile(uint8_t* dst')
load_q = base[base.index(' for(int vec=tid;vec<64*16;vec+=128)'):base.index(' __syncthreads();\n if(tid==0) {\n  asm volatile("mbarrier.arrive.expect_tx')]
qk = base[base.index('  float logits[32];'):base.index('  if(tid==0 && key_group+1<75)')]
softmax = base[base.index('  #pragma unroll\n  for(int i=0;i<32;++i){'):base.index('  if(stop_stage==3)continue;')]
pv_start = base.index('  #pragma unroll\n  for(int i=0;i<64;++i)asm volatile')
pv = base[pv_start:base.index('  if(tid==0 && key_group+1<75)', pv_start)]
store = base[base.index(' #pragma unroll\n for(int i=0;i<64;i+=2){'):base.index('\nvoid run(')]
prefix = r'''
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
'''
loop = r'''
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
'''
middle = r'''
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
'''
binding = r'''
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
'''
(root / 'fisher_c10_fp8_native_route.cu').write_text(
    '// Experimental native FP8 storage route, full C10 plus SQ1.\n' + helpers + prefix + load_q + loop + qk + softmax + middle + pv + '  __syncthreads();\n }\n' + store + binding)
