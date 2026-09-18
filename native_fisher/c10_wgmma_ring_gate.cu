#include "c10_wgmma_ring.cuh"
using namespace c10_ring;

__global__ __launch_bounds__(128,4) void gate_kernel(const bf16* q,const bf16* k,const bf16* v,
    const uint8_t* seeds,const int* checks,bf16* out_k,bf16* out_v,float* out_bias){
  extern __shared__ __align__(128) unsigned char storage[];
  bf16* sk=reinterpret_cast<bf16*>(storage);bf16* sv=sk+80*128;
  bf16* mk=sv+80*128;bf16* mv=mk+COLUMNS*RING*128;
  float* bias=reinterpret_cast<float*>(mv+COLUMNS*RING*128);
  int index=blockIdx.x,qo=checks[index*4],col=checks[index*4+1],kb=checks[index*4+2],head=checks[index*4+3];
  load_raw(k,v,sk,sv,kb,head);
  produce(q,seeds,sk,sv,mk,mv,bias,qo,col,kb,head,float(0.08838834764831845*1.4426950408889634));
  __syncthreads();
  for(int i=threadIdx.x;i<COLUMNS*RING*128;i+=128){
    int row=i/128,d=i%128;
    out_k[index*COLUMNS*RING*128+i]=mk[ring_offset(row,d)];
    out_v[index*COLUMNS*RING*128+i]=mv[ring_offset(row,d)];
  }
  if(threadIdx.x<COLUMNS*RING)out_bias[index*COLUMNS*RING+threadIdx.x]=bias[threadIdx.x];
}
void run(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor seeds,torch::Tensor checks,
         torch::Tensor out_k,torch::Tensor out_v,torch::Tensor out_bias){
  TORCH_CHECK(q.is_cuda()&&q.is_contiguous()&&q.scalar_type()==at::kBFloat16&&q.numel()==int64_t(21)*45*80*40*128,"Native packed Q required");
  for(const auto& x:{k,v,seeds,checks,out_k,out_v,out_bias})TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.device()==q.device(),"Same-device contiguous tensors required");
  TORCH_CHECK(k.sizes()==q.sizes()&&v.sizes()==q.sizes()&&k.scalar_type()==at::kBFloat16&&v.scalar_type()==at::kBFloat16,"KV mismatch");
  TORCH_CHECK(seeds.scalar_type()==at::kByte&&seeds.numel()==945*80*40*10,"Native seeds required");
  TORCH_CHECK(checks.scalar_type()==at::kInt&&checks.dim()==2&&checks.size(1)==4,"Int32 [checks,4] required");
  int n=checks.size(0);
  TORCH_CHECK(n>0&&out_k.numel()==n*5*16*128&&out_k.scalar_type()==at::kBFloat16&&out_v.sizes()==out_k.sizes()&&out_v.scalar_type()==at::kBFloat16&&out_bias.numel()==n*5*16&&out_bias.scalar_type()==at::kFloat,"Output mismatch");
  C10_CUDA_CHECK(cudaFuncSetAttribute(gate_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SHARED_BYTES));
  gate_kernel<<<n,128,SHARED_BYTES,at::cuda::getCurrentCUDAStream()>>>((const bf16*)q.data_ptr(),(const bf16*)k.data_ptr(),(const bf16*)v.data_ptr(),seeds.data_ptr<uint8_t>(),checks.data_ptr<int>(),(bf16*)out_k.data_ptr(),(bf16*)out_v.data_ptr(),out_bias.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<int> resources(){
  cudaFuncAttributes a{};int active=0;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a,gate_kernel));
  C10_CUDA_CHECK(cudaFuncSetAttribute(gate_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SHARED_BYTES));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,gate_kernel,128,SHARED_BYTES));
  return {a.numRegs,int(a.localSizeBytes),SHARED_BYTES,active};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);m.def("resources",&resources);}
