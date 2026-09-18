#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "warp_merge.cuh"
using namespace fisher_warp;

template<int CAP>
__global__ void diagnostic(const bf16* q,const bf16* k,const bf16* v,
    const int* columns,bf16* mk,bf16* mv,float* bias,float scale) {
  int lane=threadIdx.x,batch=blockIdx.x;
  __shared__ __align__(16) bf16 raw[80*128];
  q+=batch*16*128;k+=batch*80*128;v+=batch*80*128;
  for(int i=lane;i<80*128;i+=32)raw[raw_offset(i/128,i%128)]=k[i];
  __syncwarp();
  uint32_t weights[20];float residual[2];
  make_weights<CAP>(q,raw,scale,columns[batch],weights,residual,bias+batch*CAP);
  merge_values<CAP>(weights,residual,raw,mk+batch*CAP*128);
  __syncwarp();
  for(int i=lane;i<80*128;i+=32)raw[raw_offset(i/128,i%128)]=v[i];
  __syncwarp();
  merge_values<CAP>(weights,residual,raw,mv+batch*CAP*128);
}

std::vector<torch::Tensor> run(torch::Tensor q,torch::Tensor k,torch::Tensor v,
    torch::Tensor columns,int capacity,double scale) {
  TORCH_CHECK(capacity==4 || capacity==10,"Capacity must be four or ten");
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && columns.is_cuda(),"CUDA required");
  TORCH_CHECK(q.device()==k.device() && q.device()==v.device() && q.device()==columns.device(),"Device mismatch");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && columns.is_contiguous(),"Contiguous required");
  TORCH_CHECK(q.scalar_type()==at::kBFloat16 && k.scalar_type()==at::kBFloat16 && v.scalar_type()==at::kBFloat16 && columns.scalar_type()==at::kInt,"Dtype mismatch");
  TORCH_CHECK(q.dim()==3 && q.size(1)==16 && q.size(2)==128,"Q shape mismatch");
  TORCH_CHECK(k.dim()==3 && k.size(0)==q.size(0) && k.size(1)==80 && k.size(2)==128 && k.sizes()==v.sizes(),"KV shape mismatch");
  TORCH_CHECK(columns.numel()==q.size(0),"Column shape mismatch");
  auto mk=torch::empty({q.size(0),capacity,128},q.options()),mv=torch::empty_like(mk);
  auto bias=torch::empty({q.size(0),capacity},q.options().dtype(at::kFloat));
  auto stream=at::cuda::getCurrentCUDAStream();
  auto launch=[&](auto tag) {
    diagnostic<decltype(tag)::value><<<q.size(0),32,0,stream>>>(
      (const bf16*)q.data_ptr(),(const bf16*)k.data_ptr(),(const bf16*)v.data_ptr(),columns.data_ptr<int>(),
      (bf16*)mk.data_ptr(),(bf16*)mv.data_ptr(),bias.data_ptr<float>(),float(scale*1.4426950408889634));
  };
  if(capacity==4)launch(std::integral_constant<int,4>{});else launch(std::integral_constant<int,10>{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {mk,mv,bias};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("run",&run);}
