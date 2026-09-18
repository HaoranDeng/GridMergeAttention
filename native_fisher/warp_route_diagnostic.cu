#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "warp_route.cuh"
using namespace fisher_warp;

__global__ void diagnostic(const bf16* q,const bf16* k,const bf16* v,
    const float* bias,float* out,float* maximum,float* denominator,
    int tokens,int valid_rows,float scale) {
  int lane=threadIdx.x,batch=blockIdx.x;
  __shared__ bf16 rk[64*128],rv[64*128];
  __shared__ float rb[64];
  q+=batch*16*128;k+=int64_t(batch)*tokens*128;v+=int64_t(batch)*tokens*128;
  bias+=batch*tokens;out+=batch*16*128;maximum+=batch*16;denominator+=batch*16;
  RouteState state;state.clear();
  for(int start=0;start<tokens;start+=64) {
    for(int i=lane;i<64*128;i+=32) {
      int row=i/128,d=i%128;
      rk[ring_offset(row,d)]=start+row<tokens?k[(start+row)*128+d]:__float2bfloat16(0.f);
      rv[ring_offset(row,d)]=start+row<tokens?v[(start+row)*128+d]:__float2bfloat16(0.f);
    }
    for(int row=lane;row<64;row+=32)rb[row]=start+row<tokens?bias[start+row]:CUDART_INF_F;
    __syncwarp();
    consume_ring(q,128,valid_rows,rk,rv,rb,scale,state);
    __syncwarp();
  }
  int g=lane>>2,t=lane&3;
  #pragma unroll
  for(int i=0;i<64;++i) {
    int row=g+8*((i&3)>>1),d=8*(i/4)+2*t+(i&1);
    out[row*128+d]=row<valid_rows && state.sum[(i&3)>>1]>0?
        state.acc[i]/state.sum[(i&3)>>1]:0.f;
  }
  if(t==0) {
    maximum[g]=state.maximum[0];maximum[g+8]=state.maximum[1];
    denominator[g]=state.sum[0];denominator[g+8]=state.sum[1];
  }
}

std::vector<torch::Tensor> run(torch::Tensor q,torch::Tensor k,torch::Tensor v,
    torch::Tensor bias,int valid_rows,double scale) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && bias.is_cuda(),"CUDA required");
  TORCH_CHECK(q.device()==k.device() && q.device()==v.device() && q.device()==bias.device(),"Device mismatch");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && bias.is_contiguous(),"Contiguous required");
  TORCH_CHECK(q.scalar_type()==at::kBFloat16 && k.scalar_type()==at::kBFloat16 && v.scalar_type()==at::kBFloat16 && bias.scalar_type()==at::kFloat,"Dtype mismatch");
  TORCH_CHECK(q.dim()==3 && q.size(1)==16 && q.size(2)==128,"Q shape mismatch");
  TORCH_CHECK(k.dim()==3 && k.size(0)==q.size(0) && k.size(2)==128 && k.size(1)>0 && k.sizes()==v.sizes(),"KV shape mismatch");
  TORCH_CHECK(bias.dim()==2 && bias.size(0)==q.size(0) && bias.size(1)==k.size(1),"Bias shape mismatch");
  TORCH_CHECK(valid_rows>0 && valid_rows<=16,"Query row count mismatch");
  auto out=torch::empty(q.sizes(),q.options().dtype(at::kFloat));
  auto maximum=torch::empty({q.size(0),16},out.options()),denominator=torch::empty_like(maximum);
  diagnostic<<<q.size(0),32,0,at::cuda::getCurrentCUDAStream()>>>(
    (const bf16*)q.data_ptr(),(const bf16*)k.data_ptr(),(const bf16*)v.data_ptr(),bias.data_ptr<float>(),
    out.data_ptr<float>(),maximum.data_ptr<float>(),denominator.data_ptr<float>(),k.size(1),valid_rows,float(scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,maximum,denominator};
}
std::vector<int> resources() {
  cudaFuncAttributes a{};C10_CUDA_CHECK(cudaFuncGetAttributes(&a,diagnostic));
  return {a.numRegs,int(a.localSizeBytes),int(a.sharedSizeBytes)};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("run",&run);m.def("resources",&resources);}
