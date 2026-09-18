#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <math_constants.h>

__device__ __forceinline__ double sum16(double (&v)[16]) {
  #pragma unroll
  for(int stride=8;stride>0;stride/=2) {
    #pragma unroll
    for(int j=0;j<stride;++j)v[j]=__dadd_rn(v[j],v[j+stride]);
  }
  return v[0];
}
__device__ __forceinline__ double warp_sum(double value) {
  #pragma unroll
  for(int d=16;d;d/=2)value=__dadd_rn(value,__shfl_xor_sync(0xffffffff,value,d));
  return value;
}
template<int SEEDS>
__global__ void fisher_gs(const float* logits,const int* candidates,const int* anchors,
                          uint8_t* seeds,int groups,double scale) {
  const int lane=threadIdx.x&31;
  const int group=(blockIdx.x*blockDim.x+threadIdx.x)/32;
  if(group>=groups)return;
  const int key_row=(group/(80*40))%45;
  const int anchor=anchors[key_row];
  double z[16];
  #pragma unroll
  for(int p=0;p<16;++p)z[p]=__dmul_rn(double(logits[(int64_t(group)*32+lane)*16+p]),scale);
  double center=0;
  #pragma unroll
  for(int p=0;p<16;++p){
    double mean=__dmul_rn(warp_sum(z[p]),1.0/32);
    if((lane&15)==p)center=mean;
  }
  double mx=center;
  #pragma unroll
  for(int d=8;d;d/=2)mx=fmax(mx,__shfl_xor_sync(0xffffffff,mx,d));
  double rho=exp(__dadd_rn(center,-mx)),den=rho;
  #pragma unroll
  for(int d=8;d;d/=2)den=__dadd_rn(den,__shfl_xor_sync(0xffffffff,den,d));
  rho=__ddiv_rn(rho,den);
  double root=sqrt(rho),delta[16],weighted[16];
  #pragma unroll
  for(int p=0;p<16;++p){
    delta[p]=__dadd_rn(z[p],-__shfl_sync(0xffffffff,z[p],anchor));
    weighted[p]=__dmul_rn(delta[p],__shfl_sync(0xffffffff,rho,p));
  }
  double mean=sum16(weighted),res[16];
  #pragma unroll
  for(int p=0;p<16;++p)res[p]=__dmul_rn(__dadd_rn(delta[p],-mean),__shfl_sync(0xffffffff,root,p));
  unsigned used=1u<<anchor;
  if(lane==0)seeds[int64_t(group)*SEEDS]=uint8_t(candidates[key_row*32+anchor]);
  #pragma unroll
  for(int step=1;step<SEEDS;++step){
    double terms[16];
    #pragma unroll
    for(int p=0;p<16;++p)terms[p]=__dmul_rn(res[p],res[p]);
    double score=sum16(terms),best=(used&(1u<<lane))?-CUDART_INF:score;
    int picked=lane;
    #pragma unroll
    for(int d=16;d;d/=2){
      double other=__shfl_xor_sync(0xffffffff,best,d);
      int oi=__shfl_xor_sync(0xffffffff,picked,d);
      if(other>best || (other==best && oi<picked)){best=other;picked=oi;}
    }
    used|=1u<<picked;
    if(lane==0)seeds[int64_t(group)*SEEDS+step]=uint8_t(candidates[key_row*32+picked]);
    if(step<SEEDS-1){
      double norm=sqrt(__dadd_rn(__shfl_sync(0xffffffff,score,picked),1.e-12));
      double basis_scalar=0;
      #pragma unroll
      for(int p=0;p<16;++p){
        double value=__shfl_sync(0xffffffff,res[p],picked);
        if((lane&15)==p)basis_scalar=value;
      }
      basis_scalar=__ddiv_rn(basis_scalar,norm);
      double projection_terms[16];
      #pragma unroll
      for(int p=0;p<16;++p)projection_terms[p]=__dmul_rn(res[p],__shfl_sync(0xffffffff,basis_scalar,p));
      double projection=sum16(projection_terms);
      #pragma unroll
      for(int p=0;p<16;++p)res[p]=__fma_rn(-projection,__shfl_sync(0xffffffff,basis_scalar,p),res[p]);
    }
  }
}
void run(torch::Tensor logits,torch::Tensor candidates,torch::Tensor anchors,torch::Tensor seeds,double scale){
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type()==at::kFloat && logits.is_contiguous(),"Contiguous CUDA FP32 logits required");
  TORCH_CHECK(candidates.scalar_type()==at::kInt && anchors.scalar_type()==at::kInt,"Int32 candidate metadata required");
  TORCH_CHECK(seeds.scalar_type()==at::kByte && seeds.is_contiguous(),"Contiguous uint8 seeds required");
  int groups=21*45*80*40,c=seeds.size(-1);
  TORCH_CHECK(logits.numel()==int64_t(groups)*32*16 && seeds.numel()==int64_t(groups)*c && (c==4 || c==8 || c==10),"Native Fisher C4/C8/C10 shapes required");
  auto stream=at::cuda::getCurrentCUDAStream();
  if(c==4)fisher_gs<4><<<(groups+3)/4,128,0,stream>>>(logits.data_ptr<float>(),candidates.data_ptr<int>(),anchors.data_ptr<int>(),seeds.data_ptr<uint8_t>(),groups,scale);
  else if(c==8)fisher_gs<8><<<(groups+3)/4,128,0,stream>>>(logits.data_ptr<float>(),candidates.data_ptr<int>(),anchors.data_ptr<int>(),seeds.data_ptr<uint8_t>(),groups,scale);
  else fisher_gs<10><<<(groups+3)/4,128,0,stream>>>(logits.data_ptr<float>(),candidates.data_ptr<int>(),anchors.data_ptr<int>(),seeds.data_ptr<uint8_t>(),groups,scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}
