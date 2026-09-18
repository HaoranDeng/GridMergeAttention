// Experimental warp layouts distribute both probes and candidates.
// Every thread owns sixteen scalar features. FP64 arithmetic and
// balanced reductions preserve the original candidate/probe reduction tree.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <math_constants.h>

// Opaque selects stop CUDA's array-index recognition from moving the entire
// FP64 feature table into thread-local memory for two runtime gathers.
__device__ __forceinline__ double register_select(double previous,double value,bool take) {
  double out;
  asm("{ .reg .pred p; setp.ne.u32 p,%3,0; selp.f64 %0,%2,%1,p; }"
      : "=d"(out) : "d"(previous),"d"(value),"r"(int(take)));
  return out;
}

template<int N>
__device__ __forceinline__ double sum_registers(double (&v)[N]) {
  #pragma unroll
  for(int d=N/2;d;d/=2){
    #pragma unroll
    for(int j=0;j<d;++j)v[j]=__dadd_rn(v[j],v[j+d]);
  }
  return v[0];
}
template<int P>
__device__ __forceinline__ double sum_probe16(double (&v)[16/P]) {
  double x=sum_registers(v);
  #pragma unroll
  for(int d=P/2;d;d/=2)x=__dadd_rn(x,__shfl_xor_sync(0xffffffff,x,d));
  return x;
}
template<int P>
__device__ __forceinline__ double sum_candidate32(double (&v)[P]) {
  double x=sum_registers(v);
  #pragma unroll
  for(int d=16;d>=P;d/=2)x=__dadd_rn(x,__shfl_xor_sync(0xffffffff,x,d));
  return x;
}
template<int SEEDS,int P>
__global__ void fisher_gs(const float* logits,const int* candidates,const int* anchors,
                         uint8_t* seeds,int groups,double scale) {
  constexpr int CS=32/P,PR=16/P;
  const int lane=threadIdx.x&31,probe=lane%P,slot=lane/P;
  const int group=(blockIdx.x*blockDim.x+threadIdx.x)/32;
  if(group>=groups)return;
  const int key_row=(group/(80*40))%45,anchor=anchors[key_row];
  double z[P][PR],center[PR];
  #pragma unroll
  for(int p=0;p<PR;++p){
    double terms[P];
    #pragma unroll
    for(int c=0;c<P;++c){
      z[c][p]=__dmul_rn(double(logits[(int64_t(group)*32+c*CS+slot)*16+p*P+probe]),scale);
      terms[c]=z[c][p];
    }
    center[p]=__dmul_rn(sum_candidate32<P>(terms),1.0/32);
  }
  double mx=-CUDART_INF;
  #pragma unroll
  for(int p=0;p<PR;++p)mx=fmax(mx,center[p]);
  #pragma unroll
  for(int d=P/2;d;d/=2)mx=fmax(mx,__shfl_xor_sync(0xffffffff,mx,d));
  double rho[PR],root[PR],anchor_z[PR];
  // Each half-warp holds all sixteen probe coordinates. Preserve the
  // original register-first, then probe-lane balanced summation order.
  double chosen_center=0;
  #pragma unroll
  for(int p=0;p<PR;++p)chosen_center=register_select(chosen_center,center[p],slot%PR==p);
  double rho_scalar=exp(__dadd_rn(chosen_center,-mx)),den=rho_scalar;
  #pragma unroll
  for(int d=8;d;d/=2)den=__dadd_rn(den,__shfl_xor_sync(0xffffffff,den,d));
  rho_scalar=__ddiv_rn(rho_scalar,den);
  double root_scalar=sqrt(rho_scalar);
  #pragma unroll
  for(int p=0;p<PR;++p){
    rho[p]=__shfl_sync(0xffffffff,rho_scalar,p*P+probe);
    root[p]=__shfl_sync(0xffffffff,root_scalar,p*P+probe);
    double value=0;
    #pragma unroll
    for(int c=0;c<P;++c)value=register_select(value,z[c][p],c==anchor/CS);
    anchor_z[p]=__shfl_sync(0xffffffff,value,(anchor%CS)*P+probe);
  }
  double res[P][PR];
  #pragma unroll
  for(int c=0;c<P;++c){
    double delta[PR],terms[PR];
    #pragma unroll
    for(int p=0;p<PR;++p){delta[p]=__dadd_rn(z[c][p],-anchor_z[p]);terms[p]=__dmul_rn(delta[p],rho[p]);}
    double mean=sum_probe16<P>(terms);
    #pragma unroll
    for(int p=0;p<PR;++p)res[c][p]=__dmul_rn(__dadd_rn(delta[p],-mean),root[p]);
  }
  unsigned used=1u<<anchor;
  if(lane==0)seeds[int64_t(group)*SEEDS]=uint8_t(candidates[key_row*32+anchor]);
  #pragma unroll
  for(int step=1;step<SEEDS;++step){
    double best=-CUDART_INF;int picked=32;
    #pragma unroll
    for(int c=0;c<P;++c){
      double terms[PR];
      #pragma unroll
      for(int p=0;p<PR;++p)terms[p]=__dmul_rn(res[c][p],res[c][p]);
      double score=sum_probe16<P>(terms);
      int index=c*CS+slot;
      if(!(used&(1u<<index)) && (score>best || (score==best && index<picked))){best=score;picked=index;}
    }
    #pragma unroll
    for(int d=16;d>=P;d/=2){
      double other=__shfl_xor_sync(0xffffffff,best,d);int oi=__shfl_xor_sync(0xffffffff,picked,d);
      if(other>best || (other==best && oi<picked)){best=other;picked=oi;}
    }
    used|=1u<<picked;
    if(lane==0)seeds[int64_t(group)*SEEDS+step]=uint8_t(candidates[key_row*32+picked]);
    if(step<SEEDS-1){
      double norm=sqrt(__dadd_rn(best,1.e-12)),basis[PR],basis_scalar=0;
      #pragma unroll
      for(int p=0;p<PR;++p){
        double value=0;
        #pragma unroll
        for(int c=0;c<P;++c)value=register_select(value,res[c][p],c==picked/CS);
        value=__shfl_sync(0xffffffff,value,(picked%CS)*P+probe);
        basis_scalar=register_select(basis_scalar,value,slot%PR==p);
      }
      basis_scalar=__ddiv_rn(basis_scalar,norm);
      #pragma unroll
      for(int p=0;p<PR;++p)basis[p]=__shfl_sync(0xffffffff,basis_scalar,p*P+probe);
      #pragma unroll
      for(int c=0;c<P;++c){
        double terms[PR];
        #pragma unroll
        for(int p=0;p<PR;++p)terms[p]=__dmul_rn(res[c][p],basis[p]);
        double projection=sum_probe16<P>(terms);
        #pragma unroll
        for(int p=0;p<PR;++p)res[c][p]=__fma_rn(-projection,basis[p],res[c][p]);
      }
    }
  }
}
void run(torch::Tensor logits,torch::Tensor candidates,torch::Tensor anchors,torch::Tensor seeds,double scale,int probe_lanes){
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type()==at::kFloat && logits.is_contiguous(),"Contiguous CUDA FP32 logits required");
  TORCH_CHECK(candidates.scalar_type()==at::kInt && anchors.scalar_type()==at::kInt,"Int32 metadata required");
  TORCH_CHECK(seeds.scalar_type()==at::kByte && seeds.is_contiguous(),"Contiguous uint8 seeds required");
  int groups=21*45*80*40,c=seeds.size(-1);
  TORCH_CHECK(logits.numel()==int64_t(groups)*32*16 && seeds.numel()==int64_t(groups)*c && (c==4 || c==10),"Native Fisher shapes required");
  auto stream=at::cuda::getCurrentCUDAStream();
  auto launch=[&](auto pt){
    constexpr int P=decltype(pt)::value;
    if(c==4)fisher_gs<4,P><<<(groups+3)/4,128,0,stream>>>(logits.data_ptr<float>(),candidates.data_ptr<int>(),anchors.data_ptr<int>(),seeds.data_ptr<uint8_t>(),groups,scale);
    else fisher_gs<10,P><<<(groups+3)/4,128,0,stream>>>(logits.data_ptr<float>(),candidates.data_ptr<int>(),anchors.data_ptr<int>(),seeds.data_ptr<uint8_t>(),groups,scale);
  };
  TORCH_CHECK(probe_lanes==4 || probe_lanes==8 || probe_lanes==16,"Require 4/8/16 probe lanes");
  if(probe_lanes==4)launch(std::integral_constant<int,4>{});
  else if(probe_lanes==8)launch(std::integral_constant<int,8>{});
  else launch(std::integral_constant<int,16>{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}
