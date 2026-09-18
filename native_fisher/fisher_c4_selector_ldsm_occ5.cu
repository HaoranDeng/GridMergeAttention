// Combined candidate: FP32 Gram-Schmidt plus the register-pipelined frame loop.
// TIMING/NUMERICS CANDIDATE: the Fisher Gram-Schmidt runs in FP32 instead of
// FP64. The Triton reference computes this in float64, so seeds are NOT
// guaranteed to match; the seed mismatch count against the reference is the
// whole point of the measurement. Probe dots are unchanged.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include "warp_mma.cuh"
using bf16=__nv_bfloat16;
// Preserve each unrolled candidate in a register. An ordinary C++ if chain
// can be converted into an indexed local-memory array by the compiler.
__device__ __forceinline__ float choose_candidate(float prior,float value,int index,int candidate) {
  asm("{ .reg .pred p; setp.eq.s32 p,%2,%3; selp.f32 %0,%1,%0,p; }"
      : "+f"(prior) : "f"(value),"r"(index),"r"(candidate));
  return prior;
}

template<int N>
__device__ __forceinline__ float sum_registers(float (&v)[N]) {
  #pragma unroll
  for(int d=N/2;d;d/=2){
    #pragma unroll
    for(int j=0;j<d;++j)v[j]=__fadd_rn(v[j],v[j+d]);
  }
  return v[0];
}
template<int P>
__device__ __forceinline__ float sum_probe16(float (&v)[16/P]) {
  float x=sum_registers(v);
  #pragma unroll
  for(int d=P/2;d;d/=2)x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,d));
  return x;
}
template<int P>
__device__ __forceinline__ float sum_candidate32(float (&v)[P]) {
  float x=sum_registers(v);
  #pragma unroll
  for(int d=16;d>=P;d/=2)x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,d));
  return x;
}
template<int SEEDS,int P>
__device__ __forceinline__ void fisher_from_registers(const float (&local)[P][16/P],
 const int* candidates,const int* anchors,uint8_t* seeds,int group,int key_row,float scale) {
  constexpr int CS=32/P,PR=16/P;
  const int lane=threadIdx.x&31,probe=lane%P,slot=lane/P;
  const int anchor=anchors[key_row];
  float z[P][PR],center[PR];
  #pragma unroll
  for(int p=0;p<PR;++p){
    float terms[P];
    #pragma unroll
    for(int c=0;c<P;++c){
      z[c][p]=__fmul_rn(local[c][p],scale);
      terms[c]=z[c][p];
    }
    center[p]=__fmul_rn(sum_candidate32<P>(terms),1.0f/32);
  }
  float mx=-CUDART_INF_F;
  #pragma unroll
  for(int p=0;p<PR;++p)mx=fmaxf(mx,center[p]);
  #pragma unroll
  for(int d=P/2;d;d/=2)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
  float rho[PR],root[PR],anchor_z[PR];
  if constexpr(P==4) {
    // Centers are identical across candidate slots. Assign four independent
    // probe subsets to four lane groups instead of repeating all four calls.
    int subset=(lane>>2)&3;
    float argument=center[0];
    #pragma unroll
    for(int p=1;p<4;++p)if(subset==p)argument=center[p];
    float exponential=expf(__fadd_rn(argument,-mx));
    // Same balanced addition tree: p0+p2, p1+p3, then probe xor2/xor1.
    float den=exponential;
    #pragma unroll
    for(int offset=8;offset;offset>>=1)
      den=__fadd_rn(den,__shfl_xor_sync(0xffffffff,den,offset));
    float normalized=__fdiv_rn(exponential,den),owned_root=sqrtf(normalized);
    #pragma unroll
    for(int p=0;p<4;++p) {
      rho[p]=__shfl_sync(0xffffffff,normalized,p*4+probe);
      root[p]=__shfl_sync(0xffffffff,owned_root,p*4+probe);
    }
  } else {
    float den_terms[PR];
    #pragma unroll
    for(int p=0;p<PR;++p)rho[p]=den_terms[p]=expf(__fadd_rn(center[p],-mx));
    float den=sum_probe16<P>(den_terms);
    #pragma unroll
    for(int p=0;p<PR;++p){rho[p]=__fdiv_rn(rho[p],den);root[p]=sqrtf(rho[p]);}
  }
  #pragma unroll
  for(int p=0;p<PR;++p){
    float value=0;
    #pragma unroll
    for(int c=0;c<P;++c)value=choose_candidate(value,z[c][p],anchor/CS,c);
    anchor_z[p]=__shfl_sync(0xffffffff,value,(anchor%CS)*P+probe);
  }
  float res[P][PR];
  #pragma unroll
  for(int c=0;c<P;++c){
    float delta[PR],terms[PR];
    #pragma unroll
    for(int p=0;p<PR;++p){delta[p]=__fadd_rn(z[c][p],-anchor_z[p]);terms[p]=__fmul_rn(delta[p],rho[p]);}
    float mean=sum_probe16<P>(terms);
    #pragma unroll
    for(int p=0;p<PR;++p)res[c][p]=__fmul_rn(__fadd_rn(delta[p],-mean),root[p]);
  }
  unsigned used=1u<<anchor;
  if(lane==0)seeds[int64_t(group)*SEEDS]=uint8_t(candidates[key_row*32+anchor]);
  #pragma unroll
  for(int step=1;step<SEEDS;++step){
    float best=-CUDART_INF_F;int picked=32;
    #pragma unroll
    for(int c=0;c<P;++c){
      float terms[PR];
      #pragma unroll
      for(int p=0;p<PR;++p)terms[p]=__fmul_rn(res[c][p],res[c][p]);
      float score=sum_probe16<P>(terms);
      int index=c*CS+slot;
      if(!(used&(1u<<index)) && (score>best || (score==best && index<picked))){best=score;picked=index;}
    }
    #pragma unroll
    for(int d=16;d>=P;d/=2){
      float other=__shfl_xor_sync(0xffffffff,best,d);int oi=__shfl_xor_sync(0xffffffff,picked,d);
      if(other>best || (other==best && oi<picked)){best=other;picked=oi;}
    }
    used|=1u<<picked;
    if(lane==0)seeds[int64_t(group)*SEEDS+step]=uint8_t(candidates[key_row*32+picked]);
    if(step<SEEDS-1){
      float norm=sqrtf(__fadd_rn(best,1.e-12f)),basis[PR];
      #pragma unroll
      for(int p=0;p<PR;++p){
        float value=0;
        #pragma unroll
        for(int c=0;c<P;++c)value=choose_candidate(value,res[c][p],picked/CS,c);
        value=__shfl_sync(0xffffffff,value,(picked%CS)*P+probe);
        basis[p]=value;
      }
      if constexpr(P==4) {
        int subset=(lane>>2)&3;
        float value=0.0f;
        #pragma unroll
        for(int p=0;p<4;++p)value=choose_candidate(value,basis[p],subset,p);
        float owned_basis=__fdiv_rn(value,norm);
        #pragma unroll
        for(int p=0;p<4;++p)basis[p]=__shfl_sync(0xffffffff,owned_basis,p*4+probe);
      } else {
        #pragma unroll
        for(int p=0;p<PR;++p)basis[p]=__fdiv_rn(basis[p],norm);
      }
      #pragma unroll
      for(int c=0;c<P;++c){
        float terms[PR];
        #pragma unroll
        for(int p=0;p<PR;++p)terms[p]=__fmul_rn(res[c][p],basis[p]);
        float projection=sum_probe16<P>(terms);
        #pragma unroll
        for(int p=0;p<PR;++p)res[c][p]=__fmaf_rn(-projection,basis[p],res[c][p]);
      }
    }
  }
}
// Padded shared layout instead of the XOR swizzle: row stride 136 bf16 (272 B =
// 17*16 B). Every 8-element chunk stays 16-byte aligned, and the eight row
// addresses of one 8x8 ldmatrix tile land in eight distinct 16-byte bank groups
// (272*r mod 128 = 16*r), so both the STS.128 fills and the ldmatrix reads are
// conflict-free while every fragment address becomes [lane_base + immediate].
__device__ __forceinline__ int qoff(int row,int d) {return row*136+d;}
// ldmatrix x4: lane i supplies the 16-byte row address of row (i&7) of matrix
// (i>>3). Matrix j lands in r[j], where lane l receives the 32-bit word at
// byte offset 4*(l&3) of row (l>>2) - the same (elem d+2t, elem d+2t+1) word a
// 32-bit shared load of that row/column pair returns.
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4],uint32_t addr) {
 asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
  : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(addr) : "memory");
}
__global__ __launch_bounds__(128,5) void fused_selector(const bf16* __restrict__ q,
 const bf16* __restrict__ k,const int* candidates,const int* anchors,uint8_t* seeds,
 double scale,float* debug_logits) {
 __shared__ __align__(128) bf16 raw[128*136+16*136];
 bf16* sq=raw;bf16* sk=sq+128*136;
 int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
 int group1=blockIdx.x,key_row=group1%45;
 int cg=(group1/45)%20,head=group1/(45*20),column=cg*4+warp;
 for(int vec=tid;vec<4*32*16;vec+=128) {
  int row=vec/16,d=(vec%16)*8,candidate=row%32,col=cg*4+row/32;
  int qr=candidates[key_row*32+candidate];
  int qi=((qr*80+col)*40+head)*128+d;
  *reinterpret_cast<uint4*>(sq+qoff(row,d))=*reinterpret_cast<const uint4*>(q+qi);
 }
 // Stay one frame ahead: each thread owns probes p and p+8, so the whole
 // 16x128 probe tile lives in two uint4 registers. The load for frame+1 is
 // issued before this frame's dots and Gram-Schmidt and only reaches shared at
 // the top of the next frame, so its latency hides behind work.
 int pf_probe=tid>>4,pf_d=(tid&15)*8;
 uint4 pf0,pf1;
 {
  int ki=((((0*45+key_row)*80+pf_probe*5)*40+head)*128)+pf_d;
  pf0=*reinterpret_cast<const uint4*>(k+ki);
  ki=((((0*45+key_row)*80+(pf_probe+8)*5)*40+head)*128)+pf_d;
  pf1=*reinterpret_cast<const uint4*>(k+ki);
 }
 #pragma unroll 1
 for(int frame=0;frame<21;++frame) {
 __syncthreads();
 *reinterpret_cast<uint4*>(sk+qoff(pf_probe,pf_d))=pf0;
 *reinterpret_cast<uint4*>(sk+qoff(pf_probe+8,pf_d))=pf1;
 __syncthreads();
 if(frame<20) {
  int ki=((((frame+1)*45+key_row)*80+pf_probe*5)*40+head)*128+pf_d;
  pf0=*reinterpret_cast<const uint4*>(k+ki);
  ki=((((frame+1)*45+key_row)*80+(pf_probe+8)*5)*40+head)*128+pf_d;
  pf1=*reinterpret_cast<const uint4*>(k+ki);
 }
 float acc[2][2][4]={};
 // ldmatrix row addresses in bytes. The lane-dependent part is fixed per
 // thread; m contributes 16 rows (16*272 B) and d contributes 2*d B, both
 // compile-time immediates once the d/m loops unroll.
 // A (Q rows warp*32+m*16..): lanes 0-7 rows +0..7 x cols d..d+7 -> a[0];
 //   lanes 8-15 rows +8..15 x d -> a[1]; lanes 16-23 rows +0..7 x d+8 -> a[2];
 //   lanes 24-31 rows +8..15 x d+8 -> a[3]. (lane&15) is the row, lane>>4 the
 //   8-column half.
 // B (K probes): lanes 0-7 keys 0-7 x d -> b[0][0]; lanes 8-15 keys 0-7 x d+8
 //   -> b[0][1]; lanes 16-23 keys 8-15 x d -> b[1][0]; lanes 24-31 keys 8-15 x
 //   d+8 -> b[1][1].
 const uint32_t sq_a=uint32_t(__cvta_generic_to_shared(sq))
  +uint32_t(warp*32*272+(lane&15)*272+16*(lane>>4));
 const uint32_t sk_b=uint32_t(__cvta_generic_to_shared(sk))
  +uint32_t(((lane&7)+8*(lane>>4))*272+16*((lane>>3)&1));
 #pragma unroll
 for(int d=0;d<128;d+=16) {
  uint32_t bq[4];
  ldsm_x4(bq,sk_b+2*d);
  const uint32_t b[2][2]={{bq[0],bq[1]},{bq[2],bq[3]}};
  #pragma unroll
  for(int m=0;m<2;++m) {
   uint32_t a[4];
   ldsm_x4(a,sq_a+m*16*272+2*d);
   #pragma unroll
   for(int n=0;n<2;++n)fisher_warp::mma(a,b[n],acc[m][n]);
  }
 }
 // Change warp distribution from pairs of adjacent probes to four probe lanes.
 float local[4][4];
 #pragma unroll
 for(int c=0;c<4;++c) {
  #pragma unroll
  for(int p=0;p<4;++p) {
   int src=(lane&~3)+2*(p&1)+((lane&3)>>1);
   float lo=__shfl_sync(0xffffffff,acc[c/2][p/2][2*(c&1)],src);
   float hi=__shfl_sync(0xffffffff,acc[c/2][p/2][2*(c&1)+1],src);
   local[c][p]=(lane&1)?hi:lo;
  }
 }
 int group=((frame*45+key_row)*80+column)*40+head;
 if(debug_logits) {
  #pragma unroll
  for(int c=0;c<4;++c) {
   #pragma unroll
   for(int p=0;p<4;++p)debug_logits[(int64_t(group)*32+c*8+lane/4)*16+p*4+(lane&3)]=local[c][p];
  }
 }
 fisher_from_registers<4,4>(local,candidates,anchors,seeds,group,key_row,float(scale));
 }
}
void run(torch::Tensor q,torch::Tensor k,torch::Tensor candidates,torch::Tensor anchors,
 torch::Tensor seeds,double scale,torch::Tensor debug) {
 TORCH_CHECK(q.is_cuda() && k.is_cuda() && q.is_contiguous() && k.is_contiguous(),"Contiguous CUDA Q/K required");
 TORCH_CHECK(q.scalar_type()==at::kBFloat16 && k.scalar_type()==at::kBFloat16,"BF16 Q/K required");
 TORCH_CHECK(q.numel()==45*80*40*128 && k.numel()==21*45*80*40*128,"Native first-frame Q and full K required");
 TORCH_CHECK(candidates.scalar_type()==at::kInt && anchors.scalar_type()==at::kInt,"Int32 metadata required");
 TORCH_CHECK(seeds.scalar_type()==at::kByte && seeds.numel()==21*45*80*40*4,"Native C4 seeds required");
 TORCH_CHECK(!debug.numel() || (debug.scalar_type()==at::kFloat && debug.numel()==int64_t(21)*45*80*40*32*16),"Debug logits shape invalid");
 fused_selector<<<45*20*40,128,0,at::cuda::getCurrentCUDAStream()>>>(
  (const bf16*)q.data_ptr(),(const bf16*)k.data_ptr(),candidates.data_ptr<int>(),anchors.data_ptr<int>(),
  seeds.data_ptr<uint8_t>(),scale,debug.numel()?debug.data_ptr<float>():nullptr);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<int> resources() {
 cudaFuncAttributes attr{};C10_CUDA_CHECK(cudaFuncGetAttributes(&attr,fused_selector));
 int active=0;C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,fused_selector,128,0));
 return {attr.numRegs,int(attr.localSizeBytes),int(attr.sharedSizeBytes),active};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);m.def("resources",&resources);}
