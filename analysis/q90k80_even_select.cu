// Q90K80 even-capacity native RF0 selector (AXIS_CAP representatives), FP32
// tier, ported from the Q90K80C14 selector: the greedy Fisher residual loop
// runs in fp32, the cross-lane argmax is one REDUX.MAX over the non-negative
// score bit patterns plus one REDUX.MIN for the leftmost index, sums use
// balanced register trees, the picked candidate's residual is broadcast with
// SHFL.IDX, and the basis norm reuses the winning score. Scores z come from
// bf16 mma.sync with fp32 accumulation.
//
// 2026-09-23 frame loop: RF0 seeds depend on (head, query group, key row) and
// the 32 candidate queries on (head, group, row) only, so one CTA (four
// groups) keeps its candidate queries in shared memory (8 KB per warp) and
// loops over the 21 frames kg = ko*45 + row, loading only the 16 probe keys
// per frame (prefetched one frame ahead through registers into a single
// shared tile that is reused as the score transpose buffer). The arithmetic
// per (group, kg) is unchanged: seeds are bitwise those of the per-kg
// selector, at 5 CTAs/SM instead of 8 (C4 4.8 -> 3.2 ms, C10 7.8 -> 7.5 ms).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cstdint>

using bf16=__nv_bfloat16;
#ifndef AXIS_CAP
#define AXIS_CAP 8
#endif
constexpr int T=AXIS_OUTER,H=AXIS_HEADS,GROUPS=T*40,KG=T*45,C=AXIS_CAP,NPROBE=16,NCAND=32;
static_assert(C%2==0 && C>=4 && C<=14,"even capacity");
constexpr int KS_STRIDE=136,ZS_STRIDE=17,QS_STRIDE=136;
#ifndef SEL_MINBLOCKS
#define SEL_MINBLOCKS 5
#endif
// shared: sq[4 warps][32 cand][QS_STRIDE] bf16 = 34816 B; scratch = max(ks 16*136*2=4352, zs 4*32*17*4=8704) = 8704 B
constexpr int SQ_WORDS=4*NCAND*QS_STRIDE/2;      // uint32 words
constexpr int SCRATCH_WORDS=4*NCAND*ZS_STRIDE;    // floats (>= probe tile)

__device__ __forceinline__ void mma_bf16(float (&c)[4],const uint32_t (&a)[4],const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
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
__device__ __forceinline__ float dot16(const float (&a)[16],const float (&b)[16]) {
  float t[16];
  #pragma unroll
  for(int j=0;j<16;++j) t[j]=__fmul_rn(a[j],b[j]);
  return sum_registers(t);
}

__global__ __launch_bounds__(128,SEL_MINBLOCKS) void select_kernel(const bf16* __restrict__ q,const bf16* __restrict__ k,
 const int* __restrict__ table,uint8_t* __restrict__ seeds) {
  __shared__ __align__(16) uint32_t sq[SQ_WORDS];
  __shared__ __align__(16) float scratch[SCRATCH_WORDS];
  __shared__ int cids[NCAND];
  bf16* ks=reinterpret_cast<bf16*>(scratch);
  const int row=blockIdx.x,gb=blockIdx.y,head=blockIdx.z;
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,g=gb*4+warp;
  if(tid<NCAND) cids[tid]=table[row*NCAND+tid];
  auto probe_ptr=[&](int ko,int vec)->const uint4* {
    int p=vec/16,d=(vec%16)*8;
    return reinterpret_cast<const uint4*>(k+((int64_t(head)*KG+ko*45+row)*80+p*5)*128+d);
  };
  uint4 knext0=*probe_ptr(0,tid),knext1=*probe_ptr(0,tid+128);
  __syncthreads();
  // Candidate queries of this warp's group into shared memory: 32 rows x 256 B.
  bf16* wq=reinterpret_cast<bf16*>(sq)+warp*NCAND*QS_STRIDE;
  {
    const bf16* qbase=q+(int64_t(head)*GROUPS+g)*90*128;
    for(int vec=lane;vec<NCAND*16;vec+=32) {
      int c=vec/16,d=(vec%16)*8;
      *reinterpret_cast<uint4*>(wq+c*QS_STRIDE+d)=*reinterpret_cast<const uint4*>(qbase+int64_t(cids[c])*128+d);
    }
  }
  const int cid=cids[lane];
  const int anchor=__ffs(__ballot_sync(0xffffffff,cid==row))-1;
  const int second=__ffs(__ballot_sync(0xffffffff,cid==row+45))-1;
  const float scale=float(0.08838834764831845);
  float* wz=scratch+warp*(NCAND*ZS_STRIDE);
  // A-fragment shared addresses (per kk: +kk*16 elements).
  const bf16* qa0=wq+(lane>>2)*QS_STRIDE+(lane&3)*2;
  const bf16* qa1=wq+(8+(lane>>2))*QS_STRIDE+(lane&3)*2;
  #pragma unroll 1
  for(int ko=0;ko<T;++ko) {
    {
      int p0=tid/16,d0=(tid%16)*8,p1=(tid+128)/16,d1=((tid+128)%16)*8;
      *reinterpret_cast<uint4*>(&ks[p0*KS_STRIDE+d0])=knext0;
      *reinterpret_cast<uint4*>(&ks[p1*KS_STRIDE+d1])=knext1;
    }
    __syncthreads();
    if(ko+1<T) {knext0=*probe_ptr(ko+1,tid);knext1=*probe_ptr(ko+1,tid+128);}
    const int kg=ko*45+row;
    float acc[2][2][4];
    #pragma unroll
    for(int mt=0;mt<2;++mt)
      #pragma unroll
      for(int nt=0;nt<2;++nt)
        #pragma unroll
        for(int i=0;i<4;++i) acc[mt][nt][i]=0.f;
    #pragma unroll
    for(int kk=0;kk<8;++kk) {
      uint32_t b[2][2];
      #pragma unroll
      for(int nt=0;nt<2;++nt) {
        const bf16* kp=&ks[(nt*8+(lane>>2))*KS_STRIDE+kk*16+(lane&3)*2];
        b[nt][0]=*reinterpret_cast<const uint32_t*>(kp);
        b[nt][1]=*reinterpret_cast<const uint32_t*>(kp+8);
      }
      #pragma unroll
      for(int mt=0;mt<2;++mt) {
        uint32_t a[4];
        const bf16* r0=qa0+mt*16*QS_STRIDE+kk*16;const bf16* r1=qa1+mt*16*QS_STRIDE+kk*16;
        a[0]=*reinterpret_cast<const uint32_t*>(r0);a[1]=*reinterpret_cast<const uint32_t*>(r1);
        a[2]=*reinterpret_cast<const uint32_t*>(r0+8);a[3]=*reinterpret_cast<const uint32_t*>(r1+8);
        mma_bf16(acc[mt][0],a,b[0]);mma_bf16(acc[mt][1],a,b[1]);
      }
    }
    // All warps are done with the probe tile before the score transpose reuses the region.
    __syncthreads();
    #pragma unroll
    for(int mt=0;mt<2;++mt)
      #pragma unroll
      for(int nt=0;nt<2;++nt) {
        int r=mt*16+(lane>>2),p=nt*8+(lane&3)*2;
        wz[r*ZS_STRIDE+p]=acc[mt][nt][0];wz[r*ZS_STRIDE+p+1]=acc[mt][nt][1];
        wz[(r+8)*ZS_STRIDE+p]=acc[mt][nt][2];wz[(r+8)*ZS_STRIDE+p+1]=acc[mt][nt][3];
      }
    __syncwarp();
    float z[NPROBE];
    #pragma unroll
    for(int j=0;j<NPROBE;++j) z[j]=__fmul_rn(wz[lane*ZS_STRIDE+j],scale);
    // The next iteration's probe store must wait until every warp has read its scores.
    __syncthreads();
    float za[NPROBE];
    #pragma unroll
    for(int j=0;j<NPROBE;++j) za[j]=__shfl_sync(0xffffffff,z[j],anchor);
    float mx=za[0];
    #pragma unroll
    for(int j=1;j<NPROBE;++j) mx=fmaxf(mx,za[j]);
    float mine=0.f;
    #pragma unroll
    for(int j=0;j<NPROBE;++j) mine=(lane==j)?za[j]:mine;
    float ex=expf(__fadd_rn(mine,-mx));
    float den=ex;
    #pragma unroll
    for(int d=8;d>0;d>>=1) den=__fadd_rn(den,__shfl_xor_sync(0xffffffff,den,d,16));
    float rho_own=__fdiv_rn(ex,den),root_own=sqrtf(rho_own);
    float rho[NPROBE],root[NPROBE];
    #pragma unroll
    for(int j=0;j<NPROBE;++j) {rho[j]=__shfl_sync(0xffffffff,rho_own,j);root[j]=__shfl_sync(0xffffffff,root_own,j);}
    float delta[NPROBE];
    #pragma unroll
    for(int j=0;j<NPROBE;++j) delta[j]=__fadd_rn(z[j],-za[j]);
    float mean=dot16(delta,rho);
    float res[NPROBE];
    #pragma unroll
    for(int j=0;j<NPROBE;++j) res[j]=__fmul_rn(__fadd_rn(delta[j],-mean),root[j]);
    unsigned used=1u<<anchor;
    uint8_t* out=seeds+(int64_t(head*40+g)*KG+kg)*C;
    unsigned packed=(unsigned)row;
    #pragma unroll 1
    for(int slot=1;slot<C;++slot) {
      float score=dot16(res,res);
      bool valid=(((used>>lane)&1u)==0u)&(score==score);
      unsigned key=valid?__float_as_uint(score):0u;
      unsigned best_bits=__reduce_max_sync(0xffffffffu,key);
      int picked=(int)__reduce_min_sync(0xffffffffu,(key==best_bits&&valid)?(unsigned)lane:32u);
      if(slot==1) picked=second;
      used|=1u<<picked;
      unsigned chosen=(unsigned)__shfl_sync(0xffffffff,cid,picked)&0xffu;
      if(slot&1) packed|=chosen<<8; else packed=chosen;
      if((slot&1) && lane==0) *reinterpret_cast<uint16_t*>(out+slot-1)=(uint16_t)packed;
      if(slot<C-1) {
        float dir[NPROBE];
        #pragma unroll
        for(int j=0;j<NPROBE;++j) dir[j]=__shfl_sync(0xffffffff,res[j],picked);
        float dnorm=slot==1?__shfl_sync(0xffffffff,score,picked):__uint_as_float(best_bits);
        float inv=__fdiv_rn(1.f,sqrtf(__fadd_rn(dnorm,1.e-12f)));
        float basis[NPROBE];
        #pragma unroll
        for(int j=0;j<NPROBE;++j) basis[j]=__fmul_rn(dir[j],inv);
        float proj=dot16(res,basis);
        #pragma unroll
        for(int j=0;j<NPROBE;++j) res[j]=__fmaf_rn(-proj,basis[j],res[j]);
      }
    }
  }
}

void run(torch::Tensor q,torch::Tensor k,torch::Tensor table,torch::Tensor seeds) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.is_contiguous() && q.numel()==int64_t(H)*GROUPS*90*128,"Q shape");
  TORCH_CHECK(k.is_contiguous() && k.numel()==int64_t(H)*KG*80*128,"K shape");
  TORCH_CHECK(table.scalar_type()==at::kInt && table.numel()==45*NCAND,"table");
  TORCH_CHECK(seeds.scalar_type()==at::kByte && seeds.numel()==int64_t(H)*40*KG*C,"seeds");
  dim3 grid(45,10,H);
  select_kernel<<<grid,128,0,at::cuda::getCurrentCUDAStream()>>>((const bf16*)q.data_ptr(),(const bf16*)k.data_ptr(),table.data_ptr<int>(),seeds.data_ptr<uint8_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<int> resources() {
  cudaFuncAttributes a{};C10_CUDA_CHECK(cudaFuncGetAttributes(&a,select_kernel));
  int active=0;C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,select_kernel,128,0));
  return {a.numRegs,int(a.localSizeBytes),int(a.sharedSizeBytes),active};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("run",&run);m.def("resources",&resources);}
