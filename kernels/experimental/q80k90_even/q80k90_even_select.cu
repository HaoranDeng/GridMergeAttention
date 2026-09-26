// Q80K90 even-capacity native RF0 selector (AXIS_CAP representatives), FP32.
// Port of analysis/q90k80_even_select.cu (frame loop over the 21 key frames,
// candidate queries kept in shared memory, REDUX argmax, balanced register
// trees, SHFL broadcast of the picked residual) to the Q80K90 grid:
//  - the query group is row g (0..44) of frame 0 (RF0), 80 consecutive tokens;
//    its 32 candidates are the columns table[pair] of that row;
//  - the key block kg = ko*40 + pair is column pair `pair` of frame ko (90 keys,
//    key j = token (row j%45, column 2*pair + j/45)); the 16 probe keys are
//    j = p*90/16 as in the v4 selector;
//  - the first two seeds are the intersections 2*pair and 2*pair + 1.
// Q and K are read in the caller's [1, N, H, 128] layout (token-major or
// head-major storage, given by the token and head element strides). One CTA handles one
// column pair and SEL_WARPS query rows (one per warp); with four warps 45 rows
// need 12 CTAs whose last three warps idle on zeroed candidates.
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
constexpr int T=AXIS_OUTER,H=AXIS_HEADS,ROWS=45,PAIRS=40,KG=T*PAIRS,C=AXIS_CAP,NPROBE=16,NCAND=32;
static_assert(C%2==0 && C>=4 && C<=14,"even capacity");
// SEL_WARPS query rows (one per warp) per CTA: 3, 5 and 9 divide the 45 rows
// exactly; 4 needs 12 CTAs per column pair whose last three warps idle.
#ifndef SEL_WARPS
#define SEL_WARPS 4
#endif
constexpr int W=SEL_WARPS,THREADS=W*32,GBLOCKS=(ROWS+W-1)/W;
static_assert(W>=3 && W<=9,"three to nine warps");
constexpr int PROBE_VECS=NPROBE*16,PREF=(PROBE_VECS+THREADS-1)/THREADS; // probe vectors per thread
constexpr int KS_STRIDE=136,ZS_STRIDE=17,QS_STRIDE=136;
#ifndef SEL_MINBLOCKS
#define SEL_MINBLOCKS (W==3?6:W==4?5:4)
#endif
// shared (dynamic): sq[W warps][32 cand][QS_STRIDE] bf16 (W=4: 34816 B); scratch =
// max(ks 16*136*2=4352, zs W*32*17*4) B; cids 128 B (W=4: 43648 B, W=5: 54528 B).
constexpr int SQ_WORDS=W*NCAND*QS_STRIDE/2;      // uint32 words
constexpr int SCRATCH_WORDS=(W*NCAND*ZS_STRIDE>NPROBE*KS_STRIDE/2)?W*NCAND*ZS_STRIDE:NPROBE*KS_STRIDE/2; // floats (>= probe tile)
constexpr int SMEM_BYTES=(SQ_WORDS+SCRATCH_WORDS+NCAND)*4;

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

__global__ __launch_bounds__(THREADS,SEL_MINBLOCKS) void select_kernel(const bf16* __restrict__ q,const bf16* __restrict__ k,
 const int* __restrict__ table,uint8_t* __restrict__ seeds,int64_t q_ts,int64_t q_hs,int64_t k_ts,int64_t k_hs) {
  extern __shared__ __align__(16) uint32_t sq[];
  float* scratch=reinterpret_cast<float*>(sq+SQ_WORDS);
  int* cids=reinterpret_cast<int*>(scratch+SCRATCH_WORDS);
  bf16* ks=reinterpret_cast<bf16*>(scratch);
  const int pair=blockIdx.x,gb=blockIdx.y,head=blockIdx.z;
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,g=gb*W+warp;
  const bool active=g<ROWS; // warp-uniform
  if(tid<NCAND) cids[tid]=table[pair*NCAND+tid];
  auto probe_ptr=[&](int ko,int vec)->const uint4* {
    const int p=vec/16,d=(vec%16)*8,j=p*90/16;
    const int64_t token=int64_t(ko)*3600+(j%45)*80+2*pair+j/45;
    return reinterpret_cast<const uint4*>(k+token*k_ts+int64_t(head)*k_hs+d);
  };
  uint4 knext[PREF];
  #pragma unroll
  for(int i=0;i<PREF;++i) if(tid+i*THREADS<PROBE_VECS) knext[i]=*probe_ptr(0,tid+i*THREADS);
  __syncthreads();
  // Candidate queries of this warp's row into shared memory: 32 rows x 256 B.
  bf16* wq=reinterpret_cast<bf16*>(sq)+warp*NCAND*QS_STRIDE;
  for(int vec=lane;vec<NCAND*16;vec+=32) {
    const int c=vec/16,d=(vec%16)*8;
    uint4 value=make_uint4(0,0,0,0);
    if(active) value=*reinterpret_cast<const uint4*>(q+(int64_t(g)*80+cids[c])*q_ts+int64_t(head)*q_hs+d);
    *reinterpret_cast<uint4*>(wq+c*QS_STRIDE+d)=value;
  }
  const int cid=cids[lane];
  const int anchor=__ffs(__ballot_sync(0xffffffff,cid==2*pair))-1;
  const int second=__ffs(__ballot_sync(0xffffffff,cid==2*pair+1))-1;
  const float scale=float(0.08838834764831845);
  float* wz=scratch+warp*(NCAND*ZS_STRIDE);
  // A-fragment shared addresses (per kk: +kk*16 elements).
  const bf16* qa0=wq+(lane>>2)*QS_STRIDE+(lane&3)*2;
  const bf16* qa1=wq+(8+(lane>>2))*QS_STRIDE+(lane&3)*2;
  #pragma unroll 1
  for(int ko=0;ko<T;++ko) {
    #pragma unroll
    for(int i=0;i<PREF;++i) {
      const int vec=tid+i*THREADS;
      if(vec<PROBE_VECS) *reinterpret_cast<uint4*>(&ks[(vec/16)*KS_STRIDE+(vec%16)*8])=knext[i];
    }
    __syncthreads();
    if(ko+1<T) {
      #pragma unroll
      for(int i=0;i<PREF;++i) if(tid+i*THREADS<PROBE_VECS) knext[i]=*probe_ptr(ko+1,tid+i*THREADS);
    }
    const int kg=ko*PAIRS+pair;
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
    if(!active) continue;
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
    uint8_t* out=seeds+(int64_t(head*ROWS+g)*KG+kg)*C;
    unsigned packed=(unsigned)(2*pair);
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

constexpr int NTOKENS=T*3600,NHEADS=H;
static void check_input(const torch::Tensor& x,int64_t ts,int64_t hs,const char* name) {
  // Dense [1, N, H, 128] BF16 in token-major (ts = H*128, hs = 128) or head-major (ts = 128, hs = N*128) order.
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kBFloat16 && x.dim()==4 && x.size(0)==1 && x.size(1)==NTOKENS && x.size(2)==NHEADS && x.size(3)==128,
              name," must be CUDA BF16 [1, N, H, 128] of the compiled shape");
  TORCH_CHECK(x.stride(3)==1 && x.stride(1)==ts && x.stride(2)==hs &&
              ((ts==int64_t(NHEADS)*128 && hs==128) || (ts==128 && hs==int64_t(NTOKENS)*128)),name," must be token-major or head-major dense");
}
void run(torch::Tensor q,torch::Tensor k,torch::Tensor table,torch::Tensor seeds,int64_t q_ts,int64_t q_hs,int64_t k_ts,int64_t k_hs) {
  check_input(q,q_ts,q_hs,"q");check_input(k,k_ts,k_hs,"k");
  TORCH_CHECK(table.scalar_type()==at::kInt && table.is_contiguous() && table.numel()==PAIRS*NCAND,"table");
  TORCH_CHECK(seeds.scalar_type()==at::kByte && seeds.is_contiguous() && seeds.numel()==int64_t(H)*ROWS*KG*C,"seeds");
  dim3 grid(PAIRS,GBLOCKS,H);
  C10_CUDA_CHECK(cudaFuncSetAttribute(select_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM_BYTES));
  select_kernel<<<grid,THREADS,SMEM_BYTES,at::cuda::getCurrentCUDAStream()>>>((const bf16*)q.data_ptr(),(const bf16*)k.data_ptr(),table.data_ptr<int>(),seeds.data_ptr<uint8_t>(),q_ts,q_hs,k_ts,k_hs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<int> resources() {
  cudaFuncAttributes a{};C10_CUDA_CHECK(cudaFuncGetAttributes(&a,select_kernel));
  C10_CUDA_CHECK(cudaFuncSetAttribute(select_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM_BYTES));
  int active=0;C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,select_kernel,THREADS,SMEM_BYTES));
  return {a.numRegs,int(a.localSizeBytes),SMEM_BYTES,active};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("run",&run);m.def("resources",&resources);}
