// Q90K80C14 native route, transposed formulation.
// One CTA per (query group, head): 384 threads = two consumer warpgroups and
// one producer warpgroup (TMA ring of 64-key merged K/V tiles, S stages).
// Scores are computed as S^T[64 keys x 96 queries] = K_tile . Q^T (WGMMA
// m64n96k16, so the 90 queries pad to 96 instead of 128), the column softmax
// runs through shared memory, P^T is written to shared memory as the B
// operand of O^T[128 d x 96 q] += V^T . P^T, and the two consumer warpgroups
// take alternate key chunks (flash-decoding style) and merge at the end.
// Each consumer warpgroup owns two private ring stages: with a ring shared
// between the two warpgroups a fast warpgroup could wait on a stage whose
// first phase had not completed yet (a parity wait passes immediately then),
// consume an unfilled stage and desynchronize the barrier bookkeeping. P^T
// lives in the already-consumed K tile of the current stage.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cuda.h>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/arch/mma_sm90_gmma_ext.hpp>
#include <cstdint>

using bf16 = __nv_bfloat16;
using namespace cute::SM90::GMMA;

#define ROUTE_STAGES 4 // two private stages per consumer warpgroup
#ifndef ROUTE_CONSUMER_REGS
#define ROUTE_CONSUMER_REGS 232
#endif
#ifndef ROUTE_PRODUCER_REGS
#define ROUTE_PRODUCER_REGS 40
#endif

#define E8(a,o) a[o],a[o+1],a[o+2],a[o+3],a[o+4],a[o+5],a[o+6],a[o+7]
#define E48(a) E8(a,0),E8(a,8),E8(a,16),E8(a,24),E8(a,32),E8(a,40)

constexpr int T=AXIS_OUTER,KEYS=T*45*14,CHUNKS=(KEYS+63)/64,GROUPS=T*40;
constexpr int NQ=96,TILE=64*128,LOG_STRIDE=104; // logits row stride (floats)
constexpr int THREADS=384,S=ROUTE_STAGES;
// Shared memory carve-up (bytes).
constexpr int SQ_BYTES=NQ*128*2;            // 24576, Q^T tile
constexpr int RING_BYTES=S*2*TILE*2;         // K+V stages
constexpr int LOG_BYTES=64*LOG_STRIDE*4;     // 26624 per warpgroup
constexpr int MISC_BYTES=2*(NQ*4*3+16)+2*S*8+64; // alpha/max/sum per WG, flags, barriers
constexpr int OFF_RING=SQ_BYTES,OFF_LOG=OFF_RING+RING_BYTES,OFF_MISC=OFF_LOG+2*LOG_BYTES;
constexpr int SMEM=OFF_MISC+MISC_BYTES;

__device__ __forceinline__ uint64_t desc(const bf16* p,int lead,int stride) {
  uint64_t s=__cvta_generic_to_shared(p);
  return ((s&0x3ffff)>>4) | (uint64_t(lead>>4)<<16) | (uint64_t(stride>>4)<<32) | (uint64_t(1)<<62);
}
// K-major 128B-swizzled tiles: ROWS x 128 B per 64-wide k half.
template<int ROWS>
__device__ __forceinline__ int koff(int row,int k) {
  int offset=row*64+(k&63)+(k>>6)*(ROWS*64);
  return offset^(((offset>>6)&7)<<3);
}
// P^T tile: 96 query rows x 64 keys (one 128 B swizzle atom per row).
__device__ __forceinline__ int poff(int q,int key) {return q*64+(key^((q&7)*8));}
__device__ __forceinline__ uint32_t pack2(float a,float b) {
  union {__nv_bfloat162 v;uint32_t u;} p;p.v=__floats2bfloat162_rn(a,b);return p.u;
}
__device__ __forceinline__ float ex2(float x) {
  float y; asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y;
}
__device__ __forceinline__ void tma_load(bf16* dst,const CUtensorMap* map,uint32_t bar,int x,int row) {
  uint32_t sp=__cvta_generic_to_shared(dst);
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"::"r"(sp),"l"(map),"r"(x),"r"(row),"r"(bar):"memory");
}
__device__ __forceinline__ void wait_bar(uint32_t bar,int phase) {
  uint32_t ready;
  do {
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p,[%1],%2; selp.u32 %0,1,0,p; }":"=r"(ready):"r"(bar),"r"(phase):"memory");
  } while(!ready);
}
__device__ __forceinline__ void arrive_bar(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(bar):"memory");
}
// Chunk c belongs to warpgroup c&1 and uses that warpgroup's stage (c>>1)&1;
// the k-th use of a stage is chunk c with k = c>>2.
__device__ __forceinline__ int stage_of(int c) {return 2*(c&1)+((c>>1)&1);}
__device__ __forceinline__ void wg_bar(int id) {
  asm volatile("bar.sync %0,128;"::"r"(id):"memory");
}

__global__ __launch_bounds__(THREADS,1) void route_kernel(
 const bf16* __restrict__ q,const float* __restrict__ bias,bf16* __restrict__ output,int HS,float scale_log2e,
 const __grid_constant__ CUtensorMap map_k,const __grid_constant__ CUtensorMap map_v) {
  extern __shared__ __align__(1024) unsigned char raw[];
  bf16* sq=reinterpret_cast<bf16*>(raw);
  bf16* sk=reinterpret_cast<bf16*>(raw+OFF_RING);
  bf16* sv=sk+S*TILE;
  float* slog_all=reinterpret_cast<float*>(raw+OFF_LOG);
  float* salpha_all=reinterpret_cast<float*>(raw+OFF_MISC);        // [2][NQ]
  float* smax_all=salpha_all+2*NQ;                                  // [2][NQ]
  float* ssum_all=smax_all+2*NQ;                                    // [2][NQ]
  int* sflag=reinterpret_cast<int*>(ssum_all+2*NQ);                 // [2][4]
  uint64_t* full=reinterpret_cast<uint64_t*>(sflag+8);              // [S]
  uint64_t* empty=full+S;                                           // [S]
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,wg=warp>>2,wt=tid&127,seed=lane>>2;
  const int group=blockIdx.x,lh=blockIdx.y,head=HS+lh;
  const int64_t row_base=(int64_t(lh)*GROUPS+group)*KEYS;
  if(tid==0) {
    #pragma unroll
    for(int s=0;s<S;++s) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"((uint32_t)__cvta_generic_to_shared(full+s)):"memory");
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"((uint32_t)__cvta_generic_to_shared(empty+s)):"memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
  }
  // Q^T tile: 96 query rows (90 valid, rest zero).
  for(int vec=tid;vec<NQ*16;vec+=THREADS) {
    int row=vec/16,d=(vec%16)*8;uint4 value=make_uint4(0,0,0,0);
    if(row<90) value=*reinterpret_cast<const uint4*>(q+((int64_t(head)*GROUPS+group)*90+row)*128+d);
    *reinterpret_cast<uint4*>(sq+koff<NQ>(row,d))=value;
  }
  asm volatile("fence.proxy.async.shared::cta;":::"memory");
  __syncthreads();
  if(wg==2) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;"::"n"(ROUTE_PRODUCER_REGS));
    if(warp==8 && lane==0) {
      for(int chunk=0;chunk<CHUNKS;++chunk) {
        int s=stage_of(chunk);
        if(chunk>=S) wait_bar(__cvta_generic_to_shared(empty+s),((chunk>>2)-1)&1);
        uint32_t bar=__cvta_generic_to_shared(full+s);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(bar),"r"(uint32_t(2*TILE*2)):"memory");
        int row=int(row_base+chunk*64);
        tma_load(sk+s*TILE,&map_k,bar,0,row);tma_load(sk+s*TILE+4096,&map_k,bar,64,row);
        tma_load(sv+s*TILE,&map_v,bar,0,row);tma_load(sv+s*TILE+4096,&map_v,bar,64,row);
      }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;"::"n"(ROUTE_CONSUMER_REGS));
  float* slog=slog_all+wg*(64*LOG_STRIDE);
  float* salpha=salpha_all+wg*NQ;float* smax=smax_all+wg*NQ;float* ssum=ssum_all+wg*NQ;
  int* flag=sflag+wg*4;
  const int bar_id=1+wg;
  const float* gbias=bias+row_base;
  // Accumulators: O^T[2 d-halves][64 x 96] and running column stats for query wt.
  float o[2][48];
  #pragma unroll
  for(int h=0;h<2;++h)
    #pragma unroll
    for(int i=0;i<48;++i) o[h][i]=0.f;
  float run_max=-CUDART_INF_F,run_sum=0.f;
  const int kr0=(warp&3)*16+seed,kr1=kr0+8; // this thread's two key rows (0..63) within a chunk
  // Bias for this warpgroup's next chunk, prefetched one iteration ahead.
  auto fetch_bias=[&](int chunk,float& b0,float& b1) {
    int k0=chunk*64+kr0,k1=chunk*64+kr1;
    b0=k0<KEYS?__ldg(gbias+k0):-CUDART_INF_F;
    b1=k1<KEYS?__ldg(gbias+k1):-CUDART_INF_F;
  };
  float nb0,nb1;
  fetch_bias(wg,nb0,nb1);
  using ST=MMA_64x96x16_F32BF16BF16_SS<Major::K,Major::K>;
  using PV=MMA_64x96x16_F32BF16BF16_SS<Major::MN,Major::K>;
  const uint64_t qdesc=desc(sq,16,1024);
  for(int chunk=wg;chunk<CHUNKS;chunk+=2) {
    const int s=stage_of(chunk),phase=(chunk>>2)&1;
    const float b0=nb0,b1=nb1;
    if(chunk+2<CHUNKS) fetch_bias(chunk+2,nb0,nb1);
    wait_bar(__cvta_generic_to_shared(full+s),phase);
    const bf16* tk=sk+s*TILE;const bf16* tv=sv+s*TILE;
    bf16* spt=sk+s*TILE; // P^T reuses this stage's K tile once the S^T GEMM has consumed it
    // S^T = K_tile . Q^T
    float lg[48];
    asm volatile("wgmma.fence.sync.aligned;":::"memory");
    #pragma unroll
    for(int d=0;d<128;d+=16) {
      uint64_t a=desc(tk+koff<64>(0,d),16,1024);
      uint64_t b=qdesc+((d&63)>>3)+((d>>6)*(NQ*64)>>3);
      ST::fma(a,b,E48(lg),d==0?ScaleOut::Zero:ScaleOut::One);
    }
    asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
    // Scale, add key bias, and park the logits row-major [key][query].
    #pragma unroll
    for(int i=0;i<48;i+=2) {
      int kr=((i>>1)&1)?kr1:kr0;float b=((i>>1)&1)?b1:b0;
      int qc=2*(lane&3)+8*(i>>2);
      float2 v=make_float2(__fmaf_rn(lg[i],scale_log2e,__fmul_rn(b,1.4426950408889634f)),
                           __fmaf_rn(lg[i+1],scale_log2e,__fmul_rn(b,1.4426950408889634f)));
      *reinterpret_cast<float2*>(slog+kr*LOG_STRIDE+qc)=v;
    }
    wg_bar(bar_id);
    // Column softmax: thread wt handles query wt (threads 96..127 idle here).
    if(wt<NQ) {
      float l[64];
      #pragma unroll
      for(int k=0;k<64;++k) l[k]=slog[k*LOG_STRIDE+wt];
      float mx=l[0];
      #pragma unroll
      for(int k=1;k<64;++k) mx=fmaxf(mx,l[k]);
      float newmax=fmaxf(run_max,mx);
      float alpha=ex2(__fadd_rn(run_max,-newmax));
      float sum=0.f;
      #pragma unroll
      for(int k=0;k<64;++k) {l[k]=ex2(__fadd_rn(l[k],-newmax));sum=__fadd_rn(sum,l[k]);}
      run_sum=__fmaf_rn(run_sum,alpha,sum);
      salpha[wt]=alpha;
      int changed=newmax!=run_max;
      run_max=newmax;
      // P^T row for query wt: 64 keys as bf16, 16-byte chunks swizzled by (q & 7).
      #pragma unroll
      for(int c=0;c<8;++c) {
        uint4 packed=make_uint4(pack2(l[8*c],l[8*c+1]),pack2(l[8*c+2],l[8*c+3]),pack2(l[8*c+4],l[8*c+5]),pack2(l[8*c+6],l[8*c+7]));
        *reinterpret_cast<uint4*>(spt+poff(wt,8*c))=packed;
      }
      changed=__any_sync(0xffffffff,changed);
      if(lane==0) flag[warp&3]=changed;
    }
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    wg_bar(bar_id);
    if(flag[0]|flag[1]|flag[2]) {
      #pragma unroll
      for(int i=0;i<48;i+=2) {
        int qc=2*(lane&3)+8*(i>>2);
        float2 al=*reinterpret_cast<const float2*>(salpha+qc);
        #pragma unroll
        for(int h=0;h<2;++h) {o[h][i]=__fmul_rn(o[h][i],al.x);o[h][i+1]=__fmul_rn(o[h][i+1],al.y);}
      }
    }
    // O^T += V^T . P^T
    #pragma unroll
    for(int h=0;h<2;++h)
      #pragma unroll
      for(int i=0;i<48;++i) asm volatile("" : "+f"(o[h][i]) :: "memory");
    asm volatile("wgmma.fence.sync.aligned;":::"memory");
    #pragma unroll
    for(int h=0;h<2;++h) {
      #pragma unroll
      for(int ki=0;ki<64;ki+=16) {
        uint64_t a=desc(tv+h*4096+ki*64,8192,1024);
        uint64_t b=desc(spt+ki,16,1024);
        PV::fma(a,b,E48(o[h]),ScaleOut::One);
      }
    }
    asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
    if(wt==0) arrive_bar(__cvta_generic_to_shared(empty+s));
  }
  // Merge the two warpgroups' partial results. WG1 parks its state; WG0 combines.
  float* spart=reinterpret_cast<float*>(sk); // ring is idle once both warpgroups are past their loops
  asm volatile("bar.sync 3,256;":::"memory");
  if(wg==1) {
    #pragma unroll
    for(int h=0;h<2;++h)
      #pragma unroll
      for(int i=0;i<48;++i) spart[((h*48+i)*4+(warp&3))*32+lane]=o[h][i];
    if(wt<NQ) {smax[wt]=run_max;ssum[wt]=run_sum;}
  }
  asm volatile("bar.sync 3,256;":::"memory");
  if(wg==1) return;
  // WG0: fold in WG1's partials (column-wise flash merge), normalize, and stage out^T -> out.
  float* smax1=smax_all+NQ;float* ssum1=ssum_all+NQ;
  if(wt<NQ) {
    float m1=smax1[wt],m=fmaxf(run_max,m1);
    float f0=ex2(__fadd_rn(run_max,-m)),f1=ex2(__fadd_rn(m1,-m));
    float l=__fmaf_rn(run_sum,f0,__fmul_rn(ssum1[wt],f1));
    salpha[wt]=f0;smax[wt]=f1;ssum[wt]=__fdividef(1.f,l); // reuse: f0 in salpha, f1 in smax, 1/l in ssum
  }
  wg_bar(bar_id);
  bf16* sout=reinterpret_cast<bf16*>(slog_all); // 96 x 128 bf16 = 24 KB
  #pragma unroll
  for(int i=0;i<48;i+=2) {
    int qc=2*(lane&3)+8*(i>>2);
    float2 f0=*reinterpret_cast<const float2*>(salpha+qc),f1=*reinterpret_cast<const float2*>(smax+qc),inv=*reinterpret_cast<const float2*>(ssum+qc);
    #pragma unroll
    for(int h=0;h<2;++h) {
      int kr=((i>>1)&1)?kr1:kr0,d=h*64+kr;
      float v0=__fmul_rn(__fmaf_rn(o[h][i],f0.x,__fmul_rn(spart[((h*48+i)*4+(warp&3))*32+lane],f1.x)),inv.x);
      float v1=__fmul_rn(__fmaf_rn(o[h][i+1],f0.y,__fmul_rn(spart[((h*48+i+1)*4+(warp&3))*32+lane],f1.y)),inv.y);
      sout[qc*128+d]=__float2bfloat16_rn(v0);
      sout[(qc+1)*128+d]=__float2bfloat16_rn(v1);
    }
  }
  wg_bar(bar_id);
  for(int vec=wt;vec<90*16;vec+=128) {
    int row=vec/16,d=(vec%16)*8;
    *reinterpret_cast<uint4*>(output+((int64_t(head)*GROUPS+group)*90+row)*128+d)=*reinterpret_cast<const uint4*>(sout+row*128+d);
  }
}

void run(torch::Tensor q,torch::Tensor mk,torch::Tensor mv,torch::Tensor bias,torch::Tensor output,int hs,int ah) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.is_contiguous() && q.numel()==int64_t(AXIS_HEADS)*GROUPS*90*128,"Contiguous native BF16 Q of the compiled shape required");
  TORCH_CHECK(mk.is_contiguous() && mv.is_contiguous() && bias.is_contiguous() && output.is_contiguous(),"Contiguous workspace required");
  TORCH_CHECK(mk.numel()==int64_t(ah)*GROUPS*KEYS*128 && mv.numel()==mk.numel() && bias.numel()==mk.numel()/128,"Workspace shape");
  TORCH_CHECK(output.numel()==q.numel() && hs>=0 && ah==AXIS_ACTIVE_HEADS && hs+ah<=AXIS_HEADS,"Head range");
  CUtensorMap map_k{},map_v{};
  uint64_t rows=uint64_t(ah)*GROUPS*KEYS;
  uint64_t dims[2]={128,rows},strides[1]={256};
  uint32_t box[2]={64,64},element_strides[2]={1,1};
  auto encode=[&](CUtensorMap& map,torch::Tensor x) {
    auto status=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,x.data_ptr(),dims,strides,box,element_strides,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(status==CUDA_SUCCESS,"Route TMA encode failed: ",int(status));
  };
  encode(map_k,mk);encode(map_v,mv);
  C10_CUDA_CHECK(cudaFuncSetAttribute(route_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM));
  dim3 grid(GROUPS,ah);
  route_kernel<<<grid,THREADS,SMEM,at::cuda::getCurrentCUDAStream()>>>(
    (const bf16*)q.data_ptr(),bias.data_ptr<float>(),(bf16*)output.data_ptr(),hs,float(0.08838834764831845*1.4426950408889634),map_k,map_v);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<int> resources() {
  cudaFuncAttributes a{};C10_CUDA_CHECK(cudaFuncGetAttributes(&a,route_kernel));
  int active=0;
  C10_CUDA_CHECK(cudaFuncSetAttribute(route_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,route_kernel,THREADS,SMEM));
  return {a.numRegs,int(a.localSizeBytes),SMEM,active};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("run",&run);m.def("resources",&resources);}
