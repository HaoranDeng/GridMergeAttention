// Q90K80 even-capacity native route, transposed formulation.
// Identical to the Q90K80C14 route (one CTA per (query group, head): two
// consumer warpgroups on alternate key chunks with private two-stage TMA
// rings, a producer warpgroup, S^T = K_tile . Q^T with m64n96k16, column
// softmax through shared memory, O^T += V^T . P^T), except that the merged
// K/V live in the [head][kb][group][CAP][128] layout written by the compact
// merge tiles: a chunk is KBPC = 64/CAP key blocks of CAP representatives,
// fetched by one 5-D TMA box {64 d, CAP, 1 group, KBPC kb, 1 head}. For
// CAP 6 and 10 (ROUTE_PIECEWISE) a chunk is still 64 consecutive keys, fetched
// as a partial head key block, a run of whole key blocks and a partial tail
// key block with exact-height boxes (ROUTE_PIECEWISE=0 falls back to 60-key
// chunks with zeroed pad rows).
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

#ifndef AXIS_CAP
#define AXIS_CAP 8
#endif
#define ROUTE_STAGES 4 // two private stages per consumer warpgroup
#ifndef ROUTE_CONSUMER_REGS
#define ROUTE_CONSUMER_REGS 232
#endif
// ROUTE_LAZY=1: the column softmax runs in the S^T accumulator layout with a
// stale per-query offset m (FA4-style lazy rescaling). Each thread tests its
// own 48 base-2 logits against m + ROUTE_LAZY_TAU; only when some logit in the
// warpgroup exceeds it (always on the first chunk) is the exact chunk maximum
// exchanged through a small table and O / the partial sums rescaled. Then
// p = ex2(t - m) is formed in registers, summed into per-thread partial column
// sums and written as P^T with stmatrix.trans into a per-warpgroup double
// buffer. The 24 KB logits round trip and the one-thread-per-query pass are
// gone; p <= 2^TAU keeps the fp32 sums and bf16 P far from overflow.
#ifndef ROUTE_LAZY
#define ROUTE_LAZY 1
#endif
#ifndef ROUTE_LAZY_TAU
#define ROUTE_LAZY_TAU 8.0f
#endif
#ifndef ROUTE_PRODUCER_REGS
#define ROUTE_PRODUCER_REGS 40
#endif

#define E8(a,o) a[o],a[o+1],a[o+2],a[o+3],a[o+4],a[o+5],a[o+6],a[o+7]
#define E48(a) E8(a,0),E8(a,8),E8(a,16),E8(a,24),E8(a,32),E8(a,40)

constexpr int CAP=AXIS_CAP,T=AXIS_OUTER,KB=T*45,KEYS=KB*CAP,GROUPS=T*40;
// ROUTE_PIECEWISE (capacities that do not divide 64): every chunk holds 64
// consecutive keys of the group's key stream, fetched as up to three TMA boxes
// per half (a partial head key block, a run of whole key blocks, a partial
// tail key block). Otherwise a chunk is KBPC whole key blocks = 64 keys.
#ifndef ROUTE_PIECEWISE
#define ROUTE_PIECEWISE ((64%AXIS_CAP)!=0)
#endif
constexpr int KBPC=64/CAP,KPC=ROUTE_PIECEWISE?64:KBPC*CAP,CHUNKS=ROUTE_PIECEWISE?(KEYS+63)/64:(KB+KBPC-1)/KBPC;
constexpr int NF_HI=64/CAP,NF_LO=NF_HI-1; // whole key blocks per chunk (piecewise): NF_HI or NF_LO
struct RouteMaps { CUtensorMap partial[4]; CUtensorMap full[2]; }; // partial heights 2,4,6,8; full runs NF_LO, NF_HI
constexpr int NQ=96,TILE=64*128,LOG_STRIDE=104; // logits row stride (floats)
constexpr int THREADS=384,S=ROUTE_STAGES;
constexpr int SQ_BYTES=NQ*128*2;            // 24576, Q^T tile
constexpr int RING_BYTES=S*2*TILE*2;         // K+V stages
constexpr int LOG_BYTES=64*LOG_STRIDE*4;     // 26624 per warpgroup
constexpr int MISC_BYTES=2*(NQ*4*3+16)+2*S*8+64;
constexpr int OFF_RING=SQ_BYTES,OFF_LOG=OFF_RING+RING_BYTES,OFF_MISC=OFF_LOG+2*LOG_BYTES;
constexpr int SMEM=OFF_MISC+MISC_BYTES;

__device__ __forceinline__ uint64_t desc(const bf16* p,int lead,int stride) {
  uint64_t s=__cvta_generic_to_shared(p);
  return ((s&0x3ffff)>>4) | (uint64_t(lead>>4)<<16) | (uint64_t(stride>>4)<<32) | (uint64_t(1)<<62);
}
template<int ROWS>
__device__ __forceinline__ int koff(int row,int k) {
  int offset=row*64+(k&63)+(k>>6)*(ROWS*64);
  return offset^(((offset>>6)&7)<<3);
}
__device__ __forceinline__ int poff(int q,int key) {return q*64+(key^((q&7)*8));}
__device__ __forceinline__ uint32_t pack2(float a,float b) {
  union {__nv_bfloat162 v;uint32_t u;} p;p.v=__floats2bfloat162_rn(a,b);return p.u;
}
__device__ __forceinline__ void stsm_x4_trans(uint32_t addr,uint32_t r0,uint32_t r1,uint32_t r2,uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0],{%1,%2,%3,%4};"::"r"(addr),"r"(r0),"r"(r1),"r"(r2),"r"(r3):"memory");
}
__device__ __forceinline__ float ex2(float x) {
  float y; asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y;
}
__device__ __forceinline__ void tma_load(bf16* dst,const CUtensorMap* map,uint32_t bar,int x,int group,int kb0,int lh,int c0=0) {
  uint32_t sp=__cvta_generic_to_shared(dst);
  asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3,%4,%5,%6}],[%7];"
               ::"r"(sp),"l"(map),"r"(x),"r"(c0),"r"(group),"r"(kb0),"r"(lh),"r"(bar):"memory");
}
__device__ __forceinline__ const CUtensorMap* partial_map(const RouteMaps& m,int rows) {
  switch(rows) {case 2:return &m.partial[0];case 4:return &m.partial[1];case 6:return &m.partial[2];default:return &m.partial[3];}
}
// Load 64 consecutive keys starting at key k0 of this group's stream into a
// 64-row stage half (x = 0 or 64 selects the d half).
__device__ __forceinline__ void load_piecewise(bf16* dst,const RouteMaps& m,uint32_t bar,int x,int group,int k0,int lh) {
  const int kbf=k0/CAP,cf=k0%CAP;
  int row=0,kb=kbf;
  if(cf) {tma_load(dst,partial_map(m,CAP-cf),bar,x,group,kb,lh,cf);row=CAP-cf;++kb;}
  const int nf=(64-row)/CAP;
  tma_load(dst+row*64,nf==NF_HI?&m.full[1]:&m.full[0],bar,x,group,kb,lh);
  row+=nf*CAP;kb+=nf;
  if(row<64) tma_load(dst+row*64,partial_map(m,64-row),bar,x,group,kb,lh);
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
__device__ __forceinline__ int stage_of(int c) {return 2*(c&1)+((c>>1)&1);}
__device__ __forceinline__ void wg_bar(int id) {
  asm volatile("bar.sync %0,128;"::"r"(id):"memory");
}

__global__ __launch_bounds__(THREADS,1) void route_kernel(
 const bf16* __restrict__ q,const float* __restrict__ bias,bf16* __restrict__ output,int HS,float scale_log2e,
 const __grid_constant__ CUtensorMap map_k,const __grid_constant__ CUtensorMap map_v,
 const __grid_constant__ RouteMaps maps_k,const __grid_constant__ RouteMaps maps_v) {
  extern __shared__ __align__(1024) unsigned char raw[];
  bf16* sq=reinterpret_cast<bf16*>(raw);
  bf16* sk=reinterpret_cast<bf16*>(raw+OFF_RING);
  bf16* sv=sk+S*TILE;
  float* slog_all=reinterpret_cast<float*>(raw+OFF_LOG);
  float* salpha_all=reinterpret_cast<float*>(raw+OFF_MISC);
  float* smax_all=salpha_all+2*NQ;
  float* ssum_all=smax_all+2*NQ;
  int* sflag=reinterpret_cast<int*>(ssum_all+2*NQ);
  uint64_t* full=reinterpret_cast<uint64_t*>(sflag+8);
  uint64_t* empty=full+S;
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,wg=warp>>2,wt=tid&127,seed=lane>>2;
  const int group=blockIdx.x,lh=blockIdx.y,head=HS+lh;
  if(tid==0) {
    #pragma unroll
    for(int s=0;s<S;++s) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"((uint32_t)__cvta_generic_to_shared(full+s)):"memory");
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"((uint32_t)__cvta_generic_to_shared(empty+s)):"memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
  }
  for(int vec=tid;vec<NQ*16;vec+=THREADS) {
    int row=vec/16,d=(vec%16)*8;uint4 value=make_uint4(0,0,0,0);
    if(row<90) value=*reinterpret_cast<const uint4*>(q+((int64_t(head)*GROUPS+group)*90+row)*128+d);
    *reinterpret_cast<uint4*>(sq+koff<NQ>(row,d))=value;
  }
  if constexpr(KPC<64 && !ROUTE_PIECEWISE) {
    // Rows KPC..63 of every K/V stage half are never written by the TMA box.
    constexpr int per=(64-KPC)*8;
    for(int e=tid;e<S*4*per;e+=THREADS) {
      int unit=e/per,w=e%per,stage=unit>>2,kind=(unit>>1)&1,half=unit&1;
      bf16* base=(kind?sv:sk)+stage*TILE+half*4096+KPC*64;
      reinterpret_cast<uint4*>(base)[w]=make_uint4(0,0,0,0);
    }
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
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(bar),"r"(uint32_t(4*KPC*128)):"memory");
        if constexpr(ROUTE_PIECEWISE) {
          const int k0=chunk*64;
          load_piecewise(sk+s*TILE,maps_k,bar,0,group,k0,lh);load_piecewise(sk+s*TILE+4096,maps_k,bar,64,group,k0,lh);
          load_piecewise(sv+s*TILE,maps_v,bar,0,group,k0,lh);load_piecewise(sv+s*TILE+4096,maps_v,bar,64,group,k0,lh);
        } else {
          int kb0=chunk*KBPC;
          tma_load(sk+s*TILE,&map_k,bar,0,group,kb0,lh);tma_load(sk+s*TILE+4096,&map_k,bar,64,group,kb0,lh);
          tma_load(sv+s*TILE,&map_v,bar,0,group,kb0,lh);tma_load(sv+s*TILE+4096,&map_v,bar,64,group,kb0,lh);
        }
      }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;"::"n"(ROUTE_CONSUMER_REGS));
  float* slog=slog_all+wg*(64*LOG_STRIDE);
  float* salpha=salpha_all+wg*NQ;float* smax=smax_all+wg*NQ;float* ssum=ssum_all+wg*NQ;
  int* flag=sflag+wg*4;
  const int bar_id=1+wg;
  const float* gbias=bias+(int64_t(lh)*GROUPS+group)*KEYS;
  float o[2][48];
  #pragma unroll
  for(int h=0;h<2;++h)
    #pragma unroll
    for(int i=0;i<48;++i) o[h][i]=0.f;
  float run_max=-CUDART_INF_F,run_sum=0.f;
#if ROUTE_LAZY
  // Per-warpgroup scratch in the logits area (unused in this mode): two
  // 12 KB P^T buffers, a [4 warp][96] reduction table, parity-buffered flags.
  unsigned char* lazy_base=raw+OFF_LOG;
  bf16* ptbuf=reinterpret_cast<bf16*>(lazy_base+wg*2*12288);
  float* red=reinterpret_cast<float*>(lazy_base+4*12288)+wg*4*NQ;
  int* lflag=reinterpret_cast<int*>(lazy_base+4*12288+2*4*NQ*4)+wg*8;
  // Column slot c (0..23) of this thread is query 2*(lane&3)+8*(c>>1)+(c&1);
  // accumulator element i belongs to slot 2*(i>>2)+(i&1).
  float mcol[24],lsum[24];
  #pragma unroll
  for(int c=0;c<24;++c) {mcol[c]=-CUDART_INF_F;lsum[c]=0.f;}
#endif
  const int kr0=(warp&3)*16+seed,kr1=kr0+8;
  auto fetch_bias=[&](int chunk,float& b0,float& b1) {
    int k0=chunk*KPC+kr0,k1=chunk*KPC+kr1;
    b0=(kr0<KPC && k0<KEYS)?__ldg(gbias+k0):-CUDART_INF_F;
    b1=(kr1<KPC && k1<KEYS)?__ldg(gbias+k1):-CUDART_INF_F;
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
#if !ROUTE_LAZY
    bf16* spt=sk+s*TILE;
#endif
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
#if ROUTE_LAZY
    const int par=(chunk>>1)&1;
    {
      const float bl0=__fmul_rn(b0,1.4426950408889634f),bl1=__fmul_rn(b1,1.4426950408889634f);
      bool big=false;
      #pragma unroll
      for(int i=0;i<48;++i) {
        const int c=2*(i>>2)+(i&1);
        lg[i]=__fmaf_rn(lg[i],scale_log2e,((i>>1)&1)?bl1:bl0);
        big|=__fadd_rn(lg[i],-mcol[c])>ROUTE_LAZY_TAU;
      }
      big=__any_sync(0xffffffff,big);
      if(lane==0) lflag[par*4+(warp&3)]=big;
    }
    wg_bar(bar_id);
    if(lflag[par*4]|lflag[par*4+1]|lflag[par*4+2]|lflag[par*4+3]) {
      float cm[24];
      #pragma unroll
      for(int c=0;c<24;++c) {
        const int i0=4*(c>>1)+(c&1);
        float x=fmaxf(lg[i0],lg[i0+2]);
        x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,4));
        x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,8));
        x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,16));
        cm[c]=x;
      }
      if(lane<4) {
        #pragma unroll
        for(int c=0;c<24;++c) red[(warp&3)*NQ+2*lane+8*(c>>1)+(c&1)]=cm[c];
      }
      wg_bar(bar_id);
      #pragma unroll
      for(int c=0;c<24;++c) {
        const int qc=2*(lane&3)+8*(c>>1)+(c&1);
        const float x=fmaxf(fmaxf(red[qc],red[NQ+qc]),fmaxf(red[2*NQ+qc],red[3*NQ+qc]));
        const float nm=fmaxf(mcol[c],x);
        const float alpha=(nm==-CUDART_INF_F)?1.f:ex2(__fadd_rn(mcol[c],-nm));
        mcol[c]=nm;
        lsum[c]=__fmul_rn(lsum[c],alpha);
        cm[c]=alpha;
      }
      #pragma unroll
      for(int h=0;h<2;++h)
        #pragma unroll
        for(int i=0;i<48;++i) o[h][i]=__fmul_rn(o[h][i],cm[2*(i>>2)+(i&1)]);
    }
    {
      bf16* spl=ptbuf+par*(NQ*64);
      const uint32_t pt_sbase=__cvta_generic_to_shared(spl);
      const int L=lane>>3,jr=lane&7;
      #pragma unroll
      for(int g=0;g<6;++g) {
        uint32_t reg[4];
        #pragma unroll
        for(int m=0;m<4;++m) {
          const int jm=2*g+(m>>1),rm=m&1,i0=4*jm+2*rm;
          const int c0=2*jm,c1=2*jm+1;
          // A column whose offset is still -inf has only -inf logits so far.
          const float p0=mcol[c0]==-CUDART_INF_F?0.f:ex2(__fadd_rn(lg[i0],-mcol[c0]));
          const float p1=mcol[c1]==-CUDART_INF_F?0.f:ex2(__fadd_rn(lg[i0+1],-mcol[c1]));
          lsum[c0]=__fadd_rn(lsum[c0],p0);lsum[c1]=__fadd_rn(lsum[c1],p1);
          reg[m]=pack2(p0,p1);
        }
        const int jq=2*g+(L>>1),r=L&1;
        const int qrow=8*jq+jr,key=(warp&3)*16+8*r;
        stsm_x4_trans(pt_sbase+2*(qrow*64+(key^((qrow&7)*8))),reg[0],reg[1],reg[2],reg[3]);
      }
    }
    bf16* spt=ptbuf+par*(NQ*64);
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    wg_bar(bar_id);
#else
    #pragma unroll
    for(int i=0;i<48;i+=2) {
      int kr=((i>>1)&1)?kr1:kr0;float b=((i>>1)&1)?b1:b0;
      int qc=2*(lane&3)+8*(i>>2);
      float2 v=make_float2(__fmaf_rn(lg[i],scale_log2e,__fmul_rn(b,1.4426950408889634f)),
                           __fmaf_rn(lg[i+1],scale_log2e,__fmul_rn(b,1.4426950408889634f)));
      *reinterpret_cast<float2*>(slog+kr*LOG_STRIDE+qc)=v;
    }
    wg_bar(bar_id);
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
#endif
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
#if ROUTE_LAZY
  {
    #pragma unroll
    for(int c=0;c<24;++c) {
      float x=lsum[c];
      x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,4));
      x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,8));
      x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,16));
      lsum[c]=x;
    }
    wg_bar(bar_id);
    if(lane<4) {
      #pragma unroll
      for(int c=0;c<24;++c) {
        const int qc=2*lane+8*(c>>1)+(c&1);
        red[(warp&3)*NQ+qc]=lsum[c];
        if((warp&3)==0) smax[qc]=mcol[c];
      }
    }
    wg_bar(bar_id);
    if(wt<NQ) {
      run_sum=__fadd_rn(__fadd_rn(red[wt],red[NQ+wt]),__fadd_rn(red[2*NQ+wt],red[3*NQ+wt]));
      run_max=smax[wt];
    }
    wg_bar(bar_id);
  }
#endif
  float* spart=reinterpret_cast<float*>(sk);
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
  float* smax1=smax_all+NQ;float* ssum1=ssum_all+NQ;
  if(wt<NQ) {
    float m1=smax1[wt],m=fmaxf(run_max,m1);
    float f0=ex2(__fadd_rn(run_max,-m)),f1=ex2(__fadd_rn(m1,-m));
    float l=__fmaf_rn(run_sum,f0,__fmul_rn(ssum1[wt],f1));
    salpha[wt]=f0;smax[wt]=f1;ssum[wt]=__fdividef(1.f,l);
  }
  wg_bar(bar_id);
  bf16* sout=reinterpret_cast<bf16*>(slog_all);
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
  CUtensorMap map_k{},map_v{};RouteMaps maps_k{},maps_v{};
  uint64_t dims[5]={128,uint64_t(CAP),uint64_t(GROUPS),uint64_t(KB),uint64_t(ah)};
  uint64_t strides[4]={256,uint64_t(CAP)*256,uint64_t(GROUPS)*CAP*256,uint64_t(KB)*GROUPS*CAP*256};
  uint32_t element_strides[5]={1,1,1,1,1};
  auto encode=[&](CUtensorMap& map,torch::Tensor x,uint32_t reps,uint32_t blocks) {
    uint32_t box[5]={64,reps,1,blocks,1};
    auto status=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,5,x.data_ptr(),dims,strides,box,element_strides,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(status==CUDA_SUCCESS,"Route TMA encode failed: ",int(status));
  };
  encode(map_k,mk,CAP,KBPC);encode(map_v,mv,CAP,KBPC);
  if(ROUTE_PIECEWISE) {
    for(int i=0;i<4;++i) {uint32_t rows=2*(i+1);if(rows<CAP) {encode(maps_k.partial[i],mk,rows,1);encode(maps_v.partial[i],mv,rows,1);}}
    encode(maps_k.full[0],mk,CAP,NF_LO);encode(maps_v.full[0],mv,CAP,NF_LO);
    encode(maps_k.full[1],mk,CAP,NF_HI);encode(maps_v.full[1],mv,CAP,NF_HI);
  }
  C10_CUDA_CHECK(cudaFuncSetAttribute(route_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM));
  dim3 grid(GROUPS,ah);
  route_kernel<<<grid,THREADS,SMEM,at::cuda::getCurrentCUDAStream()>>>(
    (const bf16*)q.data_ptr(),bias.data_ptr<float>(),(bf16*)output.data_ptr(),hs,float(0.08838834764831845*1.4426950408889634),map_k,map_v,maps_k,maps_v);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<int> resources() {
  cudaFuncAttributes a{};C10_CUDA_CHECK(cudaFuncGetAttributes(&a,route_kernel));
  int active=0;
  C10_CUDA_CHECK(cudaFuncSetAttribute(route_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,route_kernel,THREADS,SMEM));
  return {a.numRegs,int(a.localSizeBytes),SMEM,active,CHUNKS,KPC};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("run",&run);m.def("resources",&resources);}
