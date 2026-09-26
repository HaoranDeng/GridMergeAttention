// Q80K90 exact even capacities 4, 6, 8, 10: compact merge tiles.
// Port of analysis/q90k80_even_merge.cu (TMA tensor stores from a 128B-swizzled
// stmatrix staging buffer, seeds and next-tile queries prefetched one tile
// ahead, fused 32-bit ownership argmax, 3 CTAs/SM) to the Q80K90 grid:
//  - one CTA per (key block kb = frame*40 + pair, head, tile slice); the key
//    block is the column pair `pair` of frame `frame`: 90 keys, key j = token
//    (row j%45, column 2*pair + j/45), gathered from the caller's [1, N, H, 128]
//    K and V (token-major or head-major storage, given by element strides) into
//    shared memory (N padded to 96 with zero rows, masked to -inf);
//  - one 64-row WGMMA tile holds GPT query groups (rows of a frame, flattened
//    g = frame*45 + row) of CAP representatives. Lane pair p (= warp*8 +
//    lane/4) owns rows (p, half 0) and (p, half 1): group = p / (CAP/2),
//    representative = 2*(p % (CAP/2)) + half;
//  - the score GEMM is m64n96k16 (RS, Q gathered from the fragment-ordered
//    [H, N, 128] copy), the merge GEMMs m64n128k16 over K = 96;
//  - mk/mv are laid out [head][kb][group][CAP][128], so a tile is one
//    contiguous run of GPT*CAP rows, stored directly from the accumulators
//    (MERGE_PSI_V, default for mv) or through stmatrix staging and TMA;
//  - the ownership argmax reduces over a group's CAP rows: an in-thread min of
//    the two halves, an in-warp exchange among the group's lane pairs, and
//    (MERGE_PACKED with CAP 6 or 10) a shared-memory exchange between the two
//    warps of a group whose lane pairs cross a warp boundary.
#ifndef AXIS_CAP
#define AXIS_CAP 8
#endif
// MERGE_PACKED=1 packs groups across warp boundaries (CAP 6: 10 groups per
// tile, CAP 10: 6 per tile). MERGE_PACKED=0 keeps every group inside one warp
// (CAP 6: 8 per tile, CAP 10: 4 per tile). CAP 4 and 8 are identical either way.
#ifndef MERGE_PACKED
#define MERGE_PACKED 1
#endif
#ifndef MERGE_STAGE_BUFS
#define MERGE_STAGE_BUFS 1
#endif
#ifndef MERGE_MINBLOCKS
#define MERGE_MINBLOCKS 3
#endif
#ifndef MERGE_PREFETCH_Q
#define MERGE_PREFETCH_Q 1
#endif
// MERGE_PROF=1: warp 0 lane 0 accumulates clock64 phase deltas per tile and
// adds them to prof[0..7] at the end (diagnostics; run_prof binding).
#ifndef MERGE_PROF
#define MERGE_PROF 0
#endif
// MERGE_GATHER_LATE=1 issues the next tile's query gather after the
// ownership keys (before the straddle barrier) instead of right after the
// score WGMMA.
#ifndef MERGE_GATHER_LATE
#define MERGE_GATHER_LATE 1
#endif
// MERGE_SUM_WAYS: independent partial sums in reduce/dot (1 = the serial
// 24-term chain; 2 or 4 shorten the dependency chain at a different rounding
// order). 2 is the default: interleaved A/B on synthetic QKV, merge -0.9 ms
// (C8, 19/24 rounds), -0.6 ms (C10, 16/16), neutral for C6; merged K
// rel-L2 1.8e-4 against the serial order. 4 is slower (C8 +1.6 ms, spills).
#ifndef MERGE_SUM_WAYS
#define MERGE_SUM_WAYS 2
#endif
#ifndef MERGE_SEED_PREFETCH
#define MERGE_SEED_PREFETCH 1
#endif
// MERGE_SPLIT_ISSUE=1: thread 0 issues the mk stores and thread 64 the mv
// stores (each waits for its own bulk groups), so the issue latency is not
// concentrated on warp 0.
#ifndef MERGE_SPLIT_ISSUE
#define MERGE_SPLIT_ISSUE 1
#endif
// MERGE_PSI_K / MERGE_PSI_V = 1: the K / V tile is stored in shared memory
// with its columns permuted by psi(n) = swap of n[4:3] with n[2:1] (a fixed
// involution within each 32-column block). The merge GEMM's accumulator
// column m then holds the representative's natural column psi(m), so a
// thread's eight values of one (row half, 32-column block) are one
// contiguous 16-byte vector of the natural row: mk/mv leave through direct
// global stores (64-byte lane-quad runs, full sectors), and the stmatrix
// staging buffer, its two barriers, the async-proxy fence and the TMA store
// disappear for that kind. With MERGE_PSI_K the score GEMM reduces over the
// permuted columns, which is exactly compensated by gathering the A fragment
// from the caller's natural-order Q instead of the fragment-ordered copy (the
// copy's 4x4 word transpose is psi itself), so the copy and its layout kernel
// are not needed. mv is bitwise identical to the staged path; the score
// GEMM's k-slabs group different columns, so ownership can differ on exact
// ties (numerics tier, like the fused argmax).
// Measured (C8, interleaved on one QKV): V direct is bitwise identical and
// -0.95 ms per call (15/20 rounds) in the benchmark state, neutral under a
// sustained power load; K direct as well is -1.2 ms / +0.8 ms under load
// (the raw token-major Q gather has worse DRAM locality than the copy).
// Defaults: V direct, K staged with the fragment-ordered copy.
#ifndef MERGE_PSI_K
#define MERGE_PSI_K 0
#endif
#ifndef MERGE_PSI_V
#define MERGE_PSI_V 1
#endif
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
constexpr int C=90;                           // keys per block (two columns x 45 rows)
constexpr int KC=96;                          // padded to the WGMMA N granularity
constexpr int CAP=AXIS_CAP;
static_assert(CAP%2==0 && CAP>=4 && CAP<=10,"even capacity between 4 and 10");
constexpr int LP=CAP/2;                       // lane pairs per group
constexpr int GPW=8/LP;                       // whole groups per warp (aligned mode)
constexpr int GPP=16/LP;                      // whole groups per warp pair (MERGE_PACKED 2)
constexpr int GPT=MERGE_PACKED==2?(2*GPP):MERGE_PACKED?(32/LP):(4*GPW); // groups per 64-row tile
constexpr int ROWS_STORED=GPT*CAP;
constexpr int NTOK=AXIS_OUTER*3600;
constexpr int TOTAL_GROUPS=AXIS_OUTER*45;     // query rows, flattened frame*45 + row
constexpr int KB=AXIS_OUTER*40;               // key blocks, flattened frame*40 + pair
constexpr int TILES=(TOTAL_GROUPS+GPT-1)/GPT;
constexpr bool STRADDLE=MERGE_PACKED && (8%LP)!=0;
constexpr int XBUF_WORDS=STRADDLE?GPT*2*4*(KC/4):0;
constexpr int STAGE_KINDS=(MERGE_PSI_K?0:1)+(MERGE_PSI_V?0:1); // kinds stored through the staging buffer
constexpr int TEMP_BYTES=STAGE_KINDS?MERGE_STAGE_BUFS*64*128*2:0;
static_assert(ROWS_STORED<=64,"tile rows");
using namespace cute::SM90::GMMA;

#define E8(a,o) a[o],a[o+1],a[o+2],a[o+3],a[o+4],a[o+5],a[o+6],a[o+7]
#define E32(a) E8(a,0),E8(a,8),E8(a,16),E8(a,24)
#define E48(a) E32(a),E8(a,32),E8(a,40)
#define E64(a) E32(a),E8(a,32),E8(a,40),E8(a,48),E8(a,56)

__device__ __forceinline__ uint64_t desc(const bf16* p,int lead,int stride) {
  uint64_t s=__cvta_generic_to_shared(p);
  return ((s&0x3ffff)>>4) | (uint64_t(lead>>4)<<16) | (uint64_t(stride>>4)<<32) | (uint64_t(1)<<62);
}
__device__ __forceinline__ uint64_t desc_at(uint64_t base,int element_offset) {
  return base+(uint64_t(element_offset)>>3);
}
__device__ __forceinline__ int boff(int k,int n) {
  int offset=k*64+(n&63)+(n>>6)*(KC*64);
  return offset^(((offset>>6)&7)<<3);
}
__device__ __forceinline__ int psi(int n) {return (n&~30)|((n&6)<<2)|((n&24)>>2);}
// One 16-byte vector (columns d..d+7 of key row ki) into the swizzled tile;
// with PSI the columns land at psi(d+2j) (pairs stay adjacent).
template<int PSI>
__device__ __forceinline__ void store_tile_vector(bf16* tile,int ki,int d,uint4 x) {
  if constexpr(PSI) {
    const uint32_t w[4]={x.x,x.y,x.z,x.w};
    #pragma unroll
    for(int j=0;j<4;++j) *reinterpret_cast<uint32_t*>(tile+boff(ki,psi(d+2*j)))=w[j];
  } else *reinterpret_cast<uint4*>(tile+boff(ki,d))=x;
}
// Lane-pair geometry (compile-time tables evaluated per thread once).
__device__ __forceinline__ int grp_of(int p) {
#if MERGE_PACKED==2
  int q=p&15;return q<GPP*LP?(p>>4)*GPP+q/LP:-1;
#elif MERGE_PACKED
  return p<GPT*LP?p/LP:-1;
#else
  int s=p&7;return s<GPW*LP?(p>>3)*GPW+s/LP:-1;
#endif
}
__device__ __forceinline__ int idx_of(int p) {
#if MERGE_PACKED==2
  return (p&15)%LP;
#elif MERGE_PACKED
  return p%LP;
#else
  return (p&7)%LP;
#endif
}
__device__ __forceinline__ int pstart_of(int grp) {
#if MERGE_PACKED==2
  return (grp/GPP)*16+(grp%GPP)*LP;
#elif MERGE_PACKED
  return grp*LP;
#else
  return (grp/GPW)*8+(grp%GPW)*LP;
#endif
}
// Staging row of (lane pair, half): group-major, representative-minor; pad
// rows follow the stored rows.
__device__ __forceinline__ int srow_of(int p,int half) {
  int g=grp_of(p);
  if(g>=0) return g*CAP+idx_of(p)*2+half;
#if MERGE_PACKED==2
  int wp=p>>4,q=p&15;return ROWS_STORED+(wp*(16-GPP*LP)+(q-GPP*LP))*2+half;
#elif MERGE_PACKED
  return ROWS_STORED+(p-GPT*LP)*2+half;
#else
  int w=p>>3,s=p&7;return ROWS_STORED+(w*(8-GPW*LP)+(s-GPW*LP))*2+half;
#endif
}
__device__ __forceinline__ uint4 query_vector(const bf16* p,uint64_t policy) {
  uint4 a;
  asm volatile("ld.global.L1::evict_last.L2::cache_hint.v4.b32 {%0,%1,%2,%3},[%4],%5;":"=r"(a.x),"=r"(a.y),"=r"(a.z),"=r"(a.w):"l"(p),"l"(policy));
  return a;
}
__device__ __forceinline__ float ex2(float x) {
  float y; asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y;
}
__device__ __forceinline__ float lg2(float x) {
  #if AXIS_FAST_MATH
  float y;asm("lg2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;
  #else
  return log2f(x);
  #endif
}
__device__ __forceinline__ float sum4(float x) {
  x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,2,4));
  return __fadd_rn(x,__shfl_xor_sync(0xffffffff,x,1,4));
}
__device__ __forceinline__ float max4(float x) {
  x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,1,4));
  return fmaxf(x,__shfl_xor_sync(0xffffffff,x,2,4));
}
__device__ __forceinline__ float reduce_row(float (&p)[KC/4]) {
#if MERGE_SUM_WAYS>1
  float part[MERGE_SUM_WAYS];
  #pragma unroll
  for(int w=0;w<MERGE_SUM_WAYS;++w) part[w]=p[w];
  #pragma unroll
  for(int i=MERGE_SUM_WAYS;i<KC/4;++i) part[i%MERGE_SUM_WAYS]+=p[i];
  #pragma unroll
  for(int w=MERGE_SUM_WAYS/2;w>0;w/=2)
    #pragma unroll
    for(int j=0;j<w;++j) part[j]+=part[j+w];
  return sum4(part[0]);
#else
  float sum=p[0];
  #pragma unroll
  for(int i=1;i<KC/4;++i)sum+=p[i];
  return sum4(sum);
#endif
}
__device__ __forceinline__ uint32_t owner_key(float neg_s,uint32_t owner) {
  uint32_t y;
  asm("lop3.b32 %0,%1,0xfffffff0,%2,0xea;":"=r"(y):"r"(__float_as_uint(neg_s)),"r"(owner));
  return y;
}
__device__ __forceinline__ void stsm_x4(uint32_t addr,uint32_t r0,uint32_t r1,uint32_t r2,uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0],{%1,%2,%3,%4};"::"r"(addr),"r"(r0),"r"(r1),"r"(r2),"r"(r3):"memory");
}
__device__ __forceinline__ uint32_t pack_bf162(float a,float b) {
  union {__nv_bfloat162 v;uint32_t u;} p;p.v=__floats2bfloat162_rn(a,b);return p.u;
}
__device__ __forceinline__ float reciprocal(float x){float y;
  #if AXIS_FAST_MATH
  asm("rcp.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));
  #else
  asm("div.full.f32 %0,0f3F800000,%1;":"=f"(y):"f"(x));
  #endif
  return y;}
__device__ __forceinline__ float dot_row(float (&p)[KC/4],float (&x)[KC/4]){
#if MERGE_SUM_WAYS>1
  float part[MERGE_SUM_WAYS];
  #pragma unroll
  for(int w=0;w<MERGE_SUM_WAYS;++w) part[w]=p[w]*x[w];
  #pragma unroll
  for(int i=MERGE_SUM_WAYS;i<KC/4;++i) part[i%MERGE_SUM_WAYS]=fmaf(p[i],x[i],part[i%MERGE_SUM_WAYS]);
  #pragma unroll
  for(int w=MERGE_SUM_WAYS/2;w>0;w/=2)
    #pragma unroll
    for(int j=0;j<w;++j) part[j]+=part[j+w];
  return sum4(part[0]);
#else
  float sum=p[0]*x[0];
  #pragma unroll
  for(int i=1;i<KC/4;++i)sum=fmaf(p[i],x[i],sum);
  return sum4(sum);
#endif
}
template<int TOTAL_HEADS>
__global__ __launch_bounds__(128,MERGE_MINBLOCKS) void merge_kernel(
 const bf16* __restrict__ q,const bf16* __restrict__ k,const bf16* __restrict__ v,
 const uint8_t* __restrict__ seeds,float* __restrict__ bias,int HS,float scale,
 int tile_count,const __grid_constant__ CUtensorMap map_k,const __grid_constant__ CUtensorMap map_v,long long* __restrict__ prof,
 bf16* __restrict__ mk,bf16* __restrict__ mv,int64_t q_ts,int64_t q_hs,int64_t k_ts,int64_t k_hs,int64_t v_ts,int64_t v_hs) {
  long long pacc[8]={0,0,0,0,0,0,0,0};long long pt=0;
  auto stamp=[&](int slot){ if(MERGE_PROF && threadIdx.x==0){long long now=clock64();pacc[slot]+=now-pt;pt=now;} };
  extern __shared__ __align__(1024) unsigned char raw[];
  bf16* sk=reinterpret_cast<bf16*>(raw);
  bf16* sv=sk+KC*128;
  bf16* temp=sv+KC*128;
  uint32_t* xbuf=reinterpret_cast<uint32_t*>(raw+KC*128*4+TEMP_BYTES);
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,sl=lane>>2;
  const int kb=blockIdx.x,head=blockIdx.y,gh=HS+head;
  const int p=warp*8+sl,grp=grp_of(p),idx=idx_of(p);
  const int owner0=2*idx,owner1=2*idx+1;
  const int pst=grp>=0?pstart_of(grp):p;
  uint64_t cache_last;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0,1.0;":"=l"(cache_last));
  {
    // Key block kb = column pair `pair` of frame `frame`, key j = token (row j%45, column 2*pair+j/45).
    const int frame=kb/40,pair=kb%40;
    for(int vec=tid;vec<KC*16;vec+=128) {
      const int ki=vec/16,d=(vec%16)*8;
      uint4 kx=make_uint4(0,0,0,0),vx=kx;
      if(ki<C) {
        const int64_t token=int64_t(frame)*3600+(ki%45)*80+2*pair+ki/45;
        kx=*reinterpret_cast<const uint4*>(k+token*k_ts+int64_t(gh)*k_hs+d);
        vx=*reinterpret_cast<const uint4*>(v+token*v_ts+int64_t(gh)*v_hs+d);
      }
      store_tile_vector<MERGE_PSI_K>(sk,ki,d,kx);
      store_tile_vector<MERGE_PSI_V>(sv,ki,d,vx);
    }
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;":::"memory");
  const uint64_t score_base=desc(sk,16,1024);
  const uint64_t merge_base_k=desc(sk,KC*128,1024),merge_base_v=desc(sv,KC*128,1024);
  const int start=blockIdx.z*tile_count,stop=min(start+tile_count,TILES);
  uint32_t qreg[32];
  // Seeds for the tile after the one being gathered are loaded one tile
  // ahead, so the dependent seed -> query address -> query load chain has a
  // whole tile of slack (the seed lines do not survive in L1 behind the
  // evict_last query lines).
  int nsr0=0,nsr1=0;
  auto load_seeds=[&](int tile) {
    int g=tile*GPT+grp;
    bool ok=grp>=0 && g<TOTAL_GROUPS && tile<stop;
    int gg=ok?g:0,row=gg%45;
    int sbase=((gh*45+row)*KB+kb)*CAP;
    nsr0=ok?seeds[sbase+owner0]:0;nsr1=ok?seeds[sbase+owner1]:0;
  };
  auto gather=[&](int tile) {
    int g=tile*GPT+grp;
    bool ok=grp>=0 && g<TOTAL_GROUPS;
    int gg=ok?g:0,row=gg%45;
    int sr0,sr1;
    if(MERGE_SEED_PREFETCH) {sr0=nsr0;sr1=nsr1;}
    else {
      int sbase=((gh*45+row)*KB+kb)*CAP;
      sr0=ok?seeds[sbase+owner0]:0;sr1=ok?seeds[sbase+owner1]:0;
    }
    // Query row gg is tokens gg*80 .. gg*80+79 of q: the caller's Q through
    // its token/head strides (MERGE_PSI_K) or the fragment-ordered copy.
    const bf16* q0=q+int64_t(gh)*q_hs+int64_t(gg*80+sr0)*q_ts+(lane&3)*8;
    const bf16* q1=q+int64_t(gh)*q_hs+int64_t(gg*80+sr1)*q_ts+(lane&3)*8;
    #pragma unroll
    for(int d=0;d<128;d+=32) {
      uint4 a=ok?query_vector(q0+d,cache_last):make_uint4(0,0,0,0),b=ok?query_vector(q1+d,cache_last):make_uint4(0,0,0,0);
      int i=d/4;
      qreg[i]=a.x;qreg[i+1]=b.x;qreg[i+2]=a.y;qreg[i+3]=b.y;
      qreg[i+4]=a.z;qreg[i+5]=b.z;qreg[i+6]=a.w;qreg[i+7]=b.w;
    }
    if(MERGE_SEED_PREFETCH) load_seeds(tile+1);
  };
  if(MERGE_SEED_PREFETCH) load_seeds(start);
  if(start<stop) gather(start);
  if(MERGE_PROF && threadIdx.x==0) pt=clock64();
  for(int tile=start;tile<stop;++tile) {
    stamp(7);
    const int g=tile*GPT+grp;
    const bool ok=grp>=0 && g<TOTAL_GROUPS;
    if(!MERGE_PREFETCH_Q && tile>start) gather(tile);
    #pragma unroll
    for(int i=0;i<32;++i) asm volatile("" : "+r"(qreg[i]) :: "memory");
    float s[KC/2];
    asm volatile("wgmma.fence.sync.aligned;":::"memory");
    using Score=MMA_64x96x16_F32BF16BF16_RS<Major::K,Major::K>;
    #pragma unroll
    for(int d=0;d<128;d+=16) {
      int i=d/4;
      Score::fma(qreg[i],qreg[i+1],qreg[i+2],qreg[i+3],desc_at(score_base,boff(0,d)),E48(s),d==0?ScaleOut::Zero:ScaleOut::One);
    }
    asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
    stamp(0);
    if(MERGE_PREFETCH_Q && !MERGE_GATHER_LATE && tile+1<stop) gather(tile+1);
    stamp(1);
    #pragma unroll
    for(int i=0;i<KC/2;++i) {
      int kc=2*(lane&3)+8*(i>>2)+(i&1);
      if(kc>=C)s[i]=-CUDART_INF_F;
    }
    float norm_max[2],norm_sum[2];
    #pragma unroll
    for(int half=0;half<2;++half) {
      float direct[KC/4];
      #pragma unroll
      for(int i=0;i<KC/4;++i)
        direct[i]=ex2(__fmul_rn(s[4*(i/2)+2*half+(i&1)],scale));
      float sum=reduce_row(direct);
      bool direct_ok=__all_sync(0xffffffff,sum>0x1p-64f && sum<0x1p64f);
      if(direct_ok) {
        norm_max[half]=0.f;norm_sum[half]=sum;
      } else {
      float mx=-CUDART_INF_F;
      #pragma unroll
      for(int i=0;i<KC/8;++i) mx=fmaxf(mx,fmaxf(__fmul_rn(s[4*i+2*half],scale),__fmul_rn(s[4*i+2*half+1],scale)));
      mx=max4(mx);
      float bs[KC/4];
      #pragma unroll
      for(int i=0;i<KC/4;++i)
        bs[i]=ex2(__fmaf_rn(s[4*(i/2)+2*half+(i&1)],scale,-mx));
      norm_max[half]=mx;norm_sum[half]=reduce_row(bs);
      }
    }
    float norm_log=lg2((lane&1)?norm_sum[1]:norm_sum[0]);
    #pragma unroll
    for(int half=0;half<2;++half) {
      float norm=__fadd_rn(norm_max[half],__shfl_sync(0xffffffff,norm_log,(lane&~1)+half));
      #pragma unroll
      for(int i=0;i<KC/4;++i) {
        int ix=(i/2)*4+half*2+(i&1);
        s[ix]=__fadd_rn(__fmul_rn(s[ix],scale),-norm);
      }
    }
    // Ownership: per key column, the group's best representative. Normalized
    // scores are <= 0, so the unsigned minimum of bits(s) with the owner id in
    // the low four mantissa bits is the argmax (ties -> smallest owner).
    uint32_t keys[KC/4];
    int labels[KC/4];
    #pragma unroll
    for(int i=0;i<KC/4;++i) {
      int ix=(i/2)*4+(i&1);
      float a=s[ix],b=s[ix+2];
      uint32_t key=ok?min(owner_key(a,owner0),owner_key(b,owner1)):0xffffffffu;
      if constexpr(LP==2) {
        key=min(key,__shfl_xor_sync(0xffffffff,key,4));
      } else if constexpr(LP==4) {
        key=min(key,__shfl_xor_sync(0xffffffff,key,4));
        key=min(key,__shfl_xor_sync(0xffffffff,key,8));
      } else {
        #pragma unroll
        for(int t=1;t<LP;++t) {
          int sp=grp>=0?pst+(idx+t)%LP:p;
          bool inw=(sp>>3)==warp;
          uint32_t o=__shfl_sync(0xffffffff,key,inw?(((sp&7)<<2)|(lane&3)):lane);
          key=inw?min(key,o):key;
        }
      }
      keys[i]=key;
    }
    if constexpr(STRADDLE) {
      const int pend=pst+LP-1;
      const bool strad=grp>=0 && (pst>>3)!=(pend>>3);
      const int side=(p>>3)==(pst>>3)?0:1;
      const bool writer=strad && p==max(pst,warp*8);
      if(writer) {
        uint32_t* mine=xbuf+((grp*2+side)*4+(lane&3))*(KC/4);
        #pragma unroll
        for(int i=0;i<KC/4;++i) mine[i]=keys[i];
      }
      // The next tile's query gather is issued here so its LSU work overlaps
      // the barrier wait instead of the softmax.
      if(MERGE_PREFETCH_Q && MERGE_GATHER_LATE && tile+1<stop) gather(tile+1);
      // Named barriers between adjacent warps that share a group, lower pair first.
#if MERGE_PACKED==2
      // One barrier per warp pair (ids 1 and 2); no warp waits on two pairs.
      asm volatile("bar.sync %0,64;"::"r"(1+(warp>>1)):"memory");
#else
      if(warp>0 && (8*warp)%LP!=0) asm volatile("bar.sync %0,64;"::"r"(warp):"memory");
      if(warp<3 && (8*(warp+1))%LP!=0) asm volatile("bar.sync %0,64;"::"r"(warp+1):"memory");
#endif
      if(strad) {
        const uint32_t* other=xbuf+((grp*2+(side^1))*4+(lane&3))*(KC/4);
        #pragma unroll
        for(int i=0;i<KC/4;++i) keys[i]=min(keys[i],other[i]);
      }
    }
    if(MERGE_PREFETCH_Q && MERGE_GATHER_LATE && !STRADDLE && tile+1<stop) gather(tile+1);
    #pragma unroll
    for(int i=0;i<KC/4;++i) labels[i]=(int)(keys[i]&15u);
    uint32_t packed_weights[KC/4];
    #pragma unroll
    for(int half=0;half<2;++half) {
      float winners[KC/4],probabilities[KC/4];
      const int owner=2*idx+half;
      #pragma unroll
      for(int i=0;i<KC/4;++i) {
        int ix=(i/2)*4+(i&1)+2*half;
        bool valid=2*(lane&3)+8*(i/2)+(i&1)<C;
        winners[i]=valid?s[ix]:0.f;
        probabilities[i]=(valid && ok && labels[i]==owner)?ex2(s[ix]):0.f;
      }
      float sum=reduce_row(probabilities),safe=sum>0?sum:1.f;
      float inv=reciprocal(safe),expected=dot_row(probabilities,winners);
      float correction=__fmul_rn(__fadd_rn(__fmul_rn(expected,inv),-lg2(safe)),0.6931471805599453f);
      int out=((head*TOTAL_GROUPS+(ok?g:0))*KB+kb)*CAP+owner;
      if((lane&3)==0 && ok)bias[out]=sum>0?-correction:-CUDART_INF_F;
      #pragma unroll
      for(int i=0;i<KC/8;++i) {
        union {bf16 values[2];uint32_t bits;} pair;
        pair.values[0]=__float2bfloat16_rn(__fmul_rn(probabilities[2*i],inv));
        pair.values[1]=__float2bfloat16_rn(__fmul_rn(probabilities[2*i+1],inv));
        packed_weights[2*i+half]=pair.bits;
      }
    }
    stamp(2);
    #pragma unroll
    for(int kind=0;kind<2;++kind) {
      asm volatile("wgmma.fence.sync.aligned;":::"memory");
      float acc[64];
      using Merge=MMA_64x128x16_F32BF16BF16_RS<Major::K,Major::MN>;
      #pragma unroll
      for(int ki=0;ki<KC;ki+=16) {
        auto bd=desc_at(kind==0?merge_base_k:merge_base_v,boff(ki,0));
        int wi=(ki/8)*2;
        Merge::fma(packed_weights[wi],packed_weights[wi+1],packed_weights[wi+2],packed_weights[wi+3],bd,E64(acc),ki==0?ScaleOut::Zero:ScaleOut::One);
      }
      asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
      asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
      stamp(3+2*kind);
      if(kind==0?MERGE_PSI_K:MERGE_PSI_V) {
        // Direct stores: accumulator columns 32a+8b+2q+e (b = 0..3, e = 0..1)
        // hold natural columns 32a+8q+2b+e, one 16-byte vector per (row half,
        // 32-column block) at natural offset 32a+8q of the representative's row.
        bf16* dst=kind==0?mk:mv;
        #pragma unroll
        for(int h=0;h<2;++h) {
          if(ok) {
            bf16* rowp=dst+(int64_t(head*KB+kb)*(TOTAL_GROUPS*CAP)+tile*GPT*CAP+srow_of(p,h))*128+(lane&3)*8;
            #pragma unroll
            for(int a=0;a<4;++a) {
              const int i=16*a+2*h;
              *reinterpret_cast<uint4*>(rowp+32*a)=make_uint4(pack_bf162(acc[i],acc[i+1]),pack_bf162(acc[i+4],acc[i+5]),
                                                              pack_bf162(acc[i+8],acc[i+9]),pack_bf162(acc[i+12],acc[i+13]));
            }
          }
        }
        stamp(4+2*kind);
        continue;
      }
      bf16* stage=temp+(MERGE_STAGE_BUFS==2?kind*64*128:0);
      // The staging buffer is free once the bulk store that last read it (the
      // other staged kind's, issued by its own thread) has finished reading it.
      const int issuer=MERGE_SPLIT_ISSUE?(kind?64:0):0,previous=(STAGE_KINDS==2 && MERGE_SPLIT_ISSUE)?(kind?0:64):issuer;
      if(tid==previous) {
        if(MERGE_STAGE_BUFS==2) asm volatile("cp.async.bulk.wait_group.read 1;":::"memory");
        else asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");
      }
      __syncthreads();
      {
        // Instruction g stores the four 8x8 blocks of half g>>2 (d chunks
        // 4*(g&3)+m). Lane l supplies the staging address of the row owned by
        // lane pair warp*8+(l&7) in that half, chunk 4*(g&3)+(l>>3).
        const int r=lane&7,mb=lane>>3;
        const uint32_t sbase=__cvta_generic_to_shared(stage);
        #pragma unroll
        for(int gi=0;gi<8;++gi) {
          const int row=srow_of(warp*8+r,gi>>2);
          int chunk=4*(gi&3)+mb;
          uint32_t addr=sbase+2*((chunk>>3)*4096+row*64+(((chunk&7)^(row&7))*8));
          int i0=4*(4*(gi&3)+0)+2*(gi>>2),i1=4*(4*(gi&3)+1)+2*(gi>>2),i2=4*(4*(gi&3)+2)+2*(gi>>2),i3=4*(4*(gi&3)+3)+2*(gi>>2);
          stsm_x4(addr,pack_bf162(acc[i0],acc[i0+1]),pack_bf162(acc[i1],acc[i1+1]),pack_bf162(acc[i2],acc[i2+1]),pack_bf162(acc[i3],acc[i3+1]));
        }
      }
      asm volatile("fence.proxy.async.shared::cta;":::"memory");
      __syncthreads();
      if(tid==issuer) {
        const int rowc=tile*GPT*CAP,kbc=head*KB+kb;
        const CUtensorMap* map=kind==0?&map_k:&map_v;
        #pragma unroll
        for(int half=0;half<2;++half) {
          uint32_t sp=__cvta_generic_to_shared(stage+half*4096);
          asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0,{%1,%2,%3}],[%4];"::"l"(map),"r"(half*64),"r"(rowc),"r"(kbc),"r"(sp):"memory");
        }
        asm volatile("cp.async.bulk.commit_group;":::"memory");
      }
      stamp(4+2*kind);
    }
  }
  if(STAGE_KINDS>0 && (tid==0 || (MERGE_SPLIT_ISSUE && tid==64))) asm volatile("cp.async.bulk.wait_group 0;":::"memory");
  if(MERGE_PROF && threadIdx.x==0) {
    #pragma unroll
    for(int i=0;i<8;++i) atomicAdd(reinterpret_cast<unsigned long long*>(prof+i),(unsigned long long)pacc[i]);
  }
}

constexpr int SMEM_BYTES=KC*128*4+TEMP_BYTES+XBUF_WORDS*4;
constexpr int NTOKENS=NTOK,NHEADS=AXIS_HEADS;
static void check_input(const torch::Tensor& x,int64_t ts,int64_t hs,const char* name) {
  // Dense [1, N, H, 128] BF16 in token-major (ts = H*128, hs = 128) or head-major (ts = 128, hs = N*128) order.
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kBFloat16 && x.dim()==4 && x.size(0)==1 && x.size(1)==NTOKENS && x.size(2)==NHEADS && x.size(3)==128,
              name," must be CUDA BF16 [1, N, H, 128] of the compiled shape");
  TORCH_CHECK(x.stride(3)==1 && x.stride(1)==ts && x.stride(2)==hs &&
              ((ts==int64_t(NHEADS)*128 && hs==128) || (ts==128 && hs==int64_t(NTOKENS)*128)),name," must be token-major or head-major dense");
}

void run_impl(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor seeds,
 torch::Tensor mk,torch::Tensor mv,torch::Tensor bias,int hs,int ah,int tiles,torch::Tensor prof,
 int64_t q_ts,int64_t q_hs,int64_t k_ts,int64_t k_hs,int64_t v_ts,int64_t v_hs) {
 // q: the caller's Q (MERGE_PSI_K) or the fragment-ordered copy as a head-major [1, N, H, 128] view.
 check_input(q,q_ts,q_hs,"q");check_input(k,k_ts,k_hs,"k");check_input(v,v_ts,v_hs,"v");
 TORCH_CHECK(ah==AXIS_ACTIVE_HEADS && hs>=0 && hs+ah<=AXIS_HEADS && tiles>0,"Head/tile bounds");
 TORCH_CHECK(seeds.scalar_type()==at::kByte && seeds.is_contiguous() && seeds.numel()==int64_t(AXIS_HEADS)*45*KB*CAP,"Exact seed shape");
 TORCH_CHECK(mk.is_contiguous() && mv.is_contiguous() && mk.numel()==int64_t(ah)*KB*TOTAL_GROUPS*CAP*128 && mv.numel()==mk.numel(),"Workspace shape");
 TORCH_CHECK(bias.is_contiguous() && bias.numel()==mk.numel()/128,"Bias shape");
 CUtensorMap map_k{},map_v{};
 uint64_t rows_per_kb=uint64_t(TOTAL_GROUPS)*CAP;
 uint64_t dims[3]={128,rows_per_kb,uint64_t(ah)*KB},strides[2]={256,rows_per_kb*256};
 uint32_t box[3]={64,ROWS_STORED,1},element_strides[3]={1,1,1};
 auto encode=[&](CUtensorMap& map,torch::Tensor x) {
  auto status=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,x.data_ptr(),dims,strides,box,element_strides,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(status==CUDA_SUCCESS,"Merge TMA encode failed: ",int(status));
 };
 encode(map_k,mk);encode(map_v,mv);
 C10_CUDA_CHECK(cudaFuncSetAttribute(merge_kernel<AXIS_HEADS>,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM_BYTES));
 dim3 grid(KB,ah,(TILES+tiles-1)/tiles);
 merge_kernel<AXIS_HEADS><<<grid,128,SMEM_BYTES,at::cuda::getCurrentCUDAStream()>>>(
   (const bf16*)q.data_ptr(),(const bf16*)k.data_ptr(),(const bf16*)v.data_ptr(),seeds.data_ptr<uint8_t>(),
   bias.data_ptr<float>(),hs,float(0.08838834764831845*1.4426950408889634),tiles,map_k,map_v,
   prof.defined()?reinterpret_cast<long long*>(prof.data_ptr<int64_t>()):nullptr,
   (bf16*)mk.data_ptr(),(bf16*)mv.data_ptr(),q_ts,q_hs,k_ts,k_hs,v_ts,v_hs);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void run(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor seeds,
 torch::Tensor mk,torch::Tensor mv,torch::Tensor bias,int hs,int ah,int tiles,int64_t q_ts,int64_t q_hs,int64_t k_ts,int64_t k_hs,int64_t v_ts,int64_t v_hs) {
 run_impl(q,k,v,seeds,mk,mv,bias,hs,ah,tiles,torch::Tensor(),q_ts,q_hs,k_ts,k_hs,v_ts,v_hs);
}
void run_prof(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor seeds,
 torch::Tensor mk,torch::Tensor mv,torch::Tensor bias,int hs,int ah,int tiles,torch::Tensor prof,int64_t q_ts,int64_t q_hs,int64_t k_ts,int64_t k_hs,int64_t v_ts,int64_t v_hs) {
 TORCH_CHECK(prof.is_cuda() && prof.scalar_type()==at::kLong && prof.numel()>=8,"prof tensor");
 run_impl(q,k,v,seeds,mk,mv,bias,hs,ah,tiles,prof,q_ts,q_hs,k_ts,k_hs,v_ts,v_hs);
}
std::vector<int> resources() {
 cudaFuncAttributes a{};C10_CUDA_CHECK(cudaFuncGetAttributes(&a,merge_kernel<AXIS_HEADS>));
 int active=0;
 C10_CUDA_CHECK(cudaFuncSetAttribute(merge_kernel<AXIS_HEADS>,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM_BYTES));
 C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,merge_kernel<AXIS_HEADS>,128,SMEM_BYTES));
 return {a.numRegs,int(a.localSizeBytes),SMEM_BYTES,active,TILES,GPT,MERGE_PSI_K,MERGE_PSI_V};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("run",&run);m.def("run_prof",&run_prof);m.def("resources",&resources);}
