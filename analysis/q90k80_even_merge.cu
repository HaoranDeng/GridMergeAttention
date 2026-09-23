// Q90/K80 exact even capacities 4, 6, 8, 10: compact merge tiles.
// Derived from the validated Q90K80C14 merge kernel (2026-09-21: TMA tensor
// stores from a 128B-swizzled stmatrix staging buffer, next-tile query
// prefetch, 3 CTAs/SM, fused 32-bit ownership argmax). Differences:
//  - One 64-row WGMMA tile holds GPT query groups of CAP representatives.
//    Lane pair p (= warp*8 + lane/4) owns rows (p, half 0) and (p, half 1):
//    group = p / (CAP/2), representative = 2*(p % (CAP/2)) + half. Groups are
//    consecutive flattened (frame, column) indices g = qo*40 + col.
//  - mk/mv are laid out [head][kb][group][CAP][128], so a tile is one
//    contiguous run of GPT*CAP rows and leaves through two TMA stores.
//  - The ownership argmax reduces over a group's CAP rows: an in-thread min
//    of the two halves, an in-warp exchange among the group's lane pairs, and
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
#ifndef MERGE_FUSED_ARGMAX
#define MERGE_FUSED_ARGMAX 1
#endif
#ifndef MERGE_STMATRIX
#define MERGE_STMATRIX 1
#endif
#ifndef MERGE_FFMA_NORM
#define MERGE_FFMA_NORM 0
#endif
// Diagnostics only (wrong results): MERGE_ABL_STORE_SAME sends every tile's
// TMA stores to the same rows (removes DRAM write traffic, keeps the
// instruction stream); MERGE_ABL_GATHER56 skips the query loads of lane
// pairs 6 and 7 in half 1 (the C14 kernel's 56-row gather volume).
// Tile order: 0 walks the flattened group index (tile t = groups t*GPT..);
// 1 keeps a column block fixed and walks the frames first (t -> frame t%21,
// column block t/21), the C14 kernel's order; needs 40 % GPT == 0.
#ifndef MERGE_TILE_ORDER
#define MERGE_TILE_ORDER 0
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
#ifndef MERGE_SEED_PREFETCH
#define MERGE_SEED_PREFETCH 1
#endif
#ifndef MERGE_ABL_STORE_SAME
#define MERGE_ABL_STORE_SAME 0
#endif
// MERGE_SPLIT_ISSUE=1: thread 0 issues the mk stores and thread 64 the mv
// stores (each waits for its own bulk groups), so the issue latency is not
// concentrated on warp 0.
#ifndef MERGE_SPLIT_ISSUE
#define MERGE_SPLIT_ISSUE 1
#endif
// MERGE_STORE_2D=1: 2-D store descriptors (rows flattened over kb) plus a
// tail descriptor for a partial last tile, instead of the 3-D box.
#ifndef MERGE_STORE_2D
#define MERGE_STORE_2D 0
#endif
#ifndef MERGE_ABL_NOSTRADDLE
#define MERGE_ABL_NOSTRADDLE 0
#endif
#ifndef MERGE_ABL_NOBAR
#define MERGE_ABL_NOBAR 0
#endif
#ifndef MERGE_ABL_GATHER56
#define MERGE_ABL_GATHER56 0
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
constexpr int KC=(AXIS_COLS+15)/16*16;
constexpr int CAP=AXIS_CAP;
static_assert(CAP%2==0 && CAP>=4 && CAP<=10,"even capacity between 4 and 10");
constexpr int LP=CAP/2;                       // lane pairs per group
constexpr int GPW=8/LP;                       // whole groups per warp (aligned mode)
constexpr int GPP=16/LP;                      // whole groups per warp pair (MERGE_PACKED 2)
constexpr int GPT=MERGE_PACKED==2?(2*GPP):MERGE_PACKED?(32/LP):(4*GPW); // groups per 64-row tile
constexpr int ROWS_STORED=GPT*CAP;
constexpr int TOTAL_GROUPS=AXIS_OUTER*40;
constexpr int TILES=(TOTAL_GROUPS+GPT-1)/GPT;
constexpr bool STRADDLE=MERGE_PACKED && (8%LP)!=0;
constexpr int XBUF_WORDS=STRADDLE?GPT*2*4*(KC/4):0;
static_assert(ROWS_STORED<=64,"tile rows");
static_assert(MERGE_TILE_ORDER==0 || 40%GPT==0,"frame-minor tile order needs whole column blocks");
__device__ __forceinline__ int first_group(int tile) {
#if MERGE_TILE_ORDER
  return (tile%AXIS_OUTER)*40+(tile/AXIS_OUTER)*GPT;
#else
  return tile*GPT;
#endif
}
using namespace cute::SM90::GMMA;

#define E8(a,o) a[o],a[o+1],a[o+2],a[o+3],a[o+4],a[o+5],a[o+6],a[o+7]
#define E32(a) E8(a,0),E8(a,8),E8(a,16),E8(a,24)
#define E40(a) E32(a),E8(a,32)
#define E24(a) E8(a,0),E8(a,8),E8(a,16)
#if AXIS_COLS <= 48
#define ES(a) E24(a)
#define SCORE_OP MMA_64x48x16_F32BF16BF16_RS
#else
#define ES(a) E40(a)
#define SCORE_OP MMA_64x80x16_F32BF16BF16_RS
#endif
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
// Staging layout matches a TMA SWIZZLE_128B box {64 d, rows}: two d-halves of
// 64 rows x 128 B, 16-byte chunk index XOR (row & 7).
__device__ __forceinline__ int storeoff(int row,int d) {return (d>>6)*4096+row*64+((d&63)^((row&7)*8));}
__device__ __forceinline__ uint4 query_vector(const bf16* p,uint64_t policy) {
  uint4 a;
  asm volatile("ld.global.L1::evict_last.L2::cache_hint.v4.b32 {%0,%1,%2,%3},[%4],%5;":"=r"(a.x),"=r"(a.y),"=r"(a.z),"=r"(a.w):"l"(p),"l"(policy));
  return a;
}
__device__ __forceinline__ uint4 fragment_transpose(uint4 a,int lane) {
  uint32_t t0=__shfl_xor_sync(0xffffffff,(lane&1)?a.x:a.y,1);
  uint32_t t1=__shfl_xor_sync(0xffffffff,(lane&1)?a.z:a.w,1);
  uint4 b={(lane&1)?t0:a.x,(lane&1)?a.y:t0,(lane&1)?t1:a.z,(lane&1)?a.w:t1};
  t0=__shfl_xor_sync(0xffffffff,(lane&2)?b.x:b.z,2);
  t1=__shfl_xor_sync(0xffffffff,(lane&2)?b.y:b.w,2);
  return {(lane&2)?t0:b.x,(lane&2)?t1:b.y,(lane&2)?b.z:t0,(lane&2)?b.w:t1};
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
__device__ __forceinline__ float reduce80(float (&p)[KC/4]) {
  float sum=p[0];
  #pragma unroll
  for(int i=1;i<KC/4;++i)sum+=p[i];
  return sum4(sum);
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
__device__ __forceinline__ float dot80(float (&p)[KC/4],float (&x)[KC/4]){
  float sum=p[0]*x[0];
  #pragma unroll
  for(int i=1;i<KC/4;++i)sum=fmaf(p[i],x[i],sum);
  return sum4(sum);
}
template<int TOTAL_HEADS>
__global__ __launch_bounds__(128,MERGE_MINBLOCKS) void merge_kernel(
 const bf16* __restrict__ q,const bf16* __restrict__ k,const bf16* __restrict__ v,
 const uint8_t* __restrict__ seeds,float* __restrict__ bias,int HS,float scale,
 int tile_count,const __grid_constant__ CUtensorMap map_k,const __grid_constant__ CUtensorMap map_v,
 const __grid_constant__ CUtensorMap map_k_tail,const __grid_constant__ CUtensorMap map_v_tail,long long* __restrict__ prof) {
  constexpr int Q=AXIS_OUTER,KO=Q,R=AXIS_ROWS,C=AXIS_COLS;
  long long pacc[8]={0,0,0,0,0,0,0,0};long long pt=0;
  auto stamp=[&](int slot){ if(MERGE_PROF && threadIdx.x==0){long long now=clock64();pacc[slot]+=now-pt;pt=now;} };
  extern __shared__ __align__(1024) unsigned char raw[];
  bf16* sk=reinterpret_cast<bf16*>(raw);
  bf16* sv=sk+KC*128;
  bf16* temp=sv+KC*128;
  uint32_t* xbuf=reinterpret_cast<uint32_t*>(temp+MERGE_STAGE_BUFS*64*128);
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,sl=lane>>2;
  const int kb=blockIdx.x,head=blockIdx.y,gh=HS+head;
  const int p=warp*8+sl,grp=grp_of(p),idx=idx_of(p);
  const int owner0=2*idx,owner1=2*idx+1;
  const int pst=grp>=0?pstart_of(grp):p;
  uint64_t cache_last;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0,1.0;":"=l"(cache_last));
  for(int vec=tid;vec<KC*16;vec+=128) {
    int ki=vec/16,d=(vec%16)*8;
    int src=((gh*KO*R+kb)*C+ki)*128+d;
    *reinterpret_cast<uint4*>(sk+boff(ki,d))=ki<C?*reinterpret_cast<const uint4*>(k+src):make_uint4(0,0,0,0);
    *reinterpret_cast<uint4*>(sv+boff(ki,d))=ki<C?*reinterpret_cast<const uint4*>(v+src):make_uint4(0,0,0,0);
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
    int g=first_group(tile)+grp;
    bool ok=grp>=0 && g<TOTAL_GROUPS && tile<stop;
    int gg=ok?g:0,col=gg%40;
    int sbase=((gh*40+col)*(KO*R)+kb)*CAP;
    nsr0=ok?seeds[sbase+owner0]:0;nsr1=ok?seeds[sbase+owner1]:0;
  };
  auto gather=[&](int tile) {
    int g=first_group(tile)+grp;
    bool ok=grp>=0 && g<TOTAL_GROUPS;
    int gg=ok?g:0,qo=gg/40,col=gg%40;
    int sr0,sr1;
    if(MERGE_SEED_PREFETCH) {sr0=nsr0;sr1=nsr1;}
    else {
      int sbase=((gh*40+col)*(KO*R)+kb)*CAP;
      sr0=ok?seeds[sbase+owner0]:0;sr1=ok?seeds[sbase+owner1]:0;
    }
    int qi0=(((gh*Q+qo)*40+col)*90+sr0)*128+(lane&3)*8;
    int qi1=(((gh*Q+qo)*40+col)*90+sr1)*128+(lane&3)*8;
    #pragma unroll
    for(int d=0;d<128;d+=32) {
      uint4 a=ok?query_vector(q+qi0+d,cache_last):make_uint4(0,0,0,0),b=(ok && !(MERGE_ABL_GATHER56 && sl>=6))?query_vector(q+qi1+d,cache_last):make_uint4(0,0,0,0);
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
    const int g=first_group(tile)+grp;
    const bool ok=grp>=0 && g<TOTAL_GROUPS;
    if(!MERGE_PREFETCH_Q && tile>start) gather(tile);
    #pragma unroll
    for(int i=0;i<32;++i) asm volatile("" : "+r"(qreg[i]) :: "memory");
    float s[KC/2];
    asm volatile("wgmma.fence.sync.aligned;":::"memory");
    using Score=SCORE_OP<Major::K,Major::K>;
    #pragma unroll
    for(int d=0;d<128;d+=16) {
      int i=d/4;
      Score::fma(qreg[i],qreg[i+1],qreg[i+2],qreg[i+3],desc_at(score_base,boff(0,d)),ES(s),d==0?ScaleOut::Zero:ScaleOut::One);
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
      float sum=reduce80(direct);
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
      norm_max[half]=mx;norm_sum[half]=reduce80(bs);
      }
    }
    float norm_log=lg2((lane&1)?norm_sum[1]:norm_sum[0]);
    #pragma unroll
    for(int half=0;half<2;++half) {
      float norm=__fadd_rn(norm_max[half],__shfl_sync(0xffffffff,norm_log,(lane&~1)+half));
      #pragma unroll
      for(int i=0;i<KC/4;++i) {
        int ix=(i/2)*4+half*2+(i&1);
#if MERGE_FFMA_NORM
        s[ix]=__fmaf_rn(s[ix],scale,-norm);
#else
        s[ix]=__fadd_rn(__fmul_rn(s[ix],scale),-norm);
#endif
      }
    }
    // Ownership: per key column, the group's best representative. Normalized
    // scores are <= 0, so the unsigned minimum of bits(s) with the owner id in
    // the low four mantissa bits is the argmax (ties -> smallest owner).
    int labels[KC/4];
    uint32_t keys[KC/4];
    #pragma unroll
    for(int i=0;i<KC/4;++i) {
      int ix=(i/2)*4+(i&1);
      float a=s[ix],b=s[ix+2];
#if MERGE_FUSED_ARGMAX
      uint32_t key=ok?min(owner_key(a,owner0),owner_key(b,owner1)):0xffffffffu;
#else
      // Exact argmax: keep the value and the owner in separate reductions.
      uint32_t key=ok?min(owner_key(a,owner0),owner_key(b,owner1)):0xffffffffu;
#endif
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
    if constexpr(STRADDLE && !MERGE_ABL_NOSTRADDLE) {
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
      if(!MERGE_ABL_NOBAR) asm volatile("bar.sync %0,64;"::"r"(1+(warp>>1)):"memory");
#else
      if(!MERGE_ABL_NOBAR && warp>0 && (8*warp)%LP!=0) asm volatile("bar.sync %0,64;"::"r"(warp):"memory");
      if(!MERGE_ABL_NOBAR && warp<3 && (8*(warp+1))%LP!=0) asm volatile("bar.sync %0,64;"::"r"(warp+1):"memory");
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
      float sum=reduce80(probabilities),safe=sum>0?sum:1.f;
      float inv=reciprocal(safe),expected=dot80(probabilities,winners);
      float correction=__fmul_rn(__fadd_rn(__fmul_rn(expected,inv),-lg2(safe)),0.6931471805599453f);
      int out=((head*TOTAL_GROUPS+(ok?g:0))*(KO*R)+kb)*CAP+owner;
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
      bf16* stage=temp+(MERGE_STAGE_BUFS==2?kind*64*128:0);
      // The staging buffer is free once the bulk store that last read it (the
      // other kind's, issued by its own thread) has finished reading it.
      const int issuer=MERGE_SPLIT_ISSUE?(kind?64:0):0,previous=MERGE_SPLIT_ISSUE?(kind?0:64):0;
      if(tid==previous) {
        if(MERGE_STAGE_BUFS==2) asm volatile("cp.async.bulk.wait_group.read 1;":::"memory");
        else asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");
      }
      __syncthreads();
#if MERGE_STMATRIX
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
#else
      #pragma unroll
      for(int i=0;i<64;i+=2) {
        int row=srow_of(p,(i&3)>>1);
        int d=2*(lane&3)+8*(i>>2);
        *reinterpret_cast<__nv_bfloat162*>(stage+storeoff(row,d))=__floats2bfloat162_rn(acc[i],acc[i+1]);
      }
#endif
      asm volatile("fence.proxy.async.shared::cta;":::"memory");
      __syncthreads();
      if(tid==issuer) {
        const int rowc=MERGE_ABL_STORE_SAME?0:first_group(tile)*CAP,kbc=MERGE_ABL_STORE_SAME==1?head*(KO*R):head*(KO*R)+kb;
#if MERGE_STORE_2D
        const bool tail=first_group(tile)+GPT>TOTAL_GROUPS;
        const CUtensorMap* map=kind==0?(tail?&map_k_tail:&map_k):(tail?&map_v_tail:&map_v);
        const int row2=kbc*(TOTAL_GROUPS*CAP)+rowc;
        #pragma unroll
        for(int half=0;half<2;++half) {
          uint32_t sp=__cvta_generic_to_shared(stage+half*4096);
          asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%1,%2}],[%3];"::"l"(map),"r"(half*64),"r"(row2),"r"(sp):"memory");
        }
#else
        const CUtensorMap* map=kind==0?&map_k:&map_v;
        #pragma unroll
        for(int half=0;half<2;++half) {
          uint32_t sp=__cvta_generic_to_shared(stage+half*4096);
          asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0,{%1,%2,%3}],[%4];"::"l"(map),"r"(half*64),"r"(rowc),"r"(kbc),"r"(sp):"memory");
        }
#endif
        asm volatile("cp.async.bulk.commit_group;":::"memory");
      }
      stamp(4+2*kind);
    }
  }
  if(tid==0 || (MERGE_SPLIT_ISSUE && tid==64)) asm volatile("cp.async.bulk.wait_group 0;":::"memory");
  if(MERGE_PROF && threadIdx.x==0) {
    #pragma unroll
    for(int i=0;i<8;++i) atomicAdd(reinterpret_cast<unsigned long long*>(prof+i),(unsigned long long)pacc[i]);
  }
}
__global__ void pack_query_kernel(const bf16* input,bf16* output,int64_t vectors) {
  int lane=threadIdx.x&31;
  for(int64_t vec=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
      vec<((vectors+31)/32)*32;vec+=int64_t(gridDim.x)*blockDim.x) {
    uint4 a=vec<vectors?*reinterpret_cast<const uint4*>(input+vec*8):make_uint4(0,0,0,0);
    uint4 b=fragment_transpose(a,lane);
    if(vec<vectors)*reinterpret_cast<uint4*>(output+vec*8)=b;
  }
}

void pack(torch::Tensor input,torch::Tensor output) {
  TORCH_CHECK(input.is_cuda() && output.is_cuda() && input.device()==output.device(),"CUDA tensors on the same device required");
  TORCH_CHECK(input.scalar_type()==at::kBFloat16 && output.scalar_type()==at::kBFloat16 && input.sizes()==output.sizes(),"Identical BF16 shape required");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous() && input.numel()%128==0,"Contiguous native Q shape required");
  TORCH_CHECK(input.data_ptr()!=output.data_ptr(),"Packing must be out of place");
  int64_t vectors=input.numel()/8;
  int blocks=int(std::min<int64_t>((vectors+255)/256,4096));
  pack_query_kernel<<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>(
      (const bf16*)input.data_ptr(),(bf16*)output.data_ptr(),vectors);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

constexpr int SMEM_BYTES=(KC*128*2+MERGE_STAGE_BUFS*64*128)*2+XBUF_WORDS*4;

void run_impl(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor seeds,
 torch::Tensor mk,torch::Tensor mv,torch::Tensor bias,int hs,int ah,int tiles,torch::Tensor prof) {
 constexpr int KB=AXIS_OUTER*45;
 TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.is_contiguous(),"Contiguous CUDA BF16 Q required");
 TORCH_CHECK(q.numel()==int64_t(AXIS_HEADS)*AXIS_OUTER*40*90*128,"Compiled shape mismatch");
 TORCH_CHECK(ah==AXIS_ACTIVE_HEADS && hs>=0 && hs+ah<=AXIS_HEADS && tiles>0,"Head/tile bounds");
 TORCH_CHECK(seeds.scalar_type()==at::kByte && seeds.numel()==int64_t(AXIS_HEADS)*40*KB*CAP,"Exact seed shape");
 TORCH_CHECK(mk.is_contiguous() && mv.is_contiguous() && mk.numel()==int64_t(ah)*KB*TOTAL_GROUPS*CAP*128 && mv.numel()==mk.numel(),"Workspace shape");
 TORCH_CHECK(bias.is_contiguous() && bias.numel()==mk.numel()/128,"Bias shape");
 CUtensorMap map_k{},map_v{},map_k_tail{},map_v_tail{};
 uint64_t rows_per_kb=uint64_t(TOTAL_GROUPS)*CAP;
 constexpr int TAIL_GROUPS=TOTAL_GROUPS-(TILES-1)*GPT;
 uint64_t dims[3]={128,rows_per_kb,uint64_t(ah)*KB},strides[2]={256,rows_per_kb*256};
 uint32_t box[3]={64,ROWS_STORED,1},element_strides[3]={1,1,1};
 auto encode=[&](CUtensorMap& map,torch::Tensor x) {
  auto status=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,x.data_ptr(),dims,strides,box,element_strides,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(status==CUDA_SUCCESS,"Merge TMA encode failed: ",int(status));
 };
 uint64_t dims2[2]={128,rows_per_kb*uint64_t(ah)*KB},strides2[1]={256};
 auto encode2=[&](CUtensorMap& map,torch::Tensor x,uint32_t rows) {
  uint32_t box2[2]={64,rows},es2[2]={1,1};
  auto status=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,x.data_ptr(),dims2,strides2,box2,es2,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(status==CUDA_SUCCESS,"Merge TMA encode failed: ",int(status));
 };
 if(MERGE_STORE_2D) {encode2(map_k,mk,ROWS_STORED);encode2(map_v,mv,ROWS_STORED);encode2(map_k_tail,mk,TAIL_GROUPS*CAP);encode2(map_v_tail,mv,TAIL_GROUPS*CAP);}
 else {encode(map_k,mk);encode(map_v,mv);map_k_tail=map_k;map_v_tail=map_v;}
 C10_CUDA_CHECK(cudaFuncSetAttribute(merge_kernel<AXIS_HEADS>,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM_BYTES));
 dim3 grid(KB,ah,(TILES+tiles-1)/tiles);
 merge_kernel<AXIS_HEADS><<<grid,128,SMEM_BYTES,at::cuda::getCurrentCUDAStream()>>>(
   (const bf16*)q.data_ptr(),(const bf16*)k.data_ptr(),(const bf16*)v.data_ptr(),seeds.data_ptr<uint8_t>(),
   bias.data_ptr<float>(),hs,float(0.08838834764831845*1.4426950408889634),tiles,map_k,map_v,map_k_tail,map_v_tail,
   prof.defined()?reinterpret_cast<long long*>(prof.data_ptr<int64_t>()):nullptr);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void run(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor seeds,
 torch::Tensor mk,torch::Tensor mv,torch::Tensor bias,int hs,int ah,int tiles) {
 run_impl(q,k,v,seeds,mk,mv,bias,hs,ah,tiles,torch::Tensor());
}
void run_prof(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor seeds,
 torch::Tensor mk,torch::Tensor mv,torch::Tensor bias,int hs,int ah,int tiles,torch::Tensor prof) {
 TORCH_CHECK(prof.is_cuda() && prof.scalar_type()==at::kLong && prof.numel()>=8,"prof tensor");
 run_impl(q,k,v,seeds,mk,mv,bias,hs,ah,tiles,prof);
}
std::vector<int> resources() {
 cudaFuncAttributes a{};C10_CUDA_CHECK(cudaFuncGetAttributes(&a,merge_kernel<AXIS_HEADS>));
 int active=0;
 C10_CUDA_CHECK(cudaFuncSetAttribute(merge_kernel<AXIS_HEADS>,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM_BYTES));
 C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,merge_kernel<AXIS_HEADS>,128,SMEM_BYTES));
 return {a.numRegs,int(a.localSizeBytes),SMEM_BYTES,active,TILES,GPT};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("run",&run);m.def("run_prof",&run_prof);m.def("pack",&pack);m.def("resources",&resources);}
