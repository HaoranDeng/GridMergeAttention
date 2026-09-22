// Q90/K80 exact C14. Hopper WGMMA; 14 representatives, two columns.
// 2026-09-21 optimization: mk/mv leave through TMA tensor stores from a
// 128B-swizzled staging buffer (the store drains asynchronously while the CTA
// computes the next tile), the next tile's query fragments are prefetched into
// the dead qreg registers right after the score WGMMA consumes them, and the
// launch bound asks for 3 CTAs/SM (164 registers, no spills) instead of the
// 128-register 4-CTA configuration, which spills and measures slower here.
#ifndef MERGE_STAGE_BUFS
#define MERGE_STAGE_BUFS 1
#endif
#ifndef MERGE_MINBLOCKS
#define MERGE_MINBLOCKS 3
#endif
#ifndef MERGE_PREFETCH_Q
#define MERGE_PREFETCH_Q 1
#endif
// Fused 32-bit ownership argmax (ported from the C8 D10A kernel): the key
// of a normalized score is bits(-s) with its four low mantissa bits replaced
// by the owner id, so one unsigned MIN butterfly decides value and owner at
// once. Exact ties still resolve to the smallest owner; only candidates within
// 16 ulp of each other (about 2^-19 relative) can be reassigned.
// Opt-in (Q90K80C14(merge_fused_argmax=True)): merge -4.6 ms, but on random
// QKV about 1e-3 of key ownerships move, so it is off by default.
#ifndef MERGE_FUSED_ARGMAX
#define MERGE_FUSED_ARGMAX 0
#endif
// stmatrix staging: 8 stmatrix.x4 per kind instead of 32 STS.32 plus their
// swizzled address arithmetic; same bytes land in the same staging layout.
#ifndef MERGE_STMATRIX
#define MERGE_STMATRIX 1
#endif
// Normalize with one FFMA (s*scale-norm) instead of FMUL+FADD: one rounding
// instead of two (numerics tier, not bitwise).
#ifndef MERGE_FFMA_NORM
#define MERGE_FFMA_NORM 0
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
// Every descriptor this kernel builds sits at a compile-time element offset from
// the same resident tile base, and for each offset used here the boff swizzle
// term is zero, so the offset only moves the descriptor's 14-bit address field.
// The K/V tiles start at the base of a 57344-byte dynamic window, so the field
// cannot carry. One add therefore reproduces desc() bit for bit.
__device__ __forceinline__ uint64_t desc_at(uint64_t base,int element_offset) {
  return base+(uint64_t(element_offset)>>3);
}
__device__ __forceinline__ int aoff(int row,int k) {
  int offset=row*64+(k&63)+(k>>6)*4096;
  return offset^(((offset>>6)&7)<<3);
}
__device__ __forceinline__ int boff(int k,int n) {
  int offset=k*64+(n&63)+(n>>6)*(KC*64);
  return offset^(((offset>>6)&7)<<3);
}
__device__ __forceinline__ int col_for_row(int row) {return row/16;}
__device__ __forceinline__ int seed_for_row(int row) {return row%16;}
// Staging layout matches a TMA SWIZZLE_128B box {64 d, rows}: two d-halves of
// 64 rows x 128 B, 16-byte chunk index XOR (row & 7). Conflict-free for the
// accumulator quad stores and 1024-byte aligned per 16-row column piece.
__device__ __forceinline__ int storeoff(int row,int d) {return (d>>6)*4096+row*64+((d&63)^((row&7)*8));}
__device__ __forceinline__ uint32_t load_query(const bf16* p,uint64_t policy) {
  uint32_t value;
  asm volatile("ld.global.L1::evict_last.L2::cache_hint.b32 %0,[%1],%2;":"=r"(value):"l"(p),"l"(policy));
  return value;
}
__device__ __forceinline__ uint4 query_vector(const bf16* p,uint64_t policy,int lane) {
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
__device__ __forceinline__ void store_output(bf16* p,uint4 value,uint64_t policy) {
  asm volatile("st.global.L1::evict_first.L2::cache_hint.v4.b32 [%0],{%1,%2,%3,%4},%5;"::"l"(p),"r"(value.x),"r"(value.y),"r"(value.z),"r"(value.w),"l"(policy):"memory");
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
template<int TILES,int TOTAL_HEADS>
__global__ __launch_bounds__(128,MERGE_MINBLOCKS) void merge_kernel(
 const bf16* __restrict__ q,const bf16* __restrict__ k,const bf16* __restrict__ v,
 const uint8_t* __restrict__ seeds,bf16* __restrict__ mk,bf16* __restrict__ mv,
 float* __restrict__ bias,int Q_arg,int KO_arg,int R_arg,int C_arg,int H_arg,int AH_arg,int HS,int RF_arg,float scale,
 int tile_count,const __grid_constant__ CUtensorMap map_k,const __grid_constant__ CUtensorMap map_v) {
  constexpr float* debug_scores=nullptr;
  constexpr int stop_after=0;
  constexpr int Q=AXIS_OUTER,KO=Q,R=AXIS_ROWS,C=AXIS_COLS,H=TOTAL_HEADS,AH=AXIS_ACTIVE_HEADS,RF=Q;
  extern __shared__ __align__(128) unsigned char raw[];
  bf16* sk=reinterpret_cast<bf16*>(raw);
  bf16* sv=sk+KC*128;
  bf16* temp=sv+KC*128;
  int tid=threadIdx.x,lane=tid&31,warp=tid>>5,seed=lane>>2;
  int block=blockIdx.x,head=blockIdx.y,gh=HS+head;
  int kb=block,ko=kb/R,kr=kb%R;
  uint64_t cache_last,cache_first=0;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0,1.0;":"=l"(cache_last));
  for(int vec=tid;vec<KC*16;vec+=128) {
    int ki=vec/16,d=(vec%16)*8;
    int src=((gh*KO*R+kb)*C+ki)*128+d;
    *reinterpret_cast<uint4*>(sk+boff(ki,d))=ki<C?*reinterpret_cast<const uint4*>(k+src):make_uint4(0,0,0,0);
    *reinterpret_cast<uint4*>(sv+boff(ki,d))=ki<C?*reinterpret_cast<const uint4*>(v+src):make_uint4(0,0,0,0);
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;":::"memory");
  // ptxas rematerialises desc() inside the tile loop rather than keeping eighteen
  // descriptors live, costing 131 uniform-register instructions per tile. Keeping
  // only three bases and adding a constant leaves the same bits at a fraction of
  // the issue cost.
  const uint64_t score_base=desc(sk,16,1024);
  const uint64_t merge_base_k=desc(sk,KC*128,1024),merge_base_v=desc(sv,KC*128,1024);
  int total=Q*10,start=blockIdx.z*tile_count,stop=min(start+tile_count,total);
  const int row0=warp*16+seed;
  uint32_t qreg[32];
  auto gather=[&](int tile) {
    int qo=tile%Q,cb=tile/Q;
    int col0=cb*4+col_for_row(row0);
    int si0=((gh*40+col0)*(KO*R)+kb)*14+seed_for_row(row0);
    int col1=cb*4+col_for_row(row0+8);
    int si1=((gh*40+col1)*(KO*R)+kb)*14+seed_for_row(row0+8);
    int sr0=seeds[si0],sr1=seed_for_row(row0+8)<14?seeds[si1]:0;
    int qi0=(((gh*Q+qo)*40+col0)*90+sr0)*128+(lane&3)*8;
    int qi1=(((gh*Q+qo)*40+col1)*90+sr1)*128+(lane&3)*8;
    #pragma unroll
    for(int d=0;d<128;d+=32) {
      uint4 a=seed_for_row(row0)<14?query_vector(q+qi0+d,cache_last,lane):make_uint4(0,0,0,0),b=seed_for_row(row0+8)<14?query_vector(q+qi1+d,cache_last,lane):make_uint4(0,0,0,0);
      int i=d/4;
      qreg[i]=a.x;qreg[i+1]=b.x;qreg[i+2]=a.y;qreg[i+3]=b.y;
      qreg[i+4]=a.z;qreg[i+5]=b.z;qreg[i+6]=a.w;qreg[i+7]=b.w;
    }
  };
  if(start<stop) gather(start);
  for(int tile=start;tile<stop;++tile) {
    int qo=tile%Q,cb=tile/Q,rf=0;
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
    if(MERGE_PREFETCH_Q && tile+1<stop) gather(tile+1);
    #pragma unroll
    for(int i=0;i<KC/2;++i) {
      int kc=2*(lane&3)+8*(i>>2)+(i&1);
      if(kc>=C)s[i]=-CUDART_INF_F;
    }
    if(debug_scores) {
      #pragma unroll
      for(int i=0;i<KC/2;++i) {
        int row=warp*16+seed+8*((i&3)>>1);
        int col=2*(lane&3)+8*(i>>2)+(i&1);
        debug_scores[(col_for_row(row)*8+seed_for_row(row))*80+col]=s[i];
      }
      return;
    }
    if(stop_after==1) continue;
    float norm_max[2],norm_sum[2];
    #pragma unroll
    for(int half=0;half<2;++half) {
      float direct[KC/4];
      #pragma unroll
      for(int i=0;i<KC/4;++i)
        direct[i]=ex2(__fmul_rn(s[4*(i/2)+2*half+(i&1)],scale));
      float sum=reduce80(direct);
      // A full-warp vote keeps the fallback's full-mask shuffles converged.
      // If any row needs the stable path, every lane in this warp takes it.
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
    // Each four-lane row group has the same reduced sums. Compute its two
    // independent logarithms in alternating lanes, then broadcast each.
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
    // Each warp is one paired-column group. Mask internal C16 padding.
    int labels[KC/4];
    const unsigned owner0=seed,owner1=seed+8;
    #pragma unroll
    for(int i=0;i<KC/4;++i) {
      int ix=(i/2)*4+(i&1);
      float a=s[ix],b=owner1<14?s[ix+2]:-CUDART_INF_F;
#if MERGE_FUSED_ARGMAX
      // Normalized scores are <= 0, so -s >= +0 and the unsigned order of its
      // bit pattern is the float order; -(-inf) = +inf never wins.
      uint32_t key=min(owner_key(-a,owner0),owner_key(-b,owner1));
      key=min(key,__shfl_xor_sync(0xffffffff,key,4));
      key=min(key,__shfl_xor_sync(0xffffffff,key,8));
      key=min(key,__shfl_xor_sync(0xffffffff,key,16));
      labels[i]=(int)(key&15u);
      continue;
#endif
      float best=fmaxf(a,b);
      best=fmaxf(best,__shfl_xor_sync(0xffffffff,best,4));
      best=fmaxf(best,__shfl_xor_sync(0xffffffff,best,8));
      best=fmaxf(best,__shfl_xor_sync(0xffffffff,best,16));
      unsigned label=min(a==best?owner0:16u,b==best?owner1:16u);
      label=min(label,__shfl_xor_sync(0xffffffff,label,4));
      label=min(label,__shfl_xor_sync(0xffffffff,label,8));
      label=min(label,__shfl_xor_sync(0xffffffff,label,16));
      labels[i]=label;
    }
    uint32_t packed_weights[KC/4];
    #pragma unroll
    for(int half=0;half<2;++half) {
      float winners[KC/4],probabilities[KC/4];
      int row=warp*16+seed+8*half,col=cb*4+col_for_row(row);
      int owner=seed_for_row(row);
      #pragma unroll
      for(int i=0;i<KC/4;++i) {
        int ix=(i/2)*4+(i&1)+2*half;
        bool valid=2*(lane&3)+8*(i/2)+(i&1)<C;
        winners[i]=valid?s[ix]:0.f;
        probabilities[i]=(valid && labels[i]==owner)?ex2(s[ix]):0.f;
      }
      float sum=reduce80(probabilities),safe=sum>0?sum:1.f;
      float inv=reciprocal(safe),expected=dot80(probabilities,winners);
      float correction=__fmul_rn(__fadd_rn(__fmul_rn(expected,inv),-lg2(safe)),0.6931471805599453f);
      int out=(((head*Q+qo)*40+col)*(KO*R)+kb)*14+owner;
      if((lane&3)==0 && owner<14)bias[out]=sum>0?-correction:-CUDART_INF_F;
      #pragma unroll
      for(int i=0;i<KC/8;++i) {
        union {bf16 values[2];uint32_t bits;} pair;
        pair.values[0]=__float2bfloat16_rn(__fmul_rn(probabilities[2*i],inv));
        pair.values[1]=__float2bfloat16_rn(__fmul_rn(probabilities[2*i+1],inv));
        packed_weights[2*i+half]=pair.bits;
      }
    }
    if(stop_after==2) continue;
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
      if(stop_after==3) continue;
      bf16* stage=temp+(MERGE_STAGE_BUFS==2?kind*64*128:0);
      // The staging buffer is free once the bulk store that last read it has
      // finished reading shared memory.
      if(tid==0) {
        if(MERGE_STAGE_BUFS==2) asm volatile("cp.async.bulk.wait_group.read 1;":::"memory");
        else asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");
      }
      __syncthreads();
#if MERGE_STMATRIX
      {
        // Instruction g stores the four 8x8 blocks (rows warp*16+8*(g>>2)..+7,
        // d chunks 4*(g&3)+m, m=0..3). Lane l supplies the address of row
        // (l&7) of block (l>>3): storeoff(row, 8*(4*(g&3)+(l>>3))).
        const int r=lane&7,mb=lane>>3;
        const uint32_t base=__cvta_generic_to_shared(stage)+2*((warp*16+r)*64);
        #pragma unroll
        for(int g=0;g<8;++g) {
          int chunk=4*(g&3)+mb;
          uint32_t addr=base+2*((chunk>>3)*4096+8*(g>>2)*64+(((chunk&7)^r)*8));
          int i0=4*(4*(g&3)+0)+2*(g>>2),i1=4*(4*(g&3)+1)+2*(g>>2),i2=4*(4*(g&3)+2)+2*(g>>2),i3=4*(4*(g&3)+3)+2*(g>>2);
          stsm_x4(addr,pack_bf162(acc[i0],acc[i0+1]),pack_bf162(acc[i1],acc[i1+1]),pack_bf162(acc[i2],acc[i2+1]),pack_bf162(acc[i3],acc[i3+1]));
        }
      }
#else
      #pragma unroll
      for(int i=0;i<64;i+=2) {
        int row=warp*16+seed+8*((i&3)>>1);
        int d=2*(lane&3)+8*(i>>2);
        *reinterpret_cast<__nv_bfloat162*>(stage+storeoff(row,d))=__floats2bfloat162_rn(acc[i],acc[i+1]);
      }
#endif
      asm volatile("fence.proxy.async.shared::cta;":::"memory");
      __syncthreads();
      if(tid==0) {
        const CUtensorMap* map=kind==0?&map_k:&map_v;
        #pragma unroll
        for(int col4=0;col4<4;++col4) {
          int col=cb*4+col4;
          int row=int(((int64_t(head)*Q+qo)*40+col)*(KO*R)+kb)*14;
          #pragma unroll
          for(int half=0;half<2;++half) {
            uint32_t sp=__cvta_generic_to_shared(stage+half*4096+col4*16*64);
            asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%1,%2}],[%3];"::"l"(map),"r"(half*64),"r"(row),"r"(sp):"memory");
          }
        }
        asm volatile("cp.async.bulk.commit_group;":::"memory");
      }
    }
  }
  // Wait for the bulk stores to complete (not just their shared-memory reads)
  // so the merged tiles are visible to the route kernel that follows on the
  // stream; waiting only on .read left a small fraction of tiles stale.
  if(tid==0) asm volatile("cp.async.bulk.wait_group 0;":::"memory");
}
// Pack each 32-BF16 dimension slice into the exact four-lane WGMMA
// register-fragment order. This is a lossless permutation of uint32 words.
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

void run(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor seeds,
 torch::Tensor mk,torch::Tensor mv,torch::Tensor bias,int hs,int ah,int tiles) {
 TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.is_contiguous(),"Contiguous CUDA BF16 Q required");
 TORCH_CHECK(q.numel()==int64_t(AXIS_HEADS)*AXIS_OUTER*40*90*128,"Compiled shape mismatch");
 TORCH_CHECK(ah==AXIS_ACTIVE_HEADS && hs>=0 && hs+ah<=AXIS_HEADS && tiles>0,"Head/tile bounds");
 TORCH_CHECK(seeds.scalar_type()==at::kByte && seeds.numel()==int64_t(AXIS_HEADS)*40*AXIS_OUTER*45*14,"Exact C14 seed shape");
 constexpr int bytes=(KC*128*2+MERGE_STAGE_BUFS*64*128)*2;
 TORCH_CHECK(mk.is_contiguous() && mv.is_contiguous() && mk.numel()==int64_t(ah)*AXIS_OUTER*40*AXIS_OUTER*45*14*128 && mv.numel()==mk.numel(),"Workspace shape");
 CUtensorMap map_k{},map_v{};
 uint64_t rows=uint64_t(ah)*AXIS_OUTER*40*AXIS_OUTER*45*14;
 uint64_t dims[2]={128,rows},strides[1]={256};
 uint32_t box[2]={64,14},element_strides[2]={1,1};
 auto encode=[&](CUtensorMap& map,torch::Tensor x) {
  auto status=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,x.data_ptr(),dims,strides,box,element_strides,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(status==CUDA_SUCCESS,"Merge TMA encode failed: ",int(status));
 };
 encode(map_k,mk);encode(map_v,mv);
 C10_CUDA_CHECK(cudaFuncSetAttribute(merge_kernel<21,AXIS_HEADS>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
 dim3 grid(AXIS_OUTER*45,ah,(AXIS_OUTER*10+tiles-1)/tiles);
 merge_kernel<21,AXIS_HEADS><<<grid,128,bytes,at::cuda::getCurrentCUDAStream()>>>(
   (const bf16*)q.data_ptr(),(const bf16*)k.data_ptr(),(const bf16*)v.data_ptr(),seeds.data_ptr<uint8_t>(),
   (bf16*)mk.data_ptr(),(bf16*)mv.data_ptr(),bias.data_ptr<float>(),0,0,0,0,0,0,hs,0,
   float(0.08838834764831845*1.4426950408889634),tiles,map_k,map_v);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<int> resources() {
 cudaFuncAttributes a{};C10_CUDA_CHECK(cudaFuncGetAttributes(&a,merge_kernel<21,AXIS_HEADS>));
 int active=0;constexpr int bytes=(KC*128*2+MERGE_STAGE_BUFS*64*128)*2;
 C10_CUDA_CHECK(cudaFuncSetAttribute(merge_kernel<21,AXIS_HEADS>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
 C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,merge_kernel<21,AXIS_HEADS>,128,bytes));
 return {a.numRegs,int(a.localSizeBytes),bytes,active};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("run",&run);m.def("pack",&pack);m.def("resources",&resources);}
