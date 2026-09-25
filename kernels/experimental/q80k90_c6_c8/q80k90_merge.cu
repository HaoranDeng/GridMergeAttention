// Q80/K90 exact C6/C8. Hopper WGMMA, shared source for both capacities.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cuda.h>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/arch/mma_sm90_gmma_ext.hpp>
#include <cstdint>
#include <mutex>
#include <unordered_set>

using bf16 = __nv_bfloat16;
#ifndef AXIS_CAPACITY
#error AXIS_CAPACITY must be 6 or 8
#endif
#ifndef AXIS_MIN_BLOCKS
#define AXIS_MIN_BLOCKS 2
#endif
#ifndef AXIS_DIRECT_STORE
#define AXIS_DIRECT_STORE 0
#endif
#ifndef AXIS_OWNER_MODE
#define AXIS_OWNER_MODE 0
#endif
#ifndef AXIS_FUSED_KV
#define AXIS_FUSED_KV 0
#endif
#ifndef AXIS_WARP_GROUPS
#define AXIS_WARP_GROUPS 1
#endif
#ifndef AXIS_CARVEOUT
#define AXIS_CARVEOUT -1
#endif
#ifndef AXIS_VECTOR_STORE
#define AXIS_VECTOR_STORE 0
#endif
#ifndef AXIS_SHARED_Q
#define AXIS_SHARED_Q 0
#endif
#ifndef AXIS_HOIST_SEEDS
#define AXIS_HOIST_SEEDS 0
#endif
#ifndef AXIS_CONFIGURE_ONCE
#define AXIS_CONFIGURE_ONCE 0
#endif
#ifndef AXIS_REUSE_EXP
#define AXIS_REUSE_EXP 0
#endif
#ifndef AXIS_SUM_GROUPS
#define AXIS_SUM_GROUPS 1
#endif
constexpr int CAPACITY=AXIS_CAPACITY;
static_assert(CAPACITY==6 || CAPACITY==8);
static_assert(AXIS_SEED_TILE==8 || AXIS_SEED_TILE==16);
constexpr int BLOCK_THREADS=128*AXIS_WARP_GROUPS;
constexpr int ROWS_PER_TILE=64*AXIS_WARP_GROUPS;
constexpr int GROUPS_PER_TILE=ROWS_PER_TILE/AXIS_SEED_TILE;
constexpr int COLUMN_TILES=(45+GROUPS_PER_TILE-1)/GROUPS_PER_TILE;
constexpr int KC=(AXIS_COLS+15)/16*16;
using namespace cute::SM90::GMMA;

#define E8(a,o) a[o],a[o+1],a[o+2],a[o+3],a[o+4],a[o+5],a[o+6],a[o+7]
#define E32(a) E8(a,0),E8(a,8),E8(a,16),E8(a,24)
#define E40(a) E32(a),E8(a,32)
#define E48(a) E40(a),E8(a,40)
#define E24(a) E8(a,0),E8(a,8),E8(a,16)
#if AXIS_COLS <= 48
#define ES(a) E24(a)
#define SCORE_OP MMA_64x48x16_F32BF16BF16_RS
#define SCORE_SS_OP MMA_64x48x16_F32BF16BF16_SS
#elif AXIS_COLS <= 80
#define ES(a) E40(a)
#define SCORE_OP MMA_64x80x16_F32BF16BF16_RS
#define SCORE_SS_OP MMA_64x80x16_F32BF16BF16_SS
#else
#define ES(a) E48(a)
#define SCORE_OP MMA_64x96x16_F32BF16BF16_RS
#define SCORE_SS_OP MMA_64x96x16_F32BF16BF16_SS
#endif
#define E64(a) E32(a),E8(a,32),E8(a,40),E8(a,48),E8(a,56)
#define E128(a) E64(a),E8(a,64),E8(a,72),E8(a,80),E8(a,88),E8(a,96),E8(a,104),E8(a,112),E8(a,120)

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
__device__ __forceinline__ int col_for_row(int row) {return row/AXIS_SEED_TILE;}
__device__ __forceinline__ int seed_for_row(int row) {return row%AXIS_SEED_TILE;}
__device__ __forceinline__ int storeoff(int row,int d) {return row*128+(d^((row&7)*8));}
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
__device__ __forceinline__ uint32_t pack_pair(float a,float b) {
  union {__nv_bfloat162 pair;uint32_t bits;} value;
  value.pair=__floats2bfloat162_rn(a,b);return value.bits;
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
  #if AXIS_SUM_GROUPS > 1
  float sums[AXIS_SUM_GROUPS]={};
  #pragma unroll
  for(int i=0;i<KC/4;++i)sums[i%AXIS_SUM_GROUPS]+=p[i];
  float sum=sums[0];
  #pragma unroll
  for(int i=1;i<AXIS_SUM_GROUPS;++i)sum+=sums[i];
  #else
  float sum=p[0];
  #pragma unroll
  for(int i=1;i<KC/4;++i)sum+=p[i];
  #endif
  return sum4(sum);
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
__global__ __launch_bounds__(BLOCK_THREADS,AXIS_MIN_BLOCKS) void merge_kernel(
 const bf16* __restrict__ q,const bf16* __restrict__ k,const bf16* __restrict__ v,
 const uint8_t* __restrict__ seeds,bf16* __restrict__ mk,bf16* __restrict__ mv,
 float* __restrict__ bias,int Q_arg,int KO_arg,int R_arg,int C_arg,int H_arg,int AH_arg,int HS,int RF_arg,float scale,
 int tile_count) {
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
  for(int vec=tid;vec<KC*16;vec+=BLOCK_THREADS) {
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
  #if AXIS_SHARED_Q
  const uint64_t query_base=desc(temp+(warp/4)*64*128,16,1024);
  #endif
  // Query-column tile count is independent of exact representative capacity.
  int total=Q*COLUMN_TILES,start=blockIdx.z*tile_count;
  #if AXIS_HOIST_SEEDS
  const int fixed_cb=blockIdx.z;
  const int row0=warp*16+seed;
  const int col0=fixed_cb*GROUPS_PER_TILE+col_for_row(row0);
  const int col1=fixed_cb*GROUPS_PER_TILE+col_for_row(row0+8);
  const int si0=((gh*45+col0)*(KO*R)+kb)*CAPACITY+seed_for_row(row0);
  const int si1=((gh*45+col1)*(KO*R)+kb)*CAPACITY+seed_for_row(row0+8);
  const int sr0=col0<45 && seed_for_row(row0)<CAPACITY?seeds[si0]:0;
  const int sr1=col1<45 && seed_for_row(row0+8)<CAPACITY?seeds[si1]:0;
  const int query_base0=((gh*Q*45+col0)*80+sr0)*128+(lane&3)*8;
  const int query_base1=((gh*Q*45+col1)*80+sr1)*128+(lane&3)*8;
  #endif
  for(int tile=start;tile<min(start+tile_count,total);++tile) {
    #if AXIS_HOIST_SEEDS
    int qo=tile-start,cb=fixed_cb,rf=0;
    #else
    int qo=tile%Q,cb=tile/Q,rf=0;
    #endif
    #if AXIS_SHARED_Q
    #if AXIS_DIRECT_STORE
    __syncthreads();
    #endif
    for(int vec=tid;vec<ROWS_PER_TILE*16;vec+=BLOCK_THREADS) {
      int row=vec/16,d=(vec%16)*8;
      int col=cb*GROUPS_PER_TILE+col_for_row(row),c=seed_for_row(row);
      bool valid=col<45 && c<CAPACITY;
      int sr=valid?seeds[((gh*45+col)*(KO*R)+kb)*CAPACITY+c]:0;
      int64_t qi=(((int64_t(gh)*Q+qo)*45+col)*80+sr)*128+d;
      int shared=(row/64)*64*128+aoff(row%64,d);
      *reinterpret_cast<uint4*>(temp+shared)=valid?*reinterpret_cast<const uint4*>(q+qi):make_uint4(0,0,0,0);
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    #else
    #if AXIS_HOIST_SEEDS
    const int qi0=query_base0+qo*45*80*128;
    const int qi1=query_base1+qo*45*80*128;
    #else
    int row0=warp*16+seed,col0=cb*GROUPS_PER_TILE+col_for_row(row0);
    int si0=((gh*45+col0)*(KO*R)+kb)*CAPACITY+seed_for_row(row0);
    int col1=cb*GROUPS_PER_TILE+col_for_row(row0+8);
    int si1=((gh*45+col1)*(KO*R)+kb)*CAPACITY+seed_for_row(row0+8);
    int sr0=col0<45 && seed_for_row(row0)<CAPACITY?seeds[si0]:0,sr1=col1<45 && seed_for_row(row0+8)<CAPACITY?seeds[si1]:0;
    int qi0=(((gh*Q+qo)*45+col0)*80+sr0)*128+(lane&3)*8;
    int qi1=(((gh*Q+qo)*45+col1)*80+sr1)*128+(lane&3)*8;
    #endif
    uint32_t qreg[32];
    #pragma unroll
    for(int d=0;d<128;d+=32) {
      uint4 a=col0<45 && seed_for_row(row0)<CAPACITY?query_vector(q+qi0+d,cache_last,lane):make_uint4(0,0,0,0),b=col1<45 && seed_for_row(row0+8)<CAPACITY?query_vector(q+qi1+d,cache_last,lane):make_uint4(0,0,0,0);
      int i=d/4;
      qreg[i]=a.x;qreg[i+1]=b.x;qreg[i+2]=a.y;qreg[i+3]=b.y;
      qreg[i+4]=a.z;qreg[i+5]=b.z;qreg[i+6]=a.w;qreg[i+7]=b.w;
    }
    #pragma unroll
    for(int i=0;i<32;++i) asm volatile("" : "+r"(qreg[i]) :: "memory");
    #endif
    float s[KC/2];
    asm volatile("wgmma.fence.sync.aligned;":::"memory");
    #if AXIS_SHARED_Q
    using Score=SCORE_SS_OP<Major::K,Major::K>;
    #else
    using Score=SCORE_OP<Major::K,Major::K>;
    #endif
    #pragma unroll
    for(int d=0;d<128;d+=16) {
      int i=d/4;
      #if AXIS_SHARED_Q
      Score::fma(desc_at(query_base,aoff(0,d)),desc_at(score_base,boff(0,d)),ES(s),d==0?ScaleOut::Zero:ScaleOut::One);
      #else
      Score::fma(qreg[i],qreg[i+1],qreg[i+2],qreg[i+3],desc_at(score_base,boff(0,d)),ES(s),d==0?ScaleOut::Zero:ScaleOut::One);
      #endif
    }
    asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
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
        debug_scores[(col_for_row(row)*CAPACITY+seed_for_row(row))*90+col]=s[i];
      }
      return;
    }
    if(stop_after==1) continue;
    #if AXIS_REUSE_EXP
    float saved_exp[KC/2],norm_recip[2];
    #endif
    float norm_max[2],norm_sum[2];
    #pragma unroll
    for(int half=0;half<2;++half) {
      float direct[KC/4];
      #pragma unroll
      for(int i=0;i<KC/4;++i) {
        direct[i]=ex2(__fmul_rn(s[4*(i/2)+2*half+(i&1)],scale));
        #if AXIS_REUSE_EXP
        saved_exp[4*(i/2)+2*half+(i&1)]=direct[i];
        #endif
      }
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
      for(int i=0;i<KC/4;++i) {
        bs[i]=ex2(__fmaf_rn(s[4*(i/2)+2*half+(i&1)],scale,-mx));
        #if AXIS_REUSE_EXP
        saved_exp[4*(i/2)+2*half+(i&1)]=bs[i];
        #endif
      }
      norm_max[half]=mx;norm_sum[half]=reduce80(bs);
      }
    }
    // Each four-lane row group has the same reduced sums. Compute its two
    // independent logarithms in alternating lanes, then broadcast each.
    float norm_log=lg2((lane&1)?norm_sum[1]:norm_sum[0]);
    #pragma unroll
    for(int half=0;half<2;++half) {
      float norm=__fadd_rn(norm_max[half],__shfl_sync(0xffffffff,norm_log,(lane&~1)+half));
      #if AXIS_REUSE_EXP
      norm_recip[half]=reciprocal(norm_sum[half]);
      #endif
      #pragma unroll
      for(int i=0;i<KC/4;++i) {
        int ix=(i/2)*4+half*2+(i&1);
        s[ix]=__fadd_rn(__fmul_rn(s[ix],scale),-norm);
      }
    }
    // Consume ownership immediately: no 48-element labels array remains
    // live across the weighting phase. Ties still select the smallest slot.
    uint32_t packed_weights[KC/4];
    #pragma unroll
    for(int half=0;half<2;++half) {
      float probabilities[KC/4];
      int row=warp*16+seed+8*half,col=cb*GROUPS_PER_TILE+col_for_row(row);
      int owner=seed_for_row(row);
      #pragma unroll
      for(int i=0;i<KC/4;++i) {
        int ix=(i/2)*4+(i&1);
        unsigned owner0=seed,owner1=seed+8;
        float a,b;
        if constexpr(AXIS_SEED_TILE==8) {
          a=seed<CAPACITY?s[ix+2*half]:-CUDART_INF_F;
          b=-CUDART_INF_F;
        } else {
          a=owner0<CAPACITY?s[ix]:-CUDART_INF_F;
          b=owner1<CAPACITY?s[ix+2]:-CUDART_INF_F;
        }
        #if AXIS_OWNER_MODE == 2
        // Eight seed lanes for one key have a strided participation mask.
        // Ordered IEEE bits preserve finite float ordering, including signs.
        // Compare floats in the ballot so +0/-0 remain tied.
        const unsigned key_mask=0x11111111u<<(lane&3);
        unsigned bits=__float_as_uint(a);
        unsigned ordered=bits^((bits&0x80000000u)?0xffffffffu:0x80000000u);
        unsigned largest=__reduce_max_sync(key_mask,ordered);
        unsigned decoded=largest^((largest&0x80000000u)?0x80000000u:0xffffffffu);
        float best=__uint_as_float(decoded);
        unsigned winners=__ballot_sync(0xffffffff,a==best)&key_mask;
        unsigned label=unsigned(__ffs(winners)-1)>>2;
        #else
        float best=fmaxf(a,b);
        best=fmaxf(best,__shfl_xor_sync(0xffffffff,best,4));
        best=fmaxf(best,__shfl_xor_sync(0xffffffff,best,8));
        best=fmaxf(best,__shfl_xor_sync(0xffffffff,best,16));
        #if AXIS_OWNER_MODE == 1
        unsigned winners=__ballot_sync(0xffffffff,a==best)&(0x11111111u<<(lane&3));
        unsigned label=unsigned(__ffs(winners)-1)>>2;
        #else
        unsigned label=min(a==best?owner0:16u,b==best?owner1:16u);
        label=min(label,__shfl_xor_sync(0xffffffff,label,4));
        label=min(label,__shfl_xor_sync(0xffffffff,label,8));
        label=min(label,__shfl_xor_sync(0xffffffff,label,16));
        #endif
        #endif
        bool valid=2*(lane&3)+8*(i/2)+(i&1)<C;
        #if AXIS_REUSE_EXP
        // Reuse the first stable exponential and its full-row normalizer,
        // avoiding another SFU exponential for every ownership candidate.
        probabilities[i]=(valid && label==owner)?__fmul_rn(saved_exp[ix+2*half],norm_recip[half]):0.f;
        #else
        probabilities[i]=(valid && label==owner)?ex2(s[ix+2*half]):0.f;
        #endif
      }
      float sum=reduce80(probabilities),safe=sum>0?sum:1.f;
      float inv=reciprocal(safe),expected=0.f;
      #if AXIS_SUM_GROUPS > 1
      float expected_parts[AXIS_SUM_GROUPS]={};
      #endif
      #pragma unroll
      for(int i=0;i<KC/4;++i) {
        int ix=(i/2)*4+(i&1)+2*half;
        bool valid=2*(lane&3)+8*(i/2)+(i&1)<C;
        #if AXIS_SUM_GROUPS > 1
        expected_parts[i%AXIS_SUM_GROUPS]=fmaf(probabilities[i],valid?s[ix]:0.f,expected_parts[i%AXIS_SUM_GROUPS]);
        #else
        expected=fmaf(probabilities[i],valid?s[ix]:0.f,expected);
        #endif
      }
      #if AXIS_SUM_GROUPS > 1
      #pragma unroll
      for(int i=0;i<AXIS_SUM_GROUPS;++i)expected+=expected_parts[i];
      #endif
      expected=sum4(expected);
      float correction=__fmul_rn(__fadd_rn(__fmul_rn(expected,inv),-lg2(safe)),0.6931471805599453f);
      int out=(((head*Q+qo)*45+col)*(KO*R)+kb)*CAPACITY+owner;
      if((lane&3)==0 && owner<CAPACITY && col<45)bias[out]=sum>0?-correction:-CUDART_INF_F;
      #pragma unroll
      for(int i=0;i<KC/8;++i) {
        union {bf16 values[2];uint32_t bits;} pair;
        pair.values[0]=__float2bfloat16_rn(__fmul_rn(probabilities[2*i],inv));
        pair.values[1]=__float2bfloat16_rn(__fmul_rn(probabilities[2*i+1],inv));
        packed_weights[2*i+half]=pair.bits;
      }
    }
    if(stop_after==2) continue;
    #if AXIS_FUSED_KV
    // K and V already occupy adjacent 128-column tiles of one MN descriptor.
    // One wider WGMMA reuses the weights and removes the second fence/wait.
    float both_acc[128];
    asm volatile("wgmma.fence.sync.aligned;":::"memory");
    using MergeBoth=MMA_64x256x16_F32BF16BF16_RS<Major::K,Major::MN>;
    #pragma unroll
    for(int ki=0;ki<KC;ki+=16) {
      int wi=(ki/8)*2;
      MergeBoth::fma(packed_weights[wi],packed_weights[wi+1],packed_weights[wi+2],packed_weights[wi+3],
        desc_at(merge_base_k,boff(ki,0)),E128(both_acc),ki==0?ScaleOut::Zero:ScaleOut::One);
    }
    asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
    #endif
    #pragma unroll
    for(int kind=0;kind<2;++kind) {
      #if AXIS_FUSED_KV
      float* acc=both_acc+kind*64;
      #else
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
      #endif
      if(stop_after==3) continue;
      #if AXIS_DIRECT_STORE
      // Optional barrier-free register-to-global path. Exact same BF16 pairs
      // as the shared staging path; GPU timing decides the store tradeoff.
      bf16* dst=kind==0?mk:mv;
      #if AXIS_VECTOR_STORE
      // Transpose a 4x4 matrix of BF16 pairs in registers. The four lanes
      // then write adjacent 16-byte vectors, with no shared-memory staging.
      #pragma unroll
      for(int half=0;half<2;++half) {
        #pragma unroll
        for(int block=0;block<4;++block) {
          int i=16*block+2*half;
          uint4 a={pack_pair(acc[i],acc[i+1]),pack_pair(acc[i+4],acc[i+5]),
                   pack_pair(acc[i+8],acc[i+9]),pack_pair(acc[i+12],acc[i+13])};
          uint4 b=fragment_transpose(a,lane);
          int row=warp*16+seed+8*half,d=block*32+(lane&3)*8;
          int col=cb*GROUPS_PER_TILE+col_for_row(row),c=seed_for_row(row);
          int64_t oi=((((int64_t(head)*Q+qo)*45+col)*(KO*R)+kb)*CAPACITY+c)*128+d;
          if(c<CAPACITY && col<45)store_output(dst+oi,b,cache_first);
        }
      }
      #else
      #pragma unroll
      for(int i=0;i<64;i+=2) {
        int row=warp*16+seed+8*((i&3)>>1);
        int d=2*(lane&3)+8*(i>>2);
        int col=cb*GROUPS_PER_TILE+col_for_row(row),c=seed_for_row(row);
        int64_t oi=((((int64_t(head)*Q+qo)*45+col)*(KO*R)+kb)*CAPACITY+c)*128+d;
        if(c<CAPACITY && col<45)
          *reinterpret_cast<__nv_bfloat162*>(dst+oi)=__floats2bfloat162_rn(acc[i],acc[i+1]);
      }
      #endif
      #else
      #pragma unroll
      for(int i=0;i<64;i+=2) {
        int row=warp*16+seed+8*((i&3)>>1);
        int d=2*(lane&3)+8*(i>>2);
        *reinterpret_cast<__nv_bfloat162*>(temp+storeoff(row,d))=__floats2bfloat162_rn(acc[i],acc[i+1]);
      }
      __syncthreads();
      bf16* dst=kind==0?mk:mv;
      for(int vec=tid;vec<ROWS_PER_TILE*16;vec+=BLOCK_THREADS) {
        int row=vec/16,d=(vec%16)*8,col=cb*GROUPS_PER_TILE+col_for_row(row),c=seed_for_row(row);
        int64_t oi=((((int64_t(head)*Q+qo)*45+col)*(KO*R)+kb)*CAPACITY+c)*128+d;
        if(c<CAPACITY && col<45)store_output(dst+oi,*reinterpret_cast<const uint4*>(temp+storeoff(row,d)),cache_first);
      }
      __syncthreads();
      #endif
    }
  }
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

static void configure_merge() {
 #if AXIS_CONFIGURE_ONCE
 static std::mutex mutex;
 static std::unordered_set<int> configured;
 int device=0;C10_CUDA_CHECK(cudaGetDevice(&device));
 std::lock_guard<std::mutex> guard(mutex);
 if(configured.count(device))return;
 #endif
 constexpr int bytes=(KC*128*2+((AXIS_DIRECT_STORE && !AXIS_SHARED_Q)?0:ROWS_PER_TILE*128))*2;
 C10_CUDA_CHECK(cudaFuncSetAttribute(merge_kernel<21,AXIS_HEADS>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
 #if AXIS_CARVEOUT >= 0
 C10_CUDA_CHECK(cudaFuncSetAttribute(merge_kernel<21,AXIS_HEADS>,cudaFuncAttributePreferredSharedMemoryCarveout,AXIS_CARVEOUT));
 #endif
 #if AXIS_CONFIGURE_ONCE
 configured.insert(device);
 #endif
}

void run(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor seeds,
 torch::Tensor mk,torch::Tensor mv,torch::Tensor bias,int hs,int ah,int tiles) {
 TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.is_contiguous(),"Contiguous CUDA BF16 Q required");
 TORCH_CHECK(q.numel()==int64_t(AXIS_HEADS)*AXIS_OUTER*45*80*128,"Compiled shape mismatch");
 TORCH_CHECK(ah==AXIS_ACTIVE_HEADS && hs>=0 && hs+ah<=AXIS_HEADS && tiles>0,"Head/tile bounds");
 #if AXIS_HOIST_SEEDS
 TORCH_CHECK(tiles==AXIS_OUTER,"Seed hoisting requires one complete time loop per CTA");
 #endif
 TORCH_CHECK(seeds.scalar_type()==at::kByte && seeds.numel()==int64_t(AXIS_HEADS)*45*AXIS_OUTER*40*CAPACITY,"Exact capacity seed shape");
 constexpr int bytes=(KC*128*2+((AXIS_DIRECT_STORE && !AXIS_SHARED_Q)?0:ROWS_PER_TILE*128))*2;
 configure_merge();
 dim3 grid(AXIS_OUTER*40,ah,(AXIS_OUTER*COLUMN_TILES+tiles-1)/tiles);
 merge_kernel<21,AXIS_HEADS><<<grid,BLOCK_THREADS,bytes,at::cuda::getCurrentCUDAStream()>>>(
   (const bf16*)q.data_ptr(),(const bf16*)k.data_ptr(),(const bf16*)v.data_ptr(),seeds.data_ptr<uint8_t>(),
   (bf16*)mk.data_ptr(),(bf16*)mv.data_ptr(),bias.data_ptr<float>(),0,0,0,0,0,0,hs,0,
   float(0.08838834764831845*1.4426950408889634),tiles);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<int> resources() {
 cudaFuncAttributes a{};C10_CUDA_CHECK(cudaFuncGetAttributes(&a,merge_kernel<21,AXIS_HEADS>));
 int active=0;constexpr int bytes=(KC*128*2+((AXIS_DIRECT_STORE && !AXIS_SHARED_Q)?0:ROWS_PER_TILE*128))*2;
 C10_CUDA_CHECK(cudaFuncSetAttribute(merge_kernel<21,AXIS_HEADS>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
 C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,merge_kernel<21,AXIS_HEADS>,BLOCK_THREADS,bytes));
 return {a.numRegs,int(a.localSizeBytes),bytes,active};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("run",&run);m.def("pack",&pack);m.def("resources",&resources);}
