// Directed W/H exact C8, WGMMA/TMA with r2 packed-owner selection.
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
__device__ __forceinline__ int col_for_row(int row) {return (row>>4)*2+(row&1);}
__device__ __forceinline__ int seed_for_row(int row) {return (row&15)>>1;}
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
// K/V stay resident across query tiles. All score, winner, and cluster
// reductions remain in registers. Query, weights, and store staging alias
// the same shared buffer after their respective WGMMA operations finish.
template<int TILES,int TOTAL_HEADS>
__global__ __launch_bounds__(128,4) void merge_kernel(
 const bf16* __restrict__ q,const bf16* __restrict__ k,const bf16* __restrict__ v,
 const uint8_t* __restrict__ seeds,bf16* __restrict__ mk,bf16* __restrict__ mv,
 float* __restrict__ bias,int Q_arg,int KO_arg,int R_arg,int C_arg,int H_arg,int AH_arg,int HS,int RF_arg,float scale,
 float* debug_scores_arg,int stop_after_arg,int tma_store_arg,
 const __grid_constant__ CUtensorMap map_k,const __grid_constant__ CUtensorMap map_v,int tile_count) {
  constexpr float* debug_scores=nullptr;
  constexpr int stop_after=0,tma_store=1;
  constexpr int Q=AXIS_OUTER,KO=Q,R=AXIS_ROWS,C=AXIS_COLS,H=TOTAL_HEADS,AH=AXIS_ACTIVE_HEADS,RF=Q;
  extern __shared__ __align__(128) unsigned char raw[];
  bf16* sk=reinterpret_cast<bf16*>(raw);
  bf16* sv=sk+KC*128;
  bf16* temp=sv+KC*128;
  int tid=threadIdx.x,lane=tid&31,warp=tid>>5,seed=lane>>2;
  int block=blockIdx.x,head=blockIdx.y,gh=HS+head;
  int ko=block%KO,kr=block/KO;
  int kb=ko*R+kr;
  uint64_t cache_last,cache_first=0;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0,1.0;":"=l"(cache_last));
  for(int vec=tid;vec<KC*16;vec+=128) {
    int ki=vec/16,d=(vec%16)*8;
    int src=((kb*C+ki)*H+gh)*128+d;
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
  int total=Q*((C+7)/8),start=blockIdx.z*tile_count;
  for(int tile=start;tile<min(start+tile_count,total);++tile) {
    int qo=tile%Q,cb=tile/Q,rf=0;
    int row0=warp*16+seed,col0=cb*8+col_for_row(row0);
    int si0=(((qo*C+col0)*AH+head)*(KO*R)+kb)*8+seed_for_row(row0);
    int col1=cb*8+col_for_row(row0+8);
    int si1=(((qo*C+col1)*AH+head)*(KO*R)+kb)*8+seed_for_row(row0+8);
    int sr0=col0<C?seeds[si0]:0,sr1=col1<C?seeds[si1]:0;
    int qi0=(((qo*R+sr0)*C+col0)*H+gh)*128+(lane&3)*8;
    int qi1=(((qo*R+sr1)*C+col1)*H+gh)*128+(lane&3)*8;
    uint32_t qreg[32];
    #pragma unroll
    for(int d=0;d<128;d+=32) {
      uint4 a=col0<C?query_vector(q+qi0+d,cache_last,lane):make_uint4(0,0,0,0),b=col1<C?query_vector(q+qi1+d,cache_last,lane):make_uint4(0,0,0,0);
      int i=d/4;
      qreg[i]=a.x;qreg[i+1]=b.x;qreg[i+2]=a.y;qreg[i+3]=b.y;
      qreg[i+4]=a.z;qreg[i+5]=b.z;qreg[i+6]=a.w;qreg[i+7]=b.w;
    }
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
        s[ix]=__fadd_rn(__fmul_rn(s[ix],scale),-norm);
      }
    }
    // Eight seeds share each query column. Each thread owns seed c and c+4.
    // r2 compares positive 1-log_probability with the owner in the low bits.
    // slow_reference retains full FP32 ordering and deterministic lower-index ties.
    int labels[KC/4];
    const unsigned owner0=seed_for_row(warp*16+seed),owner1=owner0+4;
    #pragma unroll
    for(int i=0;i<KC/4;++i) {
      int ix=(i/2)*4+(i&1);
      float a=s[ix],b=s[ix+2];
      unsigned label;
      #if AXIS_C8_R2
      unsigned ka=(__float_as_uint(__fadd_rn(1.f,-a))&~7u)|owner0;
      unsigned kb=(__float_as_uint(__fadd_rn(1.f,-b))&~7u)|owner1;
      unsigned key=min(ka,kb);
      key=min(key,__shfl_xor_sync(0xffffffff,key,8));
      key=min(key,__shfl_xor_sync(0xffffffff,key,16));
      label=key&7u;
      #else
      float best=fmaxf(a,b);
      best=fmaxf(best,__shfl_xor_sync(0xffffffff,best,8));
      best=fmaxf(best,__shfl_xor_sync(0xffffffff,best,16));
      label=min(a==best?owner0:8u,b==best?owner1:8u);
      label=min(label,__shfl_xor_sync(0xffffffff,label,8));
      label=min(label,__shfl_xor_sync(0xffffffff,label,16));
      #endif
      labels[i]=label;
    }
    uint32_t packed_weights[KC/4];
    #pragma unroll
    for(int half=0;half<2;++half) {
      float winners[KC/4],probabilities[KC/4];
      int row=warp*16+seed+8*half,col=cb*8+col_for_row(row);
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
      int out=((qo*C+col)*AH+head)*(KO*R*8)+kb*8+owner;
      if((lane&3)==0 && col<C)bias[out]=sum>0?correction:CUDART_INF_F;
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
      if(tma_store==2) {
        bf16* dst=kind==0?mk:mv;
        #pragma unroll
        for(int i=0;i<64;i+=2) {
          int row=warp*16+seed+8*((i&3)>>1);
          int d=2*(lane&3)+8*(i>>2),col=cb*8+col_for_row(row);
          int64_t oi=(((int64_t(qo)*C+col)*AH+head)*(KO*R*8)+kb*8+seed_for_row(row))*128+d;
          union {bf16 values[2];uint32_t bits;} pair;
          pair.values[0]=__float2bfloat16_rn(acc[i]);pair.values[1]=__float2bfloat16_rn(acc[i+1]);
          *reinterpret_cast<uint32_t*>(dst+oi)=pair.bits;
        }
        continue;
      }
      if(tma_store==1) {
        if(tid==0) asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");
        __syncthreads();
        #pragma unroll
        for(int g=0;g<8;++g) {
          uint32_t packed[4];
          #pragma unroll
          for(int j=0;j<4;++j) {
            __nv_bfloat162 value=__floats2bfloat162_rn(acc[g*8+j*2],acc[g*8+j*2+1]);
            packed[j]=*reinterpret_cast<uint32_t*>(&value);
          }
          int row=warp*16+(lane&7)+8*((lane>>3)&1);
          int canonical=col_for_row(row)*8+seed_for_row(row);
          int d=g*16+8*(lane>>4),offset=canonical*128+d;
          uint32_t sp=__cvta_generic_to_shared(temp+(offset^(((offset>>6)&7)<<3)));
          asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0],{%1,%2,%3,%4};"::"r"(sp),"r"(packed[0]),"r"(packed[1]),"r"(packed[2]),"r"(packed[3]):"memory");
        }
        asm volatile("fence.proxy.async.shared::cta;":::"memory");
        __syncthreads();
        if(tid==0) {
          uint32_t p=__cvta_generic_to_shared(temp);
          const CUtensorMap* map=kind==0?&map_k:&map_v;
          asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%0, {%1,%2,%3,%4,%5}], [%6];"::"l"(map),"r"(0),"r"(kb*16),"r"(head),"r"(cb*8),"r"(qo),"r"(p):"memory");
          asm volatile("cp.async.bulk.commit_group;":::"memory");
        }
        continue;
      }
      #pragma unroll
      for(int i=0;i<64;i+=2) {
        int row=warp*16+seed+8*((i&3)>>1);
        int d=2*(lane&3)+8*(i>>2)+(i&1);
        *reinterpret_cast<__nv_bfloat162*>(temp+storeoff(row,d))=__floats2bfloat162_rn(acc[i],acc[i+1]);
      }
      __syncthreads();
      bf16* dst=kind==0?mk:mv;
      for(int vec=tid;vec<64*16;vec+=128) {
        int row=vec/16,d=(vec%16)*8,col=cb*8+col_for_row(row);
        int64_t oi=(((int64_t(qo)*C+col)*AH+head)*(KO*R*8)+kb*8+seed_for_row(row))*128+d;
        store_output(dst+oi,*reinterpret_cast<const uint4*>(temp+storeoff(row,d)),cache_first);
      }
      __syncthreads();
    }
  }
  if(tma_store==1 && tid==0) asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");
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
 torch::Tensor mk,torch::Tensor mv,torch::Tensor bias,int hs,int ah,int tiles,double scale,torch::Tensor debug,int stop_after,int tma_store) {
 TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.dim()==6 && q.size(0)==1,"Require batch-one native BF16 QKV");
 TORCH_CHECK(q.size(3)==AXIS_COLS && q.size(5)==128,"Compiled columns/D128 required");
 TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && seeds.is_contiguous(),"Require contiguous inputs");
 TORCH_CHECK(seeds.scalar_type()==at::kByte,"Require uint8 seeds");
 int qo=q.size(1),ko=k.size(1),r=q.size(2),c=AXIS_COLS,h=q.size(4),rf=qo;
 TORCH_CHECK(qo==AXIS_OUTER && ko==qo && r==AXIS_ROWS,"Compiled axis shape required");
 TORCH_CHECK(h==AXIS_HEADS && ah==AXIS_ACTIVE_HEADS,"Compiled heads required");
 dim3 grid(ko*r,ah,(qo*((c+7)/8)+tiles-1)/tiles);
 if(debug.numel()) grid=dim3(1,1,1);
 auto stream=at::cuda::getCurrentCUDAStream();
 const int bytes=(KC*128*2+64*128)*2;
 CUtensorMap map_k{},map_v{};
 if(tma_store==1) {
   uint64_t dims[5]={64,uint64_t(ko*r*16),uint64_t(ah),uint64_t(c),uint64_t(qo)};
   uint64_t strides[4];uint64_t stride=128;
   for(int i=0;i<4;++i) {strides[i]=stride;stride*=dims[i+1];}
   uint32_t box[5]={64,16,1,8,1},element_stride[5]={1,1,1,1,1};
   auto encode=[&](CUtensorMap& map,torch::Tensor x) {
    auto status=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,5,x.data_ptr(),dims,strides,box,element_stride,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(status==CUDA_SUCCESS,"TMA descriptor creation failed: ",int(status));
   };
   encode(map_k,mk);encode(map_v,mv);
 }
 auto launch=[&](auto tag) {
   constexpr int t=decltype(tag)::value;
   auto launch_heads=[&](auto head_tag) {
   constexpr int heads=decltype(head_tag)::value;
   C10_CUDA_CHECK(cudaFuncSetAttribute(merge_kernel<t,heads>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
   merge_kernel<t,heads><<<grid,128,bytes,stream>>>(
    (const bf16*)q.data_ptr(),(const bf16*)k.data_ptr(),(const bf16*)v.data_ptr(),seeds.data_ptr<uint8_t>(),
    (bf16*)mk.data_ptr(),(bf16*)mv.data_ptr(),bias.data_ptr<float>(),qo,ko,r,c,h,ah,hs,rf,float(scale*1.4426950408889634),
    debug.numel()?debug.data_ptr<float>():nullptr,stop_after,tma_store,map_k,map_v,tiles);
   };
   launch_heads(std::integral_constant<int,AXIS_HEADS>{});
 };
 TORCH_CHECK(tiles>0,"Positive tile count required");
 launch(std::integral_constant<int,21>{});
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<int> resources() {
  cudaFuncAttributes attributes{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes,merge_kernel<21,AXIS_HEADS>));
  int active=0;
  constexpr int bytes=(KC*128*2+64*128)*2;
  C10_CUDA_CHECK(cudaFuncSetAttribute(merge_kernel<21,AXIS_HEADS>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,merge_kernel<21,AXIS_HEADS>,128,bytes));
  return {attributes.numRegs,int(attributes.localSizeBytes),bytes,active};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("run",&run); m.def("pack",&pack);m.def("resources",&resources); }
