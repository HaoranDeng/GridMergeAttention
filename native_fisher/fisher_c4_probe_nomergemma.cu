// TIMING PROBE ONLY, NOT A CORRECT KERNEL. Isolates the compressed K/V merge
// WGMMA cost by replacing both of its accumulator groups with a cheap fill
// that still consumes the packed weights and the resident K/V tile.
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

#define E8(a,o) a[o],a[o+1],a[o+2],a[o+3],a[o+4],a[o+5],a[o+6],a[o+7]
#define E32(a) E8(a,0),E8(a,8),E8(a,16),E8(a,24)
#define E40(a) E32(a),E8(a,32)
#define E64(a) E32(a),E8(a,32),E8(a,40),E8(a,48),E8(a,56)

__device__ __forceinline__ uint64_t desc(const bf16* p,int lead,int stride) {
  uint64_t s=__cvta_generic_to_shared(p);
  return ((s&0x3ffff)>>4) | (uint64_t(lead>>4)<<16) | (uint64_t(stride>>4)<<32) | (uint64_t(1)<<62);
}
__device__ __forceinline__ int aoff(int row,int k) {
  int offset=row*64+(k&63)+(k>>6)*4096;
  return offset^(((offset>>6)&7)<<3);
}
__device__ __forceinline__ int boff(int k,int n) {
  int offset=k*64+(n&63)+(n>>6)*5120;
  return offset^(((offset>>6)&7)<<3);
}
__device__ __forceinline__ int col_for_row(int row) {return (row>>3)*2+(row&1);}
__device__ __forceinline__ int seed_for_row(int row) {return (row&7)>>1;}
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
  return log2f(x);
}
__device__ __forceinline__ float sum4(float x) {
  x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,2,4));
  return __fadd_rn(x,__shfl_xor_sync(0xffffffff,x,1,4));
}
__device__ __forceinline__ float max4(float x) {
  x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,1,4));
  return fmaxf(x,__shfl_xor_sync(0xffffffff,x,2,4));
}
__device__ __forceinline__ float tree80(float (&p)[10]) {
  float main=__fadd_rn(__fadd_rn(__fadd_rn(p[0],p[1]),__fadd_rn(p[2],p[3])),
                       __fadd_rn(__fadd_rn(p[4],p[5]),__fadd_rn(p[6],p[7])));
  return __fadd_rn(sum4(main),sum4(__fadd_rn(p[8],p[9])));
}
__device__ __forceinline__ float reduce80(float (&p)[20]) {
  float main=p[0];
  #pragma unroll
  for(int i=1;i<16;++i) main=__fadd_rn(main,p[i]);
  float tail=p[16];
  #pragma unroll
  for(int i=17;i<20;++i) tail=__fadd_rn(tail,p[i]);
  return __fadd_rn(sum4(main),sum4(tail));
}

__device__ __forceinline__ float reciprocal(float x){float y;asm("div.full.f32 %0,0f3F800000,%1;":"=f"(y):"f"(x));return y;}
__device__ __forceinline__ float dot80(float (&p)[20],float (&x)[20]){
 float a=__fmaf_rn(p[0],x[0],__fmul_rn(p[1],x[1]));
 #pragma unroll
 for(int i=2;i<16;++i)a=__fmaf_rn(p[i],x[i],a);
 float b=__fmaf_rn(p[16],x[16],__fmul_rn(p[17],x[17]));
 #pragma unroll
 for(int i=18;i<20;++i)b=__fmaf_rn(p[i],x[i],b);
 return __fadd_rn(sum4(a),sum4(b));
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
  constexpr int Q=21,KO=21,R=45,C=80,H=TOTAL_HEADS,AH=TOTAL_HEADS==40?5:1,RF=1;
  extern __shared__ __align__(128) unsigned char raw[];
  bf16* sk=reinterpret_cast<bf16*>(raw);
  bf16* sv=sk+80*128;
  bf16* temp=sv+80*128;
  int tid=threadIdx.x,lane=tid&31,warp=tid>>5,seed=lane>>2;
  int block=blockIdx.x,head=blockIdx.y,gh=HS+head;
  int ko=block%KO,kr=block/KO;
  int kb=ko*R+kr;
  uint64_t cache_last,cache_first=0;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0,1.0;":"=l"(cache_last));
  for(int vec=tid;vec<80*16;vec+=128) {
    int ki=vec/16,d=(vec%16)*8;
    int src=((kb*80+ki)*H+gh)*128+d;
    *reinterpret_cast<uint4*>(sk+boff(ki,d))=*reinterpret_cast<const uint4*>(k+src);
    *reinterpret_cast<uint4*>(sv+boff(ki,d))=*reinterpret_cast<const uint4*>(v+src);
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;":::"memory");
  int total=Q*(C/16),start=blockIdx.z*tile_count;
  for(int tile=start;tile<min(start+tile_count,total);++tile) {
    int qo=tile%Q,cb=tile/Q,rf=0;
    int row0=warp*16+seed,col0=cb*16+col_for_row(row0);
    int si0=((((rf*KO+ko)*R+kr)*C+col0)*H+gh)*4+seed_for_row(row0);
    int col1=cb*16+col_for_row(row0+8);
    int si1=((((rf*KO+ko)*R+kr)*C+col1)*H+gh)*4+seed_for_row(row0+8);
    int sr0=seeds[si0],sr1=seeds[si1];
    int qi0=(((qo*R+sr0)*C+col0)*H+gh)*128+(lane&3)*8;
    int qi1=(((qo*R+sr1)*C+col1)*H+gh)*128+(lane&3)*8;
    uint32_t qreg[32];
    #pragma unroll
    for(int d=0;d<128;d+=32) {
      uint4 a=query_vector(q+qi0+d,cache_last,lane),b=query_vector(q+qi1+d,cache_last,lane);
      int i=d/4;
      qreg[i]=a.x;qreg[i+1]=b.x;qreg[i+2]=a.y;qreg[i+3]=b.y;
      qreg[i+4]=a.z;qreg[i+5]=b.z;qreg[i+6]=a.w;qreg[i+7]=b.w;
    }
    #pragma unroll
    for(int i=0;i<32;++i) asm volatile("" : "+r"(qreg[i]) :: "memory");
    float s[40];
    asm volatile("wgmma.fence.sync.aligned;":::"memory");
    using Score=MMA_64x80x16_F32BF16BF16_RS<Major::K,Major::K>;
    #pragma unroll
    for(int d=0;d<128;d+=16) {
      int i=d/4;
      Score::fma(qreg[i],qreg[i+1],qreg[i+2],qreg[i+3],desc(sk+boff(0,d),16,1024),E40(s),d==0?ScaleOut::Zero:ScaleOut::One);
    }
    asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
    if(debug_scores) {
      #pragma unroll
      for(int i=0;i<40;++i) {
        int row=warp*16+seed+8*((i&3)>>1);
        int col=2*(lane&3)+8*(i>>2)+(i&1);
        debug_scores[(col_for_row(row)*4+seed_for_row(row))*80+col]=s[i];
      }
      return;
    }
    if(stop_after==1) continue;
    float norm_max[2],norm_sum[2];
    #pragma unroll
    for(int half=0;half<2;++half) {
      float direct[20];
      #pragma unroll
      for(int i=0;i<20;++i)
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
      for(int i=0;i<10;++i) mx=fmaxf(mx,fmaxf(__fmul_rn(s[4*i+2*half],scale),__fmul_rn(s[4*i+2*half+1],scale)));
      mx=max4(mx);
      float bs[20];
      #pragma unroll
      for(int i=0;i<20;++i)
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
      for(int i=0;i<20;++i) {
        int ix=(i/2)*4+half*2+(i&1);
        s[ix]=__fadd_rn(__fmul_rn(s[ix],scale),-norm);
      }
    }
    // C4 packs sixteen independent columns into 64 WGMMA rows.
    // The two accumulator halves represent different columns, so each
    // half performs its own four-seed ownership and exact SQ1 removal.
    uint32_t packed_weights[20];
    float residual_scales[2];
    #pragma unroll
    for(int half=0;half<2;++half) {
      float winners[20],probabilities[20];
      int row=warp*16+seed+8*half,col=cb*16+col_for_row(row);
      int owner=seed_for_row(row);
      #pragma unroll
      for(int i=0;i<20;++i) {
        int ix=(i/2)*4+(i&1)+2*half;
        float best=s[ix];
        best=fmaxf(best,__shfl_xor_sync(0xffffffff,best,8));
        best=fmaxf(best,__shfl_xor_sync(0xffffffff,best,16));
        bool match=s[ix]==best;
        uint32_t matches=__ballot_sync(0xffffffff,match);
        // Equal-score owners are eight lanes apart. Only lower owners
        // from this same column/key lane may suppress the current match.
        uint32_t lower=((1u<<lane)-1u)&(0x01010101u<<(lane&7));
        bool wins=match && (matches&lower)==0;
        winners[i]=best;
        probabilities[i]=wins?ex2(best):0.f;
      }
      float sum=reduce80(probabilities),safe=sum>0?sum:1.f;
      float inv=reciprocal(safe);
      float expected=dot80(probabilities,winners);
      float selected=0.f;
      #pragma unroll
      for(int i=0;i<20;++i) {
        probabilities[i]=__fmul_rn(probabilities[i],inv);
        int ki=(i/2)*8+2*(lane&3)+(i&1);
        if(ki==col)selected=probabilities[i];
      }
      float selected_weight=sum4(selected);
      float residual=__fadd_rn(1.f,-selected_weight);
      bool valid=sum>0 && residual>1.e-6f;
      float safe_residual=valid?residual:1.f;
      residual_scales[half]=reciprocal(safe_residual);
      // Independent row reductions are identical in each four-lane group.
      // Compute the three logarithms in separate lanes with one instruction
      // stream, then broadcast; singleton selection has only one nonzero term.
      float log_argument=(lane&3)==0?safe:(lane&3)==1?fmaxf(selected_weight,1.e-20f):safe_residual;
      float logarithm=lg2(log_argument);
      float log_sum=__shfl_sync(0xffffffff,logarithm,lane&~3);
      float log_selected=__shfl_sync(0xffffffff,logarithm,(lane&~3)+1);
      float log_residual=__shfl_sync(0xffffffff,logarithm,(lane&~3)+2);
      float correction=__fmul_rn(__fadd_rn(__fmul_rn(expected,inv),-log_sum),0.6931471805599453f);
      float selected_entropy=__fmul_rn(__fmul_rn(selected_weight,log_selected),0.6931471805599453f);
      correction=__fadd_rn(__fmul_rn(__fadd_rn(correction,-selected_entropy),residual_scales[half]),-__fmul_rn(log_residual,0.6931471805599453f));
      int out=((qo*C+col)*AH+head)*(KO*R*4)+kb*4+owner;
      if((lane&3)==0)bias[out]=valid?correction:CUDART_INF_F;
      #pragma unroll
      for(int i=0;i<10;++i) {
        int ki=(i/1)*8+2*(lane&3);
        union { bf16 values[2]; uint32_t bits; } pair;
        pair.values[0]=__float2bfloat16_rn(ki==col?0.f:probabilities[2*i]);
        pair.values[1]=__float2bfloat16_rn(ki+1==col?0.f:probabilities[2*i+1]);
        packed_weights[2*i+half]=pair.bits;
      }
    }
    if(stop_after==2) continue;
    #pragma unroll
    for(int kind=0;kind<2;++kind) {
      // TIMING PROBE: the five compressed K/V WGMMA issues are replaced by a
      // cheap fill that still consumes the packed weights and reads the
      // resident K/V tile, so neither becomes dead code.
      const bf16* tile=(kind==0?sk:sv);
      float base=__bfloat162float(tile[boff(lane&63,(lane&3)*8)]);
      float acc[64];
      #pragma unroll
      for(int i=0;i<64;++i)
        acc[i]=__fmaf_rn(__int2float_rn(int(packed_weights[i&15]&0xffff)),1.e-6f,base);
      if(stop_after==3) continue;
      // The production operator rounds unpeeled normalized weights to BF16,
      // multiplies K/V, and only then divides the FP32 accumulator by residual
      // mass. Moving this division before BF16 rounding changes the operator.
      #pragma unroll
      for(int i=0;i<64;++i)acc[i]=__fmul_rn(acc[i],residual_scales[(i&3)>>1]);
      if(tma_store==2) {
        bf16* dst=kind==0?mk:mv;
        #pragma unroll
        for(int i=0;i<64;i+=2) {
          int row=warp*16+seed+8*((i&3)>>1);
          int d=2*(lane&3)+8*(i>>2),col=cb*16+col_for_row(row);
          int64_t oi=(((int64_t(qo)*C+col)*AH+head)*(KO*R*4)+kb*4+seed_for_row(row))*128+d;
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
          int canonical=col_for_row(row)*4+seed_for_row(row);
          int d=g*16+8*(lane>>4),offset=canonical*128+d;
          uint32_t sp=__cvta_generic_to_shared(temp+(offset^(((offset>>6)&7)<<3)));
          asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0],{%1,%2,%3,%4};"::"r"(sp),"r"(packed[0]),"r"(packed[1]),"r"(packed[2]),"r"(packed[3]):"memory");
        }
        asm volatile("fence.proxy.async.shared::cta;":::"memory");
        __syncthreads();
        if(tid==0) {
          uint32_t p=__cvta_generic_to_shared(temp);
          const CUtensorMap* map=kind==0?&map_k:&map_v;
          asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%0, {%1,%2,%3,%4,%5}], [%6];"::"l"(map),"r"(0),"r"(kb*8),"r"(head),"r"(cb*16),"r"(qo),"r"(p):"memory");
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
        int row=vec/16,d=(vec%16)*8,col=cb*16+col_for_row(row);
        int64_t oi=(((int64_t(qo)*C+col)*AH+head)*(KO*R*4)+kb*4+seed_for_row(row))*128+d;
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
      vec<vectors;vec+=int64_t(gridDim.x)*blockDim.x) {
    uint4 a=*reinterpret_cast<const uint4*>(input+vec*8);
    uint4 b=fragment_transpose(a,lane);
    *reinterpret_cast<uint4*>(output+vec*8)=b;
  }
}

void pack(torch::Tensor input,torch::Tensor output) {
  TORCH_CHECK(input.is_cuda() && output.is_cuda() && input.device()==output.device(),"CUDA tensors on the same device required");
  TORCH_CHECK(input.scalar_type()==at::kBFloat16 && output.scalar_type()==at::kBFloat16 && input.sizes()==output.sizes(),"Identical BF16 shape required");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous() && input.numel()%2048==0,"Contiguous native Q shape required");
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
 TORCH_CHECK(q.size(3)==80 && q.size(5)==128,"Require native80 D128");
 TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && seeds.is_contiguous(),"Require contiguous inputs");
 TORCH_CHECK(seeds.scalar_type()==at::kByte,"Require uint8 seeds");
 int qo=q.size(1),ko=k.size(1),r=q.size(2),c=80,h=q.size(4),rf=seeds.size(1);
 TORCH_CHECK(qo==21 && ko==21 && r==45 && rf==1,"Require 21 frames, 45 rows, and one router frame");
 TORCH_CHECK((h==1 && ah==1) || (h==40 && ah==5),"Require native H40/AH5 or diagnostic H1/AH1");
 dim3 grid(ko*r,ah,(qo*5+tiles-1)/tiles);
 if(debug.numel()) grid=dim3(1,1,1);
 auto stream=at::cuda::getCurrentCUDAStream();
 const int bytes=(80*128*2+64*128)*2;
 CUtensorMap map_k{},map_v{};
 if(tma_store==1) {
   uint64_t dims[5]={64,uint64_t(ko*r*8),uint64_t(ah),80,uint64_t(qo)};
   uint64_t strides[4];uint64_t stride=128;
   for(int i=0;i<4;++i) {strides[i]=stride;stride*=dims[i+1];}
   uint32_t box[5]={64,8,1,16,1},element_stride[5]={1,1,1,1,1};
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
   if(h==40) launch_heads(std::integral_constant<int,40>{});
   else launch_heads(std::integral_constant<int,1>{});
 };
 TORCH_CHECK(tiles==1 || tiles==3 || tiles==7 || tiles==21,"Require tiles 1/3/7/21");
 launch(std::integral_constant<int,21>{});
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<int> resources() {
  cudaFuncAttributes attributes{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes,merge_kernel<21,40>));
  int active=0;
  constexpr int bytes=(80*128*2+64*128)*2;
  C10_CUDA_CHECK(cudaFuncSetAttribute(merge_kernel<21,40>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,merge_kernel<21,40>,128,bytes));
  return {attributes.numRegs,int(attributes.localSizeBytes),bytes,active};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("run",&run); m.def("pack",&pack);m.def("resources",&resources); }
