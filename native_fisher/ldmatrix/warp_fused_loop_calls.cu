// Whole producer/consumer loops are separate noinline functions.
// One column per128-thread CTA; public ranges remain column pairs.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
// Stream merge fragments directly to shared storage; a separate K-overflow
// buffer avoids retaining a 64-float K accumulator across the CTA barrier.
#define FISHER_WEIGHT_QUALIFIER __forceinline__
#include "warp_route.cuh"
using namespace fisher_warp;

constexpr int PARTS=16,QUERY_ROWS=48,SHARED_BYTES=76416;
struct NativeSeedQuery {
  const bf16* first;
  const bf16* second;
  __device__ __forceinline__ uint32_t operator()(int row,int offset) const {
    return *reinterpret_cast<const uint32_t*>((row<8?first:second)+offset);
  }
};

// Four FP32 accumulators per feature panel, rather than an entire16x128
// tile live across synchronization. BF16 rounding remains after residual scale.
template<int CAP>
__device__ __forceinline__ void merge_to_ring(const uint32_t (&weights)[20],
    const float (&residual)[2],const bf16* raw,bf16* ring,bf16* overflow,int pending) {
  int lane=threadIdx.x&31,g=lane>>2,t=lane&3;
  #pragma unroll 1
  for(int d=0;d<128;d+=8) {
    float c[4]={};
    #pragma unroll
    for(int ki=0;ki<80;ki+=16) {
      int wi=ki/4;
      uint32_t a[4]={weights[wi],weights[wi+1],weights[wi+2],weights[wi+3]},b[2];
      merge_b(raw,ki,d,lane,b);mma(a,b,c);
    }
    #pragma unroll
    for(int i=0;i<4;++i) {
      int row=g+8*(i>>1),dim=d+2*t+(i&1),position=pending+row;
      if(row<CAP) {
        bf16 value=__float2bfloat16_rn(__fmul_rn(c[i],residual[i>>1]));
        if(position<64)ring[ring_offset(position,dim)]=value;
        else overflow[(position-64)*128+dim]=value;
      }
    }
  }
}

__device__ __forceinline__ void store_partial(const RouteState& state,
    float* output,float* maximum,float* denominator,int row_base) {
  int lane=threadIdx.x&31,g=lane>>2,t=lane&3;
  #pragma unroll
  for(int i=0;i<64;++i) {
    int row=row_base+g+8*((i&3)>>1),d=8*(i/4)+2*t+(i&1);
    if(row<QUERY_ROWS)output[row*128+d]=row<45?state.acc[i]:0.f;
  }
  if(t==0) {
    #pragma unroll
    for(int half=0;half<2;++half) {
      int row=row_base+g+8*half;
      if(row<QUERY_ROWS) {
        maximum[row]=row<45?state.maximum[half]:-CUDART_INF_F;
        denominator[row]=row<45?state.sum[half]:0.f;
      }
    }
  }
}

// Producer and consumer warps follow separate loops so their register live
// ranges do not overlap. Every loop executes the same named CTA barrier
// sequence; unlike conditional __syncthreads, the explicit barrier count
// and common ID rendezvous all128 threads at each matching phase.
__device__ __forceinline__ void phase_barrier() {
  asm volatile("bar.sync 0, 128;":::"memory");
}
__device__ __forceinline__ void load_raw(const bf16* k,const bf16* v,
    bf16* raw_k,bf16* raw_v,int kb,int head) {
  for(int vec=threadIdx.x;vec<80*16;vec+=128) {
    int row=vec/16,d=(vec%16)*8,offset=((kb*80+row)*40+head)*128+d;
    *reinterpret_cast<uint4*>(raw_k+raw_offset(row,d))=*reinterpret_cast<const uint4*>(k+offset);
    *reinterpret_cast<uint4*>(raw_v+raw_offset(row,d))=*reinterpret_cast<const uint4*>(v+offset);
  }
}
template<int CAP>
__device__ __forceinline__ void copy_overflow(bf16* raw_k,const bf16* overflow_k,bf16* ring_k,
    bf16* ring_v,float* ring_bias,const float* current_bias,int overflow) {
  for(int i=threadIdx.x;i<overflow*128;i+=128) {
    int row=i/128,d=i%128;
    ring_k[ring_offset(row,d)]=overflow_k[row*128+d];
    ring_v[ring_offset(row,d)]=raw_k[row*128+d];
  }
  if(threadIdx.x<overflow)
    ring_bias[threadIdx.x]=current_bias[CAP-overflow+threadIdx.x];
}
__device__ __forceinline__ void pad_ring(bf16* ring_k,bf16* ring_v,
    float* ring_bias,int pending) {
  for(int i=threadIdx.x;i<64*128;i+=128) {
    int row=i/128,d=i%128;
    if(row>=pending) {
      ring_k[ring_offset(row,d)]=__float2bfloat16(0.f);
      ring_v[ring_offset(row,d)]=__float2bfloat16(0.f);
    }
  }
  if(threadIdx.x<64 && threadIdx.x>=pending)ring_bias[threadIdx.x]=CUDART_INF_F;
}

template<int CAP,bool PRODUCER>
__device__ __noinline__ void column_loop(
    const bf16* __restrict__ packed_q,const bf16* __restrict__ k,
    const bf16* __restrict__ v,const uint8_t* __restrict__ seeds,
    float* __restrict__ partial,float* __restrict__ maxima,float* __restrict__ sums,
    int head_start,int group_start,int partition_start,float scale,float merge_scale) {
  extern __shared__ __align__(128) unsigned char storage[];
  bf16* raw_k=reinterpret_cast<bf16*>(storage);
  bf16* raw_v=raw_k+80*128;
  bf16* ring_k=raw_v+80*128;
  bf16* ring_v=ring_k+64*128;
  float* ring_bias=reinterpret_cast<float*>(ring_v+64*128);
  float* current_bias=ring_bias+64;
  bf16* overflow_k=reinterpret_cast<bf16*>(current_bias+10);
  int tid=threadIdx.x,lane=tid&31,warp=tid>>5,cp=blockIdx.x&1,role=warp;
  int group=group_start+blockIdx.x/2,ah=group%5,column_pair=(group/5)%40,frame=group/(5*40);
  int head=head_start+ah,column=column_pair*2+cp,partition=partition_start+blockIdx.y;
  int begin=partition*64,end=min(begin+64,945),pending=0;
  constexpr int64_t row_stride=80*40*128;
  const bf16* query=packed_q+((int64_t(frame)*45*80+column)*40+head)*128;
  bf16* rk=ring_k;bf16* rv=ring_v;
  if constexpr(PRODUCER) {
    uint32_t weights[20];float residual[2];
    for(int kb=begin;kb<end;++kb) {
      load_raw(k,v,raw_k,raw_v,kb,head);phase_barrier();
      int g=lane>>2,base=((kb*80+column)*40+head)*CAP;
      int row0=g<CAP?seeds[base+g]:0,row1=g+8<CAP?seeds[base+g+8]:0;
      NativeSeedQuery loader{query+row0*row_stride,query+row1*row_stride};
      make_weights_loaded<CAP>(loader,raw_k,merge_scale,column,weights,residual,current_bias);
      merge_to_ring<CAP>(weights,residual,raw_k,rk,overflow_k,pending);
      phase_barrier(); // All producers have finished reading raw K.
      merge_to_ring<CAP>(weights,residual,raw_v,rv,raw_k,pending);
      if(lane<CAP && pending+lane<64)ring_bias[pending+lane]=current_bias[lane];
      phase_barrier();pending+=CAP;
      if(pending>=64) {
        phase_barrier(); // Consumers have finished reading the full ring.
        pending-=64;
        copy_overflow<CAP>(raw_k,overflow_k,ring_k,ring_v,ring_bias,current_bias,pending);
        phase_barrier();
      }
    }
    if(pending>0) {pad_ring(ring_k,ring_v,ring_bias,pending);phase_barrier();}
  } else {
    RouteState state;state.clear();int row_start=(role-1)*16;
    for(int kb=begin;kb<end;++kb) {
      load_raw(k,v,raw_k,raw_v,kb,head);phase_barrier();
      phase_barrier(); // Producer score/weight/K phase.
      phase_barrier(); // Producer overflow/V/bias phase.
      pending+=CAP;
      if(pending>=64) {
        consume_ring(query+row_start*row_stride,row_stride,min(16,45-row_start),rk,rv,ring_bias,scale,state);
        phase_barrier();pending-=64;
        copy_overflow<CAP>(raw_k,overflow_k,ring_k,ring_v,ring_bias,current_bias,pending);
        phase_barrier();
      }
    }
    if(pending>0) {
      pad_ring(ring_k,ring_v,ring_bias,pending);phase_barrier();
      consume_ring(query+row_start*row_stride,row_stride,min(16,45-row_start),rk,rv,ring_bias,scale,state);
    }
    int64_t output_group=(frame*80+column)*5+ah;
    int64_t row_offset=(output_group*PARTS+partition)*QUERY_ROWS;
    store_partial(state,partial+row_offset*128,maxima+row_offset,sums+row_offset,row_start);
  }
}

template<int CAP>
__global__ __launch_bounds__(128,2) void fused_kernel(
    const bf16* __restrict__ packed_q,const bf16* __restrict__ k,
    const bf16* __restrict__ v,const uint8_t* __restrict__ seeds,
    float* __restrict__ partial,float* __restrict__ maxima,float* __restrict__ sums,
    int head_start,int group_start,int partition_start,float scale,float merge_scale) {
  if((threadIdx.x>>5)==0)
    column_loop<CAP,true>(packed_q,k,v,seeds,partial,maxima,sums,head_start,group_start,partition_start,scale,merge_scale);
  else
    column_loop<CAP,false>(packed_q,k,v,seeds,partial,maxima,sums,head_start,group_start,partition_start,scale,merge_scale);
}

void run(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor seeds,
    torch::Tensor partial,torch::Tensor maxima,torch::Tensor sums,int capacity,
    int head_start,int group_start,int group_count,int partition_start,int partition_count,double scale) {
  TORCH_CHECK(capacity==4 || capacity==10,"Capacity must be four or ten");
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && seeds.is_cuda() && partial.is_cuda() && maxima.is_cuda() && sums.is_cuda(),"CUDA required");
  for(const auto& x:{k,v,seeds,partial,maxima,sums})TORCH_CHECK(x.device()==q.device(),"Device mismatch");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && seeds.is_contiguous() && partial.is_contiguous() && maxima.is_contiguous() && sums.is_contiguous(),"Contiguous inputs required");
  TORCH_CHECK(q.scalar_type()==at::kBFloat16 && k.scalar_type()==at::kBFloat16 && v.scalar_type()==at::kBFloat16 && seeds.scalar_type()==at::kByte,"Input dtype mismatch");
  TORCH_CHECK(partial.scalar_type()==at::kFloat && maxima.scalar_type()==at::kFloat && sums.scalar_type()==at::kFloat,"FP32 partials required");
  TORCH_CHECK(q.dim()==6 && q.size(0)==1 && q.size(1)==21 && q.size(2)==45 && q.size(3)==80 && q.size(4)==40 && q.size(5)==128 && q.sizes()==k.sizes() && q.sizes()==v.sizes(),"Native QKV shape required");
  TORCH_CHECK(seeds.numel()==945*80*40*capacity,"Seed shape mismatch");
  TORCH_CHECK(partial.numel()==int64_t(8400)*PARTS*QUERY_ROWS*128 && maxima.numel()==8400*PARTS*QUERY_ROWS && maxima.sizes()==sums.sizes(),"Partial shape mismatch");
  TORCH_CHECK(head_start>=0 && head_start<=35 && head_start%5==0,"H5 head range required");
  TORCH_CHECK(group_start>=0 && group_count>0 && group_start+group_count<=4200 && partition_start>=0 && partition_count>0 && partition_start+partition_count<=15,"Grid range mismatch");
  auto launch=[&](auto tag) {
    constexpr int cap=decltype(tag)::value;
    C10_CUDA_CHECK(cudaFuncSetAttribute(fused_kernel<cap>,cudaFuncAttributeMaxDynamicSharedMemorySize,SHARED_BYTES));
    fused_kernel<cap><<<dim3(group_count*2,partition_count),128,SHARED_BYTES,at::cuda::getCurrentCUDAStream()>>>(
      (const bf16*)q.data_ptr(),(const bf16*)k.data_ptr(),(const bf16*)v.data_ptr(),seeds.data_ptr<uint8_t>(),
      partial.data_ptr<float>(),maxima.data_ptr<float>(),sums.data_ptr<float>(),head_start,group_start,partition_start,float(scale),float(scale*1.4426950408889634));
  };
  if(capacity==4)launch(std::integral_constant<int,4>{});else launch(std::integral_constant<int,10>{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<int> resources(int capacity) {
  cudaFuncAttributes a{};int active=0;
  auto inspect=[&](auto tag) {
    constexpr int cap=decltype(tag)::value;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&a,fused_kernel<cap>));
    C10_CUDA_CHECK(cudaFuncSetAttribute(fused_kernel<cap>,cudaFuncAttributeMaxDynamicSharedMemorySize,SHARED_BYTES));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,fused_kernel<cap>,128,SHARED_BYTES));
  };
  if(capacity==4)inspect(std::integral_constant<int,4>{});else inspect(std::integral_constant<int,10>{});
  return {a.numRegs,int(a.localSizeBytes),SHARED_BYTES,active};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("run",&run);m.def("resources",&resources);}
