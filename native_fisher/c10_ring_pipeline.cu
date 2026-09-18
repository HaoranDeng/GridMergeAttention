#include "c10_ring_consumer.cuh"
using namespace c10_ring;
constexpr int THREADS=640,PARTS=16;
constexpr int PIPE_BYTES=2*80*128*2+4*COLUMNS*RING*128*2+2*COLUMNS*RING*4;
__device__ __forceinline__ void pipeline_barrier(){asm volatile("bar.sync 0, 640;":::"memory");}

template<bool DYNAMIC>
__global__ __launch_bounds__(THREADS,1) void pipeline_kernel(const bf16* q,const bf16* k,const bf16* v,
    const uint8_t* seeds,float* partial,float* maxima,float* denominators,
    int head_start,int heads,int group_start,int part_start){
  extern __shared__ __align__(128) unsigned char storage[];
  bf16* sk=reinterpret_cast<bf16*>(storage);bf16* sv=sk+80*128;
  bf16* rings=sv+80*128;
  float* biases=reinterpret_cast<float*>(rings+4*COLUMNS*RING*128);
  int group=group_start+blockIdx.x,ah=group%heads,cg=(group/heads)%16,frame=group/(heads*16);
  int head=head_start+ah,col=cg*5,part=part_start+blockIdx.y;
  int begin=part*64,end=min(begin+64,945),tid=threadIdx.x,warp=tid>>5;
  if(tid<128){
    if constexpr(DYNAMIC)asm volatile("setmaxnreg.inc.sync.aligned.u32 128;":::"memory");
    for(int key=begin;key<=end;++key){
      if(key<end){
        int ring=(key-begin)&1;
        bf16* mk=rings+ring*2*COLUMNS*RING*128;bf16* mv=mk+COLUMNS*RING*128;
        float* bias=biases+ring*COLUMNS*RING;
        load_raw(k,v,sk,sv,key,head);
        produce(q,seeds,sk,sv,mk,mv,bias,frame,col,key,head,float(0.08838834764831845*1.4426950408889634));
      }
      pipeline_barrier();
    }
  }else{
    if constexpr(DYNAMIC)asm volatile("setmaxnreg.dec.sync.aligned.u32 88;":::"memory");
    int role=warp-4,local_col=role/3,row_base=(role%3)*16;
    RouteState state;state.clear();
    const bf16* query=q+((int64_t(frame)*45*80+col+min(local_col,4))*40+head)*128+int64_t(row_base)*80*40*128;
    for(int key=begin;key<=end;++key){
      if(key>begin&&role<15){
        int ring=(key-begin-1)&1;
        const bf16* mk=rings+ring*2*COLUMNS*RING*128+local_col*RING*128;
        const bf16* mv=mk+COLUMNS*RING*128;
        consume(query,min(16,45-row_base),mk,mv,biases+ring*COLUMNS*RING+local_col*RING,state);
      }
      pipeline_barrier();
    }
    if(role<15){
      int lane=tid&31,g=lane>>2,t=lane&3;
      int64_t base=((int64_t(frame)*80+col+local_col)*heads+ah)*PARTS*45+part*45;
      #pragma unroll
      for(int i=0;i<64;++i){
        int row=row_base+g+8*((i&3)>>1),d=8*(i/4)+2*t+(i&1);
        if(row<45)partial[(base+row)*128+d]=state.acc[i];
      }
      if(t==0){
        #pragma unroll
        for(int half=0;half<2;++half){
          int row=row_base+g+8*half;
          if(row<45){maxima[base+row]=state.maximum[half];denominators[base+row]=state.denominator[half];}
        }
      }
    }
  }
}
std::vector<int> inspect(bool dynamic){
  cudaFuncAttributes a{};int active=0;
  auto fn=dynamic?pipeline_kernel<true>:pipeline_kernel<false>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,PIPE_BYTES));
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a,fn));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,fn,THREADS,PIPE_BYTES));
  return {a.numRegs,int(a.localSizeBytes),PIPE_BYTES,active,a.maxThreadsPerBlock};
}
void run(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor seeds,
    torch::Tensor partial,torch::Tensor maxima,torch::Tensor denominator,
    int head_start,int heads,int group_start,int groups,int part_start,int parts,bool dynamic){
  TORCH_CHECK(q.is_cuda()&&q.is_contiguous()&&q.scalar_type()==at::kBFloat16&&q.numel()==int64_t(21)*45*80*40*128,"Native packed Q required");
  for(const auto& x:{k,v,seeds,partial,maxima,denominator})TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.device()==q.device(),"Same-device contiguous tensors required");
  TORCH_CHECK(k.sizes()==q.sizes()&&v.sizes()==q.sizes()&&k.scalar_type()==at::kBFloat16&&v.scalar_type()==at::kBFloat16,"KV mismatch");
  TORCH_CHECK(seeds.scalar_type()==at::kByte&&seeds.numel()==945*80*40*10,"Native seeds required");
  TORCH_CHECK(heads>=1&&heads<=5&&head_start>=0&&head_start+heads<=40,"Head range invalid");
  TORCH_CHECK(group_start>=0&&groups>0&&group_start+groups<=21*16*heads&&part_start>=0&&parts>0&&part_start+parts<=16,"Grid slice invalid");
  int64_t rows=int64_t(21)*80*heads*16*45;
  TORCH_CHECK(partial.scalar_type()==at::kFloat&&partial.numel()==rows*128&&maxima.scalar_type()==at::kFloat&&maxima.numel()==rows&&denominator.scalar_type()==at::kFloat&&denominator.numel()==rows,"Partial shape mismatch");
  auto r=inspect(dynamic);TORCH_CHECK(r[3]>=1,"No resident CTA possible");
  // Dynamic demand is128 producer registers and88 for each other thread:
  // 61440 total. Do not launch if the compiled initial CTA pool is smaller.
  TORCH_CHECK(!dynamic||r[0]==96,"Dynamic mode requires a verified96-register initial pool");
  auto fn=dynamic?pipeline_kernel<true>:pipeline_kernel<false>;
  fn<<<dim3(groups,parts),THREADS,PIPE_BYTES,at::cuda::getCurrentCUDAStream()>>>(
    (const bf16*)q.data_ptr(),(const bf16*)k.data_ptr(),(const bf16*)v.data_ptr(),seeds.data_ptr<uint8_t>(),
    partial.data_ptr<float>(),maxima.data_ptr<float>(),denominator.data_ptr<float>(),head_start,heads,group_start,part_start);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);m.def("resources",&inspect);}
