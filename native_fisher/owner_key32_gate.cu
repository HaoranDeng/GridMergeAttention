// Diagnostic only: exercise the candidate's 128-thread WGMMA score mapping
// and shared atomic reduction on externally supplied FP32 score bit patterns.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "owner_key.h"

__global__ void owner_gate(const int32_t* input, int32_t* scores,
                           int32_t* owners, int rotation) {
    __shared__ unsigned best_score[480], best_tie[480];
    const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, seed=lane>>2;
    for(int index=tid;index<480;index+=128) {
      best_score[index]=unsigned(gma_owner::encode(0xff800000u,0)>>32);
      best_tie[index]=0;
    }
    __syncthreads();
    for(int iteration=0;iteration<40;++iteration) {
      int i=(iteration+rotation)%40;
      int row=warp*16+seed+8*((i&3)>>1);
      int ki=2*(lane&3)+8*(i>>2)+(i&1);
      uint32_t bits=uint32_t(input[(blockIdx.x*64+row)*80+ki]);
      if(row<60 && !gma_owner::is_nan(bits))
        atomicMax(best_score+(row/10)*80+ki,unsigned(gma_owner::encode(bits,row%10)>>32));
    }
    __syncthreads();
    for(int iteration=0;iteration<40;++iteration) {
      int i=(iteration+rotation)%40;
      int row=warp*16+seed+8*((i&3)>>1);
      int ki=2*(lane&3)+8*(i>>2)+(i&1);
      uint32_t bits=uint32_t(input[(blockIdx.x*64+row)*80+ki]);
      if(row<60 && !gma_owner::is_nan(bits)) {
        int index=(row/10)*80+ki;
        uint64_t key=gma_owner::encode(bits,row%10);
        if(unsigned(key>>32)==best_score[index]) atomicMax(best_tie+index,unsigned(key));
      }
    }
    __syncthreads();
    for(int index=tid;index<480;index+=128) {
      uint64_t key=(uint64_t(best_score[index])<<32)|best_tie[index];
      if(best_score[index]==unsigned(gma_owner::encode(0xff800000u,0)>>32))
        key=gma_owner::encode(0xff800000u,0);
      scores[blockIdx.x*480+index]=int32_t(gma_owner::score_bits(key));
      owners[blockIdx.x*480+index]=int32_t(gma_owner::owner(key));
    }
}

std::vector<torch::Tensor> run(torch::Tensor input, int rotation) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type()==torch::kInt32);
    TORCH_CHECK(input.dim()==3 && input.size(1)==64 && input.size(2)==80);
    TORCH_CHECK(input.size(0)>0 && input.size(0)<=65535);
    TORCH_CHECK(rotation>=0 && rotation<40);
    auto scores=torch::empty({input.size(0),6,80},input.options());
    auto owners=torch::empty_like(scores);
    owner_gate<<<input.size(0),128,0,at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<int32_t>(),scores.data_ptr<int32_t>(),
        owners.data_ptr<int32_t>(),rotation);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {scores,owners};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("run",&run);}
