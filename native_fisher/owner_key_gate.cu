// Diagnostic only: exercise the candidate's 128-thread WGMMA score mapping
// and shared atomic reduction on externally supplied FP32 score bit patterns.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "owner_key.h"

__global__ void owner_gate(const int32_t* input, int32_t* scores,
                           int32_t* owners, int rotation) {
    __shared__ unsigned long long best[6*80];
    const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, seed=lane>>2;
    for (int index=tid; index<6*80; index+=128)
        best[index]=gma_owner::encode(0xff800000u,0);
    __syncthreads();
    // The same 64-row fragment mapping as the native C10 merge. Rows60-63
    // are padding and must never win. Rotate issue order to vary contention.
    for (int iteration=0; iteration<40; ++iteration) {
        const int i=(iteration+rotation)%40;
        const int row=warp*16+seed+8*((i&3)>>1);
        const int ki=2*(lane&3)+8*(i>>2)+(i&1);
        const uint32_t bits=uint32_t(input[(blockIdx.x*64+row)*80+ki]);
        if (row<60 && !gma_owner::is_nan(bits))
            atomicMax(best+(row/10)*80+ki,
                static_cast<unsigned long long>(gma_owner::encode(bits,row%10)));
    }
    __syncthreads();
    for (int index=tid; index<6*80; index+=128) {
        const uint64_t key=best[index];
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
