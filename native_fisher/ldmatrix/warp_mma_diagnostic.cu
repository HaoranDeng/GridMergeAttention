#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "warp_mma.cuh"

using namespace fisher_warp;

// Score: A[16,128] @ raw[80,128].T. Merge: A[16,80] @ raw[80,128].
// Batch entries are independent tiles, including noncontiguous query rows
// gathered by the caller. Optional packed A follows native query packing.
template<bool SCORE, bool PACKED>
__global__ void diagnostic(const bf16* a, const bf16* raw, float* out) {
  constexpr int K = SCORE ? 128 : 80, N = SCORE ? 80 : 128;
  int lane = threadIdx.x, group = lane >> 2, t = lane & 3;
  a += blockIdx.x * 16 * K;
  raw += blockIdx.x * 80 * 128;
  out += blockIdx.x * 16 * N;
  __shared__ __align__(16) bf16 shared_raw[80 * 128];
  for (int i = lane; i < 80 * 128; i += 32)
    shared_raw[raw_offset(i / 128, i % 128)] = raw[i];
  __syncwarp();
  #pragma unroll
  for (int n = 0; n < N; n += 8) {
    float c[4] = {};
    #pragma unroll
    for (int k = 0; k < K; k += 16) {
      uint32_t av[4], bv[2];
      if constexpr (PACKED) {
        // Each original 32-dimensional slice was transposed as 4x4 uint32
        // words; packed 8-element vector t owns pair 2*t in each K8 panel.
        int offset = (k / 32) * 32 + t * 8 + (k % 32) / 4;
        av[0] = *reinterpret_cast<const uint32_t*>(a + group * K + offset);
        av[1] = *reinterpret_cast<const uint32_t*>(a + (group + 8) * K + offset);
        av[2] = *reinterpret_cast<const uint32_t*>(a + group * K + offset + 2);
        av[3] = *reinterpret_cast<const uint32_t*>(a + (group + 8) * K + offset + 2);
      } else {
        av[0] = pair(a[group*K+k+2*t], a[group*K+k+2*t+1]);
        av[1] = pair(a[(group+8)*K+k+2*t], a[(group+8)*K+k+2*t+1]);
        av[2] = pair(a[group*K+k+2*t+8], a[group*K+k+2*t+9]);
        av[3] = pair(a[(group+8)*K+k+2*t+8], a[(group+8)*K+k+2*t+9]);
      }
      if constexpr (SCORE) score_b(shared_raw, n, k, lane, bv);
      else merge_b(shared_raw, k, n, lane, bv);
      mma(av, bv, c);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i)
      out[(group + 8 * (i >> 1)) * N + n + 2*t + (i & 1)] = c[i];
  }
}

torch::Tensor run(torch::Tensor a, torch::Tensor raw, bool score, bool packed) {
  TORCH_CHECK(a.is_cuda() && raw.is_cuda() && a.device()==raw.device(), "CUDA inputs required");
  TORCH_CHECK(a.scalar_type()==at::kBFloat16 && raw.scalar_type()==at::kBFloat16, "BF16 required");
  TORCH_CHECK(a.is_contiguous() && raw.is_contiguous(), "Contiguous inputs required");
  TORCH_CHECK(a.dim()==3 && raw.dim()==3 && a.size(0)==raw.size(0), "Batch shape mismatch");
  TORCH_CHECK(a.size(1)==16 && a.size(2)==(score?128:80), "A shape mismatch");
  TORCH_CHECK(raw.size(1)==80 && raw.size(2)==128, "Raw K/V shape mismatch");
  TORCH_CHECK(!packed || score, "Packed diagnostic is only defined for Q/K scores");
  auto out=torch::empty({a.size(0),16,score?80:128},a.options().dtype(at::kFloat));
  auto stream=at::cuda::getCurrentCUDAStream();
  const auto* ap=(const bf16*)a.data_ptr(); const auto* bp=(const bf16*)raw.data_ptr();
  if(score && packed) diagnostic<true,true><<<a.size(0),32,0,stream>>>(ap,bp,out.data_ptr<float>());
  else if(score) diagnostic<true,false><<<a.size(0),32,0,stream>>>(ap,bp,out.data_ptr<float>());
  else diagnostic<false,false><<<a.size(0),32,0,stream>>>(ap,bp,out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("run",&run); }
