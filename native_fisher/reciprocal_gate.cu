// Exhaustive positive-FP32 reciprocal fast-path equivalence gate.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

__global__ void reciprocal_gate_kernel(unsigned long long* stats) {
  constexpr unsigned long long lo=0x33800000ULL; // 2^-24 inclusive
  constexpr unsigned long long hi=0x43000000ULL; // 2^7 exclusive
  unsigned long long mismatches=0,first=0x7fffffffffffffffULL,max_ulp=0;
  for(unsigned long long bits=lo+blockIdx.x*blockDim.x+threadIdx.x;
      bits<hi;bits+=gridDim.x*blockDim.x) {
    float x=__uint_as_float(static_cast<unsigned>(bits)),reference,candidate;
    asm volatile("div.full.f32 %0,0f3F800000,%1;":"=f"(reference):"f"(x));
    asm volatile("rcp.approx.ftz.f32 %0,%1;":"=f"(candidate):"f"(x));
    unsigned a=__float_as_uint(reference),b=__float_as_uint(candidate);
    if(a!=b) {
      ++mismatches;
      first=min(first,bits);
      max_ulp=max(max_ulp,static_cast<unsigned long long>(a>b?a-b:b-a));
    }
  }
  if(mismatches) {
    atomicAdd(stats,mismatches);
    atomicMin(stats+1,first);
    atomicMax(stats+2,max_ulp);
  }
}

void run(torch::Tensor stats) {
  TORCH_CHECK(stats.is_cuda() && stats.is_contiguous() && stats.scalar_type()==at::kLong && stats.numel()==3,
              "Require three contiguous device int64 statistics");
  reciprocal_gate_kernel<<<4096,256,0,at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<unsigned long long*>(stats.data_ptr<int64_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("run",&run);}
