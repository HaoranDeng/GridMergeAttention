// Fragment-ordered Q copy for the Q80K90 even-capacity merge.
// forward: q [1, N, H, 128] (token order t x 45 x 80; token-major or head-major
// storage, given by its token and head element strides) -> packed [H, N, 128],
// each 64-byte slice transposed as a 4x4 matrix of 32-bit words (the WGMMA
// register-fragment order the merge gathers). A Q80K90 query group is a row of
// 80 consecutive tokens, so only the head moves to the front; the selector,
// merge and route read the caller's q, k, v in place and the route writes a
// token-major output, so no other layout kernel exists.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cstdint>

using bf16=__nv_bfloat16;
constexpr int T=AXIS_OUTER,H=AXIS_HEADS,N=T*3600;

__device__ __forceinline__ uint4 quad_transpose(uint4 a,int lane) {
  uint32_t t0=__shfl_xor_sync(0xffffffff,(lane&1)?a.x:a.y,1);
  uint32_t t1=__shfl_xor_sync(0xffffffff,(lane&1)?a.z:a.w,1);
  uint4 b={(lane&1)?t0:a.x,(lane&1)?a.y:t0,(lane&1)?t1:a.z,(lane&1)?a.w:t1};
  t0=__shfl_xor_sync(0xffffffff,(lane&2)?b.x:b.z,2);
  t1=__shfl_xor_sync(0xffffffff,(lane&2)?b.y:b.w,2);
  return {(lane&2)?t0:b.x,(lane&2)?t1:b.y,(lane&2)?b.z:t0,(lane&2)?b.w:t1};
}

// One thread per 16-byte vector of packed; a warp covers two consecutive
// (h, n) rows, and a lane quad one 64-byte slice of one row of q. The vector
// count is a multiple of the grid stride's warp size, so every shuffle has all
// 32 lanes.
__global__ void __launch_bounds__(256) forward_kernel(const bf16* __restrict__ q,bf16* __restrict__ packed,int64_t ts,int64_t hs) {
  const int64_t vectors=int64_t(N)*H*16;
  const int lane=threadIdx.x&31;
  for(int64_t vec=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;vec<vectors;vec+=int64_t(gridDim.x)*blockDim.x) {
    const int64_t row=vec>>4;const int c=int(vec&15);
    const int hd=int(row/N);const int64_t n=row%N;
    const uint4 a=*reinterpret_cast<const uint4*>(q+n*ts+hd*hs+c*8);
    *reinterpret_cast<uint4*>(packed+vec*8)=quad_transpose(a,lane);
  }
}

constexpr int NTOKENS=N,NHEADS=H;
static void check_input(const torch::Tensor& x,int64_t ts,int64_t hs,const char* name) {
  // Dense [1, N, H, 128] BF16 in token-major (ts = H*128, hs = 128) or head-major (ts = 128, hs = N*128) order.
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kBFloat16 && x.dim()==4 && x.size(0)==1 && x.size(1)==NTOKENS && x.size(2)==NHEADS && x.size(3)==128,
              name," must be CUDA BF16 [1, N, H, 128] of the compiled shape");
  TORCH_CHECK(x.stride(3)==1 && x.stride(1)==ts && x.stride(2)==hs &&
              ((ts==int64_t(NHEADS)*128 && hs==128) || (ts==128 && hs==int64_t(NTOKENS)*128)),name," must be token-major or head-major dense");
}

void forward(torch::Tensor q,torch::Tensor packed,int64_t ts,int64_t hs) {
  check_input(q,ts,hs,"q");
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type()==at::kBFloat16 && packed.is_contiguous() && packed.numel()==q.numel(),"packed shape");
  TORCH_CHECK(packed.data_ptr()!=q.data_ptr(),"out of place");
  const int64_t vectors=int64_t(N)*H*16;
  int blocks=int(std::min<int64_t>((vectors+255)/256,132*32));
  forward_kernel<<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>((const bf16*)q.data_ptr(),(bf16*)packed.data_ptr(),ts,hs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("forward",&forward);}
