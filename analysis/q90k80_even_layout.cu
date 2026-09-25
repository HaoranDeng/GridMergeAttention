// Fused layout kernels for the Q90K80 even-capacity operator.
// forward: q, k, v [1, N, H, 128] (token order t x 45 x 80) ->
//   qp     [H, T, 40, 2, 45, 128]   (query groups of 90 = two W columns x 45 rows)
//   packed same shape, each 64-byte slice transposed as a 4x4 matrix of
//          32-bit words (the WGMMA register-fragment order the merge expects)
//   kp, vp [H, T*45, 80, 128]
// backward: out [H, T, 40, 2, 45, 128] -> result [1, N, H, 128].
// Every 16-byte vector is moved once; source and destination rows are both
// 256 bytes, so reads and writes coalesce.
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

// One thread per 16-byte vector of q; a warp covers two consecutive (n, h) rows.
__global__ void __launch_bounds__(256) forward_kernel(const bf16* __restrict__ q,const bf16* __restrict__ k,const bf16* __restrict__ v,
 bf16* __restrict__ qp,bf16* __restrict__ packed,bf16* __restrict__ kp,bf16* __restrict__ vp) {
  const int64_t vectors=int64_t(N)*H*16;
  const int lane=threadIdx.x&31;
  for(int64_t vec=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;vec<vectors;vec+=int64_t(gridDim.x)*blockDim.x) {
    const int64_t row=vec>>4;const int c=int(vec&15);
    const int hd=int(row%H);const int64_t n=row/H;
    const int t=int(n/3600),rem=int(n%3600),hh=rem/80,w=rem%80;
    const uint4 a=*reinterpret_cast<const uint4*>(q+vec*8);
    const int64_t qrow=(((int64_t(hd)*T+t)*40+(w>>1))*2+(w&1))*45+hh;
    *reinterpret_cast<uint4*>(qp+qrow*128+c*8)=a;
    const uint4 b=quad_transpose(a,lane);
    *reinterpret_cast<uint4*>(packed+qrow*128+c*8)=b;
    const int64_t krow=int64_t(hd)*N+n; // [H][T*45][80] flattened = head-major token order
    *reinterpret_cast<uint4*>(kp+krow*128+c*8)=*reinterpret_cast<const uint4*>(k+vec*8);
    *reinterpret_cast<uint4*>(vp+krow*128+c*8)=*reinterpret_cast<const uint4*>(v+vec*8);
  }
}

__global__ void __launch_bounds__(256) backward_kernel(const bf16* __restrict__ out,bf16* __restrict__ result) {
  const int64_t vectors=int64_t(N)*H*16;
  for(int64_t vec=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;vec<vectors;vec+=int64_t(gridDim.x)*blockDim.x) {
    const int64_t row=vec>>4;const int c=int(vec&15);
    const int hd=int(row%H);const int64_t n=row/H;
    const int t=int(n/3600),rem=int(n%3600),hh=rem/80,w=rem%80;
    const int64_t orow=(((int64_t(hd)*T+t)*40+(w>>1))*2+(w&1))*45+hh;
    *reinterpret_cast<uint4*>(result+vec*8)=*reinterpret_cast<const uint4*>(out+orow*128+c*8);
  }
}

static void check(torch::Tensor x,const char* name) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kBFloat16 && x.is_contiguous() && x.numel()==int64_t(N)*H*128,name," must be contiguous CUDA BF16 of the compiled shape");
}
void forward(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor qp,torch::Tensor packed,torch::Tensor kp,torch::Tensor vp) {
  check(q,"q");check(k,"k");check(v,"v");check(qp,"qp");check(packed,"packed");check(kp,"kp");check(vp,"vp");
  const int64_t vectors=int64_t(N)*H*16;
  int blocks=int(std::min<int64_t>((vectors+255)/256,132*32));
  forward_kernel<<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>((const bf16*)q.data_ptr(),(const bf16*)k.data_ptr(),(const bf16*)v.data_ptr(),
    (bf16*)qp.data_ptr(),(bf16*)packed.data_ptr(),(bf16*)kp.data_ptr(),(bf16*)vp.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void backward(torch::Tensor out,torch::Tensor result) {
  check(out,"out");check(result,"result");
  const int64_t vectors=int64_t(N)*H*16;
  int blocks=int(std::min<int64_t>((vectors+255)/256,132*32));
  backward_kernel<<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>((const bf16*)out.data_ptr(),(bf16*)result.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("forward",&forward);m.def("backward",&backward);}
