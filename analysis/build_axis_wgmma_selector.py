"""Generate a WGMMA producer for the unchanged current-line Fisher selector."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'attention/axis_current_c4_selector.cu').read_text()
prefix=source[:source.index('// Padded shared layout')]
body=r'''
#include <cute/arch/mma_sm90_gmma.hpp>
using namespace cute::SM90::GMMA;
#define WG8(a) a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]
__device__ __forceinline__ int wg_sw(int x){return x^(((x>>6)&7)<<3);}
__device__ __forceinline__ int wg_qa(int r,int d){return wg_sw(r*64+(d&63)+(d>>6)*8192);}
__device__ __forceinline__ int wg_ka(int r,int d){return wg_sw(r*64+(d&63)+(d>>6)*1024);}
__device__ __forceinline__ uint64_t wg_desc(const bf16* p){
 auto s=__cvta_generic_to_shared(p);
 return ((s&0x3ffff)>>4)|(uint64_t(1)<<16)|(uint64_t(64)<<32)|(uint64_t(1)<<62);
}
constexpr int WG_SMEM=(128*128+16*128)*2+128*20*4;
__global__ __launch_bounds__(128,4) void wgmma_selector(
 const bf16* __restrict__ q,const bf16* __restrict__ k,
 const int* __restrict__ candidates,const int* __restrict__ anchors,
 uint8_t* __restrict__ seeds,int hs,float scale){
 extern __shared__ __align__(128) bf16 raw[];
 bf16* sq=raw;bf16* sk=sq+128*128;
 float* scores=reinterpret_cast<float*>(sk+16*128);
 int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
 constexpr int CG=(L+3)/4;
 int key_row=blockIdx.x%R,cg=(blockIdx.x/R)%CG;
 int head=blockIdx.x/(R*CG),qo=blockIdx.y,column=cg*4+warp;
 int anchor=anchors[key_row],cand_lane=candidates[key_row*32+lane];
 for(int vec=tid;vec<128*16;vec+=128){
  int row=vec/16,d=vec%16*8,col=cg*4+row/32;
  int qr=candidates[key_row*32+row%32];
  int qi=(((qo*R+qr)*L+col)*H+hs+head)*128+d;
  *reinterpret_cast<uint4*>(sq+wg_qa(row,d))=col<L?*reinterpret_cast<const uint4*>(q+qi):make_uint4(0,0,0,0);
 }
 int pp=tid>>4,dd=(tid&15)*8;
 int ki=((key_row*L+pp*L/16)*H+hs+head)*128+dd;
 uint4 pf0=*reinterpret_cast<const uint4*>(k+ki);
 ki=((key_row*L+(pp+8)*L/16)*H+hs+head)*128+dd;
 uint4 pf1=*reinterpret_cast<const uint4*>(k+ki);
 for(int frame=0;frame<O;++frame){
  __syncthreads();
  *reinterpret_cast<uint4*>(sk+wg_ka(pp,dd))=pf0;
  *reinterpret_cast<uint4*>(sk+wg_ka(pp+8,dd))=pf1;
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;":::"memory");
  asm volatile("wgmma.fence.sync.aligned;":::"memory");
  float a[8],b[8];
  using M=MMA_64x16x16_F32BF16BF16_SS<Major::K,Major::K>;
  #pragma unroll
  for(int d=0;d<128;d+=16){
   M::fma(wg_desc(sq+wg_qa(0,d)),wg_desc(sk+wg_ka(0,d)),WG8(a),d==0?ScaleOut::Zero:ScaleOut::One);
   M::fma(wg_desc(sq+wg_qa(64,d)),wg_desc(sk+wg_ka(0,d)),WG8(b),d==0?ScaleOut::Zero:ScaleOut::One);
  }
  asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
  #pragma unroll
  for(int i=0;i<8;++i){
   int row=warp*16+lane/4+8*((i&3)/2),p=(lane&3)*2+8*(i/4)+(i&1);
   scores[row*20+p]=a[i];scores[(row+64)*20+p]=b[i];
  }
  __syncthreads();
  if(frame<O-1){
   int ki=((((frame+1)*R+key_row)*L+pp*L/16)*H+hs+head)*128+dd;
   pf0=*reinterpret_cast<const uint4*>(k+ki);
   ki=((((frame+1)*R+key_row)*L+(pp+8)*L/16)*H+hs+head)*128+dd;
   pf1=*reinterpret_cast<const uint4*>(k+ki);
  }
  float local[4][4];
  #pragma unroll
  for(int c=0;c<4;++c){
   #pragma unroll
   for(int p=0;p<4;++p)local[c][p]=scores[(warp*32+c*8+lane/4)*20+p*4+(lane&3)];
  }
  int group=(((qo*L+column)*AH+head)*O+frame)*R+key_row;
  if(column<L)fisher_fma_trimmed<4,4>(local,cand_lane,anchor,seeds,group,scale);
 }
}
void run(torch::Tensor q,torch::Tensor k,torch::Tensor candidates,torch::Tensor anchors,
         torch::Tensor seeds,int hs,double scale){
 TORCH_CHECK(q.is_cuda() && q.is_contiguous() && k.is_contiguous(),"Contiguous CUDA tensors required");
 TORCH_CHECK(q.numel()==int64_t(O)*R*L*H*128 && k.numel()==q.numel(),"Compiled shape mismatch");
 TORCH_CHECK(q.scalar_type()==at::kBFloat16 && k.scalar_type()==at::kBFloat16,"BF16 required");
 TORCH_CHECK(seeds.numel()==int64_t(O)*L*AH*O*R*4,"Exact C4 required");
 TORCH_CHECK(hs>=0 && hs+AH<=H,"Head range invalid");
 C10_CUDA_CHECK(cudaFuncSetAttribute(wgmma_selector,cudaFuncAttributeMaxDynamicSharedMemorySize,WG_SMEM));
 wgmma_selector<<<dim3(R*((L+3)/4)*AH,O),128,WG_SMEM,at::cuda::getCurrentCUDAStream()>>>(
  (const bf16*)q.data_ptr(),(const bf16*)k.data_ptr(),candidates.data_ptr<int>(),anchors.data_ptr<int>(),
  seeds.data_ptr<uint8_t>(),hs,float(scale));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<int> resources(){
 cudaFuncAttributes a{};C10_CUDA_CHECK(cudaFuncGetAttributes(&a,wgmma_selector));
 int active=0;C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,wgmma_selector,128,WG_SMEM));
 return {a.numRegs,int(a.localSizeBytes),WG_SMEM,active};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);m.def("resources",&resources);}
'''
(ROOT/'attention/axis_current_c4_selector_wgmma.cu').write_text(prefix+body)
