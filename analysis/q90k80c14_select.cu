// Q90K80C14 native RF0 selector. One warp per (query group, key group, head):
// lane c owns candidate c of the 32-candidate pool. Scores z (32 x 16 probes)
// come from bf16 mma.sync with fp32 accumulation, are scaled in fp32 and then
// promoted to fp64 exactly like the Triton reference; the 13-step greedy
// Fisher residual selection runs in fp64 with the same formulas (no FMA
// contraction, IEEE division/sqrt, libdevice exp).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cstdint>

using bf16=__nv_bfloat16;
constexpr int T=AXIS_OUTER,H=AXIS_HEADS,GROUPS=T*40,KG=T*45,C=14,NPROBE=16,NCAND=32;
constexpr int KS_STRIDE=136,ZS_STRIDE=17;

__device__ __forceinline__ void mma_bf16(float (&c)[4],const uint32_t (&a)[4],const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
__device__ __forceinline__ double shfl64(double v,int src) {
  return __shfl_sync(0xffffffff,v,src);
}

__global__ __launch_bounds__(128) void select_kernel(const bf16* __restrict__ q,const bf16* __restrict__ k,
 const int* __restrict__ table,uint8_t* __restrict__ seeds) {
  __shared__ __align__(16) bf16 ks[NPROBE*KS_STRIDE];
  __shared__ int cids[NCAND];
  __shared__ __align__(16) float zs[4][NCAND*ZS_STRIDE];
  __shared__ __align__(16) double bcast[4][NPROBE];
  const int kg=blockIdx.x,gb=blockIdx.y,head=blockIdx.z,row=kg%45;
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,g=gb*4+warp;
  if(tid<NCAND) cids[tid]=table[row*NCAND+tid];
  // Probe keys: 16 rows (0,5,...,75) of key group kg.
  for(int vec=tid;vec<NPROBE*16;vec+=128) {
    int p=vec/16,d=(vec%16)*8;
    uint4 value=*reinterpret_cast<const uint4*>(k+((int64_t(head)*KG+kg)*80+p*5)*128+d);
    *reinterpret_cast<uint4*>(ks+p*KS_STRIDE+d)=value;
  }
  __syncthreads();
  // z = qs . ks^T for this warp's group: 32 candidate rows x 16 probes.
  float acc[2][2][4];
  #pragma unroll
  for(int mt=0;mt<2;++mt)
    #pragma unroll
    for(int nt=0;nt<2;++nt)
      #pragma unroll
      for(int i=0;i<4;++i) acc[mt][nt][i]=0.f;
  const bf16* qbase=q+(int64_t(head)*GROUPS+g)*90*128;
  const bf16* qrow[2][2];
  #pragma unroll
  for(int mt=0;mt<2;++mt) {
    qrow[mt][0]=qbase+int64_t(cids[mt*16+(lane>>2)])*128+(lane&3)*2;
    qrow[mt][1]=qbase+int64_t(cids[mt*16+8+(lane>>2)])*128+(lane&3)*2;
  }
  #pragma unroll
  for(int kk=0;kk<8;++kk) {
    uint32_t b[2][2];
    #pragma unroll
    for(int nt=0;nt<2;++nt) {
      const bf16* kp=ks+(nt*8+(lane>>2))*KS_STRIDE+kk*16+(lane&3)*2;
      b[nt][0]=*reinterpret_cast<const uint32_t*>(kp);
      b[nt][1]=*reinterpret_cast<const uint32_t*>(kp+8);
    }
    #pragma unroll
    for(int mt=0;mt<2;++mt) {
      uint32_t a[4];
      a[0]=__ldg(reinterpret_cast<const uint32_t*>(qrow[mt][0]+kk*16));
      a[1]=__ldg(reinterpret_cast<const uint32_t*>(qrow[mt][1]+kk*16));
      a[2]=__ldg(reinterpret_cast<const uint32_t*>(qrow[mt][0]+kk*16+8));
      a[3]=__ldg(reinterpret_cast<const uint32_t*>(qrow[mt][1]+kk*16+8));
      mma_bf16(acc[mt][0],a,b[0]);
      mma_bf16(acc[mt][1],a,b[1]);
    }
  }
  // Transpose through shared memory so lane c holds z[c][0..15].
  float* wz=zs[warp];
  #pragma unroll
  for(int mt=0;mt<2;++mt)
    #pragma unroll
    for(int nt=0;nt<2;++nt) {
      int r=mt*16+(lane>>2),p=nt*8+(lane&3)*2;
      wz[r*ZS_STRIDE+p]=acc[mt][nt][0];wz[r*ZS_STRIDE+p+1]=acc[mt][nt][1];
      wz[(r+8)*ZS_STRIDE+p]=acc[mt][nt][2];wz[(r+8)*ZS_STRIDE+p+1]=acc[mt][nt][3];
    }
  __syncwarp();
  const float scale=float(0.08838834764831845);
  double z[NPROBE];
  #pragma unroll
  for(int j=0;j<NPROBE;++j) z[j]=double(__fmul_rn(wz[lane*ZS_STRIDE+j],scale));
  const int cid=cids[lane];
  // Anchor row (candidate == row) and its softmax rho over probes.
  const int anchor=__ffs(__ballot_sync(0xffffffff,cid==row))-1;
  double* wb=bcast[warp];
  double za[NPROBE];
  #pragma unroll
  for(int j=0;j<NPROBE;++j) za[j]=shfl64(z[j],anchor);
  double mx=za[0];
  #pragma unroll
  for(int j=1;j<NPROBE;++j) mx=fmax(mx,za[j]);
  double ex=0.;
  if(lane<NPROBE) ex=exp(__dsub_rn(za[lane],mx));
  double exsum=ex;
  #pragma unroll
  for(int off=8;off>0;off>>=1) exsum=__dadd_rn(exsum,__shfl_xor_sync(0xffffffff,exsum,off,16));
  double rho=__ddiv_rn(ex,exsum);
  if(lane<NPROBE) wb[lane]=rho;
  __syncwarp();
  double rhos[NPROBE],srho[NPROBE];
  #pragma unroll
  for(int j=0;j<NPROBE;++j) {rhos[j]=wb[j];srho[j]=sqrt(rhos[j]);}
  __syncwarp();
  double delta[NPROBE],mean=0.;
  #pragma unroll
  for(int j=0;j<NPROBE;++j) {delta[j]=__dsub_rn(z[j],za[j]);mean=__dadd_rn(mean,__dmul_rn(delta[j],rhos[j]));}
  double residual[NPROBE];
  #pragma unroll
  for(int j=0;j<NPROBE;++j) residual[j]=__dmul_rn(__dsub_rn(delta[j],mean),srho[j]);
  bool used=cid==row;
  uint8_t* out=seeds+(int64_t(head*40+g)*KG+kg)*C;
  if(lane==0) out[0]=uint8_t(row);
  const int second=__ffs(__ballot_sync(0xffffffff,cid==row+45))-1;
  #pragma unroll 1
  for(int slot=1;slot<C;++slot) {
    double score=0.;
    #pragma unroll
    for(int j=0;j<NPROBE;++j) score=__dadd_rn(score,__dmul_rn(residual[j],residual[j]));
    if(used) score=-CUDART_INF;
    double best=score;int bi=lane;
    #pragma unroll
    for(int off=16;off>0;off>>=1) {
      double ob=__shfl_xor_sync(0xffffffff,best,off);int oi=__shfl_xor_sync(0xffffffff,bi,off);
      if(ob>best || (ob==best && oi<bi)) {best=ob;bi=oi;}
    }
    int picked=slot==1?second:bi;
    int chosen=__shfl_sync(0xffffffff,cid,picked);
    if(lane==0) out[slot]=uint8_t(chosen);
    used=used||(lane==picked);
    if(slot<C-1) {
      double dir[NPROBE];
      #pragma unroll
      for(int j=0;j<NPROBE;++j) dir[j]=shfl64(residual[j],picked);
      double norm=0.;
      #pragma unroll
      for(int j=0;j<NPROBE;++j) norm=__dadd_rn(norm,__dmul_rn(dir[j],dir[j]));
      // Reference: direction / sqrt(sum(direction^2) + 1e-12), reduced then added.
      double den=sqrt(__dadd_rn(norm,1.e-12));
      if(lane<NPROBE) wb[lane]=__ddiv_rn(dir[lane],den);
      __syncwarp();
      double basis[NPROBE];
      #pragma unroll
      for(int j=0;j<NPROBE;++j) basis[j]=wb[j];
      __syncwarp();
      double t=0.;
      #pragma unroll
      for(int j=0;j<NPROBE;++j) t=__dadd_rn(t,__dmul_rn(residual[j],basis[j]));
      #pragma unroll
      for(int j=0;j<NPROBE;++j) residual[j]=__dsub_rn(residual[j],__dmul_rn(t,basis[j]));
    }
  }
}

void run(torch::Tensor q,torch::Tensor k,torch::Tensor table,torch::Tensor seeds) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.is_contiguous() && q.numel()==int64_t(H)*GROUPS*90*128,"Q shape");
  TORCH_CHECK(k.is_contiguous() && k.numel()==int64_t(H)*KG*80*128,"K shape");
  TORCH_CHECK(table.scalar_type()==at::kInt && table.numel()==45*NCAND,"table");
  TORCH_CHECK(seeds.scalar_type()==at::kByte && seeds.numel()==int64_t(H)*40*KG*C,"seeds");
  dim3 grid(KG,10,H);
  select_kernel<<<grid,128,0,at::cuda::getCurrentCUDAStream()>>>((const bf16*)q.data_ptr(),(const bf16*)k.data_ptr(),table.data_ptr<int>(),seeds.data_ptr<uint8_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("run",&run);}
