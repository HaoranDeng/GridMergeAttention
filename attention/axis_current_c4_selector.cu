// Current-line anchor-Fisher, derived from native C4 v9c.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include "../kernels/best/c4/warp_mma.cuh"
using bf16=__nv_bfloat16;
constexpr int W=AXIS_SELECTOR_WARPS;
constexpr int O=AXIS_OUTER,R=AXIS_ROWS,L=AXIS_COLS,H=AXIS_HEADS,AH=AXIS_ACTIVE_HEADS,NC=R<32?R:32;
// Preserve each unrolled candidate in a register. An ordinary C++ if chain
// can be converted into an indexed local-memory array by the compiler.
__device__ __forceinline__ float choose_candidate(float prior,float value,int index,int candidate) {
  asm("{ .reg .pred p; setp.eq.s32 p,%2,%3; selp.f32 %0,%1,%0,p; }"
      : "+f"(prior) : "f"(value),"r"(index),"r"(candidate));
  return prior;
}

template<int N>
__device__ __forceinline__ float sum_registers(float (&v)[N]) {
  #pragma unroll
  for(int d=N/2;d;d/=2){
    #pragma unroll
    for(int j=0;j<d;++j)v[j]=__fadd_rn(v[j],v[j+d]);
  }
  return v[0];
}
template<int P>
__device__ __forceinline__ float sum_probe16(float (&v)[16/P]) {
  float x=sum_registers(v);
  #pragma unroll
  for(int d=P/2;d;d/=2)x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,d));
  return x;
}
template<int P>
__device__ __forceinline__ float sum_candidate32(float (&v)[P]) {
  float x=sum_registers(v);
  #pragma unroll
  for(int d=16;d>=P;d/=2)x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,d));
  return x;
}
// One recursive-halving level. The all-reduce butterfly at level `width`
// computes x_p(l)=__fadd_rn(x_p(l),x_p(l^width)) for every tagged value p on
// every lane. Here lane l keeps only the value whose tag bit equals `mine` and
// sends the other; the partner l^width has the opposite `mine` (the tag bit is
// a lane bit that `width` flips), so the value it sends is exactly the tagged
// value this lane keeps. The returned own+partner add is the operand pair the
// butterfly adds on this lane for tag p=mine; the partner's copy of the same
// sum is partner+own, identical bits by commutativity. Nothing is rounded
// differently, only fewer values are moved.
__device__ __forceinline__ float halve_add(float tag0,float tag1,bool mine,int width) {
  float keep=mine?tag1:tag0,send=mine?tag0:tag1;
  return __fadd_rn(keep,__shfl_xor_sync(0xffffffff,send,width));
}
template<int SEEDS,int P>
__device__ __forceinline__ void fisher_from_registers(const float (&local)[P][16/P],
 int cand_lane,int anchor,uint8_t* seeds,int group,float scale) {
  constexpr int CS=32/P,PR=16/P;
  const int lane=threadIdx.x&31,probe=lane%P,slot=lane/P;
  float z[P][PR];
  #pragma unroll
  for(int p=0;p<PR;++p){
    #pragma unroll
    for(int c=0;c<P;++c)z[c][p]=__fmul_rn(local[c][p],scale);
  }
  float rho[PR],root[PR],anchor_z[PR];
  #pragma unroll
  for(int p=0;p<PR;++p){
    float value=0;
    #pragma unroll
    for(int c=0;c<P;++c)value=choose_candidate(value,z[c][p],anchor/CS,c);
    anchor_z[p]=__shfl_sync(0xffffffff,value,(anchor%CS)*P+probe);
  }
  static_assert(P==4,"Four probe lanes");
  const int subset=(lane>>3)&3;
  float argument=0.f;
  #pragma unroll
  for(int p=0;p<4;++p)argument=choose_candidate(argument,anchor_z[p],subset,p);
  int mk=__float_as_int(argument);mk^=(mk>>31)&0x7fffffff;
  mk=__reduce_max_sync(0xffffffffu,mk);mk^=(mk>>31)&0x7fffffff;
  float exponential=__expf(argument-__int_as_float(mk));
  float den=exponential;
  den+=__shfl_xor_sync(0xffffffff,den,16);
  den+=__shfl_xor_sync(0xffffffff,den,8);
  den+=__shfl_xor_sync(0xffffffff,den,2);
  den+=__shfl_xor_sync(0xffffffff,den,1);
  float normalized=__fdividef(exponential,den),owned_root=sqrtf(normalized);
  #pragma unroll
  for(int p=0;p<4;++p){rho[p]=__shfl_sync(0xffffffff,normalized,p*8+probe);root[p]=__shfl_sync(0xffffffff,owned_root,p*8+probe);}
  float res[P][PR];
  #pragma unroll
  for(int c=0;c<P;++c){
    float delta[PR],terms[PR];
    #pragma unroll
    for(int p=0;p<PR;++p){delta[p]=__fadd_rn(z[c][p],-anchor_z[p]);terms[p]=__fmul_rn(delta[p],rho[p]);}
    float mean=sum_probe16<P>(terms);
    #pragma unroll
    for(int p=0;p<PR;++p)res[c][p]=__fmul_rn(__fadd_rn(delta[p],-mean),root[p]);
  }
  unsigned used=unsigned(~((uint64_t(1)<<NC)-1)) | (1u<<anchor);
  // Packed seed store: the four bytes seeds[group*4+0..3] (little-endian, byte
  // 0 = anchor) are assembled in a register and written once per frame with
  // one aligned 32-bit store instead of four dependent byte stores.
  static_assert(SEEDS==4,"packed seed store assumes four seeds");
  unsigned packed=__shfl_sync(0xffffffffu,cand_lane,anchor)&0xffu;
  // Halving argmax (P==4): after the two halving levels lane (probe,slot) holds
  // the score of candidate c=probe, i.e. candidate index probe*CS+slot. The map
  // lane -> probe*CS+slot is a bijection onto 0..31 (transpose of the 4x8
  // grid), so the 32 lanes cover the 32 candidates exactly once.
  const bool t1=((lane>>1)&1)!=0,t0=(lane&1)!=0;
  const int own_index=probe*CS+slot;
  #pragma unroll
  for(int step=1;step<SEEDS;++step){
    float best;int picked;
    if constexpr(P==4) {
      float s[4];
      #pragma unroll
      for(int c=0;c<4;++c){
        float terms[PR];
        #pragma unroll
        for(int p=0;p<PR;++p)terms[p]=__fmul_rn(res[c][p],res[c][p]);
        s[c]=sum_registers(terms);   // (t0+t2)+(t1+t3): v8's in-lane tree
      }
      // xor 2 (probe bit 1): keep c with bit1(c)==t1. x_c=s_c(l)+s_c(l^2).
      float x_even=halve_add(s[0],s[2],t1,2);
      float x_odd=halve_add(s[1],s[3],t1,2);
      // xor 1 (probe bit 0): keep c with bit0(c)==t0, so c==probe.
      // score=x_c(l)+x_c(l^1) = ((s_l+s_l^2)+(s_l^1+s_l^3)), v8's sum_probe16.
      float score=halve_add(x_even,x_odd,t0,1);
      // Scores are sums of squares: >= +0.0 and never -0.0, so for non-NaN
      // scores the uint32 bit order equals the float order. A used candidate
      // maps to key 0 and is excluded from the index minimum, exactly as v8's
      // (picked==32, best==-inf) lanes. score==score excludes NaN scores, which
      // v8's local compares (score>best, score==best) never pick either.
      bool valid=((((used>>own_index)&1u)==0u)&(score==score));
      unsigned key=valid?__float_as_uint(score):0u;
      unsigned best_bits=__reduce_max_sync(0xffffffffu,key);
      picked=(int)__reduce_min_sync(0xffffffffu,(key==best_bits && valid)?(unsigned)own_index:32u);
      best=__uint_as_float(best_bits);
    } else {
      best=-CUDART_INF_F;picked=32;
      #pragma unroll
      for(int c=0;c<P;++c){
        float terms[PR];
        #pragma unroll
        for(int p=0;p<PR;++p)terms[p]=__fmul_rn(res[c][p],res[c][p]);
        float score=sum_probe16<P>(terms);
        int index=c*CS+slot;
        bool ok=((used>>index)&1u)==0u;
        bool better=ok&((score>best)|((score==best)&(index<picked)));
        best=better?score:best;
        picked=better?index:picked;
      }
      unsigned key=(picked==32)?0u:__float_as_uint(best);
      unsigned best_bits=__reduce_max_sync(0xffffffffu,key);
      picked=(int)__reduce_min_sync(0xffffffffu,(key==best_bits && picked!=32)?(unsigned)picked:32u);
      best=__uint_as_float(best_bits);
    }
    used|=1u<<picked;
    packed|=(__shfl_sync(0xffffffffu,cand_lane,picked)&0xffu)<<(8*step);
    if(step<SEEDS-1){
      float norm=sqrtf(__fadd_rn(best,1.e-12f)),basis[PR];
      #pragma unroll
      for(int p=0;p<PR;++p){
        float value=0;
        #pragma unroll
        for(int c=0;c<P;++c)value=choose_candidate(value,res[c][p],picked/CS,c);
        value=__shfl_sync(0xffffffff,value,(picked%CS)*P+probe);
        basis[p]=value;
      }
      if constexpr(P==4) {
        // Subset on lane bits 4,3 (see the center block); gather from p*8+probe.
        int subset=(lane>>3)&3;
        float value=0.0f;
        #pragma unroll
        for(int p=0;p<4;++p)value=choose_candidate(value,basis[p],subset,p);
        float owned_basis=__fdividef(value,norm);
        #pragma unroll
        for(int p=0;p<4;++p)basis[p]=__shfl_sync(0xffffffff,owned_basis,p*8+probe);
      } else {
        #pragma unroll
        for(int p=0;p<PR;++p)basis[p]=__fdiv_rn(basis[p],norm);
      }
      #pragma unroll
      for(int c=0;c<P;++c){
        float terms[PR];
        #pragma unroll
        for(int p=0;p<PR;++p)terms[p]=__fmul_rn(res[c][p],basis[p]);
        float projection=sum_probe16<P>(terms);
        #pragma unroll
        for(int p=0;p<PR;++p)res[c][p]=__fmaf_rn(-projection,basis[p],res[c][p]);
      }
    }
  }
  if(lane==0)*reinterpret_cast<uint32_t*>(seeds+int64_t(group)*SEEDS)=packed;
}
template<int SEEDS,int P>
__device__ __forceinline__ void fisher_cholesky(const float (&local)[P][16/P],
 int cand_lane,int anchor,uint8_t* seeds,int group,float scale) {
  constexpr int CS=32/P,PR=16/P;
  const int lane=threadIdx.x&31,probe=lane%P,slot=lane/P;
  float z[P][PR];
  #pragma unroll
  for(int p=0;p<PR;++p){
    #pragma unroll
    for(int c=0;c<P;++c)z[c][p]=__fmul_rn(local[c][p],scale);
  }
  float rho[PR],root[PR],anchor_z[PR];
  #pragma unroll
  for(int p=0;p<PR;++p){
    float value=0;
    #pragma unroll
    for(int c=0;c<P;++c)value=choose_candidate(value,z[c][p],anchor/CS,c);
    anchor_z[p]=__shfl_sync(0xffffffff,value,(anchor%CS)*P+probe);
  }
  static_assert(P==4,"Four probe lanes");
  const int subset=(lane>>3)&3;
  float argument=0.f;
  #pragma unroll
  for(int p=0;p<4;++p)argument=choose_candidate(argument,anchor_z[p],subset,p);
  int mk=__float_as_int(argument);mk^=(mk>>31)&0x7fffffff;
  mk=__reduce_max_sync(0xffffffffu,mk);mk^=(mk>>31)&0x7fffffff;
  float exponential=__expf(argument-__int_as_float(mk));
  float den=exponential;
  den+=__shfl_xor_sync(0xffffffff,den,16);
  den+=__shfl_xor_sync(0xffffffff,den,8);
  den+=__shfl_xor_sync(0xffffffff,den,2);
  den+=__shfl_xor_sync(0xffffffff,den,1);
  float normalized=__fdividef(exponential,den),owned_root=sqrtf(normalized);
  #pragma unroll
  for(int p=0;p<4;++p){rho[p]=__shfl_sync(0xffffffff,normalized,p*8+probe);root[p]=__shfl_sync(0xffffffff,owned_root,p*8+probe);}
  float res[P][PR];
  #pragma unroll
  for(int c=0;c<P;++c){
    float delta[PR],terms[PR];
    #pragma unroll
    for(int p=0;p<PR;++p){delta[p]=__fadd_rn(z[c][p],-anchor_z[p]);terms[p]=__fmul_rn(delta[p],rho[p]);}
    float mean=sum_probe16<P>(terms);
    #pragma unroll
    for(int p=0;p<PR;++p)res[c][p]=__fmul_rn(__fadd_rn(delta[p],-mean),root[p]);
  }

  float scores[P];
  #pragma unroll
  for(int c=0;c<P;++c){
    float terms[PR];
    #pragma unroll
    for(int p=0;p<PR;++p)terms[p]=res[c][p]*res[c][p];
    scores[c]=sum_probe16<P>(terms);
  }
  unsigned used=unsigned(~((uint64_t(1)<<NC)-1)) | (1u<<anchor);
  unsigned packed=__shfl_sync(0xffffffffu,cand_lane,anchor)&0xffu;
  const int own_index=probe*CS+slot;
  #pragma unroll
  for(int step=1;step<4;++step){
    float score=0;
    #pragma unroll
    for(int c=0;c<P;++c)score=choose_candidate(score,scores[c],probe,c);
    bool valid=((used>>own_index)&1u)==0u;
    unsigned key=valid?__float_as_uint(fmaxf(score,0.f)):0u;
    unsigned bestbits=__reduce_max_sync(0xffffffffu,key);
    int picked=__reduce_min_sync(0xffffffffu,(valid && key==bestbits)?unsigned(own_index):32u);
    float best=__uint_as_float(bestbits);
    used|=1u<<picked;
    packed|=(__shfl_sync(0xffffffffu,cand_lane,picked)&0xffu)<<(8*step);
    if(step<3){
      float basis[PR],inv=rsqrtf(best+1.e-12f);
      #pragma unroll
      for(int p=0;p<PR;++p){
        float value=0;
        #pragma unroll
        for(int c=0;c<P;++c)value=choose_candidate(value,res[c][p],picked/CS,c);
        basis[p]=__shfl_sync(0xffffffff,value,(picked%CS)*P+probe)*inv;
      }
      // Pivoted Cholesky tracks residual norms analytically. Retain the first
      // residual update because the second pivot needs its actual direction.
      float factor=2.f-best/(best+1.e-12f);
      #pragma unroll
      for(int c=0;c<P;++c){
        float terms[PR];
        #pragma unroll
        for(int p=0;p<PR;++p)terms[p]=res[c][p]*basis[p];
        float projection=sum_probe16<P>(terms);
        scores[c]=fmaxf(fmaf(-projection*projection,factor,scores[c]),0.f);
        if(step==1){
          #pragma unroll
          for(int p=0;p<PR;++p)res[c][p]=fmaf(-projection,basis[p],res[c][p]);
        }
      }
    }
  }
  if(lane==0)*reinterpret_cast<uint32_t*>(seeds+int64_t(group)*4)=packed;
}
template<int SEEDS,int P>
__device__ __forceinline__ void fisher_gram(const float (&local)[P][16/P],
 int cand_lane,int anchor,uint8_t* seeds,int group,float scale,float* gram) {
  constexpr int CS=32/P,PR=16/P;
  const int lane=threadIdx.x&31,probe=lane%P,slot=lane/P;
  float z[P][PR];
  #pragma unroll
  for(int p=0;p<PR;++p){
    #pragma unroll
    for(int c=0;c<P;++c)z[c][p]=__fmul_rn(local[c][p],scale);
  }
  float rho[PR],root[PR],anchor_z[PR];
  #pragma unroll
  for(int p=0;p<PR;++p){
    float value=0;
    #pragma unroll
    for(int c=0;c<P;++c)value=choose_candidate(value,z[c][p],anchor/CS,c);
    anchor_z[p]=__shfl_sync(0xffffffff,value,(anchor%CS)*P+probe);
  }
  static_assert(P==4,"Four probe lanes");
  const int subset=(lane>>3)&3;
  float argument=0.f;
  #pragma unroll
  for(int p=0;p<4;++p)argument=choose_candidate(argument,anchor_z[p],subset,p);
  int mk=__float_as_int(argument);mk^=(mk>>31)&0x7fffffff;
  mk=__reduce_max_sync(0xffffffffu,mk);mk^=(mk>>31)&0x7fffffff;
  float exponential=__expf(argument-__int_as_float(mk));
  float den=exponential;
  den+=__shfl_xor_sync(0xffffffff,den,16);
  den+=__shfl_xor_sync(0xffffffff,den,8);
  den+=__shfl_xor_sync(0xffffffff,den,2);
  den+=__shfl_xor_sync(0xffffffff,den,1);
  float normalized=__fdividef(exponential,den),owned_root=sqrtf(normalized);
  #pragma unroll
  for(int p=0;p<4;++p){rho[p]=__shfl_sync(0xffffffff,normalized,p*8+probe);root[p]=__shfl_sync(0xffffffff,owned_root,p*8+probe);}
  float res[P][PR];
  #pragma unroll
  for(int c=0;c<P;++c){
    float delta[PR],terms[PR];
    #pragma unroll
    for(int p=0;p<PR;++p){delta[p]=__fadd_rn(z[c][p],-anchor_z[p]);terms[p]=__fmul_rn(delta[p],rho[p]);}
    float mean=sum_probe16<P>(terms);
    #pragma unroll
    for(int p=0;p<PR;++p)res[c][p]=__fmul_rn(__fadd_rn(delta[p],-mean),root[p]);
  }

  // FP16 residuals, FP32 Gram accumulation. The input QK products, Fisher
  // weighting, and pivot decisions remain in FP32. Validate near-tied pivots.
  __half* residual=reinterpret_cast<__half*>(gram+32*33);
  #pragma unroll
  for(int c=0;c<4;++c){
    #pragma unroll
    for(int p=0;p<4;++p)residual[(c*8+lane/4)*24+p*4+(lane&3)]=__float2half_rn(res[c][p]);
  }
  __syncwarp();
  #pragma unroll
  for(int m=0;m<2;++m){
    int row=m*16+lane/4,pc=2*(lane&3);
    uint32_t a[4]={*reinterpret_cast<uint32_t*>(residual+row*24+pc),
                    *reinterpret_cast<uint32_t*>(residual+(row+8)*24+pc),
                    *reinterpret_cast<uint32_t*>(residual+row*24+pc+8),
                    *reinterpret_cast<uint32_t*>(residual+(row+8)*24+pc+8)};
    #pragma unroll
    for(int n=0;n<4;++n){
      const __half* bp=residual+(n*8+lane/4)*24+pc;
      uint32_t b[2]={*reinterpret_cast<const uint32_t*>(bp),*reinterpret_cast<const uint32_t*>(bp+8)};
      float acc[4]={};
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
       "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
       : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3])
       : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
      gram[row*33+n*8+pc]=acc[0];gram[row*33+n*8+pc+1]=acc[1];
      gram[(row+8)*33+n*8+pc]=acc[2];gram[(row+8)*33+n*8+pc+1]=acc[3];
    }
  }
  __syncwarp();
  float score=gram[lane*33+lane],coefficient=0;
  unsigned used=unsigned(~((uint64_t(1)<<NC)-1)) | (1u<<anchor);
  unsigned packed=__shfl_sync(0xffffffff,cand_lane,anchor)&0xffu;
  #pragma unroll
  for(int step=1;step<4;++step){
    bool valid=((used>>lane)&1u)==0u;
    unsigned key=valid?__float_as_uint(fmaxf(score,0.f)):0u;
    unsigned bestbits=__reduce_max_sync(0xffffffff,key);
    int picked=__reduce_min_sync(0xffffffff,(valid && key==bestbits)?unsigned(lane):32u);
    float best=__uint_as_float(bestbits);
    used|=1u<<picked;packed|=(__shfl_sync(0xffffffff,cand_lane,picked)&0xffu)<<(8*step);
    if(step<3){
      float inner=gram[lane*33+picked];
      if(step==2)inner=fmaf(-coefficient,__shfl_sync(0xffffffff,coefficient,picked),inner);
      float projection=inner*rsqrtf(best+1.e-12f);
      score=fmaxf(fmaf(-projection,projection,score),0.f);
      if(step==1)coefficient=projection;
    }
  }
  if(lane==0)*reinterpret_cast<uint32_t*>(seeds+int64_t(group)*4)=packed;
  __syncwarp();
}
// Padded shared layout instead of the XOR swizzle: row stride 136 bf16 (272 B =
// 17*16 B). Every 8-element chunk stays 16-byte aligned, and the eight row
// addresses of one 8x8 ldmatrix tile land in eight distinct 16-byte bank groups
// (272*r mod 128 = 16*r), so both the STS.128 fills and the ldmatrix reads are
// conflict-free while every fragment address becomes [lane_base + immediate].
__device__ __forceinline__ int qoff(int row,int d) {return row*136+d;}
// ldmatrix x4: lane i supplies the 16-byte row address of row (i&7) of matrix
// (i>>3). Matrix j lands in r[j], where lane l receives the 32-bit word at
// byte offset 4*(l&3) of row (l>>2) - the same (elem d+2t, elem d+2t+1) word a
// 32-bit shared load of that row/column pair returns.
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4],uint32_t addr) {
 asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
  : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(addr) : "memory");
}
__global__ __launch_bounds__(W*32,W==4?4:2) void fused_selector(const bf16* __restrict__ q,
 const bf16* __restrict__ k,const int* __restrict__ candidates,const int* __restrict__ anchors,
 uint8_t* __restrict__ seeds,int HS,
 double scale,float* debug_logits) {
 extern __shared__ __align__(128) bf16 raw[];
 bf16* sq=raw;bf16* sk=sq+W*32*136;
 int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
 int group1=blockIdx.x,key_row=group1%R,qo=blockIdx.y;
 constexpr int CG=(L+W-1)/W;
 int cg=(group1/R)%CG,head=group1/(R*CG),column=cg*W+warp;
 const int anchor=anchors[key_row];
 const int cand_lane=candidates[key_row*32+lane];
 for(int vec=tid;vec<W*32*16;vec+=W*32) {
  int row=vec/16,d=(vec%16)*8,candidate=row%32,col=cg*W+row/32;
  int qr=candidates[key_row*32+candidate];
  int qi=(((qo*R+qr)*L+col)*H+HS+head)*128+d;
  *reinterpret_cast<uint4*>(sq+qoff(row,d))=col<L?*reinterpret_cast<const uint4*>(q+qi):make_uint4(0,0,0,0);
 }
 // Stay one frame ahead: each thread owns probes p and p+8, so the whole
 // 16x128 probe tile lives in two uint4 registers. The load for frame+1 is
 // issued before this frame's dots and Gram-Schmidt and only reaches shared at
 // the top of the next frame, so its latency hides behind work.
 int pf_probe=tid>>4,pf_d=(tid&15)*8;
 uint4 pf0,pf1;
 {
  int ki=(((key_row*L+pf_probe*L/16)*H+HS+head)*128)+pf_d;
  pf0=*reinterpret_cast<const uint4*>(k+ki);
  ki=(((key_row*L+(pf_probe+8)*L/16)*H+HS+head)*128)+pf_d;
  if constexpr(W==4)pf1=*reinterpret_cast<const uint4*>(k+ki);
 }
 #pragma unroll 1
 for(int frame=0;frame<O;++frame) {
 __syncthreads();
 *reinterpret_cast<uint4*>(sk+qoff(pf_probe,pf_d))=pf0;
 if constexpr(W==4)*reinterpret_cast<uint4*>(sk+qoff(pf_probe+8,pf_d))=pf1;
 __syncthreads();
 if(frame<O-1) {
  int ki=((((frame+1)*R+key_row)*L+pf_probe*L/16)*H+HS+head)*128+pf_d;
  pf0=*reinterpret_cast<const uint4*>(k+ki);
  ki=((((frame+1)*R+key_row)*L+(pf_probe+8)*L/16)*H+HS+head)*128+pf_d;
  if constexpr(W==4)pf1=*reinterpret_cast<const uint4*>(k+ki);
 }
 float acc[2][2][4]={};
 // ldmatrix row addresses in bytes. The lane-dependent part is fixed per
 // thread; m contributes 16 rows (16*272 B) and d contributes 2*d B, both
 // compile-time immediates once the d/m loops unroll.
 // A (Q rows warp*32+m*16..): lanes 0-7 rows +0..7 x cols d..d+7 -> a[0];
 //   lanes 8-15 rows +8..15 x d -> a[1]; lanes 16-23 rows +0..7 x d+8 -> a[2];
 //   lanes 24-31 rows +8..15 x d+8 -> a[3]. (lane&15) is the row, lane>>4 the
 //   8-column half.
 // B (K probes): lanes 0-7 keys 0-7 x d -> b[0][0]; lanes 8-15 keys 0-7 x d+8
 //   -> b[0][1]; lanes 16-23 keys 8-15 x d -> b[1][0]; lanes 24-31 keys 8-15 x
 //   d+8 -> b[1][1].
 const uint32_t sq_a=uint32_t(__cvta_generic_to_shared(sq))
  +uint32_t(warp*32*272+(lane&15)*272+16*(lane>>4));
 const uint32_t sk_b=uint32_t(__cvta_generic_to_shared(sk))
  +uint32_t(((lane&7)+8*(lane>>4))*272+16*((lane>>3)&1));
 #pragma unroll
 for(int d=0;d<128;d+=16) {
  uint32_t bq[4];
  ldsm_x4(bq,sk_b+2*d);
  const uint32_t b[2][2]={{bq[0],bq[1]},{bq[2],bq[3]}};
  #pragma unroll
  for(int m=0;m<2;++m) {
   uint32_t a[4];
   ldsm_x4(a,sq_a+m*16*272+2*d);
   #pragma unroll
   for(int n=0;n<2;++n)fisher_warp::mma(a,b[n],acc[m][n]);
  }
 }
 // Change warp distribution from pairs of adjacent probes to four probe lanes.
 // local[c][p] on lane l is acc[c/2][p/2][2*(c&1)+bit0(l)] of source lane
 // src(l,p)=(l&~3)+2*(p&1)+bit1(l). A source s is read only in the round with
 // p&1==bit1(s), by two destinations needing columns +0 and +1, so rounds
 // p=2q and p=2q+1 pair into two shuffles: round A exposes column bit1(s) and
 // serves p=2q+bit0(l); round B exposes column 1-bit1(s) and serves
 // p=2q+1-bit0(l). The same values move, with 16 SHFL.IDX instead of 32.
 float local[4][4];
 const bool odd=(lane&1)!=0,two=(lane&2)!=0;
 const int srcA=(lane&~3)+2*(lane&1)+((lane&3)>>1);
 const int srcB=(lane&~3)+2*(1-(lane&1))+((lane&3)>>1);
 #pragma unroll
 for(int c=0;c<4;++c) {
  #pragma unroll
  for(int q=0;q<2;++q) {
   float lo=acc[c/2][q][2*(c&1)],hi=acc[c/2][q][2*(c&1)+1];
   float va=__shfl_sync(0xffffffff,two?hi:lo,srcA);
   float vb=__shfl_sync(0xffffffff,two?lo:hi,srcB);
   local[c][2*q]=odd?vb:va;
   local[c][2*q+1]=odd?va:vb;
  }
 }
 int group=(((qo*L+column)*AH+head)*O+frame)*R+key_row;
 if(debug_logits) {
  #pragma unroll
  for(int c=0;c<4;++c) {
   #pragma unroll
   for(int p=0;p<4;++p)debug_logits[(int64_t(group)*32+c*8+lane/4)*16+p*4+(lane&3)]=local[c][p];
  }
 }
 if(column<L){
    #if AXIS_CHOLESKY == 2
    fisher_gram<4,4>(local,cand_lane,anchor,seeds,group,float(scale),
      reinterpret_cast<float*>(raw+(W*32+16)*136)+warp*1440);
    #elif AXIS_CHOLESKY == 1
    fisher_cholesky<4,4>(local,cand_lane,anchor,seeds,group,float(scale));
    #else
    fisher_from_registers<4,4>(local,cand_lane,anchor,seeds,group,float(scale));
    #endif
  }
 }
}
void run(torch::Tensor q,torch::Tensor k,torch::Tensor candidates,torch::Tensor anchors,
         torch::Tensor seeds,int hs,double scale) {
 TORCH_CHECK(q.is_cuda() && q.is_contiguous() && k.is_contiguous(),"Contiguous CUDA inputs required");
 TORCH_CHECK(q.numel()==int64_t(O)*R*L*H*128 && k.numel()==q.numel(),"Compiled shape mismatch");
 TORCH_CHECK(q.scalar_type()==at::kBFloat16 && k.scalar_type()==at::kBFloat16,"BF16 required");
 TORCH_CHECK(candidates.numel()==R*32 && candidates.scalar_type()==at::kInt,"Padded candidate table required");
 TORCH_CHECK(seeds.numel()==int64_t(O)*L*AH*O*R*4 && seeds.scalar_type()==at::kByte,"Exact C4 seeds required");
 TORCH_CHECK(hs>=0 && hs+AH<=H,"Head range invalid");
 constexpr int smem=(W*32+16)*136*2+(AXIS_CHOLESKY==2?W*5760:0);
 C10_CUDA_CHECK(cudaFuncSetAttribute(fused_selector,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
 fused_selector<<<dim3(R*((L+W-1)/W)*AH,O),W*32,smem,at::cuda::getCurrentCUDAStream()>>>(
  (const bf16*)q.data_ptr(),(const bf16*)k.data_ptr(),candidates.data_ptr<int>(),anchors.data_ptr<int>(),
  seeds.data_ptr<uint8_t>(),hs,scale,nullptr);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<int> resources() {
 cudaFuncAttributes attr{};C10_CUDA_CHECK(cudaFuncGetAttributes(&attr,fused_selector));
 int active=0;C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,fused_selector,W*32,(W*32+16)*136*2+(AXIS_CHOLESKY==2?W*5760:0)));
 return {attr.numRegs,int(attr.localSizeBytes),int((W*32+16)*136*2+(AXIS_CHOLESKY==2?W*5760:0)),active};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);m.def("resources",&resources);}
