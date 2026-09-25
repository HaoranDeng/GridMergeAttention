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
__device__ __forceinline__ void fisher_fma(const float (&local)[P][16/P],
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
    float delta[PR],mean=0.f;
    #pragma unroll
    for(int p=0;p<PR;++p){delta[p]=z[c][p]-anchor_z[p];mean=fmaf(delta[p],rho[p],mean);}
    #pragma unroll
    for(int width=P/2;width; width/=2)mean+=__shfl_xor_sync(0xffffffff,mean,width);
    #pragma unroll
    for(int p=0;p<PR;++p)res[c][p]=__fmul_rn(__fadd_rn(delta[p],-mean),root[p]);
  }

  float scores[P];
  #pragma unroll
  for(int c=0;c<P;++c){
    float norm=0.f;
    #pragma unroll
    for(int p=0;p<PR;++p)norm=fmaf(res[c][p],res[c][p],norm);
    #pragma unroll
    for(int width=P/2;width;width/=2)norm+=__shfl_xor_sync(0xffffffff,norm,width);
    scores[c]=norm;
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
        float projection=0.f;
        #pragma unroll
        for(int p=0;p<PR;++p)projection=fmaf(res[c][p],basis[p],projection);
        #pragma unroll
        for(int width=P/2;width;width/=2)projection+=__shfl_xor_sync(0xffffffff,projection,width);
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
__device__ __forceinline__ void fisher_fma_trimmed(const float (&local)[P][16/P],
 int cand_lane,int anchor,uint8_t* seeds,int group,float scale) {
  constexpr int CS=32/P,PR=16/P;
  const int lane=threadIdx.x&31,probe=lane%P,slot=lane/P;
  float z[P][PR];
  #pragma unroll
  for(int p=0;p<PR;++p){
    #pragma unroll
    for(int c=0;c<((NC+CS-1)/CS);++c)z[c][p]=__fmul_rn(local[c][p],scale);
  }
  float rho[PR],root[PR],anchor_z[PR];
  #pragma unroll
  for(int p=0;p<PR;++p){
    float value=0;
    #pragma unroll
    for(int c=0;c<((NC+CS-1)/CS);++c)value=choose_candidate(value,z[c][p],anchor/CS,c);
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
  for(int c=0;c<((NC+CS-1)/CS);++c){
    float delta[PR],mean=0.f;
    #pragma unroll
    for(int p=0;p<PR;++p){delta[p]=z[c][p]-anchor_z[p];mean=fmaf(delta[p],rho[p],mean);}
    #pragma unroll
    for(int width=P/2;width; width/=2)mean+=__shfl_xor_sync(0xffffffff,mean,width);
    #pragma unroll
    for(int p=0;p<PR;++p)res[c][p]=__fmul_rn(__fadd_rn(delta[p],-mean),root[p]);
  }

  float scores[P];
  #pragma unroll
  for(int c=0;c<((NC+CS-1)/CS);++c){
    float norm=0.f;
    #pragma unroll
    for(int p=0;p<PR;++p)norm=fmaf(res[c][p],res[c][p],norm);
    #pragma unroll
    for(int width=P/2;width;width/=2)norm+=__shfl_xor_sync(0xffffffff,norm,width);
    scores[c]=norm;
  }
  unsigned used=unsigned(~((uint64_t(1)<<NC)-1)) | (1u<<anchor);
  unsigned packed=__shfl_sync(0xffffffffu,cand_lane,anchor)&0xffu;
  const int own_index=probe*CS+slot;
  #pragma unroll
  for(int step=1;step<4;++step){
    float score=0;
    #pragma unroll
    for(int c=0;c<((NC+CS-1)/CS);++c)score=choose_candidate(score,scores[c],probe,c);
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
        for(int c=0;c<((NC+CS-1)/CS);++c)value=choose_candidate(value,res[c][p],picked/CS,c);
        basis[p]=__shfl_sync(0xffffffff,value,(picked%CS)*P+probe)*inv;
      }
      // Pivoted Cholesky tracks residual norms analytically. Retain the first
      // residual update because the second pivot needs its actual direction.
      float factor=2.f-best/(best+1.e-12f);
      #pragma unroll
      for(int c=0;c<((NC+CS-1)/CS);++c){
        float projection=0.f;
        #pragma unroll
        for(int p=0;p<PR;++p)projection=fmaf(res[c][p],basis[p],projection);
        #pragma unroll
        for(int width=P/2;width;width/=2)projection+=__shfl_xor_sync(0xffffffff,projection,width);
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
