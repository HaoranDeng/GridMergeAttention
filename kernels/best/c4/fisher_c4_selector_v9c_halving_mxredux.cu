// v9c_halving_mxredux: v9c_halving with mx from one REDUX.MAX.S32 over order-preserving int keys instead of the 4-level fmaxf butterfly (-4 SHFL.BFLY, -4 FMNMX, +1 REDUX, +4 integer ops).
// v9c_halving: v8_ldsm_packed with recursive-HALVING reductions where each lane
// only consumes one of the reduced values. Edits are confined to
// fisher_from_registers and the helper halve_add above it; the kernel body,
// shared layout, ldmatrix feed, prefetch pipeline and packed store are v8's.
//   (1) Seed-step candidate scores: v8 all-reduces all four candidate scores over
//       the 4 probe lanes (8 SHFL.BFLY per step, every lane ends with all four
//       scores, then a branchless 4-way local argmax). v9c halves: xor 2 keeps
//       the two candidates whose bit1 equals bit1(probe), xor 1 keeps the one
//       whose bit0 equals bit0(probe), so lane (probe,slot) ends with exactly the
//       score of candidate index probe*CS+slot. 3 SHFL.BFLY per step, the local
//       argmax collapses to one predicate, the REDUX cross-lane argmax is v8's.
//   (2) Probe centers: v8 all-reduces four probe sums over the 8 slot lanes
//       (xor 16, 8, 4: 12 SHFL.BFLY) although the P==4 branch consumes only
//       center[subset]. v9c moves the probe subset from lane bits 3,2 to lane
//       bits 4,3 so the halving can split on the first two exchange bits:
//       xor 16 keeps 2 of 4 (2 SHFL), xor 8 keeps 1 (1 SHFL), xor 4 is a full
//       add of the survivor (1 SHFL) = 4 SHFL.BFLY. mx becomes a 4-level fmaxf
//       butterfly over the 16 distinct centers (bits 4,3,1,0) instead of v8's
//       in-lane max plus xor 2/1. den's butterfly moves from xor 8,4,2,1 to
//       xor 16,8,2,1 (same operand pairs); rho/root/basis gathers read lane
//       p*8+probe instead of p*4+probe.
// Every FP add is the same __fadd_rn(own,partner) pair the v8 butterfly performs
// on the lane that owns the survivor (partner lanes compute partner+own, the
// same bits by commutativity); only data movement, predicate logic and the
// shuffle schedule change. Seeds are byte-identical to v8.
// v8_ldsm_packed: v4_ldsm plus a packed 32-bit seed store (one STG per frame instead of four byte stores).
// Combined candidate: FP32 Gram-Schmidt plus the register-pipelined frame loop.
// v4_ldsm: 3-way merge (git merge-file, 0 conflicts) of v3fix_abd (sha 5ab7d10c: REDUX argmax, hoisted anchor/cand_lane, 16-shuffle transpose) and ldsm (sha 8acd7056: padded 136-elem rows + ldmatrix.x4 feed) over fp32gs_regpipe (sha 62646080); union verified op-by-op.
// v3_transpose: accumulator->probe-lane transpose with 16 SHFL.IDX instead of 32.
// v1_argmax: branchless local argmax + REDUX cross-lane argmax (bitwise identical seeds).
// v2fix_abd: hoist_anchor=1 cand_shfl=1 packed=0 restrict=1 pin_picked=0
// TIMING/NUMERICS CANDIDATE: the Fisher Gram-Schmidt runs in FP32 instead of
// FP64. The Triton reference computes this in float64, so seeds are NOT
// guaranteed to match; the seed mismatch count against the reference is the
// whole point of the measurement. Probe dots are unchanged.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include "warp_mma.cuh"
using bf16=__nv_bfloat16;
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
  if constexpr(P==4) {
    // Probe subset s lives on lane bits 4,3 (v8: bits 3,2); probe t on bits
    // 1,0. Lane l's survivor is center[s] for the probe s*4+t, i.e. v8's
    // center[subset] on the lane that used to own subset s.
    // In-lane tree (z0+z2)+(z1+z3) is v8's sum_registers over the 4 candidates.
    const bool hb=((lane>>4)&1)!=0,lb=((lane>>3)&1)!=0;
    float a[4];
    #pragma unroll
    for(int p=0;p<4;++p){
      float terms[4];
      #pragma unroll
      for(int c=0;c<4;++c)terms[c]=z[c][p];
      a[p]=sum_registers(terms);
    }
    // xor 16 (slot bit 2 = s bit 1): keep p with bit1(p)==hb. u_p=a_p(l)+a_p(l^16).
    float u_even=halve_add(a[0],a[2],hb,16);
    float u_odd=halve_add(a[1],a[3],hb,16);
    // xor 8 (slot bit 1 = s bit 0): keep p with bit0(p)==lb. v_p=u_p(l)+u_p(l^8).
    float v=halve_add(u_even,u_odd,lb,8);
    // xor 4 (slot bit 0): full add of the single survivor. w_p=v_p(l)+v_p(l^4).
    float w=__fadd_rn(v,__shfl_xor_sync(0xffffffff,v,4));
    float argument=__fmul_rn(w,1.0f/32);
    // mx: max over the 16 distinct centers (s on bits 4,3; t on bits 1,0).
    // fmaxf is exact and order-independent, so this is v8's mx.
    // mx via REDUX: map each center to an order-preserving signed int (flip
    // the low 31 bits of negatives), take the warp maximum with one
    // REDUX.MAX.S32 and map back. The map is strictly monotone on non-NaN
    // floats, so the maximum int is the maximum float: v8's mx. (+0 beats -0
    // here where fmaxf may return either; x-(+0)==x-(-0) for x!=0 and
    // expf(+0)==expf(-0)==1, so a zero mx of either sign gives the same exps.)
    int mk=__float_as_int(argument);
    mk^=(mk>>31)&0x7fffffff;
    mk=__reduce_max_sync(0xffffffffu,mk);
    mk^=(mk>>31)&0x7fffffff;
    float mx=__int_as_float(mk);
    float exponential=expf(__fadd_rn(argument,-mx));
    // Same balanced addition tree as v8: (e_s+e_{s^2})+(e_{s^1}+e_{s^3}) via
    // xor 16 then xor 8 (v8: xor 8 then xor 4), then probe xor 2 / xor 1.
    float den=exponential;
    den=__fadd_rn(den,__shfl_xor_sync(0xffffffff,den,16));
    den=__fadd_rn(den,__shfl_xor_sync(0xffffffff,den,8));
    den=__fadd_rn(den,__shfl_xor_sync(0xffffffff,den,2));
    den=__fadd_rn(den,__shfl_xor_sync(0xffffffff,den,1));
    float normalized=__fdiv_rn(exponential,den),owned_root=sqrtf(normalized);
    #pragma unroll
    for(int p=0;p<4;++p) {
      rho[p]=__shfl_sync(0xffffffff,normalized,p*8+probe);
      root[p]=__shfl_sync(0xffffffff,owned_root,p*8+probe);
    }
  } else {
    float center[PR];
    #pragma unroll
    for(int p=0;p<PR;++p){
      float terms[P];
      #pragma unroll
      for(int c=0;c<P;++c)terms[c]=z[c][p];
      center[p]=__fmul_rn(sum_candidate32<P>(terms),1.0f/32);
    }
    float mx=-CUDART_INF_F;
    #pragma unroll
    for(int p=0;p<PR;++p)mx=fmaxf(mx,center[p]);
    #pragma unroll
    for(int d=P/2;d;d/=2)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
    float den_terms[PR];
    #pragma unroll
    for(int p=0;p<PR;++p)rho[p]=den_terms[p]=expf(__fadd_rn(center[p],-mx));
    float den=sum_probe16<P>(den_terms);
    #pragma unroll
    for(int p=0;p<PR;++p){rho[p]=__fdiv_rn(rho[p],den);root[p]=sqrtf(rho[p]);}
  }
  #pragma unroll
  for(int p=0;p<PR;++p){
    float value=0;
    #pragma unroll
    for(int c=0;c<P;++c)value=choose_candidate(value,z[c][p],anchor/CS,c);
    anchor_z[p]=__shfl_sync(0xffffffff,value,(anchor%CS)*P+probe);
  }
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
  unsigned used=1u<<anchor;
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
        float owned_basis=__fdiv_rn(value,norm);
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
__global__ __launch_bounds__(128,4) void fused_selector(const bf16* __restrict__ q,
 const bf16* __restrict__ k,const int* __restrict__ candidates,const int* __restrict__ anchors,
 uint8_t* __restrict__ seeds,
 double scale,float* debug_logits) {
 __shared__ __align__(128) bf16 raw[128*136+16*136];
 bf16* sq=raw;bf16* sk=sq+128*136;
 int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
 int group1=blockIdx.x,key_row=group1%45;
 int cg=(group1/45)%20,head=group1/(45*20),column=cg*4+warp;
 const int anchor=anchors[key_row];
 const int cand_lane=candidates[key_row*32+lane];
 for(int vec=tid;vec<4*32*16;vec+=128) {
  int row=vec/16,d=(vec%16)*8,candidate=row%32,col=cg*4+row/32;
  int qr=candidates[key_row*32+candidate];
  int qi=((qr*80+col)*40+head)*128+d;
  *reinterpret_cast<uint4*>(sq+qoff(row,d))=*reinterpret_cast<const uint4*>(q+qi);
 }
 // Stay one frame ahead: each thread owns probes p and p+8, so the whole
 // 16x128 probe tile lives in two uint4 registers. The load for frame+1 is
 // issued before this frame's dots and Gram-Schmidt and only reaches shared at
 // the top of the next frame, so its latency hides behind work.
 int pf_probe=tid>>4,pf_d=(tid&15)*8;
 uint4 pf0,pf1;
 {
  int ki=((((0*45+key_row)*80+pf_probe*5)*40+head)*128)+pf_d;
  pf0=*reinterpret_cast<const uint4*>(k+ki);
  ki=((((0*45+key_row)*80+(pf_probe+8)*5)*40+head)*128)+pf_d;
  pf1=*reinterpret_cast<const uint4*>(k+ki);
 }
 #pragma unroll 1
 for(int frame=0;frame<21;++frame) {
 __syncthreads();
 *reinterpret_cast<uint4*>(sk+qoff(pf_probe,pf_d))=pf0;
 *reinterpret_cast<uint4*>(sk+qoff(pf_probe+8,pf_d))=pf1;
 __syncthreads();
 if(frame<20) {
  int ki=((((frame+1)*45+key_row)*80+pf_probe*5)*40+head)*128+pf_d;
  pf0=*reinterpret_cast<const uint4*>(k+ki);
  ki=((((frame+1)*45+key_row)*80+(pf_probe+8)*5)*40+head)*128+pf_d;
  pf1=*reinterpret_cast<const uint4*>(k+ki);
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
 int group=((frame*45+key_row)*80+column)*40+head;
 if(debug_logits) {
  #pragma unroll
  for(int c=0;c<4;++c) {
   #pragma unroll
   for(int p=0;p<4;++p)debug_logits[(int64_t(group)*32+c*8+lane/4)*16+p*4+(lane&3)]=local[c][p];
  }
 }
 fisher_from_registers<4,4>(local,cand_lane,anchor,seeds,group,float(scale));
 }
}
void run(torch::Tensor q,torch::Tensor k,torch::Tensor candidates,torch::Tensor anchors,
 torch::Tensor seeds,double scale,torch::Tensor debug) {
 TORCH_CHECK(q.is_cuda() && k.is_cuda() && q.is_contiguous() && k.is_contiguous(),"Contiguous CUDA Q/K required");
 TORCH_CHECK(q.scalar_type()==at::kBFloat16 && k.scalar_type()==at::kBFloat16,"BF16 Q/K required");
 TORCH_CHECK(q.numel()==45*80*40*128 && k.numel()==21*45*80*40*128,"Native first-frame Q and full K required");
 TORCH_CHECK(candidates.scalar_type()==at::kInt && anchors.scalar_type()==at::kInt,"Int32 metadata required");
 TORCH_CHECK(seeds.scalar_type()==at::kByte && seeds.numel()==21*45*80*40*4,"Native C4 seeds required");
 TORCH_CHECK(!debug.numel() || (debug.scalar_type()==at::kFloat && debug.numel()==int64_t(21)*45*80*40*32*16),"Debug logits shape invalid");
 fused_selector<<<45*20*40,128,0,at::cuda::getCurrentCUDAStream()>>>(
  (const bf16*)q.data_ptr(),(const bf16*)k.data_ptr(),candidates.data_ptr<int>(),anchors.data_ptr<int>(),
  seeds.data_ptr<uint8_t>(),scale,debug.numel()?debug.data_ptr<float>():nullptr);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<int> resources() {
 cudaFuncAttributes attr{};C10_CUDA_CHECK(cudaFuncGetAttributes(&attr,fused_selector));
 int active=0;C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,fused_selector,128,0));
 return {attr.numRegs,int(attr.localSizeBytes),int(attr.sharedSizeBytes),active};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);m.def("resources",&resources);}
