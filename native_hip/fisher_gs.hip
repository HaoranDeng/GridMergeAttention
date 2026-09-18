// FP64 Fisher GS. W=32 is the direct logical-warp control; W=64 uses
// the whole CDNA wave to halve each lane's register-resident feature table.
// Explicit shuffle widths prevent communication between logical groups.
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>

template<int N> __device__ __forceinline__ double sum(double (&v)[N]) {
  #pragma unroll
  for(int d=N/2;d;d/=2) {
    #pragma unroll
    for(int j=0;j<d;++j) v[j]=v[j]+v[j+d];
  }
  return v[0];
}
template<int W,int P,int N>
__device__ __forceinline__ double reduce(double (&v)[N],bool probes) {
  double x=sum(v);
  if(probes) {
    #pragma unroll
    for(int d=P/2;d;d/=2) x=x+__shfl_xor(x,d,W);
  } else {
    #pragma unroll
    for(int d=W/2;d>=P;d/=2) x=x+__shfl_xor(x,d,W);
  }
  return x;
}
template<int C,int W,int P>
__global__ void fisher_gs(const float* __restrict__ logits,
    const int* __restrict__ candidates,const int* __restrict__ anchors,
    uint8_t* __restrict__ seeds,int groups,int heads,double scale) {
  constexpr int CS=W/P,CR=32/CS,PR=16/P;
  const int lane=threadIdx.x%W,probe=lane%P,slot=lane/P;
  const int group=(blockIdx.x*blockDim.x+threadIdx.x)/W;
  if(group>=groups)return;
  const int row=(group/(80*heads))%45,anchor=anchors[row];
  double z[CR][PR],center[PR];
  #pragma unroll
  for(int p=0;p<PR;++p) {
    double terms[CR];
    #pragma unroll
    for(int c=0;c<CR;++c) {
      z[c][p]=double(logits[(int64_t(group)*32+c*CS+slot)*16+p*P+probe])*scale;
      terms[c]=z[c][p];
    }
    center[p]=reduce<W,P>(terms,false)*(1.0/32);
  }
  double mx=-INFINITY;
  #pragma unroll
  for(int p=0;p<PR;++p)mx=fmax(mx,center[p]);
  #pragma unroll
  for(int d=P/2;d;d/=2)mx=fmax(mx,__shfl_xor(mx,d,W));
  double chosen=0;
  #pragma unroll
  for(int p=0;p<PR;++p)chosen=slot%PR==p?center[p]:chosen;
  double rho_scalar=exp(chosen-mx),den=rho_scalar;
  #pragma unroll
  for(int d=8;d;d/=2)den=den+__shfl_xor(den,d,W);
  rho_scalar=rho_scalar/den;
  double root_scalar=sqrt(rho_scalar),rho[PR],root[PR],anchor_z[PR];
  #pragma unroll
  for(int p=0;p<PR;++p) {
    rho[p]=__shfl(rho_scalar,p*P+probe,W);
    root[p]=__shfl(root_scalar,p*P+probe,W);
    double x=0;
    #pragma unroll
    for(int c=0;c<CR;++c)x=c==anchor/CS?z[c][p]:x;
    anchor_z[p]=__shfl(x,(anchor%CS)*P+probe,W);
  }
  double res[CR][PR];
  #pragma unroll
  for(int c=0;c<CR;++c) {
    double delta[PR],terms[PR];
    #pragma unroll
    for(int p=0;p<PR;++p) {delta[p]=z[c][p]-anchor_z[p];terms[p]=delta[p]*rho[p];}
    double mean=reduce<W,P>(terms,true);
    #pragma unroll
    for(int p=0;p<PR;++p)res[c][p]=(delta[p]-mean)*root[p];
  }
  unsigned used=1u<<anchor;
  if(lane==0)seeds[int64_t(group)*C]=uint8_t(candidates[row*32+anchor]);
  #pragma unroll
  for(int step=1;step<C;++step) {
    double best=-INFINITY;int picked=32;
    #pragma unroll
    for(int c=0;c<CR;++c) {
      double terms[PR];
      #pragma unroll
      for(int p=0;p<PR;++p)terms[p]=res[c][p]*res[c][p];
      double score=reduce<W,P>(terms,true);int index=c*CS+slot;
      if(!(used&(1u<<index)) && (score>best || (score==best && index<picked))) {best=score;picked=index;}
    }
    #pragma unroll
    for(int d=W/2;d>=P;d/=2) {
      double other=__shfl_xor(best,d,W);int oi=__shfl_xor(picked,d,W);
      if(other>best || (other==best && oi<picked)) {best=other;picked=oi;}
    }
    used|=1u<<picked;
    if(lane==0)seeds[int64_t(group)*C+step]=uint8_t(candidates[row*32+picked]);
    if(step<C-1) {
      double norm=sqrt(best+1.e-12),basis[PR],bs=0;
      #pragma unroll
      for(int p=0;p<PR;++p) {
        double x=0;
        #pragma unroll
        for(int c=0;c<CR;++c)x=c==picked/CS?res[c][p]:x;
        x=__shfl(x,(picked%CS)*P+probe,W);
        bs=slot%PR==p?x:bs;
      }
      bs=bs/norm;
      #pragma unroll
      for(int p=0;p<PR;++p)basis[p]=__shfl(bs,p*P+probe,W);
      #pragma unroll
      for(int c=0;c<CR;++c) {
        double terms[PR];
        #pragma unroll
        for(int p=0;p<PR;++p)terms[p]=res[c][p]*basis[p];
        double projection=reduce<W,P>(terms,true);
        #pragma unroll
        for(int p=0;p<PR;++p)res[c][p]=fma(-projection,basis[p],res[c][p]);
      }
    }
  }
}
template<int C,int W,int P>
void launch(const float* l,const int* t,const int* a,uint8_t* s,int n,int h,double scale,int threads,hipStream_t stream) {
  fisher_gs<C,W,P><<<(n+threads/W-1)/(threads/W),threads,0,stream>>>(l,t,a,s,n,h,scale);
}
extern "C" int run(const float* l,const int* t,const int* a,uint8_t* s,int n,int h,int c,int w,int p,int threads,double scale,void* stream) {
  if(n<=0 || h<=0 || (c!=4 && c!=8 && c!=10) || (threads!=128 && threads!=256))return int(hipErrorInvalidValue);
  #define CASE(C,W,P) if(c==C && w==W && p==P) {launch<C,W,P>(l,t,a,s,n,h,scale,threads,(hipStream_t)stream);return int(hipGetLastError());}
  CASE(4,32,4) CASE(4,32,8) CASE(4,32,16)
  CASE(4,64,2) CASE(4,64,4) CASE(4,64,8) CASE(4,64,16)
  CASE(8,32,4) CASE(8,32,8) CASE(8,32,16)
  CASE(8,64,2) CASE(8,64,4) CASE(8,64,8) CASE(8,64,16)
  CASE(10,32,4) CASE(10,32,8) CASE(10,32,16)
  CASE(10,64,2) CASE(10,64,4) CASE(10,64,8) CASE(10,64,16)
  #undef CASE
  return int(hipErrorInvalidValue);
}
