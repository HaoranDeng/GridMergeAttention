"""Derive the directed-axis CUDA kernels from the audited exact-C4 sources."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def replace(s,a,b):
    assert a in s,a
    return s.replace(a,b)

def main():
    s=(ROOT/'kernels/best/c4/fisher_c4_selector_v9c_halving_mxredux.cu').read_text()
    s=s[s.index('#include <torch/extension.h>'):]
    s=replace(s,'#include "warp_mma.cuh"','#include "../kernels/best/c4/warp_mma.cuh"')
    s=replace(s,'using bf16=__nv_bfloat16;', 'using bf16=__nv_bfloat16;\nconstexpr int O=AXIS_OUTER,R=AXIS_ROWS,L=AXIS_COLS,H=AXIS_HEADS,AH=AXIS_ACTIVE_HEADS,NC=R<32?R:32;')
    start=s.index('  float rho[PR],root[PR],anchor_z[PR];')
    end=s.index('  float res[P][PR];',start)
    s=s[:start]+'''  float rho[PR],root[PR],anchor_z[PR];
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
''' +s[end:]
    s=replace(s,'unsigned used=1u<<anchor;', 'unsigned used=unsigned(~((uint64_t(1)<<NC)-1)) | (1u<<anchor);')
    s=replace(s,'__fdiv_rn(value,norm)', '__fdividef(value,norm)')
    s=replace(s,'uint8_t* __restrict__ seeds,\n double scale','uint8_t* __restrict__ seeds,int HS,\n double scale')
    s=replace(s,'int group1=blockIdx.x,key_row=group1%45;\n int cg=(group1/45)%20,head=group1/(45*20),column=cg*4+warp;',
              'int group1=blockIdx.x,key_row=group1%R,qo=blockIdx.y;\n constexpr int CG=(L+3)/4;\n int cg=(group1/R)%CG,head=group1/(R*CG),column=cg*4+warp;')
    s=replace(s,'int qi=((qr*80+col)*40+head)*128+d;', 'int qi=(((qo*R+qr)*L+col)*H+HS+head)*128+d;')
    s=replace(s,'*reinterpret_cast<uint4*>(sq+qoff(row,d))=*reinterpret_cast<const uint4*>(q+qi);',
              '*reinterpret_cast<uint4*>(sq+qoff(row,d))=col<L?*reinterpret_cast<const uint4*>(q+qi):make_uint4(0,0,0,0);')
    s=replace(s,'int ki=((((0*45+key_row)*80+pf_probe*5)*40+head)*128)+pf_d;',
              'int ki=(((key_row*L+pf_probe*L/16)*H+HS+head)*128)+pf_d;')
    s=replace(s,'ki=((((0*45+key_row)*80+(pf_probe+8)*5)*40+head)*128)+pf_d;',
              'ki=(((key_row*L+(pf_probe+8)*L/16)*H+HS+head)*128)+pf_d;')
    s=replace(s,'for(int frame=0;frame<21;++frame)', 'for(int frame=0;frame<O;++frame)')
    s=replace(s,'if(frame<20)', 'if(frame<O-1)')
    s=replace(s,'int ki=((((frame+1)*45+key_row)*80+pf_probe*5)*40+head)*128+pf_d;',
              'int ki=((((frame+1)*R+key_row)*L+pf_probe*L/16)*H+HS+head)*128+pf_d;')
    s=replace(s,'ki=((((frame+1)*45+key_row)*80+(pf_probe+8)*5)*40+head)*128+pf_d;',
              'ki=((((frame+1)*R+key_row)*L+(pf_probe+8)*L/16)*H+HS+head)*128+pf_d;')
    s=replace(s,'int group=((frame*45+key_row)*80+column)*40+head;',
              'int group=(((qo*L+column)*AH+head)*O+frame)*R+key_row;')
    s=replace(s,'fisher_from_registers<4,4>(local,cand_lane,anchor,seeds,group,float(scale));',
              'if(column<L)fisher_from_registers<4,4>(local,cand_lane,anchor,seeds,group,float(scale));')
    start=s.index('void run(torch::Tensor q,');end=s.index('std::vector<int> resources()',start)
    s=s[:start]+'''void run(torch::Tensor q,torch::Tensor k,torch::Tensor candidates,torch::Tensor anchors,
         torch::Tensor seeds,int hs,double scale) {
 TORCH_CHECK(q.is_cuda() && q.is_contiguous() && k.is_contiguous(),"Contiguous CUDA inputs required");
 TORCH_CHECK(q.numel()==int64_t(O)*R*L*H*128 && k.numel()==q.numel(),"Compiled shape mismatch");
 TORCH_CHECK(q.scalar_type()==at::kBFloat16 && k.scalar_type()==at::kBFloat16,"BF16 required");
 TORCH_CHECK(candidates.numel()==R*32 && candidates.scalar_type()==at::kInt,"Padded candidate table required");
 TORCH_CHECK(seeds.numel()==int64_t(O)*L*AH*O*R*4 && seeds.scalar_type()==at::kByte,"Exact C4 seeds required");
 TORCH_CHECK(hs>=0 && hs+AH<=H,"Head range invalid");
 fused_selector<<<dim3(R*((L+3)/4)*AH,O),128,0,at::cuda::getCurrentCUDAStream()>>>(
  (const bf16*)q.data_ptr(),(const bf16*)k.data_ptr(),candidates.data_ptr<int>(),anchors.data_ptr<int>(),
  seeds.data_ptr<uint8_t>(),hs,scale,nullptr);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
''' +s[end:]
    prefix=s[s.index('template<int SEEDS,int P>'):s.index('  unsigned used=')]
    chol=prefix.replace('fisher_from_registers','fisher_cholesky')+'''
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
'''
    gram=prefix.replace('fisher_from_registers','fisher_gram').replace('float scale) {','float scale,float* gram) {')+'''
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
'''
    insertion=s.index('// Padded shared layout')
    s=s[:insertion]+chol+gram+s[insertion:]
    s=replace(s,'if(column<L)fisher_from_registers<4,4>(local,cand_lane,anchor,seeds,group,float(scale));',
              '''if(column<L){
    #if AXIS_CHOLESKY == 2
    fisher_gram<4,4>(local,cand_lane,anchor,seeds,group,float(scale),
      reinterpret_cast<float*>(raw+(W*32+16)*136)+warp*1440);
    #elif AXIS_CHOLESKY == 1
    fisher_cholesky<4,4>(local,cand_lane,anchor,seeds,group,float(scale));
    #else
    fisher_from_registers<4,4>(local,cand_lane,anchor,seeds,group,float(scale));
    #endif
  }''')
    s=replace(s,'constexpr int O=AXIS_OUTER,','constexpr int W=AXIS_SELECTOR_WARPS;\nconstexpr int O=AXIS_OUTER,')
    s=replace(s,'__launch_bounds__(128,4) void fused_selector','__launch_bounds__(W*32,W==4?4:2) void fused_selector')
    s=replace(s,'__shared__ __align__(128) bf16 raw[128*136+16*136];',
              'extern __shared__ __align__(128) bf16 raw[];')
    s=replace(s,'bf16* sq=raw;bf16* sk=sq+128*136;','bf16* sq=raw;bf16* sk=sq+W*32*136;')
    s=replace(s,'constexpr int CG=(L+3)/4;','constexpr int CG=(L+W-1)/W;')
    s=replace(s,'column=cg*4+warp','column=cg*W+warp')
    s=replace(s,'vec<4*32*16;vec+=128','vec<W*32*16;vec+=W*32')
    s=replace(s,'col=cg*4+row/32','col=cg*W+row/32')
    s=replace(s,'pf1=*reinterpret_cast<const uint4*>(k+ki);',
              'if constexpr(W==4)pf1=*reinterpret_cast<const uint4*>(k+ki);')
    s=replace(s,'*reinterpret_cast<uint4*>(sk+qoff(pf_probe+8,pf_d))=pf1;',
              'if constexpr(W==4)*reinterpret_cast<uint4*>(sk+qoff(pf_probe+8,pf_d))=pf1;')
    s=replace(s,'fused_selector<<<dim3(R*((L+3)/4)*AH,O),128,0,at::cuda::getCurrentCUDAStream()>>>(',
              '''constexpr int smem=(W*32+16)*136*2+(AXIS_CHOLESKY==2?W*5760:0);
 C10_CUDA_CHECK(cudaFuncSetAttribute(fused_selector,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
 fused_selector<<<dim3(R*((L+W-1)/W)*AH,O),W*32,smem,at::cuda::getCurrentCUDAStream()>>>(''')
    s=replace(s,'cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,fused_selector,128,0)',
              'cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,fused_selector,W*32,(W*32+16)*136*2+(AXIS_CHOLESKY==2?W*5760:0))')
    s=replace(s,'int(attr.sharedSizeBytes),active','int((W*32+16)*136*2+(AXIS_CHOLESKY==2?W*5760:0)),active')
    (ROOT/'attention/axis_current_c4_selector.cu').write_text('// Current-line anchor-Fisher, derived from native C4 v9c.\n'+s)

    s=(ROOT/'kernels/best/c4/fisher_c4_desc_select.cu').read_text()
    s=s[s.index('#include <torch/extension.h>'):]
    s=replace(s,'constexpr int Q=21,KO=21,R=45,C=80,H=TOTAL_HEADS,AH=TOTAL_HEADS==40?5:1,RF=1;',
              'constexpr int Q=AXIS_OUTER,KO=Q,R=AXIS_ROWS,C=AXIS_COLS,H=TOTAL_HEADS,AH=AXIS_ACTIVE_HEADS,RF=Q;')
    s=replace(s,'int src=((kb*80+ki)*H+gh)*128+d;', 'int src=((kb*C+ki)*H+gh)*128+d;')
    for name in ('k','v'):
        shared='sk' if name=='k' else 'sv'
        s=replace(s,f'*reinterpret_cast<uint4*>({shared}+boff(ki,d))=*reinterpret_cast<const uint4*>({name}+src);',
                  f'*reinterpret_cast<uint4*>({shared}+boff(ki,d))=ki<C?*reinterpret_cast<const uint4*>({name}+src):make_uint4(0,0,0,0);')
    s=replace(s,'int total=Q*(C/16)', 'int total=Q*((C+15)/16)')
    s=replace(s,'int si0=((((rf*KO+ko)*R+kr)*C+col0)*H+gh)*4+seed_for_row(row0);',
              'int si0=(((qo*C+col0)*AH+head)*(KO*R)+kb)*4+seed_for_row(row0);')
    s=replace(s,'int si1=((((rf*KO+ko)*R+kr)*C+col1)*H+gh)*4+seed_for_row(row0+8);',
              'int si1=(((qo*C+col1)*AH+head)*(KO*R)+kb)*4+seed_for_row(row0+8);')
    s=replace(s,'int sr0=seeds[si0],sr1=seeds[si1];', 'int sr0=col0<C?seeds[si0]:0,sr1=col1<C?seeds[si1]:0;')
    s=replace(s,'uint4 a=query_vector(q+qi0+d,cache_last,lane),b=query_vector(q+qi1+d,cache_last,lane);',
              'uint4 a=col0<C?query_vector(q+qi0+d,cache_last,lane):make_uint4(0,0,0,0),b=col1<C?query_vector(q+qi1+d,cache_last,lane):make_uint4(0,0,0,0);')
    marker='    if(debug_scores) {'
    s=replace(s,marker,'''    #pragma unroll
    for(int i=0;i<40;++i) {
      int kc=2*(lane&3)+8*(i>>2)+(i&1);
      if(kc>=C)s[i]=-CUDART_INF_F;
    }
'''+marker)
    s=replace(s,'winners[i]=best;\n        probabilities[i]=wins?ex2(best):0.f;',
              'bool key_valid=2*(lane&3)+8*(i/2)+(i&1)<C;\n        winners[i]=key_valid?best:0.f;\n        probabilities[i]=(wins && key_valid)?ex2(best):0.f;')
    s=replace(s,'if((lane&3)==0)bias[out]', 'if((lane&3)==0 && col<C)bias[out]')
    s=replace(s,'TORCH_CHECK(q.size(3)==80 && q.size(5)==128,"Require native80 D128");',
              'TORCH_CHECK(q.size(3)==AXIS_COLS && q.size(5)==128,"Compiled columns/D128 required");')
    s=replace(s,'int qo=q.size(1),ko=k.size(1),r=q.size(2),c=80,h=q.size(4),rf=seeds.size(1);',
              'int qo=q.size(1),ko=k.size(1),r=q.size(2),c=AXIS_COLS,h=q.size(4),rf=qo;')
    s=replace(s,'TORCH_CHECK(qo==21 && ko==21 && r==45 && rf==1,"Require 21 frames, 45 rows, and one router frame");',
              'TORCH_CHECK(qo==AXIS_OUTER && ko==qo && r==AXIS_ROWS,"Compiled axis shape required");')
    s=replace(s,'TORCH_CHECK((h==1 && ah==1) || (h==40 && ah==5),"Require native H40/AH5 or diagnostic H1/AH1");',
              'TORCH_CHECK(h==AXIS_HEADS && ah==AXIS_ACTIVE_HEADS,"Compiled heads required");')
    s=replace(s,'dim3 grid(ko*r,ah,(qo*5+tiles-1)/tiles);', 'dim3 grid(ko*r,ah,(qo*((c+15)/16)+tiles-1)/tiles);')
    s=replace(s,'uint64_t dims[5]={64,uint64_t(ko*r*8),uint64_t(ah),80,uint64_t(qo)};',
              'uint64_t dims[5]={64,uint64_t(ko*r*8),uint64_t(ah),uint64_t(c),uint64_t(qo)};')
    s=replace(s,'if(h==40) launch_heads(std::integral_constant<int,40>{});\n   else launch_heads(std::integral_constant<int,1>{});',
              'launch_heads(std::integral_constant<int,AXIS_HEADS>{});')
    s=replace(s,'TORCH_CHECK(tiles==1 || tiles==3 || tiles==7 || tiles==21,"Require tiles 1/3/7/21");','TORCH_CHECK(tiles>0,"Positive tile count required");')
    s=replace(s,'vec<vectors;vec+=int64_t(gridDim.x)*blockDim.x)', 'vec<((vectors+31)/32)*32;vec+=int64_t(gridDim.x)*blockDim.x)')
    s=replace(s,'uint4 a=*reinterpret_cast<const uint4*>(input+vec*8);', 'uint4 a=vec<vectors?*reinterpret_cast<const uint4*>(input+vec*8):make_uint4(0,0,0,0);')
    s=replace(s,'*reinterpret_cast<uint4*>(output+vec*8)=b;', 'if(vec<vectors)*reinterpret_cast<uint4*>(output+vec*8)=b;')
    s=replace(s,'input.numel()%2048==0', 'input.numel()%128==0')
    s=s.replace('merge_kernel<21,40>','merge_kernel<21,AXIS_HEADS>')
    s=replace(s,'using bf16 = __nv_bfloat16;','using bf16 = __nv_bfloat16;\nconstexpr int KC=(AXIS_COLS+15)/16*16;')
    s=replace(s,'#define E40(a) E32(a),E8(a,32)',
              '''#define E40(a) E32(a),E8(a,32)
#define E24(a) E8(a,0),E8(a,8),E8(a,16)
#if AXIS_COLS <= 48
#define ES(a) E24(a)
#define SCORE_OP MMA_64x48x16_F32BF16BF16_RS
#else
#define ES(a) E40(a)
#define SCORE_OP MMA_64x80x16_F32BF16BF16_RS
#endif''')
    start=s.index('__device__ __forceinline__ float tree80(');end=s.index('__device__ __forceinline__ float reciprocal',start)
    s=s[:start]+'''__device__ __forceinline__ float reduce80(float (&p)[KC/4]) {
  float sum=p[0];
  #pragma unroll
  for(int i=1;i<KC/4;++i)sum+=p[i];
  return sum4(sum);
}
''' +s[end:]
    start=s.index('__device__ __forceinline__ float dot80(');end=s.index('// K/V stay resident',start)
    s=s[:start]+'''__device__ __forceinline__ float dot80(float (&p)[KC/4],float (&x)[KC/4]){
  float sum=p[0]*x[0];
  #pragma unroll
  for(int i=1;i<KC/4;++i)sum=fmaf(p[i],x[i],sum);
  return sum4(sum);
}
''' +s[end:]
    for old,new in [('(n>>6)*5120','(n>>6)*(KC*64)'),('80*128','KC*128'),('80*16','KC*16'),
                    ('desc(sk,10240,','desc(sk,KC*128,'),('desc(sv,10240,','desc(sv,KC*128,'),
                    ('float s[40]','float s[KC/2]'),('using Score=MMA_64x80x16_F32BF16BF16_RS','using Score=SCORE_OP'),
                    ('E40(s)','ES(s)'),('i<40;','i<KC/2;'),('[20]','[KC/4]'),('i<20;','i<KC/4;'),
                    ('i<10;','i<KC/8;'),('ki<80;','ki<KC;')]:
        s=replace(s,old,new)
    s=replace(s,'return log2f(x);','''#if AXIS_FAST_MATH
  float y;asm("lg2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;
  #else
  return log2f(x);
  #endif''')
    s=replace(s,'float y;asm("div.full.f32 %0,0f3F800000,%1;":"=f"(y):"f"(x));return y;',
              '''float y;
  #if AXIS_FAST_MATH
  asm("rcp.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));
  #else
  asm("div.full.f32 %0,0f3F800000,%1;":"=f"(y):"f"(x));
  #endif
  return y;''')
    (ROOT/'attention/axis_current_c4_merge.cu').write_text('// Directed-axis exact C4, derived from native C4 WGMMA/TMA.\n'+s)

if __name__=='__main__':main()
