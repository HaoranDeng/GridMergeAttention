"""Two key halves reuse the same owner buffer; preserve FP32 reduction order."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'native_fisher/fisher_c10_merge.cu').read_text()
start=s.index('    float* normalized=reinterpret_cast<float*>(temp);')
end=s.index('    uint32_t packed_weights[20];',start)
s=s[:start]+'''    // Forty-key owner tiles fit below the existing 16 KiB output staging
    // footprint. Four CTAs can now share an SM without changing live C10.
    float* normalized=reinterpret_cast<float*>(temp);
    float* selected_winners=normalized+64*40;
    int* selected_owners=reinterpret_cast<int*>(selected_winners+6*40);
    float mass_main[2],mass_tail[2],entropy_main[2],entropy_tail[2];
    if(tid==0)asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");
    __syncthreads();
    #pragma unroll
    for(int chunk=0;chunk<2;++chunk) {
      #pragma unroll
      for(int i=0;i<20;++i) {
        int row=warp*16+seed+8*((i&3)>>1);
        int ki=2*(lane&3)+8*(i>>2)+(i&1);
        normalized[row*40+ki]=s[chunk*20+i];
      }
      __syncthreads();
      for(int index=tid;index<6*40;index+=128) {
        int column=index/40,ki=index%40;
        float best=-CUDART_INF_F;int owner=0;
        #pragma unroll
        for(int c=0;c<10;++c) {
          float score=normalized[(column*10+c)*40+ki];
          if(score>best){best=score;owner=c;}
        }
        selected_winners[index]=best;selected_owners[index]=owner;
      }
      __syncthreads();
      #pragma unroll
      for(int half=0;half<2;++half) {
        int row=warp*16+seed+8*half,owner=seed_for_row(row);
        #pragma unroll
        for(int pair=0;pair<5;++pair) {
          int index=min(col_for_row(row),5)*40+pair*8+2*(lane&3);
          float x0=selected_winners[index],x1=selected_winners[index+1];
          float p0=selected_owners[index]==owner?ex2(x0):0.f;
          float p1=selected_owners[index+1]==owner?ex2(x1):0.f;
          s[chunk*20+pair*4+half*2]=p0;
          s[chunk*20+pair*4+half*2+1]=p1;
          if(chunk==0 && pair==0) {
            mass_main[half]=__fadd_rn(p0,p1);
            entropy_main[half]=__fmaf_rn(p0,x0,__fmul_rn(p1,x1));
          } else if(chunk==0 || pair<3) {
            mass_main[half]=__fadd_rn(__fadd_rn(mass_main[half],p0),p1);
            entropy_main[half]=__fmaf_rn(p1,x1,__fmaf_rn(p0,x0,entropy_main[half]));
          } else if(pair==3) {
            mass_tail[half]=__fadd_rn(p0,p1);
            entropy_tail[half]=__fmaf_rn(p0,x0,__fmul_rn(p1,x1));
          } else {
            mass_tail[half]=__fadd_rn(__fadd_rn(mass_tail[half],p0),p1);
            entropy_tail[half]=__fmaf_rn(p1,x1,__fmaf_rn(p0,x0,entropy_tail[half]));
          }
        }
      }
      __syncthreads();
    }
'''+s[end:]
start=s.index('      float winners[20],probabilities[20];')
end=s.index('      float selected=0.f;',start)
s=s[:start]+'''      float probabilities[20];
      int row=warp*16+seed+8*half,col=cb*6+col_for_row(row);
      int owner=seed_for_row(row);
      #pragma unroll
      for(int i=0;i<20;++i)probabilities[i]=s[(i/2)*4+half*2+(i&1)];
      float sum=__fadd_rn(sum4(mass_main[half]),sum4(mass_tail[half]));
      float safe=sum>0?sum:1.f;
      float inv=reciprocal(safe);
      float expected=__fadd_rn(sum4(entropy_main[half]),sum4(entropy_tail[half]));
'''+s[end:]
s=s.replace('(80*128*2)*2+(64*80+2*6*80)*4','(80*128*2)*2+64*128*2')
(root/'native_fisher/fisher_c10_half_owner.cu').write_text(s)
