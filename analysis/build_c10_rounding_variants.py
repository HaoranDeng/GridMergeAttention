"""Mathematically unchanged FP32 arithmetic with ordinary rounding differences.

Fisher FP64 selection, ten seeds, ownership comparison, SQ1 and BF16 stored
operands are retained. These candidates require numerical and video validation.
"""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'native_fisher/fisher_c10_half_owner.cu').read_text()
s=s.replace('div.full.f32 %0,0f3F800000,%1;', 'rcp.approx.ftz.f32 %0,%1;')
(root/'native_fisher/fisher_c10_fast_reciprocal.cu').write_text(s)
start=s.index('__device__ __forceinline__ float reduce80(')
end=s.index('\n__device__ __forceinline__ float reciprocal',start)
s=s[:start]+'''__device__ __forceinline__ float reduce80(float (&p)[20]) {
  float a0=__fadd_rn(p[0],p[1]),a1=__fadd_rn(p[2],p[3]);
  float a2=__fadd_rn(p[4],p[5]),a3=__fadd_rn(p[6],p[7]);
  float a4=__fadd_rn(p[8],p[9]),a5=__fadd_rn(p[10],p[11]);
  float a6=__fadd_rn(p[12],p[13]),a7=__fadd_rn(p[14],p[15]);
  float main=__fadd_rn(__fadd_rn(__fadd_rn(a0,a1),__fadd_rn(a2,a3)),__fadd_rn(__fadd_rn(a4,a5),__fadd_rn(a6,a7)));
  float tail=__fadd_rn(__fadd_rn(p[16],p[17]),__fadd_rn(p[18],p[19]));
  return __fadd_rn(sum4(main),sum4(tail));
}
''' +s[end:]
s=s.replace('float mass_main[2],mass_tail[2],entropy_main[2],entropy_tail[2];','float mass_main[2],mass_tail[2],entropy_main[2],entropy_tail[2],entropy_main_odd[2],entropy_tail_odd[2];')
s=s.replace('entropy_main[half]=__fmaf_rn(p0,x0,__fmul_rn(p1,x1));','entropy_main[half]=__fmul_rn(p0,x0);entropy_main_odd[half]=__fmul_rn(p1,x1);')
s=s.replace('entropy_tail[half]=__fmaf_rn(p0,x0,__fmul_rn(p1,x1));','entropy_tail[half]=__fmul_rn(p0,x0);entropy_tail_odd[half]=__fmul_rn(p1,x1);')
s=s.replace('entropy_main[half]=__fmaf_rn(p1,x1,__fmaf_rn(p0,x0,entropy_main[half]));','entropy_main[half]=__fmaf_rn(p0,x0,entropy_main[half]);entropy_main_odd[half]=__fmaf_rn(p1,x1,entropy_main_odd[half]);')
s=s.replace('entropy_tail[half]=__fmaf_rn(p1,x1,__fmaf_rn(p0,x0,entropy_tail[half]));','entropy_tail[half]=__fmaf_rn(p0,x0,entropy_tail[half]);entropy_tail_odd[half]=__fmaf_rn(p1,x1,entropy_tail_odd[half]);')
s=s.replace('float sum=__fadd_rn(sum4(mass_main[half]),sum4(mass_tail[half]));','float sum=reduce80(probabilities);')
s=s.replace('float expected=__fadd_rn(sum4(entropy_main[half]),sum4(entropy_tail[half]));','float expected=__fadd_rn(sum4(__fadd_rn(entropy_main[half],entropy_main_odd[half])),sum4(__fadd_rn(entropy_tail[half],entropy_tail_odd[half])));')
(root/'native_fisher/fisher_c10_balanced_sums.cu').write_text(s)

# Reuse the first softmax exponentials for ownership. Comparing positive
# probabilities is mathematically equivalent to comparing log probabilities;
# FP32 rounding near a tie may select a different owner, so this is experimental.
s=(root/'native_fisher/fisher_c10_half_owner.cu').read_text()
s=s.replace('div.full.f32 %0,0f3F800000,%1;', 'rcp.approx.ftz.f32 %0,%1;')
s=s.replace('norm_max[half]=mx;norm_sum[half]=reduce80(bs);', '''float inv=reciprocal(reduce80(bs));
      #pragma unroll
      for(int i=0;i<20;++i)s[4*(i/2)+2*half+(i&1)]=__fmul_rn(bs[i],inv);''')
start=s.index('    // Each four-lane row group')
end=s.index('    // Sixty live rows',start)
s=s[:start]+s[end:]
s=s.replace('int* selected_owners=reinterpret_cast<int*>(selected_winners+6*40);',
'''int* selected_owners=reinterpret_cast<int*>(selected_winners+6*40);
    float* selected_probabilities=reinterpret_cast<float*>(selected_owners+6*40);''')
s=s.replace('selected_winners[index]=best;selected_owners[index]=owner;',
'''selected_probabilities[index]=best;
        selected_winners[index]=lg2(fmaxf(best,1.e-38f));selected_owners[index]=owner;''')
s=s.replace('selected_owners[index]==owner?ex2(x0):0.f;', 'selected_owners[index]==owner?selected_probabilities[index]:0.f;')
s=s.replace('selected_owners[index+1]==owner?ex2(x1):0.f;', 'selected_owners[index+1]==owner?selected_probabilities[index+1]:0.f;')
(root/'native_fisher/fisher_c10_probability_owner.cu').write_text(s)
