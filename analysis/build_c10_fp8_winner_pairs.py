"""Cache each winner exponential in a float2, with separate compact owners.

Unlike the rejected BF16 uint4 tuple, this uses only three words per winner
and the lower-register FP8 shared-store epilogue.
"""
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'native_fisher'
s=(root/'fisher_c10_fp8_shared_store.cu').read_text()
old='''    float* selected_winners=normalized+64*40;
    int* selected_owners=reinterpret_cast<int*>(selected_winners+6*40);'''
assert old in s
s=s.replace(old,'''    float2* selected_pairs=reinterpret_cast<float2*>(normalized+64*40);
    int* selected_owners=reinterpret_cast<int*>(selected_pairs+6*40);''')
s=s.replace('selected_winners[index]=best;selected_owners[index]=owner;',
            'selected_pairs[index]=make_float2(best,ex2(best));selected_owners[index]=owner;')
s=s.replace('''          float x0=selected_winners[index],x1=selected_winners[index+1];
          float p0=selected_owners[index]==owner?ex2(x0):0.f;
          float p1=selected_owners[index+1]==owner?ex2(x1):0.f;''','''          float2 a=selected_pairs[index],b=selected_pairs[index+1];
          float x0=a.x,x1=b.x;
          float p0=selected_owners[index]==owner?a.y:0.f;
          float p1=selected_owners[index+1]==owner?b.y:0.f;''')
s=s.replace('(64*40+2*6*40)*4','(64*40+3*6*40)*4')
(root/'fisher_c10_fp8_winner_pairs.cu').write_text(s)
