"""Lossless physical Q packing for resident-KV C10 FP8 merge.

Keep the four-lane register-fragment permutation, but store packed Q as
[head,column,frame,row,D]. The same head/column's candidate rows are then
contiguous across query frames. Original QKV, Fisher selection, BF16 weights,
FP8 intermediate layout and full route are unchanged. Packing remains one
kernel already included in each complete attention call.
"""
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'native_fisher'
s=(root/'fisher_c10_fp8_local_scores.cu').read_text()
for suffix in ('0','1'):
    old=f'(((qo*R+sr{suffix})*C+min(col{suffix},79))*H+gh)*128+(lane&3)*8'
    new=f'(((gh*C+min(col{suffix},79))*Q+qo)*R+sr{suffix})*128+(lane&3)*8'
    assert s.count(old)==1
    s=s.replace(old,new)
s=s.replace('void pack_query_kernel(const bf16* input,bf16* output,int64_t vectors)',
            'void pack_query_kernel(const bf16* input,bf16* output,int64_t vectors,int heads)')
old='    *reinterpret_cast<uint4*>(output+vec*8)=b;'
assert s.count(old)==1
s=s.replace(old,'''    int64_t token_head=vec/16;
    int head=token_head%heads,token=token_head/heads;
    int col=token%80,row=(token/80)%45,frame=token/(80*45);
    int64_t dest=(((int64_t(head)*80+col)*21+frame)*45+row)*128+(vec%16)*8;
    *reinterpret_cast<uint4*>(output+dest)=b;''')
s=s.replace('(const bf16*)input.data_ptr(),(bf16*)output.data_ptr(),vectors);',
            '(const bf16*)input.data_ptr(),(bf16*)output.data_ptr(),vectors,input.size(4));')
s=s.replace('  int64_t vectors=input.numel()/8;',
            '  TORCH_CHECK(input.dim()==6 && input.size(0)==1 && input.size(1)==21 && input.size(2)==45 && input.size(3)==80 && input.size(5)==128,"Require native query shape for head/column packing");\n  int64_t vectors=input.numel()/8;')
(root/'fisher_c10_fp8_head_column_q.cu').write_text('// Experimental physical head/column-major packed Q.\n'+s)
