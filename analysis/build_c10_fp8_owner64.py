"""64+16 ownership tiles with packed indices in unused matrix rows.

The 64x64 FP32 score tile occupies16KiB. Rows60..63 have no live C10 seed;
after the score-store barrier they hold6x64 uint16 owners (768bytes).
Owner selection reads only rows0..59. A second barrier precedes packed
owner reads, and the final barrier precedes reuse by the next key tile.
The original first64/last16 probability and entropy summation order stays
unchanged. Only shared layout, loop partitioning and index storage change.
"""
from pathlib import Path
import re

root=Path(__file__).resolve().parents[1]/'native_fisher'
s=(root/'fisher_c10_fp8_local_scores.cu').read_text()
marker='// K/V stay resident across query tiles.'
s=s.replace(marker,'''__device__ __forceinline__ int score_offset64(int row,int key) {
  return row*64+(key^(((row&6)<<2)|(row&1)));
}
'''+marker,1)
start=s.index('    float* normalized=reinterpret_cast<float*>(temp);')
end=s.index('    uint32_t packed_weights[20];',start)
t=s[start:end]
t=t.replace('int* selected_owners=reinterpret_cast<int*>(normalized+64*40);',
            'uint16_t* selected_owners=reinterpret_cast<uint16_t*>(normalized+60*64);')
t=t.replace('for(int chunk=0;chunk<2;++chunk) {',
            'for(int chunk=0;chunk<2;++chunk) {\n      const int keys=chunk==0?64:16;\n      const int score_start=chunk==0?0:32;')
t=t.replace('for(int i=0;i<20;++i)', 'for(int i=0;i<keys/2;++i)')
t=t.replace('normalized[row*40+ki]=s[chunk*20+i];',
            'normalized[score_offset64(row,ki)]=s[score_start+i];')
t=t.replace('index<6*40', 'index<6*keys').replace('int column=index/40,ki=index%40;', 'int column=index/keys,ki=index%keys;')
t,n=re.subn(r'normalized\[\(column\*10\+([0-9])\)\*40\+ki\]',r'normalized[score_offset64(column*10+\1,ki)]',t)
assert n==10
t=t.replace('selected_owners[index]=owner;', 'selected_owners[column*64+ki]=uint16_t(owner);')
t=t.replace('pair<5', 'pair<keys/8')
t=t.replace('int index=min(col_for_row(row),5)*40+pair*8+2*(lane&3);',
            'int index=min(col_for_row(row),5)*64+pair*8+2*(lane&3);\n          uint32_t packed_owners=*reinterpret_cast<const uint32_t*>(selected_owners+index);')
t=t.replace('selected_owners[index]==owner', 'int(packed_owners&65535)==owner')
t=t.replace('selected_owners[index+1]==owner', 'int(packed_owners>>16)==owner')
t=t.replace('chunk*20+', 'score_start+')
t=t.replace('else if(chunk==0 || pair<3)', 'else if(chunk==0)')
t=t.replace('else if(pair==3)', 'else if(pair==0)')
s=s[:start]+t+s[end:]
s=s.replace('(64*40+6*40)*4', '(64*64)*4')
(root/'fisher_c10_fp8_owner64.cu').write_text(s)
