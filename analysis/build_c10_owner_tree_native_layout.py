"""Balanced leftmost argmax in the retained shared layout, no transpose."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'native_fisher/fisher_c10_half_owner.cu').read_text()
start=s.index('        float best=-CUDART_INF_F;int owner=0;')
end=s.index('        selected_winners[index]=best;',start)
new=''
for i in range(5):
    new+=f'''        float z{i}=normalized[(column*10+{2*i})*40+ki];
        float b{i}=normalized[(column*10+{2*i+1})*40+ki];
        int i{i}={2*i};
        if(b{i}>z{i}){{z{i}=b{i};i{i}={2*i+1};}}
'''
for a,b in [(0,1),(2,3),(0,2),(0,4)]:
    new+=f'        if(z{b}>z{a}){{z{a}=z{b};i{a}=i{b};}}\n'
new+='        float best=z0;int owner=i0;\n'
s=s[:start]+new+s[end:]
p=ROOT/'native_fisher/fisher_c10_owner_tree_native_layout.cu';p.write_text(s);print(p)
