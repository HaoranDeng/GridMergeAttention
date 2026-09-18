"""Reuse each seed's register score after exact owner selection.

Only the winning seed contributes probability and entropy. Its register
score is identical to the winner score previously reloaded from shared
memory. Non-owner probabilities remain zero. Gate signed-zero/rounding
effects on complete storage and output before timing or promotion.
"""
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'native_fisher'
source = (root / 'fisher_c10_fp8_shared_store.cu').read_text()
old = '''    float* selected_winners=normalized+64*40;
    int* selected_owners=reinterpret_cast<int*>(selected_winners+6*40);'''
assert source.count(old) == 1
source = source.replace(old, '    int* selected_owners=reinterpret_cast<int*>(normalized+64*40);')
old = '        selected_winners[index]=best;selected_owners[index]=owner;'
assert source.count(old) == 1
source = source.replace(old, '        selected_owners[index]=owner;')
old = '          float x0=selected_winners[index],x1=selected_winners[index+1];'
assert source.count(old) == 1
source = source.replace(old, '''          float x0=s[chunk*20+pair*4+half*2];
          float x1=s[chunk*20+pair*4+half*2+1];''')
assert '(64*40+2*6*40)*4' in source
source = source.replace('(64*40+2*6*40)*4', '(64*40+6*40)*4')
(root / 'fisher_c10_fp8_local_scores.cu').write_text(source)
