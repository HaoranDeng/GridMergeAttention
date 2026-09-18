"""Specialize the retained FP8 merge for one active head per invocation."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'native_fisher'
s=(root/'fisher_c10_fp8_shared_store.cu').read_text()
s=s.replace('AH=TOTAL_HEADS==40?5:1','AH=1')
s=s.replace('(h==40 && ah==5)','(h==40 && ah==1)')
s=s.replace('H40/AH5','H40/AH1')
(root/'fisher_c10_fp8_head1.cu').write_text(s)
