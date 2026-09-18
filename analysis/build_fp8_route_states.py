"""Produce independent FP32 compressed/SQ1 states for backend validation."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'native_fisher/fp8_direct_route.py').read_text()
s=s.replace('    out_ptr,\n','    out_ptr,\n    lse_ptr,\n')
a=s.index('    if OUTPUT_HEAD_MAJOR:\n',s.index('    row_max ='))
b=s.index('    tl.store(\n        out_ptr',a)
s=s[:a]+'''    state_address=(((query_outer*COLS+query_col)*ACTIVE_HEADS+head)*ROWS+query_rows)
    output_address=state_address[:,None]*D_HEAD+dims[None,:]
    tl.store(lse_ptr+state_address,row_max+tl.log2(row_sum),mask=query_mask)
'''+s[b:]
compressed=s.replace('def fp8_direct_route(', 'def compressed_state(')
compressed=compressed.replace('    if SINGLETON_TOP_R > 0:\n        if SINGLETON_PRIMARY_ROW_MOD4_OMIT2:', '    if False:\n        if SINGLETON_PRIMARY_ROW_MOD4_OMIT2:')
singleton=s.replace('def fp8_direct_route(', 'def singleton_state(')
singleton=singleton.replace('    total_tokens = K_OUTER * ROWS * C','    total_tokens = 0')
(root/'native_fisher/fp8_route_states.py').write_text(compressed+'\n\n'+singleton)
