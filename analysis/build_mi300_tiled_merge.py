"""Derive the MI300 C4 merge with optional scheduling/division experiments.

C4 uses mode 2 (reciprocal plus FMA correction); C8 uses mode 1.
Regenerate after changes to the source kernel, then rerun GPU validation.
"""
import ast,textwrap,re
from pathlib import Path
root=Path(__file__).resolve().parents[1]
source=(root/'attention/hard_c4_wan21_14b_720p_triton.py').read_text()
node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='_hard_c4_720p_temporal_shared_merge_kernel')
s=textwrap.dedent('\n'.join(source.splitlines()[node.lineno-1:node.end_lineno]))
s=s.replace('def _hard_c4_720p_temporal_shared_merge_kernel(', 'def merge_kernel(',1)
s=s.replace('    mean_seed_ptr=None,','    mean_seed_ptr=None,\n    PACK_Q: tl.constexpr = False,\n    COL_SPLITS: tl.constexpr = 1,\n    FRAME_SPLITS: tl.constexpr = 1,\n    RESIDUAL_DIV_MODE: tl.constexpr = 0,',1)
s=s.replace('    for query_col_start in tl.range(0, COLS, BLOCK_J):',
'''    tl.static_assert((COLS // BLOCK_J) % COL_SPLITS == 0)
    tl.static_assert(Q_OUTER % FRAME_SPLITS == 0)
    split_col = tl.program_id(2) % COL_SPLITS
    split_frame = tl.program_id(2) // COL_SPLITS
    for query_col_start in tl.range(split_col * (COLS // COL_SPLITS), (split_col + 1) * (COLS // COL_SPLITS), BLOCK_J):''',1)
s=s.replace('''        for query_outer in tl.range(
            0,
            Q_OUTER,''','''        for query_outer in tl.range(
            split_frame * (Q_OUTER // FRAME_SPLITS),
            (split_frame + 1) * (Q_OUTER // FRAME_SPLITS),''',1)
needle='''            seed_query = tl.load(
                q_ptr + seed_query_offsets,'''
replacement='''            if PACK_Q:
                seed_query_offsets = (((((batch * TOTAL_HEADS + global_head) * Q_OUTER + merge_query_outer)
                    * ROWS + active_seed_rows[:, None]) * COLS + merge_query_cols[:, None]) * D_HEAD + dims[None, :])
            seed_query = tl.load(
                q_ptr + seed_query_offsets,'''
assert needle in s
s=s.replace(needle,replacement,1)
assert 'split_frame * (Q_OUTER // FRAME_SPLITS)' in s
s=s.replace('''        tl.static_assert(not STORE_PACKED_SINGLETON_KV)
    if DIRECT_BULK_NORMALIZATION:''','''        # Packed singleton storage is independent of the split bulk K/V dots.
        # This combination is validated by the opt-in C8 entry point.
        tl.static_assert(not STORE_PACKED_SINGLETON_KV or C == 8)
    if DIRECT_BULK_NORMALIZATION:''',1)
out=root/'native_hip/merge_tiled.py'
s=re.sub(r'merged_(key|value) /= residual_scale\[:, None\]',r'merged_\1 = _divide_residual(merged_\1, residual_scale, RESIDUAL_DIV_MODE)',s)
helper='''@triton.jit
def _divide_residual(value, residual, MODE: tl.constexpr):
    if MODE == 0:
        return value / residual[:, None]
    inverse = 1.0 / residual
    quotient = value * inverse[:, None]
    if MODE == 2:
        error = tl.fma(-quotient, residual[:, None], value)
        quotient = tl.fma(error, inverse[:, None], quotient)
    return quotient

'''
out.write_text('"""Generated C4 scheduling and residual-division experiments."""\nimport triton\nimport triton.language as tl\n\n'+helper+'@triton.jit\n'+s+'\n',encoding='utf-8')
print(out)
