"""Storage-only E4M3 route: identical BF16 dots, direct FP8-to-BF16 casts."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'attention/hard_c4_wan21_14b_720p_triton.py').read_text()
start=s.index('    @triton.jit\n    def _hard_c4_720p_route_kernel(')
end=s.index('\n    @triton.jit',start+20)
body=s[start:end]
body='\n'.join(line[4:] if line.startswith('    ') else line for line in body.splitlines())
body=body.replace('def _hard_c4_720p_route_kernel(', 'def fp8_direct_route(')
needle='        if SCALED_FP8_KV:\n'
assert body.count(needle)==1
body=body.replace(needle,'        merged_key=merged_key.to(query.dtype)\n        merged_value=merged_value.to(query.dtype)\n'+needle)
(root/'native_fisher/fp8_direct_route.py').write_text('"""Generated direct E4M3 storage route, BF16 arithmetic."""\nimport triton\nimport triton.language as tl\n\n'+body+'\n')
