"""Force FP8-to-FP32 vector conversion before exact BF16 narrowing."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'native_fisher'
s=(root/'fp8_direct_route.py').read_text()
s=s.replace('def fp8_direct_route(', 'def fp8_via_float_route(')
s=s.replace('merged_key=merged_key.to(query.dtype)','merged_key=_via_float(merged_key)')
s=s.replace('merged_value=merged_value.to(query.dtype)','merged_value=_via_float(merged_value)')
helper='''
@triton.jit
def _via_float(x):
    return tl.inline_asm_elementwise(
        "cvt.rn.bf16.f32 $0, $1;", constraints="=h,f",
        args=[x.to(tl.float32)], dtype=tl.bfloat16, is_pure=True, pack=1)

'''
s=s.replace('@triton.jit\ndef fp8_via_float_route',helper+'@triton.jit\ndef fp8_via_float_route')
(root/'fp8_via_float_route.py').write_text(s)
