"""Generate an exact E4M3-to-BF16 integer decoding route."""
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'native_fisher'
source = (root / 'fp8_via_float_route.py').read_text()
start = source.index('@triton.jit\ndef _via_float')
end = source.index('@triton.jit\ndef fp8_via_float_route', start)
decoder = '''@triton.jit
def _integer_decode(x):
    bits = x.to(tl.uint8, bitcast=True).to(tl.uint32)
    magnitude = bits & 127
    # Two byte permutations encode the eight subnormal BF16 bit patterns.
    subnormal = tl.inline_asm_elementwise(
        "{ .reg .b32 lo, hi; "
        "prmt.b32 lo, 0xc0800000, 0x60402000, $1; "
        "prmt.b32 hi, 0x3b3b3b00, 0x3c3c3c3c, $1; "
        "and.b32 lo, lo, 255; and.b32 hi, hi, 255; "
        "shl.b32 hi, hi, 8; or.b32 $0, lo, hi; }",
        constraints="=r,r", args=[magnitude & 7],
        dtype=tl.uint32, is_pure=True, pack=1)
    normal = (magnitude << 4) + 0x3c00
    value = tl.where(magnitude < 8, subnormal, normal)
    value = tl.where(magnitude == 127, 0x7fc0, value)
    return (value | ((bits & 128) << 8)).to(tl.uint16).to(tl.bfloat16, bitcast=True)


'''
source = source[:start] + decoder + source[end:]
source = source.replace('fp8_via_float_route', 'fp8_integer_route').replace('_via_float(', '_integer_decode(')
(root / 'fp8_integer_route.py').write_text(source)
