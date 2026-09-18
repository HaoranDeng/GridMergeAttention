"""Generate C4 route variants that differ from the retained route only in L2 hints.

The launched route kernel loads the one-shot 8.13 GB compressed streams (mk,
mv, bias) and the L2-reusable singleton raw K/V with the identical policy:
`cache_modifier=".cg"` and no eviction policy. Nothing tells L2 which lines to
keep. An eviction hint changes no value - it lowers to `createpolicy` plus a
`ld.global.L2::cache_hint`, the same mechanism the CUDA merge already uses -
so every variant here must stay bitwise equal to the retained route, and the
screen checks that.

Variants (each written as a complete copy of route_reference.py so it loads as
its own module and compiles to its own Triton cache entry):
  null            byte-identical copy: the A/A arm
  stream_first    evict_first on the mk / mv / bias stream
  singleton_last  evict_last on the singleton raw K / V loads
  both            both of the above
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'kernels/best/c4/route_reference.py'
OUT = ROOT / 'native_fisher'

MK = ('merged_k_ptr + merged_base * D_HEAD + dims[None, :],\n'
      '                mask=token_mask[:, None],\n'
      '                other=0.0,\n'
      '                cache_modifier=".cg",\n'
      '            )')
MV = MK.replace('merged_k_ptr', 'merged_v_ptr')
BIAS = ('bias_ptr + route_base + tokens,\n'
        '                mask=token_mask,\n'
        '                other=float("inf"),\n'
        '                cache_modifier=".cg",\n'
        '            )')
SK = ('k_ptr + singleton_address,\n'
      '                    mask=token_mask[:, None],\n'
      '                    other=0.0,\n'
      '                    cache_modifier=".cg",\n'
      '                )')
SV = SK.replace('k_ptr + singleton_address', 'v_ptr + singleton_address')


def hinted(pattern, policy):
    indent = pattern.rsplit('\n', 1)[0].rsplit('\n', 1)[1]
    indent = indent[:len(indent) - len(indent.lstrip())]
    return pattern.replace('cache_modifier=".cg",\n',
                           'cache_modifier=".cg",\n%seviction_policy="%s",\n' % (indent, policy))


def main():
    text = SRC.read_text(encoding='utf-8')
    start = text.index('def _hard_c4_720p_route_kernel')
    end = text.index('def _hard_c4_720p_query_column_deferred_route_kernel')
    head, body, tail = text[:start], text[start:end], text[end:]

    def patched(edits):
        out = body
        for pattern, policy in edits:
            assert out.count(pattern) == 1, pattern[:40]
            out = out.replace(pattern, hinted(pattern, policy))
        return head + out + tail

    stream = [(MK, 'evict_first'), (MV, 'evict_first'), (BIAS, 'evict_first')]
    singleton = [(SK, 'evict_last'), (SV, 'evict_last')]
    variants = {'null': text,
                'stream_first': patched(stream),
                'singleton_last': patched(singleton),
                'both': patched(stream + singleton)}
    for name, content in variants.items():
        path = OUT / ('route_c4_hint_%s.py' % name)
        path.write_text(content, encoding='utf-8', newline='\n')
        changed = sum(1 for a, b in zip(text.split('\n'), content.split('\n')) if a != b) \
            + abs(content.count('\n') - text.count('\n'))
        print(name, path.name, 'sha', hashlib.sha256(path.read_bytes()).hexdigest()[:16],
              'lines changed/added', changed, 'eviction sites', content.count('eviction_policy='))
    assert hashlib.sha256((OUT / 'route_c4_hint_null.py').read_bytes()).hexdigest() \
        == hashlib.sha256(SRC.read_bytes()).hexdigest(), 'null copy must be byte-identical'


if __name__ == '__main__':
    main()
