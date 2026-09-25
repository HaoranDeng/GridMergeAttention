"""Derive current-frame C8 from the shipped r2 without changing its selector."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'kernels/best/c8_r2'
DEST = ROOT / 'kernels/experimental/c8_current'


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Expected one occurrence: {old!r}')
    return source.replace(old, new)


def build():
    DEST.mkdir(parents=True, exist_ok=True)
    for name, original in [('probe', 'gma_c8_cuda_probe'), ('selector', 'b1_gs_regpipe_v0'),
                           ('route', 'gma_c8_route_pipeline')]:
        source = (SOURCE / (original + '.cu')).read_text()
        # Head strides and public shapes, never the MMA's 40 accumulator lanes.
        source = re.sub(r'\*40\b', '*C8_HEADS', source)
        source = source.replace('dim3(945,40)', 'dim3(945,C8_HEADS)')
        source = source.replace('dim3(45*20,40)', 'dim3(45*20,C8_HEADS)')
        source = source.replace(',40,128}', ',C8_HEADS,128}').replace('<=40', '<=C8_HEADS')
        prefix = '// Derived from kernels/best/c8_r2/' + original + '.cu; see build_c8_current_sources.py.\n'
        prefix += '#ifndef C8_HEADS\n#define C8_HEADS 40\n#endif\n'
        (DEST / (name + '.cu')).write_bytes((prefix + source).replace('\n', '\r\n').encode())
    source = (SOURCE / 'gma_c8_d10a_fused_argmax_v0.cu').read_text()
    source = replace_once(source, 'for(int qo=0;qo<Q;++qo,qi0+=R*C*H*128,qi1+=R*C*H*128)',
                          'for(int qo=0;qo<Q;++qo)')
    source = replace_once(source, 'int pf=(qo<Q-1)?R*C*H*128:0;', '''// Prefetch the next frame's own selected query rows. RF0 is a control.
          int next_qo=qo<Q-1?qo+1:qo;
          int next_rf=RF_arg==1?0:next_qo;
          int next_si=((((next_rf*KO+ko)*R+kr)*C+col0)*H+gh)*8+seed_for_row(row0);
          int next_sr0=seeds[next_si],next_sr1=seeds[next_si+4];
          int next_qi0=(((next_qo*R+next_sr0)*C+col0)*H+gh)*128+(lane&3)*8;
          int next_qi1=(((next_qo*R+next_sr1)*C+col0)*H+gh)*128+(lane&3)*8;''')
    source = source.replace('q+qi0+pf+d', 'q+next_qi0+d').replace('q+qi1+pf+d', 'q+next_qi1+d')
    source = replace_once(source, 'r==45 && rf==1', 'r==45 && (rf==1 || rf==21)')
    source = source.replace('and one router frame', 'and one or 21 router frames')
    source = source.replace('// so every span [span,span+Q) shares one cb. Seed loads and all sr/cb-dependent\n  // address components hoist out of the qo loop; the qo term strength-reduces.',
                            '// so every span shares a column block. The prologue loads frame zero;\n  // each epilogue fetches the next frame-specific seed rows.')
    source = '// Derived r2 merge; only frame-specific seed addressing changes.\n' + source
    (DEST / 'merge.cu').write_bytes(source.replace('\n', '\r\n').encode())


if __name__ == '__main__':
    build()
