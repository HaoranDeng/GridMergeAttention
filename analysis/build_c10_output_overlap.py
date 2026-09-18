"""Separate owner scratch from one/two asynchronous TMA output buffers."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
original = (ROOT / 'native_fisher/fisher_c10_half_owner.cu').read_text()
for buffers in (1, 2):
    s = original.replace('__launch_bounds__(128,4)', f'__launch_bounds__(128,{3 if buffers == 1 else 2})')
    s = s.replace('float* normalized=reinterpret_cast<float*>(temp);',
                  f'float* normalized=reinterpret_cast<float*>(sv+80*128+{buffers}*64*128);')
    # Owner data never aliases a pending asynchronous output transfer.
    s = s.replace('    if(tid==0)asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");\n', '', 1)
    s = s.replace('(80*128*2)*2+64*128*2', f'(80*128*2)*2+{buffers}*64*128*2+12288')
    if buffers == 2:
        s = s.replace('    for(int kind=0;kind<2;++kind) {',
                      '    for(int kind=0;kind<2;++kind) {\n      temp=sv+80*128+kind*64*128;')
        # Reuse the buffer from two commits ago; the newest opposite-kind
        # transfer can continue while this buffer is repopulated.
        s = s.replace('        if(tid==0) asm volatile("cp.async.bulk.wait_group.read 0;',
                      '        if(tid==0) asm volatile("cp.async.bulk.wait_group.read 1;')
    path = ROOT / f'native_fisher/fisher_c10_output_buffers{buffers}.cu'
    path.write_text(s)
    print(path)
