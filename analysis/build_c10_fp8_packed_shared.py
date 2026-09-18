"""Combine adjacent FP8 halfword stores into one aligned shared word store.

Only the TMA staging epilogue changes. Even lanes collect the adjacent lane's
two bytes without converting them; global tail stores and all math stay exact.
This is separate from the rejected vector global-store experiment.
"""
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'native_fisher'
source = (root / 'fisher_c10_fp8_shared_store.cu').read_text()
old = '        *reinterpret_cast<uint16_t*>(fp8temp+(offset^(((offset>>7)&7)<<4)))=bits;'
assert source.count(old) == 1
source = source.replace(old, '''        uint32_t adjacent=__shfl_xor_sync(0xffffffff,uint32_t(bits),1);
        uint32_t word=uint32_t(bits)|(adjacent<<16);
        if((lane&1)==0)
          *reinterpret_cast<uint32_t*>(fp8temp+(offset^(((offset>>7)&7)<<4)))=word;''')
(root / 'fisher_c10_fp8_packed_shared.cu').write_text(source)
