"""Independent integer-layout checks for multi-WG scheduling and N40 scores.

These check index geometry; GPU bitwise gates remain required for instruction
semantics, memory ordering and the complete attention call.
"""
import argparse
import collections
import json
from pathlib import Path


def boff(key, dimension):
    offset=key*64+(dimension&63)+(dimension>>6)*5120
    return offset^(((offset>>6)&7)<<3)


def audit():
    expected={(q,c) for q in range(21) for c in range(10)}
    five=[]
    for start in (0,105):
        for wg in range(5):
            five.extend((q,span//21) for span in range(start+wg*21,start+105,5*21)
                        for q in range(21))
    six=[(q,span//21) for wg in range(6) for span in range((wg//3)*21,210,42)
         for q in range(wg%3,21,3)]
    four=[(q,span//21) for wg in range(4) for span in range((wg//2)*21,210,42)
          for q in range(wg%2,21,2)]
    for visits in (four,five,six):
        assert set(visits)==expected
        assert all(count==1 for count in collections.Counter(visits).values())
    fragments=[]
    scratch=set()
    for warp in range(4):
        for lane in range(32):
            for i in range(20):
                address=warp*4096+i*128+lane*4
                assert warp*4096<=address and address+4<=(warp+1)*4096
                assert address not in scratch
                scratch.add(address)
                for half in range(2):
                    row=warp*16+(lane>>2)+8*((i&3)>>1)
                    key=2*(lane&3)+8*(i>>2)+(i&1)+half*40
                    old_i=i+half*20
                    assert row==warp*16+(lane>>2)+8*((old_i&3)>>1)
                    assert key==2*(lane&3)+8*(old_i>>2)+(old_i&1)
                    fragments.append((row,key))
    assert len(set(fragments))==len(fragments)==64*80
    for half in range(2):
        for key in range(40):
            for dimension in range(128):
                assert boff(half*40+key,dimension)==(
                    boff(half*40,dimension)+boff(key,dimension)-boff(0,dimension))
    return dict(four_wg_unique_query_column_pairs=len(four),
                five_wg_unique_query_column_pairs=len(five),
                six_wg_unique_query_column_pairs=len(six),
                exact_score_elements=len(fragments),
                scratch_fp32_elements=len(scratch),
                scratch_bytes_per_warp=20*32*4,
                descriptor_offset_pairs_checked=2*40*128,
                passed=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    a=p.parse_args()
    result=audit()
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
