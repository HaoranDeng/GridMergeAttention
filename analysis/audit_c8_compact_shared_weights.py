"""Check compact SS-weight and N64-output geometry before GPU bitwise gates."""
import argparse
import collections
import json
from pathlib import Path


def weight_offset(row, key):
    return (row & 7)*8 + (key & 7) + (row >> 3)*640 + (key >> 3)*64


def audit():
    # Independent canonical 8x8 atoms, in K-major order inside each M8 slab.
    canonical = [((m8*8+r, k8*8+k), m8*640+k8*64+r*8+k)
                 for m8 in range(8) for k8 in range(10)
                 for r in range(8) for k in range(8)]
    assert {offset for _, offset in canonical} == set(range(5120))
    assert all(weight_offset(*coordinate) == offset for coordinate, offset in canonical)
    # Every packed RS pair maps once to the same logical matrix in SS storage.
    writes=[]
    for warp in range(4):
        for lane in range(32):
            for half in range(2):
                row=warp*16+(lane >> 2)+half*8
                for i in range(10):
                    key=8*i+2*(lane & 3)
                    address=weight_offset(row,key)
                    assert address % 2 == 0
                    assert weight_offset(row,key+1)==address+1
                    writes.extend((address,address+1))
    assert collections.Counter(writes)==collections.Counter(range(5120))
    # Descriptor LBO=128 B and SBO=1280 B, no swizzle, for each K16 step.
    for start in range(0,80,16):
        for row in range(64):
            for key in range(16):
                relative=(row & 7)*16+(row >> 3)*1280+(key >> 3)*128+(key & 7)*2
                assert 2*weight_offset(row,start+key)==2*weight_offset(0,start)+relative
    scratch={warp*2560+i*128+lane*4 for warp in range(4)
             for i in range(20) for lane in range(32)}
    assert scratch==set(range(0,10240,4))
    query={((row*64+(key & 63)+(key >> 6)*4096) ^
            (((row*64+(key & 63)+(key >> 6)*4096)>>6 & 7)<<3))
           for row in range(64) for key in range(128)}
    assert query==set(range(8192))
    assert 16384 < 18432
    # An output warp owns 16x64 BF16 words, swizzled at 128 B granularity.
    output={((row*64+d)^(((row*64+d)>>6 & 7)<<3))
            for row in range(16) for d in range(64)}
    assert output==set(range(1024))
    resources={}
    for groups in (3,4,6):
        frames=3 if groups==3 else groups//2
        columns=1 if groups==3 else 2
        visits=[(q,column) for wg in range(groups)
                for column in range(wg//frames,10,columns) for q in range(wg%frames,21,frames)]
        expected=[(q,column) for q in range(21) for column in range(10)]
        assert collections.Counter(visits)==collections.Counter(expected)
        for wg in range(groups):
            base=40960+wg*18432
            assert base % 128 == 0 and (base+10240) % 128 == 0
            assert base+10240+8192==40960+(wg+1)*18432
        resources[groups]={'dynamic_shared_bytes':40960+groups*18432,
                           'query_column_pairs':len(visits)}
    return {'passed':True,'exact_weight_elements':5120,'score_scratch_bytes':10240,
            'temporary_query_bytes':16384,'query_output_overlap_bytes':16384-10240,
            'output_bytes_per_wg':8192,'resources':resources,
            'scope':'Integer geometry only; GPU bitwise gates required for semantics and ordering.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    result=audit()
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
