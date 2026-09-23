"""GPU-free checks of address maps, exact capacity, and warp ownership."""
import json
from pathlib import Path
import numpy as np


def test_packing():
    # This is a data permutation, so bitwise equality is required here.
    t,h,d = 2,3,128
    raw=np.arange(t*3600*h*d,dtype=np.int32).reshape(t,45,80,h,d)
    expected_q=raw.transpose(3,0,1,2,4).copy().reshape(-1)
    expected_k=raw.reshape(t,45,40,2,h,d).transpose(4,0,2,3,1,5).copy().reshape(-1)
    x=np.arange(raw.size,dtype=np.int64)
    dim=x%128;token=x//128;head=token//(t*3600);pos=token%(t*3600)
    src=(pos*h+head)*128+dim
    assert np.array_equal(raw.reshape(-1)[src],expected_q)
    frame=pos//3600;within=pos%3600;pair=within//90;j=within%90
    original=(frame*45+j%45)*80+pair*2+j//45
    assert np.array_equal(raw.reshape(-1)[(original*h+head)*128+dim],expected_k)
    pd=dim//32*32+(dim%8//2)*8+(dim%32//8)*2+dim%2
    packed=np.empty_like(expected_q);packed[token*128+pd]=expected_q
    # Independent WGMMA four-lane uint32 transpose description.
    expected_packed=expected_q.reshape(-1,4,4,2).transpose(0,2,1,3).copy().reshape(-1)
    assert np.array_equal(packed,expected_packed)
    # Route now writes [N,H,D] directly, without an output transpose.
    assert len(np.unique(src))==raw.size
    recovered=np.empty_like(expected_q);recovered[src]=expected_q
    assert np.array_equal(recovered,raw.reshape(-1))


def test_output_coverage():
    for cap in (6,8):
        for seed_tile in (8,16):
            groups=64//seed_tile;column_tiles=(45+groups-1)//groups
            coverage=np.zeros((45,cap,128),dtype=np.uint8)
            # WGMMA accumulator register mapping, independent of scalar
            # source substitutions. Verify every output is written once.
            for cb in range(column_tiles):
                for tid in range(128):
                    warp,lane=divmod(tid,32);seed=lane//4
                    for i in range(0,64,2):
                        row=warp*16+seed+8*((i&3)//2)
                        dim=2*(lane&3)+8*(i//4)
                        col=cb*groups+row//seed_tile;c=row%seed_tile
                        if col<45 and c<cap:
                            coverage[col,c,dim:dim+2]+=1
            assert np.all(coverage==1),(cap,seed_tile,np.unique(coverage))
            for tiles in (1,21,42,63,126,252):
                visited=[tile for block in range((21*column_tiles+tiles-1)//tiles)
                         for tile in range(block*tiles,min((block+1)*tiles,21*column_tiles))]
                assert visited==list(range(21*column_tiles))


def test_winner_reduction():
    rng=np.random.default_rng(9045)
    cases=[rng.normal(size=(64,96)).astype(np.float32),np.zeros((64,96),np.float32)]
    # Invalid C6 rows would otherwise win; force an adversarial pattern.
    adversarial=np.full((64,96),-100,np.float32)
    adversarial[np.arange(64)%8>=6]=100
    cases.append(adversarial)
    for cap in (6,8):
        for st in (8,16):
            for scores in cases:
                for row in range(64):
                    warp=row//16;half=row%16//8;group=row//st
                    values=np.full((8,2,96),-np.inf,np.float32)
                    for seed in range(8):
                        if st==8:
                            if seed<cap:values[seed,0]=scores[warp*16+seed+8*half]
                        else:
                            if seed<cap:values[seed,0]=scores[warp*16+seed]
                            if seed+8<cap:values[seed,1]=scores[warp*16+seed+8]
                    best=values.max(axis=(0,1))
                    labels=np.where(values==best,np.arange(16).reshape(2,8).T[:,:,None],16).min(axis=(0,1))
                    expected=scores[group*st:group*st+cap].argmax(axis=0)
                    assert np.array_equal(labels,expected)


if __name__=='__main__':
    checks={}
    for test in (test_packing,test_output_coverage,test_winner_reduction):
        test();checks[test.__name__]='passed'
    result=dict(gpu_used=False,checks=checks,
                scope='CPU address algebra and ownership checks; no compiled CUDA execution')
    Path(__file__).with_name('cpu_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
