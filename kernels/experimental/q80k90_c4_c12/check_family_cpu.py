"""Verify native row coverage and strided warp ownership without a GPU."""
import ast
import json
from pathlib import Path
import numpy as np


def main():
    rng=np.random.default_rng(9045)
    checked=[]
    for cap in (4,6,8,10,12):
        for tile in (4,8,16):
            if cap>tile:continue
            groups=64//tile
            coverage=np.zeros((45,cap,128),np.uint8)
            for block in range((45+groups-1)//groups):
                for tid in range(128):
                    warp,lane=divmod(tid,32)
                    for i in range(0,64,2):
                        row=warp*16+lane//4+8*((i&3)//2)
                        col=block*groups+row//tile;seed=row%tile
                        d=2*(lane&3)+8*(i//4)
                        if col<45 and seed<cap:coverage[col,seed,d:d+2]+=1
            assert np.all(coverage==1),(cap,tile)
            scores=rng.normal(size=(64,96)).astype(np.float32)
            scores[:,0]=0; scores[1::2,0]=-0.
            scores[:,1]=-np.inf
            for row in range(64):
                if row%tile>=cap:scores[row]=1e6
            for warp in range(4):
                for half in range(2):
                    for key in range(4):
                        for kc in range(key,96,4):
                            low=np.full(8,-np.inf);high=np.full(8,-np.inf)
                            for seed in range(8):
                                if tile<=8:
                                    if seed%tile<cap:low[seed]=scores[warp*16+seed+8*half,kc]
                                else:
                                    if seed<cap:low[seed]=scores[warp*16+seed,kc]
                                    if seed+8<cap:high[seed]=scores[warp*16+seed+8,kc]
                            for segment in range(0,8,4 if tile==4 else 8):
                                count=4 if tile==4 else 8
                                best=max(low[segment:segment+count].max(),high[segment:segment+count].max())
                                lo_mask=sum(1<<(4*s+key) for s in range(segment,segment+count) if low[s]==best)
                                hi_mask=sum(1<<(4*s+key) for s in range(segment,segment+count) if high[s]==best)
                                ffs=lambda x:(x&-x).bit_length()-1
                                label=((ffs(lo_mask)//4)%tile) if lo_mask else 8+ffs(hi_mask)//4
                                row=warp*16+segment+8*half;group=row//tile
                                expected=int(scores[group*tile:group*tile+cap,kc].argmax())
                                assert label==expected,(cap,tile,row,kc,label,expected)
            checked.append(dict(capacity=cap,seed_tile=tile,coverage=True,ballot_ties=True))
    for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    report=dict(passed=True,gpu_used=False,cases=checked)
    Path(__file__).with_name('cpu_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
