"""Only CTA visitation order changes; every query/key/head pair remains."""
from pathlib import Path
import argparse
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--source',default='fisher_c10_half_owner.cu')
p.add_argument('--prefix',default='fisher_c10')
args=p.parse_args()
s=(root/'native_fisher'/args.source).read_text()
s=s.replace('  int ko=block%KO,kr=block/KO;','  int ko=block/R,kr=block%R;')
(root/'native_fisher'/f'{args.prefix}_key_order.cu').write_text(s)
s=s.replace('int block=blockIdx.x,head=blockIdx.y,gh=HS+head;','int block=blockIdx.x/AH,head=blockIdx.x%AH,gh=HS+head;')
s=s.replace('dim3 grid(ko*r,ah,','dim3 grid(ko*r*ah,1,')
(root/'native_fisher'/f'{args.prefix}_key_head_order.cu').write_text(s)
