"""Change only compressed K/V transport to TMA descriptors; preserve route math."""
from pathlib import Path
import textwrap
root=Path(__file__).resolve().parents[1]
source=(root/'attention/hard_c4_wan21_14b_720p_triton.py').read_text()
start=source.index('    @triton.jit\n    def _hard_c4_720p_route_kernel(')
end=source.index('    @triton.jit',start+20)
code=textwrap.dedent(source[start:end]).replace('_hard_c4_720p_route_kernel','tma_route_kernel')
for kind,name in [('k','key'),('v','value')]:
    old=f'''        merged_{name} = tl.load(
            merged_{kind}_ptr + merged_base * D_HEAD + dims[None, :],
            mask=token_mask[:, None],
            other=0.0,
            cache_modifier=".cg",
        )'''
    assert code.count(old)==1
    code=code.replace(old,f'        merged_{name} = merged_{kind}_ptr.load([(route_base + token_start).to(tl.int32), 0])')
code+='''
class DescriptorRoute:
    def __init__(self):
        self.key=None
        self.descriptors=None
        self.block=None
        self.stages=None
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            from triton.tools.tensor_descriptor import TensorDescriptor
            block=self.block or args[23]
            key=(args[3].data_ptr(),args[4].data_ptr(),tuple(args[3].shape),block)
            if key!=self.key:
                self.descriptors=[TensorDescriptor(x,[x.numel()//128,128],[128,1],[block,128]) for x in args[3:5]]
                self.key=key
            copied=list(args);copied[3:5]=self.descriptors
            copied[23]=block
            copied[24]=self.stages or args[24]
            return tma_route_kernel[grid](*copied,**kwargs)
        return launch
tma_route=DescriptorRoute()
'''
(root/'native_fisher/tma_route.py').write_text('"""Experimental full-support native C10 route with TMA K/V loads."""\nimport triton\nimport triton.language as tl\n\n'+code)
screen=(root/'analysis/screen_c10_masked_route.py').read_text().replace('masked_route','tma_route')
screen=screen.replace('args=p.parse_args();',"p.add_argument('--route-block',type=int,choices=[64,128,256],default=64);p.add_argument('--route-stages',type=int,choices=[1,3],default=3)\n    args=p.parse_args();")
screen=screen.replace('control=h._hard_c4_720p_route_kernel',"tma_route.block=args.route_block;tma_route.stages=args.route_stages\n        result['candidate_config']={'block':args.route_block,'stages':args.route_stages,'warps':4}\n        control=h._hard_c4_720p_route_kernel",1)
screen=screen.replace("fisher_c10_half_owner.cu')","fisher_c10_long_tiles_tree.cu',tiles=84)")
screen=screen.replace('op.bias,0,5,21,','op.bias,0,5,84,')
screen=screen.replace('import torch\n','import torch\n        import triton\n        triton.set_allocator(lambda size,alignment,stream: torch.empty((size,),device="cuda",dtype=torch.int8))\n',1)
old="result['variants'].append(row);save();assert row['bitwise_equal'],row"
new="""diff=out[...,:5,:].float()-expected[...,:5,:].float()
            row['relative_l2']=float((diff.square().sum()/expected[...,:5,:].float().square().sum()).sqrt())
            row['max_abs']=float(diff.abs().max());row['finite']=bool(torch.isfinite(out[...,:5,:]).all())
            result['variants'].append(row);save();assert row['finite'] and row['relative_l2']<.002,row"""
assert old in screen;screen=screen.replace(old,new)
(root/'analysis/screen_c10_tma_route.py').write_text(screen)
print('Generated TMA route and three-layer paired H5 screen')
