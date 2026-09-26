"""Q80K90 exact even capacities 4, 6, 8, 10: RF0 two-intersection Fisher, native kernels.

Port of the Q90K80 even-capacity kernels (analysis/q90k80_even_*) to the Q80K90
grid of the 21x45x80 latent video: a query group is one row of a frame (80
consecutive tokens, 45 groups per frame) and a key block is two adjacent columns
(90 keys, 40 blocks per frame); key j of block (frame, pair) is the token at row
j%45, column 2*pair+j//45. The operator is the one of
kernels/experimental/q80k90_c4_c12 (same candidate table, RF0 Fisher selector
with the two row/column-pair intersections as the first two seeds, ownership
merge, entropy bias): exactly CAP representatives per (query group, key block),
empty representatives have zero K/V and -inf bias, no singleton or residual
route.

Layouts: seeds [h, 45 row, 840 kb, CAP] (as the v4 kernels); merged K/V
[ah, 840 kb, 945 group, CAP, 128] (group = frame*45 + row, so a compact merge
tile is one contiguous run of rows); bias [ah, 945 group, 840 kb, CAP]
(group-major, what the route reads; identical to the v4 bias layout). A query
group is 80 consecutive tokens of the caller's [1, N, H, 128] tensors, so the
selector, merge and route read Q, K and V in place, through their token and
head strides: token-major storage (contiguous [1, N, H, 128]) and head-major
storage (a transposed view of a contiguous [1, H, N, 128], as attention
processors hold it) are both read without a copy. The output is a new
token-major [1, N, H, 128] tensor; the only copy is the merge's
fragment-ordered Q.
"""
import os
import ctypes,functools,hashlib
from pathlib import Path
import torch

def _native(stem,flags):
    from torch.utils.cpp_extension import load
    ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libstdc++.so.6',mode=ctypes.RTLD_GLOBAL)
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST','9.0a');os.environ.setdefault('MAX_JOBS','2')
    os.environ.setdefault('TORCH_EXTENSIONS_DIR','/local2/denghaoran/GridMergeAttention/q80even_cuda_extensions_20260925')
    source=Path(__file__).with_name(f'q80k90_even_{stem}.cu')
    flags=['-O3','--expt-relaxed-constexpr','-lineinfo']+flags
    digest=hashlib.sha256(source.read_bytes()+' '.join(flags).encode()).hexdigest()[:16]
    cutlass=Path(torch.__file__).resolve().parent.parent/'flashinfer/data/cutlass/include'
    return load(name=f'q80even_{stem}_'+digest,sources=[str(source)],extra_include_paths=[str(cutlass)],extra_cuda_cflags=flags,extra_ldflags=['-lcuda'],verbose=False)

def _shape_flags(t,h,ah,cap):
    return [f'-DAXIS_OUTER={t}',f'-DAXIS_HEADS={h}',f'-DAXIS_ACTIVE_HEADS={ah}',f'-DAXIS_CAP={cap}']

@functools.lru_cache(maxsize=32)
def native_merge(t,h,ah,cap,packed=1,minblocks=3,extra=()):
    return _native('merge',_shape_flags(t,h,ah,cap)+['-DAXIS_FAST_MATH=1',f'-DMERGE_PACKED={int(packed)}',f'-DMERGE_MINBLOCKS={int(minblocks)}']+list(extra))

@functools.lru_cache(maxsize=32)
def native_route(t,h,ah,cap,extra=()):
    return _native('route',_shape_flags(t,h,ah,cap)+list(extra))

@functools.lru_cache(maxsize=8)
def native_layout(t,h):
    return _native('layout',[f'-DAXIS_OUTER={t}',f'-DAXIS_HEADS={h}'])

@functools.lru_cache(maxsize=32)
def native_select(t,h,cap,extra=()):
    return _native('select',[f'-DAXIS_OUTER={t}',f'-DAXIS_HEADS={h}',f'-DAXIS_CAP={cap}']+list(extra))


def candidate_table():
    """The v4 Q80K90 table: 32 candidate columns of a query row per column pair,
    always containing the two intersection columns 2*pair and 2*pair+1."""
    def original_pool(anchor):
        chosen={anchor}
        chosen.update(range(max(0,anchor-11),min(80,anchor+12)))
        chosen.update(round(i*79/10) for i in range(11))
        while len(chosen)<32:
            chosen.add(max(set(range(80))-chosen,key=lambda x:(min(abs(x-y) for y in chosen),-x)))
        return chosen
    tables=[]
    for group in range(40):
        first=2*group; second=first+1
        chosen={first,second};pool=original_pool(first)|chosen
        if len(pool)<=32:chosen=pool.copy()
        while len(chosen)<32:
            available=pool-chosen
            if not available:available=set(range(80))-chosen
            chosen.add(max(available,key=lambda x:(min(abs(x-y) for y in chosen),-x)))
        tables.append(sorted(chosen))
    return tables


def input_strides(x):
    """(tensor, (token stride, head stride)) of a [1, N, H, 128] input without copying it when
    it is token-major or head-major dense; anything else is made token-major contiguous."""
    n,h=x.shape[1],x.shape[2]
    if x.is_contiguous():return x,(h*128,128)
    if x.transpose(1,2).is_contiguous():return x,(128,n*128)
    return x.contiguous(),(h*128,128)


def groups_per_tile(cap,packed=1):
    lp=cap//2
    if int(packed)==2:return 2*(16//lp)
    return 32//lp if packed else 4*(8//lp)

# Merge tile slices per (key block, head) = CTAs sharing one key block's tiles.
# Interleaved A/B on synthetic QKV: C4 three slices (20 tiles/CTA) -0.77 ms
# against four (18/20 rounds), C8 four slices best (three neutral).
SLICES={4:3,6:4,8:4,10:4}

def merge_tiles(t,cap,packed=1):
    g=groups_per_tile(cap,packed)
    return (t*45+g-1)//g


class Q80K90Even:
    """Exact-capacity Q80K90 operator for cap in (4, 6, 8, 10), native kernels only.

    native_tiles is the number of merge tiles per CTA (None = SLICES[cap]
    slices of the tile range). head_chunk=8 (default) keeps the merged K/V of eight heads
    live: 2 x cap x 1.63 GB of workspace (C10: 32.6 GB). merge_packed: 1 packs
    groups across warps (default), 0 keeps every group inside one warp, 2 packs
    per warp pair. Every call returns a new token-major output tensor; the
    workspace is reusable on one stream (use one instance per concurrent
    stream). Inputs may be token-major or head-major (see input_strides)."""
    def __init__(self,cap,head_chunk=8,native_tiles=None,merge_packed=1,merge_minblocks=3,merge_flags=(),route_flags=(),select_flags=()):
        assert cap in (4,6,8,10)
        self.cap=cap;self.hc=int(head_chunk);self.nt=native_tiles
        self.packed=int(merge_packed);self.minblocks=int(merge_minblocks)
        self.merge_flags=tuple(merge_flags);self.route_flags=tuple(route_flags);self.select_flags=tuple(select_flags)
        assert self.hc>0 and self.packed in (0,1,2)
        self.cache=None

    def prepare(self,q):
        b,n,h,d=q.shape;t=n//3600
        assert b==1 and n==t*3600 and d==128 and q.dtype==torch.bfloat16 and q.is_cuda
        key=(t,h,q.device)
        if self.cache!=key:
            ah=min(self.hc,h);cap=self.cap
            assert h%ah==0,'the native kernels are compiled for one active-head count'
            self.table=torch.tensor(candidate_table(),device=q.device,dtype=torch.int32)
            self.seeds=torch.empty((h,45,t*40,cap),device=q.device,dtype=torch.uint8)
            self.mk=torch.empty((ah,t*40,t*45,cap,128),device=q.device,dtype=torch.bfloat16);self.mv=torch.empty_like(self.mk)
            self.bias=torch.empty((ah,t*45,t*40,cap),device=q.device,dtype=torch.float32)
            self.packed_q=torch.empty((h,n,128),device=q.device,dtype=torch.bfloat16)
            self.tiles=self.nt if self.nt is not None else max(1,(merge_tiles(t,cap,self.packed)+SLICES[cap]-1)//SLICES[cap])
            self.cache=key
        return t,h

    def merge_module(self,t,h,ah):
        return native_merge(t,h,ah,self.cap,self.packed,self.minblocks,self.merge_flags)

    @torch.no_grad()
    def __call__(self,q,k,v,return_debug=False):
        t,h=self.prepare(q);cap=self.cap
        assert k.shape==v.shape==q.shape and k.dtype==v.dtype==q.dtype and k.device==v.device==q.device
        (q,qs),(k,ks),(v,vs)=input_strides(q),input_strides(k),input_strides(v)
        with torch.cuda.device(q.device):
            native_layout(t,h).forward(q,self.packed_q,*qs)
            native_select(t,h,cap,self.select_flags).run(q,k,self.table,self.seeds,*qs,*ks)
            out=torch.empty((1,t*3600,h,128),device=q.device,dtype=q.dtype)
            for hs in range(0,h,self.hc):
                ah=min(self.hc,h-hs)
                self.merge_module(t,h,ah).run(self.packed_q,k,v,self.seeds,self.mk,self.mv,self.bias,hs,ah,self.tiles,*ks,*vs)
                native_route(t,h,ah,cap,self.route_flags).run(q,self.mk,self.mv,self.bias,out,hs,ah,*qs)
        return (out,self.seeds) if return_debug else out


def Q80K90C4(**kw):return Q80K90Even(4,**kw)
def Q80K90C6(**kw):return Q80K90Even(6,**kw)
def Q80K90C8(**kw):return Q80K90Even(8,**kw)
def Q80K90C10(**kw):return Q80K90Even(10,**kw)
