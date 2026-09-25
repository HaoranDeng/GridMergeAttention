"""Exact C4 on directed THW axes, with current-line anchor-Fisher seeds.

The operator matches analysis.quick_axis_c4_qkv: every key belongs to one
of four merged slots, empty slots are disabled, and all key groups participate.
No singleton, shared RF0 selector, learned state, or dense fallback is used.
"""
from functools import lru_cache
import os
import torch
import triton
import triton.language as tl

from analysis.quick_axis_c4_qkv import candidate_table
from kernels.best.c4.route_reference import _hard_c4_720p_route_kernel


@triton.jit
def _probe_logits(Q, K, Z, O:tl.constexpr, R:tl.constexpr, L:tl.constexpr,
                  H:tl.constexpr, AH:tl.constexpr, HS:tl.constexpr):
    qg = tl.program_id(0)*4 + tl.arange(0,4)
    kg = tl.program_id(1)*4 + tl.arange(0,4)
    bh = tl.program_id(2)
    b, h = bh//AH, bh%AH
    rows = tl.arange(0,32)
    probes = tl.arange(0,16)
    d = tl.arange(0,128)
    qt = ((b*O+qg[:,None,None]//L)*R+rows[None,:,None])*L+qg[:,None,None]%L
    q = tl.load(Q+(qt*H+HS+h)*128+d[None,None,:], (qg[:,None,None]<O*L)&(rows[None,:,None]<R),0)
    kt = b*O*R*L+kg[:,None,None]*L+probes[None,:,None]*L//16
    k = tl.load(K+(kt*H+HS+h)*128+d[None,None,:],kg[:,None,None]<O*R,0)
    z = tl.dot(tl.reshape(q,(128,128)),tl.trans(tl.reshape(k,(64,128))))*(128**-.5)
    z = tl.trans(tl.reshape(z,(4,32,4,16)),(0,2,1,3))
    offset = (((b.to(tl.int64)*O*L+qg[:,None,None,None])*AH+h)*(O*R)+kg[None,:,None,None])*512+rows[None,None,:,None]*16+probes[None,None,None,:]
    tl.store(Z+offset,z,(qg[:,None,None,None]<O*L)&(kg[None,:,None,None]<O*R))


@triton.jit
def _merge(Q, K, V, Candidates, Anchors, MK, MV, Bias, Seeds, ProbeLogits,
           O: tl.constexpr, R: tl.constexpr, L: tl.constexpr,
           H: tl.constexpr, AH: tl.constexpr, HS: tl.constexpr,
           NC: tl.constexpr, BJ: tl.constexpr, BL: tl.constexpr,
           AUDIT: tl.constexpr, FISHER_FP64: tl.constexpr, PRECOMPUTE:tl.constexpr):
    # Q/K/V are contiguous [batch, outer, query-axis, key-axis, head, 128].
    colblock = tl.program_id(0)
    kg = tl.program_id(1)
    bh = tl.program_id(2)
    b, h = bh // AH, bh % AH
    kr = kg % R
    j = colblock * BJ + tl.arange(0, BJ)
    c = tl.arange(0, 32)
    p = tl.arange(0, 16)
    d = tl.arange(0, 128)
    ks = tl.arange(0, BL)
    rows = tl.load(Candidates + kr * NC + c, c < NC, 0)
    anchor = tl.load(Anchors + kr)
    probes = p * L // 16
    kb = ((b * O * R * L + kg * L + ks[:, None]) * H + HS + h) * 128 + d[None, :]
    key = tl.load(K + kb, ks[:, None] < L, 0)
    value = tl.load(V + kb, ks[:, None] < L, 0)
    kp = ((b * O * R * L + kg * L + probes[:, None]) * H + HS + h) * 128 + d[None, :]
    if not PRECOMPUTE:
        probe_key = tl.load(K + kp)
    for qo in range(O):
        if PRECOMPUTE:
            offset = ((((b.to(tl.int64)*O+qo)*L+j[:,None,None])*AH+h)*(O*R)+kg)*512+c[None,:,None]*16+p[None,None,:]
            z = tl.load(ProbeLogits+offset,j[:,None,None]<L,0.)
        else:
            qb = (((b * O + qo) * R + rows[None, :, None]) * L + j[:, None, None]) * H + HS + h
            query_candidates = tl.load(Q + qb * 128 + d[None, None, :],
                                       (j[:, None, None] < L) & (c[None, :, None] < NC), 0)
            z = tl.dot(tl.reshape(query_candidates, (BJ * 32, 128)), tl.trans(probe_key))
            z = tl.reshape(z * (128 ** -.5), (BJ, 32, 16))
        if FISHER_FP64:
            z = z.to(tl.float64)
        za = tl.sum(tl.where(c[None, :, None] == anchor, z, 0.), 1)
        rho = tl.exp((za - tl.max(za, 1)[:, None]).to(tl.float32))
        if FISHER_FP64:
            rho = rho.to(tl.float64)
        rho = rho / tl.sum(rho, 1)[:, None]
        delta = z - za[:, None, :]
        res = (delta - tl.sum(delta * rho[:, None, :], 2)[:, :, None]) * tl.sqrt(rho[:, None, :])
        used = (c[None, :] == anchor) | (c[None, :] >= NC)
        s1 = tl.full((BJ,), 0, tl.int32)
        s2 = tl.full((BJ,), 0, tl.int32)
        s3 = tl.full((BJ,), 0, tl.int32)
        for step in tl.static_range(1, 4):
            score = tl.where(used, -float('inf'), tl.sum(res * res, 2))
            pick = tl.argmax(score, 1, tie_break_left=True)
            selected = tl.sum(tl.where(c[None, :] == pick[:, None], rows[None, :], 0), 1)
            if step == 1:
                s1 = selected
            elif step == 2:
                s2 = selected
            else:
                s3 = selected
            used = used | (c[None, :] == pick[:, None])
            if step < 3:
                direction = tl.sum(tl.where(c[None, :, None] == pick[:, None, None], res, 0.), 1)
                basis = direction / tl.sqrt(tl.sum(direction * direction, 1) + 1.e-12)[:, None]
                res = res - tl.sum(res * basis[:, None, :], 2)[:, :, None] * basis[:, None, :]
        slots = tl.arange(0, 4)
        sr = tl.where(slots[None, :] == 0, kr,
                      tl.where(slots[None, :] == 1, s1[:, None],
                               tl.where(slots[None, :] == 2, s2[:, None], s3[:, None])))
        sqb = (((b * O + qo) * R + sr[:, :, None]) * L + j[:, None, None]) * H + HS + h
        sq = tl.load(Q + sqb * 128 + d[None, None, :], j[:, None, None] < L, 0)
        logits = tl.reshape(tl.dot(tl.reshape(sq, (BJ * 4, 128)), tl.trans(key)) * (128 ** -.5), (BJ, 4, BL))
        logits = tl.where(ks[None, None, :] < L, logits, -float('inf'))
        zmax = tl.max(logits, 2)
        lse = zmax + tl.log(tl.sum(tl.exp(logits - zmax[:, :, None]), 2))
        owner = tl.argmax(logits - lse[:, :, None], 1, tie_break_left=True)
        assigned = (slots[None, :, None] == owner[:, None, :]) & (ks[None, None, :] < L)
        active = tl.sum(assigned.to(tl.int32), 2) > 0
        masked = tl.where(assigned, logits, -float('inf'))
        maximum = tl.where(active, tl.max(masked, 2), 0.)
        weights = tl.exp(masked - maximum[:, :, None])
        denom = tl.sum(weights, 2)
        weights = weights / tl.maximum(denom[:, :, None], 1.e-30)
        negentropy = tl.sum(weights * tl.log(tl.maximum(weights, 1.e-30)), 2)
        bias = tl.where(active, negentropy, float('inf'))
        wb = tl.reshape(weights, (BJ * 4, BL)).to(tl.bfloat16)
        mk = tl.reshape(tl.dot(wb, key), (BJ, 4, 128))
        mv = tl.reshape(tl.dot(wb, value), (BJ, 4, 128))
        base = (((b.to(tl.int64) * O + qo) * L + j[:, None]) * AH + h) * (O * R * 4) + kg * 4 + slots[None, :]
        tl.store(MK + base[:, :, None] * 128 + d[None, None, :], mk, j[:, None, None] < L)
        tl.store(MV + base[:, :, None] * 128 + d[None, None, :], mv, j[:, None, None] < L)
        tl.store(Bias + base, bias, j[:, None] < L)
        if AUDIT:
            tl.store(Seeds + base, sr, j[:, None] < L)


@lru_cache(maxsize=16)
def _tables(r, device):
    table, anchors = candidate_table(r)
    return (torch.tensor(table, dtype=torch.int32, device=device),
            torch.tensor(anchors, dtype=torch.int32, device=device), len(table[0]))


def merged_attention(q, k, v, *, head_start=0, head_count=None, audit=False, output=None):
    """Generic relaid-out grid; useful for numerical validation as well."""
    b, o, r, l, h, d = q.shape
    assert d == 128 and q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert all(x.is_contiguous() for x in (q, k, v)) and l >= 16 and r >= 4
    ah = head_count or h
    candidates, anchors, nc = _tables(r, str(q.device))
    shape = (b, o, l, ah, o, r, 4)
    mk = torch.empty((*shape, d), dtype=q.dtype, device=q.device)
    mv = torch.empty_like(mk)
    bias = torch.empty(shape, dtype=torch.float32, device=q.device)
    seeds = torch.empty(shape, dtype=torch.int32, device=q.device) if audit else bias
    precompute = r <= 32 and os.environ.get('AXIS_C4_PRECOMPUTE','0') == '1'
    probe_logits = torch.empty((b,o*l,ah,o*r,32,16),dtype=torch.float32,device=q.device) if precompute else bias
    if precompute:
        _probe_logits[(triton.cdiv(o*l,4),triton.cdiv(o*r,4),b*ah)](
            q,k,probe_logits,o,r,l,h,ah,head_start,num_warps=4,num_stages=1)
    bj = int(os.environ.get('AXIS_C4_BLOCK_J', '4'))
    _merge[(triton.cdiv(l, bj), o * r, b * ah)](
        q, k, v, candidates, anchors, mk, mv, bias, seeds, probe_logits,
        o, r, l, h, ah, head_start, nc, bj, triton.next_power_of_2(l), audit,
        os.environ.get('AXIS_C4_FISHER_FP64', '0') == '1', precompute,
        num_warps=4 if bj <= 4 else 8, num_stages=1, enable_fp_fusion=False)
    out = torch.empty_like(q) if output is None else output
    _hard_c4_720p_route_kernel[(1, b * ah, o * l)](
        q, k, v, mk, mv, bias, bias, bias, bias, bias, out,
        Q_OUTER=o, K_OUTER=o, TOTAL_HEADS=h, ACTIVE_HEADS=ah, HEAD_START=head_start,
        D_HEAD=128, HEAD_MAJOR=False, OUTPUT_HEAD_MAJOR=False, ROWS=r, COLS=l, C=4,
        BLOCK_I=triton.next_power_of_2(r), BLOCK_TOKENS=64, LOOP_STAGES=2,
        INPUT_PRECISION='ieee', EXACT_INTRA_FRAME=False, EXACT_NEIGHBOR_WINDOW_COLS=0,
        RECORD_DISTORTION=False, QUERY_OUTER_OFFSET=0, SINGLETON_TOP_R=0,
        SINGLETON_QUERY_COLUMN=False, SINGLETON_SECONDARY_FRAME_STRIDE=1,
        SINGLETON_PRIMARY_ROW_MOD4_OMIT2=False, QUERY_COLUMN_MAJOR=False,
        PACKED_SINGLETON_KV=False, SCALED_FP8_KV=False, COMPRESSED_FP8=False,
        SCALE=128 ** -.5, num_warps=4, num_stages=2)
    return (out, mk, mv, bias, seeds) if audit else out


class AxisCurrentC4:
    def __init__(self, axes, grid=(21, 45, 80)):
        assert axes in ('tw', 'wh', 'th')
        self.axes, self.grid = axes, grid
        self.qa, self.ka = ['thw'.index(a) for a in axes]
        self.oa = next(a for a in range(3) if a not in (self.qa, self.ka))
        self.chunk = ({'tw': 2, 'wh': 8, 'th': 1} if os.environ.get('AXIS_C4_PRECOMPUTE','0') == '1'
                      else {'tw': 4, 'wh': 8, 'th': 2})[axes]
        self.calls = 0

    def __call__(self, q, k, v):
        b, h, n, d = q.shape
        assert n == self.grid[0] * self.grid[1] * self.grid[2] and d == 128
        order = (0, 2 + self.oa, 2 + self.qa, 2 + self.ka, 1, 5)
        relaid = [x.reshape(b, h, *self.grid, d).permute(order).contiguous() for x in (q, k, v)]
        out = torch.empty_like(relaid[0])
        for hs in range(0, h, self.chunk):
            hc = min(self.chunk, h - hs)
            merged_attention(*relaid, head_start=hs, head_count=hc, output=out)
        inverse = tuple(order.index(i) for i in range(6))
        self.calls += 1
        return out.permute(inverse).reshape(b, h, n, d).contiguous()
