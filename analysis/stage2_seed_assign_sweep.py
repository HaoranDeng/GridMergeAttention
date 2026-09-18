"""Rank step-1 (seed selection) and step-2 (key assignment) variants.

Scope is deliberately narrow: only how the four seed query rows are chosen and
how the 52 keys are assigned to them. C stays 4, the block layout is unchanged,
the merge is the unchanged masked cluster softmax, and the second stage is full
attention over the merged tokens. Every arm therefore sits at identical
compressed-token density, so rel-L2 is directly comparable between them.

Runs on tensors from capture_wan_qkv.py, so no model is loaded and every arm
sees identical inputs. Seed sets are computed once and shared across the
assignment arms that build on them, which is most of the runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attention.adaptive_seed_pytorch import (  # noqa: E402
    _adaptive_selection, _gather_position, _gather_probes, _probe_columns,
    adaptive_seed_queries,
)
from attention.hard_route_pytorch import (  # noqa: E402
    cached_local_c4_seed_rows, gather_seed_queries,
)

NEG_INF = torch.finfo(torch.float32).min
C4 = 4


# ---------------------------------------------------------------- merge core
def seed_key_logits(seeds, k, scale):
    return torch.einsum("bafkjchd,bfklhd->bhafkjcl", seeds.float(), k.float()) * scale


def merge_and_attend(logits, assignment, k, v, q, scale):
    """Unchanged hard-C4 merge and full compressed attention, given a partition."""
    masked = logits.masked_fill(~assignment, NEG_INF)
    log_w = torch.log_softmax(masked, dim=-1).masked_fill(~assignment, 0.0)
    w = log_w.exp().masked_fill(~assignment, 0.0)
    bias = (w * log_w).sum(-1).masked_fill(~assignment.any(-1), float("inf"))
    mk = torch.einsum("bhafkjcl,bfklhd->bhafkjcd", w, k.float())
    mv = torch.einsum("bhafkjcl,bfklhd->bhafkjcd", w, v.float())
    B, H, A, F, K, J, C, D = mk.shape
    I = q.size(2)
    qq = q.float().permute(0, 4, 1, 2, 3, 5)                       # [B,H,A,I,J,D]
    qq = qq.permute(0, 1, 2, 4, 3, 5).reshape(B * H * A * J, I, D)
    kk = mk.permute(0, 1, 2, 5, 3, 4, 6, 7).reshape(B * H * A * J, F * K * C, D)
    vv = mv.permute(0, 1, 2, 5, 3, 4, 6, 7).reshape(B * H * A * J, F * K * C, D)
    bb = bias.permute(0, 1, 2, 5, 3, 4, 6).reshape(B * H * A * J, F * K * C)
    out = torch.nn.functional.scaled_dot_product_attention(
        qq.unsqueeze(1), kk.unsqueeze(1), vv.unsqueeze(1),
        attn_mask=(-bb).unsqueeze(1).unsqueeze(2), scale=scale).squeeze(1)
    return out.reshape(B, H, A, J, I, D)


def dense_reference(q, k, v, scale):
    B, A, I, J, H, D = q.shape
    F, K, L = k.size(1), k.size(2), k.size(3)
    qq = q.float().permute(0, 4, 1, 2, 3, 5).reshape(B * H, A * I * J, D)
    kk = k.float().permute(0, 4, 1, 2, 3, 5).reshape(B * H, F * K * L, D)
    vv = v.float().permute(0, 4, 1, 2, 3, 5).reshape(B * H, F * K * L, D)
    out = torch.nn.functional.scaled_dot_product_attention(qq, kk, vv, scale=scale)
    return out.reshape(B, H, A, I, J, D).permute(0, 1, 2, 4, 3, 5)   # [B,H,A,J,I,D]


# ------------------------------------------------------------ step 2 variants
def assign_key_softmax(logits):
    """Shipped rule: normalise per seed over keys, each key takes its best seed."""
    best = torch.log_softmax(logits, dim=-1).argmax(dim=-2, keepdim=True)
    a = torch.zeros_like(logits, dtype=torch.bool)
    return a.scatter_(-2, best, True)


def _merged_keys(logits, assignment, k):
    masked = logits.masked_fill(~assignment, NEG_INF)
    w = torch.log_softmax(masked, dim=-1).exp().masked_fill(~assignment, 0.0)
    return torch.einsum("bhafkjcl,bfklhd->bhafkjcd", w, k.float())


def assign_tightness(logits, k, base=None, lam=None):
    """One Lloyd step in key space: reassign to the nearest merged key.

    The exactness identity holds for any subset, so membership is free to
    optimise anything. The second-order error term is the within-cluster key
    covariance, which max-likelihood assignment never targets.
    """
    base = assign_key_softmax(logits) if base is None else base
    mk = _merged_keys(logits, base, k)                      # [B,H,A,F,K,J,C,D]
    # ||mk - k||^2 expanded, because materialising the difference would be
    # [B,H,A,F,K,J,C,L,D] and cost tens of GiB. Note J (query column) and L
    # (key column) are both 52 here and must not be broadcast against each
    # other -- they are different axes.
    cross = torch.einsum("bhafkjcd,bfklhd->bhafkjcl", mk, k.float())
    mk_sq = mk.pow(2).sum(-1).unsqueeze(-1)                 # [...,C,1]
    k_sq = k.float().pow(2).sum(-1).permute(0, 4, 1, 2, 3)  # [B,H,F,K,L]
    k_sq = k_sq[:, :, None, :, :, None, None, :]            # [B,H,1,F,K,1,1,L]
    dist = mk_sq - 2.0 * cross + k_sq
    if lam is None:
        best = dist.argmin(dim=-2, keepdim=True)
    else:
        score = torch.log_softmax(logits, dim=-1) - lam * dist
        best = score.argmax(dim=-2, keepdim=True)
    a = torch.zeros_like(logits, dtype=torch.bool)
    return a.scatter_(-2, best, True)


def assign_capacity(logits, cap):
    """Key-softmax choice, but no cluster may exceed `cap` keys.

    Sinkhorn balancing failed because equal clusters fight real attention mass.
    A ceiling is a weaker constraint: it only blocks the degenerate case where
    one seed swallows most of the row and the others come back empty.
    """
    score = torch.log_softmax(logits, dim=-1)               # [...,C,L]
    order = score.max(dim=-2).values.argsort(dim=-1, descending=True)
    C = score.size(-2)
    pref = score.argmax(dim=-2)                             # [...,L]
    pref_sorted = torch.gather(pref, -1, order)
    counts = torch.zeros(*score.shape[:-2], C, device=score.device, dtype=torch.int32)
    out = torch.empty_like(pref_sorted)
    ranked = score.argsort(dim=-2, descending=True)         # [...,C,L]
    ranked_sorted = torch.gather(ranked, -1, order.unsqueeze(-2).expand_as(ranked))
    for i in range(score.size(-1)):
        placed = torch.zeros_like(pref_sorted[..., i])
        done = torch.zeros_like(placed, dtype=torch.bool)
        for r in range(C):
            cand = ranked_sorted[..., r, i]
            room = torch.gather(counts, -1, cand.unsqueeze(-1)).squeeze(-1) < cap
            take = room & ~done
            placed = torch.where(take, cand, placed)
            done = done | take
        placed = torch.where(done, placed, ranked_sorted[..., 0, i])
        counts.scatter_add_(-1, placed.unsqueeze(-1),
                            torch.ones_like(placed.unsqueeze(-1), dtype=torch.int32))
        out[..., i] = placed
    final = torch.empty_like(out)
    final.scatter_(-1, order, out)
    a = torch.zeros_like(logits, dtype=torch.bool)
    return a.scatter_(-2, final.unsqueeze(-2), True)


# ------------------------------------------------------------ step 1 variants
def cheap_seed_queries(q, k, v, scale, pool=30, probes=16, min_gap=3):
    """No greedy search: anchor plus the highest-mass rows kept a fixed gap apart.

    Tests whether span's benefit comes from importance (cheap to compute, one
    dot per candidate) or from the spread that the Gram-Schmidt loop provides
    at 99% of the runtime.
    """
    chosen, window, bank = _adaptive_selection(
        q, k, v, scale, alpha=0.0, candidate_count=pool,
        probe_count=probes, value_dimensions=32)
    probe_index = _probe_columns(q, k, scale, probes, max(1, probes // 2))
    lg = torch.einsum("bafkjhpd,bafkjhsd->bafkjhps", bank,
                      _gather_probes(k, probe_index, q.size(1), q.size(3))) * scale
    mass = torch.logsumexp(lg, dim=-1)                       # [...,P]
    anchor = chosen[0]
    order = mass.argsort(dim=-1, descending=True)
    picks = [anchor]
    for _ in range(C4 - 1):
        ok = torch.ones_like(order, dtype=torch.bool)
        for p in picks:
            ok &= (order - p.unsqueeze(-1)).abs() >= min_gap
        idx = torch.where(ok.any(-1, keepdim=True), ok.float().argmax(-1, keepdim=True),
                          torch.zeros_like(order[..., :1]))
        picks.append(torch.gather(order, -1, idx).squeeze(-1))
    return torch.stack([_gather_position(bank, p) for p in picks], dim=-2).transpose(-3, -2).to(q.dtype)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", required=True)
    p.add_argument("--output", default="out/stage2_seed_assign.json")
    p.add_argument("--query_frames", type=int, default=2)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    blob = torch.load(args.capture)
    tensors = blob["tensors"]
    keys = sorted(tensors, key=lambda s: tuple(int(x) for x in s.split("_")))
    D = tensors[keys[0]]["q"].shape[-1]
    scale = D ** -0.5
    rows = {}

    for key in keys:
        t = tensors[key]
        q = t["q"][:, : args.query_frames].to(args.device)
        k = t["k"].to(args.device)
        v = t["v"].to(args.device)
        ref = dense_reference(q, k, v, scale)
        rel = lambda o: ((o - ref).norm() / ref.norm()).item()

        # step-1 arms: seed sets, computed once and shared downstream
        fixed_rows = cached_local_c4_seed_rows(
            q.device, 0, q.size(1), k.size(1), k.size(2), q.size(3), q.size(2))
        seeds = {
            "fixed":      gather_seed_queries(q, fixed_rows),
            "span_a0":    adaptive_seed_queries(q, k, v, scale, alpha=0.0, candidate_count=30),
            "span_a03":   adaptive_seed_queries(q, k, v, scale, alpha=0.3, candidate_count=30),
            "cheap":      cheap_seed_queries(q, k, v, scale),
        }
        lg = {name: seed_key_logits(s, k, scale) for name, s in seeds.items()}

        arms = {}
        for name in seeds:
            arms[f"{name}|keysoftmax"] = assign_key_softmax(lg[name])
        base = arms["span_a03|keysoftmax"]
        arms["span_a03|tightness"] = assign_tightness(lg["span_a03"], k, base=base)
        for lam in (0.5, 2.0):
            arms[f"span_a03|hybrid{lam}"] = assign_tightness(lg["span_a03"], k, base=base, lam=lam)
        for cap in (20, 30):
            arms[f"span_a03|cap{cap}"] = assign_capacity(lg["span_a03"], cap)

        for name, a in arms.items():
            seed_name = name.split("|")[0]
            out = merge_and_attend(lg[seed_name], a, k, v, q, scale)
            rows.setdefault(name, []).append(rel(out))
        del q, k, v, ref, lg, arms, seeds
        torch.cuda.empty_cache()
        print(f"  {key} done", flush=True)

    summary = {n: sum(v) / len(v) for n, v in rows.items()}
    baseline = summary["fixed|keysoftmax"]
    print(f"\n{'arm':34s} {'rel-L2':>9} {'vs fixed':>9} {'vs span_a03':>12}")
    ref03 = summary["span_a03|keysoftmax"]
    for n, val in sorted(summary.items(), key=lambda kv: kv[1]):
        print(f"{n:34s} {val:9.5f} {100*(val/baseline-1):8.1f}% {100*(val/ref03-1):11.2f}%")
    Path(args.output).write_text(json.dumps(
        {"per_point": rows, "mean": summary, "points": keys}, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
