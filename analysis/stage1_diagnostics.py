"""Stage-1 screens: one cheap statistic per architecture idea.

Each function answers "is the signal this idea depends on actually present in
real Wan attention inputs?" before anyone implements the idea. They run on the
tensors saved by capture_wan_qkv.py, so no model or pipeline is involved.

Every diagnostic prints its own kill signal, because a number without a
threshold is not a decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attention.adaptive_seed_pytorch import adaptive_seed_queries  # noqa: E402


def cluster_assignment(q, k, v, scale, alpha=0.3, pool=30, query_frames=2):
    """Key -> seed assignment for every block, [F,K,J,H,L] int8.

    This is the partition, not the seed identity. Two different seed sets can
    induce the same partition, which is exactly what idea 1 is betting on.

    Restricted to a few query frames: the full 21x21 frame-pair tensor is tens
    of GiB, and a partition-agreement rate does not need every block.
    """
    q = q[:, :query_frames]
    seeds = adaptive_seed_queries(q, k, v, scale, alpha=alpha, candidate_count=pool)
    logits = torch.einsum("bafkjchd,bfklhd->bhafkjcl", seeds.float(), k.float()) * scale
    return torch.log_softmax(logits, dim=-1).argmax(dim=-2).to(torch.int8)


def agreement(a, b):
    """Fraction of keys assigned to the same cluster index."""
    return (a == b).float().mean().item()


def temporal_vs_spatial_variance(k):
    """Idea 5. Compare key variance along time against along width.

    The method compresses 52 keys across width. If keys vary less along time
    than along width, the compressible axis is time and the layout is
    compressing the wrong one.
    """
    x = k.float()                                     # [B,F,R,C,H,D]
    var_time = x.var(dim=1, correction=0).mean().item()
    var_width = x.var(dim=3, correction=0).mean().item()
    var_height = x.var(dim=2, correction=0).mean().item()
    return var_time, var_height, var_width


def motion_consistency(q, k, scale, frames=(6, 7), sample_rows=6):
    """Idea 6. Does the best-matching key column shift coherently between frames?

    For a fixed query, find its best key column in frame f and in frame f+1. If
    motion is present and exploitable, the shift should concentrate around a
    small non-zero displacement rather than scatter uniformly.
    """
    f0, f1 = frames
    B, F, R, C, H, D = q.shape
    rows = torch.linspace(0, R - 1, sample_rows).round().long()
    qs = q[0, f0, rows][:, :, 0].float()               # [rows, C, D] query frame f0
    shifts = []
    for r_i, r in enumerate(rows):
        k0 = k[0, f0, r, :, 0].float()                 # [C, D]
        k1 = k[0, f1, r, :, 0].float()
        a0 = (qs[r_i] @ k0.T).argmax(dim=-1)           # [C]
        a1 = (qs[r_i] @ k1.T).argmax(dim=-1)
        shifts.append((a1 - a0).float())
    s = torch.cat(shifts)
    frac_zero = (s == 0).float().mean().item()
    return s.abs().mean().item(), s.std().item(), frac_zero


def dispersion_spread(q, k, v, scale, alpha=0.3, pool=30):
    """Idea 7. How unequal is per-block cluster dispersion?

    Adaptive C only pays if blocks differ in how hard they are to compress. A
    flat distribution means a uniform budget is already near-optimal.
    """
    q = q[:, :2]
    seeds = adaptive_seed_queries(q, k, v, scale, alpha=alpha, candidate_count=pool)
    logits = torch.einsum("bafkjchd,bfklhd->bhafkjcl", seeds.float(), k.float()) * scale
    w = torch.softmax(logits, dim=-1)
    entropy = -(w * torch.log(w.clamp_min(1e-9))).sum(-1)      # effective cluster size
    per_block = entropy.mean(dim=-1).flatten()
    return (per_block.mean().item(), per_block.std().item(),
            (per_block.std() / per_block.mean().clamp_min(1e-9)).item())


def query_neighborhood_variance(q, k, v, scale):
    """Idea 9. How similar are the dense outputs of adjacent queries?

    Clustering queries only works if neighbours want the same answer. Compares
    vertical neighbours against a random-pair baseline.
    """
    B, F, R, C, H, D = q.shape
    qq = q[0, :, :, :, 0].reshape(-1, D).float()
    kk = k[0, :, :, :, 0].reshape(-1, D).float()
    vv = v[0, :, :, :, 0].reshape(-1, D).float()
    out = torch.softmax(qq @ kk.T * scale, dim=-1) @ vv
    out = out.reshape(F, R, C, D)
    adj = (out[:, 1:] - out[:, :-1]).norm(dim=-1).mean().item()
    idx = torch.randperm(R * C)[: R * C // 2]
    flat = out.reshape(F, R * C, D)
    rnd = (flat[:, idx] - flat[:, torch.randperm(R * C)[: len(idx)]]).norm(dim=-1).mean().item()
    return adj, rnd, adj / max(rnd, 1e-9)


def quantization_error(q, k, v, scale, bits=(8, 4)):
    """Idea 10. Dense attention with quantized K/V, no merging at all.

    Quantization attacks the same memory traffic that merging does, but adds no
    selection cost. If its error at 4 bits beats hard C4's, the whole selection
    problem is avoidable.
    """
    B, F, R, C, H, D = q.shape
    qq = q[0, :, :, :, 0].reshape(-1, D).float()
    kk = k[0, :, :, :, 0].reshape(-1, D).float()
    vv = v[0, :, :, :, 0].reshape(-1, D).float()
    ref = torch.softmax(qq @ kk.T * scale, dim=-1) @ vv
    results = {}
    for b in bits:
        levels = 2 ** b - 1
        def qz(x):
            lo, hi = x.amin(dim=0, keepdim=True), x.amax(dim=0, keepdim=True)
            step = (hi - lo).clamp_min(1e-9) / levels
            return ((x - lo) / step).round().clamp(0, levels) * step + lo
        got = torch.softmax(qq @ qz(kk).T * scale, dim=-1) @ qz(vv)
        results[f"{b}bit"] = ((got - ref).norm() / ref.norm()).item()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", required=True)
    p.add_argument("--output", default="out/stage1_diagnostics.json")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    blob = torch.load(args.capture)
    tensors = blob["tensors"]
    keys = sorted(tensors, key=lambda s: tuple(int(x) for x in s.split("_")))
    dev = args.device
    get = lambda key, name: tensors[key][name].to(dev)
    D = tensors[keys[0]]["q"].shape[-1]
    scale = D ** -0.5
    report = {"points": keys}

    print(f"{len(keys)} (step, layer) points, head dim {D}\n")

    # --- idea 5 -----------------------------------------------------------
    vt = vh = vw = 0.0
    for key in keys:
        a, b, c = temporal_vs_spatial_variance(get(key, "k"))
        vt, vh, vw = vt + a, vh + b, vw + c
    n = len(keys)
    report["idea5_key_variance"] = {"time": vt / n, "height": vh / n, "width": vw / n}
    print(f"[5] key variance   time {vt/n:.4f}   height {vh/n:.4f}   width {vw/n:.4f}")
    print(f"    -> compress the LOWEST-variance axis. currently compressing width.\n")

    # --- idea 6 -----------------------------------------------------------
    m_abs, m_std, m_zero = motion_consistency(get(keys[0], "q"), get(keys[0], "k"), scale)
    report["idea6_motion"] = {"mean_abs_shift": m_abs, "std_shift": m_std, "frac_zero": m_zero}
    print(f"[6] best-key shift between adjacent frames: mean |shift| {m_abs:.2f} cols,"
          f" std {m_std:.2f}, exactly-zero {m_zero:.1%}")
    print(f"    -> high frac_zero means no motion to compensate; wide std means incoherent.\n")

    # --- idea 7 -----------------------------------------------------------
    mu, sd, cv = dispersion_spread(get(keys[0], "q"), get(keys[0], "k"), get(keys[0], "v"), scale)
    report["idea7_dispersion"] = {"mean": mu, "std": sd, "cv": cv}
    print(f"[7] cluster dispersion: mean {mu:.3f} nats, std {sd:.3f}, CV {cv:.3f}")
    print(f"    -> CV below ~0.2 means blocks are uniformly hard, adaptive C buys little.\n")

    # --- idea 9 -----------------------------------------------------------
    adj, rnd, ratio = query_neighborhood_variance(
        get(keys[0], "q"), get(keys[0], "k"), get(keys[0], "v"), scale)
    report["idea9_query_similarity"] = {"adjacent": adj, "random": rnd, "ratio": ratio}
    print(f"[9] dense-output distance: adjacent queries {adj:.4f}, random pairs {rnd:.4f},"
          f" ratio {ratio:.3f}")
    print(f"    -> ratio near 1 means neighbours are no more alike than strangers.\n")

    # --- idea 10 ----------------------------------------------------------
    qe = quantization_error(get(keys[0], "q"), get(keys[0], "k"), get(keys[0], "v"), scale)
    report["idea10_quantization"] = qe
    print(f"[10] dense attention with quantized K/V: " +
          "  ".join(f"{b} {e:.5f}" for b, e in qe.items()))
    print(f"    -> compare against hard C4 at rel-L2 ~0.047-0.071 on real QKV.\n")

    # --- ideas 1 and 8 ----------------------------------------------------
    parts = {key: cluster_assignment(get(key, "q"), get(key, "k"), get(key, "v"), scale)
             for key in keys}
    steps = sorted({int(key.split("_")[0]) for key in keys})
    layers = sorted({int(key.split("_")[1]) for key in keys})
    across_steps, across_layers = [], []
    for l in layers:
        for i in range(1, len(steps)):
            a, b = f"{steps[i-1]}_{l}", f"{steps[i]}_{l}"
            if a in parts and b in parts:
                across_steps.append(agreement(parts[a], parts[b]))
    for s in steps:
        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                a, b = f"{s}_{layers[i]}", f"{s}_{layers[j]}"
                if a in parts and b in parts:
                    across_layers.append(agreement(parts[a], parts[b]))
    head_pairs = []
    for key in keys:
        pa = parts[key]
        if pa.shape[-2] >= 2:
            head_pairs.append(agreement(pa[..., 0, :], pa[..., 1, :]))
    report["idea1_partition_across_steps"] = sum(across_steps) / max(len(across_steps), 1)
    report["idea1_partition_across_layers"] = sum(across_layers) / max(len(across_layers), 1)
    report["idea8_partition_across_heads"] = sum(head_pairs) / max(len(head_pairs), 1)
    print(f"[1] partition agreement, consecutive kept steps: "
          f"{report['idea1_partition_across_steps']:.1%}")
    print(f"    (seed-identity agreement was 7-11%; partition can be higher)")
    print(f"[1] partition agreement across layers: {report['idea1_partition_across_layers']:.1%}")
    print(f"[8] partition agreement across heads in a layer: "
          f"{report['idea8_partition_across_heads']:.1%}")
    print(f"    -> random baseline is ~25% for four clusters.\n")

    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
