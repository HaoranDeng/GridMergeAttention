"""Fit foldable per-head transforms from GMA calibration statistics."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def truncate_residual(transform: torch.Tensor, rank: int):
    if rank <= 0 or rank >= transform.size(-1):
        return transform
    identity = torch.eye(
        transform.size(-1), device=transform.device, dtype=transform.dtype
    )
    residual = transform - identity
    u, s, vh = torch.linalg.svd(residual, full_matrices=False)
    return identity + torch.einsum(
        "hdr,hr,hre->hde", u[..., :rank], s[..., :rank], vh[..., :rank, :]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--ridge-relative", type=float, default=0.1)
    parser.add_argument(
        "--rank", type=int, default=0, help="Residual rank; zero keeps the full map."
    )
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used only for the offline ridge solves and optional SVDs.",
    )
    parser.add_argument(
        "--by-timestep",
        action="store_true",
        help="Fit one projection-folded transform per captured diffusion timestep.",
    )
    parser.add_argument(
        "--query-conditioned",
        action="store_true",
        help="Fit Y_gma T_y + Q T_q + b from query-conditioned statistics.",
    )
    parser.add_argument(
        "--query-rank",
        type=int,
        default=0,
        help="Rank of T_q; zero keeps the full query transform.",
    )
    args = parser.parse_args()

    merged_by_timestep = {}
    sources = []
    for path in args.stats:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        sources.append(str(path))
        timestep_states = (
            payload.get("timestep_layer_stats")
            if args.by_timestep
            else {"static": payload["layer_stats"]}
        )
        if not timestep_states:
            raise ValueError(f"{path} has no timestep_layer_stats")
        for timestep, layer_states in timestep_states.items():
            merged = merged_by_timestep.setdefault(timestep, {})
            for label, state in layer_states.items():
                query_state = "ztz" in state
                if query_state != args.query_conditioned:
                    raise ValueError(
                        f"{path} query-conditioned={query_state}, but "
                        f"--query-conditioned={args.query_conditioned}"
                    )
                matrix_keys = (
                    ("ztz", "zty", "zsum", "ysum")
                    if query_state
                    else ("xtx", "xty")
                )
                target = merged.setdefault(label, {"samples": 0, "calls": 0})
                for key in matrix_keys:
                    if key not in target:
                        target[key] = torch.zeros_like(
                            state[key], dtype=torch.float64
                        )
                    target[key].add_(state[key].double())
                target["samples"] += int(state["samples"])
                target["calls"] += int(state["calls"])

    if not merged_by_timestep:
        raise ValueError("no layer statistics found")
    first_merged = next(iter(merged_by_timestep.values()))
    first_state = next(iter(first_merged.values()))
    if args.query_conditioned:
        heads, feature_dim, _ = first_state["ztz"].shape
        dim = first_state["zty"].size(-1)
        if feature_dim != 2 * dim:
            raise ValueError(
                f"expected [Y_gma,Q] feature dim {2 * dim}, got {feature_dim}"
            )
    else:
        heads, dim, _ = first_state["xtx"].shape
        feature_dim = dim
    solve_device = torch.device(args.device)
    identity = torch.eye(
        dim, device=solve_device, dtype=torch.float64
    ).expand(heads, -1, -1)
    zero = torch.zeros_like(identity)
    joint_prior = torch.cat((identity, zero), dim=1)
    output_dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    def solve_layers(merged):
        transforms = {}
        query_transforms = {}
        biases = {}
        metadata = {}
        for layer_idx in range(args.layers):
            label = str(layer_idx)
            if label not in merged:
                transforms[label] = identity.float().to(
                    device="cpu", dtype=output_dtype
                )
                if args.query_conditioned:
                    query_transforms[label] = zero.float().to(
                        device="cpu", dtype=output_dtype
                    )
                    biases[label] = torch.zeros(
                        heads, dim, dtype=output_dtype
                    )
                metadata[label] = {"samples": 0, "calls": 0, "identity": True}
                continue
            state = merged[label]
            if args.query_conditioned:
                samples = int(state["samples"])
                zsum = state["zsum"].to(solve_device)
                ysum = state["ysum"].to(solve_device)
                xtx = state["ztz"].to(solve_device) - torch.einsum(
                    "hp,hq->hpq", zsum, zsum
                ) / samples
                xty = state["zty"].to(solve_device) - torch.einsum(
                    "hp,hd->hpd", zsum, ysum
                ) / samples
                prior = joint_prior
            else:
                xtx = state["xtx"].to(solve_device)
                xty = state["xty"].to(solve_device)
                prior = identity
            mean_energy = (
                xtx.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / feature_dim
            )
            ridge = float(args.ridge_relative) * mean_energy.clamp_min(
                torch.finfo(torch.float64).tiny
            )
            transform = torch.linalg.solve(
                xtx
                + ridge[:, None, None]
                * torch.eye(
                    feature_dim, device=solve_device, dtype=torch.float64
                ),
                xty + ridge[:, None, None] * prior,
            ).float()
            if args.query_conditioned:
                gma_transform = truncate_residual(
                    transform[:, :dim], args.rank
                )
                query_transform = transform[:, dim:]
                if 0 < args.query_rank < dim:
                    u, s, vh = torch.linalg.svd(
                        query_transform, full_matrices=False
                    )
                    query_transform = torch.einsum(
                        "hdr,hr,hre->hde",
                        u[..., : args.query_rank],
                        s[..., : args.query_rank],
                        vh[..., : args.query_rank, :],
                    )
                feature_mean = (zsum / state["samples"]).float()
                target_mean = (ysum / state["samples"]).float()
                bias = target_mean - torch.einsum(
                    "hpd,hde->hpe",
                    feature_mean[:, None, :dim],
                    gma_transform,
                ).squeeze(1) - torch.einsum(
                    "hpd,hde->hpe",
                    feature_mean[:, None, dim:],
                    query_transform,
                ).squeeze(1)
                transform = gma_transform
                query_transforms[label] = query_transform.to(
                    device="cpu", dtype=output_dtype
                )
                biases[label] = bias.to(device="cpu", dtype=output_dtype)
            else:
                transform = truncate_residual(transform, args.rank)
            transforms[label] = transform.to(
                device="cpu", dtype=output_dtype
            )
            metadata[label] = {
                "samples": state["samples"],
                "calls": state["calls"],
                "identity": False,
            }
        return transforms, query_transforms, biases, metadata

    solved = {
        timestep: solve_layers(merged)
        for timestep, merged in merged_by_timestep.items()
    }
    common = {
        "sources": sources,
        "ridge_relative": args.ridge_relative,
        "residual_rank": args.rank or dim,
        "num_layers": args.layers,
        "num_heads": heads,
        "head_dim": dim,
    }
    if args.query_conditioned:
        if not args.by_timestep:
            raise ValueError("query-conditioned calibration currently requires --by-timestep")
        result = {
            "version": 3,
            "method": "timestep-conditioned query-aware per-head GMA calibration",
            **common,
            "query_rank": args.query_rank or dim,
            "timestep_layer_transforms": {
                timestep: item[0] for timestep, item in solved.items()
            },
            "timestep_layer_query_transforms": {
                timestep: item[1] for timestep, item in solved.items()
            },
            "timestep_layer_biases": {
                timestep: item[2] for timestep, item in solved.items()
            },
            "timestep_layer_metadata": {
                timestep: item[3] for timestep, item in solved.items()
            },
        }
    elif args.by_timestep:
        result = {
            "version": 2,
            "method": (
                "timestep-conditioned projection-folded per-head GMA calibration"
            ),
            **common,
            "timestep_layer_transforms": {
                timestep: item[0] for timestep, item in solved.items()
            },
            "timestep_layer_metadata": {
                timestep: item[3] for timestep, item in solved.items()
            },
        }
    else:
        transforms, _, _, metadata = solved["static"]
        result = {
            "version": 1,
            "method": "projection-folded per-head GMA calibration",
            **common,
            "layer_transforms": transforms,
            "layer_metadata": metadata,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(
        f"Saved {len(solved) * args.layers} transforms ({heads} heads, dim {dim}, "
        f"rank {args.rank or dim}) to {args.output}"
    )


if __name__ == "__main__":
    main()
