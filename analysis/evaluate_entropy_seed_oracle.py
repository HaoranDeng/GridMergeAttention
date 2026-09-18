"""Evaluate entropy-objective seed oracles on real Wan2.1-14B QKV.

For a target query ``q`` and a real seed query ``s``, the loss in the
entropy-regularized variational objective is ``KL(p_s || p_q)``.  This script
uses that direction to build an offline, key-aware C4 medoid oracle inside
every native 80-key GCA block.  It then evaluates the selected rows with the
unchanged hard assignment, entropy-aware merge, and global attention stages.

The oracle is intentionally not a deployable selector: exact block masses and
all 80 keys are read during selection.  Its purpose is to test whether the
derived objective predicts lower final GCA error, and whether the production
Hellinger-Span selector is a useful cheap surrogate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attention.adaptive_seed_pytorch import multiscale_candidate_table
from attention.adaptive_seed_triton import hellinger_span_seed_rows_triton
from sweep_simple_weighted_span_qkv import _dense_frame
from sweep_simple_weighted_span_selectors import _fixed_rows, evaluate_rows


FRAMES = 21
ROWS = 45
COLS = 80
C4 = 4


def _parse_indices(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split()]


def _load_grid(path: str, heads: list[int]) -> tuple[torch.Tensor, ...]:
    with np.load(path) as payload:
        return tuple(
            torch.from_numpy(payload[name][0, heads].copy()).reshape(
                len(heads), FRAMES, ROWS, COLS, -1
            )
            for name in ("q", "k", "v")
        )


def _gather_query_rows(query: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Gather query[J,R,D] with rows[J,B,C]."""

    columns = torch.arange(query.size(0), device=query.device)[:, None, None]
    return query[columns, rows]


def _distance_column(distance: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather one candidate column from distance[..., demand, candidate]."""

    demand_count = distance.size(-2)
    gather_index = index[..., None, None].expand(*index.shape, demand_count, 1)
    return distance.gather(-1, gather_index).squeeze(-1)


def _anchored_facility(
    distance: torch.Tensor,
    anchor: torch.Tensor,
    demand_weight: torch.Tensor,
) -> torch.Tensor:
    """Greedily reduce a weighted anchored C4 facility objective."""

    prefix = distance.shape[:-2]
    candidate_count = distance.size(-1)
    anchor = anchor.expand(prefix)
    selected = [anchor]
    used = torch.zeros(
        *prefix, candidate_count, device=distance.device, dtype=torch.bool
    )
    used.scatter_(-1, anchor.unsqueeze(-1), True)
    minimum = _distance_column(distance, anchor)
    for _ in range(1, C4):
        improved = torch.minimum(minimum.unsqueeze(-1), distance)
        gain = (
            (minimum.unsqueeze(-1) - improved) * demand_weight.unsqueeze(-1)
        ).sum(-2)
        index = gain.masked_fill(used, -float("inf")).argmax(dim=-1)
        selected.append(index)
        used.scatter_(-1, index.unsqueeze(-1), True)
        minimum = torch.minimum(minimum, _distance_column(distance, index))
    return torch.stack(selected, dim=-1)


def _conditional_kl_distances(
    target_log_probability: torch.Tensor,
    candidate_log_probability: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reverse and forward KL as [J,B,target,candidate]."""

    target_probability = target_log_probability.exp()
    candidate_probability = candidate_log_probability.exp()

    candidate_negative_entropy = (
        candidate_probability * candidate_log_probability
    ).sum(-1)
    candidate_under_target = torch.einsum(
        "jbcl,jbrl->jbrc", candidate_probability, target_log_probability
    )
    reverse = (
        candidate_negative_entropy.unsqueeze(-2) - candidate_under_target
    ).clamp_min_(0.0)

    target_negative_entropy = (
        target_probability * target_log_probability
    ).sum(-1)
    target_under_candidate = torch.einsum(
        "jbrl,jbcl->jbrc", target_probability, candidate_log_probability
    )
    forward = (
        target_negative_entropy.unsqueeze(-1) - target_under_candidate
    ).clamp_min_(0.0)
    return reverse, forward


@torch.inference_mode()
def select_entropy_medoid_rows(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    candidate_count: int,
    candidate_geometry_count: int,
    block_chunk: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float], dict[str, float]]:
    """Select real seed rows with exact per-grid-block conditional KL.

    ``query`` is one router frame with shape [column,row,dimension], while
    ``key`` contains all native GCA blocks with shape [block,80,dimension].
    The mass-weighted objective is

        sum_i m_{ib} min_s KL(p_{s,b} || p_{i,b}),

    where ``m_{ib}`` is the target query's exact dense-attention mass on block
    ``b`` and every candidate ``s`` is an actual query row.
    """

    if query.shape[:2] != (COLS, ROWS):
        raise ValueError(f"expected query [{COLS},{ROWS},D], got {tuple(query.shape)}")
    if key.shape[:2] != (FRAMES * ROWS, COLS):
        raise ValueError(
            f"expected key [{FRAMES * ROWS},{COLS},D], got {tuple(key.shape)}"
        )
    scale = query.size(-1) ** -0.5
    device = query.device
    candidate_table, anchor_table = multiscale_candidate_table(
        ROWS,
        candidate_count,
        device,
        geometry_count=candidate_geometry_count,
    )
    block_rows = torch.arange(key.size(0), device=device) % ROWS
    candidate_rows = candidate_table[block_rows].to(torch.long)
    anchor = anchor_table[block_rows].to(torch.long)
    started = time.perf_counter()

    # Exact global log-partition for each of the 80x45 router queries.  It is
    # accumulated blockwise so the full [J,R,B,L] score tensor is never kept.
    global_log_partition = torch.full(
        (COLS, ROWS), -float("inf"), device=device, dtype=torch.float32
    )
    for start in range(0, key.size(0), block_chunk):
        key_chunk = key[start : start + block_chunk].float()
        logits = torch.einsum("jrd,bld->jrbl", query.float(), key_chunk) * scale
        chunk_partition = torch.logsumexp(logits.flatten(-2), dim=-1)
        global_log_partition = torch.logaddexp(
            global_log_partition, chunk_partition
        )

    outputs: dict[str, list[torch.Tensor]] = {
        "reverse_kl_uniform": [],
        "reverse_kl_mass": [],
        "forward_kl_mass": [],
    }
    objective_sum = defaultdict(float)
    objective_weight = defaultdict(float)

    for start in range(0, key.size(0), block_chunk):
        end = min(start + block_chunk, key.size(0))
        key_chunk = key[start:end].float()
        rows_chunk = candidate_rows[start:end]
        anchors_chunk = anchor[start:end].unsqueeze(0).expand(COLS, -1)
        candidates = _gather_query_rows(
            query, rows_chunk.unsqueeze(0).expand(COLS, -1, -1)
        ).float()

        target_logits = (
            torch.einsum("jrd,bld->jbrl", query.float(), key_chunk) * scale
        )
        candidate_logits = (
            torch.einsum("jbcd,bld->jbcl", candidates, key_chunk) * scale
        )
        target_log_partition = torch.logsumexp(target_logits, dim=-1)
        target_log_probability = torch.log_softmax(target_logits, dim=-1)
        candidate_log_probability = torch.log_softmax(candidate_logits, dim=-1)
        reverse, forward = _conditional_kl_distances(
            target_log_probability, candidate_log_probability
        )

        block_mass = torch.exp(
            target_log_partition - global_log_partition.unsqueeze(1)
        )
        uniform_weight = torch.ones_like(block_mass)
        selections = {
            "reverse_kl_uniform": _anchored_facility(
                reverse, anchors_chunk, uniform_weight
            ),
            "reverse_kl_mass": _anchored_facility(
                reverse, anchors_chunk, block_mass
            ),
            "forward_kl_mass": _anchored_facility(
                forward, anchors_chunk, block_mass
            ),
        }
        expanded_rows = rows_chunk.unsqueeze(0).expand(COLS, -1, -1)
        for name, positions in selections.items():
            rows = expanded_rows.gather(-1, positions)
            outputs[name].append(rows.cpu())
            distance = reverse if name.startswith("reverse") else forward
            weight = uniform_weight if name.endswith("uniform") else block_mass
            chosen_distance = distance.gather(
                -1,
                positions.unsqueeze(-2).expand(-1, -1, ROWS, -1),
            ).amin(-1)
            objective_sum[name] += float((chosen_distance * weight).sum())
            objective_weight[name] += float(weight.sum())

        del target_logits, candidate_logits, reverse, forward

    if query.is_cuda:
        torch.cuda.synchronize(query.device)
    elapsed = time.perf_counter() - started
    rows = {
        name: torch.cat(chunks, dim=1).to(device)
        for name, chunks in outputs.items()
    }
    objective = {
        name: objective_sum[name] / max(objective_weight[name], 1.0e-30)
        for name in outputs
    }
    timing = {
        "exact_global_partition_and_medoid_seconds": elapsed,
    }
    return rows, objective, timing


@torch.inference_mode()
def evaluate_attention_divergence(
    query: torch.Tensor,
    key: torch.Tensor,
    rows: torch.Tensor,
    *,
    column_chunk: int,
) -> dict[str, float]:
    """Measure both KL directions between dense and induced GCA attention."""

    scale = query.size(-1) ** -0.5
    forward_sum = 0.0
    reverse_sum = 0.0
    squared_hellinger_sum = 0.0
    queries = 0
    active_sum = 0.0
    active_count = 0

    for start in range(0, query.size(0), column_chunk):
        end = min(start + column_chunk, query.size(0))
        query_chunk = query[start:end].float()
        rows_chunk = rows[start:end]
        seeds = _gather_query_rows(query_chunk, rows_chunk).float()
        seed_logits = torch.einsum("jbcd,bld->jbcl", seeds, key.float()) * scale
        owner = torch.log_softmax(seed_logits, dim=-1).argmax(dim=-2)
        cluster_id = torch.arange(C4, device=query.device).view(1, 1, C4, 1)
        assignment = owner.unsqueeze(-2) == cluster_id
        active = assignment.any(dim=-1)
        active_sum += float(active.float().sum())
        active_count += active.numel()

        masked_seed_logits = seed_logits.masked_fill(~assignment, -float("inf"))
        safe_seed_logits = torch.where(
            active.unsqueeze(-1), masked_seed_logits, torch.zeros_like(seed_logits)
        )
        seed_log_partition = torch.logsumexp(masked_seed_logits, dim=-1)
        seed_log_probability = torch.log_softmax(safe_seed_logits, dim=-1)
        seed_probability = torch.where(
            assignment, seed_log_probability.exp(), torch.zeros_like(seed_logits)
        )
        merged_key = torch.einsum("jbcl,bld->jbcd", seed_probability, key.float())
        tangent_bias = (seeds * merged_key).sum(-1) * scale - seed_log_partition

        cluster_logits = (
            torch.einsum("jrd,jbcd->jrbc", query_chunk, merged_key) * scale
            - tangent_bias.unsqueeze(1)
        )
        cluster_logits = cluster_logits.masked_fill(
            ~active.unsqueeze(1), -float("inf")
        )
        cluster_log_probability = torch.log_softmax(
            cluster_logits.flatten(-2), dim=-1
        ).reshape_as(cluster_logits)

        dense_logits = torch.einsum("jrd,bld->jrbl", query_chunk, key.float()) * scale
        dense_log_probability = dense_logits - torch.logsumexp(
            dense_logits.flatten(-2), dim=-1
        )[..., None, None]

        owner_for_queries = owner.unsqueeze(1).expand(
            -1, query_chunk.size(1), -1, -1
        )
        cluster_for_key = cluster_log_probability.gather(-1, owner_for_queries)
        within_for_key = (
            seed_log_probability.permute(0, 1, 3, 2)
            .gather(-1, owner.unsqueeze(-1))
            .squeeze(-1)
            .unsqueeze(1)
        )
        induced_log_probability = cluster_for_key + within_for_key
        dense_probability = dense_log_probability.exp()
        induced_probability = induced_log_probability.exp()

        forward = (
            dense_probability * (dense_log_probability - induced_log_probability)
        ).sum(dim=(-2, -1))
        reverse = (
            induced_probability * (induced_log_probability - dense_log_probability)
        ).sum(dim=(-2, -1))
        squared_hellinger = 1.0 - (
            torch.sqrt(dense_probability * induced_probability)
        ).sum(dim=(-2, -1))
        forward_sum += float(forward.clamp_min(0.0).sum())
        reverse_sum += float(reverse.clamp_min(0.0).sum())
        squared_hellinger_sum += float(squared_hellinger.clamp_min(0.0).sum())
        queries += forward.numel()

    return {
        "mean_forward_kl": forward_sum / queries,
        "mean_reverse_kl": reverse_sum / queries,
        "mean_squared_hellinger": squared_hellinger_sum / queries,
        "active_cluster_fraction": active_sum / active_count,
        "queries": queries,
    }


def _shared_seed_count(rows: torch.Tensor, reference: torch.Tensor) -> float:
    shared = (rows[..., :, None] == reference[..., None, :]).any(-1).float().sum(-1)
    return float(shared.mean())


def _add(total: dict[str, float], metrics: dict[str, float]) -> None:
    total["squared_error"] += metrics["squared_error"]
    total["reference_squared"] += metrics["reference_squared"]
    total["numel"] += metrics["numel"]
    total["forward_kl_sum"] += metrics["mean_forward_kl"] * metrics["queries"]
    total["reverse_kl_sum"] += metrics["mean_reverse_kl"] * metrics["queries"]
    total["hellinger_sum"] += metrics["mean_squared_hellinger"] * metrics["queries"]
    total["queries"] += metrics["queries"]
    total["active_sum"] += metrics["active_cluster_fraction"] * metrics["queries"]
    total["points"] += 1


def _new_total() -> dict[str, float]:
    return {
        "squared_error": 0.0,
        "reference_squared": 0.0,
        "numel": 0.0,
        "forward_kl_sum": 0.0,
        "reverse_kl_sum": 0.0,
        "hellinger_sum": 0.0,
        "queries": 0.0,
        "active_sum": 0.0,
        "points": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--heads", default="0,10,20,30")
    parser.add_argument("--frames", default="0,10,20")
    parser.add_argument("--router_frame", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate_count", type=int, default=32)
    parser.add_argument("--candidate_geometry_count", type=int, default=45)
    parser.add_argument("--block_chunk", type=int, default=21)
    parser.add_argument("--column_chunk", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--top_probes", type=int, default=8)
    parser.add_argument("--uniform_probes", type=int, default=8)
    parser.add_argument("--probe_anchor_count", type=int, default=2)
    parser.add_argument("--block_j", type=int, default=4)
    args = parser.parse_args()

    heads = _parse_indices(args.heads)
    frames = _parse_indices(args.frames)
    if not 0 <= args.router_frame < FRAMES:
        raise ValueError(f"router_frame must be in [0,{FRAMES})")
    device = torch.device(args.device)
    totals: dict[str, dict[str, float]] = defaultdict(_new_total)
    records = {}
    selector_records = {}

    for capture in args.qkv:
        q_grid, k_grid, v_grid = _load_grid(capture, heads)
        q_grid = q_grid.to(device)
        k_grid = k_grid.to(device)
        v_grid = v_grid.to(device)
        key_blocks_all = k_grid.reshape(len(heads), FRAMES * ROWS, COLS, -1)
        value_blocks_all = v_grid.reshape(len(heads), FRAMES * ROWS, COLS, -1)

        production_q = (
            q_grid[:, args.router_frame : args.router_frame + 1]
            .permute(1, 2, 3, 0, 4)
            .unsqueeze(0)
            .contiguous()
        )
        production_k = k_grid.permute(1, 2, 3, 0, 4).unsqueeze(0).contiguous()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        hellinger_table = hellinger_span_seed_rows_triton(
            production_q,
            production_k,
            alpha=args.alpha,
            candidate_count=args.candidate_count,
            candidate_geometry_count=args.candidate_geometry_count,
            top_probes=args.top_probes,
            uniform_probes=args.uniform_probes,
            probe_anchor_count=args.probe_anchor_count,
            block_j=args.block_j,
            group_tile=32,
            output_dtype=torch.uint8,
            gs_num_warps=8,
            selection_mode="gs",
        )
        torch.cuda.synchronize(device)
        hellinger_seconds = time.perf_counter() - started

        capture_record = {}
        selector_record = {
            "hellinger_seconds_all_heads": hellinger_seconds,
            "heads": {},
        }
        for head_slot, head in enumerate(heads):
            router_query = q_grid[head_slot, args.router_frame].permute(1, 0, 2)
            key_blocks = key_blocks_all[head_slot]
            value_blocks = value_blocks_all[head_slot]
            rows, objectives, timing = select_entropy_medoid_rows(
                router_query,
                key_blocks,
                candidate_count=args.candidate_count,
                candidate_geometry_count=args.candidate_geometry_count,
                block_chunk=args.block_chunk,
            )
            rows["hellinger_span"] = (
                hellinger_table[0, 0, ..., head_slot, :]
                .reshape(FRAMES * ROWS, COLS, C4)
                .permute(1, 0, 2)
                .to(torch.long)
            )
            rows["fixed_c4"] = _fixed_rows(router_query, key_blocks.size(0))
            selector_record["heads"][str(head)] = {
                "objectives": objectives,
                "timing": timing,
                "overlap_with_reverse_kl_mass": {
                    name: _shared_seed_count(table, rows["reverse_kl_mass"])
                    for name, table in rows.items()
                },
            }

            for frame in frames:
                query = q_grid[head_slot, frame].permute(1, 0, 2).contiguous()
                dense = _dense_frame(
                    q_grid[head_slot], k_grid[head_slot], v_grid[head_slot], frame
                )
                point_name = f"{Path(capture).parent.name}:h{head}:f{frame}"
                point = {}
                for name, seed_rows in rows.items():
                    metrics = evaluate_rows(
                        query, key_blocks, value_blocks, dense, seed_rows
                    )
                    metrics.update(
                        evaluate_attention_divergence(
                            query,
                            key_blocks,
                            seed_rows,
                            column_chunk=args.column_chunk,
                        )
                    )
                    metrics["shared_seeds_vs_reverse_kl_mass"] = _shared_seed_count(
                        seed_rows, rows["reverse_kl_mass"]
                    )
                    point[name] = metrics
                    _add(totals[name], metrics)
                capture_record[point_name] = point
                order = sorted(point, key=lambda name: point[name]["mean_reverse_kl"])
                print(
                    f"{point_name} "
                    + " ".join(
                        f"{name}={point[name]['mean_reverse_kl']:.6f}"
                        for name in order
                    ),
                    flush=True,
                )
                del dense
                torch.cuda.empty_cache()
            del rows
            torch.cuda.empty_cache()

        records[Path(capture).parent.name] = capture_record
        selector_records[Path(capture).parent.name] = selector_record
        del q_grid, k_grid, v_grid, key_blocks_all, value_blocks_all
        torch.cuda.empty_cache()

    aggregate = {}
    for name, total in totals.items():
        aggregate[name] = {
            "rel_l2_vs_dense": math.sqrt(
                total["squared_error"] / max(total["reference_squared"], 1.0e-30)
            ),
            "mse_vs_dense": total["squared_error"] / total["numel"],
            "mean_forward_kl": total["forward_kl_sum"] / total["queries"],
            "mean_reverse_kl": total["reverse_kl_sum"] / total["queries"],
            "mean_squared_hellinger": total["hellinger_sum"] / total["queries"],
            "active_cluster_fraction": total["active_sum"] / total["queries"],
            "points": int(total["points"]),
            "queries": int(total["queries"]),
        }
    ranking = sorted(aggregate, key=lambda name: aggregate[name]["mean_reverse_kl"])
    result = {
        "settings": {
            **vars(args),
            "heads": heads,
            "frames": frames,
            "qkv": args.qkv,
            "device_name": torch.cuda.get_device_name(device),
            "all_seeds_are_real_query_rows": True,
            "router_is_shared_across_query_frames": True,
            "entropy_objective_gap": "KL(induced_gma || dense)",
            "oracle_deployable": False,
            "oracle_reason": (
                "selection reads all keys and exact dense block masses; it is an "
                "upper-bound diagnostic for the derived seed objective"
            ),
        },
        "aggregate": aggregate,
        "ranking_by_final_reverse_kl": ranking,
        "selector": selector_records,
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("\naggregate", flush=True)
    for name in ranking:
        value = aggregate[name]
        print(
            f"{name:>24s} reverse={value['mean_reverse_kl']:.8f} "
            f"forward={value['mean_forward_kl']:.8f} "
            f"rel_l2={value['rel_l2_vs_dense']:.8f} "
            f"mse={value['mse_vs_dense']:.10f}",
            flush=True,
        )
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
