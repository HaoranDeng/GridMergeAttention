"""Approximate the best key-independent C4 seed positions for each Q block.

The deployable object studied here is one set of four query-row indices for
each ``(layer, head, query frame, query column)``.  Those four indices are
shared over every key frame and key row.  The offline oracle is allowed to
read real keys only while choosing the indices; the resulting table itself
contains no key-block-dependent entries.

For a representative subset of key blocks, every query row is represented by
its blockwise attention distributions.  We then solve the complete
``45 choose 4`` Hellinger k-medoids problem.  Final quality is measured with
the repository's exact ``KL(dense || induced_gma)`` evaluator over all 945 key
blocks, so the Hellinger objective is only a search surrogate.
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

from analysis.evaluate_query_only_seed_kl import (
    C4,
    COLS,
    FRAMES,
    ROWS,
    _aggregate,
    _global_farthest_order,
    _load_qk,
    _pairwise_squared,
    _production_wide_rows,
    evaluate_attention_kl,
)


def _parse_indices(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split()]


def _key_block_indices(frames: list[int], rows: list[int]) -> list[int]:
    if any(not 0 <= frame < FRAMES for frame in frames):
        raise ValueError(f"key frames must be in [0, {FRAMES})")
    if any(not 0 <= row < ROWS for row in rows):
        raise ValueError(f"key rows must be in [0, {ROWS})")
    return [frame * ROWS + row for frame in frames for row in rows]


@torch.inference_mode()
def attention_hellinger_distance(
    query: torch.Tensor,
    key_blocks: torch.Tensor,
) -> torch.Tensor:
    """Return mean blockwise squared Hellinger distance ``[column,45,45]``."""

    if query.shape[:2] != (COLS, ROWS):
        raise ValueError("query must have shape [80,45,D]")
    if key_blocks.ndim != 3 or key_blocks.size(1) != COLS:
        raise ValueError("key_blocks must have shape [blocks,80,D]")
    scale = query.size(-1) ** -0.5
    logits = torch.einsum(
        "jrd,bld->jrbl", query.float(), key_blocks.float()
    ) * scale
    embedding = logits.softmax(dim=-1).sqrt().flatten(start_dim=2)
    embedding = embedding / math.sqrt(key_blocks.size(0))
    # Squared Euclidean distance between sqrt-probabilities is twice the
    # conventional squared Hellinger distance.  The constant does not alter
    # the optimal medoids.
    return _pairwise_squared(embedding)


@torch.inference_mode()
def attention_forward_kl_distance(
    query: torch.Tensor,
    key_blocks: torch.Tensor,
) -> torch.Tensor:
    """Return mean blockwise directed KL from each demand row to each medoid."""

    scale = query.size(-1) ** -0.5
    logits = torch.einsum(
        "jrd,bld->jrbl", query.float(), key_blocks.float()
    ) * scale
    log_probability = logits.log_softmax(dim=-1)
    probability = log_probability.exp()
    flat_probability = probability.flatten(start_dim=2)
    flat_log_probability = log_probability.flatten(start_dim=2)
    block_count = key_blocks.size(0)
    self_term = (
        flat_probability * flat_log_probability
    ).sum(dim=-1) / block_count
    cross_term = torch.einsum(
        "jrn,jsn->jrs", flat_probability, flat_log_probability
    ) / block_count
    return (self_term.unsqueeze(-1) - cross_term).clamp_min_(0.0)


@torch.inference_mode()
def exact_c4_pmedian(
    distance: torch.Tensor,
    combinations: torch.Tensor,
    combo_chunk: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve all ``45 choose 4`` uniform p-median problems in parallel."""

    if distance.ndim != 3 or distance.shape[-2:] != (ROWS, ROWS):
        raise ValueError("distance must have shape [batch,45,45]")
    if combinations.ndim != 2 or combinations.size(1) != C4:
        raise ValueError("combinations must have shape [n,4]")
    best_cost = torch.full(
        (distance.size(0),), float("inf"), device=distance.device
    )
    best_combination = torch.zeros(
        distance.size(0), dtype=torch.long, device=distance.device
    )
    for start in range(0, combinations.size(0), combo_chunk):
        candidates = combinations[start : start + combo_chunk]
        # Advanced indexing yields [batch, demand_row, combination, medoid].
        cost = distance[:, :, candidates].amin(dim=-1).mean(dim=1)
        chunk_cost, chunk_index = cost.min(dim=-1)
        improved = chunk_cost < best_cost
        best_cost = torch.where(improved, chunk_cost, best_cost)
        best_combination = torch.where(
            improved, chunk_index + start, best_combination
        )
    return combinations[best_combination], best_cost


def _repeat_over_key_blocks(rows: torch.Tensor) -> torch.Tensor:
    if rows.shape != (COLS, C4):
        raise ValueError("rows must have shape [80,4]")
    return rows[:, None, :].expand(-1, FRAMES * ROWS, -1)


def _fixed_wide_table(device: torch.device) -> torch.Tensor:
    key_row = torch.arange(FRAMES * ROWS, device=device) % ROWS
    return _production_wide_rows(device)[key_row][None].expand(COLS, -1, -1)


@torch.inference_mode()
def evaluate_attention_kl_per_column(
    query: torch.Tensor,
    key: torch.Tensor,
    rows: torch.Tensor,
    column_chunk: int = 4,
) -> torch.Tensor:
    """Return exact induced-GMA mean KL for every query column."""

    scale = query.size(-1) ** -0.5
    values = []
    for start in range(0, query.size(0), column_chunk):
        end = min(start + column_chunk, query.size(0))
        query_chunk = query[start:end].float()
        rows_chunk = rows[start:end]
        columns = torch.arange(end - start, device=query.device)[:, None, None]
        seeds = query_chunk[columns, rows_chunk].float()
        seed_logits = torch.einsum("jbcd,bld->jbcl", seeds, key.float()) * scale
        owner = torch.log_softmax(seed_logits, dim=-1).argmax(dim=-2)
        cluster_id = torch.arange(C4, device=query.device).view(1, 1, C4, 1)
        assignment = owner.unsqueeze(-2) == cluster_id
        active = assignment.any(dim=-1)
        masked_seed_logits = seed_logits.masked_fill(~assignment, -float("inf"))
        safe_seed_logits = torch.where(
            active.unsqueeze(-1), masked_seed_logits, torch.zeros_like(masked_seed_logits)
        )
        seed_log_partition = torch.logsumexp(masked_seed_logits, dim=-1)
        seed_log_probability = torch.log_softmax(safe_seed_logits, dim=-1)
        seed_probability = torch.where(
            assignment, seed_log_probability.exp(), torch.zeros_like(seed_log_probability)
        )
        merged_key = torch.einsum("jbcl,bld->jbcd", seed_probability, key.float())
        tangent_bias = (seeds * merged_key).sum(-1) * scale - seed_log_partition
        cluster_logits = (
            torch.einsum("jrd,jbcd->jrbc", query_chunk, merged_key) * scale
            - tangent_bias.unsqueeze(1)
        ).masked_fill(~active.unsqueeze(1), -float("inf"))
        cluster_log_probability = torch.log_softmax(
            cluster_logits.flatten(-2), dim=-1
        ).reshape_as(cluster_logits)
        dense_logits = torch.einsum("jrd,bld->jrbl", query_chunk, key.float()) * scale
        dense_log_partition = torch.logsumexp(dense_logits.flatten(-2), dim=-1)
        dense_log_probability = dense_logits - dense_log_partition[..., None, None]
        dense_probability = dense_log_probability.exp()
        owner_for_queries = owner.unsqueeze(1).expand(
            -1, query_chunk.size(1), -1, -1
        )
        cluster_log_probability_for_key = cluster_log_probability.gather(
            -1, owner_for_queries
        )
        within_log_probability_for_key = (
            seed_log_probability.permute(0, 1, 3, 2)
            .gather(-1, owner.unsqueeze(-1))
            .squeeze(-1)
            .unsqueeze(1)
        )
        induced_log_probability = (
            cluster_log_probability_for_key + within_log_probability_for_key
        )
        full_kl = (
            dense_probability * (dense_log_probability - induced_log_probability)
        ).sum(dim=(-2, -1)).clamp_min(0.0)
        values.append(full_kl.mean(dim=-1).cpu())
    return torch.cat(values)


def _select_proposal_per_column(
    proposals: dict[str, torch.Tensor],
    proposal_kl: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, int]]:
    names = list(proposals)
    costs = torch.stack([proposal_kl[name] for name in names])
    winner = costs.argmin(dim=0)
    stacked_rows = torch.stack([proposals[name].cpu() for name in names])
    columns = torch.arange(COLS)
    selected = stacked_rows[winner, columns]
    counts = {name: int((winner == index).sum()) for index, name in enumerate(names)}
    return selected, counts


def _set_agreement(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    if left.shape != right.shape or left.size(-1) != C4:
        raise ValueError("seed sets must have matching [...,4] shapes")
    overlap = (
        left[..., :, None] == right[..., None, :]
    ).any(dim=-1).sum(dim=-1).float()
    exact = (
        left.sort(dim=-1).values == right.sort(dim=-1).values
    ).all(dim=-1)
    return {
        "mean_overlap_of_4": float(overlap.mean()),
        "mean_overlap_fraction": float((overlap / C4).mean()),
        "mean_jaccard": float((overlap / (2 * C4 - overlap)).mean()),
        "exact_set_fraction": float(exact.float().mean()),
    }


def _position_summary(all_rows: torch.Tensor) -> dict[str, object]:
    rows = all_rows.to(torch.long).reshape(-1, C4)
    histogram = torch.bincount(rows.flatten(), minlength=ROWS).to(torch.float64)
    probability = histogram / histogram.sum().clamp_min(1)
    support = probability > 0
    entropy = float(
        -(probability[support] * probability[support].log()).sum()
    )
    ordered_rows = rows.sort(dim=-1).values.to(torch.float64)
    pairwise = torch.stack(
        [
            (ordered_rows[:, right] - ordered_rows[:, left]).abs()
            for left in range(C4)
            for right in range(left + 1, C4)
        ],
        dim=-1,
    )
    top_values, top_indices = histogram.topk(12)
    return {
        "sets": rows.size(0),
        "normalized_entropy": entropy / math.log(ROWS),
        "effective_support": math.exp(entropy),
        "top_rows": [
            {
                "row": int(index),
                "count": int(value),
                "mass": float(value / histogram.sum()),
            }
            for value, index in zip(top_values, top_indices)
        ],
        "sorted_slot_mean": ordered_rows.mean(dim=0).tolist(),
        "sorted_slot_std": ordered_rows.std(dim=0).tolist(),
        "mean_span": float(
            (ordered_rows[:, -1] - ordered_rows[:, 0]).mean()
        ),
        "mean_pairwise_row_distance": float(pairwise.mean()),
        "row_histogram": histogram.to(torch.long).tolist(),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", default="0,10,20,30")
    parser.add_argument("--frames", default="0,10,20")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--column_chunk", type=int, default=4)
    parser.add_argument("--combo_chunk", type=int, default=1024)
    parser.add_argument("--search_a_frames", default="0,3,6,9,12,15,18,20")
    parser.add_argument("--search_a_rows", default="0,6,12,18,24,30,36,44")
    parser.add_argument("--search_b_frames", default="1,4,7,10,13,16,19,20")
    parser.add_argument("--search_b_rows", default="3,9,15,21,27,33,39,42")
    args = parser.parse_args()

    heads = _parse_indices(args.heads)
    frames = _parse_indices(args.frames)
    search_a = _key_block_indices(
        _parse_indices(args.search_a_frames),
        _parse_indices(args.search_a_rows),
    )
    search_b = _key_block_indices(
        _parse_indices(args.search_b_frames),
        _parse_indices(args.search_b_rows),
    )
    device = torch.device(args.device)
    combinations = torch.combinations(
        torch.arange(ROWS, device=device), r=C4
    )
    totals: dict[str, list[dict[str, float]]] = defaultdict(list)
    cases: dict[str, dict[str, object]] = {}
    oracle_rows_all = []
    agreements = []
    search_seconds = []
    proposal_winner_counts: dict[str, int] = defaultdict(int)

    for capture_index, path in enumerate(args.qkv, start=1):
        q_heads, k_heads = _load_qk(path, heads)
        for head_slot, head in enumerate(heads):
            q_grid = q_heads[head_slot].reshape(FRAMES, ROWS, COLS, -1).to(device)
            k_grid = k_heads[head_slot].reshape(FRAMES, ROWS, COLS, -1).to(device)
            key_blocks = k_grid.reshape(FRAMES * ROWS, COLS, -1)
            fixed_wide = _fixed_wide_table(device)
            for frame in frames:
                query = q_grid[frame].permute(1, 0, 2).contiguous()
                started = time.perf_counter()
                distance_a = attention_hellinger_distance(
                    query, key_blocks[search_a]
                )
                hellinger_a, surrogate_a = exact_c4_pmedian(
                    distance_a, combinations, args.combo_chunk
                )
                distance_b = attention_hellinger_distance(
                    query, key_blocks[search_b]
                )
                hellinger_b, surrogate_b = exact_c4_pmedian(
                    distance_b, combinations, args.combo_chunk
                )
                search_ab = sorted(set(search_a + search_b))
                hellinger_ab, _ = exact_c4_pmedian(
                    attention_hellinger_distance(query, key_blocks[search_ab]),
                    combinations,
                    args.combo_chunk,
                )
                forward_kl_a, _ = exact_c4_pmedian(
                    attention_forward_kl_distance(query, key_blocks[search_a]),
                    combinations,
                    args.combo_chunk,
                )
                forward_kl_b, _ = exact_c4_pmedian(
                    attention_forward_kl_distance(query, key_blocks[search_b]),
                    combinations,
                    args.combo_chunk,
                )
                forward_kl_ab, _ = exact_c4_pmedian(
                    attention_forward_kl_distance(query, key_blocks[search_ab]),
                    combinations,
                    args.combo_chunk,
                )
                elapsed = time.perf_counter() - started
                agreement = _set_agreement(hellinger_a, hellinger_b)
                agreements.append(agreement)
                search_seconds.append(elapsed)

                qglobal = _global_farthest_order(query[..., :32], C4)
                proposals = {
                    "hellinger_a": hellinger_a,
                    "hellinger_b": hellinger_b,
                    "hellinger_ab": hellinger_ab,
                    "forward_kl_a": forward_kl_a,
                    "forward_kl_b": forward_kl_b,
                    "forward_kl_ab": forward_kl_ab,
                    "query_global_d32": qglobal,
                }
                proposal_kl = {
                    name: evaluate_attention_kl_per_column(
                        query,
                        key_blocks,
                        _repeat_over_key_blocks(rows),
                        args.column_chunk,
                    )
                    for name, rows in proposals.items()
                }
                refined_cpu, winner_counts = _select_proposal_per_column(
                    proposals, proposal_kl
                )
                refined = refined_cpu.to(device)
                oracle_rows_all.append(refined_cpu)
                for name, count in winner_counts.items():
                    proposal_winner_counts[name] += count
                method_rows = {
                    "fixed_wide_keyrow_structural": fixed_wide,
                    "query_only_global_c4_d32": _repeat_over_key_blocks(qglobal),
                    "offline_k_aware_fixed_c4_hellinger": _repeat_over_key_blocks(
                        hellinger_a
                    ),
                    "offline_k_aware_fixed_c4_refined_oracle": _repeat_over_key_blocks(
                        refined
                    ),
                }
                metrics = {}
                for name, rows in method_rows.items():
                    value = evaluate_attention_kl(
                        query, key_blocks, rows, args.column_chunk
                    )
                    metrics[name] = value
                    totals[name].append(value)

                case_name = f"{path.stem}:h{head}:f{frame}"
                cases[case_name] = {
                    "metrics": metrics,
                    "oracle_rows_by_column": refined_cpu.tolist(),
                    "hellinger_a_rows_by_column": hellinger_a.cpu().tolist(),
                    "hellinger_b_rows_by_column": hellinger_b.cpu().tolist(),
                    "search_subset_agreement": agreement,
                    "proposal_winner_counts": winner_counts,
                    "search_surrogate_mean_a": float(surrogate_a.mean()),
                    "search_surrogate_mean_b": float(surrogate_b.mean()),
                    "search_seconds": elapsed,
                }
                print(
                    f"{case_name} "
                    f"fixed={metrics['fixed_wide_keyrow_structural']['mean_key_kl']:.6f} "
                    f"qglobal={metrics['query_only_global_c4_d32']['mean_key_kl']:.6f} "
                    f"h-oracle={metrics['offline_k_aware_fixed_c4_hellinger']['mean_key_kl']:.6f} "
                    f"refined={metrics['offline_k_aware_fixed_c4_refined_oracle']['mean_key_kl']:.6f} "
                    f"overlap={agreement['mean_overlap_of_4']:.3f}/4 "
                    f"search={elapsed:.2f}s",
                    flush=True,
                )
                del distance_a, distance_b, hellinger_a, hellinger_b
                torch.cuda.empty_cache()
            del q_grid, k_grid, key_blocks
            torch.cuda.empty_cache()
        print(
            f"processed {capture_index}/{len(args.qkv)} {path.name}",
            flush=True,
        )

    # _aggregate expects the conventional baseline name.
    aggregate_input = {
        "fixed_wide": totals["fixed_wide_keyrow_structural"],
        "query_only_global_c4_d32": totals["query_only_global_c4_d32"],
        "offline_k_aware_fixed_c4_hellinger": totals[
            "offline_k_aware_fixed_c4_hellinger"
        ],
        "offline_k_aware_fixed_c4_refined_oracle": totals[
            "offline_k_aware_fixed_c4_refined_oracle"
        ],
    }
    aggregate = _aggregate(aggregate_input)
    aggregate["fixed_wide_keyrow_structural"] = aggregate.pop("fixed_wide")
    baseline = aggregate["fixed_wide_keyrow_structural"]["mean_key_kl"]
    for value in aggregate.values():
        value["mean_key_kl_ratio_vs_fixed"] = value["mean_key_kl"] / baseline

    result = {
        "settings": {
            **vars(args),
            "qkv": [str(path) for path in args.qkv],
            "heads": heads,
            "frames": frames,
            "search_a_key_block_indices": search_a,
            "search_b_key_block_indices": search_b,
            "search_blocks_per_split": len(search_a),
            "combinations_per_q_block": combinations.size(0),
            "device_name": torch.cuda.get_device_name(device),
            "attention_kl_direction": "KL(dense || induced_gma)",
        },
        "interpretation": {
            "selector_fixed_per_q_block": True,
            "seed_rows_shared_over_all_key_blocks": True,
            "offline_search_reads_key": True,
            "deployable_q_only": False,
            "all_seeds_are_real_query_rows": True,
            "search_objective": (
                "exact C(45,4) uniform p-median under mean blockwise "
                "attention Hellinger or forward-KL distance on sampled key blocks"
            ),
            "final_objective": (
                "exact induced-GMA attention KL on all 945 key blocks"
            ),
            "refinement": (
                "choose one of seven fixed-C4 proposals independently for each "
                "Q block using exact full-key induced-GMA KL"
            ),
        },
        "aggregate_kl": aggregate,
        "position_summary": _position_summary(torch.stack(oracle_rows_all)),
        "search_split_stability": {
            key: float(np.mean([item[key] for item in agreements]))
            for key in agreements[0]
        },
        "search_runtime": {
            "mean_seconds_per_head_frame": float(np.mean(search_seconds)),
            "total_seconds": float(np.sum(search_seconds)),
        },
        "proposal_winner_counts": dict(proposal_winner_counts),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({
        "aggregate_kl": result["aggregate_kl"],
        "position_summary": result["position_summary"],
        "search_split_stability": result["search_split_stability"],
        "search_runtime": result["search_runtime"],
    }, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
