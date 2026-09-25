"""Screen variational seed rules with the complete production C4 operator.

All captured queries, frames, keys, and values enter the unchanged production
merge and route kernels. Dense attention is an external diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

import torch

from attention.adaptive_seed_pytorch import multiscale_candidate_table
from attention.adaptive_seed_triton import hellinger_span_seed_rows_triton
from attention.core import flash_attention
from attention.token_merging import _hard_c4_attention
from evaluate_entropy_seed_oracle import _load_grid, _parse_indices
from partition_variational_selector import select_partition_variational_seeds
from population_span_selector import select_population_span
from sweep_simple_weighted_span_selectors import _greedy_span


BASE_MODE = "hellinger_span_alpha0p2_c32_p16_a2_j4_t1_u16_cg45_sq1_qm_fp8score_skv_nl_ml2"
JOINT_MASS_VARIANTS = ("gs_c16_256", "gs_joint2_c16_256", "gs_joint2_c32_256")


def _joint_mass_variant_config(name, frames):
    """Fixed row banks and unaveraged router frames for the named controls."""
    if name not in JOINT_MASS_VARIANTS:
        raise ValueError(f"unknown joint mass variant {name!r}")
    if name == "gs_c16_256":
        if frames <= 10:
            raise ValueError("gs_c16_256 requires router frame 10")
        return 16, 16, [10]
    if frames < 2:
        raise ValueError("joint2 requires at least two distinct frames")
    router_frames = [((2 * slot + 1) * frames) // 4 for slot in range(2)]
    count = 32 if name == "gs_joint2_c32_256" else 16
    return count, 45 if count == 32 else 16, router_frames


def _independent_frame_mass_gs(profile, log_mass, anchor, num_seeds=4):
    """Shared row pivots, independent per-frame projection coefficients.

    ``profile`` is [T, ..., candidates, probes], ``log_mass`` is
    [T, ..., candidates], and ``anchor`` is [...]. Masses have already had
    their own frame/query global log-partition subtracted. One common maximum
    across frames AND candidates stabilizes exp without changing frame mass
    ratios. At each step score_i = sum_t rho_ti ||residual_ti||^2.
    """
    if profile.ndim < 3 or tuple(log_mass.shape) != tuple(profile.shape[:-1]):
        raise ValueError("profile/log_mass must have matching frame and candidate axes")
    if tuple(anchor.shape) != tuple(profile.shape[1:-2]):
        raise ValueError("anchor must match the non-frame batch axes")
    candidate_count, probe_count = profile.shape[-2:]
    if profile.shape[0] < 1 or probe_count < 1 or not 1 <= num_seeds <= candidate_count:
        raise ValueError("require nonempty frames/probes and 1 <= num_seeds <= candidates")
    rho = torch.exp(log_mass - torch.amax(log_mass, dim=(0, -1), keepdim=True))
    candidate_ids = torch.arange(candidate_count, device=profile.device)
    used = torch.zeros_like(anchor[..., None] == candidate_ids, dtype=torch.bool)
    residual = profile
    chosen = []
    for step in range(num_seeds):
        if step == 0:
            position = anchor
        else:
            score = torch.sum(torch.sum(residual * residual, dim=-1) * rho, dim=0)
            position = torch.argmax(torch.where(used, -torch.inf, score), dim=-1)
        chosen.append(position)
        used = used | (candidate_ids == position[..., None])
        if step + 1 < num_seeds:
            selected = torch.broadcast_to(
                position[None, ..., None, None], (*residual.shape[:-2], 1, probe_count)
            )
            basis = torch.gather(residual, dim=-2, index=selected)[..., 0, :]
            basis = basis * torch.rsqrt(torch.sum(basis * basis, dim=-1, keepdim=True) + 1e-12)
            # Both the basis norm and coefficient retain their frame axis.
            coefficient = torch.sum(residual * basis[..., None, :], dim=-1, keepdim=True)
            residual = residual - coefficient * basis[..., None, :]
    return torch.stack(chosen, dim=-1)


def _joint_mass_tables(q_grid, k_grid, names, *, chunk, probes):
    """Torch-only C16/RF10 and independent-projection two-frame controls."""
    heads, frames, rows, columns, dim = q_grid.shape
    if not 1 <= probes <= columns or chunk < 1:
        raise ValueError("require a positive chunk and 1 <= probes <= columns")
    if frames * rows * columns < 256:
        raise ValueError("G256 requires at least 256 distinct key tokens")
    block_rows = torch.arange(frames * rows, device=q_grid.device) % rows
    probe_cols = torch.arange(probes, device=q_grid.device) * (columns // probes)
    output = {}
    for name in names:
        count, geometry, router_frames = _joint_mass_variant_config(name, frames)
        candidates, anchors = multiscale_candidate_table(
            rows, count, q_grid.device, geometry_count=geometry
        )
        candidate_rows, anchor_rows = candidates[block_rows].long(), anchors[block_rows].long()
        table = torch.empty((1, 1, frames, rows, columns, heads, 4),
                            device=q_grid.device, dtype=torch.uint8)
        for head in range(heads):
            # Keep [router_frame, column, row, dim] separate throughout.
            query = q_grid[head, router_frames].permute(0, 2, 1, 3).float()
            key = k_grid[head].reshape(frames * rows, columns, dim).float()
            flattened_key = key.flatten(0, 1)
            positions = torch.arange(256, device=q_grid.device) * (flattened_key.size(0) // 256)
            normalizer = (
                torch.einsum("tjrd,pd->tjrp", query, flattened_key[positions]) * dim**-0.5
            ).logsumexp(-1)
            flat_out = table[0, 0, :, :, :, head].reshape(frames * rows, columns, 4)
            for start in range(0, frames * rows, chunk):
                end = min(start + chunk, frames * rows)
                bank_rows = candidate_rows[start:end]
                logits = torch.einsum(
                    "tjbcd,bpd->tjbcp", query[:, :, bank_rows], key[start:end, probe_cols]
                ) * dim**-0.5
                profile = logits.softmax(-1).sqrt()
                profile = profile - profile.mean(-1, keepdim=True)
                mass = logits.logsumexp(-1) - normalizer[:, :, bank_rows]
                anchor = anchor_rows[start:end][None].expand(columns, -1)
                if name == "gs_c16_256":
                    # Identical one-frame GS formula to gs_256, only C16/G16.
                    selected = _greedy_span(profile[0], torch.softmax(mass[0], -1), anchor)
                else:
                    selected = _independent_frame_mass_gs(profile, mass, anchor)
                selected_rows = bank_rows[None].expand(columns, -1, -1).gather(-1, selected)
                flat_out[start:end].copy_(selected_rows.permute(1, 0, 2).to(torch.uint8))
            print(f"{name} selector head {head + 1}/{heads} complete", flush=True)
        output[name] = table
    return output


def _refine_gs_medoid(logits, selected, rho, diagnostics=None):
    """One owner-preserving pass after GS, using unchanged natural-log scores."""
    from refine_partition_seed_medoids import refine_partition_seed_medoids

    result = refine_partition_seed_medoids(
        logits, selected, rho, return_diagnostics=diagnostics is not None
    )
    if diagnostics is None:
        return result
    refined, info = result
    gain = info["weighted_local_lse_gain"]
    changed = info["changed_seed_count"]
    values = {
        "groups": changed.numel(),
        "changed_seed_slots": int(changed.sum()),
        "groups_with_changed_seeds": int((changed > 0).sum()),
        "owner_violations": int((~info["owner_preserved"]).sum()),
        "anchor_violations": int((~info["anchor_preserved"]).sum()),
        "weighted_local_lse_before_sum": float(info["weighted_local_lse_before"].double().sum()),
        "weighted_local_lse_after_sum": float(info["weighted_local_lse_after"].double().sum()),
        "weighted_local_lse_gain_sum": float(gain.double().sum()),
        "negative_gain_groups": int((gain < 0).sum()),
    }
    for key, value in values.items():
        diagnostics[key] = diagnostics.get(key, 0) + value
    minimum = float(gain.min())
    diagnostics["minimum_weighted_local_lse_gain"] = min(
        diagnostics.get("minimum_weighted_local_lse_gain", minimum), minimum
    )
    diagnostics["scope"] = (
        "One owner-preserving pass with fixed rho and initial local route probabilities; "
        "natural-log probe-local variational objective, not global attention or PSNR. "
        "Negative-gain counts are raw and may include FP32 roundoff."
    )
    return refined


def production(q, k, v, *, table=None, mode=BASE_MODE):
    kwargs = dict(h=45, w=80, h_reduce=1, w_reduce=1, sm_scale=128**-0.5,
                  query_outer_chunk=21, head_chunk=8, hard_route_backend="hard_c4",
                  exact_intra_frame_attention=False, exact_neighbor_window_columns=0,
                  exact_query_rows=0, first_frame_full_attention=False,
                  full_attention_anchor_time_count=0, full_attention_anchor_row_count=0,
                  full_attention_anchor_scale=1., seed_mode=mode)
    if table is None:
        return _hard_c4_attention(q, k, v, **kwargs)
    with patch("attention.token_merging.hellinger_span_seed_rows_triton", return_value=table):
        return _hard_c4_attention(q, k, v, **kwargs)


@torch.inference_mode()
def custom_tables(q_grid, k_grid, names, *, chunk=21, probes=16, router="rf10", diagnostics=None):
    heads, frames, rows, columns, dim = q_grid.shape
    if not 1 <= probes <= columns:
        raise ValueError("probes must be between one and the number of columns")
    if "gs_bulk_256" in names and probes < 2:
        raise ValueError("gs_bulk_256 requires at least two distinct probes")
    joint_names = [name for name in names if name in JOINT_MASS_VARIANTS]
    if joint_names and router != "rf10":
        raise ValueError("named C16/joint2 controls use fixed router frames; require --router rf10")
    joint_tables = _joint_mass_tables(q_grid, k_grid, joint_names, chunk=chunk, probes=probes) if joint_names else {}
    names = [name for name in names if name not in JOINT_MASS_VARIANTS]
    if not names:
        return joint_tables
    candidates, anchor = multiscale_candidate_table(rows, 32, q_grid.device, geometry_count=45)
    block_rows = torch.arange(frames * rows, device=q_grid.device) % rows
    candidate_rows = candidates[block_rows].long()
    anchor_rows = anchor[block_rows].long()
    if router == "rf10":
        router_q = q_grid[:, 10]
    elif router == "simpson":
        router_q = ((q_grid[:, 0].float() + 4 * q_grid[:, 10].float() + q_grid[:, 20].float()) / 6).to(q_grid.dtype)
    else:
        raise ValueError(router)
    output = {name: torch.empty((1, 1, frames, rows, columns, heads, 4),
                               device=q_grid.device, dtype=torch.uint8) for name in names}
    probe_cols = torch.arange(probes, device=q_grid.device) * (columns // probes)
    if "gs_bulk_256" in names:
        # A query-column singleton is already exact in production SQ1. Remove
        # it only if present in the existing probe set; never load extra keys.
        bulk_probe_valid = (
            probe_cols[None, None, None, :]
            != torch.arange(columns, device=q_grid.device)[:, None, None, None]
        )
        bulk_probe_count = bulk_probe_valid.sum(-1, keepdim=True)
    for head in range(heads):
        query = router_q[head].permute(1, 0, 2).float()
        key = k_grid[head].reshape(frames * rows, columns, dim).float()
        flattened_key = key.flatten(0, 1)
        normalizers = {}
        for count in (16, 256):
            positions = torch.arange(count, device=q_grid.device) * (flattened_key.size(0) // count)
            global_scores = torch.einsum("jrd,pd->jrp", query, flattened_key[positions]) * dim**-0.5
            normalizers[str(count)] = global_scores.logsumexp(-1)
        normalizers["mean"] = torch.einsum("jrd,d->jr", query, flattened_key.mean(0)) * dim**-0.5
        flat_out = {name: item[0, 0, :, :, :, head].reshape(frames * rows, columns, 4)
                    for name, item in output.items()}
        for start in range(0, frames * rows, chunk):
            end = min(start + chunk, frames * rows)
            bank_rows = candidate_rows[start:end]
            logits = torch.einsum("jbcd,bpd->jbcp", query[:, bank_rows], key[start:end, probe_cols]) * dim**-0.5
            anc = anchor_rows[start:end][None].expand(columns, -1)
            mass = logits.logsumexp(-1)
            z = mass - mass.mean(-1, keepdim=True)
            weight = torch.exp(.4 * z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6))
            for name in names:
                if name.startswith("gs_") or name.startswith("deficit_"):
                    policy = name.rsplit("_", 1)[-1]
                    if name == "gs_bulk_256":
                        local_logits = logits.masked_fill(~bulk_probe_valid, -torch.inf)
                        local_mass = local_logits.logsumexp(-1)
                        profile = local_logits.softmax(-1).sqrt()
                        profile = torch.where(
                            bulk_probe_valid,
                            profile - profile.sum(-1, keepdim=True) / bulk_probe_count,
                            0.0,
                        )
                    else:
                        local_mass = mass
                        profile = logits.softmax(-1).sqrt()
                        profile = profile - profile.mean(-1, keepdim=True)
                    normalized_mass = local_mass if policy == "raw" else local_mass - normalizers[policy][:, bank_rows]
                    if name.startswith("gs_z_") or name.startswith("deficit_z_"):
                        deviation = normalized_mass - normalized_mass.mean(-1, keepdim=True)
                        derived_weight = torch.exp(.4 * deviation * torch.rsqrt(deviation.square().mean(-1, keepdim=True) + 1e-6))
                    else:
                        derived_weight = torch.softmax(normalized_mass, -1)
                    if name.startswith("gs_"):
                        selected = _greedy_span(profile, derived_weight, anc)
                        if name == "gs_medoid_256":
                            # This is exactly gs_256's initialization and rho.
                            # logits are QK/sqrt(D), with no log2 conversion.
                            # The refiner fixes selected[..., 0], the same anc.
                            info = diagnostics.setdefault(name, {}) if diagnostics is not None else None
                            selected = _refine_gs_medoid(logits, selected, derived_weight, info)
                    else:
                        selected = select_partition_variational_seeds(logits, anc, variant="gap_deficit_weighted", weights=derived_weight)
                elif name.startswith("population"):
                    selected = select_population_span(logits, anc, weights=weight if name.endswith("weighted") else None)
                else:
                    selected = select_partition_variational_seeds(logits, anc, variant=name,
                                                                  weights=weight if name.endswith("weighted") else None)
                selected_rows = bank_rows[None].expand(columns, -1, -1).gather(-1, selected)
                flat_out[name][start:end].copy_(selected_rows.permute(1, 0, 2).to(torch.uint8))
        print(f"selector head {head + 1}/{heads} complete", flush=True)
    output.update(joint_tables)
    return output


def metrics(output, dense):
    error = (output.float() - dense.float()).square()
    ref = dense.float().square()
    frame_mse = error.reshape(1, 21, 45 * 80, -1).mean((0, 2, 3))
    return {"mse": float(error.mean()), "relative_l2": float((error.sum() / ref.sum()).sqrt()),
            "frame_mse": frame_mse.tolist(), "head_mse": error.mean((0, 1, 3)).tolist(),
            "numel": error.numel()}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", nargs="+", required=True)
    parser.add_argument("--heads", default="0,10,20,30")
    parser.add_argument("--variants", default="gap_farthest,gap_weighted,gap_gain,gap_gain_weighted,population,population_weighted",
                        help="Comma-separated rules; gs_bulk_256 excludes SQ1 probes; "
                             "gs_c16_256, gs_joint2_c16_256, gs_joint2_c32_256 use fixed RF10/two-frame banks")
    parser.add_argument("--probes", type=int, default=16)
    parser.add_argument("--router", default="rf10", choices=("rf10", "simpson"))
    parser.add_argument("--block-chunk", type=int, default=7)
    parser.add_argument("--include-temporal", action="store_true")
    parser.add_argument("--include-key-matched", action="store_true")
    parser.add_argument("--include-transport", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    heads = _parse_indices(args.heads)
    names = [s for s in args.variants.split(",") if s]
    all_records = {}
    for capture in args.qkv:
        started = time.perf_counter()
        grids = [x.to(device="cuda", dtype=torch.bfloat16) for x in _load_grid(capture, heads)]
        qg, kg, vg = grids
        tensors = [g.permute(1, 2, 3, 0, 4).reshape(1, 21 * 45 * 80, len(heads), 128).contiguous() for g in grids]
        q, k, v = tensors
        dense = flash_attention(q, k, v, softmax_scale=128**-0.5)
        torch.cuda.synchronize()
        print(f"loaded dense {capture} in {time.perf_counter() - started:.1f}s", flush=True)
        result = {}
        baseline = production(q, k, v)
        result["rf10"] = metrics(baseline, dense)
        result["simpson"] = metrics(production(q, k, v, mode=BASE_MODE.replace("_u16_cg", "_u16_ts3_cg")), dense)
        if args.include_key_matched:
            native_q = q.reshape(1, 21, 45, 80, len(heads), 128)
            native_k = k.reshape(1, 21, 45, 80, len(heads), 128)
            matched_tables = []
            for frame in range(21):
                matched_tables.append(hellinger_span_seed_rows_triton(
                    native_q[:, frame:frame+1], native_k[:, frame:frame+1],
                    alpha=.2, candidate_count=32, candidate_geometry_count=45,
                    top_probes=0, uniform_probes=16, block_j=4, group_tile=32,
                    output_dtype=torch.uint8, gs_num_warps=8))
            matched = torch.cat(matched_tables, dim=2)
            result["key_matched"] = metrics(production(q, k, v, table=matched), dense)
            del matched_tables, matched
        if args.include_transport:
            from temporal_transport_seeds import temporal_transport_router
            native_k = k.reshape(1, 21, 45, 80, len(heads), 128)
            for transport_mode in ("mean21", "simpson3"):
                transported = temporal_transport_router(qg, mode=transport_mode)
                native_router = transported.permute(1, 2, 3, 0, 4).unsqueeze(0).contiguous()
                table = hellinger_span_seed_rows_triton(
                    native_router, native_k, alpha=.2, candidate_count=32,
                    candidate_geometry_count=45, top_probes=0, uniform_probes=16,
                    block_j=4, group_tile=32, output_dtype=torch.uint8, gs_num_warps=8)
                result[f"transport_{transport_mode}"] = metrics(production(q, k, v, table=table), dense)
        if args.include_temporal:
            for count in (3, 4):
                result[f"hellinger_t{count}"] = metrics(production(q, k, v, mode=BASE_MODE.replace("_t1_", f"_t{count}_")), dense)
        selector_diagnostics = {}
        tables = custom_tables(qg, kg, names, chunk=args.block_chunk, probes=args.probes,
                               router=args.router, diagnostics=selector_diagnostics)
        for name, table in tables.items():
            result[name] = metrics(production(q, k, v, table=table), dense)
            if name in JOINT_MASS_VARIANTS:
                count, geometry, router_frames = _joint_mass_variant_config(name, qg.shape[1])
                result[name]["selector_metadata"] = {
                    "candidate_count": count, "candidate_geometry_count": geometry,
                    "router_frames": router_frames, "local_probe_count": args.probes,
                    "global_probe_count": 256, "seed_count": 4,
                    "fixed_matched_row_anchor": True, "raw_query_averaging": False,
                    "projection_rule": "independent_per_frame" if len(router_frames) > 1 else "single_frame",
                    "mass_rule": "exp_one_common_frame_candidate_max" if len(router_frames) > 1 else "candidate_softmax",
                }
            if name in selector_diagnostics:
                info = selector_diagnostics[name]
                info["mean_weighted_local_lse_gain"] = info["weighted_local_lse_gain_sum"] / info["groups"]
                result[name]["selector_diagnostics"] = info
        base = result["rf10"]["mse"]
        for name, value in result.items():
            value["mse_change_vs_rf10"] = value["mse"] / base - 1
            print(f"{Path(capture).parent.name} {name}: mse={value['mse']:.9f} change={value['mse_change_vs_rf10']:+.2%}", flush=True)
        all_records[capture] = result
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"settings": vars(args), "records": all_records}, default=str, indent=2) + "\n")
        del grids, qg, kg, vg, tensors, q, k, v, dense, baseline, tables
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
