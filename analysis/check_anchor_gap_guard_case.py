"""Synthetic guard-case checks, executed only by the monitored native parent.

No video or timing conclusion follows. Actual Anchor32 is the incumbent;
the correction is compared to an independent FP64 partition oracle.
"""
import hashlib

GAP_ATOL = 2e-10
GAP_RTOL = 2e-11
MISMATCH_DIAGNOSTIC_LIMIT = 8


def append_decision_diagnostic(diagnostics, payload):
    """Keep at most eight failed banks per case; never affect acceptance.

    The caller copies original FP64 tensors directly to Python float lists,
    without FP32 conversion. JSON preserves them for pure-CPU replay. Native
    base/trial means are recomputed from audit vectors by Torch FP64, NOT read
    from the kernel's reduction scalars; full vectors are retained so that
    this reduction-order distinction is visible during forensic analysis.
    """
    if len(diagnostics) >= MISMATCH_DIAGNOSTIC_LIMIT:
        return False
    diagnostics.append(payload)
    print(f"decision_mismatch bank={payload['bank_id']} "
          f"index=({payload['frame']},{payload['row']},{payload['col']},{payload['head']}) "
          f"actual={payload['actual_decision']} expected={payload['expected_decision']}", flush=True)
    return True


def validate_guard_record(record, banks, frames):
    zero_fields = ("anchor_mismatches", "nonunique_banks", "invalid_candidate_banks",
                   "prefix_mismatched_banks", "decision_mismatched_banks", "self_accepted_banks",
                   "row_change_decision_mismatched_banks",
                   "score_audit_mismatched_banks")
    if (any(record.get(name) != 0 for name in zero_fields)
            or record.get("score_audit_on_off_equal") is not True
            or record.get("score_comparison_passed") is not True
            or record.get("score_atol") != GAP_ATOL or record.get("score_rtol") != GAP_RTOL
            or record.get("score_values_checked") != banks * 3 * 32
            or not 0 <= record.get("max_score_tolerance_ratio", float("inf")) <= 1
            or record.get("frame_banks") != [7200] * frames
            or record.get("frame_mismatched_banks") != [0] * frames
            or record.get("q_dtype") != "torch.bfloat16"
            or record.get("k_dtype") != "torch.bfloat16"
            or record.get("k_shape") != [1, frames, 45, 80, 2, 128]
            or record.get("singleton_already_probed_banks") != banks // 5
            or record.get("singleton_added_banks") != banks * 4 // 5
            or not 0 < record.get("accepted_banks", -1) < banks
            or record.get("incumbent_implementation") != "fisher_anchor_seed_rows_triton"):
        raise RuntimeError("anchor gap guard structure, decisions, or fixed FP64 gap gate failed")


def check_guard_case(torch, seed, frames, scale):
    from analysis.anchor_gap_guard_seed import anchor_gap_guard
    from analysis.check_barycentric_native import Q_SHAPE, decode_bank_ids
    from attention.adaptive_seed_pytorch import multiscale_candidate_table
    from attention.affine_hellinger_seed_triton import fisher_anchor_seed_rows_triton
    from attention.anchor_gap_guard_seed_triton import anchor_gap_guard_seed_rows_triton

    torch.manual_seed(seed)
    q = torch.randint(-4, 5, Q_SHAPE, device="cuda").to(torch.bfloat16)
    k_shape = (1, frames, 45, 80, 2, 128)
    k = torch.randint(-4, 5, k_shape, device="cuda").to(torch.bfloat16)
    kwargs = dict(candidate_count=32, candidate_geometry_count=45, block_j=4,
                  probe_count=16, global_probe_count=256, gs_num_warps=4)
    incumbent = fisher_anchor_seed_rows_triton(q, k, scale, **kwargs)
    native = anchor_gap_guard_seed_rows_triton(q, k, scale, **kwargs)
    gaps = torch.full((1, 1, frames, 45, 80, 2, 3, 32), float("nan"),
                      dtype=torch.float64, device=q.device)
    decisions = torch.full((1, 1, frames, 45, 80, 2, 3), -99,
                           dtype=torch.int32, device=q.device)
    audited = anchor_gap_guard_seed_rows_triton(q, k, scale, gap_audit=gaps,
                                              decision_audit=decisions, **kwargs)
    shape = (1, 1, frames, 45, 80, 2, 4)
    for value in (incumbent, native, audited):
        if tuple(value.shape) != shape or value.dtype != torch.uint8:
            raise AssertionError("wrong real-row table shape or dtype")
    incumbent, native, audited = (value.reshape(-1, 4) for value in (incumbent, native, audited))
    gaps, decisions = gaps.reshape(-1, 3, 32), decisions.reshape(-1, 3)
    banks = len(native)
    table, _ = multiscale_candidate_table(45, 32, q.device, geometry_count=45)
    expected = torch.empty_like(native)
    decision_errors = invalid = accepted = self_accepted = 0
    decision_diagnostics = []
    max_error = max_ratio = 0.0
    passed = True
    for start in range(0, banks, 256):
        all_ids = torch.arange(start, min(start + 256, banks), device=q.device)
        _, _, all_cols, _ = decode_bank_ids(all_ids)
        # Distinct supports: 16 bins when SQ is in U16, otherwise 17 bins.
        # No zero-measure duplicate bin is passed to the independent reference.
        for already_probed in (True, False):
            ids = all_ids[(all_cols % 5 == 0) == already_probed]
            if not len(ids):
                continue
            frame, row, col, head = decode_bank_ids(ids)
            real = table[row].long()
            support = torch.arange(0, 80, 5, device=q.device)[None].expand(len(ids), -1)
            if already_probed:
                singleton = col // 5
                measures = torch.full((len(ids), 16), 79 / 15, dtype=torch.float64, device=q.device)
            else:
                support = torch.cat((support, col[:, None]), -1)
                singleton = torch.full_like(col, 16)
                measures = torch.full((len(ids), 17), 79 / 16, dtype=torch.float64, device=q.device)
            measures.scatter_(1, singleton[:, None], 1.0)
            qq = q[0, 0, real, col[:, None], head[:, None]].float()
            kk = k[0, frame[:, None], row[:, None], support, head[:, None]].float()
            logits = torch.bmm(qq, kk.transpose(1, 2)).double() * scale
            membership = real[:, :, None] == incumbent[ids, None, :].long()
            if not bool((membership.sum(1) == 1).all()):
                raise AssertionError("incumbent outside candidate table")
            slots = membership.long().argmax(1)
            oracle = anchor_gap_guard(logits, slots, singleton, measures=measures)
            expected[ids] = real.gather(1, oracle["seeds"]).to(torch.uint8)
            wanted_decision = torch.stack((oracle["challenger"], oracle["accepted"].long(), slots[:, -1]), -1)
            decision_disagreement = (decisions[ids].long() != wanted_decision).any(-1)
            decision_errors += int(decision_disagreement.sum())
            # Compare against the independently recovered actual incumbent,
            # not the native decision audit's claimed old-slot field.
            self_accepted += int(((decisions[ids, 0] == slots[:, -1]) & (decisions[ids, 1] != 0)).sum())
            accepted += int(oracle["accepted"].sum())
            wanted_gaps = torch.stack([oracle[name] for name in ("prefix_gap", "base_gap", "trial_gap")], 1)
            observed = gaps[ids]
            if not bool(torch.isfinite(observed).all()) or not bool(torch.isfinite(wanted_gaps).all()):
                raise AssertionError("missing or nonfinite gap audit")
            error = (observed - wanted_gaps).abs()
            ratio = error / (GAP_ATOL + GAP_RTOL * wanted_gaps.abs())
            max_error = max(max_error, float(error.max()))
            max_ratio = max(max_ratio, float(ratio.max()))
            passed = passed and bool((ratio <= 1).all())
            invalid += int(((real[:, :, None] == native[ids, None, :].long()).sum(1) != 1).any(-1).sum())
            remaining = MISMATCH_DIAGNOSTIC_LIMIT - len(decision_diagnostics)
            if remaining > 0:
                # Index copies are bounded before any logits/gap host transfer.
                positions = decision_disagreement.nonzero().flatten()[:remaining].cpu().tolist()
                for position in positions:
                    actual_gap_vectors = observed[position].cpu().tolist()
                    oracle_gap_vectors = wanted_gaps[position].cpu().tolist()
                    actual_means = observed[position].mean(-1).cpu().tolist()
                    append_decision_diagnostic(decision_diagnostics, {
                        "bank_id": int(ids[position]), "frame": int(frame[position]),
                        "row": int(row[position]), "col": int(col[position]), "head": int(head[position]),
                        "actual_decision": decisions[ids[position]].cpu().tolist(),
                        "expected_decision": wanted_decision[position].cpu().tolist(),
                        "decision_fields": ["challenger_candidate_slot", "accepted", "incumbent_last_slot"],
                        "incumbent_slots": slots[position].cpu().tolist(),
                        "candidate_rows": real[position].cpu().tolist(),
                        "actual_prefix_gap": actual_gap_vectors[0],
                        "actual_base_gap": actual_gap_vectors[1],
                        "actual_trial_gap": actual_gap_vectors[2],
                        "oracle_prefix_gap": oracle_gap_vectors[0],
                        "oracle_base_gap": oracle_gap_vectors[1],
                        "oracle_trial_gap": oracle_gap_vectors[2],
                        "actual_base_mean": actual_means[1], "actual_trial_mean": actual_means[2],
                        "oracle_base_mean": float(oracle["base_mean"][position]),
                        "oracle_trial_mean": float(oracle["trial_mean"][position]),
                        "actual_mean_source": "Torch FP64 mean of native audit vectors; not native reduction scalars",
                        "logits": logits[position].cpu().tolist(),
                        "logits_dtype": str(logits.dtype), "measures": measures[position].cpu().tolist(),
                        "measures_dtype": str(measures.dtype), "singleton_index": int(singleton[position]),
                        "support_columns": support[position].cpu().tolist(), "scale": scale,
                    })
    equal = (native == expected).all(-1)
    audit_equal = (native == audited).all(-1)
    _, rows, _, _ = decode_bank_ids(torch.arange(banks, device=q.device))
    record = {"input_seed": seed, "key_frames": frames, "scale": scale,
        "method": "anchor_gap_guard", "banks": banks, "q_dtype": str(q.dtype), "k_dtype": str(k.dtype),
        "k_shape": list(k.shape), "group_agreement": float(equal.float().mean()),
        "mismatched_banks": int((~equal).sum()), "anchor_mismatches": int((native[:, 0] != rows).sum()),
        "nonunique_banks": int((native.sort(dim=-1).values.diff(dim=-1) == 0).any(-1).sum()),
        "invalid_candidate_banks": invalid,
        "prefix_mismatched_banks": int((native[:, :3] != incumbent[:, :3]).any(-1).sum()),
        "decision_mismatched_banks": decision_errors, "accepted_banks": accepted,
        "self_accepted_banks": self_accepted,
        "row_change_decision_mismatched_banks": int((decisions[:, 1].bool() != (audited != incumbent).any(-1)).sum()),
        "decision_mismatch_diagnostic_limit": MISMATCH_DIAGNOSTIC_LIMIT,
        "decision_mismatch_diagnostics": decision_diagnostics,
        "singleton_already_probed_banks": banks // 5, "singleton_added_banks": banks * 4 // 5,
        "incumbent_implementation": "fisher_anchor_seed_rows_triton",
        "frame_banks": [7200] * frames,
        "frame_mismatched_banks": [int((~part).sum()) for part in equal.reshape(frames, -1)],
        "score_audit_on_off_equal": bool(audit_equal.all()),
        "score_audit_mismatched_banks": int((~audit_equal).sum()),
        "score_atol": GAP_ATOL, "score_rtol": GAP_RTOL,
        "score_values_checked": banks * 3 * 32, "score_comparison_passed": passed,
        "max_score_abs_error": max_error, "max_score_tolerance_ratio": max_ratio,
        "native_sha256": hashlib.sha256(native.cpu().numpy().tobytes()).hexdigest()}
    return record
