import argparse
import json
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised only when Triton is missing.
    triton = None
    tl = None

from attention.hard_fused_triton import (
    HARD_BLOCK_M,
    HARD_BLOCK_N,
    HARD_INPUT_PRECISION,
    HARD_NUM_STAGES,
    HARD_NUM_WARPS,
    hard_route_attention_triton_fused,
)


STAGES = [
    ("load_qkv", 0),
    ("seed_logits", 1),
    ("hard_assign", 2),
    ("merge_kv", 3),
    ("second_scores", 4),
    ("online_accum_checksum", 5),
]


if triton is not None:

    @triton.jit
    def _hard_c4_fused_profile_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        checksum_ptr,
        Q_OUTER: tl.constexpr,
        K_OUTER: tl.constexpr,
        Q_ROWS: tl.constexpr,
        K_ROWS: tl.constexpr,
        Q_COLS: tl.constexpr,
        HEADS: tl.constexpr,
        K_COLS: tl.constexpr,
        D_HEAD: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SCALE: tl.constexpr,
        LOGIT_INPUT_PRECISION: tl.constexpr,
        STAGE: tl.constexpr,
    ):
        program = tl.program_id(0)

        query_col = program % Q_COLS
        tmp = program // Q_COLS
        query_outer = tmp % Q_OUTER
        tmp = tmp // Q_OUTER
        head = tmp % HEADS
        batch = tmp // HEADS

        q_rows = tl.arange(0, BLOCK_Q)
        k_cols = tl.arange(0, BLOCK_K)
        dims = tl.arange(0, BLOCK_D)
        q_valid = q_rows < Q_ROWS
        k_valid = k_cols < K_COLS
        d_valid = dims < D_HEAD

        q_base = (batch * Q_OUTER + query_outer) * Q_ROWS * Q_COLS * HEADS * D_HEAD
        q = tl.load(
            q_ptr
            + q_base
            + (((q_rows[:, None] * Q_COLS + query_col) * HEADS + head) * D_HEAD + dims[None, :]),
            mask=q_valid[:, None] & d_valid[None, :],
            other=0.0,
        )
        q_for_dot = q.to(tl.float16)
        checksum = tl.sum(q.to(tl.float32))

        row_max = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
        row_sum = tl.zeros((BLOCK_Q,), tl.float32)
        row_acc = tl.zeros((BLOCK_Q, BLOCK_D), tl.float32)

        for block in range(0, K_OUTER * K_ROWS):
            key_outer = block // K_ROWS
            key_row = block - key_outer * K_ROWS

            seed_ids = tl.arange(0, 4)
            seed_row_1 = tl.where(key_row > 0, key_row - 1, 1)
            seed_row_2 = tl.where(
                key_row == 0,
                2,
                tl.where(key_row == Q_ROWS - 1, Q_ROWS - 3, key_row + 1),
            )
            seed_row_3 = tl.where(
                key_row <= 1,
                3,
                tl.where(key_row >= Q_ROWS - 2, Q_ROWS - 4, key_row - 2),
            )
            seed_rows = tl.where(
                seed_ids == 0,
                key_row,
                tl.where(seed_ids == 1, seed_row_1, tl.where(seed_ids == 2, seed_row_2, seed_row_3)),
            )

            key_base = k_ptr + (((batch * K_OUTER + key_outer) * K_ROWS + key_row) * K_COLS * HEADS + head) * D_HEAD
            key_block = tl.load(
                key_base + k_cols[:, None] * HEADS * D_HEAD + dims[None, :],
                mask=k_valid[:, None] & d_valid[None, :],
                other=0.0,
            )
            value_base = v_ptr + (((batch * K_OUTER + key_outer) * K_ROWS + key_row) * K_COLS * HEADS + head) * D_HEAD
            value_block = tl.load(
                value_base + k_cols[:, None] * HEADS * D_HEAD + dims[None, :],
                mask=k_valid[:, None] & d_valid[None, :],
                other=0.0,
            )
            checksum += tl.sum(key_block.to(tl.float32)) + tl.sum(value_block.to(tl.float32))

            if STAGE >= 1:
                seed_queries = tl.load(
                    q_ptr
                    + q_base
                    + (((seed_rows[:, None] * Q_COLS + query_col) * HEADS + head) * D_HEAD + dims[None, :]),
                    mask=d_valid[None, :],
                    other=0.0,
                )
                seed_logits = tl.dot(
                    seed_queries,
                    tl.trans(key_block),
                    input_precision=LOGIT_INPUT_PRECISION,
                ) * SCALE
                seed_logits = tl.where(k_valid[None, :], seed_logits, -float("inf"))
                checksum += tl.sum(tl.where(seed_logits > -float("inf"), seed_logits, 0.0))

                if STAGE >= 2:
                    logits_0 = tl.sum(tl.where(seed_ids[:, None] == 0, seed_logits, 0.0), axis=0)
                    logits_1 = tl.sum(tl.where(seed_ids[:, None] == 1, seed_logits, 0.0), axis=0)
                    logits_2 = tl.sum(tl.where(seed_ids[:, None] == 2, seed_logits, 0.0), axis=0)
                    logits_3 = tl.sum(tl.where(seed_ids[:, None] == 3, seed_logits, 0.0), axis=0)

                    best_logits = logits_0
                    assigned_seed = tl.zeros((BLOCK_K,), tl.int32)
                    take = logits_1 > best_logits
                    best_logits = tl.where(take, logits_1, best_logits)
                    assigned_seed = tl.where(take, 1, assigned_seed)
                    take = logits_2 > best_logits
                    best_logits = tl.where(take, logits_2, best_logits)
                    assigned_seed = tl.where(take, 2, assigned_seed)
                    take = logits_3 > best_logits
                    best_logits = tl.where(take, logits_3, best_logits)
                    assigned_seed = tl.where(take, 3, assigned_seed)
                    checksum += tl.sum(best_logits) + tl.sum(assigned_seed.to(tl.float32))

                    if STAGE >= 3:
                        routed = k_valid[None, :] & (assigned_seed[None, :] == seed_ids[:, None])
                        cluster_max = tl.max(tl.where(routed, best_logits[None, :], -float("inf")), axis=1)
                        cluster_has_key = cluster_max > -float("inf")
                        safe_cluster_max = tl.where(cluster_has_key, cluster_max, 0.0)
                        cluster_prob = tl.where(
                            routed,
                            tl.exp2((best_logits[None, :] - safe_cluster_max[:, None]) * 1.4426950408889634),
                            0.0,
                        )
                        cluster_sum = tl.sum(cluster_prob, axis=1)
                        safe_cluster_sum = tl.where(cluster_has_key, cluster_sum, 1.0)
                        merge_weight = cluster_prob / safe_cluster_sum[:, None]
                        log_cluster_sum = safe_cluster_max + tl.log2(safe_cluster_sum) * 0.6931471805599453

                        merged_key = tl.dot(merge_weight.to(key_block.dtype), key_block).to(tl.float16)
                        merged_value = tl.dot(merge_weight.to(value_block.dtype), value_block).to(tl.float16)
                        expected_logit = tl.sum(tl.where(routed, merge_weight * best_logits[None, :], 0.0), axis=1)
                        routing_bias = tl.where(cluster_has_key, expected_logit - log_cluster_sum, float("inf"))
                        checksum += (
                            tl.sum(merged_key.to(tl.float32))
                            + tl.sum(merged_value.to(tl.float32))
                            + tl.sum(tl.where(cluster_has_key, routing_bias, 0.0))
                        )

                        if STAGE >= 4:
                            scores = tl.dot(q_for_dot, tl.trans(merged_key)) * SCALE - routing_bias[None, :]
                            scores = tl.where(q_valid[:, None] & cluster_has_key[None, :], scores, -float("inf"))
                            checksum += tl.sum(tl.where(scores > -float("inf"), scores, 0.0))

                            if STAGE >= 5:
                                block_max = tl.max(scores, axis=1)
                                next_row_max = tl.maximum(row_max, block_max)
                                prob = tl.exp2((scores - next_row_max[:, None]) * 1.4426950408889634)
                                old_scale = tl.exp2((row_max - next_row_max) * 1.4426950408889634)
                                row_acc = row_acc * old_scale[:, None] + tl.dot(prob.to(merged_value.dtype), merged_value)
                                row_sum = row_sum * old_scale + tl.sum(prob, axis=1)
                                row_max = next_row_max

        if STAGE >= 5:
            checksum += tl.sum(row_acc / row_sum[:, None])
        tl.store(checksum_ptr + program, checksum)


def run_profile_stage(q, k, v, stage):
    batch, query_outer, query_rows, query_columns, heads, dim = q.shape
    _, key_outer, key_rows, key_columns, _, _ = k.shape
    checksum = torch.empty(batch * heads * query_outer * query_columns, device=q.device, dtype=torch.float32)
    block_n = max(HARD_BLOCK_N, triton.next_power_of_2(key_columns))
    grid = (checksum.numel(),)
    _hard_c4_fused_profile_kernel[grid](
        q,
        k,
        v,
        checksum,
        Q_OUTER=query_outer,
        K_OUTER=key_outer,
        Q_ROWS=query_rows,
        K_ROWS=key_rows,
        Q_COLS=query_columns,
        HEADS=heads,
        K_COLS=key_columns,
        D_HEAD=dim,
        BLOCK_Q=HARD_BLOCK_M,
        BLOCK_K=block_n,
        BLOCK_D=triton.next_power_of_2(dim),
        SCALE=float(dim ** -0.5),
        LOGIT_INPUT_PRECISION=HARD_INPUT_PRECISION,
        STAGE=int(stage),
        num_warps=HARD_NUM_WARPS,
        num_stages=HARD_NUM_STAGES,
    )
    return checksum


def time_call(fn, warmup, repeats):
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        out = fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / max(repeats, 1)), float(out.float().sum().item())


def parse_args():
    parser = argparse.ArgumentParser(description="Profile cumulative stages of the experimental fused hard C4 Triton kernel.")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--query_outer", type=int, default=1)
    parser.add_argument("--key_outer", type=int, default=21)
    parser.add_argument("--rows", type=int, default=30)
    parser.add_argument("--columns", type=int, default=52)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm GPU is required.")
    if triton is None:
        raise RuntimeError("Triton is required.")

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    q_shape = (args.batch, args.query_outer, args.rows, args.columns, args.heads, args.dim)
    kv_shape = (args.batch, args.key_outer, args.rows, args.columns, args.heads, args.dim)
    q = torch.randn(q_shape, device="cuda", dtype=dtype)
    k = torch.randn(kv_shape, device="cuda", dtype=dtype)
    v = torch.randn(kv_shape, device="cuda", dtype=dtype)

    cumulative = {}
    for name, stage in STAGES:
        mean_ms, checksum = time_call(lambda stage=stage: run_profile_stage(q, k, v, stage), args.warmup, args.repeats)
        cumulative[name] = {"stage": stage, "mean_ms": mean_ms, "checksum": checksum}

    full_ms, full_checksum = time_call(
        lambda: hard_route_attention_triton_fused(q, k, v, args.dim ** -0.5),
        args.warmup,
        args.repeats,
    )

    previous_name = None
    estimated_parts = {}
    for name, _stage in STAGES:
        if previous_name is None:
            estimated_parts[name] = cumulative[name]["mean_ms"]
        else:
            estimated_parts[name] = cumulative[name]["mean_ms"] - cumulative[previous_name]["mean_ms"]
        previous_name = name
    estimated_parts["full_kernel_minus_second_scores"] = full_ms - cumulative["second_scores"]["mean_ms"]
    estimated_parts["full_kernel_minus_online_checksum"] = full_ms - cumulative["online_accum_checksum"]["mean_ms"]

    payload = {
        "device": torch.cuda.get_device_name(0),
        "dtype": args.dtype,
        "shape_q": list(q_shape),
        "shape_kv": list(kv_shape),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "note": "Stage timings are cumulative profiling kernels with checksum stores. Per-part timings are adjacent differences and are approximate.",
        "cumulative_ms": cumulative,
        "full_fused": {"mean_ms": full_ms, "checksum": full_checksum},
        "estimated_part_ms": estimated_parts,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
