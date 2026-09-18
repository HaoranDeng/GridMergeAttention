import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from attention.hard_c4_wan21_14b_720p_triton import (
    hard_c4_720p_attention_triton_blocks,
    hard_c4_720p_exact_query_top3_attention_triton_blocks,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch, frames, rows, columns, dim = 1, 21, 45, 80, 128
    shape = (batch, frames, rows, columns, args.heads, dim)
    query = torch.randn(shape, device=device, dtype=dtype)
    key = torch.randn(shape, device=device, dtype=dtype)
    value = torch.randn(shape, device=device, dtype=dtype)
    scale = dim**-0.5

    baseline = hard_c4_720p_attention_triton_blocks(
        query, key, value, scale=scale
    )
    refined = hard_c4_720p_exact_query_top3_attention_triton_blocks(
        query, key, value, scale=scale
    )
    torch.cuda.synchronize()

    row_mse = (refined.float() - baseline.float()).square().mean(dim=-1)
    row_mse = row_mse.permute(0, 1, 3, 4, 2)
    selected_rows = row_mse.topk(3, dim=-1).indices
    selected_mask = torch.zeros_like(row_mse, dtype=torch.bool)
    selected_mask.scatter_(dim=-1, index=selected_rows, value=True)

    query_by_row = query.permute(0, 1, 3, 4, 2, 5)
    gather_index = selected_rows.unsqueeze(-1).expand(
        *selected_rows.shape, dim
    )
    selected_query = query_by_row.gather(4, gather_index)
    selected_query = selected_query.permute(0, 3, 1, 2, 4, 5).reshape(
        batch, args.heads, -1, dim
    )
    dense_key = key.permute(0, 4, 1, 2, 3, 5).reshape(
        batch, args.heads, -1, dim
    )
    dense_value = value.permute(0, 4, 1, 2, 3, 5).reshape(
        batch, args.heads, -1, dim
    )
    dense_selected = F.scaled_dot_product_attention(
        selected_query,
        dense_key,
        dense_value,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    ).reshape(
        batch, args.heads, frames, columns, 3, dim
    ).permute(0, 2, 3, 1, 4, 5)
    refined_selected = refined.permute(0, 1, 3, 4, 2, 5).gather(
        4, gather_index
    )

    exact_delta = refined_selected.float() - dense_selected.float()
    exact_max_abs = float(exact_delta.abs().max().item())
    exact_rel_l2 = float(
        exact_delta.norm().item()
        / dense_selected.float().norm().clamp_min(1e-20).item()
    )
    full_delta = (refined.float() - baseline.float()).abs().permute(
        0, 1, 3, 4, 2, 5
    )
    unselected = ~selected_mask.unsqueeze(-1).expand_as(full_delta)
    unselected_max_abs = float(full_delta.masked_select(unselected).max().item())
    selected_min_row_mse = float(
        row_mse.gather(-1, selected_rows).min().item()
    )

    payload = {
        "device": torch.cuda.get_device_name(0),
        "shape": list(shape),
        "selected_rows_per_group": 3,
        "groups": int(batch * frames * columns * args.heads),
        "selected_min_row_mse_vs_c4": selected_min_row_mse,
        "unselected_max_abs_vs_c4": unselected_max_abs,
        "selected_max_abs_vs_dense": exact_max_abs,
        "selected_rel_l2_vs_dense": exact_rel_l2,
        "finite": bool(torch.isfinite(refined).all()),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
