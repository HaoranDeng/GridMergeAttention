"""Summarize fixed-head rescue candidates from Hard-C4 real-QKV probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration_json", required=True)
    parser.add_argument("--validation_json", required=True)
    parser.add_argument("--group_width", type=int, default=4)
    parser.add_argument(
        "--layer_widths",
        help="Optional comma-separated dense-head counts, one per layer.",
    )
    parser.add_argument("--output_json")
    return parser.parse_args()


def layer_head_errors(payload):
    heads = [int(head) for head in payload["heads"]]
    calls_total = int(payload["calls_total"])
    layers = 40
    if calls_total % layers:
        raise ValueError(f"calls_total={calls_total} is not divisible by {layers}")
    values = [[[] for _ in heads] for _ in range(layers)]
    per_call = payload["per_call_head_errors_vs_dense"]
    for call_text, head_values in per_call.items():
        layer = (int(call_text) - 1) % layers
        for position, head in enumerate(heads):
            values[layer][position].append(
                float(head_values[str(head)]["fixed_wide"]["mse_vs_dense"])
            )
    return heads, [
        [sum(samples) / len(samples) for samples in layer]
        for layer in values
    ]


def top_indices(values, count):
    return sorted(range(len(values)), key=values.__getitem__, reverse=True)[:count]


def best_contiguous(values, width):
    return max(
        range(len(values) - width + 1),
        key=lambda start: sum(values[start : start + width]),
    )


def selected_share(matrix, selections):
    selected = sum(
        sum(row[index] for index in indices)
        for row, indices in zip(matrix, selections)
    )
    total = sum(sum(row) for row in matrix)
    return selected / total


def main():
    args = parse_args()
    calibration = json.loads(Path(args.calibration_json).read_text())
    validation = json.loads(Path(args.validation_json).read_text())
    calibration_heads, calibration_matrix = layer_head_errors(calibration)
    validation_heads, validation_matrix = layer_head_errors(validation)
    if calibration_heads != validation_heads:
        raise ValueError("calibration and validation head lists differ")

    widths = (
        [int(value) for value in args.layer_widths.split(",")]
        if args.layer_widths
        else [int(args.group_width)] * len(calibration_matrix)
    )
    if len(widths) != len(calibration_matrix):
        raise ValueError("layer_widths must contain one value per layer")
    if any(width < 0 or width > len(calibration_heads) for width in widths):
        raise ValueError("layer_widths values must be valid head counts")
    arbitrary = [
        top_indices(row, width) for row, width in zip(calibration_matrix, widths)
    ]
    contiguous_starts = [
        best_contiguous(row, width) if width else None
        for row, width in zip(calibration_matrix, widths)
    ]
    contiguous = [
        list(range(start, start + width)) if width else []
        for start, width in zip(contiguous_starts, widths)
    ]
    validation_arbitrary = [
        top_indices(row, width) for row, width in zip(validation_matrix, widths)
    ]
    validation_contiguous_starts = [
        best_contiguous(row, width) if width else None
        for row, width in zip(validation_matrix, widths)
    ]
    validation_contiguous = [
        list(range(start, start + width)) if width else []
        for start, width in zip(validation_contiguous_starts, widths)
    ]
    calibration_global = [
        sum(row[head] for row in calibration_matrix)
        for head in range(len(calibration_heads))
    ]
    uniform_width = sum(widths) // len(widths)
    global_top = top_indices(calibration_global, uniform_width)
    dense_fraction = sum(widths) / (len(widths) * len(calibration_heads))
    theoretical_dense_ratio = (1.0 - dense_fraction) * (11.0 / 60.0) + dense_fraction

    result = {
        "group_width": int(args.group_width) if not args.layer_widths else None,
        "layer_widths": widths,
        "theoretical_dense_ratio": theoretical_dense_ratio,
        "ratio_vs_svg_ear": theoretical_dense_ratio / 0.37906453310046345,
        "calibration": {
            "global_top_group_share": selected_share(
                calibration_matrix, [global_top] * len(calibration_matrix)
            ),
            "layerwise_arbitrary_share": selected_share(
                calibration_matrix, arbitrary
            ),
            "layerwise_contiguous_share": selected_share(
                calibration_matrix, contiguous
            ),
        },
        "validation": {
            "calibrated_layerwise_arbitrary_share": selected_share(
                validation_matrix, arbitrary
            ),
            "calibrated_layerwise_contiguous_share": selected_share(
                validation_matrix, contiguous
            ),
            "oracle_layerwise_arbitrary_share": selected_share(
                validation_matrix, validation_arbitrary
            ),
            "oracle_layerwise_contiguous_share": selected_share(
                validation_matrix, validation_contiguous
            ),
            "mean_arbitrary_head_overlap": sum(
                len(set(left) & set(right)) / width if width else 1.0
                for left, right, width in zip(
                    arbitrary, validation_arbitrary, widths
                )
            )
            / len(arbitrary),
            "contiguous_start_match_fraction": sum(
                left == right
                for left, right in zip(
                    contiguous_starts, validation_contiguous_starts
                )
            )
            / len(contiguous_starts),
        },
        "calibrated_arbitrary_heads_by_layer": [
            [calibration_heads[index] for index in indices] for indices in arbitrary
        ],
        "calibrated_contiguous_start_by_layer": [
            calibration_heads[start] if start is not None else None
            for start in contiguous_starts
        ],
        "validation_oracle_contiguous_start_by_layer": [
            validation_heads[start] if start is not None else None
            for start in validation_contiguous_starts
        ],
    }
    rendered = json.dumps(result, indent=2)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
