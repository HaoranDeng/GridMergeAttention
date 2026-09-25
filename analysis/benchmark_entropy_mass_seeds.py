"""Production-path parity and interleaved latency for entropy-mass selectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from attention.entropy_mass_seed_triton import entropy_mass_seed_rows_triton
from attention.partition_mass_seed_triton import partition_mass_seed_rows_triton
from evaluate_entropy_seed_oracle import _load_grid, _parse_indices
from screen_partition_population_seeds import custom_tables, metrics, production, BASE_MODE


MODES = {
    "rf10": BASE_MODE,
    "mass_span": "entropy_mass_span_c32_p16_g256_j4_t1_u16_cg45_sq1_qm_fp8score_skv_nl_ml2",
    "partition_mass": "entropy_partition_mass_c32_p16_g256_j4_t1_u16_cg45_sq1_qm_fp8score_skv_nl_ml2",
}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", nargs="+", required=True)
    parser.add_argument("--heads", default=",".join(map(str, range(40))))
    parser.add_argument("--modes", default="rf10,mass_span,partition_mass")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = {}
    for capture in args.qkv:
        heads = _parse_indices(args.heads)
        qg, kg, vg = [t.to(device="cuda", dtype=torch.bfloat16) for t in _load_grid(capture, heads)]
        q, k, v = [t.permute(1, 2, 3, 0, 4).reshape(1, 21 * 45 * 80, len(heads), 128).contiguous() for t in (qg, kg, vg)]
        record = {}
        if args.parity:
            tables = custom_tables(qg, kg, ["gs_mass_256", "deficit_mass_256"], chunk=7)
            router = q.reshape(1, 21, 45, 80, len(heads), 128)[:, 10:11].contiguous()
            key = k.reshape(1, 21, 45, 80, len(heads), 128)
            for name, selector, reference_name in (
                ("mass_span", entropy_mass_seed_rows_triton, "gs_mass_256"),
                ("partition_mass", partition_mass_seed_rows_triton, "deficit_mass_256"),
            ):
                fused = selector(router, key, gs_num_warps=8)
                ref = tables[reference_name]
                overlap = (fused[..., :, None] == ref[..., None, :]).any(-1).float().sum(-1).mean()
                fused_output = production(q, k, v, mode=MODES[name])
                reference_output = production(q, k, v, table=ref)
                if not bool(torch.isfinite(fused_output).all()):
                    raise AssertionError(f"nonfinite {name}")
                record[name + "_parity"] = {
                    "slot_agreement": float((fused == ref).float().mean()),
                    "mean_shared_seeds": float(overlap),
                    "output_relative_l2": metrics(fused_output, reference_output)["relative_l2"],
                }
                print(name, record[name + "_parity"], flush=True)
                if float(overlap) < 3.98:
                    raise AssertionError(f"seed parity failed for {name}: {overlap}")
            del tables, router, key, ref, fused, fused_output, reference_output
        names = args.modes.split(",")
        for name in names:
            production(q, k, v, mode=MODES[name])
        torch.cuda.synchronize()
        samples = {name: [] for name in names}
        for repeat in range(args.repeats):
            for name in names if repeat % 2 == 0 else names[::-1]:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iterations):
                    output = production(q, k, v, mode=MODES[name])
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) / args.iterations)
        record["latency"] = {name: {"mean_ms": float(np.mean(values)), "std_ms": float(np.std(values)),
                                    "samples_ms": values} for name, values in samples.items()}
        print(capture, json.dumps(record), flush=True)
        records[capture] = record
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"settings": vars(args), "device": torch.cuda.get_device_name(),
                                          "records": records}, default=str, indent=2) + "\n")
        del qg, kg, vg, q, k, v, output
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
