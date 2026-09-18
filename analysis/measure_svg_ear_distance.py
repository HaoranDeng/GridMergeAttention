"""Measure SVG-EAR's attention approximation error on our captured Wan QKV.

PSNR transfers between setups; rel-L2 against dense does not, because it is
measured on specific Q/K/V. To compare SVG-EAR against GridMerge on the same
footing, EAR has to run on the same captured tensors. Triton/torch version
differences do not matter here: only the distance is being measured, not speed.

EAR falls back to dense attention for early layers and early timesteps, so
first_layers_fp and first_times_fp are forced to 0 and the timestep is 0 --
otherwise this would silently measure full attention against itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", required=True)
    p.add_argument("--ear_root", default="/xuanwu-tank/east/zrl/ear-test")
    p.add_argument("--output", default="out/svg_ear_distance.json")
    # Defaults follow scripts/wan/wan_t2v_720p_ear.sh, the configuration
    # SVG-EAR is actually run at, not the argparse defaults. kmeans_iter_init
    # of 0 leaves the cluster assignment unbound.
    p.add_argument("--num_q_centroids", type=int, default=300)
    p.add_argument("--num_k_centroids", type=int, default=1000)
    p.add_argument("--top_p_list", default="0.9",
                   help="comma-separated top_p values to sweep")
    p.add_argument("--min_kc_ratio", type=float, default=0.10)
    p.add_argument("--kmeans_iter_init", type=int, default=50)
    p.add_argument("--kmeans_iter_step", type=int, default=2)
    args = p.parse_args()
    sys.path.insert(0, args.ear_root)

    from svg.models.wan.attention import WanAttn_EARAttn_Processor as EAR

    blob = torch.load(args.capture, weights_only=False)
    frames, rows, cols = blob["grid"]
    keys = sorted(blob["tensors"], key=lambda s: tuple(int(x) for x in s.split("_")))

    EAR.context_length = 0
    EAR.num_frame = frames
    EAR.frame_size = rows * cols
    EAR.num_q_centroids = args.num_q_centroids
    EAR.num_k_centroids = args.num_k_centroids
    EAR.top_p_kmeans = None  # set per sweep point
    EAR.min_kc_ratio = args.min_kc_ratio
    EAR.kmeans_iter_init = args.kmeans_iter_init
    EAR.kmeans_iter_step = args.kmeans_iter_step
    EAR.num_layers = 30
    # EAR logs avg_density per call when a logging file is set. Density is
    # the fraction of key blocks actually attended, and without it a rel-L2
    # comparison against a fixed-ratio method like C4 is meaningless.
    density_log = Path(args.output).with_suffix(".density.jsonl")
    density_log.parent.mkdir(parents=True, exist_ok=True)
    density_log.write_text("")
    EAR.logging_file = str(density_log)
    EAR.first_layers_fp = 0      # 0 so no layer takes the dense fallback
    EAR.first_times_fp = 0       # 0 so no timestep takes the dense fallback
    for attr, val in (("zero_step_kmeans_init", False), ("num_sampled_rows", 32),
                      ("sample_mse_max_row", 10000)):
        if not hasattr(EAR, attr):
            setattr(EAR, attr, val)

    tensors = {}
    for key in keys:
        t = blob["tensors"][key]
        flat = lambda x: x.permute(0, 4, 1, 2, 3, 5).reshape(1, x.size(4), -1, x.size(5))
        tensors[key] = tuple(flat(t[n]) for n in ("q", "k", "v"))

    summary = []
    for top_p in [float(x) for x in args.top_p_list.split(",")]:
        EAR.top_p_kmeans = top_p
        density_log.write_text("")
        rows_out, errs = run_one(EAR, tensors, keys, top_p)
        densities = [json.loads(l)["avg_density"]
                     for l in density_log.read_text().splitlines() if l.strip()]
        if errs:
            summary.append({
                "top_p": top_p,
                "rel_l2": sum(errs) / len(errs),
                "density": (sum(densities) / len(densities)) if densities else None,
                "n": len(errs)})
            r = summary[-1]
            print(f"top_p {top_p:.2f}   density {r['density']:.4f}   "
                  f"rel-L2 {r['rel_l2']:.5f}   ({r['n']} points)", flush=True)

    print(f"\n{'top_p':>6} {'density':>9} {'rel-L2':>9}     versus GridMerge")
    for r in summary:
        print(f"{r['top_p']:6.2f} {r['density']:9.4f} {r['rel_l2']:9.5f}")
    print(f"{'ours':>6} {0.0769:9.4f} {0.07005:9.5f}   weighted span a=0.3")
    print(f"{'C4':>6} {0.0769:9.4f} {0.09742:9.5f}   key-row C4")
    Path(args.output).write_text(json.dumps(
        {"sweep": summary, "qc": args.num_q_centroids, "kc": args.num_k_centroids,
         "gridmerge": {"density": 0.0769, "span_a03": 0.07005, "key_row_c4": 0.09742}},
        indent=2))
    print(f"wrote {args.output}")


def run_one(EAR, tensors, keys, top_p):
    rows_out, errs = [], []
    for key in keys:
        step, layer = (int(x) for x in key.split("_"))
        q, k, v = (x.cuda() for x in tensors[key])
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        try:
            proc = EAR(layer_idx=max(layer, 1))
            got = proc.attention_core_logic(q, k, v, torch.tensor([0.0], device="cuda"))
            rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
            rows_out.append({"step": step, "layer": layer, "rel_l2": rel})
            errs.append(rel)
        except Exception as exc:
            print(f"  step {step:2d} layer {layer:2d}   FAILED {type(exc).__name__}: {exc}")
        del q, k, v, ref
        torch.cuda.empty_cache()

    return rows_out, errs


if __name__ == "__main__":
    main()
