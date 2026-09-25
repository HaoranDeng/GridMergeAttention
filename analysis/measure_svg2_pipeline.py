"""SVG2 (SAP) rel-L2 and density, measured inside the live pipeline.

Driving the EAR processor standalone on captured tensors reports ~0.98 density
against ~0.36 in the pipeline, because the processors persist across calls there
and initialise their centroids once. Rather than reproduce that state outside
the pipeline, this measures in it: subclass the processor, compute dense
attention alongside its output, and record both error and density per call.

That is the same approach benchmark_wan14b_720p_methods.py takes with its
TimedSparseProcessor wrapper.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    p.add_argument("--ear_root", default="/xuanwu-tank/east/zrl/ear-test")
    p.add_argument("--output", default="out/svg_ear_pipeline.json")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--measure_steps", type=int, default=2)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--num_q_centroids", type=int, default=300)
    p.add_argument("--num_k_centroids", type=int, default=1000)
    p.add_argument("--top_p_kmeans", type=float, default=0.9)
    p.add_argument("--min_kc_ratio", type=float, default=0.10)
    args = p.parse_args()
    sys.path.insert(0, args.ear_root)

    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from svg.models.wan.inference import replace_wan_attention
    from svg.models.wan.attention import WanAttn_SAPAttn_Processor

    log = Path(args.output).with_suffix(".density.jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("")
    errors = []

    class MeasuredEAR(WanAttn_SAPAttn_Processor):
        """EAR, plus a dense reference computed on the same inputs."""

        def attention_core_logic(self, query, key, value, timestep):
            out = super().attention_core_logic(query, key, value, timestep)
            dense = F.scaled_dot_product_attention(query, key, value)
            rel = ((out.float() - dense.float()).norm() / dense.float().norm()).item()
            errors.append({"layer": self.layer_idx, "rel_l2": rel})
            return out

    vae = AutoencoderKLWan.from_pretrained(
        args.model, subfolder="vae", torch_dtype=torch.float32, local_files_only=True)
    pipe = WanPipeline.from_pretrained(
        args.model, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction", use_flow_sigmas=True,
        num_train_timesteps=1000, flow_shift=5.0)
    pipe.to("cuda")

    # first_times_fp=1001 and first_layers_fp=0 is how the reference configures
    # the all-layer variant: every call takes the sparse path.
    replace_wan_attention(
        pipe, height=args.height, width=args.width, num_frames=args.num_frames,
        first_layers_fp=0, first_times_fp=1001, pattern="SAP",
        num_q_centroids=args.num_q_centroids, num_k_centroids=args.num_k_centroids,
        top_p_kmeans=args.top_p_kmeans, min_kc_ratio=args.min_kc_ratio,
        kmeans_iter_init=50, kmeans_iter_step=2, logging_file=str(log))
    for layer, block in enumerate(pipe.transformer.blocks):
        block.attn1.set_processor(MeasuredEAR(layer_idx=layer))

    prompt = Path(args.prompt_path).read_text().splitlines()[0].strip()

    class Stop(Exception):
        pass

    def stop_after(_p, i, _t, kw):
        if i + 1 >= args.measure_steps:
            raise Stop
        return kw

    try:
        pipe(prompt=prompt, height=args.height, width=args.width,
             num_frames=args.num_frames, guidance_scale=5.0,
             num_inference_steps=args.steps, callback_on_step_end=stop_after)
    except Stop:
        pass

    dens = [json.loads(l)["avg_density"] for l in log.read_text().splitlines() if l.strip()]
    errs = [e["rel_l2"] for e in errors]
    mean_d = sum(dens) / len(dens) if dens else float("nan")
    mean_e = sum(errs) / len(errs) if errs else float("nan")
    print(f"\nSVG2 (SAP) in-pipeline, {args.measure_steps} step(s), {len(errs)} calls")
    print(f"  density  {mean_d:.4f}")
    print(f"  rel-L2   {mean_e:.5f}")
    print(f"\n  GridMerge on the same model, density 0.0769:")
    print(f"    key-row C4   0.09742")
    print(f"    span a=0.3   0.06644")
    Path(args.output).write_text(json.dumps(
        {"mean_density": mean_d, "mean_rel_l2": mean_e,
         "per_call": errors, "densities": dens,
         "config": {k: v for k, v in vars(args).items() if k not in ("ear_root", "model")}},
        indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
