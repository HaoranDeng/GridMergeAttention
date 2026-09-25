"""Stream frozen seed operators over every call of a canonical dense trajectory.

Candidates are off-path diagnostics: only dense output advances the model.
No QKV archive, training, fitted parameters, or candidate video is produced.
Sampled-head timing is diagnostic, not the full-40-head performance gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODES = {
    "rf0": "hellinger_span_alpha0p2_c32_p16_a2_j4_t1_rf0_u16_cg45",
    "mass_c32": "entropy_mass_span_c32_p16_g256_j4_t1_rf0_u16_cg45",
    "mass_c16": "entropy_mass_span_c16_p16_g256_j4_t1_rf0_u16_cg16",
    "partition_c16": "entropy_partition_mass_c16_p16_g256_j4_t1_rf0_u16_cg16",
}


def check_capture_environment(environment):
    for key in ("TOKEN_MERGING_SAVE_FIRST_QKV_PATH", "TOKEN_MERGING_EXIT_AFTER_QKV_SAVE"):
        if environment.get(key):
            raise ValueError(f"Unset {key}: this diagnostic must not save full QKV or exit early")


def summarize(records, methods):
    """Weight by scalar output count, not by per-call relative errors."""
    result = {}
    for name in methods:
        values = [record["methods"][name] for record in records]
        count = sum(item["numel"] for item in values)
        if not count:
            continue
        error = sum(item["squared_error_sum"] for item in values)
        energy = sum(item["reference_squared_sum"] for item in values)
        result[name] = {
            "calls": len(values), "numel": count, "mse": error / count,
            "relative_l2": (error / energy) ** 0.5 if energy else None,
            "diagnostic_sampled_head_mean_attention_ms": sum(item["attention_ms"] for item in values) / len(values),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dense_wan_14b_720p_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--heads", default="0,10,20,30")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_capture_environment(os.environ)
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing diagnostic")
    if not 1 <= args.num_prompts <= 20:
        raise ValueError("Use a canonical prefix of 1..20 prompts")
    heads = [int(item) for item in args.heads.split(",")]
    if not heads or len(set(heads)) != len(heads) or any(not 0 <= h < 40 for h in heads):
        raise ValueError("heads must be distinct indices in 0..39")

    from utils.prompt_contract import build_prompt_manifest, load_nonempty_prompts, validate_canonical_prompt_contract
    prompts = load_nonempty_prompts(args.prompt_path)
    validate_canonical_prompt_contract(args.prompt_path, prompts)
    manifest = build_prompt_manifest(args.prompt_path, prompts, selected_count=args.num_prompts,
                                     seed=0, num_samples=1, vbench_filenames=False, allow_noncanonical=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_name("prompt_preflight_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    args.output.with_name("prompt_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    import torch
    from omegaconf import OmegaConf
    from generate import build_pipeline, load_generator_checkpoint, set_seed
    import wan.modules.model as wan_model
    import wan.rmsnorm_ops as rmsnorm_ops
    import attention.core as attention_core
    import attention.token_merging as token_module
    from attention.partition_mass_seed_triton import partition_mass_seed_rows_triton

    config = OmegaConf.load(args.config)
    if config.get("causal", False) or config.get("token_merging_args", {}).get("enable", False):
        raise ValueError("The external trajectory must use the bidirectional dense reference")
    if config.get("token_merging_args", {}).get("attention_backend", "sdpa") != "sdpa":
        raise ValueError("The external reference must use dense SDPA")
    if list(config.image_or_video_shape) != [1, 21, 16, 90, 160] or list(config.denoising_step_list) != [1000, 750, 500, 250]:
        raise ValueError("Expected canonical 720p four-step distilled configuration")
    if not config.get("warp_denoising_step", False) or config.model_kwargs.get("timestep_shift") != 5.0 or config.model_kwargs.get("model_name") != "Wan2.1-T2V-14B-StepDistill-CfgDistill":
        raise ValueError("Expected the audited model, warped schedule, and timestep shift 5")
    torch.cuda.init()  # Advertise the active job before CPU-side model loading.
    pipeline = build_pipeline(config, torch.device("cuda"))
    loaded_weights = load_generator_checkpoint(pipeline, args.checkpoint, True)
    if loaded_weights != "raw_state_dict":
        raise ValueError("This comparison requires the audited raw checkpoint")
    pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to("cuda")
    pipeline.generator.to("cuda")
    pipeline.vae.to("cuda")
    pipeline.eval()
    if len(pipeline.generator.model.blocks) != 40 or len(pipeline.denoising_step_list) != 4:
        raise ValueError("Expected all 40 layers and all four warped steps")
    modes = dict(MODES)
    checkpoint_stat = Path(args.checkpoint).stat()
    payload = {
        "settings": vars(args) | {"output": str(args.output)}, "prompt_manifest": manifest,
        "checkpoint_weights": loaded_weights, "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "device": torch.cuda.get_device_name(), "heads": heads, "methods": modes,
        "actual_warped_timesteps": pipeline.denoising_step_list.tolist(),
        "checkpoint_identity": {"path": str(Path(args.checkpoint).resolve()),
                                "bytes": checkpoint_stat.st_size, "mtime_ns": checkpoint_stat.st_mtime_ns,
                                "note": "File metadata and loaded branch, not a new full-file checksum verification"},
        "environment": {key: os.environ.get(key) for key in
                        ("WAN_MODEL_ROOT", "WAN_SDPA_BACKEND", "TOKEN_MERGING_FORCE_TORCH_RMSNORM")},
        "resolved_model_root": str(Path(os.environ.get("WAN_MODEL_ROOT", "wan_models")).resolve()),
        "actual_use_torch_rmsnorm": rmsnorm_ops._USE_TORCH,
        "actual_forced_sdpa_backend": attention_core._FORCED_SDPA_BACKEND,
        "rmsnorm_implementation": f"{wan_model.rmsnorm.__module__}.{wan_model.rmsnorm.__name__}",
        "rope_implementation": f"{wan_model.apply_rope_video_tokens.__module__}.{wan_model.apply_rope_video_tokens.__name__}",
        "partition_launch_override": {"gs_num_warps": 4, "reuse_owner_logp": True},
        "router_frame": 0,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                          ("analysis/screen_native_seed_trajectory.py", "generate.py", "pipeline/token_merging_wan.py",
                           "wan/modules/model.py", "wan/rmsnorm_ops.py", "attention/core.py", "attention/token_merging.py",
                           "attention/adaptive_seed_triton.py", "attention/entropy_mass_seed_triton.py",
                           "attention/partition_mass_seed_triton.py")},
        "scope": "All 160 self-attention calls per prompt, all 21 query frames, selected heads. Dense trajectory only; candidates never advance the model.",
        "limitations": ["Sampled-head QKV diagnostic, not video PSNR or full-head timing.",
                        "First calls include compilation; do not promote diagnostic latency as a speed gate.",
                        "No cross-layer/step/head candidate exceptions are introduced."],
        "records": [], "complete": False,
    }
    state = {"layer": None, "prompt": None, "call": 0}
    head_index = torch.tensor(heads, device="cuda")
    original_attention = wan_model.flash_attention
    original_forward = wan_model.WanSelfAttention.forward

    def partition_override(*pargs, **kwargs):
        return partition_mass_seed_rows_triton(*pargs, **(kwargs | {"gs_num_warps": 4, "reuse_owner_logp": True}))

    def forward(module, *fargs, **kwargs):
        if state["layer"] is not None:
            raise RuntimeError("Unexpected nested self attention")
        if module.enable_token_merging or module.attention_backend != "sdpa" or tuple(module.window_size) != (-1, -1):
            raise RuntimeError("Every reference layer must use full dense SDPA")
        state["layer"] = int(module.layer_idx)
        try:
            return original_forward(module, *fargs, **kwargs)
        finally:
            state["layer"] = None

    def attention(q, k, v, *fargs, **kwargs):
        dense = original_attention(q, k, v, *fargs, **kwargs)
        if state["layer"] is None:
            return dense
        call = state["call"]
        if call >= 160 or state["layer"] != call % 40 or tuple(q.shape) != (1, 75600, 40, 128) or k.shape != q.shape or v.shape != q.shape:
            raise RuntimeError("Unexpected trajectory call ordering or QKV shape")
        selected = [x.index_select(2, head_index).contiguous() for x in (q, k, v)]
        reference = dense.index_select(2, head_index).float()
        row = {"prompt_index": state["prompt"], "call_index": call,
               "step_index": call // 40, "layer_index": state["layer"], "methods": {}}
        order = list(modes) if call % 2 == 0 else list(reversed(modes))
        for name in order:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            output = token_module._hard_c4_attention(
                *selected, h=45, w=80, h_reduce=1, w_reduce=1, sm_scale=128 ** -0.5,
                query_outer_chunk=21, head_chunk=8, hard_route_backend="hard_c4", seed_mode=modes[name],
                exact_intra_frame_attention=False, exact_neighbor_window_columns=0, exact_query_rows=0,
                first_frame_full_attention=False, full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0, full_attention_anchor_scale=1.0).contiguous()
            end.record()
            error = output.float() - reference
            if not bool(torch.isfinite(error).all()):
                raise RuntimeError(f"Nonfinite candidate output: {name}")
            row["methods"][name] = {
                "numel": error.numel(), "squared_error_sum": float(error.square().sum()),
                "reference_squared_sum": float(reference.square().sum()),
                "frame_mse": error.square().reshape(1, 21, 3600, len(heads), 128).mean((0, 2, 3, 4)).tolist(),
                "head_mse": error.square().mean((0, 1, 3)).tolist(),
                "attention_ms": start.elapsed_time(end),
            }
            del output, error
        payload["records"].append(row)
        state["call"] += 1
        if state["call"] % 40 == 0:
            payload["summary"] = summarize(payload["records"], modes)
            args.output.write_text(json.dumps(payload, indent=2) + "\n")
            print(json.dumps({"prompt": state["prompt"], "completed_calls": state["call"], "summary": payload["summary"]}), flush=True)
        return dense

    started = time.perf_counter()
    with torch.inference_mode(), patch.object(wan_model.WanSelfAttention, "forward", forward), \
            patch.object(wan_model, "flash_attention", attention), \
            patch.object(token_module, "partition_mass_seed_rows_triton", partition_override):
        for index in range(args.num_prompts):
            state.update(prompt=index, call=0)
            set_seed(index)
            noise = torch.randn(list(config.image_or_video_shape), device="cuda", dtype=torch.bfloat16)
            video = pipeline.inference(noise=noise, text_prompts=[prompts[index]])
            if state["call"] != 160:
                raise RuntimeError(f"Incomplete four-step trajectory: {state['call']} calls")
            del noise, video
    payload.update(complete=True, wall_seconds=time.perf_counter() - started)
    payload["summary"] = summarize(payload["records"], modes)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
