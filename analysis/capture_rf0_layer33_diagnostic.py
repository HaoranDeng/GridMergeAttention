"""Prepare L33 evidence on the canonical dense trajectory; no candidate runs.

Capture Q/K/V and the original dense output, BF16 unchanged, at calls
033/073/113/153 for heads 0/10/20/30 and the first three canonical prompts.
All four actual steps and all 40 layers still execute densely. No MP4 is saved.
An external launcher must enforce one clean GPU. This script has not run on GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "analysis"):
    sys.path.insert(0, str(directory))
from screen_native_seed_trajectory import check_capture_environment
from utils.prompt_contract import build_prompt_manifest, load_nonempty_prompts, validate_canonical_prompt_contract

HEADS, TARGET_CALLS = (0, 10, 20, 30), (33, 73, 113, 153)
FULL_SHAPE, SAVED_SHAPE = (1, 75600, 40, 128), (1, 75600, 4, 128)
ACTUAL_TIMESTEPS = (1000.0, 937.5, 833.3333129882812, 625.0)
FROZEN_STREAM_SHA256 = "cc2fdcdd3fd83823c31548d7248529a52c781a81358bb717413d22fff969b538"
RAW_CAPTURE_BYTES = 12 * 4 * math.prod(SAVED_SHAPE) * 2
SOURCES = ("analysis/capture_rf0_layer33_diagnostic.py", "analysis/screen_native_seed_trajectory.py",
           "analysis/verify_prompt_contract.py", "utils/prompt_contract.py", "generate.py",
           "pipeline/token_merging_wan.py", "utils/wan_wrapper.py", "wan/modules/model.py",
           "wan/rmsnorm_ops.py", "attention/core.py")


def validate_config(config):
    tm, model = config.get("token_merging_args", {}), config.get("model_kwargs", {})
    if config.get("causal", False) or tm.get("enable", False) or tm.get("attention_backend", "sdpa") != "sdpa":
        raise ValueError("Require bidirectional full dense SDPA with token merging disabled")
    if config.get("image_or_video_shape") != [1, 21, 16, 90, 160] or config.get("denoising_step_list") != [1000, 750, 500, 250]:
        raise ValueError("Require the canonical 720p four-step configuration")
    if not config.get("warp_denoising_step", False) or model.get("timestep_shift") != 5.0:
        raise ValueError("Require the original warped timestep schedule")
    if model.get("model_name") != "Wan2.1-T2V-14B-StepDistill-CfgDistill" or model.get("load_pretrained_weights") is not False:
        raise ValueError("Require the audited 14B model loaded through the explicit checkpoint")


def call_record(prompt, call, layer, step, timestep):
    if prompt not in range(3) or call not in range(160) or layer != call % 40 or step != call // 40:
        raise ValueError("Unexpected prompt/call/layer/step order")
    if timestep != ACTUAL_TIMESTEPS[step]:
        raise ValueError("Actual timestep differs from the frozen dense trajectory")
    return {"prompt_index": prompt, "call_index": call, "layer_index": layer,
            "step_index": step, "actual_timestep": timestep,
            "capture_file": f"prompt{prompt:03d}_call{call:03d}_layer33.pt" if call in TARGET_CALLS else None}


def validate_complete(records):
    expected = [call_record(p, c, c % 40, c // 40, ACTUAL_TIMESTEPS[c // 40]) for p in range(3) for c in range(160)]
    if records != expected:
        raise ValueError("Require exactly 480 ordered calls: 3 prompts x 4 actual steps x 40 layers")
    return [row["capture_file"] for row in records if row["capture_file"] is not None]


def save_status(root, payload):
    (root / "capture_manifest.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def capture(args, prompts, payload):
    import torch
    from omegaconf import OmegaConf
    from generate import build_pipeline, load_generator_checkpoint, set_seed
    import wan.modules.model as wan_model
    import wan.rmsnorm_ops as rmsnorm_ops
    import attention.core as attention_core
    config = OmegaConf.load(args.config)
    validate_config(OmegaConf.to_container(config, resolve=True))
    if attention_core._FORCED_SDPA_BACKEND != "flash" or rmsnorm_ops._USE_TORCH:
        raise ValueError("Match frozen run: WAN_SDPA_BACKEND=flash and native (not Torch) RMSNorm")
    payload.update(actual_forced_sdpa_backend=attention_core._FORCED_SDPA_BACKEND,
        actual_use_torch_rmsnorm=rmsnorm_ops._USE_TORCH,
        rmsnorm_implementation=f"{wan_model.rmsnorm.__module__}.{wan_model.rmsnorm.__name__}",
        rope_implementation=f"{wan_model.apply_rope_video_tokens.__module__}.{wan_model.apply_rope_video_tokens.__name__}")
    save_status(args.output_root, payload)
    torch.cuda.init()  # Advertise the job before the large CPU checkpoint load.
    pipeline = build_pipeline(config, torch.device("cuda"))
    payload["checkpoint_weights"] = load_generator_checkpoint(pipeline, args.checkpoint, True)
    if payload["checkpoint_weights"] != "raw_state_dict":
        raise ValueError("Require the audited raw_state_dict, not a generator/EMA branch")
    pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to("cuda")
    pipeline.generator.to("cuda")
    pipeline.vae.to("cuda")
    pipeline.eval()
    actual_steps = pipeline.denoising_step_list.tolist()
    if len(pipeline.generator.model.blocks) != 40 or tuple(actual_steps) != ACTUAL_TIMESTEPS:
        raise ValueError("Require all 40 layers and the exact four actual warped timesteps")
    payload.update(actual_warped_timesteps=actual_steps, device=torch.cuda.get_device_name(),
                   torch_version=torch.__version__, cuda_version=torch.version.cuda)
    state = {"prompt": None, "call": 0, "layer": None, "step": None, "timestep": None, "generator_calls": 0}
    head_index = torch.tensor(HEADS, device="cuda")
    original_attention, original_forward = wan_model.flash_attention, wan_model.WanSelfAttention.forward
    original_generator = pipeline.generator.forward

    def generator_forward(*inputs, **options):
        step = state["generator_calls"]
        if step not in range(4) or state["call"] != step * 40:
            raise RuntimeError("Generator call count does not match complete 40-layer steps")
        timestep = options["timestep"]
        if tuple(timestep.shape) != (1, 21) or not bool((timestep == ACTUAL_TIMESTEPS[step]).all()):
            raise RuntimeError("Generator received unexpected actual timestep")
        state.update(step=step, timestep=float(timestep.flatten()[0]))
        result = original_generator(*inputs, **options)
        if state["call"] != (step + 1) * 40:
            raise RuntimeError("Generator step did not execute exactly 40 ordered self-attention calls")
        state["generator_calls"] += 1
        return result

    def forward(module, *inputs, **options):
        if state["layer"] is not None or module.enable_token_merging or module.attention_backend != "sdpa" or tuple(module.window_size) != (-1, -1):
            raise RuntimeError("Require nonnested full dense self attention in every layer")
        state["layer"] = int(module.layer_idx)
        try:
            return original_forward(module, *inputs, **options)
        finally:
            state["layer"] = None

    def attention(q, k, v, *inputs, **options):
        dense = original_attention(q, k, v, *inputs, **options)
        if state["layer"] is None:  # Cross attention is neither captured nor counted.
            return dense
        if any(tuple(x.shape) != FULL_SHAPE or x.dtype != torch.bfloat16 for x in (q, k, v, dense)):
            raise RuntimeError("Require original BF16 [1,75600,40,128] QKV and dense output")
        if inputs or any(options.get(key) is not None for key in ("q_lens", "k_lens", "q_scale", "softmax_scale")):
            raise RuntimeError("Unexpected scaled/masked self-attention capture")
        if options.get("causal", False) or options.get("dropout_p", 0) != 0 or options.get("deterministic", False) or tuple(options.get("window_size", (-1, -1))) != (-1, -1):
            raise RuntimeError("Capture requires unmasked noncausal forced Flash SDPA")
        row = call_record(state["prompt"], state["call"], state["layer"], state["step"], state["timestep"])
        if row["capture_file"]:
            tensors = {name: value.index_select(2, head_index).contiguous().cpu()
                       for name, value in zip(("q", "k", "v", "dense_output"), (q, k, v, dense))}
            if any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
                raise RuntimeError("Nonfinite captured tensor")
            metadata = row | {"heads": list(HEADS), "shape": list(SAVED_SHAPE), "dtype": "torch.bfloat16",
                "prompt": payload["prompt_manifest"]["prompts"][state["prompt"]],
                "sample_seed": state["prompt"], "checkpoint_weights": payload["checkpoint_weights"],
                "checkpoint_identity": payload["checkpoint_identity"], "config_sha256": payload["config_sha256"],
                "capture_source_sha256": payload["source_sha256"]["analysis/capture_rf0_layer33_diagnostic.py"],
                "frozen_stream_sha256": FROZEN_STREAM_SHA256,
                "qk_position": "after Q/K RMSNorm and RoPE; V after projection",
                "scale": 128 ** -0.5, "reference": "Same call's original 40-head forced Flash SDPA output, sliced to saved heads"}
            destination = args.output_root / row["capture_file"]
            with destination.open("xb") as stream:
                torch.save(tensors | {"metadata": metadata}, stream)
            payload["captures"].append(metadata | {"bytes": destination.stat().st_size})
            del tensors
        payload["records"].append(row)
        state["call"] += 1
        if state["call"] % 40 == 0:
            save_status(args.output_root, payload)
            print(f"prompt={state['prompt']} calls={state['call']} captures={len(payload['captures'])}", flush=True)
        return dense  # Only the untouched dense output advances the model.

    save_status(args.output_root, payload)
    with torch.inference_mode(), patch.object(wan_model.WanSelfAttention, "forward", forward), \
            patch.object(wan_model, "flash_attention", attention), patch.object(pipeline.generator, "forward", generator_forward):
        for index in range(3):
            state.update(prompt=index, call=0, generator_calls=0, step=None, timestep=None)
            set_seed(index)  # Canonical base seed 0 + prompt index, one sample, same as generate/frozen stream.
            noise = torch.randn(list(config.image_or_video_shape), device="cuda", dtype=torch.bfloat16)
            video = pipeline.inference(noise=noise, text_prompts=[prompts[index]])
            if state["call"] != 160 or state["generator_calls"] != 4:
                raise RuntimeError("Incomplete four-step dense trajectory")
            del noise, video  # Original decoder may run; no image/video encoder or MP4 writer is invoked.
    expected = validate_complete(payload["records"])
    if len(payload["captures"]) != 12 or sorted(p.name for p in args.output_root.glob("*.pt")) != sorted(expected):
        raise RuntimeError("Require exactly 12 expected capture files")
    payload.update(complete=True, total_self_attention_calls=len(payload["records"]), total_captures=12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dense_wan_14b_720p_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    check_capture_environment(os.environ)
    if args.output_root.exists():
        parser.error("Refusing to reuse an existing capture root")
    prompts = load_nonempty_prompts(args.prompt_path)
    validate_canonical_prompt_contract(args.prompt_path, prompts)
    manifest = build_prompt_manifest(args.prompt_path, prompts, selected_count=3, seed=0, num_samples=1,
                                     vbench_filenames=False, allow_noncanonical=False)
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    if hashes["analysis/screen_native_seed_trajectory.py"] != FROZEN_STREAM_SHA256:
        parser.error("Frozen trajectory source changed; review the protocol before capturing")
    checkpoint = Path(args.checkpoint)
    info = checkpoint.stat()
    args.output_root.mkdir(parents=True, exist_ok=False)
    payload = {"complete": False, "records": [], "captures": [], "prompt_manifest": manifest,
        "settings": vars(args) | {"output_root": str(args.output_root)}, "heads": list(HEADS),
        "capture_calls_per_prompt": list(TARGET_CALLS), "source_sha256": hashes,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(), "use_ema_requested": True,
        "checkpoint_identity": {"path": str(checkpoint.resolve()), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
                                "note": "Metadata plus actual loaded branch; no new full-file checksum claim"},
        "environment": {key: os.environ.get(key) for key in ("WAN_MODEL_ROOT", "WAN_SDPA_BACKEND", "TOKEN_MERGING_FORCE_TORCH_RMSNORM", "CUDA_VISIBLE_DEVICES")},
        "planned_raw_tensor_bytes": RAW_CAPTURE_BYTES, "expected_total_calls": 480, "expected_captures": 12,
        "scope": "External dense trajectory only; no selector, candidate, fitting, layer exception, or video generation experiment",
        "rf0_note": "RF0 is only a later selector policy: saved QKV contain all 21 frames, and capture runs no selection",
        "storage": "torch.save preserves BF16 Q/K/V and original same-call dense output; no FP16 conversion",
        "purity": "External launcher must enforce and monitor one exclusively clean GPU",
        "limitations": "Four selected heads and L33 only; no candidate quality/runtime claim; VAE decode is discarded, no MP4 saved"}
    save_status(args.output_root, payload)
    try:
        if shutil.disk_usage(args.output_root).free < RAW_CAPTURE_BYTES + 1024 ** 3:
            raise RuntimeError("Require capture payload plus 1 GiB free disk reserve")
        subprocess.run([sys.executable, str(ROOT / "analysis/verify_prompt_contract.py"), "--prompt_path", args.prompt_path,
            "--num_videos", "3", "--seed", "0", "--num_samples", "1", "--manifest_out",
            str(args.output_root / "prompt_preflight_manifest.json")], check=True, capture_output=True, text=True)
        preflight = json.loads((args.output_root / "prompt_preflight_manifest.json").read_text(encoding="utf-8"))
        if preflight != manifest:
            raise RuntimeError("Independent prompt preflight and capture manifest disagree")
        (args.output_root / "prompt_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        capture(args, prompts, payload)
    except BaseException as exc:
        payload.update(complete=False, error=f"{type(exc).__name__}: {exc}")
        save_status(args.output_root, payload)
        (args.output_root / "exit_code").write_text("1\n", encoding="utf-8")
        raise
    save_status(args.output_root, payload)
    (args.output_root / "exit_code").write_text("0\n", encoding="utf-8")


if __name__ == "__main__":
    main()
