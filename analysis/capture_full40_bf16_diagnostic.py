"""External dense-only full40 BF16 evidence, never a GCA inference variant.

Run all 480 self-attention calls of three canonical prompts, four steps and
40 layers. Save original Q/K/V and same-call dense output for layers 0/33
at each step (24 files, about 74 GB). No head replication, dtype conversion,
selection, fitting, MP4 export, or candidate attention is performed.
An independently monitoring clean-GPU parent is mandatory.
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
from analysis.capture_rf0_layer33_diagnostic import validate_config, ACTUAL_TIMESTEPS, FROZEN_STREAM_SHA256
from analysis.screen_native_seed_trajectory import check_capture_environment
from utils.prompt_contract import build_prompt_manifest, load_nonempty_prompts, validate_canonical_prompt_contract

HEADS = tuple(range(40))
TARGET_CALLS = (0, 33, 40, 73, 80, 113, 120, 153)
FULL_SHAPE = (1, 75600, 40, 128)
EXPECTED_CALLS, EXPECTED_CAPTURES = 480, 24
BYTES_PER_CAPTURE = 4 * math.prod(FULL_SHAPE) * 2
RAW_CAPTURE_BYTES = EXPECTED_CAPTURES * BYTES_PER_CAPTURE
DISK_RESERVE_BYTES = 1024 ** 3
SOURCES = (
    "analysis/capture_full40_bf16_diagnostic.py", "analysis/capture_rf0_layer33_diagnostic.py",
    "analysis/screen_native_seed_trajectory.py", "analysis/verify_prompt_contract.py",
    "utils/prompt_contract.py", "generate.py", "pipeline/token_merging_wan.py",
    "utils/wan_wrapper.py", "wan/modules/model.py", "wan/rmsnorm_ops.py", "attention/core.py",
)


def require_parent(environment=None, parent_pid=None):
    environment = os.environ if environment is None else environment
    parent_pid = os.getppid() if parent_pid is None else parent_pid
    if parent_pid <= 1 or environment.get("FULL40_CAPTURE_PARENT") != str(parent_pid):
        raise RuntimeError("full40 capture requires its live monitoring parent")


def call_record(prompt, call, layer, step, timestep):
    if prompt not in range(3) or call not in range(160) or layer != call % 40 or step != call // 40:
        raise ValueError("unexpected prompt/call/layer/step order")
    if timestep != ACTUAL_TIMESTEPS[step]:
        raise ValueError("actual timestep differs from canonical dense trajectory")
    return {"prompt_index": prompt, "call_index": call, "layer_index": layer,
            "step_index": step, "actual_timestep": timestep,
            "capture_file": f"prompt{prompt:03d}_call{call:03d}_layer{layer}.pt" if call in TARGET_CALLS else None}


def validate_complete(records):
    expected = [call_record(p, c, c % 40, c // 40, ACTUAL_TIMESTEPS[c // 40])
                for p in range(3) for c in range(160)]
    if records != expected:
        raise ValueError("require exactly 480 ordered dense calls, not a shorter or hybrid trajectory")
    files = [row["capture_file"] for row in records if row["capture_file"] is not None]
    if len(files) != EXPECTED_CAPTURES or len(set(files)) != EXPECTED_CAPTURES:
        raise ValueError("require 24 disjoint full40 capture filenames")
    return files


def require_disk_budget(free_bytes, *, remaining_captures=EXPECTED_CAPTURES):
    if (not isinstance(remaining_captures, int) or isinstance(remaining_captures, bool)
            or remaining_captures not in range(EXPECTED_CAPTURES + 1)):
        raise ValueError("invalid remaining capture count")
    required = remaining_captures * BYTES_PER_CAPTURE + DISK_RESERVE_BYTES
    if free_bytes < required:
        raise RuntimeError(f"require at least {required} free bytes for full40 capture plus reserve")
    return required


def fingerprint(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("capture changed while its post-save checksum was computed")
    return {"path": str(path.resolve()), "bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns, "sha256": digest.hexdigest()}


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def save_status(root, payload):
    (root / "capture_manifest.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def capture(args, prompts, payload):
    require_parent()  # Before Torch import, model loading, or GPU initialization.
    import torch
    from omegaconf import OmegaConf
    from generate import build_pipeline, load_generator_checkpoint, set_seed
    import wan.modules.model as wan_model
    import wan.rmsnorm_ops as rmsnorm_ops
    import attention.core as attention_core

    config = OmegaConf.load(args.config)
    validate_config(OmegaConf.to_container(config, resolve=True))
    if attention_core._FORCED_SDPA_BACKEND != "flash" or rmsnorm_ops._USE_TORCH:
        raise ValueError("require canonical forced Flash SDPA and native RMSNorm")
    payload.update(actual_forced_sdpa_backend=attention_core._FORCED_SDPA_BACKEND,
        actual_use_torch_rmsnorm=rmsnorm_ops._USE_TORCH,
        rmsnorm_implementation=f"{wan_model.rmsnorm.__module__}.{wan_model.rmsnorm.__name__}",
        rope_implementation=f"{wan_model.apply_rope_video_tokens.__module__}.{wan_model.apply_rope_video_tokens.__name__}")
    save_status(args.output_root, payload)
    torch.cuda.init()  # Register on the parent's monitored GPU before checkpoint load.
    pipeline = build_pipeline(config, torch.device("cuda"))
    payload["checkpoint_weights"] = load_generator_checkpoint(pipeline, args.checkpoint, True)
    if payload["checkpoint_weights"] != "raw_state_dict":
        raise ValueError("require the audited raw_state_dict, not a generator/EMA branch")
    pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to("cuda")
    pipeline.generator.to("cuda")
    pipeline.vae.to("cuda")
    pipeline.eval()
    if len(pipeline.generator.model.blocks) != 40 or tuple(pipeline.denoising_step_list.tolist()) != ACTUAL_TIMESTEPS:
        raise ValueError("require all 40 layers and exact four actual warped timesteps")
    payload.update(actual_warped_timesteps=list(ACTUAL_TIMESTEPS), device=torch.cuda.get_device_name(),
                   torch_version=torch.__version__, cuda_version=torch.version.cuda)
    state = {"prompt": None, "call": 0, "layer": None, "step": None, "timestep": None, "generator_calls": 0}
    original_attention = wan_model.flash_attention
    original_forward = wan_model.WanSelfAttention.forward
    original_generator = pipeline.generator.forward

    def generator_forward(*inputs, **options):
        step = state["generator_calls"]
        if step not in range(4) or state["call"] != step * 40:
            raise RuntimeError("generator call count does not match complete 40-layer steps")
        timestep = options["timestep"]
        if tuple(timestep.shape) != (1, 21) or not bool((timestep == ACTUAL_TIMESTEPS[step]).all()):
            raise RuntimeError("generator received an unexpected actual timestep")
        state.update(step=step, timestep=float(timestep.flatten()[0]))
        result = original_generator(*inputs, **options)
        if state["call"] != (step + 1) * 40:
            raise RuntimeError("generator step did not execute 40 ordered self-attention calls")
        state["generator_calls"] += 1
        return result

    def forward(module, *inputs, **options):
        if (state["layer"] is not None or module.enable_token_merging
                or module.attention_backend != "sdpa" or tuple(module.window_size) != (-1, -1)):
            raise RuntimeError("require nonnested full dense self-attention in every layer")
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
            raise RuntimeError("require original BF16 [1,75600,40,128] QKV and same-call dense output")
        if inputs or any(options.get(key) is not None for key in ("q_lens", "k_lens", "q_scale", "softmax_scale")):
            raise RuntimeError("unexpected scaled/masked self-attention capture")
        if (options.get("causal", False) or options.get("dropout_p", 0) != 0
                or options.get("deterministic", False) or tuple(options.get("window_size", (-1, -1))) != (-1, -1)):
            raise RuntimeError("capture requires unmasked noncausal forced Flash SDPA")
        row = call_record(state["prompt"], state["call"], state["layer"], state["step"], state["timestep"])
        if row["capture_file"]:
            require_parent()
            require_disk_budget(shutil.disk_usage(args.output_root).free,
                                remaining_captures=EXPECTED_CAPTURES - len(payload["captures"]))
            # Preserve every original head and bit of BF16 tensor storage.
            tensors = {name: value.detach().contiguous().cpu()
                       for name, value in zip(("q", "k", "v", "dense_output"), (q, k, v, dense))}
            if any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
                raise RuntimeError("nonfinite full captured tensor")
            metadata = row | {"heads": list(HEADS), "shape": list(FULL_SHAPE), "dtype": "torch.bfloat16",
                "prompt": payload["prompt_manifest"]["prompts"][state["prompt"]],
                "sample_seed": state["prompt"], "checkpoint_weights": payload["checkpoint_weights"],
                "checkpoint_identity": payload["checkpoint_identity"], "config_sha256": payload["config_sha256"],
                "capture_source_sha256": payload["source_sha256"]["analysis/capture_full40_bf16_diagnostic.py"],
                "frozen_stream_sha256": FROZEN_STREAM_SHA256, "all_tensors_finite": True,
                "qk_position": "after Q/K RMSNorm and RoPE; V after projection", "scale": 128 ** -0.5,
                "reference": "Same call's original full40 forced Flash SDPA output; no head slicing or replication"}
            destination = args.output_root / row["capture_file"]
            with destination.open("xb") as stream:
                torch.save(tensors | {"metadata": metadata}, stream)
            receipt = fingerprint(destination)
            if receipt["bytes"] < BYTES_PER_CAPTURE:
                raise RuntimeError("saved full40 capture is smaller than its tensor payload")
            payload["captures"].append(metadata | {"file": receipt,
                "bytes": receipt["bytes"], "sha256": receipt["sha256"]})
            del tensors
            save_status(args.output_root, payload)
        payload["records"].append(row)
        state["call"] += 1
        if state["call"] % 40 == 0:
            save_status(args.output_root, payload)
            print(f"prompt={state['prompt']} calls={state['call']} full40_captures={len(payload['captures'])}", flush=True)
        return dense  # Only the untouched original dense result advances the model.

    save_status(args.output_root, payload)
    with torch.inference_mode(), patch.object(wan_model.WanSelfAttention, "forward", forward), \
            patch.object(wan_model, "flash_attention", attention), patch.object(pipeline.generator, "forward", generator_forward):
        for index in range(3):
            state.update(prompt=index, call=0, generator_calls=0, step=None, timestep=None)
            set_seed(index)  # Canonical base seed 0 + prompt index, as in generate.py.
            noise = torch.randn(list(config.image_or_video_shape), device="cuda", dtype=torch.bfloat16)
            video = pipeline.inference(noise=noise, text_prompts=[prompts[index]])
            if state["call"] != 160 or state["generator_calls"] != 4:
                raise RuntimeError("incomplete four-step dense trajectory")
            del noise, video  # Decoder output is discarded; no MP4 writer is invoked.
    expected = validate_complete(payload["records"])
    if (len(payload["captures"]) != EXPECTED_CAPTURES
            or sorted(p.name for p in args.output_root.glob("*.pt")) != sorted(expected)):
        raise RuntimeError("require exactly the 24 expected full40 capture files")
    for item in payload["captures"]:
        info = Path(item["file"]["path"]).stat()
        if (info.st_size, info.st_mtime_ns) != (item["file"]["bytes"], item["file"]["mtime_ns"]):
            raise RuntimeError("capture metadata changed after its post-save checksum")
    payload.update(complete=True, total_self_attention_calls=EXPECTED_CALLS, total_captures=EXPECTED_CAPTURES)


def main():
    require_parent()  # Must fail before importing model/GPU dependencies.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dense_wan_14b_720p_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    check_capture_environment(os.environ)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or visible == "-1":
        parser.error("require one parent-selected visible GPU")
    if args.output_root.exists():
        parser.error("refusing to reuse an existing capture root")
    prompts = load_nonempty_prompts(args.prompt_path)
    validate_canonical_prompt_contract(args.prompt_path, prompts)
    manifest = build_prompt_manifest(args.prompt_path, prompts, selected_count=3, seed=0,
                                    num_samples=1, vbench_filenames=False, allow_noncanonical=False)
    hashes = source_hashes()
    if hashes["analysis/screen_native_seed_trajectory.py"] != FROZEN_STREAM_SHA256:
        parser.error("frozen canonical trajectory helper changed; review before capture")
    checkpoint = Path(args.checkpoint)
    checkpoint_info = checkpoint.stat()
    args.output_root.mkdir(parents=True, exist_ok=False)
    payload = {"complete": False, "records": [], "captures": [], "prompt_manifest": manifest,
        "settings": vars(args) | {"output_root": str(args.output_root)}, "heads": list(HEADS),
        "capture_calls_per_prompt": list(TARGET_CALLS), "source_sha256": hashes,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(), "use_ema_requested": True,
        "checkpoint_identity": {"path": str(checkpoint.resolve()), "bytes": checkpoint_info.st_size,
            "mtime_ns": checkpoint_info.st_mtime_ns, "note": "metadata and actual loaded branch; not a full-file checksum"},
        "parent_pid": os.getppid(), "worker_pid": os.getpid(),
        "environment": {key: os.environ.get(key) for key in ("WAN_MODEL_ROOT", "WAN_SDPA_BACKEND",
            "TOKEN_MERGING_FORCE_TORCH_RMSNORM", "CUDA_VISIBLE_DEVICES")},
        "planned_raw_tensor_bytes": RAW_CAPTURE_BYTES, "expected_total_calls": EXPECTED_CALLS,
        "expected_captures": EXPECTED_CAPTURES, "minimum_disk_reserve_bytes": DISK_RESERVE_BYTES,
        "scope": "External pure dense diagnostic; every layer/step remains dense, no candidate method or training",
        "storage": "torch.save original full40 BF16 Q/K/V and same-call dense output; post-save SHA256 per file",
        "purity": "Parent must independently admit and continuously monitor one exclusively idle H200",
        "limitations": "Three prompts, early/late layers at all four steps; not a video metric or runtime result"}
    save_status(args.output_root, payload)
    try:
        require_disk_budget(shutil.disk_usage(args.output_root).free)
        subprocess.run([sys.executable, str(ROOT / "analysis/verify_prompt_contract.py"),
            "--prompt_path", args.prompt_path, "--num_videos", "3", "--seed", "0", "--num_samples", "1",
            "--manifest_out", str(args.output_root / "prompt_preflight_manifest.json")],
            check=True, capture_output=True, text=True)
        preflight = json.loads((args.output_root / "prompt_preflight_manifest.json").read_text(encoding="utf-8"))
        if preflight != manifest:
            raise RuntimeError("independent prompt preflight differs from capture manifest")
        (args.output_root / "prompt_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        capture(args, prompts, payload)
        require_parent()
        if source_hashes() != hashes or hashlib.sha256(Path(args.config).read_bytes()).hexdigest() != payload["config_sha256"]:
            raise RuntimeError("producer sources or config changed during capture")
        final_checkpoint = checkpoint.stat()
        if (final_checkpoint.st_size, final_checkpoint.st_mtime_ns) != (checkpoint_info.st_size, checkpoint_info.st_mtime_ns):
            raise RuntimeError("checkpoint metadata changed during capture")
    except BaseException as exc:
        payload.update(complete=False, error=f"{type(exc).__name__}: {exc}")
        save_status(args.output_root, payload)
        (args.output_root / "exit_code").write_text("1\n", encoding="utf-8")
        raise
    save_status(args.output_root, payload)
    (args.output_root / "exit_code").write_text("0\n", encoding="utf-8")


if __name__ == "__main__":
    main()
