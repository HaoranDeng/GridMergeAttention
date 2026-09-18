"""Strict CPU-only artifact audit for completed RF0 Hellinger/partition/Mass-GS videos.

No decoding, quality calculation, model import, GPU call, or remote access.
Only an explicitly requested new --output file is written. Missing generation
completion is invalid; absent optional quality remains unavailable, not zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "analysis"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from utils.prompt_contract import (load_nonempty_prompts, prompt_sha256, validate_canonical_prompt_contract)
from validate_joint_frame_video_run import (
    CONTRACT, NAMES, PROTOCOL, clean_stage, manifest_signature, quality_summary,
    read_json, require, runtime_summary,
)

MODES = {
    "hellinger_rf0": "hellinger_span_alpha0p2_c32_p16_a2_j4_t1_rf0_u16_cg45",
    "partition_rf0": "entropy_partition_mass_c16_p16_g256_j4_t1_rf0_u16_cg16",
    "mass_c32_rf0": "entropy_mass_span_c32_p16_g256_j4_t1_rf0_u16_cg45",
}
QUALITY_KEYS = {"hellinger_rf0": {"hellinger_uniform16_rf0", "rf0"},
                "partition_rf0": {"partition_c16_rf0"},
                "mass_c32_rf0": {"mass_c32_rf0"}}


def artifact(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def canonical_signature():
    path = ROOT / "prompts" / "MovieGenVideoBench_extended.txt"
    prompts = load_nonempty_prompts(path)
    validate_canonical_prompt_contract(path, prompts)
    return [(text, prompt_sha256(text)) for text in prompts[:20]]


def strict_prompt_manifest(manifest, label):
    signature = manifest_signature(manifest)
    require(signature == canonical_signature(), f"{label}: one or more of the 20 canonical prompt text/hashes differ")
    require(manifest.get("canonical_contract_enforced") is True, f"{label}: canonical prompt enforcement not recorded")
    return signature


def _portable_path(value):
    return PurePosixPath(str(value).replace("\\", "/"))


def _method_label(label, method_key, field):
    text = str(label).lower()
    require("rf0" in text and method_key.split("_")[0] in text,
            f"{field}: must identify {method_key}, not an RF10 or different-method control")
    require("rf10" not in text and "joint" not in text, f"{field}: contradictory router/method")
    if method_key == "mass_c32_rf0":
        require("c32" in text and not any(other in text for other in ("c16", "partition", "hellinger")),
                f"{field}: must identify Mass-GS C32 RF0 without contradictory method labels")


def verify_provenance(run_dir, manifest, runtime, source):
    """Verify frozen copies, never compare against today's possibly changed checkout."""
    entries = source.get("files")
    require(isinstance(entries, list) and bool(entries), "source_provenance.files must list frozen source hashes")
    files = {}
    for entry in entries:
        relative = _portable_path(entry["path"])
        require(not relative.is_absolute() and ".." not in relative.parts and ":" not in str(relative),
                "source provenance paths must stay inside source/")
        name = str(relative)
        require(name not in files, f"duplicate source provenance path: {name}")
        frozen = run_dir / "source" / Path(*relative.parts)
        actual = artifact(frozen)
        require(actual["sha256"] == entry.get("sha256"), f"frozen source hash mismatch: {name}")
        files[name] = actual
    config_name = _portable_path(manifest["config"]).name
    config_key = "configs/" + config_name
    require(config_key in files, "run config missing from frozen source provenance")
    config_artifact = files[config_key]
    require(manifest.get("config_sha256") == config_artifact["sha256"], "run config SHA256 differs from frozen source")
    config_text = Path(config_artifact["path"]).read_text(encoding="utf-8")
    matches = re.findall(r"^\s*seed_mode:\s*([^\r\n#]+)", config_text, re.MULTILINE)
    require(len(matches) == 1, "frozen config must contain exactly one seed_mode")
    mode = matches[0].strip().strip("\"'")
    require(mode in MODES.values(), "frozen config is not one of the supported exact RF0 methods")
    method_key = next(key for key, value in MODES.items() if value == mode)
    selectors = {
        "hellinger_rf0": {"attention/adaptive_seed_triton.py"},
        "partition_rf0": {"attention/partition_mass_seed_triton.py", "attention/entropy_mass_seed_triton.py"},
        "mass_c32_rf0": {"attention/entropy_mass_seed_triton.py"},
    }
    required = {"generate.py", "wan/modules/model.py", "attention/token_merging.py", config_key} | selectors[method_key]
    if method_key == "mass_c32_rf0":
        require(config_key == "configs/token_merging_wan_720p_entropy_mass_span_rf0_fewstep.yaml",
                "Mass-GS C32 RF0 requires its declared few-step config")
    require(required <= files.keys(), "missing required generation/selector source snapshots")
    for document, label in ((manifest, "run_manifest"), (source, "source_provenance")):
        _method_label(document.get("method"), method_key, label + ".method")
        if document.get("seed_mode") is not None:
            require(document["seed_mode"] == mode, label + ".seed_mode differs from frozen config")
    require(_portable_path(runtime.get("config")) == _portable_path(manifest["config"]), "runtime/run config paths differ")
    require(runtime.get("checkpoint") == manifest.get("checkpoint") and bool(runtime.get("checkpoint")),
            "runtime/run checkpoint paths differ or are missing")
    require(runtime.get("checkpoint_weights") == "raw_state_dict", "actual loaded checkpoint branch must be raw_state_dict")
    require(manifest.get("checkpoint_weights") in (None, "raw_state_dict"), "run manifest contradicts actual raw checkpoint branch")
    if source.get("remote_base_commit") is not None:
        require(source["remote_base_commit"] == manifest.get("git_commit"), "recorded remote source commit differs")
    for key, expected in {"router_frame": 0, "num_videos": 20, "seed": 0, "num_samples": 1,
                          "frames": 81, "denoise_steps": 4, "attention_calls_per_video": 160}.items():
        if key in source.get("planned", {}):
            require(source["planned"][key] == expected, f"source planned.{key} contradicts RF0 protocol")
    return {"method_key": method_key, "seed_mode": mode, "config": config_artifact,
            "source_files": files, "recorded_git_commit": manifest.get("git_commit"),
            "checkpoint_weights": "raw_state_dict", "checkpoint": runtime["checkpoint"],
            "manifest_checkpoint_backfilled": manifest.get("checkpoint_weights") == "raw_state_dict"}


def validate(run_dir, *, quality_path=None, quality_method=None, quality_stage_dir=None):
    run_dir = Path(run_dir)
    generation = clean_stage(run_dir, "generation.exitcode") | {"valid": True}
    video_dir = run_dir / "videos"
    videos = sorted(path.name for path in video_dir.iterdir() if path.suffix.lower() == ".mp4")
    require(videos == NAMES, "videos must contain exactly 000-00.mp4 through 019-00.mp4")
    require(all((video_dir / name).is_file() and (video_dir / name).stat().st_size > 0 for name in NAMES),
            "all 20 video files must be nonempty regular files")
    input_paths = {"runtime": run_dir / "runtime.json", "run_manifest": run_dir / "run_manifest.json",
                   "source_provenance": run_dir / "source_provenance.json",
                   "prompt_manifest": video_dir / "prompt_manifest.json",
                   "generation_exitcode": run_dir / "generation.exitcode", "generation_purity": run_dir / "purity.log"}
    runtime, manifest, source = (read_json(input_paths[name]) for name in ("runtime", "run_manifest", "source_provenance"))
    signature = strict_prompt_manifest(read_json(input_paths["prompt_manifest"]), "video prompt manifest")
    require(strict_prompt_manifest(runtime["prompt_manifest"], "runtime prompt manifest") == signature, "runtime prompts differ")
    preflights = [run_dir / name for name in ("prompt_preflight_manifest.json", "preflight_prompt_manifest.json")]
    preflights = [path for path in preflights if path.is_file()]
    require(bool(preflights), "missing preflight prompt manifest")
    for index, path in enumerate(preflights):
        require(strict_prompt_manifest(read_json(path), str(path)) == signature, "preflight prompt manifest differs")
        input_paths["prompt_preflight" if index == 0 else "prompt_preflight_extra"] = path
    for key, expected in {"num_videos": 20, "frames": 81, "seed": 0, "denoise_steps": 4,
                          "attention_calls_per_video": 160}.items():
        require(type(manifest.get(key)) is int and manifest[key] == expected, f"run_manifest.{key} mismatch")
    if "num_samples" in manifest:
        require(manifest["num_samples"] == 1, "run_manifest.num_samples mismatch")
    provenance = verify_provenance(run_dir, manifest, runtime, source)
    scopes = runtime_summary(runtime, signature)
    for item in scopes.values():
        item.update(speedup_vs_sdpa=None, mean_sdpa_attention_ms=None,
                    mean_inference_plus_export_seconds=item["mean_inference_seconds"] + item["mean_export_seconds"])
    stages = {"generation": generation, "quality": {"valid": None, "status": "not_provided"}}
    quality = None
    if quality_path is not None:
        quality_path = Path(quality_path)
        stage_dir = Path(quality_stage_dir) if quality_stage_dir else run_dir / "quality_evaluation"
        stages["quality"] = clean_stage(stage_dir, "exit_code") | {"valid": True, "status": "complete"}
        document = read_json(quality_path)
        for key, expected in PROTOCOL.items():
            require(document["protocol"].get(key) == expected, f"quality protocol.{key} mismatch")
        methods = document["methods"]
        if quality_method is None:
            require(len(methods) == 1, "--quality-method is required for multi-method quality JSON")
            quality_method = next(iter(methods))
        require(quality_method in methods, "quality method not found")
        require(quality_method in QUALITY_KEYS[provenance["method_key"]],
                "quality method key does not identify this RF0 run; RF10/other controls are not accepted")
        item = methods[quality_method]
        if item.get("directory") is not None:
            require(_portable_path(item["directory"]).parent.name == run_dir.name
                    and _portable_path(item["directory"]).name == "videos", "quality method directory does not identify this run")
        summary = quality_summary(item, quality_method)
        reference = document.get("reference")
        require(isinstance(reference, str) and bool(reference), "quality reference missing")
        planned_reference = source.get("planned", {}).get("quality_reference")
        if planned_reference is not None:
            require(_portable_path(reference) == _portable_path(planned_reference), "quality reference differs from predeclared source protocol")
        quality = {"valid": True, "method": quality_method, "reference": reference, "summary": summary,
                   "videos": 20, "frames": 1620, "aggregation": "arithmetic mean of 20 per-video metrics",
                   "protocol": document["protocol"], "method_directory_recorded": item.get("directory") is not None,
                   "association": "Caller-selected quality artifact/method; evaluator may not record its method directory"}
        input_paths.update(quality=quality_path, quality_exitcode=stage_dir / "exit_code", quality_purity=stage_dir / "purity.log")
    return {"valid": True, "method_key": provenance["method_key"], "method": manifest["method"],
            "run_dir": str(run_dir.resolve()), "stages": stages, "provenance": provenance,
            "prompt_contract": CONTRACT, "verified_prompt_indices": list(range(20)),
            "generation_runtime": scopes, "quality": quality,
            "inputs": {key: artifact(path) for key, path in input_paths.items()},
            "video_files": [{"name": name, "bytes": (video_dir / name).stat().st_size} for name in NAMES],
            "runtime_definitions": {"mean_attention_ms": "Complete GCA attention operator: selector + merge + route; excludes projections/RoPE",
                                    "mean_full_attention_ms": "Full self-attention module including projections/RoPE/output projection",
                                    "steady19": "Indices 1..19 only, excluding video 000-00; all20 remains the primary complete-run summary"},
            "limits": ["MP4 files are counted but not decoded; 81 decoded frames are certified only when strict quality records are supplied.",
                       "Loaded raw weights are verified from runtime provenance, not rehashed checkpoint bytes.",
                       "Purity is verified from terminal launcher artifacts; no GPU is contacted.",
                       "No matched dense video runtime is supplied: SDPA speedup is unavailable, not borrowed from QKV benchmarks.",
                       "Inference timing excludes model loading and MP4 export; export is reported separately."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-json", type=Path)
    parser.add_argument("--quality-method")
    parser.add_argument("--quality-stage-dir", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing audit report")
    if args.quality_json is None and (args.quality_method is not None or args.quality_stage_dir is not None):
        parser.error("quality options require --quality-json")
    try:
        report = validate(args.run_dir, quality_path=args.quality_json,
                          quality_method=args.quality_method, quality_stage_dir=args.quality_stage_dir)
    except (ValueError, KeyError, TypeError, OSError) as error:
        report = {"valid": False, "run_dir": str(args.run_dir.resolve()), "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"valid": report["valid"], "output": str(args.output), "error": report.get("error")}))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
