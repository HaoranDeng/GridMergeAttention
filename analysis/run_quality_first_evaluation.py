"""Frozen BF16 RF0 candidate evaluation against historical Uniform16 HF RF10.

Exactly 20 matched videos, every one of 81 full-resolution RGB frames. Reuses
evaluate_video_quality.py PSNR/SSIM/Alex-LPIPS functions without modifying them.
No generation, metric resizing, timing benchmark, default edit or resume.
Default/legacy --mass-dir evaluates pinned Mass e. Explicit
--candidate-method hellinger_rf0 --candidate-dir evaluates pinned Hellinger g.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import launch_quality_first_videos as guard
from analysis.quality_first_mode_contract import method_contract

EXPERIMENTS = Path("/local2/denghaoran/GridMergeAttention/experiments")
REFERENCE = EXPERIMENTS / "wan14b_720p_4step_dense_flash_matched_extended_ema_20260826/flash/videos"
MASS = EXPERIMENTS / "wan14b_720p_4step_mass_rf0_bf16score_extended_20260905e/videos"
HELLINGER_RF0 = EXPERIMENTS / "wan14b_720p_4step_hellinger_rf0_bf16score_extended_20260905g/videos"
HELLINGER = EXPERIMENTS / "wan14b_720p_4step_hellinger_uniform16_20260904/videos"
METHODS = ("mass_rf0_bf16score", "historical_hellinger_uniform16_rf10_fp8score")
AUDIT_SHA = "5d75c179120fa9e2f25b8f4a9b860578da3f0489eb56aea412a0925d7f8714a5"
MASS_PROVENANCE_SHA = "bf4380917f4d320893bf34810c46132abf8d85510b2c12ab04712fb4e7385a1e"
HELLINGER_RF0_PROVENANCE_SHA = "6e150a32842aa85af48c4528bdb5f46264448396e6b93dfb7f6e90201cafb8d8"
HELLINGER_RF0_BASE_COMMIT = "56491164e091d7a5887663d0dd6aa0772b446546"
CANDIDATE_METHODS = ("mass_rf0", "hellinger_rf0")
SOURCE = "analysis/run_quality_first_evaluation.py"
SOURCES = (SOURCE, "analysis/evaluate_video_quality.py", "utils/prompt_contract.py",
           "utils/__init__.py", "analysis/launch_quality_first_videos.py",
           "analysis/run_quality_first_videos.py", "analysis/quality_first_mode_contract.py")
METRIC_SOURCE_SHAS = {
    "analysis/evaluate_video_quality.py": "ea28e3c629165ef09a123f8145b8701996ea69daed7797bcfc14068910e717bb",
    "utils/prompt_contract.py": "7b99b7318f6236f1885c6af46e06845ae8407650cee30946ec9c8d3a8e9d666b",
}
PROTOCOL = {
    "expected_videos": 20, "expected_frames": 81, "frame_shape": [720, 1280, 3],
    "frame_dtype": "uint8", "pairing": "same filename/prompt/seed; every decoded frame",
    "resize": False, "crop": False, "subsample": False,
    "psnr": "mean of per-frame RGB PSNR, data_range=255",
    "ssim": "per-frame skimage structural_similarity, channel_axis=2, data_range=255",
    "lpips": "AlexNet LPIPS; original RGB /127.5 - 1; no resize",
    "lpips_batch_size": 8, "aggregation": "frame-count-weighted per-video means",
    "target": {"metric": "psnr", "method": METHODS[0], "strictly_greater_than": 12.0},
    "ssim_lpips_policy": "report both; no invented noninferiority or promotion threshold",
    "timing": "not measured; historical RF10 timing not established; no borrowed speedup",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def candidate_spec(method="mass_rf0"):
    """Only two explicitly pinned completed-generation identities are admitted."""
    require(method in CANDIDATE_METHODS, "unsupported pinned candidate method")
    if method == "mass_rf0":
        return {"method": method, "label": METHODS[0], "display": "Mass",
                "directory": MASS, "provenance_sha": MASS_PROVENANCE_SHA,
                "generation_key": "mass_generation", "base_commit": None,
                "selector_source": "attention/entropy_mass_seed_triton.py"}
    return {"method": method, "label": "hellinger_rf0_bf16score", "display": "Hellinger",
            "directory": HELLINGER_RF0, "provenance_sha": HELLINGER_RF0_PROVENANCE_SHA,
            "generation_key": "hellinger_generation", "base_commit": HELLINGER_RF0_BASE_COMMIT,
            "selector_source": "attention/adaptive_seed_triton.py"}


def selected_candidate(args):
    method = getattr(args, "candidate_method", "mass_rf0")
    candidate_spec(method)
    return method


def candidate_directory(args):
    """Keep legacy Mass CLI compatibility without calling Hellinger 'mass'."""
    method = selected_candidate(args)
    current, legacy = getattr(args, "candidate_dir", None), getattr(args, "mass_dir", None)
    require((current is None) != (legacy is None), "supply exactly one of --candidate-dir or --mass-dir")
    require(legacy is None or method == "mass_rf0", "--mass-dir is only valid for mass_rf0")
    return current if current is not None else legacy


def methods_for(method="mass_rf0"):
    return (candidate_spec(method)["label"], METHODS[1])


def protocol_for(method="mass_rf0"):
    # All numerical and historical-control conventions are unchanged.
    return {**PROTOCOL, "target": {**PROTOCOL["target"], "method": methods_for(method)[0]}}


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write_json(path, value, *, exclusive=False):
    guard.worker.write_json(path, value, exclusive=exclusive)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_receipt(path, expected):
    actual = fingerprint(path)
    require(actual["bytes"] == expected["bytes"] and actual["sha256"] == expected["sha256"],
            f"input receipt mismatch: {path}")
    return actual


def pinned_directories(args):
    spec = candidate_spec(selected_candidate(args))
    mapping = {"reference": args.reference_dir, spec["label"]: candidate_directory(args), METHODS[1]: args.hellinger_dir}
    for name, supplied, expected in zip(mapping, mapping.values(), (REFERENCE, spec["directory"], HELLINGER)):
        require(Path(supplied).is_absolute() and Path(supplied).resolve() == expected.resolve(),
                f"{name} must use its exact pinned experiment directory")
    return {name: Path(path).resolve(strict=True) for name, path in mapping.items()}


def verify_inventory(directory, expected=None):
    paths = sorted(Path(directory).glob("*.mp4"))
    names = [f"{index:03d}-00.mp4" for index in range(20)]
    require([path.name for path in paths] == names, "require exactly canonical 000-00 through019-00 MP4s")
    if expected is not None:
        require([record["name"] for record in expected] == names, "reference audit inventory is not exact20")
    records = []
    for index, path in enumerate(paths):
        value = fingerprint(path)
        require(value["bytes"] > 0, "empty encoded video")
        if expected is not None:
            require(value["bytes"] == expected[index]["bytes"] and value["sha256"] == expected[index]["sha256"],
                    f"encoded video changed since independent audit: {path}")
        records.append({"name": path.name, **value})
    return records


def validate_mass_generation(video_dir):
    """Backward-compatible entry point for the original pinned Mass e run."""
    return validate_candidate_generation(video_dir, "mass_rf0")


def validate_candidate_generation(video_dir, method="mass_rf0"):
    """Read only; no model import, checkpoint load or inference."""
    spec = candidate_spec(method)
    display = spec["display"]
    run = Path(video_dir).parent
    provenance_receipt = fingerprint(run / "source_provenance.json")
    require(provenance_receipt["sha256"] == spec["provenance_sha"], f"{display} generation source identity differs from its pinned run")
    provenance = read_json(run / "source_provenance.json")
    declaration = method_contract(method)
    require(provenance["method"] == spec["label"] and provenance["declaration"] == declaration, f"not the BF16 RF0 {display} generation")
    if spec["base_commit"] is not None:
        require(provenance.get("remote_base_commit") == spec["base_commit"], f"{display} generation base commit differs")
    require(provenance["planned"] == {"router_frame": 0, "num_videos": 20, "num_samples": 1, "seed": 0,
                                     "frames": 81, "denoise_steps": 4, "attention_calls_per_video": 160},
            f"{display} generation protocol changed")
    source_root, seen = run / "source", set()
    for record in provenance["files"]:
        path = (source_root / record["path"]).resolve(strict=True)
        require(path.is_relative_to(source_root.resolve()) and record["path"] not in seen, f"unsafe or duplicate {display} source receipt")
        verify_receipt(path, record)
        seen.add(record["path"])
    require({"generate.py", "attention/token_merging.py", spec["selector_source"],
             "analysis/run_quality_first_videos.py", provenance["config"]} <= seen, f"incomplete {display} source provenance")
    completion = read_json(run / "completion.json")
    require(completion.get("valid") is True and completion.get("timing_valid") is True
            and type(completion.get("worker_exit_code")) is int and completion["worker_exit_code"] == 0
            and completion.get("method") == spec["label"] and completion.get("foreign") == []
            and completion.get("failure") is None and completion.get("monitor_polls", 0) > 0,
            f"{display} generation did not complete under its clean-GPU parent")
    require((run / "purity.log").stat().st_size == 0 and (run / "generation.exitcode").read_text().strip() == "0",
            f"{display} generation exit/purity mismatch")
    files = {"source_provenance.json": provenance_receipt}
    for name, key in (("dispatch_audit.json", "dispatch_audit_sha256"), ("runtime.json", "runtime_sha256"),
                      ("generation.exitcode", "generation_exitcode_sha256"), ("source_provenance.json", "source_provenance_sha256")):
        files[name] = fingerprint(run / name)
        require(files[name]["sha256"] == completion["verified_artifacts"][key], f"{display} parent artifact receipt mismatch")
    dispatch = read_json(run / "dispatch_audit.json")
    for key, value in {"complete": True, "error": None, "videos_observed": 20,
        "self_attention_calls": 3200, "core_calls": 3200, "selector_calls": 3200,
        "merge_route_head_chunk_calls": 16000, "per_layer_calls": [80] * 40, "declaration": declaration}.items():
        require(dispatch.get(key) == value, f"{display} dispatch not complete/pure: {key}")
    runtime = read_json(run / "runtime.json")
    require(runtime["checkpoint_weights"] == "raw_state_dict" and len(runtime["records"]) == 20,
            f"{display} runtime checkpoint or coverage mismatch")
    require([item["prompt_index"] for item in runtime["records"]] == list(range(20))
            and all(item["attention_calls"] == item["full_attention_calls"] == 160 for item in runtime["records"]),
            f"{display} runtime omitted prompts/layers/steps")
    files.update({name: fingerprint(run / name) for name in ("completion.json", "purity.log", "run_manifest.json")})
    return {"valid": True, "files": files, "declaration": declaration,
            "timing_scope": "generation runtime retained as provenance only; evaluator makes no timing comparison"}


def preflight(args, audit_path):
    """Verify every source and all three manifests before any Torch/LPIPS import."""
    method = selected_candidate(args)
    spec, methods = candidate_spec(method), methods_for(method)
    directories = pinned_directories(args)
    audit_receipt = fingerprint(audit_path)
    require(audit_receipt["sha256"] == AUDIT_SHA, "reference audit is not the independent pinned receipt")
    audit = read_json(audit_path)
    require(audit.get("valid") is True, "reference audit is invalid")
    historical = audit["hellinger_rf10"]
    require(historical["policy"]["router_frame"] == 10 and historical["policy"]["probe_policy"] == "Uniform16, not historical A8U8"
            and historical["policy"]["pure_all_layer_config"] is True, "historical HF identity changed")
    require(Path(audit["references"]["canonical_20260826"]["root"]).resolve() == directories["reference"]
            and (Path(historical["root"]) / "videos").resolve() == directories[methods[1]], "historical audit roots differ")
    result = {"reference_audit": audit_receipt, "directories": {k: str(v) for k, v in directories.items()},
              spec["generation_key"]: validate_candidate_generation(directories[methods[0]], method), "inputs": {}}
    for name, section in (("reference", audit["references"]["canonical_20260826"]), (methods[1], historical)):
        result["inputs"][name] = {
            "manifest": verify_receipt(directories[name] / "prompt_manifest.json", section["manifest"]),
            "videos": verify_inventory(directories[name], section["files"]),
        }
    historical_sources = {}
    for name in ("config", "quality", "runtime", "generation_log"):
        path = Path(historical[name]["path"]).resolve(strict=True)
        require(path.is_relative_to(directories[methods[1]].parent), "historical receipt escapes its source run")
        historical_sources[name] = verify_receipt(path, historical[name])
    result["historical_source_receipts"] = historical_sources
    result["historical_identity"] = {"policy": historical["policy"], "recorded_git_commit": historical["recorded_git_commit"],
                                     "historical_rf10_timing_established": False}
    result["inputs"][methods[0]] = {"manifest": fingerprint(directories[methods[0]] / "prompt_manifest.json"),
                                  "videos": verify_inventory(directories[methods[0]])}
    from analysis import evaluate_video_quality as quality
    from utils.prompt_contract import CANONICAL_PROMPT_SHA256, CANONICAL_FIRST_PROMPT_SHA256, load_output_prompt_manifest
    comparisons = []
    for name in methods:
        pairs, contract = quality.prepare_comparison(directories["reference"], directories[name], expected_videos=20)
        require(contract["prompt_file_sha256"] == CANONICAL_PROMPT_SHA256 and contract["seed"] == 0
                and contract["verified_prompt_indices"] == list(range(20)), "not canonical matched prompt/seed coverage")
        comparisons.append((name, pairs, contract))
    for directory in directories.values():
        manifest = load_output_prompt_manifest(directory)
        require(manifest["selected_prompt_count"] == 20 and manifest["prompts"][0]["sha256"] == CANONICAL_FIRST_PROMPT_SHA256,
                "not the canonical exact20 Tokyo-woman prefix")
    result["all_three_prompt_manifests_verified"] = True
    return result, comparisons


def prepare(args):
    output = Path(args.output).resolve()
    require(not output.exists(), "fresh output stage required; no resume or overwrite")
    for name, expected in METRIC_SOURCE_SHAS.items():
        require(fingerprint(ROOT / name)["sha256"] == expected, "metric or prompt source changed; protocol requires review")
    inputs, _ = preflight(args, args.reference_audit)
    output.mkdir(parents=True)
    files = []
    for name in SOURCES:
        source = (ROOT / name).resolve(strict=True)
        require(source.is_relative_to(ROOT.resolve()), "source escapes repository")
        before = fingerprint(source)
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        require(before["sha256"] == fingerprint(source)["sha256"] == fingerprint(target)["sha256"], "source changed while freezing")
        files.append({"path": name, "bytes": before["bytes"], "sha256": before["sha256"]})
    shutil.copyfile(args.reference_audit, output / "reference_audit.json")
    require(fingerprint(output / "reference_audit.json")["sha256"] == AUDIT_SHA, "audit freeze mismatch")
    # Normalize the audit's own local staging path; byte identity is unchanged.
    inputs["reference_audit"] = fingerprint(output / "reference_audit.json")
    write_json(output / "input_validation.json", inputs, exclusive=True)
    write_json(output / "protocol.json", protocol_for(selected_candidate(args)), exclusive=True)
    provenance = {"files": files, "metric_source_shas": METRIC_SOURCE_SHAS,
                  "protocol": fingerprint(output / "protocol.json"),
                  "input_validation": fingerprint(output / "input_validation.json"),
                  "reference_audit": fingerprint(output / "reference_audit.json")}
    write_json(output / "source_provenance.json", provenance, exclusive=True)
    return provenance


def verify_snapshot(output, candidate_method="mass_rf0"):
    output = Path(output).resolve()
    provenance = read_json(output / "source_provenance.json")
    require(provenance["metric_source_shas"] == METRIC_SOURCE_SHAS, "metric source pins changed")
    require([record["path"] for record in provenance["files"]] == list(SOURCES), "snapshot source whitelist changed")
    for record in provenance["files"]:
        verify_receipt(output / "source" / record["path"], record)
    for name, field in (("protocol.json", "protocol"), ("input_validation.json", "input_validation"),
                        ("reference_audit.json", "reference_audit")):
        verify_receipt(output / name, provenance[field])
    require(read_json(output / "protocol.json") == protocol_for(candidate_method), "evaluation protocol changed")
    return provenance


@contextmanager
def full_resolution_decoder(quality):
    """Observe unchanged decoded frames; no resize, crop, sampling or casting."""
    original = quality.frames
    def checked(path):
        for frame in original(path):
            require(list(frame.shape) == PROTOCOL["frame_shape"] and str(frame.dtype) == "uint8",
                    f"decoded frame is not full-resolution720p RGB uint8: {path}")
            yield frame
    quality.frames = checked
    try:
        yield
    finally:
        quality.frames = original


def validate_video_metric(video, index):
    require(video["video"] == f"{index:03d}-00.mp4" and video["frames"] == 81, "metric frame/prompt coverage differs")
    require(all(isinstance(video[key], (float, int)) and math.isfinite(video[key]) for key in ("psnr", "ssim", "lpips")),
            "nonfinite quality metric; no silent clipping/replacement")


def result_complete(result, candidate_method="mass_rf0"):
    try:
        methods, protocol = methods_for(candidate_method), protocol_for(candidate_method)
    except ValueError:
        return False
    if not (result.get("valid") is True and result.get("complete") is True and result.get("protocol") == protocol
            and set(result.get("methods", {})) == set(methods)
            and result.get("mean_attention_ms") is None and result.get("speedup_vs_sdpa") is None
            and result.get("historical_rf10_timing_established") is False):
        return False
    try:
        for method in methods:
            row = result["methods"][method]
            if row.get("mean_attention_ms") is not None or row.get("speedup_vs_sdpa") is not None:
                return False
            if len(row["videos"]) != 20 or row["summary"]["videos"] != 20 or row["summary"]["frames"] != 1620:
                return False
            for index, video in enumerate(row["videos"]):
                validate_video_metric(video, index)
            for metric in ("psnr", "ssim", "lpips"):
                expected = math.fsum(v[metric] * v["frames"] for v in row["videos"]) / 1620
                if not math.isclose(row["summary"][metric], expected, rel_tol=1e-12, abs_tol=1e-12):
                    return False
        return (result.get("inputs_reverified_at_end") is True and result.get("sources_reverified_at_end") is True
                and result.get("psnr_above_12") is (result["methods"][methods[0]]["summary"]["psnr"] > 12.0))
    except (KeyError, TypeError, ValueError):
        return False


def worker(args):
    output = Path(args.output).resolve()
    candidate = selected_candidate(args)
    methods, protocol = methods_for(candidate), protocol_for(candidate)
    status = 1
    owns_stage = False
    try:
        require(Path(__file__).resolve() == output / "source" / SOURCE, "worker must execute the frozen snapshot")
        startup = read_json(output / "startup.json")
        require(startup["parent_pid"] == os.getppid()
                and os.environ.get("GCA_QUALITY_EVALUATION_PARENT_PID") == str(os.getppid()), "worker lacks its monitoring parent")
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == startup["immediately_before_spawn"]["gpu"]["uuid"],
                "worker GPU differs from clean parent receipt")
        require(not (output / "results.json").exists() and not (output / "evaluation.exitcode").exists(), "refuse worker resume")
        owns_stage = True
        verify_snapshot(output, candidate)
        inputs, comparisons = preflight(args, output / "reference_audit.json")
        require(inputs == read_json(output / "input_validation.json"), "inputs changed after parent preflight")
        result = {"valid": False, "complete": False, "protocol": protocol, "reference": str(REFERENCE),
                  "input_validation": fingerprint(output / "input_validation.json"),
                  "source_provenance": fingerprint(output / "source_provenance.json"),
                  "methods": {name: {"summary": None, "videos": [], "mean_attention_ms": None,
                                     "speedup_vs_sdpa": None, "vbench": None} for name in methods},
                  "mean_attention_ms": None, "speedup_vs_sdpa": None,
                  "historical_rf10_timing_established": False, "psnr_above_12": None,
                  "limitations": ["This is decoded-video fidelity, not a new attention timing experiment.",
                                  f"Historical HF uses RF10 and FP8 merge scores; new {candidate_spec(candidate)['display']} uses RF0 and BF16 merge scores.",
                                  "Both are recomputed against canonical20260826; old HF quality numbers are not reused.",
                                  "SSIM and LPIPS are reported without invented acceptance margins; VBench is unmeasured.",
                                  "Parent completion and clean purity log are required for valid evaluation."]}
        write_json(output / "results.json", result, exclusive=True)
        # All directory/source/manifest/completion checks above precede these
        # imports and model creation. No fallback to a CPU metric run.
        import torch
        import lpips
        from analysis import evaluate_video_quality as quality
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "one parent-selected CUDA device required")
        torch.set_num_threads(4)
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        model = lpips.LPIPS(net="alex").to(device).eval()
        result["environment"] = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0), "lpips": importlib.metadata.version("lpips"),
            "numpy": importlib.metadata.version("numpy"), "scikit_image": importlib.metadata.version("scikit-image"),
            "imageio": importlib.metadata.version("imageio")}
        with full_resolution_decoder(quality):
            for method, pairs, contract in comparisons:
                row = result["methods"][method]
                row["prompt_contract"] = contract
                row["directory"] = inputs["directories"][method]
                for index, (reference, generated) in enumerate(pairs):
                    verify_receipt(reference, inputs["inputs"]["reference"]["videos"][index])
                    verify_receipt(generated, inputs["inputs"][method]["videos"][index])
                    video = quality.evaluate_pair(reference, generated, model,
                        batch_size=PROTOCOL["lpips_batch_size"], device=device, expected_frames=81)
                    validate_video_metric(video, index)
                    row["videos"].append(video)
                    row["summary"] = quality.aggregate(row["videos"])
                    write_json(output / "results.json", result)
                    print(json.dumps({"method": method, **video}, allow_nan=False), flush=True)
        ending, _ = preflight(args, output / "reference_audit.json")
        require(ending == inputs, "input/source artifacts changed during evaluation")
        verify_snapshot(output, candidate)
        result.update(valid=True, complete=True, inputs_reverified_at_end=True, sources_reverified_at_end=True,
                      psnr_above_12=bool(result["methods"][methods[0]]["summary"]["psnr"] > 12.0))
        require(result_complete(result, candidate), "evaluation output failed its terminal numerical/coverage check")
        write_json(output / "results.json", result)
        status = 0
    except BaseException as error:
        if owns_stage and (output / "results.json").exists():
            failed = read_json(output / "results.json")
            failed.update(valid=False, complete=False, error=f"{type(error).__name__}: {error}")
            write_json(output / "results.json", failed)
        raise
    finally:
        if owns_stage:
            (output / "evaluation.exitcode").write_text(str(status) + "\n", encoding="utf-8")


def verify_completion(output, candidate_method="mass_rf0"):
    output = Path(output)
    require((output / "evaluation.exitcode").read_text().strip() == "0", "evaluation worker exit receipt is nonzero")
    require(result_complete(read_json(output / "results.json"), candidate_method), "evaluation results incomplete or internally inconsistent")
    verify_snapshot(output, candidate_method)
    return {name: fingerprint(output / name) for name in ("results.json", "evaluation.exitcode",
            "source_provenance.json", "input_validation.json", "protocol.json")}


def launch(args):
    output = Path(args.output).resolve()
    candidate = selected_candidate(args)
    require(not output.exists(), "fresh output stage required; no resume")
    pinned_directories(args)
    initial = guard.require_clean(args.gpu)
    process, status, failure, foreign, artifacts = None, None, None, [], None
    previous, polls = {}, 0
    started = guard.now()
    def interrupted(number, frame):
        raise KeyboardInterrupt(f"signal {number}")
    try:
        previous = {number: signal.signal(number, interrupted) for number in (signal.SIGINT, signal.SIGTERM)}
        prepare(args)
        verify_snapshot(output, candidate)
        immediate = guard.require_clean(args.gpu, expected_uuid=initial["gpu"]["uuid"])
        write_json(output / "startup.json", {"parent_pid": os.getpid(), "initial": initial,
            "immediately_before_spawn": immediate, "requested_gpu": args.gpu,
            "monitor_interval_seconds": 5, "protocol": protocol_for(candidate)}, exclusive=True)
        command = [sys.executable, "-u", str(output / "source" / SOURCE), "--worker",
            "--candidate-method", candidate, "--candidate-dir", str(candidate_directory(args)),
            "--reference-dir", str(args.reference_dir),
            "--hellinger-dir", str(args.hellinger_dir), "--output", str(output)]
        environment = guard.child_environment(output, initial["gpu"]["uuid"])
        environment["GCA_QUALITY_EVALUATION_PARENT_PID"] = str(os.getpid())
        with (output / "worker.log").open("x") as log, (output / "purity.log").open("x") as purity, \
             (output / "gpu_monitor.jsonl").open("x") as monitor:
            process = subprocess.Popen(command, cwd=output / "source", env=environment, stdout=log, stderr=subprocess.STDOUT)
            write_json(output / "process.json", {"parent_pid": os.getpid(), "worker_pid": process.pid,
                "command": command, "cwd": str(output / "source"),
                "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"]}, exclusive=True)
            try:
                while True:
                    active = guard.gpu_processes(initial["gpu"]["uuid"])
                    polls += 1
                    monitor.write(json.dumps({"utc": guard.now(), "compute_processes": active}) + "\n")
                    monitor.flush()
                    foreign = [entry for entry in active if entry["pid"] != process.pid]
                    require(not foreign, "foreign GPU process detected; stop only own evaluation worker")
                    if process.poll() is not None:
                        break
                    time.sleep(5)
                status = process.wait(timeout=15)
                require(status == 0, f"evaluation worker exited with code{status}")
                artifacts = verify_completion(output, candidate)
            except BaseException as error:
                purity.write(json.dumps({"utc": guard.now(), "foreign": foreign,
                    "guard_failure": f"{type(error).__name__}: {error}"}) + "\n")
                purity.flush()
                raise
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        for number in previous:
            signal.signal(number, signal.SIG_IGN)
        try:
            guard.stop_own_child(process)
            if process is not None:
                status = process.returncode
        except BaseException as error:
            failure = (failure + "; " if failure else "") + f"cleanup: {type(error).__name__}: {error}"
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
        valid = failure is None and status == 0 and artifacts is not None and not foreign
        completion = {"valid": valid, "complete": valid, "started_utc": started, "completed_utc": guard.now(),
            "worker_exit_code": status, "parent_pid": os.getpid(), "worker_pid": process.pid if process else None,
            "requested_gpu": args.gpu, "gpu_uuid": initial["gpu"]["uuid"], "foreign": foreign,
            "failure": failure, "monitor_polls": polls, "monitor_interval_seconds": 5,
            "verified_artifacts": artifacts, "mean_attention_ms": None, "speedup_vs_sdpa": None,
            "historical_rf10_timing_established": False}
        if output.exists():
            if status is not None:
                (output / "worker.exitcode").write_text(str(status) + "\n", encoding="utf-8")
            write_json(output / "completion.json", completion)
    require(valid, f"quality evaluation failed: {failure}; inspect completion/worker log")
    return completion


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--candidate-method", choices=CANDIDATE_METHODS, default="mass_rf0")
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--candidate-dir", type=Path, help="Exact pinned candidate video directory")
    candidate.add_argument("--mass-dir", type=Path, help="Legacy alias, only for mass_rf0")
    for name in ("reference-dir", "hellinger-dir", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--reference-audit", type=Path, default=EXPERIMENTS / "quality_first_reference_audit.json")
    args = parser.parse_args(argv)
    candidate_directory(args)
    return args


def main():
    args = parse_args()
    return worker(args) if args.worker else launch(args)


if __name__ == "__main__":
    main()
