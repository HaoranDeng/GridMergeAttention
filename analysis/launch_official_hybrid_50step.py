"""Guard one fresh 20-video Wan2.1 run using the released SVG settings.

SVG/SAP/EAR and explicit gma_c4_mix retain the official hybrid schedule;
the stable Anchor-Fisher RF0 backends remain all-layer. No busy-device queue
or resume. A requested two-device allocation is scoped to this invocation.
The parent reuses the existing clean-H200/account-cap guard, while the child
independently checks canonical prompts before importing the model pipeline.
"""
from __future__ import annotations

import argparse
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
from utils.prompt_contract import (
    assert_matching_prompt_manifests, build_prompt_manifest,
    ensure_output_prompt_manifest, load_nonempty_prompts,
    validate_canonical_prompt_contract,
)

GCA_METHODS = ("gca_fisher_anchor_stable_rf0_all", "gca_fisher_anchor_stable_rf0_hw_swap_all")
METHODS = ("svg", "svg2", "ear", *GCA_METHODS, "gma_c4_mix")
SVG_COMMIT = "f89aedaf169ac2ae5b186bda674e53c3dc08c476"
MODEL = Path("/home/denghaoran/wan_models/Wan2.1-T2V-14B-Diffusers")
SVG_ROOT = Path("/local2/denghaoran/GridMergeAttention_worktrees/svg_official_20260906ch")
REFERENCE = Path("/local2/denghaoran/GridMergeAttention/experiments/wan14b_720p_50step_hellinger_p16_c32g45_a0p2_20260829/dense_reference_exact_prompt_20260816")
CALLS, SPARSE_CALLS, DENSE_CALLS, VIDEOS = 4000, 3120, 880, 20


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def flashinfer_workspace(output):
    # /local2 is shared across hosts with different CUDA toolchains.
    return str(Path(output).resolve() / "flashinfer_cache")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def prompt_manifest():
    path = ROOT / "prompts/MovieGenVideoBench_extended.txt"
    prompts = load_nonempty_prompts(path)
    validate_canonical_prompt_contract(path, prompts)
    return build_prompt_manifest(path, prompts, selected_count=VIDEOS, seed=0,
                                 num_samples=1, vbench_filenames=False, allow_noncanonical=False)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def source_inventory():
    require(git(SVG_ROOT, "rev-parse", "HEAD") == SVG_COMMIT, "wrong official SVG source revision")
    result = {}
    for label, root in (("gca", ROOT), ("svg", SVG_ROOT)):
        require(not git(root, "diff", "HEAD", "--name-only"), "tracked source edits in " + label)
        for name in git(root, "ls-files").splitlines():
            path = root / name
            if path.is_file() and path.suffix in (".py", ".yaml", ".json", ".sh", ".cu", ".cuh", ".h", ".hpp"):
                result[label + "/" + name] = {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
    for path in (SVG_ROOT / "svg/kernels/build").glob("_kernels*.so"):
        result["native/" + path.name] = {"path": str(path.resolve()), "sha256": digest(path), "bytes": path.stat().st_size}
    require(any(name.startswith("native/") for name in result), "compiled SVG RoPE extension is missing")
    return result


def verify_sources(inventory):
    for row in inventory.values():
        path = Path(row["path"])
        require(path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], "source changed: " + str(path))


def reference_preflight(output):
    report = json.loads((REFERENCE / "source_runtime.json").read_text())
    protocol = report["protocol"]
    require(report["backend"] == "dense" and len(report["videos"]) == VIDEOS, "incomplete dense reference")
    for name, expected in (("num_steps", 50), ("num_frames", 81), ("seed", 0),
                           ("guidance_scale", 5.0), ("resolution", "1280x720")):
        require(protocol[name] == expected, "dense reference differs in " + name)
    require(Path(protocol["model"]).resolve() == MODEL.resolve(), "reference checkpoint path mismatch")
    manifest = prompt_manifest()
    records = {row["index"]: row for row in report["videos"]}
    require(set(records) == set(range(VIDEOS)), "reference index coverage mismatch")
    for entry in manifest["prompts"]:
        row = records[entry["index"]]
        require(row["prompt"] == entry["text"] and row["seed"] == 0
                and row["attention_calls"] == CALLS, "reference prompt/seed/call mismatch")
        require((REFERENCE / row["video"]).is_file(), "missing reference video")
    paired = assert_matching_prompt_manifests(REFERENCE, output / "videos", prompt_indices=range(VIDEOS))
    return {"directory": str(REFERENCE), "runtime_sha256": digest(REFERENCE / "source_runtime.json"),
            "manifest_sha256": digest(REFERENCE / "prompt_manifest.json"), "prompt_contract": paired,
            "timing": "historical reference only; no matched SDPA speedup claimed",
            "preprocessing": "historical dense implementation; equivalence to current fast preprocessing not assumed"}


def summarize_density(path, runtime, backend):
    if backend not in ("svg2", "ear"):
        return None
    values = []
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            value = float(row["avg_density"])
            require(math.isfinite(value) and 0 <= value <= 1, "invalid realized density")
            values.append(value)
    require(len(values) == VIDEOS * SPARSE_CALLS, "incomplete sparse-branch density log")
    per_video = []
    for i in range(VIDEOS):
        subtotal = sum(values[i * SPARSE_CALLS:(i + 1) * SPARSE_CALLS])
        per_video.append({"index": i, "sparse_call_density": subtotal / SPARSE_CALLS,
                          "all_call_density": (subtotal + DENSE_CALLS) / CALLS})
    return {"sparse_calls": len(values), "dense_calls": VIDEOS * DENSE_CALLS,
            "all_calls": VIDEOS * CALLS, "sparse_call_density": sum(values) / len(values),
            "all_call_density": (sum(values) + VIDEOS * DENSE_CALLS) / (VIDEOS * CALLS),
            "sparse_call_sparsity": 1 - sum(values) / len(values),
            "all_call_sparsity": 1 - (sum(values) + VIDEOS * DENSE_CALLS) / (VIDEOS * CALLS),
            "definition": "retained token-pair area weighted by actual Q/K cluster populations; compensated EAR blocks are not counted as exact blocks",
            "per_video": per_video, "source_sha256": digest(path)}


def validate_runtime(output, method):
    video_dir = output / "videos"
    runtime = json.loads((video_dir / "runtime.json").read_text())
    require(runtime["backend"] == method and len(runtime["videos"]) == VIDEOS, "wrong or incomplete runtime")
    require({p.name for p in video_dir.glob("*.mp4")} == {f"{i:03d}-00.mp4" for i in range(VIDEOS)}, "incomplete canonical video set")
    require(runtime["protocol"]["num_steps"] == 50 and runtime["protocol"]["num_frames"] == 81,
            "generation schedule changed")
    require(runtime["summary"]["attention_calls"] == VIDEOS * CALLS, "incomplete full attention coverage")
    require(all(row["attention_calls"] == row["full_attention_calls"] == CALLS
                for row in runtime["videos"]), "per-video core/module call count mismatch")
    require(runtime["protocol"]["canonical_names"] is True
            and runtime["protocol"]["original_script_seeding"] is True,
            "canonical naming or per-video original seeding disabled")
    require({row["index"] for row in runtime["videos"]} == set(range(VIDEOS)), "video indices changed")
    for row in runtime["videos"]:
        dispatch = row["attention_dispatch"]
        counts = {name: dispatch[name] for name in ("dense", "sparse", "gca")}
        expected = (dict(dense=DENSE_CALLS, sparse=SPARSE_CALLS, gca=0)
                    if method in ("svg", "svg2", "ear") else
                    dict(dense=DENSE_CALLS, sparse=0, gca=SPARSE_CALLS)
                    if method == "gma_c4_mix" else dict(dense=0, sparse=0, gca=CALLS))
        require(counts == expected, "actual hybrid/all-layer branch counts changed")
        require(len(dispatch["per_layer"]) == 40
                and all(sum(layer.values()) == 100 for layer in dispatch["per_layer"]),
                "per-layer dispatch coverage changed")
        if method in GCA_METHODS:
            require(dispatch["gca_dispatch"] == dict(core_calls=CALLS, selector_calls=CALLS,
                        merge_wrapper_calls=5 * CALLS, stable_kernel_calls=5 * CALLS),
                    "GCA selector/stable merge coverage changed")
    if method == GCA_METHODS[1]:
        validation = json.loads((output / "native_validation.json").read_text())
        require(validation.get("valid") is True, "missing successful transpose native validation")
        layout = runtime["protocol"]["gca_stable"]["layout"]
        require(all(layout.get(key) == value for key, value in {
            "key_axis": "H", "query_axis": "W", "key_group_size": 45,
            "query_group_size": 80, "key_groups_per_frame": 80,
            "query_groups_per_frame": 45, "compressed_slots_per_group": 5,
            "full_route": True}.items()), "spatial transpose layout changed")
    density = summarize_density(video_dir / "density.jsonl", runtime, method)
    if density is not None:
        write_json(output / "density_summary.json", density)
    return runtime


def worker(args):
    require(os.environ.get("GCA_OFFICIAL_50STEP_PARENT") == str(os.getppid()), "live guarded parent required")
    startup = json.loads((args.output / "startup.json").read_text())
    verify_sources(startup["sources"])
    ensure_output_prompt_manifest(args.output / "videos", prompt_manifest())
    require(not list((args.output / "videos").glob("*.mp4")), "fresh videos only; no resume")
    require(not (args.output / "videos/runtime.json").exists(), "fresh runtime only")
    if args.method == GCA_METHODS[1]:
        from attention.spatial_transpose_gca_triton import validate_native
        validation = validate_native(args.output / "native_validation")
        require(validation.get("valid") is True, "transpose native differential check failed")
        write_json(args.output / "native_validation.json", validation)
    from analysis import benchmark_wan14b_720p_methods as benchmark
    sys.argv = [str(ROOT / "analysis/benchmark_wan14b_720p_methods.py"),
                "--backend", args.method, "--model", str(MODEL),
                "--sparse_video_gen_root", str(SVG_ROOT),
                "--prompt_path", str(ROOT / "prompts/MovieGenVideoBench_extended.txt"),
                "--output_dir", str(args.output / "videos"), "--start", "0", "--count", "20",
                "--seed", "0", "--height", "720", "--width", "1280", "--num_frames", "81",
                "--num_steps", "50", "--gma_head_chunk", "8", "--svg_pattern_density", "0.3",
                "--svg_top_p", "0.9", "--svg_min_kc_ratio", "0.10", "--attention_error_calls", "0",
                "--canonical_names", "--original_script_seeding"]
    benchmark.main()
    validate_runtime(args.output, args.method)
    verify_sources(startup["sources"])


def parent(args):
    require(not args.output.exists(), "fresh run directory required")
    initial = guard.require_clean(args.gpu)
    limit = args.max_account_gpus or guard.account_limit()
    require(len(guard.account_devices()) < limit, "account GPU cap has no free slot")
    require((MODEL / "model_index.json").is_file(), "original Wan2.1 model unavailable")
    sources = source_inventory()
    args.output.mkdir(parents=True)
    ensure_output_prompt_manifest(args.output / "videos", prompt_manifest())
    subprocess.run([sys.executable, str(ROOT / "analysis/verify_prompt_contract.py"),
                    "--prompt_path", str(ROOT / "prompts/MovieGenVideoBench_extended.txt"),
                    "--num_videos", "20", "--seed", "0", "--num_samples", "1",
                    "--manifest_out", str(args.output / "prompt_contract.json")],
                   check=True, stdout=subprocess.DEVNULL)
    reference = reference_preflight(args.output)
    for name, row in sources.items():
        target = args.output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["path"], target)
        require(digest(target) == row["sha256"], "frozen source copy mismatch")
    environment = guard.child_environment(args.output, initial["gpu"]["uuid"])
    environment.update(GCA_OFFICIAL_50STEP_PARENT=str(os.getpid()),
                       PYTHONPATH=str(ROOT) + os.pathsep + str(SVG_ROOT),
                       SVG_DENSITY_LOG=str(args.output / "videos/density.jsonl"),
                       FLASHINFER_WORKSPACE_BASE=flashinfer_workspace(args.output))
    environment["LD_LIBRARY_PATH"] += os.pathsep + "/usr/local/cuda-12.9/lib64"
    process, status, failure = None, None, None
    polls, foreign, artifacts, previous = 0, [], None, {}
    started = guard.now()
    def interrupted(number, frame):
        raise KeyboardInterrupt(f"received signal {number}")
    try:
        previous = {number: signal.signal(number, interrupted) for number in (signal.SIGTERM, signal.SIGINT)}
        immediate = guard.require_clean(args.gpu, expected_uuid=initial["gpu"]["uuid"])
        require(len(guard.account_devices()) < limit, "account GPU cap reached during preparation")
        write_json(args.output / "startup.json", dict(method=args.method, source_root=str(ROOT),
            gca_commit=git(ROOT, "rev-parse", "HEAD"), svg_commit=SVG_COMMIT, sources=sources,
            initial=initial, immediate=immediate, account_limit=limit, reference=reference,
            packages={name: importlib.metadata.version(name) for name in ("torch", "triton", "diffusers", "flashinfer-python")},
            flashinfer_workspace=environment["FLASHINFER_WORKSPACE_BASE"],
            settings="official released Wan2.1 scripts, not SVG2 paper-table settings", started_utc=started))
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--method", args.method,
                   "--gpu", str(args.gpu), "--output", str(args.output)]
        with (args.output / "worker.log").open("x") as log, (args.output / "gpu_monitor.jsonl").open("x") as monitor:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            write_json(args.output / "process.json", dict(parent_pid=os.getpid(), worker_pid=process.pid,
                       command=command, spawned_utc=guard.now(), gpu_uuid=initial["gpu"]["uuid"]))
            while True:
                active = guard.gpu_processes(initial["gpu"]["uuid"])
                owned = guard.account_devices()
                polls += 1
                monitor.write(json.dumps(dict(utc=guard.now(), processes=active, account_devices=owned)) + "\n")
                monitor.flush()
                foreign = [row for row in active if row["pid"] != process.pid]
                require(not foreign, "foreign GPU process detected; stopping only own child")
                require(len(owned) <= limit, "account GPU cap exceeded; stopping only own child")
                if process.poll() is not None:
                    break
                time.sleep(5)
            status = process.wait(timeout=15)
            require(status == 0, f"video worker exited {status}")
            validate_runtime(args.output, args.method)
            verify_sources(sources)
            artifacts = {str(path.relative_to(args.output)): digest(path) for path in
                         (args.output / "videos/runtime.json", args.output / "videos/prompt_manifest.json")}
            if args.method == GCA_METHODS[1]:
                artifacts["native_validation.json"] = digest(args.output / "native_validation.json")
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
            failure = (failure + "; " if failure else "") + f"cleanup: {error}"
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
        completion = dict(valid=failure is None and status == 0 and artifacts is not None,
                          method=args.method, started_utc=started, completed_utc=guard.now(),
                          worker_exit_code=status, worker_pid=process.pid if process else None,
                          parent_pid=os.getpid(), failure=failure, foreign=foreign, monitor_polls=polls,
                          gpu_uuid=initial["gpu"]["uuid"], account_limit=limit, verified_artifacts=artifacts)
        write_json(args.output / "completion.json", completion)
    require(completion["valid"], f"invalid run: {failure}")
    print(json.dumps(completion, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output", type=lambda x: Path(x).resolve(), required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--max-account-gpus", type=int, choices=(1, 2),
                        help="Explicit per-run account allocation override; default host policy is unchanged.")
    args = parser.parse_args()
    worker(args) if args.worker else parent(args)


if __name__ == "__main__":
    main()
