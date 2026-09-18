"""Run four fresh official-schedule comparisons on two explicitly allocated H200s.

Each lane checks its device before each generation/evaluation process; it never
waits for a busy GPU. A foreign process invalidates the active stage. The two
lanes run independently so an evaluation failure cannot discard other videos.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import launch_official_hybrid_50step as run
from analysis import launch_quality_first_videos as guard

LANES = (("ear", "svg2"), ("gma_c4_mix", "svg"))
DIMENSIONS = ("subject_consistency", "background_consistency", "motion_smoothness",
              "dynamic_degree", "aesthetic_quality", "imaging_quality")
VBENCH = Path("/home/denghaoran/workspace/VBench")
CACHE = Path("/local2/denghaoran/GridMergeAttention/vbench_home/.cache")


def guarded_evaluation(command, output, gpu, label):
    state = guard.require_clean(gpu)
    run.require(len(guard.account_devices()) < 2, "two-device allocation exhausted")
    environment = guard.child_environment(output, state["gpu"]["uuid"])
    environment.update(PYTHONPATH=str(ROOT), VBENCH_CACHE_DIR=str(CACHE / "vbench"),
                       XDG_CACHE_HOME=str(CACHE), TORCH_HOME=str(CACHE / "torch"),
                       HF_HOME=str(CACHE / "huggingface"))
    child, failure, foreign, status = None, None, [], None
    started = guard.now()
    try:
        with (output / (label + ".log")).open("x") as log, (output / (label + "_gpu.jsonl")).open("x") as audit:
            guard.require_clean(gpu, expected_uuid=state["gpu"]["uuid"])
            child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                     stdout=log, stderr=subprocess.STDOUT)
            while True:
                active = guard.gpu_processes(state["gpu"]["uuid"])
                foreign = [row for row in active if row["pid"] != child.pid]
                audit.write(json.dumps(dict(utc=guard.now(), worker_pid=child.pid, processes=active)) + "\n")
                audit.flush()
                run.require(not foreign, "foreign GPU process; active evaluation invalidated")
                run.require(len(guard.account_devices()) <= 2, "two-device allocation exceeded")
                if child.poll() is not None:
                    break
                time.sleep(5)
            status = child.wait()
            run.require(status == 0, f"evaluation exited {status}")
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        guard.stop_own_child(child)
    receipt = dict(valid=failure is None and status == 0, failure=failure, foreign=foreign,
                   started_utc=started, completed_utc=guard.now(), command=command,
                   gpu_uuid=state["gpu"]["uuid"], exit_code=status)
    run.write_json(output / (label + "_completion.json"), receipt)
    run.require(receipt["valid"], failure)


def evaluate(output, method, gpu):
    results = output / "results"
    results.mkdir()
    failures = {}
    quality = [sys.executable, "-u", str(ROOT / "analysis/evaluate_video_quality.py"),
               "--reference_dir", str(run.REFERENCE), "--method_dir", str(output / "videos"),
               "--method_name", method, "--output", str(results / "quality.json"),
               "--expected_videos", "20", "--expected_frames", "81", "--lpips_batch_size", "8"]
    try:
        guarded_evaluation(quality, results, gpu, "quality")
    except Exception as error:
        failures["quality"] = str(error)
    try:
        vbench = results / "vbench"
        vbench.mkdir()
        subprocess.run([sys.executable, str(ROOT / "analysis/prepare_vbench_prompt_map.py"),
                        "--prompt_path", str(ROOT / "prompts/MovieGenVideoBench_extended.txt"),
                        "--video_dir", str(output / "videos"), "--output", str(vbench / "prompts.json")], check=True)
        command = [sys.executable, "-u", str(VBENCH / "evaluate.py"),
                   "--videos_path", str(output / "videos"), "--dimension", *DIMENSIONS,
                   "--mode", "custom_input", "--load_ckpt_from_local", "True",
                   "--prompt_file", str(vbench / "prompts.json"), "--output_path", str(vbench)]
        guarded_evaluation(command, results, gpu, "vbench")
    except Exception as error:
        failures["vbench"] = str(error)
    run.write_json(results / "evaluation_status.json", dict(valid=not failures, failures=failures,
        vbench_scope=list(DIMENSIONS), vbench_official_total_score=False))
    return failures


def lane(args):
    gpu = args.gpus[args.lane]
    methods = LANES[args.lane]
    lock_path = Path("/local2/denghaoran/GridMergeAttention/gpu_locks") / f"h200_gpu{gpu}.lock"
    lock_path.parent.mkdir(exist_ok=True, parents=True)
    results = {}
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for method in methods:
            output = args.output / method
            command = [sys.executable, "-u", str(ROOT / "analysis/launch_official_hybrid_50step.py"),
                       "--gpu", str(gpu), "--method", method, "--output", str(output),
                       "--max-account-gpus", "2"]
            guard.require_clean(gpu)
            with (args.output / (method + "_launcher.log")).open("x") as log:
                generation = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            entry = dict(generation_exit=generation.returncode, method=method, gpu=gpu)
            results[method] = entry
            if generation.returncode == 0:
                entry["evaluation_failures"] = evaluate(output, method, gpu)
            run.write_json(args.output / f"lane_{args.lane}_status.json", results)
            # Fail closed if the GPU ceased to be exclusive. No busy-device wait.
            guard.require_clean(gpu)
    run.write_json(args.output / f"lane_{args.lane}_completion.json", dict(
        completed_utc=guard.now(), methods=results,
        valid=all(r["generation_exit"] == 0 and not r.get("evaluation_failures") for r in results.values())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs=2, type=int, required=True)
    parser.add_argument("--output", type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--lane", type=int, choices=(0, 1))
    args = parser.parse_args()
    run.require(len(set(args.gpus)) == 2, "two different physical GPUs required")
    if args.lane is not None:
        lane(args)
        return
    run.require(not args.output.exists(), "fresh comparison output required")
    initial = [guard.require_clean(gpu) for gpu in args.gpus]
    run.require(len(guard.account_devices()) == 0, "both account allocation slots must be available")
    run.source_inventory()
    args.output.mkdir(parents=True)
    run.write_json(args.output / "comparison.json", dict(
        created_utc=guard.now(), source_commit=run.git(ROOT, "rev-parse", "HEAD"),
        svg_commit=run.SVG_COMMIT, initial_gpus=initial, lanes=LANES, account_gpu_limit=2,
        user_override="2026-09-08: four hybrid methods on two GPUs",
        generation=dict(model=str(run.MODEL), steps=50, resolution="1280x720", frames=81,
            videos_per_method=20, seed=0, cfg=5.0, dense_calls=880, replacement_calls=3120,
            gma="standard Hard C4; entropy bias; four fixed seed queries; full compressed attention"),
        quality_reference=str(run.REFERENCE),
        quality_reference_scope="historical same protocol; old/current preprocessing equivalence not established",
        speedup_vs_sdpa=None, speedup_status="no matched current SDPA stream measured",
        vbench_dimensions=list(DIMENSIONS), vbench_official_total_score=False))
    children = []
    for index in range(2):
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--gpus", *map(str, args.gpus),
                   "--output", str(args.output), "--lane", str(index)]
        with (args.output / f"lane_{index}.log").open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        children.append(child)
    run.write_json(args.output / "processes.json", dict(parent_pid=os.getpid(),
        lanes=[dict(lane=i, pid=child.pid, gpu=args.gpus[i], methods=LANES[i]) for i, child in enumerate(children)]))
    codes = [child.wait() for child in children]
    run.write_json(args.output / "comparison_completion.json", dict(
        completed_utc=guard.now(), lane_exit_codes=codes,
        valid=all(code == 0 and json.loads((args.output / f"lane_{i}_completion.json").read_text())["valid"]
                  for i, code in enumerate(codes))))


if __name__ == "__main__":
    main()
