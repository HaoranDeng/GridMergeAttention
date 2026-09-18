"""Validate and summarize a completed RF0 sampled-head dense-trajectory screen.

CPU/standard-library only. Reads artifacts without following recorded remote
paths; writes only the explicitly requested, previously nonexistent output.
These output errors are not video PSNR or a full-40-head performance gate.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.prompt_contract import build_prompt_manifest, load_nonempty_prompts, validate_canonical_prompt_contract

SOURCE_NAME = "analysis/screen_native_seed_trajectory.py"
SOURCE = ROOT / SOURCE_NAME
HEADS = [0, 10, 20, 30]
NUMEL = 1 * 21 * 45 * 80 * len(HEADS) * 128
MANIFEST_NAMES = ("launch_prompt_preflight_manifest.json", "prompt_preflight_manifest.json", "prompt_manifest.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, label, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), f"{label}: expected finite numeric value")
    require(value > 0 if positive else value >= 0, f"{label}: invalid sign")
    return float(value)


def check_all_finite(value, label="screen"):
    if isinstance(value, float):
        require(math.isfinite(value), f"{label}: nonfinite value")
    elif isinstance(value, dict):
        for key, item in value.items():
            check_all_finite(item, f"{label}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            check_all_finite(item, f"{label}/{index}")


def expected_modes():
    """Read the frozen producer's literal modes without importing Torch."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == "MODES" for target in node.targets)]
    require(len(nodes) == 1, "Producer must define exactly one literal MODES table")
    modes = ast.literal_eval(nodes[0].value)
    require(set(modes) == {"rf0", "mass_c32", "mass_c16", "partition_c16"}, "Unexpected producer control set")
    require(all("_t1_rf0_u16_" in mode for mode in modes.values()), "Producer controls must all use RF0 Uniform16")
    return modes


def canonical_manifest():
    path = ROOT / "prompts" / "MovieGenVideoBench_extended.txt"
    prompts = load_nonempty_prompts(path)
    validate_canonical_prompt_contract(path, prompts)
    return build_prompt_manifest(path, prompts, selected_count=3, seed=0, num_samples=1,
                                 vbench_filenames=False, allow_noncanonical=False)


def check_manifest(manifest, expected, label):
    require(isinstance(manifest, dict), f"{label}: missing manifest object")
    # Absolute source paths legitimately differ between local and remote copies.
    for key, value in expected.items():
        if key != "prompt_path":
            require(type(manifest.get(key)) is type(value) and manifest[key] == value,
                    f"{label}: canonical {key} mismatch")
    for entry in manifest["prompts"]:
        require(type(entry.get("index")) is int, f"{label}: prompt index must be integer")


def aggregate(records, modes):
    result = {}
    baseline_error = math.fsum(row["methods"]["rf0"]["squared_error_sum"] for row in records)
    for name in modes:
        values = [row["methods"][name] for row in records]
        count = sum(item["numel"] for item in values)
        error = math.fsum(item["squared_error_sum"] for item in values)
        energy = math.fsum(item["reference_squared_sum"] for item in values)
        wins = sum(row["methods"][name]["squared_error_sum"] < row["methods"]["rf0"]["squared_error_sum"] for row in records)
        ties = sum(row["methods"][name]["squared_error_sum"] == row["methods"]["rf0"]["squared_error_sum"] for row in records)
        result[name] = {
            "calls": len(values), "numel": count, "squared_error_sum": error,
            "reference_squared_sum": energy, "mse": error / count,
            "relative_l2": math.sqrt(error / energy),
            "mse_relative_change_vs_rf0": error / baseline_error - 1 if baseline_error else None,
            "wins_vs_rf0": wins, "ties_vs_rf0": ties, "losses_vs_rf0": len(values) - wins - ties,
            "compared_calls": len(values),
            "diagnostic_sampled_head_mean_attention_ms": math.fsum(item["attention_ms"] for item in values) / len(values),
        }
    return result


def validate_and_summarize(screen, manifests, *, exit_code, purity_text):
    check_all_finite(screen)
    require(isinstance(screen, dict) and screen.get("complete") is True, "Screen is not complete")
    require(str(exit_code).strip() == "0", "Run exit_code must be 0")
    require(not purity_text.strip(), "purity.log contains foreign-process evidence")
    modes = expected_modes()
    require(screen.get("methods") == modes, "Screen modes differ from the four frozen RF0 controls")
    require(screen.get("heads") == HEADS and all(type(h) is int for h in screen["heads"]), "Expected heads [0,10,20,30]")
    require(type(screen.get("router_frame")) is int and screen["router_frame"] == 0, "Expected router_frame 0")
    require(screen.get("checkpoint_weights") == "raw_state_dict", "Expected recorded raw_state_dict branch")
    require(screen.get("settings", {}).get("num_prompts") == 3, "Expected three prompts")
    require(screen.get("source_sha256", {}).get(SOURCE_NAME) == hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "Producer source SHA256 differs from the local frozen script")
    finite(screen.get("wall_seconds"), "wall_seconds", positive=True)
    steps = screen.get("actual_warped_timesteps")
    require(isinstance(steps, list) and len(steps) == 4, "Expected four warped timesteps")
    for step in steps:
        finite(step, "warped timestep", positive=True)
    require(all(a > b for a, b in zip(steps, steps[1:])), "Warped timesteps must descend")
    expected = canonical_manifest()
    check_manifest(screen.get("prompt_manifest"), expected, "embedded prompt manifest")
    require(set(manifests) == set(MANIFEST_NAMES), "Expected all three standalone manifests")
    for name in MANIFEST_NAMES:
        check_manifest(manifests[name], expected, name)

    records = screen.get("records")
    require(isinstance(records, list) and len(records) == 480, "Expected exactly 480 call records")
    seen = set()
    for position, row in enumerate(records):
        require(isinstance(row, dict), f"record {position}: expected object")
        fields = ("prompt_index", "call_index", "step_index", "layer_index")
        require(all(type(row.get(key)) is int for key in fields), f"record {position}: indices must be integers")
        prompt, call, step, layer = (row[key] for key in fields)
        require((prompt, call) not in seen, f"Duplicate call {(prompt, call)}")
        seen.add((prompt, call))
        require((prompt, call, step, layer) == (position // 160, position % 160, (position % 160) // 40, position % 40),
                f"record {position}: missing, reordered, or incorrect prompt/step/layer call")
        require(isinstance(row.get("methods"), dict) and set(row["methods"]) == set(modes), f"record {position}: method set mismatch")
        reference_energy = None
        for name in modes:
            item = row["methods"][name]
            label = f"record {position}/{name}"
            require(isinstance(item, dict), f"{label}: expected object")
            require(type(item.get("numel")) is int and item["numel"] == NUMEL, f"{label}: numel mismatch")
            error = finite(item.get("squared_error_sum"), f"{label}/squared_error_sum")
            energy = finite(item.get("reference_squared_sum"), f"{label}/reference_squared_sum", positive=True)
            finite(item.get("attention_ms"), f"{label}/attention_ms")
            if reference_energy is None:
                reference_energy = energy
            require(math.isclose(energy, reference_energy, rel_tol=1e-7), f"{label}: inconsistent dense reference energy")
            for key, length in (("frame_mse", 21), ("head_mse", 4)):
                values = item.get(key)
                require(isinstance(values, list) and len(values) == length, f"{label}/{key}: length mismatch")
                mean = math.fsum(finite(value, f"{label}/{key}") for value in values) / length
                require(math.isclose(mean, error / NUMEL, rel_tol=2e-5, abs_tol=1e-8),
                        f"{label}/{key}: inconsistent with total squared error")

    overall = aggregate(records, modes)
    stored_summary = screen.get("summary")
    require(isinstance(stored_summary, dict) and set(stored_summary) == set(modes), "Producer summary method set mismatch")
    for name, result in overall.items():
        for key in ("calls", "numel", "mse", "relative_l2", "diagnostic_sampled_head_mean_attention_ms"):
            actual = finite(stored_summary[name].get(key), f"producer summary/{name}/{key}")
            require(math.isclose(actual, result[key], rel_tol=1e-10, abs_tol=1e-10), f"Producer summary mismatch: {name}/{key}")
    report = {
        "valid": True, "complete": True, "exit_code": 0, "purity_violations": 0,
        "calls": 480, "prompts": 3, "steps_per_prompt": 4, "layers_per_step": 40,
        "heads": HEADS, "query_frames": 21, "numel_per_call": NUMEL, "router_frame": 0,
        "methods": modes, "checkpoint_weights": screen["checkpoint_weights"],
        "source_sha256": screen["source_sha256"], "actual_warped_timesteps": steps,
        "prompt_file_sha256": expected["prompt_file_sha256"],
        "prompt_hashes": [entry["sha256"] for entry in expected["prompts"]], "seed": 0,
        "aggregation": "MSE = sum(squared_error_sum) / sum(numel); relative L2 = sqrt(sum(error)/sum(reference energy)).",
        "comparison": "Relative change is candidate MSE / RF0 MSE - 1 (negative is better); null only if RF0 error is zero. Wins are strictly lower same-call MSE; no tolerance or statistical significance claim.",
        "scope": "Four sampled heads on an external dense trajectory, all 21 query frames and all 480 calls; candidates never advance the model. Not video PSNR/SSIM/LPIPS.",
        "timing_scope": "Sampled-head diagnostic timings include first-call compilation and are not a full-40-head speed gate or video speedup.",
        "validation_tolerance": {"marginal_mse_relative": 2e-5, "marginal_mse_absolute": 1e-8},
        "overall": overall,
        "per_prompt": {str(prompt): aggregate([row for row in records if row["prompt_index"] == prompt], modes) for prompt in range(3)},
        "per_step": {str(step): aggregate([row for row in records if row["step_index"] == step], modes) for step in range(4)},
    }
    check_all_finite(report, "report")
    return report


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_run(run_dir):
    directory = Path(run_dir)
    report = validate_and_summarize(
        read_json(directory / "screen.json"),
        {name: read_json(directory / name) for name in MANIFEST_NAMES},
        exit_code=(directory / "exit_code").read_text(encoding="utf-8"),
        purity_text=(directory / "purity.log").read_text(encoding="utf-8"),
    )
    report["artifacts"] = {name: str((directory / name).resolve()) for name in ("screen.json", "exit_code", "purity.log", *MANIFEST_NAMES)}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Refusing to overwrite an existing output")
    try:
        report = summarize_run(args.run_dir)
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"valid": True, "calls": report["calls"], "output": str(args.output)}, allow_nan=False))


if __name__ == "__main__":
    main()
