"""CPU-only artifact validation for a completed canonical 20-video GCA run.

Reads local artifacts even when their provenance paths refer to a remote host.
Frame counts are verified against strict evaluator records, not decoded again.
Never loads a model or invokes a GPU. Only the explicit --output is written.
Run --self-test for in-memory standard-library tests without experiment data.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.prompt_contract import (
    CANONICAL_FIRST_PROMPT_SHA256, CANONICAL_PROMPT_SHA256, _validated_prompt_index,
)

NAMES = [f"{index:03d}-00.mp4" for index in range(20)]
METRICS = ("psnr", "ssim", "lpips")
PROTOCOL = {"psnr_data_range": 255, "ssim_channel_axis": 2,
            "lpips_backbone": "alex", "expected_videos": 20, "expected_frames": 81}
CONTRACT = {"prompt_file_sha256": CANONICAL_PROMPT_SHA256, "seed": 0,
            "num_samples": 1, "vbench_filenames": False}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, label, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), f"{label}: not finite numeric")
    require(not positive or value > 0, f"{label}: must be positive")
    return float(value)


def close(actual, expected, label):
    require(math.isclose(finite(actual, label), expected, rel_tol=1e-9, abs_tol=1e-9),
            f"{label}: recorded {actual!r} != recomputed {expected!r}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def contract_check(contract, label):
    for key, value in CONTRACT.items():
        require(type(contract.get(key)) is type(value) and contract[key] == value,
                f"{label}: {key} mismatch")


def manifest_signature(manifest):
    contract_check(manifest, "prompt manifest")
    index = _validated_prompt_index(manifest, "artifact manifest")
    require(list(sorted(index)) == list(range(20)), "manifest must contain exactly indices 0..19")
    require(index[0]["sha256"] == CANONICAL_FIRST_PROMPT_SHA256, "noncanonical first prompt")
    return [(index[i]["text"], index[i]["sha256"]) for i in range(20)]


def quality_summary(item, label):
    contract_check(item["prompt_contract"], label)
    require(item["prompt_contract"].get("verified_prompt_indices") == list(range(20)),
            f"{label}: all 20 prompt indices must be verified")
    videos = item["videos"]
    require(len(videos) == 20 and sorted(v["video"] for v in videos) == NAMES,
            f"{label}: missing/duplicate/noncanonical quality video records")
    require(all(type(v["frames"]) is int and v["frames"] == 81 for v in videos),
            f"{label}: expected exactly 81 evaluated frames/video")
    summary = {metric: statistics.fmean(finite(v[metric], f"{label}/{v['video']}/{metric}")
                                      for v in videos) for metric in METRICS}
    for key, value in {**summary, "videos": 20, "frames": 1620}.items():
        close(item["summary"][key], value, f"{label}/summary/{key}")
    return summary


def quality_comparison(quality, baseline):
    require(quality["reference"] == baseline["reference"], "quality dense references differ")
    for label, document in (("candidate", quality), ("baseline", baseline)):
        for key, value in PROTOCOL.items():
            require(document["protocol"].get(key) == value, f"{label}: protocol {key} mismatch")
    require(len(quality["methods"]) == 1, "candidate quality.json must contain exactly one method")
    name, item = next(iter(quality["methods"].items()))
    summary = quality_summary(item, name)
    # Validate every recorded method, not only the two chosen comparison rows.
    controls = {key: quality_summary(value, key) for key, value in baseline["methods"].items()}
    require({"mass_span", "rf10_reference_matched"} <= controls.keys(),
            "baseline quality requires mass_span and rf10_reference_matched")
    return {"method": name, "reference": quality["reference"], "summary": summary,
            "videos": 20, "frames": 1620,
            "delta_vs_same_reference": {key: {metric: summary[metric] - controls[key][metric]
                                              for metric in METRICS}
                                        for key in ("mass_span", "rf10_reference_matched")}}


def runtime_summary(runtime, signature):
    require(runtime.get("checkpoint_weights") == "raw_state_dict", "runtime weights must be raw_state_dict")
    require(manifest_signature(runtime["prompt_manifest"]) == signature, "runtime manifest differs")
    records = runtime["records"]
    require(len(records) == 20, "runtime must contain 20 records")
    for index, record in enumerate(records):
        name = PurePosixPath(record["output"].replace("\\", "/")).name
        require(name == NAMES[index] and record["prompt_index"] == index and record["sample_index"] == 0,
                "runtime records must align with canonical files in zero-based order")
        require((record["prompt_text"], record["prompt_sha256"]) == signature[index],
                f"runtime prompt mismatch at {index}")
        for prefix in ("", "full_"):
            calls = record[f"{prefix}attention_calls"]
            require(type(calls) is int and calls == 160, f"{name}: expected 160 {prefix}attention calls")
            total = finite(record[f"{prefix}attention_total_ms"], name, positive=True)
            close(record[f"mean_{prefix}attention_ms"], total / calls, f"{name}/{prefix}mean_ms")
        finite(record["inference_seconds"], name, positive=True)
        require(finite(record["export_seconds"], name) >= 0, f"{name}: negative export time")

    def aggregate(items):
        calls = sum(item["attention_calls"] for item in items)
        full_calls = sum(item["full_attention_calls"] for item in items)
        return {"videos": len(items), "attention_calls": calls, "full_attention_calls": full_calls,
                "mean_attention_ms": sum(r["attention_total_ms"] for r in items) / calls,
                "mean_full_attention_ms": sum(r["full_attention_total_ms"] for r in items) / full_calls,
                "mean_inference_seconds": statistics.fmean(r["inference_seconds"] for r in items),
                "mean_export_seconds": statistics.fmean(r["export_seconds"] for r in items),
                "attention_calls_per_video": calls / len(items)}

    scopes = {"all20": aggregate(records), "steady19": aggregate(records[1:])}
    for scope, prefix in (("all20", ""), ("steady19", "steady_state_")):
        for key in ("videos", "mean_attention_ms", "mean_full_attention_ms",
                    "mean_inference_seconds", "attention_calls_per_video"):
            close(runtime["summary"][prefix + key], scopes[scope][key], f"runtime/summary/{prefix}{key}")
    require(scopes["all20"]["attention_calls"] == scopes["all20"]["full_attention_calls"] == 3200,
            "runtime must total 3200 core and full attention calls")
    return scopes


def benchmark_summary(benchmark):
    require(benchmark["timing_heads"] == list(range(40)), "QKV benchmark must time all 40 heads")
    repeats, iterations = benchmark["settings"]["repeats"], benchmark["settings"]["iterations"]
    require(type(repeats) is int and repeats > 0 and type(iterations) is int and iterations > 0,
            "invalid benchmark repeat/iteration counts")
    require(set(benchmark["records"]) == set(benchmark["settings"]["qkv"])
            and bool(benchmark["records"]), "benchmark capture set incomplete")
    captures, pooled = {}, {}
    for capture, record in benchmark["records"].items():
        timings = record["attention_latency"]
        require({"rf10", "mass_original", "joint_c16", "joint_c32", "sdpa"} <= timings.keys(),
                f"{capture}: missing timed methods")
        means = {}
        for name, item in timings.items():
            samples = item["samples_ms"]
            require(len(samples) == repeats and item["calls_per_sample"] == iterations,
                    f"{capture}/{name}: incomplete timing samples")
            samples = [finite(v, f"{capture}/{name}", positive=True) for v in samples]
            means[name] = statistics.fmean(samples)
            close(item["mean_attention_ms"], means[name], f"{capture}/{name}/mean")
            pooled.setdefault(name, []).extend(samples)
        captures[capture] = {}
        for name, mean in means.items():
            ratio = means["sdpa"] / mean
            close(timings[name]["speedup_vs_sdpa"], ratio, f"{capture}/{name}/speedup")
            captures[capture][name] = {"mean_attention_ms": mean, "speedup_vs_sdpa": ratio}
    means = {name: statistics.fmean(values) for name, values in pooled.items()}
    return {"scope": benchmark["scope"], "captures": captures,
            "pooled_equal_call_count": {name: {"mean_attention_ms": mean,
                                               "speedup_vs_sdpa": means["sdpa"] / mean}
                                         for name, mean in means.items()},
            "not_video_speedup": True,
            "note": "Matched frozen-QKV complete-operator latency only; not a video-generation speedup."}


def clean_stage(directory, exit_name):
    exit_path, purity = directory / exit_name, directory / "purity.log"
    require(exit_path.is_file() and exit_path.read_text().strip() == "0", f"{exit_path}: missing/nonzero exit")
    require(purity.is_file() and purity.stat().st_size == 0, f"{purity}: missing/nonempty purity log")
    return {"exit_code": 0, "purity_log_bytes": 0}


def validate(run_dir, baseline_path, benchmark_path):
    video_dir = run_dir / "videos"
    files = sorted(p.name for p in video_dir.iterdir() if p.suffix.lower() == ".mp4")
    require(files == NAMES, "video directory must contain exactly 000-00..019-00.mp4")
    require(all((video_dir / name).is_file() and (video_dir / name).stat().st_size > 0 for name in NAMES),
            "video files must be nonempty regular files")
    signature = manifest_signature(read_json(video_dir / "prompt_manifest.json"))
    preflights = [run_dir / name for name in ("prompt_preflight_manifest.json", "preflight_prompt_manifest.json")]
    preflights = [path for path in preflights if path.is_file()]
    require(bool(preflights), "missing preflight prompt manifest")
    for path in preflights:
        require(manifest_signature(read_json(path)) == signature, f"{path}: preflight differs")
    manifest = read_json(run_dir / "run_manifest.json")
    for key, value in {"num_videos": 20, "frames": 81, "seed": 0,
                       "attention_calls_per_video": 160, "denoise_steps": 4,
                       "checkpoint_weights": "raw_state_dict"}.items():
        require(manifest.get(key) == value, f"run manifest: {key} mismatch")
    stages = {"generation": clean_stage(run_dir, "generation.exitcode"),
              "quality": clean_stage(run_dir / "quality_evaluation", "exit_code"),
              "qkv_benchmark": clean_stage(benchmark_path.parent, "exit_code")}
    runtime = read_json(run_dir / "runtime.json")
    quality = quality_comparison(read_json(run_dir / "quality.json"), read_json(baseline_path))
    return {"valid": True, "run_dir": str(run_dir.resolve()), "stages": stages,
            "checkpoint_weights": "raw_state_dict", "prompt_contract": CONTRACT,
            "verified_prompt_indices": list(range(20)), "quality": quality,
            "generation_runtime": runtime_summary(runtime, signature),
            "matched_qkv_benchmark": benchmark_summary(read_json(benchmark_path)),
            "limits": ["81-frame counts come from strict evaluator records; MP4s are not decoded again.",
                       "Checkpoint provenance is validated from recorded raw_state_dict labels, not weight bytes.",
                       "Purity is validated from launcher logs; no GPU is contacted.",
                       "Inference timing excludes model loading and MP4 export; export time is separate."]}


class HelperTests(unittest.TestCase):
    def fixture(self):
        import hashlib
        prompts = [{"index": i, "text": f"prompt {i}",
                    "sha256": hashlib.sha256(f"prompt {i}".encode()).hexdigest()} for i in range(20)]
        first = (ROOT / "prompts/MovieGenVideoBench_extended.txt").read_text(encoding="utf-8").splitlines()[0].strip()
        prompts[0] = {"index": 0, "text": first, "sha256": hashlib.sha256(first.encode()).hexdigest()}
        manifest = {**CONTRACT, "selected_prompt_count": 20, "prompts": prompts}
        records = [{"output": NAMES[i], "prompt_index": i, "sample_index": 0,
                    "prompt_text": p["text"], "prompt_sha256": p["sha256"],
                    "attention_calls": 160, "full_attention_calls": 160,
                    "attention_total_ms": 1600., "full_attention_total_ms": 3200.,
                    "mean_attention_ms": 10., "mean_full_attention_ms": 20.,
                    "inference_seconds": 5., "export_seconds": .1} for i, p in enumerate(prompts)]
        summary = {}
        for prefix, count in (("", 20), ("steady_state_", 19)):
            summary.update({prefix + k: v for k, v in {"videos": count, "mean_attention_ms": 10.,
                            "mean_full_attention_ms": 20., "mean_inference_seconds": 5.,
                            "attention_calls_per_video": 160.}.items()})
        runtime = {"checkpoint_weights": "raw_state_dict", "prompt_manifest": manifest,
                   "records": records, "summary": summary}
        item = {"prompt_contract": {**CONTRACT, "verified_prompt_indices": list(range(20))},
                "summary": {"videos": 20, "frames": 1620, "psnr": 12., "ssim": .4, "lpips": .5},
                "videos": [{"video": name, "frames": 81, "psnr": 12., "ssim": .4, "lpips": .5}
                           for name in NAMES]}
        return manifest, runtime, item

    def test_runtime_recomputed(self):
        manifest, runtime, _ = self.fixture()
        result = runtime_summary(runtime, manifest_signature(manifest))
        self.assertEqual(result["all20"]["attention_calls"], 3200)
        self.assertEqual(result["steady19"]["attention_calls"], 3040)

    def test_manifest_tamper(self):
        manifest, _, _ = self.fixture()
        manifest["prompts"][2]["text"] = "changed"
        with self.assertRaises(ValueError):
            manifest_signature(manifest)

    def test_runtime_rejects_corruption(self):
        manifest, original, _ = self.fixture()
        for field, value in (("attention_calls", 159), ("mean_attention_ms", 11.),
                             ("inference_seconds", float("nan")), ("prompt_index", 1)):
            runtime = copy.deepcopy(original)
            runtime["records"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                runtime_summary(runtime, manifest_signature(manifest))
        original["summary"]["steady_state_mean_attention_ms"] = 11.
        with self.assertRaises(ValueError):
            runtime_summary(original, manifest_signature(manifest))

    def test_quality_summary_and_rejections(self):
        _, _, original = self.fixture()
        self.assertEqual(quality_summary(original, "test")["psnr"], 12.)
        for field, value in (("frames", 80), ("video", NAMES[1]), ("lpips", float("inf"))):
            item = copy.deepcopy(original)
            item["videos"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                quality_summary(item, "test")

    def test_same_reference_delta(self):
        _, _, item = self.fixture()
        candidate = {"reference": "/remote/dense/videos", "protocol": PROTOCOL, "methods": {"joint": item}}
        baseline = {**candidate, "methods": {"mass_span": item, "rf10_reference_matched": item}}
        self.assertEqual(quality_comparison(candidate, baseline)["delta_vs_same_reference"]["mass_span"]["psnr"], 0)
        baseline["reference"] = "/remote/other/videos"
        with self.assertRaises(ValueError):
            quality_comparison(candidate, baseline)

    def test_benchmark_ratio(self):
        timings = {name: {"samples_ms": [ms, ms], "mean_attention_ms": ms,
                          "calls_per_sample": 5, "speedup_vs_sdpa": 300 / ms}
                   for name, ms in (("rf10", 130.), ("mass_original", 130.),
                                    ("joint_c16", 125.), ("joint_c32", 145.), ("sdpa", 300.))}
        benchmark = {"timing_heads": list(range(40)), "scope": "complete operator",
                     "settings": {"repeats": 2, "iterations": 5, "qkv": ["capture"]},
                     "records": {"capture": {"attention_latency": timings}}}
        result = benchmark_summary(benchmark)
        self.assertTrue(result["not_video_speedup"])
        self.assertEqual(result["captures"]["capture"]["joint_c16"]["speedup_vs_sdpa"], 2.4)
        timings["joint_c16"]["speedup_vs_sdpa"] = 3.
        with self.assertRaises(ValueError):
            benchmark_summary(benchmark)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("run-dir", "baseline-quality", "benchmark-json", "output"):
        parser.add_argument("--" + option, type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        require(not any((args.run_dir, args.baseline_quality, args.benchmark_json, args.output)),
                "--self-test does not accept artifact/output paths")
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(HelperTests))
        return 0 if result.wasSuccessful() else 1
    if not all((args.run_dir, args.baseline_quality, args.benchmark_json, args.output)):
        parser.error("--run-dir, --baseline-quality, --benchmark-json and --output are required")
    require(not args.output.exists(), f"Refusing to overwrite {args.output}")
    try:
        report = validate(args.run_dir, args.baseline_quality, args.benchmark_json)
    except (ValueError, KeyError, TypeError, OSError) as error:
        report = {"valid": False, "error": str(error), "run_dir": str(args.run_dir.resolve())}
    report["inputs"] = {key: str(value.resolve()) for key, value in {
        "baseline_quality": args.baseline_quality, "benchmark_json": args.benchmark_json}.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"valid": report["valid"], "output": str(args.output), "error": report.get("error")}))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
