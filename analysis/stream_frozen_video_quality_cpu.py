"""Stream CPU fidelity through an unchanged frozen evaluator; publish only audited full20."""
from __future__ import annotations

import argparse
import errno
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

DEADLINE = time.monotonic() + 7200


def retry(operation):
    while True:
        try:
            return operation()
        except (OSError, json.JSONDecodeError) as exc:
            transient = isinstance(exc, json.JSONDecodeError) or exc.errno in {
                errno.EEXIST, errno.ESTALE, errno.ENOENT,
            }
            if not transient or time.monotonic() >= DEADLINE:
                raise
            time.sleep(1)


def read_json(path):
    return retry(lambda: json.loads(Path(path).read_text()))


def digest(path):
    def read():
        result = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1048576), b""):
                result.update(chunk)
        return result.hexdigest()
    return retry(read)


def save(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    retry(lambda: temporary.write_text(json.dumps(payload, indent=2) + "\n"))
    retry(lambda: os.replace(temporary, path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--method", required=True)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in {"", "-1"}
    run, reference = args.run.resolve(), args.reference.resolve()
    output, progress = run / "quality_cpu.json", run / "quality_cpu.progress.json"
    assert not output.exists() and not progress.exists(), "Use fresh quality artifacts"
    expected_evaluator = "ea28e3c629165ef09a123f8145b8701996ea69daed7797bcfc14068910e717bb"
    assert digest(args.evaluator) == expected_evaluator
    spec = importlib.util.spec_from_file_location("frozen_quality", args.evaluator)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    videos_dir = run / "videos"
    contract = retry(lambda: evaluator.assert_matching_prompt_manifests(
        reference, videos_dir, prompt_indices=range(20)))
    for directory in (reference, videos_dir):
        manifest = read_json(directory / "prompt_manifest.json")
        assert manifest["selected_prompt_count"] == 20
        assert manifest["num_samples"] == 1 and manifest["vbench_filenames"] is False
        assert manifest["seed"] == 0
        assert manifest["prompt_file_sha256"] == "50293c2717c8f2df5773154d4680c74e636ecb08eb4fe13cdce7357d0ab9d121"
    import torch
    import lpips
    assert not torch.cuda.is_available()
    assert (Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth").is_file()
    assert (Path(lpips.__file__).parent / "weights/v0.1/alex.pth").is_file()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    save(run / "quality_cpu_process.json", {
        "pid": os.getpid(), "started_unix": time.time(), "device": "cpu",
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "threads": 4,
        "evaluator": str(args.evaluator), "evaluator_sha256": expected_evaluator,
        "driver": str(Path(__file__).resolve()), "driver_sha256": digest(__file__),
        "reference": str(reference), "method": args.method,
        "prompt_contract": contract, "command": sys.argv,
    })
    model = lpips.LPIPS(net="alex").to("cpu").eval()
    result = {
        "reference": str(reference),
        "protocol": {"pairing": "same filename/prompt/seed; framewise comparison",
                     "psnr_data_range": 255, "ssim_channel_axis": 2,
                     "lpips_backbone": "alex", "expected_videos": 20, "expected_frames": 81},
        "methods": {},
    }
    scored, inputs = [], []
    deadline = DEADLINE
    for index in range(20):
        filename = f"{index:03d}-00.mp4"
        while True:
            assert time.monotonic() < deadline, "Generation wait exceeded two hours"
            if (run / "completion.json").exists():
                assert read_json(run / "completion.json")["valid"] is True, "Generation invalid"
            if (run / "runtime.json").exists():
                runtime = read_json(run / "runtime.json")
                rows = [r for r in runtime["records"] if r["output"] == filename]
                if rows and (videos_dir / filename).is_file():
                    assert len(rows) == 1 and rows[0]["prompt_index"] == index
                    assert rows[0]["attention_calls"] == rows[0]["full_attention_calls"] == 160
                    break
            time.sleep(5)
        pair = (reference / filename, videos_dir / filename)
        before = [digest(path) for path in pair]
        row = evaluator.evaluate_pair(*pair, model, 1, torch.device("cpu"), expected_frames=81)
        assert row["frames"] == 81 and all(math.isfinite(row[key]) for key in ("psnr", "ssim", "lpips"))
        assert [digest(path) for path in pair] == before, "Video changed during scoring"
        inputs.append([{ "path": str(path), "sha256": sha} for path, sha in zip(pair, before)])
        scored.append(row)
        result["methods"][args.method] = {"prompt_contract": contract,
            "summary": evaluator.aggregate(scored), "videos": scored}
        save(progress, {"final": False, "scope": "Incomplete streaming quality; not final result", **result})
        print(json.dumps({"final": False, **row}), flush=True)
    while not (run / "completion.json").exists():
        assert time.monotonic() < deadline, "Completion wait exceeded two hours"
        time.sleep(5)
    completion = read_json(run / "completion.json")
    assert completion["valid"] is True and completion["worker_exit_code"] == 0
    file_names = {"dispatch_audit_sha256": "dispatch_audit.json", "runtime_sha256": "runtime.json",
                  "generation_exitcode_sha256": "generation.exitcode",
                  "source_provenance_sha256": "source_provenance.json", "stable_kernel_sha256": "stable_kernel.py",
                  "native_validation_sha256": "native_validation.json"}
    for key, expected in completion["verified_artifacts"].items():
        assert key in file_names, f"Unrecognized completion artifact: {key}"
        assert digest(run / file_names[key]) == expected, key
    runtime, audit = read_json(run / "runtime.json"), read_json(run / "dispatch_audit.json")
    assert len(runtime["records"]) == 20 and runtime["checkpoint_weights"] == "raw_state_dict"
    for index, row in enumerate(runtime["records"]):
        assert row["output"] == f"{index:03d}-00.mp4" and row["prompt_index"] == index
        assert row["attention_calls"] == row["full_attention_calls"] == 160
    assert audit["complete"] is True and audit["self_attention_calls"] == audit["core_calls"] == audit["selector_calls"] == 3200
    assert audit["per_layer_calls"] == [80] * 40
    evaluator.prepare_comparison(reference, videos_dir, expected_videos=20)
    for pair in inputs:
        for item in pair:
            assert digest(item["path"]) == item["sha256"], "Input changed after scoring"
    assert digest(args.evaluator) == expected_evaluator
    result["validation"] = {"valid": True, "generation_completion_sha256": digest(run / "completion.json"),
        "runtime_sha256": digest(run / "runtime.json"), "audit_sha256": digest(run / "dispatch_audit.json"),
        "evaluator_sha256": expected_evaluator, "inputs": inputs,
        "streaming_cpu_only": True, "frame_subsampling": False, "image_resizing": False}
    save(output, result)
    save(run / "quality_cpu_completion.json", {"valid": True, "pid": os.getpid(),
        "completed_unix": time.time(), "quality_sha256": digest(output),
        "summary": evaluator.aggregate(scored), "input_files_rechecked": 40})
    print(json.dumps({"final": True, "summary": evaluator.aggregate(scored), "quality_sha256": digest(output)}), flush=True)


if __name__ == "__main__":
    main()
