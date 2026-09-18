"""Make verified canonical-name copies of a historical 50-step reference.

Never modifies the historical reference or a completed generation run.  The
reference is matched at the saved protocol level, not bitwise/source-equivalent
to current fast-preprocessing implementations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.prompt_contract import (  # noqa: E402
    CANONICAL_PROMPT_SHA256,
    assert_matching_prompt_manifests,
    load_output_prompt_manifest,
)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--method-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference, run = args.reference.resolve(), args.method_run.resolve()
    method = run / "videos"
    output = args.output.absolute()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse alias output: {output}")
    contract = assert_matching_prompt_manifests(reference, method, prompt_indices=range(20))
    manifest = load_output_prompt_manifest(reference)
    assert contract["prompt_file_sha256"] == CANONICAL_PROMPT_SHA256
    assert contract["seed"] == 0 and contract["num_samples"] == 1
    assert contract["vbench_filenames"] is False
    assert manifest["selected_prompt_count"] == 20
    assert {p.name for p in reference.glob("*.mp4")} == {f"{i:03d}.mp4" for i in range(20)}
    assert {p.name for p in method.glob("*.mp4")} == {f"{i:03d}-00.mp4" for i in range(20)}
    completion = json.loads((run / "completion.json").read_text())
    assert completion["valid"] is True and completion["worker_exit_code"] == 0
    for relative, expected in completion["verified_artifacts"].items():
        assert digest(run / relative) == expected, relative
    original_runtime = reference / "source_runtime.json"
    assert digest(original_runtime) == "b7a44477bd71eec0ef6f6cac6d19a68f33fe776b4dea946670f1aefd4e146c77"
    old = json.loads(original_runtime.read_text())
    new = json.loads((method / "runtime.json").read_text())
    expected_protocol = {
        "model": "/home/denghaoran/wan_models/Wan2.1-T2V-14B-Diffusers",
        "resolution": "1280x720", "num_frames": 81, "num_steps": 50,
        "guidance_scale": 5.0, "seed": 0,
    }
    for key, expected in expected_protocol.items():
        assert old["protocol"][key] == new["protocol"][key] == expected, key
    assert len(old["videos"]) == len(new["videos"]) == 20
    for i, (left, right, prompt) in enumerate(zip(old["videos"], new["videos"], manifest["prompts"], strict=True)):
        assert left["index"] == right["index"] == prompt["index"] == i
        assert left["prompt"] == right["prompt"] == prompt["text"]
        assert left["seed"] == right["seed"] == 0
        assert left["attention_calls"] == right["attention_calls"] == 4000
        assert left["video"] == f"{i:03d}.mp4"
        assert right["video"] == f"{i:03d}-00.mp4"
    # All provenance checks precede any output creation. Copies have independent
    # inodes, so later review tools cannot write through to historical files.
    output.mkdir(parents=True, exist_ok=False)
    for filename in ("prompt_manifest.json", "source_runtime.json"):
        shutil.copyfile(reference / filename, output / filename)
        assert digest(reference / filename) == digest(output / filename)
    mappings = []
    for i in range(20):
        source = reference / f"{i:03d}.mp4"
        destination = output / f"{i:03d}-00.mp4"
        source_sha = digest(source)
        shutil.copyfile(source, destination)
        assert digest(destination) == source_sha
        mappings.append({
            "index": i, "source": str(source), "destination": str(destination),
            "sha256": source_sha, "size_bytes": source.stat().st_size,
            "method_sha256": digest(method / destination.name),
            "prompt_sha256": manifest["prompts"][i]["sha256"],
        })
    receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "valid": True,
        "reference_classification": "historical same-protocol canonical quality reference",
        "limitations": [
            "Not bitwise/source-equivalent to the current fast-preprocessing stack.",
            "Historical timing is not a matched current-run speedup denominator.",
            "Decoded frame counts and shapes must be checked by the evaluator.",
        ],
        "source_reference": str(reference), "method_run": str(run),
        "matched_protocol": expected_protocol, "prompt_contract": contract,
        "source_runtime_sha256": digest(original_runtime),
        "method_runtime_sha256": digest(method / "runtime.json"),
        "method_completion_sha256": digest(run / "completion.json"),
        "mappings": mappings,
    }
    (output / "alias_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"valid": True, "videos": len(mappings), "output": str(output)}))


if __name__ == "__main__":
    main()
