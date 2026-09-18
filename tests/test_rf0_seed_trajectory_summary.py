"""Synthetic-only CPU checks; no models, Torch, CUDA, or experiment writes."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "summarize_rf0_seed_trajectory.py"
spec = importlib.util.spec_from_file_location("rf0_trajectory_summary", SCRIPT)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


@pytest.fixture
def fixture():
    modes = summary.expected_modes()
    manifest = summary.canonical_manifest()
    rows = []
    for prompt in range(3):
        for call in range(160):
            base = 1 + prompt + (call // 40) / 10 + (call % 40) / 1000
            row = {"prompt_index": prompt, "call_index": call, "step_index": call // 40,
                   "layer_index": call % 40, "methods": {}}
            for name, factor in zip(modes, [1, .9, 1.2, .8 if call % 2 == 0 else 1.2]):
                mse = base * factor
                row["methods"][name] = {"numel": summary.NUMEL, "squared_error_sum": mse * summary.NUMEL,
                    "reference_squared_sum": 4 * summary.NUMEL, "frame_mse": [mse] * 21,
                    "head_mse": [mse] * 4, "attention_ms": 5.0}
            rows.append(row)
    data = {"complete": True, "methods": modes, "heads": summary.HEADS.copy(), "router_frame": 0,
            "settings": {"num_prompts": 3}, "checkpoint_weights": "raw_state_dict",
            "source_sha256": {summary.SOURCE_NAME: hashlib.sha256(summary.SOURCE.read_bytes()).hexdigest()},
            "wall_seconds": 100, "actual_warped_timesteps": [1000, 900, 800, 600],
            "prompt_manifest": manifest, "records": rows, "summary": {}}
    for name in modes:
        values = [row["methods"][name] for row in rows]
        count = sum(item["numel"] for item in values)
        error = sum(item["squared_error_sum"] for item in values)
        energy = sum(item["reference_squared_sum"] for item in values)
        data["summary"][name] = {"calls": 480, "numel": count, "mse": error / count,
            "relative_l2": (error / energy) ** .5, "diagnostic_sampled_head_mean_attention_ms": 5.0}
    manifests = {name: copy.deepcopy(manifest) for name in summary.MANIFEST_NAMES}
    return data, manifests


def validate(fixture, **kwargs):
    return summary.validate_and_summarize(*fixture, exit_code=kwargs.get("exit_code", "0\n"), purity_text=kwargs.get("purity_text", ""))


def test_weighted_aggregation_counts_and_scope(fixture):
    report = validate(fixture)
    assert report["valid"] and report["calls"] == 480
    total = report["overall"]
    assert total["rf0"]["mse"] == pytest.approx(2.1695)
    assert total["mass_c32"]["mse_relative_change_vs_rf0"] == pytest.approx(-.1)
    assert total["mass_c32"]["wins_vs_rf0"] == 480
    assert total["mass_c16"]["losses_vs_rf0"] == 480
    assert total["partition_c16"]["wins_vs_rf0"] == 240
    assert total["rf0"]["ties_vs_rf0"] == 480
    assert all(group["rf0"]["calls"] == 160 for group in report["per_prompt"].values())
    assert all(group["rf0"]["calls"] == 120 for group in report["per_step"].values())
    assert "Not video PSNR" in report["scope"]
    assert "not a full-40-head speed gate" in report["timing_scope"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(complete=False),
    lambda d: d["records"].pop(),
    lambda d: d["records"].__setitem__(479, copy.deepcopy(d["records"][0])),
    lambda d: d["records"][40].update(layer_index=1),
    lambda d: d["records"][0].update(prompt_index=False),
    lambda d: d.update(heads=[0, 10, 20, 31]),
    lambda d: d["methods"].update(rf0=d["methods"]["rf0"].replace("rf0", "rf10")),
    lambda d: d["records"][0]["methods"]["mass_c32"].update(numel=1),
    lambda d: d["records"][0]["methods"]["mass_c32"].update(frame_mse=[1] * 20),
    lambda d: d["records"][0]["methods"]["mass_c32"].update(head_mse=[1] * 3),
    lambda d: d["records"][0]["methods"]["mass_c32"].update(squared_error_sum=float("nan")),
    lambda d: d["records"][0]["methods"]["mass_c32"].update(attention_ms=-1),
    lambda d: d["records"][0]["methods"]["mass_c32"].update(reference_squared_sum=1),
    lambda d: d["records"][0]["methods"]["mass_c32"].update(head_mse=[100] * 4),
    lambda d: d["prompt_manifest"].update(seed=1),
    lambda d: d["prompt_manifest"]["prompts"][2].update(sha256="a" * 64),
    lambda d: d["source_sha256"].update({summary.SOURCE_NAME: "a" * 64}),
    lambda d: d["summary"]["rf0"].update(mse=99),
    lambda d: d.update(actual_warped_timesteps=[1000, 750, 500]),
])
def test_rejects_incomplete_or_inconsistent_records(fixture, mutation):
    mutation(fixture[0])
    with pytest.raises(ValueError):
        validate(fixture)


def test_external_manifest_is_checked_but_remote_path_is_not_followed(fixture):
    fixture[1][summary.MANIFEST_NAMES[0]]["prompt_path"] = "/remote/no-access/prompts.txt"
    assert validate(fixture)["valid"]
    fixture[1][summary.MANIFEST_NAMES[0]]["prompts"][1]["text"] = "wrong prompt"
    with pytest.raises(ValueError, match="canonical prompts mismatch"):
        validate(fixture)


@pytest.mark.parametrize("kwargs", [{"exit_code": "1"}, {"purity_text": "foreign pid 999\n"}])
def test_rejects_unclean_or_failed_run(fixture, kwargs):
    with pytest.raises(ValueError):
        validate(fixture, **kwargs)


def write_fixture(directory, fixture):
    data, manifests = fixture
    (directory / "screen.json").write_text(json.dumps(data), encoding="utf-8")
    for name, manifest in manifests.items():
        (directory / name).write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "exit_code").write_text("0\n", encoding="utf-8")
    (directory / "purity.log").write_text("", encoding="utf-8")


def test_cli_local_paths_and_no_overwrite(tmp_path, fixture):
    write_fixture(tmp_path, fixture)
    output = tmp_path / "validated_summary.json"
    args = ["--run-dir", str(tmp_path), "--output", str(output)]
    summary.main(args)
    original = output.read_bytes()
    assert json.loads(original)["valid"]
    with pytest.raises(SystemExit):
        summary.main(args)
    assert output.read_bytes() == original


def test_missing_completion_artifact_writes_no_output(tmp_path, fixture):
    write_fixture(tmp_path, fixture)
    (tmp_path / "exit_code").unlink()
    output = tmp_path / "validated_summary.json"
    with pytest.raises(SystemExit):
        summary.main(["--run-dir", str(tmp_path), "--output", str(output)])
    assert not output.exists()
