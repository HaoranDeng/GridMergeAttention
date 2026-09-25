"""CPU-only checks for strict video pairing; no Torch/LPIPS import is needed."""

import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from analysis import evaluate_video_quality as quality
from utils.prompt_contract import build_prompt_manifest, ensure_output_prompt_manifest


def _directory(root, *, count=2, samples=1, vbench=False):
    prompts = ["Tokyo woman", "Mammoths", "Space traveler"]
    manifest = build_prompt_manifest(
        "MovieGenVideoBench_extended.txt", prompts, selected_count=count,
        seed=0, num_samples=samples, vbench_filenames=vbench, allow_noncanonical=True,
    )
    ensure_output_prompt_manifest(root, manifest)
    for index in range(count):
        (root / f"{index:03d}-00.mp4").touch()
    return root


def _pair(tmp_path, **kwargs):
    return (
        _directory(tmp_path / "reference", **kwargs),
        _directory(tmp_path / "method", **kwargs),
    )


def test_strict_pairing_accepts_exact_names_and_verifies_every_index(tmp_path):
    reference, method = _pair(tmp_path)
    pairs, contract = quality.prepare_comparison(reference, method, expected_videos=2)
    assert [left.name for left, _ in pairs] == ["000-00.mp4", "001-00.mp4"]
    assert contract["verified_prompt_indices"] == [0, 1]


@pytest.mark.parametrize("directory_index", [0, 1])
@pytest.mark.parametrize("extra", [False, True])
def test_strict_pairing_rejects_missing_or_extra_file(tmp_path, directory_index, extra):
    directories = _pair(tmp_path)
    if extra:
        (directories[directory_index] / "002-00.mp4").touch()
    else:
        (directories[directory_index] / "001-00.mp4").unlink()
    with pytest.raises(ValueError, match="Expected exactly 2 canonical paired videos"):
        quality.prepare_comparison(*directories, expected_videos=2)


@pytest.mark.parametrize("kwargs", [{"samples": 2}, {"vbench": True}])
def test_strict_pairing_rejects_unsupported_sample_or_filename_policy(tmp_path, kwargs):
    directories = _pair(tmp_path, **kwargs)
    with pytest.raises(ValueError, match="requires num_samples=1 and vbench_filenames=false"):
        quality.prepare_comparison(*directories, expected_videos=2)


def test_strict_pairing_rejects_truncation_and_invalid_expected_count(tmp_path):
    directories = _pair(tmp_path)
    with pytest.raises(ValueError, match="max_videos must equal expected_videos"):
        quality.prepare_comparison(*directories, max_videos=1, expected_videos=2)
    with pytest.raises(ValueError, match="expected_videos must be positive"):
        quality.prepare_comparison(*directories, expected_videos=0)


def test_strict_pairing_rejects_corrupted_per_index_hash(tmp_path):
    reference, method = _pair(tmp_path)
    path = method / "prompt_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["prompts"][1]["sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="index 1 SHA256 does not match"):
        quality.prepare_comparison(reference, method, expected_videos=2)


def test_legacy_max_videos_preserved_and_empty_pairing_rejected(tmp_path):
    reference, method = _pair(tmp_path)
    pairs, _ = quality.prepare_comparison(reference, method, max_videos=1)
    assert len(pairs) == 1
    with pytest.raises(ValueError, match="No paired videos"):
        quality.prepare_comparison(reference, method, max_videos=0)


def test_all_method_preflights_run_before_loading_metric_dependencies(tmp_path, monkeypatch):
    reference, method = _pair(tmp_path)
    invalid_method = _directory(tmp_path / "invalid_method")
    (invalid_method / "001-00.mp4").unlink()
    args = SimpleNamespace(
        reference_dir=reference, method_dir=[method, invalid_method],
        method_name=["valid", "invalid"], expected_videos=2, expected_frames=81,
        max_videos=None, lpips_batch_size=8, output=tmp_path / "quality.json",
    )
    monkeypatch.setattr(quality, "parse_args", lambda: args)
    original_import = builtins.__import__

    def guarded_import(name, *arguments, **keywords):
        if name in {"torch", "lpips"}:
            raise AssertionError("Metric dependency loaded before all pair preflights")
        return original_import(name, *arguments, **keywords)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ValueError, match="Expected exactly 2 canonical paired videos"):
        quality.main()


def _mock_frames(monkeypatch, reference_count, method_count=None):
    if method_count is None:
        method_count = reference_count
    reference, method = Path("reference.mp4"), Path("method.mp4")
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    counts = {reference: reference_count, method: method_count}
    monkeypatch.setattr(quality, "frames", lambda path: iter([frame] * counts[path]))
    return reference, method


def test_exact_81_frames_are_all_returned(monkeypatch):
    paths = _mock_frames(monkeypatch, 81)
    assert len(list(quality.paired_frames(*paths, expected_frames=81))) == 81


@pytest.mark.parametrize("count", [80, 82])
def test_short_or_long_video_is_rejected_without_truncation(monkeypatch, count):
    paths = _mock_frames(monkeypatch, count)
    with pytest.raises(ValueError, match="Frame count mismatch"):
        list(quality.paired_frames(*paths, expected_frames=81))


def test_unequal_paired_lengths_are_rejected(monkeypatch):
    paths = _mock_frames(monkeypatch, 81, 80)
    with pytest.raises(ValueError, match="zip"):
        list(quality.paired_frames(*paths, expected_frames=81))


def test_empty_video_is_rejected(monkeypatch):
    paths = _mock_frames(monkeypatch, 0)
    with pytest.raises(ValueError, match="No frames decoded"):
        list(quality.paired_frames(*paths))
