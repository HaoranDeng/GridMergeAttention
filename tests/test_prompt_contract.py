import json

import pytest

from utils.prompt_contract import (
    CANONICAL_FIRST_PROMPT_PREFIX,
    assert_matching_prompt_manifests,
    build_prompt_manifest,
    ensure_output_prompt_manifest,
    load_nonempty_prompts,
    validate_canonical_prompt_contract,
)


def _canonical_prompts():
    return load_nonempty_prompts("prompts/MovieGenVideoBench_extended.txt")


def _manifest(prompts, *, seed=0, selected_count=2, num_samples=1):
    return build_prompt_manifest(
        "prompts/MovieGenVideoBench_extended.txt",
        prompts,
        selected_count=selected_count,
        seed=seed,
        num_samples=num_samples,
        vbench_filenames=False,
        allow_noncanonical=False,
    )


def test_canonical_prompt_zero_is_tokyo_woman():
    prompts = _canonical_prompts()
    validate_canonical_prompt_contract(
        "prompts/MovieGenVideoBench_extended.txt", prompts
    )
    assert prompts[0].startswith(CANONICAL_FIRST_PROMPT_PREFIX)


def test_probe_subset_is_rejected():
    prompts = load_nonempty_prompts("prompts/moviegen_probe_20x1.txt")
    with pytest.raises(ValueError, match="Prompt contract check failed"):
        validate_canonical_prompt_contract("prompts/moviegen_probe_20x1.txt", prompts)


def test_manifest_refuses_legacy_videos(tmp_path):
    (tmp_path / "000-00.mp4").write_bytes(b"legacy")
    with pytest.raises(ValueError, match="no prompt_manifest.json"):
        ensure_output_prompt_manifest(tmp_path, _manifest(_canonical_prompts()))


def test_manifest_refuses_seed_or_prompt_mismatch(tmp_path):
    prompts = _canonical_prompts()
    ensure_output_prompt_manifest(tmp_path, _manifest(prompts))
    with pytest.raises(ValueError, match="Prompt manifest mismatch"):
        ensure_output_prompt_manifest(tmp_path, _manifest(prompts, seed=1))

    manifest = json.loads((tmp_path / "prompt_manifest.json").read_text())
    assert manifest["prompts"][0]["text"].startswith(
        CANONICAL_FIRST_PROMPT_PREFIX
    )


def test_quality_manifest_check_accepts_matching_runs(tmp_path):
    prompts = _canonical_prompts()
    reference = tmp_path / "reference"
    method = tmp_path / "method"
    ensure_output_prompt_manifest(reference, _manifest(prompts))
    ensure_output_prompt_manifest(method, _manifest(prompts))
    contract = assert_matching_prompt_manifests(reference, method)
    assert contract["seed"] == 0
    assert contract["verified_prompt_indices"] == [0, 1]


def test_quality_manifest_check_rejects_missing_or_mismatched_runs(tmp_path):
    prompts = _canonical_prompts()
    reference = tmp_path / "reference"
    method = tmp_path / "method"
    ensure_output_prompt_manifest(reference, _manifest(prompts))
    with pytest.raises(FileNotFoundError, match="prompt_manifest.json"):
        assert_matching_prompt_manifests(reference, method)
    ensure_output_prompt_manifest(method, _manifest(prompts, seed=1))
    with pytest.raises(ValueError, match="refusing cross-prompt quality metrics"):
        assert_matching_prompt_manifests(reference, method)


def _matching_pair(tmp_path, *, reference_count=2, method_count=2):
    prompts = _canonical_prompts()
    reference = tmp_path / "reference"
    method = tmp_path / "method"
    ensure_output_prompt_manifest(
        reference, _manifest(prompts, selected_count=reference_count)
    )
    ensure_output_prompt_manifest(method, _manifest(prompts, selected_count=method_count))
    return reference, method


def _corrupt_manifest(directory, mutate):
    path = directory / "prompt_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("field,value", [("sha256", "0" * 64), ("text", "Wrong prompt")])
def test_quality_manifest_rejects_corrupted_prompt_hash_or_text(tmp_path, field, value):
    reference, method = _matching_pair(tmp_path)
    _corrupt_manifest(method, lambda item: item["prompts"][0].update({field: value}))
    with pytest.raises(ValueError, match="SHA256 does not match its text"):
        assert_matching_prompt_manifests(reference, method)


@pytest.mark.parametrize("index", [0, 2, -1, "1"])
def test_quality_manifest_rejects_corrupted_prompt_index(tmp_path, index):
    reference, method = _matching_pair(tmp_path)
    _corrupt_manifest(method, lambda item: item["prompts"][1].update(index=index))
    with pytest.raises(ValueError, match="prompt index|prompt indices"):
        assert_matching_prompt_manifests(reference, method)


def test_quality_manifest_rejects_swapped_index_mapping(tmp_path):
    reference, method = _matching_pair(tmp_path)

    def swap(item):
        item["prompts"][0]["index"], item["prompts"][1]["index"] = 1, 0

    _corrupt_manifest(method, swap)
    with pytest.raises(ValueError, match="mismatch at index 0"):
        assert_matching_prompt_manifests(reference, method)


def test_quality_manifest_accepts_reordered_entries_with_same_index_mapping(tmp_path):
    reference, method = _matching_pair(tmp_path)
    _corrupt_manifest(method, lambda item: item["prompts"].reverse())
    assert assert_matching_prompt_manifests(reference, method)["verified_prompt_indices"] == [0, 1]


def test_quality_manifest_allows_different_counts_for_explicit_intersection(tmp_path):
    reference, method = _matching_pair(tmp_path, reference_count=3, method_count=2)
    contract = assert_matching_prompt_manifests(reference, method, prompt_indices=range(2))
    assert contract["verified_prompt_indices"] == [0, 1]
    # Preserve legacy evaluator --max_videos behavior, which passes no indices.
    assert assert_matching_prompt_manifests(reference, method) == contract
    with pytest.raises(ValueError, match="index 2 is missing"):
        assert_matching_prompt_manifests(reference, method, prompt_indices=[0, 1, 2])


@pytest.mark.parametrize("field,value", [("seed", -1), ("num_samples", 0)])
def test_quality_manifest_rejects_invalid_seed_or_sample_count_in_both_runs(tmp_path, field, value):
    reference, method = _matching_pair(tmp_path)
    for directory in (reference, method):
        _corrupt_manifest(directory, lambda item: item.update({field: value}))
    with pytest.raises(ValueError, match=field):
        assert_matching_prompt_manifests(reference, method)


def test_quality_manifest_rejects_sample_count_mismatch(tmp_path):
    reference, method = _matching_pair(tmp_path)
    _corrupt_manifest(method, lambda item: item.update(num_samples=2))
    with pytest.raises(ValueError, match="num_samples"):
        assert_matching_prompt_manifests(reference, method)


def test_quality_manifest_rejects_inconsistent_selected_count(tmp_path):
    reference, method = _matching_pair(tmp_path)
    _corrupt_manifest(method, lambda item: item.update(selected_prompt_count=20))
    with pytest.raises(ValueError, match="selected_prompt_count"):
        assert_matching_prompt_manifests(reference, method)
