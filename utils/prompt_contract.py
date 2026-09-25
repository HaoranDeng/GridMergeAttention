import hashlib
import json
from collections.abc import Iterable
from pathlib import Path


CANONICAL_PROMPT_FILENAME = "MovieGenVideoBench_extended.txt"
CANONICAL_PROMPT_SHA256 = (
    "50293c2717c8f2df5773154d4680c74e636ecb08eb4fe13cdce7357d0ab9d121"
)
CANONICAL_FIRST_PROMPT_SHA256 = (
    "d0bf2d0fa5fd5f0a08da801d40232bb9e960c84d5d4889c175a0da4e4cc0aa05"
)
CANONICAL_FIRST_PROMPT_PREFIX = "A stylish woman strolls down a bustling Tokyo street"
PROMPT_MANIFEST_FILENAME = "prompt_manifest.json"


def load_nonempty_prompts(path: str | Path) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def normalized_prompt_file_sha256(prompts: list[str]) -> str:
    normalized = "\n".join(prompts) + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_canonical_prompt_contract(
    prompt_path: str | Path,
    prompts: list[str],
    *,
    allow_noncanonical: bool = False,
) -> None:
    if allow_noncanonical:
        return
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_path}")

    file_sha256 = normalized_prompt_file_sha256(prompts)
    first_sha256 = prompt_sha256(prompts[0])
    errors = []
    if Path(prompt_path).name != CANONICAL_PROMPT_FILENAME:
        errors.append(
            f"filename={Path(prompt_path).name!r}, expected={CANONICAL_PROMPT_FILENAME!r}"
        )
    if file_sha256 != CANONICAL_PROMPT_SHA256:
        errors.append(
            f"prompt_sha256={file_sha256}, expected={CANONICAL_PROMPT_SHA256}"
        )
    if first_sha256 != CANONICAL_FIRST_PROMPT_SHA256:
        errors.append(
            f"first_prompt_sha256={first_sha256}, "
            f"expected={CANONICAL_FIRST_PROMPT_SHA256}"
        )
    if not prompts[0].startswith(CANONICAL_FIRST_PROMPT_PREFIX):
        errors.append(
            f"first_prompt={prompts[0]!r}, expected a Tokyo-woman prompt"
        )
    if errors:
        raise ValueError(
            "Prompt contract check failed; refusing to generate a mismatched "
            "video set:\n- " + "\n- ".join(errors)
        )


def build_prompt_manifest(
    prompt_path: str | Path,
    prompts: list[str],
    *,
    selected_count: int,
    seed: int,
    num_samples: int,
    vbench_filenames: bool,
    allow_noncanonical: bool,
) -> dict:
    selected_prompts = prompts if selected_count < 0 else prompts[:selected_count]
    return {
        "schema_version": 1,
        "prompt_path": str(Path(prompt_path).resolve()),
        "prompt_filename": Path(prompt_path).name,
        "prompt_file_sha256": normalized_prompt_file_sha256(prompts),
        "prompt_count": len(prompts),
        "selected_prompt_count": len(selected_prompts),
        "canonical_contract_enforced": not allow_noncanonical,
        "seed": seed,
        "num_samples": num_samples,
        "vbench_filenames": vbench_filenames,
        "prompts": [
            {
                "index": index,
                "sha256": prompt_sha256(prompt),
                "text": prompt,
            }
            for index, prompt in enumerate(selected_prompts)
        ],
    }


def _comparison_payload(manifest: dict) -> dict:
    return {
        key: manifest[key]
        for key in (
            "prompt_file_sha256",
            "selected_prompt_count",
            "seed",
            "num_samples",
            "vbench_filenames",
            "prompts",
        )
    }


def ensure_output_prompt_manifest(output_dir: str | Path, manifest: dict) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / PROMPT_MANIFEST_FILENAME

    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _comparison_payload(existing) != _comparison_payload(manifest):
            raise ValueError(
                f"Prompt manifest mismatch in {output_dir}; use a new output "
                "directory instead of mixing prompt sets or seeds."
            )
        return manifest_path

    existing_videos = sorted(output_dir.glob("*.mp4"))
    if existing_videos:
        raise ValueError(
            f"Refusing to resume {output_dir}: it contains {len(existing_videos)} "
            f"videos but no {PROMPT_MANIFEST_FILENAME}. Backfill and verify the "
            "manifest explicitly before resuming."
        )

    payload = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    try:
        with manifest_path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _comparison_payload(existing) != _comparison_payload(manifest):
            raise ValueError(
                f"Concurrent prompt manifest mismatch in {output_dir}."
            )
    return manifest_path


def load_output_prompt_manifest(output_dir: str | Path) -> dict:
    manifest_path = Path(output_dir) / PROMPT_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}; refuse to compare videos without verified "
            "prompt identity."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _validated_prompt_index(manifest: dict, directory: str | Path) -> dict[int, dict]:
    """Validate the stored per-index identity rather than trusting its file hash."""
    def invalid(reason: str) -> None:
        raise ValueError(f"Invalid prompt manifest in {directory}: {reason}")

    file_hash = manifest.get("prompt_file_sha256")
    if not isinstance(file_hash, str) or len(file_hash) != 64 or any(
        character not in "0123456789abcdef" for character in file_hash
    ):
        invalid("prompt_file_sha256 must be a SHA256 digest")
    seed = manifest.get("seed")
    samples = manifest.get("num_samples")
    if type(seed) is not int or seed < 0:
        invalid("seed must be a nonnegative integer")
    if type(samples) is not int or samples < 1:
        invalid("num_samples must be a positive integer")
    if type(manifest.get("vbench_filenames")) is not bool:
        invalid("vbench_filenames must be a boolean")

    prompts = manifest.get("prompts")
    count = manifest.get("selected_prompt_count")
    if not isinstance(prompts, list) or type(count) is not int or count != len(prompts):
        invalid("selected_prompt_count must match the prompts list length")
    index = {}
    for entry in prompts:
        if not isinstance(entry, dict):
            invalid("every prompt entry must be an object")
        position = entry.get("index")
        if type(position) is not int or position < 0 or position in index:
            invalid(f"invalid or duplicate prompt index {position!r}")
        text = entry.get("text")
        if not isinstance(text, str) or not text.strip():
            invalid(f"prompt index {position} has no text")
        if entry.get("sha256") != prompt_sha256(text):
            invalid(f"prompt index {position} SHA256 does not match its text")
        index[position] = entry
    if set(index) != set(range(count)):
        invalid("prompt indices must cover the zero-based selected prefix")
    return index


def assert_matching_prompt_manifests(
    reference_dir: str | Path,
    method_dir: str | Path,
    *,
    prompt_indices: Iterable[int] | None = None,
) -> dict:
    """Verify metadata and paired per-index text/hash before video comparison.

    Different selected counts are allowed: existing evaluators can restrict
    pairing with ``--max_videos`` and already reject missing paired files.
    By default all shared manifest indices are checked. A caller evaluating
    an explicit intersection can pass its exact ``prompt_indices``; every
    requested index must exist in both manifests. File existence, sample
    filenames, and frame counts remain the evaluator's responsibility.
    """
    reference = load_output_prompt_manifest(reference_dir)
    method = load_output_prompt_manifest(method_dir)
    comparison_keys = (
        "prompt_file_sha256",
        "seed",
        "num_samples",
        "vbench_filenames",
    )
    mismatches = {
        key: {"reference": reference.get(key), "method": method.get(key)}
        for key in comparison_keys
        if reference.get(key) != method.get(key)
    }
    if mismatches:
        raise ValueError(
            "Prompt manifest mismatch; refusing cross-prompt quality metrics: "
            + json.dumps(mismatches, sort_keys=True)
        )
    reference_index = _validated_prompt_index(reference, reference_dir)
    method_index = _validated_prompt_index(method, method_dir)
    if prompt_indices is None:
        paired_indices = sorted(reference_index.keys() & method_index.keys())
    else:
        paired_indices = list(prompt_indices)
        if any(type(index) is not int or index < 0 for index in paired_indices):
            raise ValueError("paired prompt indices must be nonnegative integers")
        if len(set(paired_indices)) != len(paired_indices):
            raise ValueError("paired prompt indices must not contain duplicates")
        paired_indices.sort()
    if not paired_indices:
        raise ValueError("No shared prompt indices; refusing empty quality comparison")
    for index in paired_indices:
        if index not in reference_index or index not in method_index:
            raise ValueError(f"Paired prompt index {index} is missing from a manifest")
        left, right = reference_index[index], method_index[index]
        if left["sha256"] != right["sha256"] or left["text"] != right["text"]:
            raise ValueError(
                f"Prompt manifest mismatch at index {index}; "
                "refusing cross-prompt quality metrics"
            )
    return {
        key: reference[key]
        for key in comparison_keys
    } | {"verified_prompt_indices": paired_indices}
