"""Verify retained source bytes without importing PyTorch or allocating a GPU."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("c4", "c8", "c8_r2", "c10", "fp8/c10")


def verify(package_file):
    directory = Path(package_file).resolve().parent
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = directory / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Retained kernel source changed: {path}")
    for name, expected in manifest["shared_dependencies"].items():
        path = ROOT / name
        # Shared repository Python files use universal-newline text hashes.
        actual = hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Retained kernel dependency changed: {path}")
    return manifest


if __name__ == "__main__":
    for variant in VARIANTS:
        manifest = verify(Path(__file__).parent / variant / "__init__.py")
        print(f"{variant}: verified {len(manifest['files'])} files")
