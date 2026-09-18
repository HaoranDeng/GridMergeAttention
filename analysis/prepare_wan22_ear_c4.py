"""Check the pinned checkpoint and exercise the isolated Wan2.2 environment."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import run_wan22_ear_c4 as run


def verify_model(output):
    url = f"https://huggingface.co/api/models/{run.MODEL_REPO}/revision/{run.MODEL_REVISION}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        info = json.load(response)
    run.require(info["sha"] == run.MODEL_REVISION, "checkpoint revision changed")
    files = [row for row in info["siblings"] if not row["rfilename"].startswith("assets/")
             and row["rfilename"] != ".gitattributes"]
    # Reject partial downloads before doing the full content verification.
    for row in files:
        path = run.MODEL / row["rfilename"]
        run.require(path.is_file() and path.stat().st_size == row["size"], f"download incomplete: {path}")
    receipts = []
    for row in files:
        path = run.MODEL / row["rfilename"]
        expected = row.get("lfs", {}).get("sha256")
        digest = hashlib.sha256() if expected else hashlib.sha1()
        if not expected:
            expected = row["blobId"]
            digest.update(f"blob {row['size']}\0".encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(16 << 20), b""):
                digest.update(chunk)
        run.require(digest.hexdigest() == expected, f"checkpoint hash mismatch: {path}")
        receipts.append(dict(file=row["rfilename"], bytes=row["size"], hash=expected,
                             algorithm="sha256" if row.get("lfs") else "git-blob-sha1"))
        print(f"Verified {row['rfilename']}", flush=True)
    run.write_json(output / "setup/model_verified.json", dict(valid=True, repo=run.MODEL_REPO,
        revision=run.MODEL_REVISION, files=receipts, total_bytes=sum(r["bytes"] for r in receipts), completed_utc=run.now()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify-model", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validate-ear", action="store_true")
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--attempt", default="compatibility")
    args = parser.parse_args()
    if args.verify_model:
        verify_model(args.output)
    if args.validate:
        run.guarded([sys.executable, "-u", str(ROOT / "analysis/run_wan22_ear_c4.py"),
            "--validate", "--output", str(args.output)], args.output / "setup", args.gpu, args.attempt)
    if args.validate_ear:
        run.guarded([sys.executable, "-u", str(ROOT / "analysis/run_wan22_ear_c4.py"),
            "--validate-ear", "--output", str(args.output)], args.output / "setup", args.gpu, args.attempt)


if __name__ == "__main__":
    main()
