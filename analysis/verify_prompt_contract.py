import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.prompt_contract import (
    build_prompt_manifest,
    load_nonempty_prompts,
    validate_canonical_prompt_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and record the canonical GMA generation prompt contract."
    )
    parser.add_argument(
        "--prompt_path",
        default="prompts/MovieGenVideoBench_extended.txt",
    )
    parser.add_argument("--num_videos", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--manifest_out")
    args = parser.parse_args()

    prompts = load_nonempty_prompts(args.prompt_path)
    validate_canonical_prompt_contract(args.prompt_path, prompts)
    manifest = build_prompt_manifest(
        args.prompt_path,
        prompts,
        selected_count=args.num_videos,
        seed=args.seed,
        num_samples=args.num_samples,
        vbench_filenames=False,
        allow_noncanonical=False,
    )
    if args.manifest_out:
        output = Path(args.manifest_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
