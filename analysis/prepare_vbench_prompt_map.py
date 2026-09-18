"""Create VBench custom-input prompt maps for matched video directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prompts = [
        line.strip()
        for line in Path(args.prompt_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    videos = sorted(Path(args.video_dir).resolve().glob("*.mp4"))
    mapping = {str(video): prompts[int(video.stem)] for video in videos}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(mapping, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
