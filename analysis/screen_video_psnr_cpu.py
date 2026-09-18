"""CPU-only, full-resolution PSNR for a fixed canonical prefix; not a final eval."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from evaluate_video_quality import prepare_comparison, paired_frames
from skimage.metrics import peak_signal_noise_ratio


def identity(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--method-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stream-from-run", type=Path,
                        help="Score completed runtime records while this existing generation runs")
    args = parser.parse_args()
    if not 1 <= args.count <= 20 or args.output.exists():
        raise ValueError("require a fixed canonical prefix of 1..20 and a fresh output")
    deadline=time.monotonic()+3600
    def read_json_if_ready(path):
        try: return json.loads(path.read_text())
        except (FileNotFoundError,json.JSONDecodeError): return None
    def wait_for(predicate):
        while not predicate():
            completion=read_json_if_ready(args.stream_from_run/'completion.json')
            if completion and not completion.get('valid'): raise ValueError('generation failed')
            if time.monotonic()>deadline: raise TimeoutError('generation did not become ready within one hour')
            time.sleep(5)
    if args.stream_from_run:
        if args.method_dir.resolve() != (args.stream_from_run/'videos').resolve():
            raise ValueError('stream run and video directory differ')
        wait_for(lambda:(args.method_dir/'prompt_manifest.json').exists())
        from utils.prompt_contract import assert_matching_prompt_manifests
        contract=assert_matching_prompt_manifests(args.reference_dir,args.method_dir,prompt_indices=range(args.count))
        pairs=[(args.reference_dir/f'{i:03d}-00.mp4',args.method_dir/f'{i:03d}-00.mp4') for i in range(args.count)]
    else:
        pairs, contract = prepare_comparison(args.reference_dir, args.method_dir, max_videos=args.count)
    if [a.name for a, _ in pairs] != [f"{i:03d}-00.mp4" for i in range(args.count)]:
        raise ValueError("not the predefined canonical prefix")
    rows = []
    for index,(reference, method) in enumerate(pairs):
        if args.stream_from_run:
            def ready():
                r=read_json_if_ready(args.stream_from_run/'runtime.json')
                return r and method.exists() and any(x.get('prompt_index')==index and x.get('attention_calls')==160 for x in r.get('records',[]))
            wait_for(ready)
        before = [identity(reference), identity(method)]
        values = []
        for a, b in paired_frames(reference, method, expected_frames=81):
            if a.shape != (720, 1280, 3) or a.dtype != np.uint8 or b.dtype != np.uint8:
                raise ValueError("require original full-resolution RGB uint8 frames")
            values.append(float(peak_signal_noise_ratio(a, b, data_range=255)))
        if before != [identity(reference), identity(method)] or not np.isfinite(values).all():
            raise ValueError("input changed or nonfinite PSNR")
        row = {"video": method.name, "frames": len(values), "psnr": float(np.mean(values)),
               "frame_psnr": values, "inputs": before}
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("video", "frames", "psnr")}), flush=True)
        if args.stream_from_run:
            args.output.with_suffix('.progress.json').write_text(json.dumps({'complete':False,'videos':rows},allow_nan=False))
    if args.stream_from_run:
        wait_for(lambda:read_json_if_ready(args.stream_from_run/'completion.json') is not None)
        if not read_json_if_ready(args.stream_from_run/'completion.json').get('valid'):
            raise ValueError('generation did not complete validly')
        prepare_comparison(args.reference_dir,args.method_dir,max_videos=args.count,expected_videos=args.count)
    report = {"scope": "Fixed-prefix CPU screening; not full metric-suite or generation validity",
              "prompt_contract": contract, "videos": rows, "psnr": float(np.mean([r["psnr"] for r in rows])),
              "frames_per_video": 81, "resolution": [720, 1280], "data_range": 255,
              "ssim": None, "lpips": None, "vbench": None, "mean_attention_ms": None,
              "speedup_vs_sdpa": None, "evaluator": identity(Path(__file__)),
              "decoder": identity(Path(__file__).with_name("evaluate_video_quality.py"))}
    if args.stream_from_run:
        report['generation_completion']=identity(args.stream_from_run/'completion.json')
        report['timing_caveat']='CPU PSNR ran concurrently with generation; not a matched speed benchmark'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"videos": len(rows), "mean_psnr": report["psnr"]}), flush=True)


if __name__ == "__main__":
    main()
