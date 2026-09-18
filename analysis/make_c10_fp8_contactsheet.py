"""Read completed paired videos and make thumbnails for visual review only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=2)
    args = parser.parse_args()
    run = args.run.resolve()
    sys.path.insert(0, str(run / 'source'))
    from analysis.evaluate_video_quality import frames
    from utils.prompt_contract import assert_matching_prompt_manifests
    assert 0 <= args.start < 20 and 1 <= args.count <= 4 and args.start + args.count <= 20
    indices = range(args.start, args.start + args.count)
    contract = assert_matching_prompt_manifests(run / 'control/videos', run / 'candidate/videos', prompt_indices=indices)
    selected = (0, 27, 54, 80)
    canvas = Image.new('RGB', (1280, args.count * 404), 'white')
    draw = ImageDraw.Draw(canvas)
    inputs = []
    for index in indices:
        for offset, variant in enumerate(('control', 'candidate')):
            row = (index-args.start)*2+offset
            runtime = json.loads((run / variant / 'runtime.json').read_text())
            filename = f'{index:03d}-00.mp4'
            records = [r for r in runtime['records'] if r['output'] == filename]
            assert len(records) == 1 and records[0]['prompt_index'] == index
            assert records[0]['attention_calls'] == records[0]['full_attention_calls'] == 160
            path = run / variant / 'videos' / filename
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            seen = []
            for number, frame in enumerate(frames(path)):
                if number in selected:
                    column = selected.index(number)
                    thumbnail = Image.fromarray(frame).resize((320, 180), Image.Resampling.LANCZOS)
                    canvas.paste(thumbnail, (column*320, row*202+22))
                    draw.text((column*320+4, row*202+4), f'{filename} | {"BF16" if offset==0 else "FP8"} | frame {number}', fill='black')
                    seen.append(number)
            assert seen == list(selected) and number == 80
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
            inputs.append({'path': str(path), 'sha256': digest})
    basename = 'first2_contactsheet' if args.start == 0 and args.count == 2 else f'contactsheet_{args.start:02d}_{args.start+args.count-1:02d}'
    output = run / (basename + '.png')
    assert not output.exists()
    canvas.save(output)
    output.with_suffix('.json').write_text(json.dumps({'scope': f'Paired thumbnail review, prompt indices {list(indices)}; metrics use every full-resolution frame',
        'frame_indices': selected, 'inputs': inputs, 'prompt_contract': contract}, indent=2))


if __name__ == '__main__':
    main()
