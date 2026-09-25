"""Summarize CUDA activity intervals without treating profiler time as a benchmark."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def summarize(path):
    events = json.loads(path.read_text())['traceEvents']
    kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X']
    if not kernels:
        raise ValueError('Trace contains no CUDA kernel intervals')
    by_name = defaultdict(lambda: {'count': 0, 'total_ms': 0, 'streams': set()})
    boundaries = []
    for e in kernels:
        row = by_name[e['name']]
        row['count'] += 1
        row['total_ms'] += e['dur'] / 1000
        row['streams'].add(e.get('args', {}).get('stream'))
        boundaries.extend([(e['ts'], 1), (e['ts'] + e['dur'], -1)])
    active = 0
    previous = min(x[0] for x in boundaries)
    union = overlap = 0
    for timestamp, delta in sorted(boundaries):
        if active:
            union += timestamp - previous
        if active > 1:
            overlap += timestamp - previous
        active += delta
        previous = timestamp
    span = max(x[0] for x in boundaries) - min(x[0] for x in boundaries)
    for row in by_name.values():
        row['streams'] = sorted(row['streams'], key=str)
    return {'trace': str(path), 'scope': 'Instrumented CUDA activity; not headline latency or hardware counters',
            'kernel_span_ms': span / 1000, 'kernel_active_union_ms': union / 1000,
            'kernel_overlap_ms': overlap / 1000, 'gap_ms': (span - union) / 1000,
            'kernels': dict(by_name)}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('trace', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = summarize(a.trace)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
