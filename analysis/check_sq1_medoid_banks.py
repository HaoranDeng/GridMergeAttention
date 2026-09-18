"""Fixed nine-bank CPU pilot: C32 RF0 refinement, all45 full80 diagnostics only."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time

CAPTURE = Path('/local2/denghaoran/GridMergeAttention/experiments/rf0_layer33_dense_capture_20260904n/captures/prompt001_call033_layer33.pt')
OUTPUT = Path('/local2/denghaoran/GridMergeAttention/experiments/sq1_conditional_medoid_cpu_20260906cd')
BANKS = [(j, f, r) for j in (0, 40, 79) for f, r in ((0, 0), (10, 22), (20, 44))]
FRAMES = (0, 10, 20)
SCALE = 128 ** -.5


def require(value, message):
    if not value: raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=lambda p: Path(p).resolve(), required=True)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'explicitly disable all CUDA devices')
    require(os.sched_getaffinity(0) == {120, 121, 122, 123}, 'require fixed CPU affinity120-123')
    require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')), 'require one CPU thread')
    require(not OUTPUT.exists(), 'fresh output only; no resume/overwrite')
    sys.path.insert(0, str(args.source_root))
    import numpy as np
    import torch
    from attention.adaptive_seed_pytorch import multiscale_candidate_table
    from analysis.anchor_fullgrid_seed import anchor_indices
    from analysis.sq1_conditional_medoid_reference import refine_sq1_conditional_medoids, uniform16_sq1_support
    from analysis.anchor_gap_guard_seed import quadrature_partition_gap_numpy
    from analysis.diagnose_sq1_fixed_seed_witness import input_contract, OLD_ROOT, CAPTURE_SHA
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), 'CPU only; CUDA must remain uninitialized')
    helpers = [multiscale_candidate_table, anchor_indices, refine_sq1_conditional_medoids, quadrature_partition_gap_numpy, input_contract]
    sources = [Path(__file__).resolve()]
    for helper in helpers:
        path = Path(inspect.getsourcefile(helper)).resolve()
        require(path.is_relative_to(args.source_root), 'helper imported outside fixed source root')
        sources.append(path)
    hashes = {str(p): digest(p) for p in sources}
    _, row, _ = input_contract(OLD_ROOT)  # Small locked receipts; do not load old Mass seeds/cache.
    require(Path(row['input']['path']) == CAPTURE and CAPTURE.stat().st_size == 309661725, 'wrong single capture')
    before_stat = (CAPTURE.stat().st_size, CAPTURE.stat().st_mtime_ns)
    require(digest(CAPTURE) == CAPTURE_SHA, 'capture SHA mismatch')
    OUTPUT.mkdir()
    (OUTPUT / 'sources').mkdir()
    for number, path in enumerate(sources): shutil.copy2(path, OUTPUT / 'sources' / f'{number}_{path.name}')
    report = dict(complete=False, scope='9 predeclared banks from one prompt1/L33/step0/head30 capture; not global/all-layer/output/video quality or speed',
                  optimization='RF0 C32 targets AND C32 real candidates; U16+SQ positive measures; anchor fixed; one owner-preserving pass',
                  measurement='all45 current-frame targets on full80 keys; FP64 QK from original BF16 operands; rebuild full80 ownership/SQ/conditionals/route independently for each seed set',
                  selector_precision='original BF16 operands cast FP32 for CPU bmm, then FP64 scale/Fisher/GS; not native row parity evidence',
                  source_sha256=hashes, capture=dict(path=str(CAPTURE), sha256=CAPTURE_SHA, bytes=before_stat[0], mtime_ns=before_stat[1]),
                  head=30, stored_head_slot=3, banks=BANKS, frames=FRAMES, records=[])
    def save(): (OUTPUT / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    save()
    started = time.perf_counter()
    saved = torch.load(CAPTURE, map_location='cpu', weights_only=True)
    require(set(saved) == {'q', 'k', 'v', 'dense_output', 'metadata'}, 'unexpected payload')
    require(saved['metadata'] == {k: v for k, v in row['metadata'].items() if k not in ('bytes', 'sha256')}, 'metadata mismatch')
    for name in ('q', 'k', 'v', 'dense_output'):
        require(tuple(saved[name].shape) == (1, 75600, 4, 128) and saved[name].dtype == torch.bfloat16 and saved[name].device.type == 'cpu', 'not original CPU BF16 capture')
    q, k = saved['q'].reshape(21, 45, 80, 4, 128), saved['k'].reshape(21, 45, 80, 4, 128)
    del saved
    table, anchors = multiscale_candidate_table(45, 32, 'cpu', geometry_count=45)
    require(tuple(table.shape) == (45, 32), 'wrong canonical candidate table')
    for column, key_frame, key_row in BANKS:
        rows = table[key_row].long()
        require(int(rows[anchors[key_row]]) == key_row and len(set(rows.tolist())) == 32, 'matched anchor/table mismatch')
        spec = uniform16_sq1_support(column)
        qq = q[0, rows, column, 3].float()
        kk = k[key_frame, key_row, :, 3].float()
        z16 = torch.bmm(qq[None], kk[::5].T[None]).double() * SCALE
        slots = anchor_indices(z16, anchors[key_row:key_row + 1])[0].numpy()
        zp = (torch.bmm(qq[None], kk[spec['probe_columns']].T[None]).double() * SCALE)[0].numpy()
        refined = refine_sq1_conditional_medoids(zp, rows.numpy(), slots, spec['probe_columns'], column)
        steps = [dict(cell=s['cell'], active=s['active'], feasible_count=int(s['feasible'].sum()), accepted=bool(s['accepted']),
                      before_gap=s['before']['mean_gap'], after_gap=s['after']['mean_gap']) for s in refined['steps']]
        record = dict(query_column=column, key_frame=key_frame, key_row=key_row, candidate_rows=rows.tolist(),
                      before_rows=refined['incumbent_rows'].tolist(), after_rows=refined['seed_rows'].tolist(),
                      changed_seed_count=refined['changed_seed_count'], probe_before=refined['initial']['mean_gap'], probe_after=refined['final']['mean_gap'], steps=steps, full80=[])
        for frame in FRAMES:
            z = (q[frame, :, column, 3].double() @ k[key_frame, key_row, :, 3].double().T).numpy() * SCALE
            states = [quadrature_partition_gap_numpy(z, selected, column, measures=np.ones(80), return_details=True)
                      for selected in (refined['incumbent_rows'], refined['seed_rows'])]
            errors = []
            for state in states:
                a = state['reconstruction']; positive = a > 0; terms = np.zeros_like(a)
                terms[positive] = a[positive] * (np.log(a[positive]) - state['log_p'][positive])
                errors.append(float(np.max(np.abs(terms.sum(-1) - state['gap']))))
                require(np.allclose(a.sum(-1), 1, rtol=0, atol=2e-12), 'invalid reconstructed distribution')
            require(max(errors) <= 2e-10, 'direct reverse KL identity mismatch')
            a, b = states
            record['full80'].append(dict(query_frame=frame, before_mean=float(a['gap'].mean()), after_mean=float(b['gap'].mean()),
                delta_mean=float((b['gap'] - a['gap']).mean()), before_gap=a['gap'].tolist(), after_gap=b['gap'].tolist(),
                owner_changes=int(np.count_nonzero(a['owner'] != b['owner'])), direct_kl_max_errors=errors))
        report['records'].append(record); save()
    records = report['records']
    report['summary'] = dict(changed_banks=sum(r['changed_seed_count'] > 0 for r in records), total_banks=9,
        changed_slots=sum(r['changed_seed_count'] for r in records), adjustable_slots=27,
        probe_before_mean=float(np.mean([r['probe_before'] for r in records])), probe_after_mean=float(np.mean([r['probe_after'] for r in records])),
        max_probe_step_increase=max(s['after_gap'] - s['before_gap'] for r in records for s in r['steps']), full80={})
    require(report['summary']['max_probe_step_increase'] <= 2e-10, 'probe coordinate monotonicity failed')
    for frame in FRAMES:
        cases = [x for r in records for x in r['full80'] if x['query_frame'] == frame]
        report['summary']['full80'][str(frame)] = dict(before_mean=float(np.mean([c['before_mean'] for c in cases])),
            after_mean=float(np.mean([c['after_mean'] for c in cases])), delta_mean=float(np.mean([c['delta_mean'] for c in cases])),
            improved_banks=sum(c['delta_mean'] < 0 for c in cases), worsened_banks=sum(c['delta_mean'] > 0 for c in cases),
            changed_owner_count=sum(c['owner_changes'] for c in cases), total_bulk_owners=9 * 79)
    require((CAPTURE.stat().st_size, CAPTURE.stat().st_mtime_ns) == before_stat, 'capture changed during load')
    require({str(p): digest(p) for p in sources} == hashes and not torch.cuda.is_initialized(), 'sources changed or CUDA initialized')
    report.update(complete=True, cpu_diagnostic_seconds=time.perf_counter() - started, cuda_initialized=False)
    save()
    print(json.dumps(report['summary'], indent=2, allow_nan=False))


if __name__ == '__main__': main()
