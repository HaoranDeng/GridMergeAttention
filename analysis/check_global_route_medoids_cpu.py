"""Fixed six-case, RF0-only global/local frozen-beta medoid CPU diagnostic.

Preparation is not permission to execute this pilot. No native/GPU operation,
video-quality result, or latency claim is produced. Full80 measurements use
original current-frame Q and all945 original K/V banks, independently of the
positive-measure probe objective used for selection. No other capture is read.
"""

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
OUTPUT = Path('/local2/denghaoran/GridMergeAttention/experiments/sq1_global_route_medoid_cpu_20260906cf')
CASES = tuple((head, column) for head in (0, 30) for column in (0, 40, 41))
FRAMES = (0, 10, 20)
METHODS = ('anchor', 'global', 'bank_local')
SCALE = 128 ** -.5


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def full80_state(z, values, selected, singleton, dense_log_z, dense_output):
    """Independent ideal FP64 global reconstruction, not probe-helper reuse.

    z[b,q,k] includes ALL945 banks, ALL45 target rows and ALL80 real keys.
    Ownership compares full80 normalized seed profiles before SQ is peeled.
    Every structurally nonempty bulk cell is normalized directly; there is no
    cancellation-prone 1-alpha normalization, BF16 merge, or residual cutoff.
    """
    import numpy as np

    require(z.shape == (945, 45, 80) and values.shape == (945, 80, 128), 'full80 geometry changed')
    require(z.dtype == values.dtype == np.float64 and np.isfinite(z).all()
            and np.isfinite(values).all(), 'finite original-operand FP64 measurement required')
    require(selected.shape == (945, 4) and selected.dtype.kind in 'iu'
            and np.all((selected >= 0) & (selected < 45))
            and all(len(np.unique(row)) == 4 for row in selected), 'four distinct real seed rows required')
    seed_logits = z[np.arange(945)[:, None], selected]
    centered = seed_logits - seed_logits.max(-1, keepdims=True)
    log_p = centered - np.logaddexp.reduce(centered, axis=-1, keepdims=True)
    owners_before_sq = log_p.argmax(1)
    owners = owners_before_sq.copy()
    owners[:, singleton] = 4
    mask = owners[:, None, :] == np.arange(4)[None, :, None]
    valid = mask.any(-1)
    maximum = np.where(mask, seed_logits, -np.inf).max(-1)
    centered = seed_logits - np.where(valid, maximum, 0)[..., None]
    exponent = np.exp(np.where(mask, centered, -np.inf))
    normalizer = np.where(valid, exponent.sum(-1), 1)
    weights = exponent / normalizer[..., None]
    log_weights = np.where(mask, centered - np.log(normalizer)[..., None], 0)
    entropy = -(weights * log_weights).sum(-1)
    # w dot z + H(w) is exactly q dot merged_k + H(w) in ideal arithmetic.
    bulk = np.einsum('bqp,bcp->bqc', z, weights, optimize=True) + entropy[:, None, :]
    bulk = np.where(valid[:, None, :], bulk, -np.inf)
    components = np.concatenate((bulk, z[..., singleton:singleton + 1]), axis=-1)
    represented = np.logaddexp.reduce(np.logaddexp.reduce(components, axis=-1), axis=0)
    beta = np.exp(components - represented[None, :, None])
    merged_v = np.concatenate((weights @ values, values[:, singleton:singleton + 1]), axis=1)
    output = beta.transpose(1, 0, 2).reshape(45, -1) @ merged_v.reshape(-1, 128)
    gap = dense_log_z - represented
    # Direct reverse KL checks the reconstructed distribution against the
    # global dense distribution, including each bank's absolute probability.
    reconstruction = beta[..., :4] @ weights
    reconstruction[..., singleton] += beta[..., 4]
    log_a = np.zeros_like(reconstruction)
    np.log(reconstruction, out=log_a, where=reconstruction > 0)
    direct_kl = (reconstruction * (log_a - z + dense_log_z[None, :, None])).sum(axis=(0, 2))
    mass_error = float(np.max(np.abs(reconstruction.sum(axis=(0, 2)) - 1)))
    kl_error = float(np.max(np.abs(direct_kl - gap)))
    require(mass_error <= 2e-11 and kl_error <= 2e-9, 'independent full80 global reverse-KL identity failed')
    require(np.isfinite(output).all() and np.isfinite(gap).all(), 'nonfinite measurement')
    return dict(gap=gap, output=output, per_row_mse=np.square(output - dense_output).mean(-1),
                owners=owners, owners_before_sq=owners_before_sq, valid=valid,
                direct_kl_max_error=kl_error, distribution_mass_max_error=mass_error)


def summarize_steps(result):
    """Retain every coordinate decision; raw log-demand stays in the NPZ."""
    rows = []
    for step in result['steps']:
        rows.append(dict(bank=int(step['bank']), cell=int(step['cell']), active=bool(step['active']),
                         accepted=bool(step['accepted']), old_slot=int(step['old_slot']),
                         chosen_slot=int(step['chosen_slot']),
                         feasible_count=int(step['feasible_candidates'].sum()),
                         normalized_surrogate_improvement=float(step['normalized_surrogate_improvement'])))
    require(len(rows) == 945 * 3, 'incomplete medoid coordinate pass')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=lambda p: Path(p).resolve(), required=True)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'explicitly disable CUDA')
    require(os.sched_getaffinity(0) == {124, 125, 126, 127}, 'require CPU affinity124-127')
    require(all(os.environ.get(name) == '1' for name in
                ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')), 'require one CPU thread')
    require(not OUTPUT.exists(), 'fresh fixed output only; no resume or overwrite')
    sys.path.insert(0, str(args.source_root))
    import numpy as np
    import torch
    from attention.adaptive_seed_pytorch import multiscale_candidate_table
    from analysis.anchor_fullgrid_seed import anchor_indices
    from analysis.sq1_conditional_medoid_reference import uniform16_sq1_support
    from analysis.global_route_medoid_reference import refine_global_route_medoids
    from analysis.diagnose_sq1_fixed_seed_witness import input_contract, OLD_ROOT, CAPTURE_SHA

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), 'CUDA must remain uninitialized')
    helpers = (multiscale_candidate_table, anchor_indices, uniform16_sq1_support,
               refine_global_route_medoids, input_contract)
    sources = [Path(__file__).resolve()]
    for helper in helpers:
        path = Path(inspect.getsourcefile(helper)).resolve()
        require(path.is_relative_to(args.source_root), 'helper imported outside fixed source root')
        if path not in sources:
            sources.append(path)
    source_hashes = {str(path): digest(path) for path in sources}
    _, input_row, _ = input_contract(OLD_ROOT)  # Small receipts only; old Mass seeds/cache are NOT loaded.
    stat = CAPTURE.stat()
    before_stat = (stat.st_size, stat.st_mtime_ns)
    require(Path(input_row['input']['path']) == CAPTURE and stat.st_size == 309661725, 'wrong locked capture')
    require(digest(CAPTURE) == CAPTURE_SHA, 'single capture SHA mismatch')
    OUTPUT.mkdir()
    (OUTPUT / 'sources').mkdir()
    archived = []
    for index, path in enumerate(sources):
        destination = OUTPUT / 'sources' / f'{index}_{path.name}'
        shutil.copy2(path, destination)
        archived.append(dict(path=str(destination), original=str(path), sha256=source_hashes[str(path)]))
    report = dict(complete=False, valid=False, candidate=False, video_quality_result=False, timing_benchmark=False,
        scope='Fixed six head/column cases from one prompt1/step0/L33 original BF16 capture; no population extension or post-result tuning',
        selection='Only RF0 Q; all945 K banks; Anchor32 C32/g45/U16 incumbent; all45 RF0 targets and the same C32 candidates for both initially frozen-beta medoid passes; anchor fixed',
        control='global versus bank_local changes only the initial frozen route demand; neither mode refreshes beta during its three-slot pass',
        measurement='Only frames0/10/20: independent ideal FP64 original BF16 QK, full80 reownership then direct SQ1 bulk normalization, all945x5 compressed components in one global route',
        selector_precision='Original BF16 Q/K cast FP32 for CPU bmm; then FP64 scale/Fisher/medoid. This is not native selector or output parity evidence',
        limitations='Probe monotonicity does not imply full80 gap, output MSE, temporal transfer, all-layer or video quality; CPU diagnostic time is not attention latency',
        no_large_capture_library_reads=True, capture_copied=False, source_sha256=source_hashes, source_archive=archived,
        capture=dict(path=str(CAPTURE), sha256=CAPTURE_SHA, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns),
        capture_metadata=input_row['metadata'], cases=CASES, frames=FRAMES, key_banks=945,
        cpu_affinity=sorted(os.sched_getaffinity(0)), cpu_threads=1, records=[])
    def save():
        (OUTPUT / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    save()
    started = time.perf_counter()
    saved = torch.load(CAPTURE, map_location='cpu', weights_only=True)
    require(set(saved) == {'q', 'k', 'v', 'dense_output', 'metadata'}, 'unexpected capture payload')
    require(saved['metadata'] == {k: v for k, v in input_row['metadata'].items() if k not in ('bytes', 'sha256')}, 'capture metadata changed')
    for name in ('q', 'k', 'v', 'dense_output'):
        tensor = saved[name]
        require(tuple(tensor.shape) == (1, 75600, 4, 128) and tensor.dtype == torch.bfloat16
                and tensor.device.type == 'cpu', 'not original CPU BF16 capture')
    q, k, v = (saved[name].reshape(21, 45, 80, 4, 128) for name in ('q', 'k', 'v'))
    del saved, tensor  # Saved BF16 dense output is not the independent FP64 reference.
    table, anchor_slots = multiscale_candidate_table(45, 32, 'cpu', geometry_count=45)
    require(tuple(table.shape) == (45, 32), 'canonical C32 table shape changed')
    candidates = table.long().repeat(21, 1)
    anchors = anchor_slots.repeat(21)
    require(torch.equal(candidates.gather(1, anchors[:, None].long()).squeeze(1), torch.arange(45).repeat(21)), 'matched anchor provenance failed')
    arrays = dict(candidate_rows=candidates.numpy())
    for head in (0, 30):
        stored_head = (0, 10, 20, 30).index(head)
        # Only the selected stored head is promoted; never materialize a full40 capture.
        keys = k[..., stored_head, :].reshape(945, 80, 128)
        keys64 = keys.double().numpy()
        values64 = v[..., stored_head, :].reshape(945, 80, 128).double().numpy()
        for column in (0, 40, 41):
            prefix = f'head{head}_column{column}'
            support = uniform16_sq1_support(column)
            probes = support['probe_columns']
            rf0 = q[0, :, column, stored_head].float()  # Selection's only Q frame read.
            kp = keys[:, probes].float()
            z32 = torch.bmm(rf0[candidates], kp[:, :16].transpose(1, 2)).double() * SCALE
            slots = anchor_indices(z32, anchors)
            incumbent = candidates.gather(1, slots).numpy()
            zp = (torch.bmm(rf0[None].expand(945, -1, -1), kp.transpose(1, 2)).double() * SCALE).numpy()
            updates = {mode: refine_global_route_medoids(zp, candidates.numpy(), incumbent, probes, column, weighting=mode)
                       for mode in ('global', 'bank_local')}
            seed_sets = dict(anchor=incumbent, **{mode: result['seed_rows'] for mode, result in updates.items()})
            record = dict(head=head, stored_head_slot=stored_head, query_column=column,
                          probe_columns=probes.tolist(), probe_measures=support['measures'].tolist(),
                          key_banks=945, adjustable_slots=2835, probe={}, full80=[])
            arrays[prefix + '_probe_logits'] = zp
            arrays[prefix + '_incumbent_rows'] = incumbent
            for mode, result in updates.items():
                require(np.array_equal(result['seed_rows'][:, 0], incumbent[:, 0]), 'anchor changed')
                require(np.array_equal(result['initial']['owners'], result['final']['owners']), 'probe bulk ownership changed')
                old_local = float(result['initial']['local_gap'].mean())
                new_local = float(result['final']['local_gap'].mean())
                if mode == 'global':
                    require(result['new_mean_global_gap'] <= result['old_mean_global_gap'] + 2e-9, 'global probe monotonicity failed')
                else:
                    require(new_local <= old_local + 2e-9, 'bank-local probe monotonicity failed')
                record['probe'][mode] = dict(changed_banks=int(result['changed_slots'].any(1).sum()),
                    changed_slots=int(result['changed_slots'].sum()), old_mean_global_gap=result['old_mean_global_gap'],
                    new_mean_global_gap=result['new_mean_global_gap'], old_mean_bank_local_gap=old_local,
                    new_mean_bank_local_gap=new_local, steps=summarize_steps(result))
                for name in ('seed_rows', 'changed_slots', 'frozen_log_beta', 'old_global_gap', 'new_global_gap'):
                    arrays[prefix + '_' + mode + '_' + name] = result[name]
            record['global_vs_local_different_banks'] = int(np.any(seed_sets['global'] != seed_sets['bank_local'], axis=1).sum())
            # Only after BOTH seed sets are final may diagnostic frames10/20 be read.
            for frame in FRAMES:
                query = q[frame, :, column, stored_head].double().numpy()
                full = (query @ keys64.reshape(-1, 128).T) * SCALE
                require(np.isfinite(full).all(), 'nonfinite full80 QK')
                dense_log_z = np.logaddexp.reduce(full, axis=1)
                dense_output = np.exp(full - dense_log_z[:, None]) @ values64.reshape(-1, 128)
                z = full.reshape(45, 945, 80).transpose(1, 0, 2)
                states = {mode: full80_state(z, values64, rows, column, dense_log_z, dense_output)
                          for mode, rows in seed_sets.items()}
                item = dict(query_frame=frame, methods={})
                arrays[f'{prefix}_frame{frame}_dense_output'] = dense_output
                arrays[f'{prefix}_frame{frame}_dense_log_z'] = dense_log_z
                for mode, state in states.items():
                    item['methods'][mode] = dict(mean_global_gap=float(state['gap'].mean()),
                        mse_vs_dense=float(state['per_row_mse'].mean()),
                        per_row_global_gap=state['gap'].tolist(), per_row_mse=state['per_row_mse'].tolist(),
                        delta_mean_gap_vs_anchor=float((state['gap'] - states['anchor']['gap']).mean()),
                        delta_mse_vs_anchor=float((state['per_row_mse'] - states['anchor']['per_row_mse']).mean()),
                        owner_changes_vs_anchor=int(np.count_nonzero(state['owners'] != states['anchor']['owners'])),
                        singleton_pre_owner_changes_vs_anchor=int(np.count_nonzero(state['owners_before_sq'][:, column] != states['anchor']['owners_before_sq'][:, column])),
                        structurally_valid_bulk_cells=int(state['valid'].sum()), direct_kl_max_error=state['direct_kl_max_error'],
                        distribution_mass_max_error=state['distribution_mass_max_error'])
                    for name in ('output', 'owners', 'owners_before_sq'):
                        arrays[f'{prefix}_frame{frame}_{mode}_{name}'] = state[name]
                record['full80'].append(item)
            report['records'].append(record)
            save()
            del updates, z32, zp, rf0, kp, states, full, z
        del keys64, values64
    require([(r['head'], r['query_column']) for r in report['records']] == list(CASES), 'fixed case coverage changed')
    require(all([item['query_frame'] for item in r['full80']] == list(FRAMES) for r in report['records']), 'frame coverage incomplete')
    report['summary'] = {'probe': {}, 'full80': {}}
    for mode in ('global', 'bank_local'):
        rows = [record['probe'][mode] for record in report['records']]
        report['summary']['probe'][mode] = dict(changed_banks=sum(r['changed_banks'] for r in rows),
            total_banks=6 * 945, changed_slots=sum(r['changed_slots'] for r in rows), adjustable_slots=6 * 2835,
            old_mean_global_gap=float(np.mean([r['old_mean_global_gap'] for r in rows])),
            new_mean_global_gap=float(np.mean([r['new_mean_global_gap'] for r in rows])))
    for frame in FRAMES:
        rows = [item for r in report['records'] for item in r['full80'] if item['query_frame'] == frame]
        report['summary']['full80'][str(frame)] = {mode: dict(
            mean_global_gap=float(np.mean([r['methods'][mode]['mean_global_gap'] for r in rows])),
            mean_mse_vs_dense=float(np.mean([r['methods'][mode]['mse_vs_dense'] for r in rows])),
            owner_changes_vs_anchor=sum(r['methods'][mode]['owner_changes_vs_anchor'] for r in rows),
            total_bulk_owners=6 * 945 * 79) for mode in METHODS}
    array_path = OUTPUT / 'arrays.npz'
    np.savez_compressed(array_path, **arrays)
    report['arrays'] = dict(path=str(array_path), bytes=array_path.stat().st_size, sha256=digest(array_path))
    require((CAPTURE.stat().st_size, CAPTURE.stat().st_mtime_ns) == before_stat, 'capture changed')
    require({str(path): digest(path) for path in sources} == source_hashes, 'source changed')
    require(all(digest(row['path']) == row['sha256'] for row in archived), 'source archive mismatch')
    require(not torch.cuda.is_initialized(), 'CUDA unexpectedly initialized')
    report.update(complete=True, valid=True, cuda_initialized=False,
                  cpu_diagnostic_seconds=time.perf_counter() - started)
    save()
    print(json.dumps(report['summary'], indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
