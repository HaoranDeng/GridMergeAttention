"""Quick exact-C4 axis diagnostic on frozen 720p QKV; no video inference.

All six directed axis pairs use the same current-line anchor-Fisher rule.
Spatial-query pairs additionally use physical RF0 as a production-policy
control. All key groups participate; exactly four slots, no singleton.
This reference computes a common sampled query set, so its timing is NOT a
complete attention call or an optimized-kernel speedup.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

AXES = "thw"
GRID = (21, 45, 80)
SAMPLES = ((0, 10, 20), (0, 22, 44), (0, 40, 79))
CAPTURE_ROOT = Path('/local2/denghaoran/GridMergeAttention/experiments/rf0_layer33_dense_capture_20260904n/captures')


def save(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_table(length):
    """Existing C32/g45 geometry for H/W; all21 candidates for T."""
    count, geometry = min(32, length), min(45, length)
    table, anchors = [], []
    for anchor in range(length):
        if length <= 32:
            chosen = set(range(length))
        else:
            radius, uniforms = max(1, geometry // 4), max(4, geometry // 4)
            chosen = {anchor}
            chosen.update(range(max(0, anchor - radius), min(length, anchor + radius + 1)))
            chosen.update(round(i * (length - 1) / (uniforms - 1)) for i in range(uniforms))
            if len(chosen) > count:
                raise ValueError('candidate geometry exceeds budget')
            while len(chosen) < count:
                remaining = [i for i in range(length) if i not in chosen]
                chosen.add(max(remaining, key=lambda i: (min(abs(i-j) for j in chosen), -i)))
        selected = sorted(chosen)
        table.append(selected)
        anchors.append(selected.index(anchor))
    return table, anchors


def prepare(k, v, qa, ka):
    import torch
    remaining = [axis for axis in range(3) if axis != ka]
    # Key groups preserve both fixed physical coordinates in lexicographic order.
    kb = k.permute(*remaining, ka, 3).reshape(-1, k.shape[ka], k.shape[-1]).contiguous()
    vb = v.permute(*remaining, ka, 3).reshape_as(kb).contiguous()
    coords = torch.cartesian_prod(*[torch.arange(k.shape[a], device=k.device) for a in remaining])
    anchor = coords[:, remaining.index(qa)]
    table, anchor_slots = candidate_table(k.shape[qa])
    candidate = torch.tensor(table, device=k.device)[anchor]
    slots = torch.tensor(anchor_slots, device=k.device)[anchor]
    probes = torch.arange(16, device=k.device) * k.shape[ka] // 16
    if k.shape[ka] < 16:
        probes = torch.arange(k.shape[ka], device=k.device)
    return kb, vb, candidate, slots, probes


def select_fisher(query, kb, candidates, anchor_slots, probes, scale):
    import torch
    z_all = torch.einsum('id,gpd->gip', query, kb[:, probes]) * scale
    z = z_all.gather(1, candidates[..., None].expand(-1, -1, len(probes))).double()
    groups = torch.arange(len(kb), device=query.device)
    anchor_z = z[groups, anchor_slots]
    rho = anchor_z.softmax(-1)
    delta = z - anchor_z[:, None]
    residual = (delta - (delta * rho[:, None]).sum(-1, keepdim=True)) * rho[:, None].sqrt()
    used = torch.zeros(candidates.shape, dtype=torch.bool, device=query.device)
    used[groups, anchor_slots] = True
    selected = [candidates[groups, anchor_slots]]
    for step in range(1, 4):
        score = residual.square().sum(-1).masked_fill(used, -float('inf'))
        picked = score.argmax(-1)
        selected.append(candidates[groups, picked])
        used[groups, picked] = True
        if step < 3:
            direction = residual[groups, picked]
            basis = direction / (direction.square().sum(-1, keepdim=True) + 1e-12).sqrt()
            residual -= (residual * basis[:, None]).sum(-1, keepdim=True) * basis[:, None]
    return torch.stack(selected, -1)


def line_attention(query, reference_query, kb, vb, candidates, slots, probes, sampled_indices, *, quantize=True):
    import torch
    scale = query.shape[-1] ** -.5
    seeds = select_fisher(reference_query, kb, candidates, slots, probes, scale)
    z = torch.bmm(query[seeds], kb.transpose(-1, -2)) * scale
    owner = z.log_softmax(-1).argmax(-2)
    assigned = owner[:, None] == torch.arange(4, device=query.device)[None, :, None]
    active = assigned.any(-1)
    selected = z.masked_fill(~assigned, -float('inf'))
    maximum = selected.amax(-1, keepdim=True)
    maximum = torch.where(active[..., None], maximum, 0.0)
    weights = torch.exp(selected - maximum)
    weights /= weights.sum(-1, keepdim=True).clamp_min(1e-30)
    entropy = -(weights * weights.clamp_min(1e-30).log()).sum(-1)
    # Match the BF16 representation boundary, preserving FP32 entropy.
    if quantize:
        merge_weights = weights.to(torch.bfloat16).float()
        mk = torch.bmm(merge_weights, kb).to(torch.bfloat16).float()
        mv = torch.bmm(merge_weights, vb).to(torch.bfloat16).float()
    else:
        mk, mv = torch.bmm(weights, kb), torch.bmm(weights, vb)
    logits = query[sampled_indices] @ mk.flatten(0, 1).T * scale
    logits += entropy.flatten()[None]
    logits.masked_fill_(~active.flatten()[None], -float('inf'))
    output = logits.softmax(-1) @ mv.flatten(0, 1)
    return output, {'empty': int((~active).sum()), 'slots': active.numel()}


def evaluate(q, k, v, qa, ka, policy, samples=SAMPLES, quantize=True):
    import torch
    kb, vb, candidates, slots, probes = prepare(k, v, qa, ka)
    other = [a for a in range(3) if a != qa]
    rows, empty, total = {}, 0, 0
    for fixed in itertools.product(*[samples[a] for a in other]):
        index = [slice(None)] * 3
        for axis, value in zip(other, fixed):
            index[axis] = value
        line = q[tuple(index)]
        reference_index = list(index)
        if policy == 'rf0':
            if qa == 0:
                raise ValueError('RF0 cannot supply four temporal query seeds')
            reference_index[0] = 0
        reference_line = q[tuple(reference_index)]
        out, counts = line_attention(line, reference_line, kb, vb, candidates, slots, probes,
                                    list(samples[qa]), quantize=quantize)
        empty += counts['empty']; total += counts['slots']
        for inner, value in enumerate(samples[qa]):
            coordinate = list(index); coordinate[qa] = value
            rows[tuple(coordinate)] = out[inner]
    return torch.stack([rows[c] for c in itertools.product(*samples)]), empty / total


def selftest():
    import torch
    torch.manual_seed(719)
    q, k = [torch.randn(4, 5, 6, 8, dtype=torch.float64) for _ in range(2)]
    v = torch.randn_like(q)
    for qa, ka in itertools.permutations(range(3), 2):
        kb, vb, candidates, slots, probes = prepare(k, v, qa, ka)
        others = [a for a in range(3) if a != ka]
        for g, fixed in enumerate(itertools.product(*[range(k.shape[a]) for a in others])):
            index = [slice(None)] * 3
            for a, x in zip(others, fixed): index[a] = x
            assert torch.equal(kb[g], k[tuple(index)])
            assert candidates[g, slots[g]] == fixed[others.index(qa)]
        # Identical Q gives one nonempty route. With Q=0, entropy restores
        # exact key multiplicity and uniform dense averaging across all keys.
        zero = torch.zeros_like(q)
        out, fraction = evaluate(zero, k, v, qa, ka, 'current', samples=((0,3),(0,4),(0,5)), quantize=False)
        assert torch.allclose(out, v.mean((0,1,2)).expand_as(out), atol=1e-12, rtol=1e-12)
        assert fraction == .75
        # Constant V must be preserved for arbitrary Q/K, including RF0.
        constant = torch.ones_like(v) * 2.5
        out, _ = evaluate(q, k, constant, qa, ka, 'current', samples=((0,3),(0,4),(0,5)), quantize=False)
        assert torch.allclose(out, torch.full_like(out, 2.5), atol=1e-12, rtol=1e-12)
    # Independent literal block computation checks ownership, entropy sign,
    # all four routes and empty cells on a nontrivial asymmetric grid.
    qa, ka = 1, 2
    kb, vb, candidates, slots, probes = prepare(k, v, qa, ka)
    line = q[2, :, 3]
    seeds = select_fisher(line, kb, candidates, slots, probes, 8 ** -.5)
    keys, vals, biases = [], [], []
    for group in range(len(kb)):
        z = line[seeds[group]] @ kb[group].T * (8 ** -.5)
        owners = z.log_softmax(-1).argmax(0)
        for c in range(4):
            members = owners == c
            if members.any():
                w = z[c, members].softmax(0)
                keys.append(w @ kb[group, members]); vals.append(w @ vb[group, members])
                biases.append(-(w * w.log()).sum())
    logits = line @ torch.stack(keys).T * (8 ** -.5) + torch.stack(biases)
    expected = logits.softmax(-1) @ torch.stack(vals)
    actual, _ = line_attention(line, line, kb, vb, candidates, slots, probes, list(range(5)), quantize=False)
    assert torch.allclose(expected, actual, atol=1e-12, rtol=1e-12)
    return {'passed': True, 'axes': 6, 'checks': ['physical index mapping', 'matched anchors',
            'exact C4 and empty routes', 'uniform dense identity', 'constant V', 'literal scalar reference']}


def summarize(rows):
    methods = sorted({r['method'] for r in rows})
    result = []
    for method in methods:
        selected = [r for r in rows if r['method'] == method]
        qa, ka = selected[0]['query_axis'], selected[0]['key_axis']
        result.append({'method': method, 'query_axis': qa, 'key_axis': ka,
            'query_tokens': GRID[AXES.index(qa)], 'key_tokens': GRID[AXES.index(ka)],
            'compressed_tokens_per_query': 75600 * 4 // GRID[AXES.index(ka)],
            'compressed_token_density': 4 / GRID[AXES.index(ka)], 'cases': len(selected),
            'relative_l2_mean': sum(r['relative_l2'] for r in selected) / len(selected),
            'relative_l2_pooled': (sum(r['squared_error'] for r in selected) / sum(r['reference_squared'] for r in selected)) ** .5,
            'sampled_27query_reference_ms_mean': sum(r['sampled_ms'] for r in selected) / len(selected),
            'empty_slot_fraction_mean': sum(r['empty_slot_fraction'] for r in selected) / len(selected),
            'mean_attention_ms': None, 'speedup_vs_sdpa': None})
    return sorted(result, key=lambda r: r['relative_l2_mean'])


def worker(args):
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tests = selftest()
    if args.selftest:
        print(json.dumps(tests)); return
    if os.environ.get('AXIS_C4_PARENT') != str(os.getppid()):
        raise RuntimeError('GPU diagnostic requires the monitored parent')
    torch.cuda.set_device(0)
    source_manifest = json.loads((args.capture_root / 'capture_manifest.json').read_text())
    prompt_manifest = json.loads((args.capture_root / 'prompt_manifest.json').read_text())
    if not source_manifest.get('complete'):
        raise ValueError('incomplete source capture')
    report = {'scope': 'Frozen 4-step Wan2.1-14B-720p dense QKV, sampled queries; no videos',
              'capacity': 4, 'grid': GRID, 'samples': SAMPLES, 'selftest': tests,
              'policies': {'current': 'Fisher on current physical query line, matched anchor, Cmin(32,Lq), Uniform16 keys',
                           'rf0': 'same spatial query line at physical t=0 for seed selection; current queries for merge'},
              'precision': 'original BF16 inputs; FP32 dot/reductions, FP64 Fisher/GS, BF16 merge weights and KV; FP32 route',
              'timing_note': 'Python reference over 27 sampled queries per head. Complete attention latency and SDPA speedup unmeasured.',
              'prompt_manifest': prompt_manifest, 'captures': [], 'rows': [], 'summary': []}
    variants = [(qa, ka, 'current') for qa, ka in itertools.permutations(range(3), 2)]
    variants += [(qa, ka, 'rf0') for qa, ka in itertools.permutations(range(3), 2) if qa != 0]
    coordinates = list(itertools.product(*SAMPLES))
    for p in range(args.prompts):
        for call in args.calls:
            layer = call % 40
            path = args.capture_root / f'prompt{p:03d}_call{call:03d}_layer{layer}.pt'
            payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            saved_heads = len(payload['metadata']['heads'])
            for name in ('q','k','v','dense_output'):
                if tuple(payload[name].shape) != (1,75600,saved_heads,128) or payload[name].dtype != torch.bfloat16:
                    raise ValueError(f'invalid capture {path}/{name}')
            report['captures'].append({'path': str(path), 'sha256': sha(path), 'metadata': payload['metadata']})
            for head in args.head_slots:
                q, k, v, saved = [payload[name][0,:,head].to('cuda').float().reshape(*GRID,128)
                                  for name in ('q','k','v','dense_output')]
                sampled = torch.stack([q[c] for c in coordinates])
                # FP32 dense diagnostic target uses exactly the same Q/K/V.
                dense = (sampled @ k.reshape(-1,128).T * (128 ** -.5)).softmax(-1) @ v.reshape(-1,128)
                saved_subset = torch.stack([saved[c] for c in coordinates])
                dense_gap = float((dense-saved_subset).norm()/dense.norm())
                for qa, ka, policy in variants:
                    torch.cuda.synchronize(); start = time.perf_counter()
                    output, empty = evaluate(q,k,v,qa,ka,policy)
                    torch.cuda.synchronize(); elapsed = 1000*(time.perf_counter()-start)
                    difference = output-dense
                    per_query = (difference.norm(dim=-1)/dense.norm(dim=-1).clamp_min(1e-30)).cpu().tolist()
                    row = {'prompt': p, 'call': call, 'layer': layer, 'step': call//40,
                           'head': int(payload['metadata']['heads'][head]),
                           'method': f'q{AXES[qa]}_k{AXES[ka]}_c4_{policy}',
                           'query_axis': AXES[qa], 'key_axis': AXES[ka], 'policy': policy,
                           'relative_l2': float(difference.norm()/dense.norm()),
                           'squared_error': float(difference.square().sum()),
                           'reference_squared': float(dense.square().sum()),
                           'rmse': float(difference.square().mean().sqrt()),
                           'per_query_relative_l2': per_query, 'sampled_ms': elapsed,
                           'empty_slot_fraction': empty, 'fp32_dense_vs_saved_bf16_relative_l2': dense_gap}
                    report['rows'].append(row)
                report['summary'] = summarize(report['rows'])
                save(args.output / 'results.json', report)
                print(json.dumps({'prompt':p,'call':call,'head':row['head'],
                                  'errors':{r['method']:round(r['relative_l2'],5) for r in report['rows'][-len(variants):]}}), flush=True)
            del payload, q, k, v, saved
    report['complete'] = True
    save(args.output / 'results.json', report)


def gpu_state(gpu):
    def query(fields, kind):
        value = subprocess.check_output(['nvidia-smi','-i',str(gpu),f'--query-{kind}={fields}',
                                         '--format=csv,noheader,nounits'],text=True,timeout=10)
        return [[x.strip() for x in row] for row in csv.reader(value.splitlines()) if row]
    state = query('name,uuid,memory.used,utilization.gpu','gpu')[0]
    return {'name':state[0],'uuid':state[1],'memory_mib':int(state[2]),'utilization':int(state[3]),
            'pids':[int(row[0]) for row in query('pid','compute-apps')]}


def parent(args):
    state = gpu_state(args.gpu)
    if 'H200' not in state['name'] or state['pids'] or state['memory_mib'] > 100 or state['utilization'] != 0:
        raise RuntimeError(f'GPU not clean; no launch or queue: {state}')
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'preflight.json',state)
    source = Path(__file__).resolve()
    frozen = args.output / source.name
    frozen.write_bytes(source.read_bytes())
    save(args.output/'source.json',{'source':str(source),'sha256':sha(frozen)})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), AXIS_C4_PARENT=str(os.getpid()),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [sys.executable,str(frozen),'--worker','--output',str(args.output),
               '--capture-root',str(args.capture_root),
               '--prompts',str(args.prompts),'--calls',*[str(v) for v in args.calls],
               '--head-slots',*[str(v) for v in args.head_slots]]
    state = gpu_state(args.gpu)
    if state['pids'] or state['memory_mib'] > 100 or state['utilization']:
        raise RuntimeError('GPU changed before spawn; not launching')
    child = None
    failure = None
    try:
        with (args.output/'worker.log').open('w') as log, (args.output/'gpu_monitor.jsonl').open('w') as monitor:
            child = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(args.output/'process.json',{'parent_pid':os.getpid(),'child_pid':child.pid,'command':command})
            while child.poll() is None:
                state = gpu_state(args.gpu)
                monitor.write(json.dumps({'time':time.time(),**state})+'\n'); monitor.flush()
                foreign = set(state['pids'])-{child.pid}
                if foreign:
                    raise RuntimeError(f'GPU contamination: {sorted(foreign)}')
                time.sleep(2)
            if child.returncode:
                raise RuntimeError(f'worker exited {child.returncode}')
    except BaseException as error:
        failure = str(error)
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            child.wait(timeout=20)
        raise
    finally:
        save(args.output/'completion.json',{'valid':failure is None,'failure':failure,
            'worker_exit_code':None if child is None else child.returncode,
            'complete_attention_latency_measured':False,'source_sha256':sha(frozen)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--capture-root',type=Path,default=CAPTURE_ROOT)
    parser.add_argument('--prompts',type=int,default=3)
    parser.add_argument('--calls',type=int,nargs='+',default=[33,73,113,153])
    parser.add_argument('--head-slots',type=int,nargs='+',default=[0,1,2,3])
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--selftest',action='store_true')
    args = parser.parse_args()
    if args.selftest or args.worker: worker(args)
    else: parent(args)


if __name__ == '__main__':
    main()
