"""Validate simple native axis C4 against independent sampled reference."""
import argparse
import json
import time
from pathlib import Path

import torch
from attention.axis_current_c4_triton import AxisCurrentC4, merged_attention
from analysis.quick_axis_c4_qkv import candidate_table, select_fisher, line_attention, evaluate, SAMPLES, CAPTURE_ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--axes', choices=['tw', 'wh', 'th'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {'axes': args.axes, 'capacity': 4, 'cases': [], 'passed': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Production R/L dimensions, two outer slices, including tail columns.
        r, l = {'tw': (21, 80), 'wh': (80, 45), 'th': (21, 45)}[args.axes]
        gen = torch.Generator(device='cuda').manual_seed(719)
        k, v = [torch.randn((1, 2, r, l, 1, 128), device='cuda', dtype=torch.bfloat16, generator=gen) for _ in range(2)]
        q0 = torch.randn_like(k)
        for zero in (False, True):
            q = torch.zeros_like(q0) if zero else q0
            out, mk, mv, bias, seeds = merged_attention(q, k, v, audit=True)
            torch.cuda.synchronize()
            assert bool(torch.isfinite(out).all())
            table, anchors = candidate_table(r)
            candidates = torch.tensor(table, device='cuda').repeat(2, 1)
            anchors = torch.tensor(anchors, device='cuda').repeat(2)
            kb, vb = [x[0, ..., 0, :].float().reshape(2*r, l, 128) for x in (k, v)]
            probes = torch.arange(16, device='cuda') * l // 16
            errors, mismatch = [], 0
            for o, j in ((0, 0), (0, l-1), (1, l//2), (1, l-1)):
                query = q[0, o, :, j, 0].float()
                refseeds = select_fisher(query, kb, candidates, anchors, probes, 128**-.5)
                mismatch += int((refseeds != seeds[0,o,j,0].reshape(-1,4)).any(-1).sum())
                expected, _ = line_attention(query, query, kb, vb, candidates, anchors, probes, list(range(r)))
                got = out[0,o,:,j,0].float()
                errors.append(((got-expected).norm()/expected.norm()).item())
            empties = int(torch.isposinf(bias).sum())
            if zero:
                assert empties == bias.numel() * 3 // 4
            assert max(errors) < .012, errors
            assert mismatch <= (8 if not zero else 0), mismatch
            report['cases'].append({'zero_query': zero, 'sampled_relative_l2': errors,
                                    'seed_mismatched_cells': mismatch, 'empty_slots': empties})
        del q0, q, k, v, out, mk, mv, bias, seeds
        torch.cuda.empty_cache()
        core = AxisCurrentC4(args.axes)
        capture = CAPTURE_ROOT/'prompt000_call033_layer33.pt'
        payload = torch.load(capture, map_location='cpu', weights_only=True, mmap=True)
        real = [payload[n][0,:,0].to('cuda').reshape(21,45,80,128) for n in ('q','k','v')]
        actual = core(*[x.reshape(1,1,75600,128) for x in real])[0,0].reshape(21,45,80,128)
        expected, _ = evaluate(*[x.float() for x in real],core.qa,core.ka,'current')
        import itertools
        sampled = torch.stack([actual[c].float() for c in itertools.product(*SAMPLES)])
        real_error = float((sampled-expected).norm()/expected.norm())
        assert real_error < .012, real_error
        report['frozen_real_qkv_relative_l2'] = real_error
        del payload, real, actual, expected, sampled
        tensors = [torch.randn((1,40,75600,128), device='cuda', dtype=torch.bfloat16, generator=gen) for _ in range(3)]
        started = time.perf_counter()
        out = core(*tensors)
        torch.cuda.synchronize()
        report['compile_and_smoke_seconds'] = time.perf_counter()-started
        assert bool(torch.isfinite(out).all()) and out.shape == tensors[0].shape
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); out = core(*tensors); end.record(); end.synchronize()
        report['full_40head_attention_ms'] = start.elapsed_time(end)
        if args.axes == 'tw':
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                out = core(*tensors)
                torch.cuda.synchronize()
            report['kernel_profile'] = [{'name':e.key,'device_time_us':e.device_time_total,'count':e.count}
                                         for e in prof.key_averages() if e.device_time_total > 1000]
        report['peak_memory_gib'] = torch.cuda.max_memory_allocated()/2**30
        report['passed'] = True
    finally:
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
