"""Diagnostic CUDA-event boundaries for the exact-C8 complete call.

Events are recorded on the original serial stream, with one synchronization
after the entire call. These instrumented samples never replace ABBA timing.
"""

import statistics


def profile_stages(op, q, k, v, output, clean, repetitions=6):
    import torch
    from kernels.best.c8_r2.gma_c8_r2 import MERGE_TILES

    names = ['selector', 'pack'] + [name for h in range(0, 40, 5)
                                     for name in (f'merge_h{h}', f'route_h{h}')]
    samples = []
    backend = op.backend
    for _ in range(repetitions):
        clean()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(len(names)+1)]
        events[0].record()
        seeds = op.select(q, k)
        events[1].record()
        backend.merge.pack(q, backend.packed_q)
        events[2].record()
        for index, head in enumerate(range(0, 40, 5)):
            backend.merge.run(backend.packed_q, k, v, seeds,
                              backend.mk, backend.mv, backend.bias, head, 5,
                              MERGE_TILES, 128**-0.5, backend.empty, 0, 1)
            events[3+index*2].record()
            backend.route.run(q, backend.mk, backend.mv, backend.bias,
                              output, head, 5, 128**-0.5)
            events[4+index*2].record()
        events[-1].synchronize()
        clean()
        row = {name: events[i].elapsed_time(events[i+1]) for i, name in enumerate(names)}
        row.update(complete_instrumented=events[0].elapsed_time(events[-1]),
                   merge_total=sum(v for k, v in row.items() if k.startswith('merge_')),
                   route_total=sum(v for k, v in row.items() if k.startswith('route_')))
        samples.append(row)
    return {'repetitions': repetitions,
            'scope': 'Instrumented serial stages, diagnostic only; use separate ABBA full-call times.',
            'means_ms': {name: statistics.fmean(row[name] for row in samples) for name in samples[0]},
            'samples_ms': samples}
