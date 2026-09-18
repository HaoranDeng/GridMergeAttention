"""Sample SM clock intervals without privileged performance counters.

Thread-0 elapsed intervals include scheduler stalls and instrumentation overhead.
They are not hardware-counter stall attribution or isolated execution times.
"""


def profile_merge_clocks(module, backend, k, v, seeds, references, clean):
    import torch

    names = ('qk_and_previous_v_store', 'scalar_reductions_and_k_issue',
             'k_wait', 'k_store_prefetch_and_v_issue', 'v_wait',
             'v_conversion_store_and_prefetch')
    trace = torch.zeros((5*945*10, 7, 2), dtype=torch.float32, device=k.device)
    clean()
    module.run(backend.packed_q,k,v,seeds,backend.mk,backend.mv,backend.bias,
               0,5,105,128**-0.5,trace,0,1)
    torch.cuda.synchronize()
    clean()
    matches = {name: bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))
               for name,a,b in zip(('mk','mv','bias'),
                                  (backend.mk,backend.mv,backend.bias),references)}
    timestamps = trace.view(torch.int64).reshape(-1, 7).cpu()
    del trace
    intervals = timestamps[:, 1:] - timestamps[:, :-1]
    if not bool((timestamps > 0).all() and (intervals >= 0).all()):
        raise RuntimeError('missing or nonmonotonic clock sample')
    if not all(matches.values()):
        raise RuntimeError(f'clock instrumentation changed merge outputs: {matches}')
    intervals = intervals.to(torch.float64)
    means = intervals.mean(0)
    return {
        'scope': 'One sampled qo=10 iteration per CTA/column span, H5 at layer capture. Instrumented scheduling and elapsed SM clocks only; not a full-call performance candidate.',
        'sample_count': len(timestamps),
        'instrumented_resources': dict(zip(('registers','local_bytes','shared_bytes','ctas_per_sm'),module.resources())),
        'intermediate_bitwise_equal': matches,
        'total_mean_cycles': float(means.sum()),
        'phases': {name: {
            'mean_cycles': float(means[i]),
            'fraction_of_mean_total': float(means[i]/means.sum()),
            'p10_median_p90_cycles': torch.quantile(intervals[:, i],
                torch.tensor([.1, .5, .9], dtype=torch.float64)).tolist(),
        } for i,name in enumerate(names)},
    }
