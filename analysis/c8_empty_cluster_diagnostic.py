"""Measure unused exact-C8 slots outside all latency timing regions."""


def empty_cluster_histogram(backend, retained, q, k, v, seeds, clean):
    import torch
    from kernels.best.c8_r2.gma_c8_r2 import MERGE_TILES

    rows = []
    for head in range(0, 40, 5):
        clean()
        retained.run(backend.packed_q, k, v, seeds, backend.mk, backend.mv,
                     backend.bias, head, 5, MERGE_TILES, 128**-0.5,
                     backend.empty, 0, 1)
        b = backend.bias.reshape(-1, 8)
        empty = torch.isposinf(b)
        count = empty.sum(dim=1)
        hist = torch.bincount(count, minlength=9).cpu().tolist()
        invalid = int((torch.isnan(b) | torch.isneginf(b)).sum())
        rows.append({'head_start': head, 'heads': 5,
                     'blocks_by_empty_count_0_to_8': hist,
                     'invalid_bias_count': invalid,
                     'empty_fraction': sum(i*n for i,n in enumerate(hist))/(8*sum(hist))})
        clean()
    total = [sum(row['blocks_by_empty_count_0_to_8'][i] for row in rows) for i in range(9)]
    return {'scope': 'Retained merge, all H40 heads and every exact-C8 block; outside timing.',
            'chunks': rows, 'blocks_by_empty_count_0_to_8': total,
            'empty_fraction': sum(i*n for i,n in enumerate(total))/(8*sum(total)),
            'invalid_bias_count': sum(row['invalid_bias_count'] for row in rows)}
