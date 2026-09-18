"""Compare three fixed first-seed rules on cached QK; no GPU or video claims."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sq1_partition_seed_reference import sq1_partition_metrics, select_sq1_partition_seeds


def first_seed_costs(z, mu, importance):
    weighted = z + np.log(mu)[None, None, :]
    logp = weighted - np.logaddexp.reduce(weighted, axis=-1, keepdims=True)
    lr = z[..., :-1] - np.logaddexp.reduce(z[..., :-1], axis=-1, keepdims=True)
    # Axes: bank, seed candidate s, represented target i.
    kl = (np.exp(lr) * lr).sum(-1)[:, :, None] - np.einsum('bsp,bip->bsi', np.exp(lr), lr)
    bulk_log_mass = np.logaddexp.reduce(logp[..., :-1], axis=-1)
    gap = -np.logaddexp(logp[:, None, :, -1], bulk_log_mass[:, None, :] - kl)
    pi = importance / importance.sum(-1, keepdims=True)
    exact = (gap * pi[:, None, :]).sum(-1)
    upper = (kl * (pi * np.exp(bulk_log_mass))[:, None, :]).sum(-1)
    if np.max(exact - upper) > 1e-11:
        raise ValueError('Jensen upper-bound identity failed')
    return exact, upper


def fisher_pivots(z, anchor, importance, weighted_center, weighted_score):
    """Fisher features on real U16 logits, optionally weighted by block mass."""
    pi = importance / importance.sum(-1, keepdims=True)
    center = (z*pi[..., None]).sum(1) if weighted_center else z.mean(1)
    rho = np.exp(center - np.logaddexp.reduce(center, axis=-1, keepdims=True))
    delta = z - z[np.arange(len(z)), anchor, None]
    residual = np.sqrt(rho[:, None])*(delta-(delta*rho[:, None]).sum(-1,keepdims=True))
    chosen = anchor[:, None].copy()
    used = np.arange(32)[None] == anchor[:, None]
    for _ in range(3):
        score = (residual**2).sum(-1)
        if weighted_score: score *= importance
        pick = np.argmax(np.where(used, -np.inf, score), axis=-1)
        chosen = np.concatenate((chosen,pick[:, None]),axis=-1)
        used[np.arange(len(z)),pick] = True
        basis = residual[np.arange(len(z)),pick]
        basis = basis/np.sqrt((basis*basis).sum(-1,keepdims=True)+1e-12)
        residual -= (residual*basis[:, None]).sum(-1,keepdims=True)*basis[:, None]
    return chosen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fisher',action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('fresh result required')
    results = []
    for prompt in range(3):
        path = args.cache_dir / f'prompt{prompt:03d}_sq1_native_parity.npz'
        with np.load(path, allow_pickle=False) as raw:
            prefix = 'stored_seed_logp__'
            coords = raw['bank_coordinates']
            support = raw[prefix+'quadrature_support_count']
            for size in (16, 17):
                mask = support == size
                z = raw[prefix+'quadrature_logits_natural_fp64'][mask, :, :size]
                mu = raw[prefix+'quadrature_measures'][mask, :size][0]
                pi = raw[prefix+'reference_mass_weights'][mask]
                anchors = raw['anchors'][mask]
                exact, upper = first_seed_costs(z, mu, pi)
                rules = {'anchor': anchors, 'exact_sq1_medoid': exact.argmin(-1),
                         'sq1_bound_medoid': upper.argmin(-1)}
                selected_rules = {name:select_sq1_partition_seeds(z,first,size-1,measures=mu,importance=pi)
                                  for name,first in rules.items()}
                if args.fisher:
                    z16=raw['local_logits_natural_fp64'][mask]
                    mass=np.logaddexp.reduce(z16,axis=-1)-raw['reference_global_log2_normalizer'][mask]*np.log(2)
                    weight=np.exp(mass-mass.max(-1,keepdims=True))
                    selected_rules.update({name:fisher_pivots(z16,anchors,weight,wc,ws) for name,wc,ws in (
                        ('fisher_unweighted',False,False),('fisher_mass_score',False,True),('fisher_mass_center_score',True,True))})
                    with np.load(args.cache_dir/f'prompt{prompt:03d}_native_seeds.npz',allow_pickle=False) as seeds:
                        hh,jj,ff,rr=coords[mask].T
                        real=seeds['hellinger_rf0'][0,0,ff,rr,jj,hh]
                        selected_rules['hellinger_rf0']=(real[:,:,None]==raw['candidate_rows'][mask,None,:]).argmax(-1)
                for name, selected in selected_rules.items():
                    first=selected[:,0]
                    loss = sq1_partition_metrics(z, z[np.arange(len(z))[:, None], selected],
                                                 size-1, measures=mu)['reverse_gap']
                    bank_loss = (pi * loss).sum(-1) / pi.sum(-1)
                    results.append({'prompt': prompt, 'support': size, 'rule': name, 'banks': len(z),
                        'first_seed_gap': float(exact[np.arange(len(z)), first].mean()),
                        'c4_gap_mean': float(bank_loss.mean()),
                        'c4_gap_by_head': {str(h): float(bank_loss[coords[mask, 0] == h].mean()) for h in range(4)},
                        'first_seed_changed': int((first != anchors).sum()),
                        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    summary = {name: float(sum(r['c4_gap_mean']*r['banks'] for r in results if r['rule']==name)/1800)
               for name in selected_rules}
    output = {'scope': 'CPU quadrature objective only; not native output MSE, PSNR or runtime',
              'records': results, 'mean_c4_gap': summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(output, f, indent=2, allow_nan=False)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
