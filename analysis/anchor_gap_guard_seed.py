"""One last-slot correction of an externally supplied Anchor32 incumbent.

Keep its first three real candidate slots. The sole challenger maximizes the
CURRENT prefix-three partition reverse-KL gap among slots not in that prefix
(the original fourth slot is eligible). Accept the challenger only when its
completed four-cell mean gap is STRICTLY smaller than the incumbent's. Raw
FP64 gaps are not clipped; exact ranking ties choose the lowest unused slot,
and equal objective values retain the incumbent.
An identical challenger/old-fourth slot is never accepted: the ordered seed
tuple, hence its mathematical objective, is unchanged. Independently computed
raw gaps are still retained verbatim for numerical auditing.

The supplied support must include the explicitly observed SQ key. U16+SQ
uses singleton measure 1 and bulk measures 79/m, m=15 or 16. Owners are
computed with the entire normalized support BEFORE peeling the singleton.
This is the exact restricted reverse KL on the positive-measure quadrature,
not whole-block tangent coverage. No G256, Fisher reimplementation, synthetic
query, Q/K/V modification, merge override, calibration or GPU launch occurs.

The guard is locally non-worsening in exact arithmetic; it does NOT guarantee
the full80/global objective, current-frame transfer, BF16 moments or video
quality. Prefix C3 plus two C4 evaluations cost O(11*R*P), with O(R*P+C*P)
working storage per bank, rather than evaluating every candidate insertion.
"""


def uniform16_sq1_quadrature(singleton_column):
    """Return distinct original U16 columns plus SQ, and their positive masses.

An absent SQ is appended, not replaced by a sampled key or assigned zero
probability. The caller must actually evaluate that additional QK logit.
This function describes width80 geometry only and never reads Q/K.
"""
    import numpy as np

    sq = np.asarray(singleton_column)
    if sq.ndim != 0 or sq.dtype.kind not in "iu" or not 0 <= int(sq) < 80:
        raise ValueError("singleton_column must be an integer in [0,80)")
    columns = np.arange(0, 80, 5, dtype=np.int64)
    matches = np.flatnonzero(columns == int(sq))
    position = int(matches[0]) if matches.size else 16
    if not matches.size:
        columns = np.concatenate((columns, [int(sq)]))
    measure = np.full(len(columns), 79.0 / (len(columns) - 1), dtype=np.float64)
    measure[position] = 1.0
    return {"probe_columns": columns, "singleton_index": position,
            "measures": measure, "bulk_probe_count": len(columns) - 1}


def _numpy_inputs(logits, selected, singleton_index, measures):
    import numpy as np

    raw = np.asarray(logits)
    if raw.dtype.kind != "f":
        raise TypeError("logits must be floating-point")
    if raw.ndim < 2 or min(raw.shape) < 1 or raw.shape[-1] < 2:
        raise ValueError("require nonempty [...,candidates,probes>=2] logits")
    z = raw.astype(np.float64)
    if not np.isfinite(z).all():
        raise ValueError("logits must be finite")
    batch, (rows, probes) = z.shape[:-2], z.shape[-2:]
    slots = np.asarray(selected)
    if slots.ndim < 1 or slots.dtype.kind not in "iu" or not 1 <= slots.shape[-1] <= rows:
        raise ValueError("selected must contain integer candidate slots")
    try:
        slots = np.broadcast_to(slots, (*batch, slots.shape[-1])).astype(np.int64)
    except ValueError as error:
        raise ValueError("selected does not broadcast to the logit batch") from error
    if np.any((slots < 0) | (slots >= rows)) or np.any(np.diff(np.sort(slots, axis=-1)) == 0):
        raise ValueError("selected slots must be distinct and in range")
    sq = np.asarray(singleton_index)
    if sq.dtype.kind not in "iu":
        raise ValueError("singleton_index must contain explicit integers")
    try:
        sq = np.broadcast_to(sq, batch).astype(np.int64)
    except ValueError as error:
        raise ValueError("singleton_index does not broadcast to the batch") from error
    if np.any((sq < 0) | (sq >= probes)):
        raise ValueError("an observed singleton index in [0,P) is required")
    mu = np.asarray(measures)
    if mu.dtype.kind not in "fiu":
        raise ValueError("measures must be real, finite and positive")
    try:
        mu = np.broadcast_to(mu.astype(np.float64), (*batch, probes))
    except ValueError as error:
        raise ValueError("measures do not broadcast to [...,P]") from error
    if not np.isfinite(mu).all() or np.any(mu <= 0):
        raise ValueError("measures must be finite and strictly positive")
    if np.any(np.take_along_axis(mu, sq[..., None], -1) != 1.0):
        raise ValueError("the explicit singleton must have unit measure")
    with np.errstate(over="ignore", invalid="ignore"):
        centered = z - z.max(axis=-1, keepdims=True)
        weighted = centered + np.log(mu)[..., None, :]
        logp = weighted - np.logaddexp.reduce(weighted, axis=-1, keepdims=True)
    if not np.isfinite(logp).all():
        raise ValueError("logit dynamic range exceeds FP64 arithmetic")
    return logp, slots, sq


def _numpy_gap(logp, selected, singleton, details=False):
    import numpy as np

    batch, (rows, probes) = logp.shape[:-2], logp.shape[-2:]
    count = selected.shape[-1]
    all_gap, all_owner, all_weights, all_reconstructed = [], [], [], []
    for lp, slots, sq in zip(logp.reshape(-1, rows, probes),
                             selected.reshape(-1, count), singleton.reshape(-1)):
        owner = lp[slots].argmax(axis=0)
        owner[int(sq)] = count  # Ownership was computed before SQ extraction.
        weight = np.zeros((count + 1, probes), dtype=np.float64)
        component = np.full((rows, count + 1), -np.inf, dtype=np.float64)
        weight[count, int(sq)] = 1.0
        component[:, count] = lp[:, int(sq)]
        for cell, slot in enumerate(slots):
            mask = owner == cell
            if not mask.any():
                continue
            logw = lp[slot, mask] - np.logaddexp.reduce(lp[slot, mask])
            w = np.exp(logw)
            weight[cell, mask] = w
            component[:, cell] = lp[:, mask] @ w - w @ logw
        represented = np.logaddexp.reduce(component, axis=-1)
        all_gap.append(-represented)
        if details:
            all_owner.append(owner)
            all_weights.append(weight)
            all_reconstructed.append(np.exp(component - represented[:, None]) @ weight)
    result = np.asarray(all_gap).reshape(*batch, rows)
    if not np.isfinite(result).all():
        raise ValueError("partition gap exceeds finite FP64 arithmetic")
    if not details:
        return result
    return {"gap": result, "log_p": logp, "p": np.exp(logp),
            "owner": np.asarray(all_owner).reshape(*batch, probes),
            "weights": np.asarray(all_weights).reshape(*batch, count + 1, probes),
            "reconstruction": np.asarray(all_reconstructed).reshape(*batch, rows, probes)}


def quadrature_partition_gap_numpy(logits, selected, singleton_index, *, measures,
                                    return_details=False):
    """FP64 per-candidate min KL(a||p) for the final disjoint C+SQ family.

The bin distribution is p_i=softmax(z_i+log(mu)). Bulk conditional weights
and BIN entropy give represented mass p_i[SQ]+sum_c exp(w_c log(p_i)+H(w_c)).
Positive measures model the original-key multiplicity; a plain 17-way
uniform softmax is not the width80 quadrature. No negative gap is clipped.
"""
    return _numpy_gap(*_numpy_inputs(logits, selected, singleton_index, measures),
                      details=return_details)


def anchor_gap_guard_numpy(logits, incumbent, singleton_index, *, measures):
    """Return real candidate slots and all three raw FP64 candidate-gap vectors.

Inputs: logits [...,R,P], incumbent [...,4], observed singleton [...],
measures [...,P]. Broadcasting over the logit batch is supported. The
incumbent is supplied by the existing Anchor-Fisher rule, never recomputed.
"""
    import numpy as np

    logp, base, sq = _numpy_inputs(logits, incumbent, singleton_index, measures)
    if base.shape[-1] != 4:
        raise ValueError("exactly four distinct incumbent slots are required")
    prefix = base[..., :3]
    prefix_gap = _numpy_gap(logp, prefix, sq)
    used = (np.arange(logp.shape[-2]) == prefix[..., :, None]).any(axis=-2)
    challenger = np.where(used, -np.inf, prefix_gap).argmax(axis=-1)
    trial = np.concatenate((prefix, challenger[..., None]), axis=-1)
    base_gap, trial_gap = _numpy_gap(logp, base, sq), _numpy_gap(logp, trial, sq)
    base_mean, trial_mean = base_gap.mean(axis=-1), trial_gap.mean(axis=-1)
    # An unchanged ordered tuple cannot strictly improve its own objective,
    # even if two floating-point evaluations differ. Keep the raw gap audits.
    accepted = (challenger != base[..., 3]) & (trial_mean < base_mean)
    output = np.where(accepted[..., None], trial, base)
    return {"seeds": output, "incumbent": base.copy(), "challenger": challenger,
            "accepted": accepted, "prefix_gap": prefix_gap, "base_gap": base_gap,
            "trial_gap": trial_gap, "base_mean": base_mean, "trial_mean": trial_mean}


def _torch_inputs(logits, selected, singleton_index, measures):
    import torch

    if not logits.is_floating_point():
        raise TypeError("logits must be floating-point")
    if logits.ndim < 2 or min(logits.shape) < 1 or logits.shape[-1] < 2:
        raise ValueError("require nonempty [...,candidates,probes>=2] logits")
    z = logits.double()
    if not bool(torch.isfinite(z).all()):
        raise ValueError("logits must be finite")
    batch, (rows, probes) = z.shape[:-2], z.shape[-2:]
    slots = torch.as_tensor(selected, device=z.device)
    if (slots.ndim < 1 or slots.is_floating_point() or slots.is_complex()
            or slots.dtype == torch.bool or not 1 <= slots.shape[-1] <= rows):
        raise ValueError("selected must contain integer candidate slots")
    try:
        slots = torch.broadcast_to(slots.long(), (*batch, slots.shape[-1]))
    except RuntimeError as error:
        raise ValueError("selected does not broadcast to the logit batch") from error
    if bool(((slots < 0) | (slots >= rows)).any()) or bool((slots.sort(-1).values.diff(dim=-1) == 0).any()):
        raise ValueError("selected slots must be distinct and in range")
    sq = torch.as_tensor(singleton_index, device=z.device)
    if sq.is_floating_point() or sq.is_complex() or sq.dtype == torch.bool:
        raise ValueError("singleton_index must contain explicit integers")
    try:
        sq = torch.broadcast_to(sq.long(), batch)
    except RuntimeError as error:
        raise ValueError("singleton_index does not broadcast to the batch") from error
    if bool(((sq < 0) | (sq >= probes)).any()):
        raise ValueError("an observed singleton index in [0,P) is required")
    raw_measure = torch.as_tensor(measures, device=z.device)
    if raw_measure.is_complex() or raw_measure.dtype == torch.bool:
        raise ValueError("measures must be real, finite and positive")
    # Python floating lists contain double-precision values. Do not route
    # 79/15 through the default FP32 dtype before promotion to FP64.
    mu = torch.as_tensor(measures, dtype=torch.float64, device=z.device)
    try:
        mu = torch.broadcast_to(mu, (*batch, probes))
    except RuntimeError as error:
        raise ValueError("measures do not broadcast to [...,P]") from error
    if not bool(torch.isfinite(mu).all()) or bool((mu <= 0).any()):
        raise ValueError("measures must be finite and strictly positive")
    if bool((mu.gather(-1, sq[..., None]) != 1.0).any()):
        raise ValueError("the explicit singleton must have unit measure")
    weighted = z - z.amax(dim=-1, keepdim=True) + mu.log()[..., None, :]
    logp = torch.log_softmax(weighted, dim=-1)
    if not bool(torch.isfinite(logp).all()):
        raise ValueError("logit dynamic range exceeds FP64 arithmetic")
    return logp, slots, sq


def _torch_gap(logp, selected, singleton, details=False):
    import torch

    batch, (rows, probes) = logp.shape[:-2], logp.shape[-2:]
    count = selected.shape[-1]
    lp, slots, sq = logp.reshape(-1, rows, probes), selected.reshape(-1, count), singleton.reshape(-1)
    bank_count = lp.shape[0]
    seed_lp = lp.gather(1, slots[..., None].expand(-1, -1, probes))
    owner = seed_lp.argmax(dim=1)
    columns = torch.arange(probes, device=lp.device)
    bulk = columns[None] != sq[:, None]
    cells = (owner[:, None] == torch.arange(count, device=lp.device)[None, :, None]) & bulk[:, None]
    active = cells.any(dim=-1)
    masked = seed_lp.masked_fill(~cells, -float("inf"))
    norm = torch.logsumexp(masked, dim=-1, keepdim=True)
    logw = (seed_lp - torch.where(active[..., None], norm, 0.0)).masked_fill(~cells, -float("inf"))
    weights = logw.exp()
    entropy = -(weights * torch.where(cells, logw, 0.0)).sum(dim=-1)
    component = torch.bmm(lp, weights.transpose(1, 2)) + entropy[:, None]
    component = component.masked_fill(~active[:, None], -float("inf"))
    sq_logp = lp.gather(-1, sq[:, None, None].expand(-1, rows, 1))
    component = torch.cat((component, sq_logp), dim=-1)
    represented = torch.logsumexp(component, dim=-1)
    gap = -represented.reshape(*batch, rows)
    if not bool(torch.isfinite(gap).all()):
        raise ValueError("partition gap exceeds finite FP64 arithmetic")
    if not details:
        return gap
    sq_weight = (columns[None] == sq[:, None]).to(torch.float64)[:, None]
    weights = torch.cat((weights, sq_weight), dim=1)
    reconstruction = torch.bmm(torch.softmax(component, dim=-1), weights)
    owner = torch.where(bulk, owner, count)
    return {"gap": gap, "log_p": logp, "p": logp.exp(),
            "owner": owner.reshape(*batch, probes),
            "weights": weights.reshape(*batch, count + 1, probes),
            "reconstruction": reconstruction.reshape(*batch, rows, probes)}


def quadrature_partition_gap(logits, selected, singleton_index, *, measures,
                              return_details=False):
    """Independent vectorized PyTorch FP64 counterpart of the NumPy metric."""
    return _torch_gap(*_torch_inputs(logits, selected, singleton_index, measures),
                      details=return_details)


def anchor_gap_guard(logits, incumbent, singleton_index, *, measures):
    """PyTorch FP64 two-choice reference; see anchor_gap_guard_numpy contract."""
    import torch

    logp, base, sq = _torch_inputs(logits, incumbent, singleton_index, measures)
    if base.shape[-1] != 4:
        raise ValueError("exactly four distinct incumbent slots are required")
    prefix = base[..., :3]
    prefix_gap = _torch_gap(logp, prefix, sq)
    used = (torch.arange(logp.shape[-2], device=logp.device) == prefix[..., :, None]).any(dim=-2)
    challenger = prefix_gap.masked_fill(used, -float("inf")).argmax(dim=-1)
    trial = torch.cat((prefix, challenger[..., None]), dim=-1)
    base_gap, trial_gap = _torch_gap(logp, base, sq), _torch_gap(logp, trial, sq)
    base_mean, trial_mean = base_gap.mean(dim=-1), trial_gap.mean(dim=-1)
    # Exact tuple identity, not a tolerance or an approximate profile test.
    # Preserve the unmodified strict comparison for every different tuple.
    accepted = (challenger != base[..., 3]) & (trial_mean < base_mean)
    output = torch.where(accepted[..., None], trial, base)
    return {"seeds": output, "incumbent": base.clone(), "challenger": challenger,
            "accepted": accepted, "prefix_gap": prefix_gap, "base_gap": base_gap,
            "trial_gap": trial_gap, "base_mean": base_mean, "trial_mean": trial_mean}


__all__ = ["uniform16_sq1_quadrature", "quadrature_partition_gap_numpy",
           "quadrature_partition_gap", "anchor_gap_guard_numpy", "anchor_gap_guard"]
