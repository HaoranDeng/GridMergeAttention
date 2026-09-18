"""CPU FP64 references and index witnesses for query-W/key-T GCA."""
from __future__ import annotations

import numpy as np


def uniform_probes(length=21, count=16):
    if length < count or count < 1:
        raise ValueError("require distinct valid probes")
    return np.arange(count, dtype=np.int64) * length // count


def candidate_table(rows=80, count=32, geometry=45):
    """Literal counterpart of the unchanged production W80 candidate table."""
    if not 4 <= count <= geometry <= rows:
        raise ValueError("invalid candidate geometry")
    table, anchors = [], []
    radius, uniforms = max(1, geometry // 4), max(4, geometry // 4)
    for anchor in range(rows):
        chosen = {anchor}
        chosen.update(range(max(0, anchor - radius), min(rows, anchor + radius + 1)))
        chosen.update(round(i * (rows - 1) / (uniforms - 1)) for i in range(uniforms))
        if len(chosen) > count:
            raise ValueError("mandatory geometry exceeds C32")
        while len(chosen) < count:
            remaining = [i for i in range(rows) if i not in chosen]
            chosen.add(max(remaining, key=lambda i: (min(abs(i - j) for j in chosen), -i)))
        selected = sorted(chosen)
        table.append(selected)
        anchors.append(selected.index(anchor))
    return np.asarray(table, dtype=np.int32), np.asarray(anchors, dtype=np.int32)


def softmax(z, axis=-1):
    z = np.asarray(z, dtype=np.float64)
    exponential = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return exponential / exponential.sum(axis=axis, keepdims=True)


def fisher_seeds(logits, candidates, anchor_slot):
    """Anchor-Fisher metric and three GS pivots, with leftmost unused ties."""
    z = np.asarray(logits, dtype=np.float64)
    candidates = np.asarray(candidates)
    if z.shape != (len(candidates), 16):
        raise ValueError("expected [candidate,Uniform16] logits")
    rho = softmax(z[anchor_slot])
    delta = z - z[anchor_slot]
    residual = (delta - (delta * rho).sum(-1, keepdims=True)) * np.sqrt(rho)
    used = np.zeros(len(candidates), dtype=bool)
    used[anchor_slot] = True
    selected, score_history = [anchor_slot], []
    for step in range(1, 4):
        score = (residual * residual).sum(-1)
        score_history.append(score.copy())
        pivot = int(np.argmax(np.where(used, -np.inf, score)))
        selected.append(pivot)
        used[pivot] = True
        if step < 3:
            basis = residual[pivot] / np.sqrt((residual[pivot] ** 2).sum() + 1e-12)
            residual -= (residual * basis).sum(-1, keepdims=True) * basis
    return candidates[selected], np.stack(score_history)


def stable_bulk(logits, singleton):
    """Full T21 log-softmax ownership BEFORE removing SQ1 at query time.

    Return normalized bulk weights, negative entropy bias, and full-support
    owners. Empty bulks have zero weights and positive infinite route bias.
    """
    z = np.asarray(logits, dtype=np.float64)
    if z.shape != (4, 21) or not 0 <= singleton < 21:
        raise ValueError("expected four real seed logits over exactly21 temporal keys")
    maximum = z.max(-1, keepdims=True)
    logp = z - maximum - np.log(np.exp(z - maximum).sum(-1, keepdims=True))
    owner = logp.argmax(axis=0)
    weights = np.zeros_like(z)
    bias = np.full(4, np.inf)
    for slot in range(4):
        members = (owner == slot) & (np.arange(21) != singleton)
        if members.any():
            weights[slot, members] = softmax(z[slot, members])
            positive = weights[slot, members]
            bias[slot] = (positive * np.log(positive)).sum()
    return weights, bias, owner


def original_to_hwt(x, height=45):
    """[B,heads,T21*H*W80,D] -> [B,H,W80,T21,heads,D]."""
    if x.ndim != 4 or x.shape[2] != 21 * height * 80 or height < 1:
        raise ValueError("invalid original T21/H/W80 shape")
    batch, heads, _, dim = x.shape
    return x.reshape(batch, heads, 21, height, 80, dim).transpose(0, 3, 4, 2, 1, 5)


def hwt_to_original(x):
    if x.ndim != 6 or x.shape[2:4] != (80, 21):
        raise ValueError("expected HWT with W80/T21")
    batch, height, width, frames, heads, dim = x.shape
    return x.transpose(0, 4, 3, 1, 2, 5).reshape(batch, heads, frames * height * width, dim)


def physical_rf0(hwt):
    """Physical time zero, [B,1,W80,Hq,heads,D]; Hq is not reference time."""
    if hwt.ndim != 6 or hwt.shape[2:4] != (80, 21):
        raise ValueError("expected HWT with W80/T21")
    return hwt[:, :, :, 0].transpose(0, 2, 1, 3, 4)[:, None]


def rf0_seed_shape(batch=1, key_height=45, query_height=45, heads=40):
    """Compact width indices independent of query time; uint8 in native code."""
    if min(batch, key_height, query_height, heads) < 1:
        raise ValueError("invalid seed dimensions")
    return batch, 1, key_height, 80, query_height, heads, 4
