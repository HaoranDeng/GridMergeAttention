"""CPU-only mathematical references for the isolated spatial-axis ablation."""
from __future__ import annotations

import numpy as np


def uniform_probes(length, count=16):
    if length < count or count < 1:
        raise ValueError("require distinct valid probes")
    return np.arange(count, dtype=np.int64) * length // count


def candidate_table(rows=80, count=32, geometry=45):
    """Literal CPU counterpart of the unchanged production candidate table."""
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
    exp = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return exp / exp.sum(axis=axis, keepdims=True)


def fisher_seeds(logits, candidates, anchor_slot):
    """FP64 Anchor-Fisher metric and left-tie-breaking three-step GS."""
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
    """Ownership uses full45 log-softmax BEFORE singleton removal.

    Returns normalized bulk weights, negative entropy bias, and full-support
    ownership. Empty bulks have exactly zero weight and +inf route bias.
    """
    z = np.asarray(logits, dtype=np.float64)
    if z.shape != (4, 45) or not 0 <= singleton < 45:
        raise ValueError("expected four real seed logits over exactly45 keys")
    maximum = z.max(-1, keepdims=True)
    logp = z - maximum - np.log(np.exp(z - maximum).sum(-1, keepdims=True))
    owner = logp.argmax(axis=0)
    weights = np.zeros_like(z)
    bias = np.full(4, np.inf)
    for slot in range(4):
        members = (owner == slot) & (np.arange(45) != singleton)
        if members.any():
            weights[slot, members] = softmax(z[slot, members])
            positive = weights[slot, members]
            bias[slot] = (positive * np.log(positive)).sum()
    return weights, bias, owner


def original_to_twh(x):
    """Numpy indexing witness, [B,heads,THW,D] -> [B,T,W,H,heads,D]."""
    if x.shape[-2] % (45 * 80):
        raise ValueError("invalid original token count")
    b, heads, tokens, d = x.shape
    return x.reshape(b, heads, tokens // (45 * 80), 45, 80, d).transpose(0, 2, 4, 3, 1, 5)


def twh_to_original(x):
    b, frames, width, height, heads, d = x.shape
    return x.transpose(0, 4, 1, 3, 2, 5).reshape(b, heads, frames * height * width, d)
