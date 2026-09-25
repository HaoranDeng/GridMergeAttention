"""RF0 real-query GS seeds using the candidate-average categorical Fisher.

G = mean_r(diag(p_r) - p_r p_r.T), p_r = softmax(z_r), uses each of the
existing candidates equally. It is NOT Fisher at mean(p) or softmax(mean(z)):
G = Fisher(mean(p)) - Cov_r(p_r). This changes curvature, not the ordinary
largest-residual pivot criterion. No mass normalizer, fitting, mean-query
insertion, hard-partition trial, KV merge, or route is performed here.

The reference forms only P-by-P metric and R-by-R residual Gram matrices,
never the equivalent R-by-(R*P) stacked features. All geometry is FP64.
The empirical average is a local surrogate, not a global KL/hard-partition
bound, video-quality result, or speed claim. Only real row indices are output.
"""

METHOD = "fisher_expected"
GS_EPSILON = 1e-12


def _torch_logits(z):
    import torch

    if (z.ndim != 3 or z.shape[0] < 1 or z.shape[1] < 4 or z.shape[2] < 1):
        raise ValueError("require logits [banks, candidates>=4, probes>=1]")
    if not z.is_floating_point():
        raise TypeError("logits must be floating-point")
    if not bool(torch.isfinite(z).all()):
        raise ValueError("logits must be finite")
    return z.double()


def _torch_anchor(anchor, z):
    import torch

    anchor = torch.as_tensor(anchor, device=z.device)
    if (anchor.shape != (z.shape[0],) or anchor.is_floating_point()
            or anchor.is_complex() or anchor.dtype == torch.bool):
        raise ValueError("one integer anchor candidate slot per bank required")
    anchor = anchor.long()
    if bool(((anchor < 0) | (anchor >= z.shape[1])).any()):
        raise ValueError("anchor outside candidate bank")
    return anchor


def expected_fisher_metric(z):
    """Return [banks,P,P] FP64 mean Hessian, not Hessian of the mean profile."""
    import torch

    z = _torch_logits(z)
    p = torch.softmax(z, dim=-1)
    metric = torch.diag_embed(p.mean(dim=1))
    metric = metric - torch.bmm(p.transpose(1, 2), p) / z.shape[1]
    return (metric + metric.transpose(1, 2)) * 0.5


def expected_fisher_gram(z, anchor):
    """Return delta_i.T G delta_j in FP64 for each bank.

    Remove a constant logit gauge from each delta before the products. This
    is mathematically invisible because every categorical Fisher kills ones,
    and avoids amplifying cancellation from irrelevant row-wise offsets.
    """
    import torch

    z = _torch_logits(z)
    anchor = _torch_anchor(anchor, z)
    batch = torch.arange(z.shape[0], device=z.device)
    delta = z - z[batch, anchor, None, :]
    delta = delta - delta[..., :1]
    gram = torch.bmm(torch.bmm(delta, expected_fisher_metric(z)), delta.transpose(1, 2))
    return (gram + gram.transpose(1, 2)) * 0.5


def _gram_indices(gram, anchor):
    """Three max-diagonal residual pivots, anchor first and never repeated.

    If d=R_pp and v=R[:,p], the existing safeguarded feature GS operation is
    x <- x - <x,x_p> x_p/(d+eps). Its EXACT residual-Gram counterpart is
    R <- R - vv.T*(d+2eps)/(d+eps)^2, not R-vv.T/(d+eps).
    For eps=0 this is the standard pivoted-Cholesky rank-one update.
    Roundoff-negative residual diagonals are clamped only for scoring/norms;
    no tolerance-based change to the largest-residual criterion is applied.
    Exact score ties choose the first unused candidate slot.
    """
    import torch

    residual = gram.clone()
    batch = torch.arange(gram.shape[0], device=gram.device)
    used = torch.arange(gram.shape[1], device=gram.device)[None] == anchor[:, None]
    selected = [anchor]
    for step in range(3):
        score = residual.diagonal(dim1=-2, dim2=-1).clamp_min(0)
        pick = score.masked_fill(used, -float("inf")).argmax(dim=-1)
        selected.append(pick)
        used[batch, pick] = True
        if step < 2:
            norm = score[batch, pick]
            column = residual[batch, :, pick]
            factor = (norm + 2 * GS_EPSILON) / (norm + GS_EPSILON).square()
            residual = residual - column[:, :, None] * column[:, None, :] * factor[:, None, None]
    return torch.stack(selected, dim=-1)


def expected_fisher_indices(z, anchor):
    """Return four distinct real-candidate slots [banks,4], anchor first."""
    z = _torch_logits(z)
    anchor = _torch_anchor(anchor, z)
    return _gram_indices(expected_fisher_gram(z, anchor), anchor)


def _numpy_logits(z):
    import numpy as np

    z = np.asarray(z)
    if z.ndim != 3 or z.shape[0] < 1 or z.shape[1] < 4 or z.shape[2] < 1:
        raise ValueError("require logits [banks, candidates>=4, probes>=1]")
    if z.dtype.kind != "f":
        raise TypeError("logits must be floating-point")
    if not np.isfinite(z).all():
        raise ValueError("logits must be finite")
    return z.astype(np.float64)


def _numpy_anchor(anchor, z):
    import numpy as np

    anchor = np.asarray(anchor)
    if anchor.shape != (len(z),) or anchor.dtype.kind not in "iu":
        raise ValueError("one integer anchor candidate slot per bank required")
    if ((anchor < 0) | (anchor >= z.shape[1])).any():
        raise ValueError("anchor outside candidate bank")
    return anchor.astype(np.int64)


def expected_fisher_metric_numpy(z):
    """Independent NumPy FP64 reference, explicitly average small Hessians."""
    import numpy as np

    z = _numpy_logits(z)
    metrics = []
    for bank in z:
        p = np.exp(bank - bank.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        metrics.append(sum(np.diag(row) - np.outer(row, row) for row in p) / len(p))
    return np.asarray(metrics)


def expected_fisher_gram_numpy(z, anchor):
    import numpy as np

    z = _numpy_logits(z)
    anchor = _numpy_anchor(anchor, z)
    metric = expected_fisher_metric_numpy(z)
    result = []
    for bank, first, g in zip(z, anchor, metric):
        delta = bank - bank[first]
        delta -= delta[:, :1]
        gram = delta @ g @ delta.T
        result.append((gram + gram.T) * 0.5)
    return np.asarray(result)


def expected_fisher_indices_numpy(z, anchor):
    """NumPy four-real-slot reference with deterministic max-residual pivots."""
    import numpy as np

    z = _numpy_logits(z)
    anchor = _numpy_anchor(anchor, z)
    output = []
    for gram, first in zip(expected_fisher_gram_numpy(z, anchor), anchor):
        chosen = [int(first)]
        residual = gram.copy()
        for step in range(3):
            score = np.maximum(np.diag(residual), 0).copy()
            score[chosen] = -np.inf
            pick = int(score.argmax())
            chosen.append(pick)
            if step < 2:
                norm = score[pick]
                column = residual[:, pick].copy()
                coefficient = (norm + 2 * GS_EPSILON) / (norm + GS_EPSILON) ** 2
                residual -= np.outer(column, column) * coefficient
        output.append(chosen)
    return np.asarray(output, dtype=np.int64)


def select(q, k, *, bank_chunk=512):
    """Return {fisher_expected: uint8 [B,1,F,45,80,H,4]} real-row table.

    Original BF16 Q/K have shape [B,frames*3600,H,D]. Read RF0 Q only and
    every key frame. C32/g45 and Uniform16 columns 0,5,...,75 are fixed.
    BF16 operands are cast to FP32 for dot products; scaling, probabilities,
    average Fisher and GS are FP64. This is not a true FP64-QK reference.
    Each current frame must reload its original Q at the returned rows.
    """
    import torch
    from attention.adaptive_seed_pytorch import multiscale_candidate_table

    if not isinstance(bank_chunk, int) or isinstance(bank_chunk, bool) or bank_chunk < 1:
        raise ValueError("bank_chunk must be a positive integer")
    if q.ndim != 4 or k.ndim != 4 or min(q.shape) < 1 or min(k.shape) < 1:
        raise ValueError("Q/K must be nonempty [B,frames*3600,H,D]")
    if (q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]
            or q.shape[1] % 3600 or k.shape[1] % 3600):
        raise ValueError("Q/K must share batch/head/dim and complete 45x80 frames")
    if q.device != k.device or q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise ValueError("Q/K must be original BF16 on the same device")
    if q.is_cuda and torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("disable TF32 to preserve FP32 dot products")
    batches, _, heads, dim = q.shape
    frames = k.shape[1] // 3600
    table, anchors = multiscale_candidate_table(45, 32, q.device, geometry_count=45)
    total = batches * frames * 45 * 80 * heads
    output = torch.empty((total, 4), device=q.device, dtype=torch.uint8)
    probes = torch.arange(0, 80, 5, device=q.device)
    for start in range(0, total, bank_chunk):
        ids = torch.arange(start, min(start + bank_chunk, total), device=q.device)
        head = ids % heads
        column = (ids // heads) % 80
        row = (ids // (heads * 80)) % 45
        frame = (ids // (heads * 80 * 45)) % frames
        batch = ids // (heads * 80 * 45 * frames)
        real = table[row].long()
        qq = q[batch[:, None], real * 80 + column[:, None], head[:, None]].float()
        kk = k[batch[:, None], frame[:, None] * 3600 + row[:, None] * 80 + probes,
               head[:, None]].float()
        z = torch.bmm(qq, kk.transpose(1, 2)).double() * (dim ** -0.5)
        slots = expected_fisher_indices(z, anchors[row])
        output[start:start + len(ids)] = real.gather(1, slots).to(torch.uint8)
    return {METHOD: output.reshape(batches, 1, frames, 45, 80, heads, 4)}


__all__ = ["METHOD", "expected_fisher_metric", "expected_fisher_gram",
           "expected_fisher_indices", "expected_fisher_metric_numpy",
           "expected_fisher_gram_numpy", "expected_fisher_indices_numpy", "select"]
