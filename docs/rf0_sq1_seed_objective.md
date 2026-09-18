# SQ1 changes the first-seed variational objective

Mathematical note, 2026-09-05. **No candidate implementation, GPU experiment,
or paper edit.** This identifies a limitation of the existing
[full-block first-medoid proposal](rf0_first_seed_medoid_proposal.md), not an
error in its stated one-cell theorem. It supplies no C4 optimum, stepwise
monotonicity, runtime, or PSNR guarantee.

## Exact one-bulk-cell plus singleton gap

Use natural logarithms, finite target logits on a fixed block with at least
two keys, and fixed nonnegative population weights `pi_i` summing to one.
Let `p_i = softmax(z_i)`, let `j0` be the SQ1 key, and define

\[
t_i=p_i(j_0),\qquad R=\{j:j\ne j_0\},\qquad
r_i=p_i(\cdot\mid R),\qquad J(w;z)=w^Tz+H(w).
\]

For one real seed `s`, the bulk distribution is `r_s`; the other component
is the exact singleton. If `Z_i = sum_j exp(z_ij)`, entropy merging gives

\[
\frac{\widetilde Z_i(s)}{Z_i}
=t_i+(1-t_i)\exp[-\mathrm{KL}(r_s\|r_i)],\qquad
\epsilon_i(s)=-\log\left[t_i+(1-t_i)e^{-\mathrm{KL}(r_s\|r_i)}\right].
\]

Proof: the bulk corrected logit is `J(r_s; z_i|R)` and its exponential is
`Z_i (1-t_i) exp(-KL(r_s || r_i))`; the singleton contributes `Z_i t_i`.
The singleton logit is exact, but its final normalized attention weight need
not equal `t_i` because the bulk represented mass can be deficient.

These are ideal exact-arithmetic identities. Native BF16 moments, weight
rounding, and the SQ1 residual-mass cutoff are additional implementation
differences; this note does not claim native numerical equality.

## What the old medoid actually optimizes

The existing isolated implementation
`attention/partition_mass_medoid_seed_triton.py:127` minimizes

\[
D_s=\sum_i\pi_i\mathrm{KL}(p_s\|p_i).
\]

For a single **unpeeled** cell, this is exactly the weighted variational gap.
Over free distributions `w`, its unique minimizer is
`softmax(sum_i pi_i log p_i)`. With linear QK logits this is the distribution
of the weighted mean query. Over the existing real-query bank, the best seed
is instead the candidate minimizing `D_s`; no synthetic mean query is inserted.

After SQ1, however, KL decomposes as

\[
\mathrm{KL}(p_s\|p_i)
=\mathrm{KL}(\operatorname{Bern}(t_s)\|\operatorname{Bern}(t_i))
 +(1-t_s)\mathrm{KL}(r_s\|r_i).
\]

Thus the old score penalizes disagreement in the seed's singleton mass even
though SQ1 already makes that token a separate component. Neither this
binary-mass term nor its seed-dependent bulk multiplier is the exact SQ1
gap above. The old first-seed guarantee `D_best <= D_anchor` remains true,
but it says nothing comparable about the SQ1 gap.

## Independently checked counterexample

Take the first coordinate as the singleton:

\[
p_1=(0.999,0.0009,0.0001),\quad
p_2=(0.001,0.4995,0.4995),\quad \pi=(0.8,0.2).
\]

| Real seed | Old full-cell score `D_s` | Actual one-bulk + SQ1 score `sum_i pi_i epsilon_i(s)` |
| --- | ---: | ---: |
| 1 | 1.3786618666596850 | 0.0735238742622939 |
| 2 | 5.9226048537867841 | 0.0003200640170718 |

The full-cell medoid chooses seed 1; the actual SQ1 objective chooses seed 2.
These are mathematical toy values, **not attention-output or video metrics**.
Recomputed independently with NumPy float64 on 2026-09-05:

```python
import numpy as np
p = np.array([[.999, .0009, .0001], [.001, .4995, .4995]], dtype=np.float64)
pi = np.array([.8, .2])
r = p[:, 1:] / (1 - p[:, :1])
lp, lr = np.log(p), np.log(r)
full_kl = (p[:, None, :] * (lp[:, None, :] - lp[None, :, :])).sum(-1)
bulk_kl = (r[:, None, :] * (lr[:, None, :] - lr[None, :, :])).sum(-1)
gap = -np.logaddexp(np.log(p[:, 0])[None, :],
                    np.log1p(-p[:, 0])[None, :] - bulk_kl)
print(full_kl @ pi, gap @ pi)  # candidate s is the first axis
```

The direct probability-domain expression and the `logaddexp` expression
agreed to maximum absolute error `1.11e-16` in this check.

## Possible SQ1-aware score: unimplemented

A precisely specified future first-seed criterion is

\[
s^\star=\arg\min_{s\ \text{in the real-query bank}}
A_s,\qquad A_s=\sum_i\pi_i\epsilon_i(s).
\]

This would optimize the stated **one-bulk + singleton weighted block gap**
over the available real seeds. It is not a C4 or global-attention optimum.
The corresponding free-distribution problem replaces `r_s` by `w` inside
`A_s`; the outer logarithms prevent applying the old fixed-weight geometric
barycenter formula directly to that exact problem.

For comparison, Jensen's inequality gives the different upper-bound score

\[
A_s\le B_s=\sum_i\pi_i(1-t_i)\mathrm{KL}(r_s\|r_i).
\]

The free-distribution minimizer of **this bound** is the geometric barycenter
of `r_i` weighted by `pi_i(1-t_i)` after normalization. Its real-query medoid
is another discrete approximation; lowering `B_s` does not guarantee lowering
`A_s`. Neither bound minimization nor a synthetic barycenter is implemented here.

The exact score presumes the singleton is in the modeled support. Uniform-16
probes omit it for most query columns; a future approximation must state how
`t_i` and the bulk conditional are obtained, rather than silently treating
an unobserved singleton as zero mass. Current P16/G256 mass weights are also
estimated population importance, not exact global compressed-route weights.

SQ1 itself does not require a matched-row first seed: the active native merge
loads all four supplied row ids and peels the singleton from its dynamic
owner. An anchor guarantees an exact conditional on its own cell for its own
query in ideal arithmetic, not an exact entire block. Historical anchor-free
failures remain relevant cautions; this note does not justify dropping the
anchor from production or weakening its existing verifier.

Adding later seeds changes ownership and conditional distributions, so the
feasible families need not be nested; see [barycenter limits](seed_barycenter_limits.md).
If current Mass BF16 fails the video target, any follow-up still needs matched
BF16 full-output screening, including L33/head30 and all query frames, before
considering 20 videos. No new experiment is authorized or launched by this note.
