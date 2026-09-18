# Limits of query-barycenter seed arguments

Mathematical note, 2026-09-04. This is not a paper edit or an implementation
decision. The proposed one-uniform-mean seed plus three real-query seeds has
not been found as an exact tested configuration in the bounded local audit.
Adjacent centroid/mean-logit negative results do not establish its failure.
No experiment or implementation of this configuration is committed here.

## Exact result for one fixed cell

For a nonempty fixed key cell R, finite logits z_i = K_R q_i / sqrt(d), and
nonnegative query weights pi_i summing to one, define
J(w; z) = w^T z + H(w), with w in the probability simplex on R. Then

\[
\sum_i\pi_i\operatorname{KL}(w\|\operatorname{softmax}(z_i))
=\sum_i\pi_i\operatorname{LSE}(z_i)-J(w;\bar z),
\qquad \bar z=\sum_i\pi_i z_i.
\]

Consequently the unique minimizing distribution is

\[
w^\star=\operatorname{softmax}(\bar z),\qquad
\bar q=\sum_i\pi_iq_i,\qquad \bar z=K_R\bar q/\sqrt d.
\]

The query vector need not be unique: key-null directions and constant logit
shifts can represent the same distribution. Uniform pi_i gives the ordinary
query mean. Norm-preserving rescaling generally destroys this exact result.
The result concerns average conditional KL on this cell, not the global
compressed-attention objective with query-dependent mass across cells/blocks.

## Fixed-partition C4 still needs route weights

For fixed disjoint cells and component distributions w_c, the objective is

\[
\Phi(w)=\sum_i\pi_i\log\sum_c\exp(J(w_c;z_i))
=\max_{\{\beta_i\}}\sum_i\pi_i
\left[\sum_c\beta_{ic}J(w_c;z_i)+H(\beta_i)\right].
\]

**Only with beta held fixed**, the w_c subproblem has the closed-form solution

\[
q_c^{\mathrm{new}}=
\frac{\sum_i\pi_i\beta_{ic}q_i}{\sum_i\pi_i\beta_{ic}},
\qquad w_c^{\mathrm{new}}=
\operatorname{softmax}_{R_c}(Kq_c^{\mathrm{new}}/\sqrt d).
\]

Skip zero-demand cells. If beta is the current exact global compressed route,
this conditional update followed by route reoptimization cannot decrease
Phi, provided the cells remain fixed and the update is feasible. In contrast,
substituting beta(w) into the expression gives an implicit fixed-point or
stationarity condition, **not an explicit C4 global optimum**. Uniform means
are justified only when the effective route weights are constant across
queries, or when weighted and uniform means induce identical cell Softmaxes.
Probe-local routes or externally estimated block masses give surrogate updates.

## Adding a seed can decrease the restricted objective

Use three equally weighted queries with logits

\[
z_1=(-1,0,0),\quad z_2=(0,1,0),\quad z_3=(-2,2,2),
\qquad \bar z=(-1,1,2/3).
\]

Initially the sole seed is bar(z), with w = Softmax(bar(z)). Now add seed z_1
and use maximum normalized seed likelihood for hard ownership. The mean seed
owns key {2}; the new seed owns keys {1,3}, with conditional probabilities
(r, 1-r), where r = 1/(1+e). Let h(r) be binary entropy. The new per-query
restricted value is LSE(z_i2, r z_i1 + (1-r) z_i3 + h(r)).

| Mean over the three queries | One mean seed | Mean seed plus z_1 |
| --- | ---: | ---: |
| Restricted objective | 1.616194100049748 | 1.614806064486900 |
| Gap from dense LSE | 0.089040179760051 | 0.090428215322900 |

The objective decreases by 0.001388035562849. Independently rechecked using
70-digit Decimal arithmetic; these are mathematical toy values, not model
measurements. Repeating each query 15 times gives the same 45-query example.

Disjoint support makes the issue precise: the old distribution belongs to the
new structured simplex only if each new component equals its conditional
restriction on that cell. A new seed generally changes those conditionals.
The feasible sets therefore need not be nested. This refutes a general
stepwise-improvement argument, not the empirical usefulness of every mean
anchor or every possible choice of the remaining three seeds.

## Scope

A current-frame mean has the fixed-cell result for that frame. Reusing an
RF10 mean vector for other frames does not inherit it. Repartitioning, probe
restriction, row-index sharing, and final video quality require separate
justification. Four-seed optimality and improved PSNR do not follow from the
one-cell barycenter identity. No decision to implement or run the proposal
has been made in this note.
