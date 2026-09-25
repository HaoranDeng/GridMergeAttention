# SQ1-aware partition-deficit seed selection

Experimental proposal, 2026-09-05. This is a correction to the existing
partition-deficit seed heuristic, not a new greedy algorithm family, an
optimal C4 solution or evidence of improved video PSNR. The paper and
production defaults are unchanged. Native parity, full-output BF16
screening and canonical video evaluation are required before promotion.

## Score the represented family after singleton extraction

Fix one query column, one key block, a real-query candidate bank and the
existing first-seed anchor. Let `j0` denote the separately represented SQ1
key and `p_i = softmax(z_i)` the candidate's full-block distribution. For a
selected set of seeds `S`, first form the same hard ownership as merge:

    owner(j) = argmax_{s in S} log p_s(j).

Ties favor the earlier selected seed. Only then remove `j0` from its owned
cell. Write `C_s` for the remaining cell and `w_s = p_s(. | C_s)` for its
conditional merge weights. Empty cells contribute nothing. In exact
arithmetic the represented fraction of target candidate `i`'s partition
function is

    r_i(S) = p_i(j0) + sum_s exp(<w_s, log p_i> + H(w_s)).
    epsilon_i(S) = -log r_i(S).
    delta_i(S) = 1 - r_i(S).

The sum uses only nonempty bulk cells. Because the cells and singleton are
disjoint, this is the exact ideal variational lower bound for that fixed
family: `0 <= delta_i < 1` for finite logits. Equivalently, each bulk term is
the target cell mass times `exp(-KL(w_s || p_i(. | C_s)))`. The exact singleton
term does not imply an exact final singleton attention weight, since the
compressed global partition function may remain deficient.

Choose each additional real seed by

    next = argmax_{i not already selected} pi_i * delta_i(S),

where `pi_i` is uniform in a diagnostic or the existing nonnegative
G256-based block-importance estimate in the proposed native path. No weight
is learned or calibrated. This chooses the currently under-represented
candidate, not the candidate with the best population loss after insertion.
It therefore avoids the candidate-by-candidate lookahead used in the
previously unsuccessful `gap_gain_weighted` experiments. Reassignment can
change all cells after insertion; there is no monotonicity or optimality
guarantee even on the candidate bank.

## Why this differs from the old objectives

The existing `partition_mass_seed_triton.py` computes its deficit before SQ1
is extracted. The existing singleton-complement GS experiment instead masks
the support before normalization and scores a span residual. Neither is the
ownership-before-peel variational deficit above. The distinction matters
even with a fixed anchor.

For example, take singleton key 0 and four real candidate distributions:

    p0 = (.10, .30, .30, .30)  # fixed anchor
    p1 = (.97, .01, .01, .01)
    p2 = (.10, .50, .20, .20)
    p3 = (.10, .20, .50, .20).

The old unpeeled current-deficit rule selects `p1`. Under SQ1, however, the
singleton is already represented and the anchor represents `p1`'s uniform
bulk conditional exactly. The corrected rule gives `p1` zero deficit and
selects `p2` or its symmetric counterpart `p3` (`p2` on an exact score tie).
Floating-point reductions can perturb a mathematical tie. This is a mathematical
counterexample to treating the two scores as equivalent, not an
attention-output or video-quality experiment.

## Uniform16 plus an explicit singleton

The singleton is absent from Uniform16 for 64 of 80 query columns. It
cannot be assigned zero mass just because it was not sampled. Retain the
original Uniform16 columns, remove `j0` from the bulk sample if present, and
reuse or compute its separate QK logit. If `m` sampled bulk columns remain,
set `eta = 79/m`, so `m` is 15 or 16 at width 80. Use

    Zhat_i = exp(z_i(j0)) + eta * sum_{j in sampled bulk} exp(z_i(j)).

This is equivalent to pseudo-bin logits `z_i(j) + log eta` on the bulk
sample and `z_i(j0)` on the singleton. The measure sums to 80. A plain
17-way softmax would give the singleton the wrong relative weight.
Ownership uses this full normalizer before peeling. The same quadrature
also supplies the bulk entropy correction; omitting its `log eta` term is
not equivalent. The shared scale cancels in some final relative importance
scores, but not in the singleton-to-bulk ratio or ownership normalization.

This gives an exact variational gap for the positive-measure quadrature
model, only an approximation to the actual 80-key block. G256 importance
and RF0-only candidate rows introduce further approximations. Native BF16
moments, finite-precision comparisons and the residual-mass cutoff are also
outside the ideal identity. Full-output and video tests must expose these
differences rather than assuming the probe objective predicts PSNR.

## Validation status

The [NumPy reference](../analysis/sq1_partition_seed_reference.py) passes 14
CPU unit tests covering a separately reconstructed variational objective and
KL, integer-measure/repeated-key equivalence, the toy reversal, ownership
before peeling, observed and unobserved singleton quadrature, shifts,
batching, independent per-prefix greedy scoring, empty cells and input guards. These are mathematical/reference
tests, not evidence of native numerical equality.

The [isolated native kernel](../attention/sq1_partition_mass_seed_triton.py)
has no production wiring and is not part of the running Mass BF16 experiment.
Only static checks and formula review have been performed. Its optional
score trace records every candidate's weighted deficit at the actual native
prefix, allowing subsequent score and rank checks against the reference.
No GPU compilation, native parity result, output-MSE result, video metric or
runtime result is claimed. Direct FP32 subtraction in the deficit and the
different singleton/bulk dot reductions must be checked, especially near
ties; tolerances cannot be loosened merely to pass a candidate.

The [native-score verifier](../analysis/verify_sq1_partition_native.py) has
18 additional CPU-only tests. It checks all 32 candidate scores at each of
the three actual selected prefixes, using FP64 products of the original
BF16-valued local, singleton and G256 vectors. The fixed tolerance is
`2e-5 + 1e-4 * abs(reference)` on dimensionless mass-deficit scores, with no
log-base conversion. Rank defects use the same tolerance rule. Both native
owner-reuse paths and audit-on/off row identity must pass independently;
cross-path row identity is descriptive, not a substitute for score checks.
The planned GPU audit covers 600 fixed banks per path plus structural checks
on all 302,400 groups. The 32 combined CPU tests and independent integration
review pass; the verifier has not yet been executed on a GPU.

The isolated [BF16 output-screen runner](../analysis/experiment_sq1_partition_bf16.py)
freezes its complete local source dependency set, admits one idle H200 and
records every five-second process inventory. It requires both parity paths
before evaluating the candidate's unchanged full compressed merge/route.
The candidate always uses the stored-profile path; the owner-reuse path is
only a numerical diagnostic. Hellinger, Mass and Jeffreys controls must
replicate the three frozen L33 BF16 captures. Its 16 CPU tests cover the
admission and failure paths, source freeze, native dispatch and complete
metric aggregation. This prepares a quality screen; no GPU result is claimed.

Related history: [SQ1 first-seed objective](rf0_sq1_seed_objective.md),
[old partition selector](../attention/partition_mass_seed_triton.py),
[old population lookahead](../analysis/partition_variational_selector.py),
and [singleton-complement screen](singleton_complement_seed_selection.md).
