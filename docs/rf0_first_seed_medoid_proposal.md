# RF0 first-seed reverse-KL medoid proposal

Status: isolated experimental implementation, 2026-09-04. **No GPU or video
result yet.** `attention/partition_mass_medoid_seed_triton.py` is not connected
to production modes. Ten CPU math/static/reference tests passed in the
CUDA-disabled `rf0_medoid_cpu_validation_20260904m/source` staging copy. They
check KL direction, finite underflow handling, deterministic distinct choices,
and unchanged later greedy steps; they do not compile or validate the GPU kernel.
This is not the current 20-video candidate: the current RF0 partition selector
still fixes its first seed to the matched key row. No C4 optimality, PSNR
improvement, or equal-runtime guarantee is claimed. This note does not change
production defaults, historical RF10 results, or the paper.

## One bounded candidate

Keep RF0, the existing C16/geometry-16 candidate bank, Uniform-16 local probes,
G256 global-normalizer probes, and four distinct real query-row seeds. Replace
only the forced first seed with the weighted reverse-KL medoid below. Select
the remaining three seeds with the existing partition-deficit greedy rule,
unchanged. The original matched row remains eligible but is no longer required.

Keep SQ1, hard ownership, entropy merge, temporal row-index reuse, and full
compressed attention unchanged. Do not insert a mean/synthetic query, change
the probe support to exclude the singleton, or add a dense rescue. Any future
experiment would use this same training-free operator in every layer and call.

## Exact fixed-cell statement

For one fixed nonempty probe cell, let the M candidate distributions be

\[
z_i=K_Rq_i/\sqrt d,\qquad p_i=\operatorname{softmax}(z_i),
\qquad i=1,\ldots,M.
\]

Assume finite logits. Take the existing nonnegative block-mass importance
weights rho_i with positive total weight, normalize pi_i = rho_i / sum_i rho_i,
and hold them fixed during
first-seed selection. In the current kernel, rho is proportional to
exp(local LSE - global-probe LSE); this is an estimated importance weight,
not an exact global compressed-attention route probability.

With natural logarithms, define

\[
J(w;z)=w^Tz+H(w),\qquad
\mu_j=\sum_i\pi_i\log p_{ij},\qquad
D_s=\sum_jp_{sj}(\log p_{sj}-\mu_j).
\]

Then, exactly,

\[
D_s=\sum_i\pi_i\operatorname{KL}(p_s\|p_i)
=\sum_i\pi_i\operatorname{LSE}(z_i)-J(p_s;\bar z),
\qquad \bar z=\sum_i\pi_i z_i.
\]

Thus s_star = argmin_s D_s is the best available **real candidate** for this
weighted one-cell entropy-regularized approximation. Because the original
anchor is in the same bank, D_s_star <= D_anchor. Ties require a deterministic
rule; this mathematical comparison does not depend on the chosen tie rule.

The unrestricted optimal distribution is

\[
r=\operatorname{softmax}(\mu)
=\operatorname{softmax}(K_R\bar q/\sqrt d),\qquad
\bar q=\sum_i\pi_iq_i.
\]

Since D_s = KL(p_s || r) - log(sum_j exp(mu_j)), the discrete choice is also
the candidate nearest to this barycenter in **KL(p_s || r)**. The direction
matters: KL(r || p_s) is a different criterion. Ordinary Euclidean proximity
to the mean query, or norm-preserving rescaling of that mean, is not equivalent.
The mean is only an interpretation of the score; it is not inserted into C4.

The current kernel uses base-2 log probabilities. The corresponding D_s is
scaled by 1/ln(2), which leaves its argmin unchanged. Its existing importance
weights use exp2 with base-2 local/global normalizers; those weights should not
be changed when using the equivalent base-2 score.

## Relation to the current partition criterion, and limits

With only seed s and the whole probe cell, the current represented target mass
for candidate i is exp(-KL(p_s || p_i)). Hence the weighted missing mass obeys

\[
\sum_i\pi_i\bigl[1-e^{-\operatorname{KL}(p_s\|p_i)}\bigr]
\le D_s.
\]

The proposal minimizes this upper bound, not the missing mass itself. Decreasing
this upper bound does not guarantee a lower actual missing-mass value. In base-2
units the right side is ln(2) times the base-2 KL score.

More importantly, the one-cell result does not solve C4. For fixed cells, the
full restricted objective contains query-dependent route weights. Its component
barycenter update uses effective weights pi_i * beta_ic with beta held fixed;
the current mass-probe proxy is not that exact update. Adding seeds also changes
ownership and conditional component distributions. The feasible distribution
families need not be nested, so neither the later greedy steps nor the final
C4 objective inherit monotonicity from the first-seed result. See
[the fixed-partition derivation and counterexample](seed_barycenter_limits.md).

An exhaustive optimizer over four unrestricted real seeds would have a feasible
set containing the anchored four-seed choices. That abstract optimum cannot be
worse for the same objective. This greedy initialization is not an exhaustive
optimizer and does not inherit that guarantee.

Additional limitations are the sampled candidate population, P16/G256 estimates,
the unchanged SQ1 versus full-probe surrogate, and reusing RF0-selected row
indices at other frames. None of these gives an all-frame or final-video quality
guarantee. Fixed-QKV gains alone would not justify promotion.

## Cost expectation, not a timing result

The weighted mean log profile and all D_s scores require O(MP) elementwise work
and reductions (M=16, P=16), not an M-by-M pairwise KL matrix. The required QK
logits, log probabilities, and importance weights already exist. Probabilities
can be recovered from the existing local-LSE exponentials and denominators if
retained and reused. No extra QK products, query/key probe loads, candidate rows,
compressed cells, or global-normalizer pass are mathematically needed.

Keeping/reusing those temporaries may increase live registers, reductions, or
other instruction costs. A compiler may also change occupancy or scheduling.
Therefore small overhead is only a hypothesis: no equal-speed or zero-overhead
claim is made. Any future speed comparison must time the complete operator on
matched QKV with a clean GPU, separately from video generation timing.

## Why the current SQ1 path does not require the anchor

Read-only code audit, line numbers as inspected on 2026-09-04:

- `attention/partition_mass_seed_triton.py:117` explicitly initializes `used`
  from `anchor_index`; `attention/entropy_mass_seed_triton.py:226` does the
  analogous initialization. These are selector decisions.
- `attention/hard_c4_wan21_14b_720p_triton.py:1412` constructs the native external
  seed-table offsets for all four slots; line 1428 loads all four row ids.
  The merge uses those row ids to load actual queries and computes ownership
  across all four seeds. Slot 0 is not overwritten with the matched key row.
- In the same merge kernel, line 1773 enters the query-column singleton policy;
  rank 0 selects `global_query_cols` at line 1791. The singleton is removed
  from its dynamically computed owning cluster, with per-cluster residual-mass
  handling around line 1899. SQ1 does not require ownership by seed 0.
- The temporal-row-map branch around line 1537 preserves structural slots 0/1,
  but the active SQ1 dispatch in
  `attention/singleton_c4_wan21_14b_720p_triton.py:1487` explicitly passes
  `USE_TEMPORAL_ROW_MAP=False`. This audit is not permission to generalize to
  that inactive branch or every historical merge implementation.

Downstream routing consumes merged components and biases; the inspected active
path has no matched-row requirement. Empty components and ownership tie breaking
remain generic. Existing production verifiers that assert seed0 == row encode
the current anchored-method contract. A separate candidate would need a separate
verification policy preserving shape, range, distinctness, and output checks;
the existing anchored verifier must not be silently weakened.

## Distinguish earlier failures

The bounded audit of [negative results](negative_results.md) found adjacent,
but not identical, experiments:

- **No-Ith Adaptive Greedy Seeds**: removing the anchor from greedy coverage
  worsened the larger screen after an encouraging pilot. This rejects repeating
  that heuristic and cautions against claiming that free anchors are beneficial.
- **Anchor-Free Portfolio And Mean-Query Merge**: anchored/free proposal selection
  used a pooled value proxy, while the mean-query variant modified merge logits.
  Both differ from choosing one existing row with the objective above.
- **Owner-Preserving Conditional Medoid Seed Refinement**: `gs_medoid_256` kept
  the anchor and refined the other three slots under fixed-owner conditions.
  It helped one capture and hurt another; it was not promoted to 20 videos.

No exactly matching RF0 C16 weighted reverse-KL first-seed initialization followed
by unchanged partition-deficit greedy selection was found in that bounded audit.
This is not proof that no such experiment exists elsewhere, nor evidence that
this proposal will succeed. It supplies one specific, falsifiable initialization
criterion while retaining the current candidate unchanged.
