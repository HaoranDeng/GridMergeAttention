# Singleton-complement seed selection for Grid Compressed Attention

**Status: rejected after two matched frozen-QKV screens (2026-09-04).**
The singleton-complement mask increased attention-output MSE by 1.2118% and
1.1518% relative to its mass-GS starting point. It should not be promoted or
advanced to a 20-video run on this evidence. The derivation below remains a
valid explanation of the fixed-cell objective, not evidence that this
particular probe/span correction improves the implemented operator. There
is no video-quality or runtime result for this candidate, and the paper
repository is unchanged.

The motivation is specific to the measured `_sq1` operator: every key block
already contributes a query-column singleton, in addition to four
seed-conditioned bulk cells. The singleton key is removed from all bulk
cells and participates in the same global compressed softmax. It is not an
additional dense-attention or rescue computation. Its contribution is exact
and does not depend on which seed rows are selected. Seed selection should
therefore focus on the remaining, imperfectly represented bulk keys.

For the existing mass-selector derivation and measured baseline scope, see
[entropy-mass seed selection](entropy_mass_seed_selection.md). The present
note changes only the proposed selector's scoring support; the existing
hard ownership, conditional merge, entropy correction, and route remain
unchanged.

## Exact global gap with fixed singleton cells

Fix one query frame, query column j, and head. Let i index the query rows,
b index width-aligned key blocks, and z_i collect a query's scaled logits
against all original keys. In each block, let u_b be the key at column j,
and define the bulk complement D_b = b \ {u_b}. For width W=80, D_b has
79 keys.

Define the dense singleton and bulk partition functions by

\[
E_i=\sum_b e^{z_{i,u_b}},
\qquad
Z_{ib}^{-}=\sum_{\ell\in D_b}e^{z_{i\ell}},
\qquad
Z_i=E_i+\sum_b Z_{ib}^{-}.
\]

For any fixed seed-induced partition of the bulk complements, let w_bc be
the normalized conditional distribution of nonempty bulk cell c in block b.
Its entropy-corrected logit and compressed bulk partition function are

\[
\widetilde L_{ibc}
=w_{bc}^{\top}z_i+\mathcal H(w_{bc}),
\qquad
\widetilde Z_{ib}^{-}=\sum_c e^{\widetilde L_{ibc}}.
\]

Every singleton has weight one and entropy zero. Its corrected logit is
exactly z_i,u_b. The complete compressed partition function is consequently

\[
\widetilde Z_i=E_i+\sum_b\widetilde Z_{ib}^{-}.
\]

The bulk cells and singleton cells have disjoint supports. The structured
simplex variational identity therefore gives

\[
\Delta_i
=\log Z_i-\log\widetilde Z_i
=\min_{a\in\mathcal A^{\mathrm{GCA}}}
\operatorname{KL}(a\|\operatorname{softmax}(z_i)).
\]

This identity fixes the induced cells and their conditional distributions;
it does not state that their generating seed set is optimal. It applies
regardless of how the existing ownership rule constructed those cells.

**Proposition.** Define

\[
\epsilon_{ib}^{-}
=\log Z_{ib}^{-}-\log\widetilde Z_{ib}^{-}\geq0,
\qquad
\rho_i^{\mathrm{singleton}}=E_i/Z_i,
\qquad
\rho_{ib}^{-}=Z_{ib}^{-}/Z_i.
\]

Then

\[
\Delta_i
=-\log\left[
\rho_i^{\mathrm{singleton}}
+\sum_b\rho_{ib}^{-}e^{-\epsilon_{ib}^{-}}
\right]
=-\log(1-D_i),
\]

where

\[
D_i=\sum_b\rho_{ib}^{-}(1-e^{-\epsilon_{ib}^{-}}),
\qquad
0\leq\Delta_i\leq\sum_b\rho_{ib}^{-}\epsilon_{ib}^{-}.
\]

For uniformly small bulk gaps,

\[
\Delta_i
=\sum_b\rho_{ib}^{-}\epsilon_{ib}^{-}
+O\!\left(\max_b|\epsilon_{ib}^{-}|^2\right).
\]

*Proof.* Substitute tilde Z_ib^- = Z_ib^- exp(-epsilon_ib^-) into the
compressed partition function and divide by Z_i. The singleton and bulk
masses sum to one; subtracting the exponential deficits gives 1-D_i.
For the bound, apply Jensen's inequality to a distribution with singleton
mass rho_i^singleton at gap zero and bulk masses rho_ib^- at their respective
gaps. Expanding the exponential and logarithm proves the first-order
expression. QED.

Thus the first-order importance factor is the **global attention mass of
the residual bulk**, not the whole block's mass. The singleton contributes
no approximation deficit. It still affects the global normalizer, which
must not be removed from the importance calculation.

## A provably irrelevant source of span novelty

Consider one block with strictly positive normalized candidate distributions

\[
p_i=(u_i,(1-u_i)r),
\qquad 0<u_i<1,
\qquad r\in\Delta_{W-1},\quad r_\ell>0.
\]

The first coordinate is the already-exact singleton. The bulk conditional
r is identical for every candidate; only singleton mass u_i changes.

For any nonempty seed set, existing normalized-likelihood hard ownership
assigns every bulk key to the seed with largest 1-u_s, with a common
deterministic winner in a tie. Its conditional bulk distribution is r;
other bulk cells are empty. Singleton plus this one bulk distribution
already represents every target p_i exactly:

\[
p_i=u_i e_{u_b}+(1-u_i)(0,r).
\]

The block variational gap is therefore zero for every candidate and every
nonempty seed set. Adding seeds cannot improve this block's representation.
Nevertheless, centered square-root features of the full p_i vary with u_i,
so an ordinary span criterion can spend selection capacity distinguishing
these already-represented variations.

A small arithmetic check used r=(0.7,0.2,0.1) and
u=(0.01,0.2,0.8,0.99), with the first candidate as anchor. Full-profile
centered-root residual squared norms after projecting onto that anchor were
approximately (0, 0.048885, 0.276291, 0.371430), despite all actual
singleton-plus-bulk gaps being zero to numerical precision. Conditioning
the profiles on the bulk produces identical features, so their subsequent
span novelty vanishes.

This is a mathematical counterexample to treating all full-profile novelty
as unresolved compression error. It is not an experiment demonstrating a
failure rate or an improvement on real Wan tensors.

## Proposed rule: mass-GS on the singleton complement

Keep the existing Uniform16 probe columns

\[
\mathcal P=\{0,5,10,\ldots,75\}.
\]

For query column j, retain only P_j = |P \ {j}| valid probes. Thus P_j is
15 when j is itself a probe column and 16 otherwise. Given the already
computed candidate logits x_i, form

\[
\ell_{ib}^{-}
=\operatorname{LSE}_{\ell\in\mathcal P\setminus\{j\}}x_{i\ell},
\qquad
\widehat p_{ib\ell}^{-}
=\frac{\mathbf1[\ell\ne j]e^{x_{i\ell}}}
{\sum_{v\in\mathcal P\setminus\{j\}}e^{x_{iv}}}.
\]

Masked positions remain zero in the centered-root feature:

\[
\phi_{i\ell}^{-}
=\mathbf1[\ell\ne j]\left[
\sqrt{\widehat p_{ib\ell}^{-}}
-\frac1{P_j}\sum_{v\in\mathcal P\setminus\{j\}}
\sqrt{\widehat p_{ibv}^{-}}
\right].
\]

The proposed greedy score is the existing mass-GS score on this complement:

\[
s_t=\arg\max_{i\notin S_{t-1}}
\widehat\rho_{ib}^{-}
\|(I-\Pi_{t-1})\phi_i^{-}\|_2^2.
\]

The first seed remains the matched row. The candidate bank, RF10 router,
four total seed rows, and all downstream computations remain unchanged.
This retains the existing span surrogate; it does not turn Gram-Schmidt
into an exact minimizer of the global gap.

### Mass quadrature and candidate-independent factors

Keep the existing G=256 global probe normalizer ell_i^global over all keys,
including singleton positions whenever those global probes encounter them.
Using the same equal-weight quadrature convention gives

\[
\widehat\rho_{ib}^{-}
=\frac{(W-1)/P_j}{N/G}
\exp(\ell_{ib}^{-}-\ell_i^{\mathrm{global}}),
\qquad N=THW.
\]

For one block and query column, W-1, P_j, N, and G are the same for every
candidate row. This prefactor cancels from candidate ranking. The stable
kernel weight can therefore remain

\[
v_i=\exp(d_i-\max_r d_r),
\qquad d_i=\ell_{ib}^{-}-\ell_i^{\mathrm{global}}.
\]

The implementation's base-2 LSE/exp2 convention is equivalent. The factor
must not be silently dropped if reporting an absolute mass estimate or
comparing mass totals across differently sampled groups. Deterministic
probe quadrature is only an approximation; it is not guaranteed unbiased
and can miss large logits.

### Minimal implementation changes

In the current
[mass-GS kernel](../attention/entropy_mass_seed_triton.py), retain the
existing 32-by-16 candidate-probe dot products and change only their scoring:

1. Combine probe validity with `probe_column != query_column` before the
   local LSE and softmax.
2. Normalize and center over the remaining 15 or 16 probes, leaving the
   masked feature entry zero.
3. Use that local LSE with the unchanged global normalizer and existing
   mass-GS iterations and deterministic selection rules.

No extra QK products, key loads, normalization pass, compressed slots, or
learned parameters are required. The global normalizer is not recomputed.
There are 16 affected columns among the width of 80, so **20% of query
columns** can receive different scores; the other 80% have the same
mathematical selector as before. Bitwise parity still requires a numerical
implementation check. The smaller valid-probe count does not change the
stored 16-lane feature shape.

## Matched screening result: no improvement

The reference implementation was screened as `gs_bulk_256` against its
unmasked `gs_256` starting point. Both use RF10, the same 32-row candidate
bank and fixed first seed, Uniform16 local probes, Uniform256 global
normalization, and the unchanged production C4-plus-singleton merge/route.
The screen evaluates all 21 query frames of two frozen holdout QKV captures
on heads 0, 10, 20, and 30 against dense attention.

| Capture | Unmasked mass-GS MSE | Singleton-complement MSE | Change vs mass-GS |
| --- | ---: | ---: | ---: |
| call020 | 0.0053246538154780865 | 0.005389178171753883 | +1.2118% |
| call100 | 0.0072697666473686695 | 0.0073534962721168995 | +1.1518% |

Both masked results remain better than the older RF10 heuristic in this
screen, but are worse than the mass-GS operator they modify; that is the
relevant comparison for judging the mask. These are attention-output MSE
measurements, not generated-video PSNR, SSIM, LPIPS, or latency results.

The exact fixed-cell identity and the irrelevant-novelty example do not
predict improvement on general inputs. The screen does not isolate whether
the degradation comes from fewer valid probes, full-block ownership offsets,
temporal reuse, or the remaining span/output mismatch. No causal explanation
is established by these two captures alone. Preserve this as a bounded
negative result; do not repeat the unchanged mask or promote it into the
default selector.

Implementation: [production-path screening reference](../analysis/screen_partition_population_seeds.py).
Artifact: `C:\GMA\experiments\singleton_complement_screen_20260904c\screen.json`.

## Precise limitations and acceptance status

- **Ownership still uses full-block normalization.** Production assigns
  each key using the selected seeds' normalized probabilities over all
  80 keys, and only then removes the singleton and renormalizes bulk cells.
  Proposed selector profiles are normalized on the probe complement.
  Singleton mass can affect full-block ownership offsets when candidate
  bulk conditionals differ. This proposal does not change that rule and
  cannot claim its feature geometry fully matches the resulting cells.
- **Only one known mismatch is removed.** The exact identities concern the
  actual final cells; P16 scores still approximate them. Excluding one probe
  can increase sampling error, and most query columns already exclude the
  singleton from their probes. This limits the possible scope of the change.
- **Temporal sharing is unchanged.** RF10 chooses row indices that are
  reused across query frames. The current-frame merge still loads real
  current-frame queries. No temporal-average objective or temporal
  optimality is established by this correction.
- **Span novelty remains a surrogate.** Hard repartitioning can change the
  feasible simplex discontinuously. No monotone global-gap, optimal C4
  seed-set, or attention-output-error guarantee follows from the proposed
  score, even with exact bulk probabilities.
- **No video or runtime claim.** Scalar variational gap, fixed-QKV output
  error, and generated-video PSNR/SSIM/LPIPS are different quantities.
  This candidate failed both matched capture screens. It has no 20-video
  result, has not achieved PSNR above 12, and has not verified
  no-slower-than-RF10 runtime.

The matched selector/production-path check is complete and negative; no
further generation or production integration is justified by this result.
The theoretical identity is not evidence that the remaining quality target
is achievable by this mask.
