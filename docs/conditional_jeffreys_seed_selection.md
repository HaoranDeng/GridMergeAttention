# Conditional-Jeffreys seed selection: derivation and RF0 diagnostic

CPU phase completed 2026-09-04. This is an isolated CPU reference experiment, not a
promoted GCA method. Original Mass-GS C32/g45 with RF0 remains the default;
no paper, production selector, merge, routing kernel or generation config
was changed. No video or GPU experiment was run in this CPU phase. The
subsequent [native complete-operator experiment](conditional_jeffreys_native_experiment.md)
tests this same candidate with actual GPU selection, merging and routing.

## Main finding

A partition-aware, mass-weighted conditional-Jeffreys pivot has an exact
connection to the entropy-regularized objective. On the fixed RF0/L33/step0
sample, replacing the Mass-GS residual with this score reduces weighted
local conditional-output MSE by **2.6319%** and weighted reverse-KL gap by
**0.8679%**. The output proxy improves on all three prompts.

This is only a small local diagnostic gain. It does not establish better
complete attention, video quality, or runtime. The tested RF0-only local
metric already ranks Mass above Hellinger, opposite to the earlier severe
all-frame L33 complete-attention regression. Therefore it cannot identify
the cause of that failure or establish that this candidate repairs it.

## Exact variational connection

Use temperature-one logits z (including the attention scale), natural logs,
and H(a) = -a^T log a. The objective and its loss are

\[
 f(a,z)=a^Tz+H(a),\qquad p=\operatorname{softmax}(z),\qquad
 \max_{a\in\Delta}f(a,z)-f(a,z)=\mathrm{KL}(a\Vert p).
\]

Fix disjoint cells R_c and normalized distributions w_c supported on them.
Restrict the approximation to a_j = beta_c w_cj for j in R_c. Entropy's
chain rule gives the compressed logits

\[
 \widetilde L_c=w_c^Tz+H(w_c).
\]

For P_c = p(R_c), p_c = p(.|R_c), define

\[
 d_c=\mathrm{KL}(w_c\Vert p_c),\qquad
 e_c=\mathrm{KL}(p_c\Vert w_c),\qquad
 C=\sum_cP_c e^{-d_c}.
\]

The exact optimum within this restricted family is
beta_c = P_c exp(-d_c)/C and p_hat_j = beta_c w_cj. Consequently,

\[
 \Delta=\mathrm{KL}(\widehat p\Vert p)=-\log C,\qquad
 \mathrm{KL}(p\Vert\widehat p)=\sum_cP_c(d_c+e_c)+\log C.
\]

Adding the two equations produces the useful identity

\[
 \boxed{\Gamma=\sum_cP_c\bigl[\mathrm{KL}(w_c\Vert p_c)
 +\mathrm{KL}(p_c\Vert w_c)\bigr]
 =\mathrm{KL}(\widehat p\Vert p)+\mathrm{KL}(p\Vert\widehat p)
 \;\geq\;\Delta.}
\]

This is an exact identity for the stated fixed-cell family, not a small-error
Taylor approximation. Empty cells can be omitted. A singleton has d_c=e_c=0.
For a complete disjoint block collection, it also decomposes as
Gamma_i = sum_b rho_ib Gamma_ib, where rho_ib is true block attention mass.
The fixed sample below is not a complete collection, so its normalized
weighted means are diagnostics, not global KL values.

The softmax/logsumexp relationship is standard; see
[Gao and Pavel](https://arxiv.org/abs/1704.00805).
The conditional identity above follows directly from the displayed algebra;
no literature-novelty claim is made.

## Proposed score, relative to Mass-GS

Both existing GS methods use centered square-root attention profiles and
their residual squared norms. Hellinger-GS multiplies by a standardized
local-response weight. Mass-GS instead uses a local/global partition ratio
to approximate block attention mass.

The isolated candidate keeps that mass factor but replaces the span residual:

\[
 i_{\mathrm{next}}=\arg\max_{i\notin S}
      \widehat\rho_{ib}\,\widehat\Gamma_{ib}(S).
\]

Gamma is evaluated against the current hard partition on the selection probe
support, induced by the already selected real seed rows, rather than their
linear span. P16 selection does not see full80 ownership. This makes
the score directly sensitive to the approximation used by entropy-aware
merging. It does not imply that hard assignment is universally optimal or
that no soft-assignment formulation could work. Hard cells define the
particular restricted family being analyzed.

For each observed key, the owner is the seed with largest log probability
normalized over the entire supplied support (16 or 80 probes). Only after
this comparison is the matching-column singleton peeled out, if observed.
Final evaluation separately reconstructs ownership over all 80 keys and then
peels SQ1, for every method. The first, index-matched seed stays fixed; three
additional distinct rows are selected from the unchanged production C32/g45
bank. No medoid initialization, population loss, extra dense route, or
layer/head exception is introduced.

One efficient evaluation uses u_ic = w_c^T log p_i + H(w_c):

\[
 \Gamma_i=\sum_jp_{ij}(\log p_{ij}-\log w_{o(j),j})
                 -\sum_cP_{ic}u_{ic}.
\]

This needs cell-mass reductions/products, not an all-pairs candidate matrix.
Nevertheless, partition reconstruction can cost more than GS. This CPU
implementation provides no evidence of acceptable native attention latency.

## Why the identity is not an optimality guarantee

1. The pivot chooses a poorly represented query. It does not evaluate the
   post-insertion population loss of every possible seed.
2. Adding a seed changes ownership. The feasible families need not be nested,
   so neither the reverse gap nor Gamma is guaranteed to decrease.
3. Even a smaller exact Gamma need not mean a smaller reverse gap or output
   error. For p=(0.99,0.01), compare a=(0.999999,0.000001) and
   b=(0.97,0.03). Gamma drops from about 0.09219 to 0.02238, while reverse
   KL rises from about 0.01004 to 0.01316 and TV rises from 0.01000 to 0.02000.
4. Probe-estimated Gamma and mass can disagree with their full-block values.
   Row-shift invariance holds for Mass and this new score when the same
   per-query shift is applied to both local and global logits; this property
   alone does not prove better approximation.
5. A simple variance/Fisher expansion can miss rare logit excursions. With
   w=(1-exp(-A),exp(-A)) and perturbation (0,2A), half its variance tends to
   zero, while log E_w exp(perturbation) - E_w perturbation grows like A.
   Such an expansion needs explicit tail/range assumptions.

## Fixed experiment and metrics

Artifact root:
`C:/GMA/experiments/rf0_conditional_jeffreys_banks_20260904s`.

Inputs are the three audited step-0, layer-33 BF16 captures from
`rf0_layer33_dense_capture_20260904n` and their previously computed exact
all-key normalizers from `rf0_layer33_exact_normalizer_oracle_20260904q`.
No new dense QK normalizer or reference trajectory was generated. Captures
use the canonical first three prompts and raw checkpoint weights.

The fixed sample contains heads 0/10/20/30, query columns
0/1/20/21/40/41/60/61/75/76, key frames 0/10/20 and key rows
0/11/22/33/44: 600 banks per prompt, 1,800 total. Each bank evaluates all
45 query rows from RF0 against all 80 keys, giving 81,000 block/query
observations. These are overlapping deterministic samples, not independent
replicates or a held-out model benchmark.

Saved BF16 Q/K are cast to FP32 for CPU matrix multiplication, then stored
as float64. Metrics, selectors and mathematical entropy-aware projection use
float64. Values retain their original BF16 values after exact FP32 casting.
This is not bitwise parity with Triton selection or FP8 merging.

For each bank/query, the true block mass is rho=Z_block/Z_global. The reported
mean reverse gap is sum(rho*Delta)/sum(rho). The output proxy is

\[
 \mathrm{MSE}_{\mathrm{conditional}}
 =\frac{\sum_{b,i}\rho_{ib}
       \| (\widehat p_{ib}-p_{ib})^T V_b\|_2^2}
       {128\sum_{b,i}\rho_{ib}}.
\]

This is deliberately a **rho-weighted local conditional MSE**, not complete
attention MSE or a squared global block contribution (which would involve
rho squared and cross-block interactions). It excludes global block
reweighting, other query frames, native FP8 rounding and network evolution.

## Nine prespecified comparisons

All final evaluations use full80 ownership before SQ1. P16 denotes uniform
columns 0,5,...,75 during selection; G256 is the unchanged native global
probe normalizer. Exact-Z uses the previously saved all-75,600-key denominator.
For P16/exact-Z the numerator is still P16, **not exact full-block mass**.
P80 and exact-Z are off-path information oracles, not deployable proposals.

| Selection | Weighted reverse gap | Weighted Gamma | Conditional MSE |
| --- | ---: | ---: | ---: |
| Hellinger P16 | 0.007061057 | 0.015624932 | 0.000375750756 |
| Mass P16/G256 | 0.005565352 | 0.011509337 | 0.000171910935 |
| Jeffreys P16/G256 | 0.005517048 | 0.011446741 | 0.000167386418 |
| Jeffreys P16/G256, no SQ in scoring | 0.005535494 | 0.011505163 | 0.000168222645 |
| Mass P16/exact-Z | 0.005695601 | 0.011843095 | 0.000197351773 |
| Jeffreys P16/exact-Z | 0.005580992 | 0.011682053 | 0.000182953626 |
| Hellinger P80 | 0.007322734 | 0.015802018 | 0.000370310392 |
| Mass P80/exact-Z | 0.005734599 | 0.011863558 | 0.000196190438 |
| Jeffreys P80/exact-Z | 0.005627279 | 0.011639034 | 0.000191265445 |

For Jeffreys P16/G256 vs Mass P16/G256, conditional MSE changes by
-5.6473%, -1.5683% and -3.0419% on prompts 0, 1 and 2. Reverse-gap changes
are -0.9108%, -0.6878% and -1.0826%. Pooled Gamma improves by 0.5439%, but
prompt 0 Gamma worsens by 0.7666%; even this score does not improve every
reported quantity. Head30 conditional MSE changes from 0.000174006879 to
0.000171275655. Conditional MSE decreases in all 12 prompt-by-head aggregates,
but pooled head30 Gamma rises by 0.1753%. At finer scope, prompt0/head30 Gamma
rises by 11.1277%; prompt1/head20 reverse gap and Gamma rise by 1.0057% and
1.0116%. Output-proxy wins/losses/ties by bank are 767/498/535; these are
dependent descriptive comparisons, not evidence of statistical significance.

Replacing just the Mass denominator by exact-Z worsens this proxy by
14.7988%. Using P80 plus exact-Z with the Jeffreys pivot gives 14.2658%
higher proxy MSE than its P16/G256 version. These measured comparisons do
not establish that additional information is intrinsically harmful: the
fixed greedy heuristic is not solving the full objective. They do reject
the simple expectation that these particular information substitutions
automatically repair the local approximation.

## Decision and next discriminating check

Follow-up: the native experiment linked above has now reproduced the fixed
L33 controls and measured the complete candidate output. Its all-frame MSE
is 0.3669% worse than Mass on those three captures, including regression in
RF0 itself. The local-proxy gain below is therefore not a complete-operator
gain; see the follow-up for separate full40-head timing and data scope.

Keep the new score as a research candidate and retain the current Mass-GS
default. Do not launch another 20-video run or claim a better paper method
from this result alone. PSNR, SSIM, LPIPS, VBench, attention time and SDPA
speedup are all unmeasured for this candidate.

The next useful diagnostic is to separate three omitted effects on the
existing captures: seed-row transfer from RF0 to other query frames, global
reweighting across blocks, and native merge precision. In production, RF0
shares **seed row indices**; seed query vectors, ownership and merge weights
are recomputed for each actual query frame. A temporal audit must reproduce
that behavior, not reuse RF0 merge weights unchanged. Reproducing the known
failure under these controlled settings should precede another seed/probe
redesign. This proposes a diagnostic, not a dense inference exception or a
change away from RF0.

## Reproducibility

- `analysis/conditional_jeffreys_seed.py`: NumPy reference identity and pivot.
- `analysis/cache_rf0_probe_banks.py`: CPU-only audited input/cache exporter.
- `analysis/screen_conditional_jeffreys_banks.py`: fixed nine-method comparison.
- The three associated test files contain 34 passing CPU tests.
- `banks_v2/manifest.json` records input bytes/hashes, candidate-table source,
  coordinates, precision and successful CPU export with CUDA uninitialized.
- `audit/diagnostic.json` records source/cache receipts, all per-prompt and
  per-head sums, normalizer roundoff checks, and final validity/completion.
  Three metrics NPZ files preserve seed rows and per-query metric arrays.
- `execution_receipt.json` records root-observed successful foreground tool
  exits. It is not a remote process-authored wrapper receipt.
- `independent_validation.json` checks 12 artifact/source receipts and
  independently recomputes 2,520 aggregate values (maximum relative discrepancy
  5.24e-16). A separate scalar cell-by-cell implementation reconstructs 9,720
  query projections across all nine methods: maximum KL discrepancy 2.23e-15
  and output-SSE discrepancy 2.21e-12. No block mass exceeded one, so the
  evaluator's positive-roundoff clamp was not used on these inputs.
- The first exporter attempt failed before producing banks because it checked
  an incorrect tensor key, `dense` instead of the original `dense_output`.
  The error and original attempt were retained. The corrected attempt uses a
  fresh output directory and a strict schema; no input artifact was altered.

Reproduction requires the frozen candidate-table source in an `attention/`
sibling of the exporter directory. Run with CUDA hidden and at most four CPU
threads. Both exporter and evaluator refuse to overwrite existing output
directories. Full input tensors and diagnostic artifacts are retained outside
Git; the code and this note are tracked.
