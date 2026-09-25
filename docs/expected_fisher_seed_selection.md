# Expected-Fisher real-query seeds

Exploratory, 2026-09-06 UTC. This is a new curvature surrogate with a CPU
reference and an isolated native J1 implementation, not an established video
improvement. No paper or production
default is changed. It keeps a real matched anchor plus three real rows,
RF0-only selection, C32/g45 and Uniform16 probes. It adds no training,
calibration, global normalizer, synthetic query, or change to KV merging
and core attention with compressed KV.

## Average the local curvature, not the distributions

Let z_r be the existing probe logits of candidate r and p_r=softmax(z_r).
The proposed metric is

    G_avg = mean_r [diag(p_r) - p_r p_r^T].

For Phi(u)=mean_r F(z_r+u), this is exactly the Hessian at u=0. Equivalently,
for a small common logit perturbation u and finite logits,

    mean_r KL(p_r || softmax(z_r+u)) = 0.5 u^T G_avg u + O(||u||^3).

Thus the geometry averages local sensitivity across the candidate
population instead of selecting one anchor or mean-logit expansion point.
The covariance identity makes the distinction from earlier proposals clear:

    G_avg = [diag(mean_r p_r) - (mean_r p_r)(mean_r p_r)^T]
            - Cov_r(p_r).

It is not Fisher at the arithmetic mean profile, Fisher at the geometric
mean profile, or the earlier population-gain pivot rule. The latter changed
the criterion for choosing a direction while retaining one common Hessian.

Keep the matched anchor a and set delta_i=z_i-z_a. The feature geometry has
Gram matrix R_ij=delta_i^T G_avg delta_j. Choose three ordinary largest
residual-norm pivots, with the anchor first, exactly as in the existing GS
heuristic. Return the original row indices; current-frame merge loads the
current-frame queries at those rows.

An explicit but inefficient equivalent feature is the concatenation over r
of sqrt(p_r/R) * (delta_i - <p_r,delta_i>). The reference uses only the
small P-by-P metric and R-by-R Gram matrices, never a bank-sized stacked
feature tensor. Constant logit offsets are removed before the Gram products.

## Verification and current boundary

`analysis/expected_fisher_seed.py` provides independent NumPy and Torch FP64
references. Original BF16 Q/K values are multiplied in FP32, then scaling,
probabilities, metric and selection use FP64. The existing 1e-12 GS
safeguard is retained; its exact residual-Gram update uses
(d+2*epsilon)/(d+epsilon)^2, not merely 1/(d+epsilon).

All 12 tests passed on an actual single-thread Torch CPU run on scai6
(5.167 seconds, CUDA_VISIBLE_DEVICES empty). They check explicit stacked
features, the covariance identity, constant shifts, PSD/nullspace, stable
degeneracy and ties, epsilon-sensitive cases, true Torch operations,
cross-batch/head/frame indexing, and the standard 21-frame output table.
Local NumPy checks also pass; missing local Torch tests are explicitly
skipped, not counted as executed Torch evidence.

## Isolated native J1 implementation and acceptance gate

Commit `cb3bd87` adds `attention/expected_fisher_seed_triton.py`, the isolated
`fisher_expected_rf0` video contract, and the explicit
`analysis/check_barycentric_native.py --family expected_fisher` checker.
The wrapper supports actual J1 only: `block_j=1` is the default and J4 is
rejected. Each program owns one query-column/key-row/head/batch bank and
loops over key frames; it does not share seed decisions across columns.
RF0 selection, four real rows, SQ1 merging and full compressed attention
remain unchanged. The `global_probe_count=256`/G256 dispatch field is only
a compatibility check: no G256 sketch, global-normalizer estimate or mass
weighting is computed or consumed by this selector.

BF16 QK dot products accumulate in FP32, then logits are promoted to FP64
before scaling; all geometry remains FP64. A logical program-local 32-by-32
matrix `Y[i,r] = <delta_i,p_r>`, `rho=mean_r p_r`, and two covariance columns
replace a full residual Gram workspace. Three pivots need only two
safeguarded rank updates; the second pivot column includes the first update.
No HBM Gram/stacked-feature workspace is allocated. Logical local storage is
not a claim of physical register residency: compiler layout and spilling
have not been measured. Optional FP64 score stores are diagnostic only and
are disabled on the normal path.

The local regression run recorded 72 tests: 67 passed and 5 missing-Torch
tests were explicitly skipped. This includes CPU/AST contracts and does not
constitute native GPU parity. The separate actual Torch CPU reference run
above remains the evidence for those Torch operations.

The guarded GPU parity run is
`/local2/denghaoran/GridMergeAttention/experiments/expected_fisher_native_check_20260906bn`
on scai6. Directly inspected `results.json` and `completion.json` show that
it passed at `2026-09-06T00:41:39.524206+00:00`: `valid=true`, worker exit 0,
no foreign process, and unchanged source fingerprints. This establishes
synthetic native numerical parity, not video quality or attention speed.
Its predeclared gate was:

- Original independently generated integer BF16 Q/K, RF0 Q shape
  `[1,1,45,80,2,128]`, C32/g45 and Uniform16; no head replication or captures.
- Seed 17 uses one key frame and scale `0.125` (7,200 banks); seed 53 uses
  two independent key frames and scale `128**-0.5` (14,400 banks). The latter
  separately exercises key-frame decoding and actual FP64 attention scaling.
- All 21,600 banks must have exact row equality with independent Torch FP64
  `expected_fisher_indices`, the matched anchor, and four distinct in-table
  real rows. Native score-audit on/off must return exactly the same rows.
- All `21,600 * 3 * 32 = 2,073,600` scores are compared at the actual native
  pivot prefixes with the independent FP64 Gram oracle. The fixed gate is
  `abs(error) <= 2e-10 + 2e-11 * abs(reference)`; no tolerance override is
  exposed, and score agreement cannot substitute for exact row agreement.
- Parent/worker source snapshots and before/after fingerprints, two clean
  H200 checks, five-second exclusivity monitoring and fail-closed completion
  are required. The check leaves attention latency and speedup null.

Both cases passed every gate with zero row/anchor/uniqueness/membership or
score-audit-on/off mismatches, including zero mismatches in each key frame.
For seed 17, all 691,200 scores passed; maximum absolute error was
`1.624478329631529e-12` and maximum fixed-tolerance ratio was
`0.004971794958812079`. For seed 53, all 1,382,400 scores passed; the
corresponding values were `6.989964163039986e-13` and
`0.002296627195301443`. No tolerance was enlarged. The accepted native source
SHA256 is `cb65b300288c9d8771cbfc522411450d422a2c7d263dfb5529518671c5e00d99`;
the checker SHA256 is
`26219dbeab38aeac9bbcde583cc0944f4ceb0f055d07cf439bdc395da6b4b1e2`.

## Complete 20-video result: quality target not reached

The frozen `cb3bd87` run
`wan14b_720p_4step_expected_fisher_rf0_bf16score_extended_20260906bo`
completed on scai6 at `2026-09-06T01:10:42.594519+00:00` with
`valid=true`, worker exit 0 and no foreign process in 327 monitor samples.
It used the canonical extended prompts 0 through 19, seed 0, four steps,
the raw 14B checkpoint, original BF16 score inputs, RF0 real-row selection
and the same operator in all 40 layers: 3,200 complete attention calls.

All 20 videos (81 frames each) were scored against the matching dense
reference. Mean PSNR is **11.612386415987558 dB**. This exceeds the historical
Hellinger C4 RF10 result, 11.529624325204969 dB, but is below the current
Anchor32 result, 11.636408427866783 dB, and does not reach the requested
12 dB. No prefix score was used to reject or rank the method.

The instrumented generation run reports **233.48598093509673 ms** per
complete GCA call and **270.3577928161621 ms** including the full wrapper.
These measurements include metadata checks and overlap with CPU video
scoring. No matched SDPA or Hellinger timing was measured for this candidate,
so `speedup_vs_sdpa` is unavailable and these figures do not establish a
speed ranking. SSIM and LPIPS are not yet measured for this run.

The PSNR artifact SHA256 is
`64881a7abc9c9001bcae536c7f622aef91aef1dce3feb902eff4131c91625ff0`.
The result does not justify promoting this candidate or prioritizing its
speed optimization. Its curvature remains a local perturbation surrogate,
not a bound for the final seed-induced hard partition or video error. The
original joint quality, speed and simplicity objective remains unmet.
