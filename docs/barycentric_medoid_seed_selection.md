# A real barycentric medoid as the first Fisher seed

2026-09-05. Isolated RF0-only prototype; not a production or paper change.
The four returned seed rows are real queries. C32/g45, Uniform16 probes,
BF16 merge, SQ1, entropy and full compressed route are unchanged. No training,
calibration, synthetic mean, extra query frame, or global mass probe is used.

## First seed: exact discrete one-cell solution

For the R=32 RF0 candidates on one fixed probe cell, let

    z_i = K_probe q_i / sqrt(d), p_i = softmax(z_i),
    z_bar = (1/R) sum_i z_i, rho_bar = softmax(z_bar).

The first seed minimizes KL(p_s || rho_bar) over those real candidates.
This is not the opposite KL direction. The exact identity

    (1/R) sum_i KL(p_s || p_i)
      = KL(p_s || rho_bar) + (1/R) sum_i LSE(z_i) - LSE(z_bar)

shows that the selected real seed minimizes average reverse KL on this
fixed probe cell. The matched key row remains eligible but is not forced.
The unrestricted optimizer is rho_bar, represented by the mean candidate Q,
but this method never inserts that synthetic query. It retains an actual
candidate's Q, including its magnitude and positional information.

## Three further seeds: unchanged anchor-Fisher geometry

Use rho = p_first, delta_i = z_i - z_first, and

    x_i = sqrt(rho) * (delta_i - dot(rho, delta_i)).

Choose the three remaining real rows by maximum squared GS residual norm.
All four selected rows must be distinct. Exact ties use the first available
candidate-table index. The denominator safeguard remains 1e-12. Candidate
Q/K operands are original BF16 values promoted to FP32 for QK accumulation;
scaling, stable log-softmax, KL costs, Fisher features and GS are FP64.
TF32 is disabled in the guarded screen. This is not a true FP64-QK oracle.

Selection reads RF0 Q only. It returns one table indexed by key frame, key
row, query column and head. During merge, each current query frame reloads
the original Q at those row indices. Slot zero is a real medoid; the
synthetic/mean-seed adapter must not be enabled.

The first-seed cost requires O(RP) extra arithmetic using existing logits,
without new QK products or an R-by-R KL matrix. Kernel resource usage and
complete-attention latency still require measurement; no equal-speed claim
follows from this operation count.

## What the theorem does not establish

The exact result is for one fixed probe cell and equally weighted candidate
queries. It does not optimize the final four-cell partition or all 45 query
rows, establish monotone improvement after adding seeds, or predict video
quality. For a fixed multi-cell global objective, component barycenters use
query weights multiplied by query-dependent route masses with those masses
held fixed. Hard repartitioning also changes the feasible compressed family;
these families need not be nested. RF0 row reuse adds another approximation.
See seed_barycenter_limits.md for the identities and counterexample.

## Distinguish previous experiments

- mean_seed_bf16_screen_20260905y inserts the current-frame mean over all
  45 query rows: mean_fisher32 uses three Fisher pivots; mean_anchor_fisher32
  includes the matched real row and two further pivots. Their L33 MSEs were
  0.001494061712 and 0.001367319218. Neither is this all-real medoid rule.
- query_medoid_bf16_screen_20260905ak fixes the matched anchor and updates
  the other three query-cluster medoids after GS. Its KL/Fisher L33 MSEs
  were 0.001458905266 and 0.001476816519. This proposal instead initializes
  one free first seed, then performs the ordinary anchor-Fisher GS steps.
- rf0_medoid_cpu_validation_20260904m checks a C16, mass-weighted first
  medoid followed by partition-deficit selection, not C32 uniform-query
  weighting followed by Fisher-GS.
- owner_preserving_medoid_screen_20260904c keeps the original anchor and
  conditions refinements on existing key owners; it is another distinct rule.

Prior L33 rankings are not treated as video-quality gates. The prototype's
initial GPU/video/timing status was pending; the completed native and
complete20 results below supersede it. A final goal decision additionally
requires a matched complete-attention speed check.

## CPU verification

Six tests passed on scai6 with CUDA_VISIBLE_DEVICES empty and one CPU thread
in 12.337 seconds. The test modules were isolated in
/tmp/barycentric_medoid_cpu_20260905_t7twYv and imported only unchanged
candidate-table dependencies from the frozen run-j source. Checks include
the exact reverse-KL/mean-query identity, an independent all-pairs KL
first-seed oracle and three GS steps, distinction from the opposite KL
direction, constant-logit shifts, deterministic ties, underflow-scale
logits, input validation, and the full BF16 [1,75600,4,128] layout with
every non-RF0 query value set to NaN. Cross-frame/key-row/query-column/head
banks agree with independently indexed NumPy references. This is CPU
implementation validation, not native GPU parity or quality evidence.

## Native validation and direct-video test

The native C32 RF0 selector `fisher_barycentric_medoid_seed_rows_triton`
implements the same first-seed costs and three subsequent GS pivots in FP64,
with BF16 QK operands and FP32 accumulation. No extra QK dot products,
synthetic seed, global mass pass, or merge override is used. Actual native
feature/pivot AST tests match an independent all-pairs-KL NumPy oracle.

`barycentric_native_check_20260905bc`, frozen bf49da4, completed validly on
an exclusive scai6 H200 at23:26:42 UTC. For each of two synthetic integer-BF16
input tensors, all7200 native real-medoid seed groups matched the independent
PyTorch oracle exactly. This establishes that bounded implementation check,
not exact equivalence on every real input, video quality, or latency.

The canonical full20 run
`wan14b_720p_4step_barycentric_medoid_rf0_bf16score_extended_20260905be`
was launched from the same immutable commit and completed validly at
2026-09-05 23:57:13 UTC. Full20 CPU PSNR is 11.485429304894780, below
existing Anchor32 (11.636408427866783) and below the requested 12. It is
not promoted. This is a full20 comparison, not a prefix or single-layer
MSE ranking. Full-resolution SSIM/LPIPS evaluation bi is still running.

The separate original-full40 BF16 benchmark bf measures 160.069936ms for
the complete operator, 2.055165x versus matched SDPA and 23.7709% slower
than Hellinger RF10. It also fails the final speed requirement. See
[the complete20 follow-up](barycentric_complete20_followup.md) for the
24-input/480-timed-invocation protocol and its limitations.
