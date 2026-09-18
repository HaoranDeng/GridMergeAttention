# One mean seed plus three real query seeds

Exploratory, 2026-09-05. No production default or paper change.

For a fixed nonempty key cell, softmax of the mean query logits uniquely
minimizes average KL(w || softmax(z_i)). See seed_barycenter_limits.md for
the proof and why this does not establish four-cell or video optimality.
Norm rescaling would destroy the exact statement, so none is used.

The proposed change reserves seed zero for the uniform row-mean query.
Selection still uses RF0 and the original C32/g45, Uniform16 support. The
three selected real-row indices are reused across frames as before; merge
loads those rows from each current query frame. The analytic seed likewise
uses the current query frame's row mean, not a temporally frozen RF0 vector.
Means are accumulated in FP32 and rounded to BF16 before selection logits
and merge. All other Q/K/V inputs, BF16 merge, entropy, SQ1 and full route
are unchanged. The arithmetic theorem is exact before this rounding.

Prespecified comparisons:

- mean_fisher32: three real rows chosen by Fisher-GS about the mean tangent.
- mean_anchor_fisher32: keep the actual matched row among the three real
  rows; project its direction first, then choose two more Fisher-GS pivots.
- mean_hellinger32: three GS pivots on sqrt(p_i)-sqrt(p_mean).

The first is the primary proposal. A repeated row identifier between the
dummy slot zero and a real seed is allowed: slot zero contains the mean
vector, not that real row. The other three row identifiers must be distinct.
The optional merge-kernel flag defaults off; only the isolated experiment
adapter enables it. Route queries are never replaced by means.

The first gate is complete-output MSE on three frozen L33 captures, all21
frames and four heads. Include the original Fisher32 PyTorch control,
whose previous pooled MSE is 0.001257257061. This gate cannot establish
PSNR, SSIM, LPIPS or speed. A promising candidate requires native selector
verification and all20 canonical video evaluation before promotion.

Run mean_seed_bf16_screen_20260905y completed cleanly. Its Fisher control
exactly reproduced the previous pooled MSE 0.001257257061. The three mean
variants gave 0.001494061712, 0.001367319218 and 0.001527821510 respectively.
At that screen's decision point, none advanced to videos. Mean-Fisher and mean-Hellinger improved RF0 MSE
but increased error over the other20 frames. This limits the empirical
value of the fixed-cell barycenter theorem under this row-sharing scheme;
it does not refute that theorem. No video metrics or timing were measured.

## Reopened direct-video test, 2026-09-05

The subsequent zero-anchor experiment exposed a limitation of the above
four-head, single-layer gate: an approximately 80% pooled local-MSE gain
was concentrated in original head30 and led to **worse**, not better,
canonical full20 video PSNR (11.069348277730807). Therefore the old local
ranking is not sufficient evidence to exclude the mean-query candidate.
This reopens the same clearly identified mean_fisher32 rule for a direct
all40-layer/all4-step/full20 test, not a claim of a newly invented formula.

The native video mode is `mean_fisher_rf0`. Its first slot is the current
frame's unrescaled mean over all45 spatial query rows; the other three
slots are actual RF0-selected row indices. The mean is formed once per
complete attention call and reused only across that call's five head chunks.
No mean is cached across attention calls. RF0 selection independently uses
the same FP32-average/BF16-round rule on frame zero. `mean_anchor_fisher_rf0`
is a separate available control, not implicitly part of the primary test.

At the initial launch, neither native video quality nor matched speed had
been established. The completed follow-up below supersedes that pending
status. The fixed-cell theorem and its limits remain unchanged; the paper
repository and production defaults are untouched.

Native correctness check `barycentric_native_check_20260905bc`, frozen
bf49da4, completed validly on an exclusive scai6 H200 at23:26:42 UTC.
Both mean modes and the separate real-medoid mode each matched the independent
oracle on all7200 banks of each of two synthetic integer-BF16 input tensors.
This is synthetic implementation evidence, not real-video equivalence or quality.
The full20 mean_fisher run `wan14b_720p_4step_mean_fisher_rf0_bf16score_extended_20260905bd`
was then launched from the same immutable commit.

## Complete20 outcome, 2026-09-06 UTC

Mean-Fisher bd completed all 20 videos with verified all-layer execution.
Its full-resolution PSNR is 11.030490212430504, below existing Anchor32
(11.636408427866783). Keeping the matched real row in the separate
mean_anchor_fisher bg control gives 11.288276912442422, also below Anchor32.
Neither mean candidate is promoted. These are full20 results, not prefix
or fixed-QKV predictions.

Full metrics bh independently gives Mean-Fisher PSNR 11.030490212430506,
SSIM 0.33774328250803387 and LPIPS 0.5669000752178239. The matched complete
operator benchmark bf measures 149.406114ms, 2.201852x versus matched SDPA
and 15.5253% slower than Hellinger RF10. Mean + matched anchor has not been
included in that benchmark; its matched speed is unmeasured.

See [the full20 follow-up](barycentric_complete20_followup.md) for exact
run identities, pending metrics and the separate timing protocol.
