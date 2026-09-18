# A maximum-entropy seed and three real query directions

Prospective RF0-only experiment, 2026-09-05. No quality or speed improvement
has been established. The paper repository is unchanged.

For a fixed nonempty cell R of m keys, the entropy-regularized objective at
q=0 is maximized by uniform weights. Its exact tangent representative is

    k_bar = mean_{j in R} k_j,
    v_bar = mean_{j in R} v_j,
    corrected_logit(q) = q^T k_bar / sqrt(d) + log(m).

This motivates reserving one seed for the analytic zero query, and choosing
three real queries to represent departures from this maximum-entropy state.
On P fixed probes, with u=(1/P,...,1/P), the exact identity is

    KL(u || softmax(z)) = logsumexp(z) - mean(z) - log(P).

Near zero logits, it equals ||z-mean(z)1||^2/(2P) plus higher-order terms.
Thus use x_i=(z_i-mean(z_i)1)/sqrt(P), then select three distinct real rows
by largest residual-norm Gram--Schmidt pivots. C32/g45 and Uniform16 remain
fixed. There is no forced real anchor and no candidate-average or learned
reference. The local quadratic and greedy span are heuristics, not an
optimizer or bound for the induced hard partition or final video PSNR.

The zero query competes with the three real seeds under the original
normalized full-block hard ownership. Its response is exactly 1/80 in
ideal arithmetic. It owns keys that the other seeds do not favor above
that reference, with ties resolved in seed order. After the query-column
singleton is removed, its nonempty cell still has uniform conditional
weights and entropy log(m). Other cells keep ordinary conditional Softmax
and their actual entropies. All representatives from every key block still
participate in the full compressed route.

Only RF0 actual Q scores select the three real row indices. At each merge,
slots 1--3 use that frame's actual Q, and slot 0 uses zero. A dummy matched
row identifier occupies slot 0 in the index table solely for kernel layout;
it is not a fourth real query and may equal one of the three real indices.
The table is invalid if used without the explicit zero-query merge override.
Unlike the earlier mean-Q seed experiment, no query vectors are averaged.

Important limits: removing the real matched anchor may hurt quality; the
zero cell may be empty or too large; large logits weaken the approximation.
The complete operator is not exact at q=0, because the other cells generally
have nonuniform weights. The four-slot merge still runs, so no speedup may
be inferred from the analytic zero seed. This is a different seed family,
not a relabeling of the paper's four-real-query selector.

## Completed full20 result: negative

`wan14b_720p_4step_zero_fisher_rf0_bf16score_extended_20260905az`, frozen
fc8f552, completed 2026-09-05 23:04 UTC with all20 canonical videos, all40
layers/all4 steps (3200 attention calls), raw_state_dict checkpoint and
no observed foreign GPU process. Full20 PSNR is **11.069348277730807**,
versus Anchor32 **11.636408427866783**. Reject this version for quality.

The prior L33 four-head diagnostic's pooled MSE was 0.00026608638920290494;
its roughly 80% gain came almost entirely from original head30 and failed
to predict video ranking. Native/prototype seed tables were not exactly
identical. Neither that diagnostic nor its mathematical motivation is a
video-quality guarantee.

Instrumented complete GCA attention averaged 153.3797714138031 ms per call
(steady last19: 146.17478996076082 ms). Full module time including surrounding
work averaged 191.81433202266692 ms. These are not matched interleaved
Hellinger/SDPA timings; speedup_vs_sdpa is unmeasured. Completed full20 metrics
in `zero_fisher_video_metrics_20260905ba` are SSIM 0.33926977114523005 and
LPIPS 0.5651841654821679, also worse than Anchor32. All20 videos, source and supporting
artifacts are archived locally; the exploratory workbook row is 578.
