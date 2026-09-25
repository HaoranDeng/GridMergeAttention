# RF0 neighboring-column Fisher-GS

Prospective candidate, 2026-09-05. No paper/default changes.

The current J4 launch computes four independent selectors concurrently.
This candidate instead selects one set of four real row indices for four
adjacent columns, while retaining each column's independent actual-Q merge.
It never shares merged K/V, averages Q, or reads other query frames for
selection. Current-frame Q at the selected row is still loaded in merge.
RF0, C32/g45, U16, anchor+C3, SQ1 and full compressed route are unchanged.

For each key block and each neighboring query column j, compute its
candidate logits z_i^j, geometric Fisher center rho_j=softmax(mean_i z_i^j),
and anchor-relative feature

    x_i^j = sqrt(rho_j) * [z_i^j-z_anchor^j
                           - <rho_j,z_i^j-z_anchor^j>].

Concatenate X_i=(x_i^0,x_i^1,x_i^2,x_i^3) without averaging. Keep the
actual anchor and select three pivots by residual-norm Gram-Schmidt on X.
Both the squared norm and projection coefficient sum across columns and
probes. Four independent projections with a shared pivot are NOT this rule.
The four-column group is now an algorithm parameter, fixed at width4.

The sum of four logsumexp objectives has a block-diagonal Fisher Hessian.
The concatenated squared distance is its joint local quadratic geometry.
This is a local approximation and a subspace-coverage heuristic, not an
optimal hard partition or a bound on generated-video PSNR. Its motivation
is to avoid over-specializing shared RF0 row indices to one column's
responses; spatial neighbors need not proxy temporal changes, and forcing
common rows may hurt edges or small objects.

The prototype and native versions reuse existing U16 RF0 logits. Native
BF16 QK products accumulate in FP32; curvature and GS use FP64. No added
QK products or merge/route changes. Cross-column reductions still require
measurement: no speedup or no-slowdown is asserted in advance.

## First real-QKV results and bounded video check

Prototype an and native ao completed cleanly on scai6 GPU1. Pooled MSE
was approximately 0.001546068 for both, versus 0.001257260 for independent
geometric Fisher32 and 0.001298445 for Hellinger RF0 BF16. Thus sharing
neighboring-column row indices worsened this all-frame L33 check. Native
seed-set agreement with the prototype exceeded99.993% for every capture;
this is not exact row equivalence. Independent CPU tests also execute the
actual kernel arithmetic against an explicitly concatenated NumPy GS.

One bounded full20-video check ap is retained to test the hypothesized
multi-layer temporal benefit directly, not as a promotion of its negative
single-layer result. Anchor-Fisher's opposite local/video ranking makes
that distinction material. No early rejection on a three-video prefix.
The generation uses frozen7ff8226, all40 layers/all4 steps and BF16 Q/K/V;
all metrics and timing conclusions require its own completion and audit.

Run ap completed validly at2026-09-05T15:25:33Z with all20 canonical
videos/all3200 calls. PSNR was11.57804054981833, below anchor-Fisher
11.636408427866783 and geometric Fisher32 11.617539025066193, although
above matched-reference historical Hellinger RF10 11.529624325204969.
It is not promoted. This does not support replacing independent columns
with shared row indices for the present goal. Full metric suite is as;
CPU-overlapped generation timings remain provisional, not a matched speed
or no-slowdown result. Strict RF0-only research has not reached PSNR12.
