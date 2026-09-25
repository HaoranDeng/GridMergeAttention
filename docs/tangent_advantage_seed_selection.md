# Seed selection by complementary variational gains

Prospective RF0-only candidate, 2026-09-05; no paper/default change.

For F(z)=logsumexp(z) and p_s=softmax(z_s), the seed tangent is
t_s(z_i)=p_s dot z_i+H(p_s). Its exact gap on the same U16 support is

    D[s,i] = F(z_i)-t_s(z_i) = KL(p_s || p_i).

The represented partition fraction is A[s,i]=exp(-D[s,i]), so the increase
from adding tangent s to the retained anchor's envelope is exactly

    X[s,i] = max(A[s,i]-A[anchor,i],0).

Choose three residual-norm GS pivots in rows X[s,:], retaining the actual
anchor. Unlike Fisher-GS, feature coordinates index target queries, not
keys or local logit directions. Unlike the already-tested tangent_coverage,
this covers linear directions of advantage patterns instead of greedily
maximizing their pointwise maximum sum. There are no weights, learned
parameters, tuned temperatures or additional QK products.

This avoids assuming a shared local Hessian describes large seed/target
differences. It does not make the complete seed-selection problem exact:
actual GCA creates separate hard-owned key cells and conditions each seed
inside its cell, and reuses RF0 row indices across frames. A small span
residual in X is not a bound on that operator or video quality. U16 support
and the finite candidate set are also approximations.

The extra R-by-R directed KL matrix costs O(R^2 P), followed by O(C R^2)
GS work. With fixed R32/P16/C4 it is bounded, but no runtime or no-slowdown
claim is made before native complete-attention measurement. All merges and
routes remain unchanged BF16/current-frame actual Q, C4+SQ1, all layers.

## Completed bounded screen

Run tangent_advantage_bf16_screen_20260905aq completed cleanly on scai6
GPU4. Complete-output MSE was approximately 0.001411210372, versus
0.001298445288 for Hellinger RF0 BF16 and 0.001257260255 for Fisher32.
The gain-space formulation improved on the earlier L1 tangent-coverage
screen (0.001566768618), but did not improve the matched reference. No
native kernel or video run is planned for it in this round. This is not
evidence that all exact KL objectives fail; the fixed support, surrogate
GS objective and subsequent hard partition are still different problems.
