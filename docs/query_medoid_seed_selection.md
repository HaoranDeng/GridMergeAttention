# Query-cluster medoids from the entropy objective

Prospective RF0-only screen, 2026-09-05. No other-frame statistics.
Four real seeds, real anchor, C32/g45, U16 and unchanged BF16 merge/route.
This is not the earlier key-owner-preserving medoid or partition lookahead.

Initialize with geometric Fisher-GS. Assign each candidate query i to its
best actual seed tangent, minimizing KL(p_s || p_i) on the probe support.
For a fixed query cluster U, its best real seed solves

    min_s sum_{i in U} KL(p_s || p_i).

Writing mean(z_U) for its average logit vector, the seed-dependent part is
exactly |U| KL(p_s || softmax(mean(z_U))). This gives a direct, finite
search for the best representative of that fixed query cluster. Keep the
geometrically matched seed fixed and update the other three once. Restrict
each search to its query-cluster members plus its current seed, reserving
the other initial/updated seed rows to ensure four distinct real indices.
Empty clusters and ties keep their previous seed.

This update cannot increase the stated fixed-query-cluster objective in
exact arithmetic. It does not assert improvement after the separate hard
key partition, temporal row reuse, or generated-video dynamics.

query_kl_medoid is the primary proposal. query_fisher_medoid makes the same
one-pass representative update using the paper's shared quadratic Fisher
metric instead of exact KL. Both need only the existing RF0 local logits;
there is no additional temporal input, global mass sketch, or fitted parameter.

## Completed bounded check

Run query_medoid_bf16_screen_20260905ak completed cleanly on scai6 GPU1.
Three canonical captured prompts, step0/layer33, four heads and all21
frames yielded complete-output MSE about 0.001458905266 for exact KL
medoids and 0.001476816519 for shared-Fisher medoids. Both are worse than
the matched Hellinger RF0 BF16 reference (0.001298445288) and original
Fisher32 (0.001257260255). These are local output checks, not video metrics.
Neither is advanced to native selection or generation in this round.
