# Hellinger-Span C4 Seed Concentration and Q-Only Oracle

## Scope

This note analyzes the production Hellinger-Span C4 selector on captured
Wan2.1-14B attention tensors. The selector uses `alpha=0.3`, `C=22`,
`P=16`, `A=2`, `J=4`, Gram-Schmidt diversification, and router frame 10.
All statistics use the exact row indices emitted by the production selector.

The concentration study covers 40 attention calls and heads 0, 10, 20, and
30. To avoid counting the four-column probe-group replication as four
independent choices, row-frequency statistics use one representative column
per group. This leaves 12,096,000 seed selections, including 9,072,000
adaptive selections after excluding the fixed anchor.

## Are the selected queries concentrated?

No. The selected query rows are broadly dispersed and strongly conditioned
on the current key block, column group, and attention head.

| Statistic | Production Hellinger-Span C4 | Fixed-wide C4 reference |
|---|---:|---:|
| Global effective row support (adaptive seeds) | 40.80 / 45 | 4 / 45 |
| Global top-1 row mass (adaptive seeds) | 4.62% | -- |
| Global top-4 row mass (adaptive seeds) | 17.29% | 100% |
| Global top-8 row mass (adaptive seeds) | 31.68% | 100% |
| Mean effective rows within one key-frame/column-group/head context | 24.65 / 45 | 4 / 45 |
| Mean unique rows within one context | 33.91 / 45 | 4 / 45 |
| Mean C4 row span | 19.52 rows | 4.78 rows |
| Median C4 row span | 18 rows | -- |
| Mean pairwise row distance inside C4 | 10.53 rows | 2.56 rows |

The adaptive-row distribution has normalized entropy 0.9743. Within a fixed
`(key frame, query column group, head)` context, its mean top-4 mass is only
35.93% and its mean top-8 mass is 54.39%. Thus even after conditioning on a
local routing context, a small globally shared query-row set does not cover
most selections.

There is still a local bias relative to the target key row: 22.32% of adaptive
seeds lie within one row, 35.11% within two rows, 53.64% within four rows, and
67.80% within eight rows. This is a broad, target-dependent neighborhood rather
than concentration on a few absolute rows. No C4 set contains duplicate rows.

The `J=4` implementation shares probe positions across four adjacent query
columns, not the final selected seed rows. Among non-leading columns in a
group, only 0.127% exactly reproduce the leader's complete C4 row set.

## Shared-seed diagnostic

We also compressed the production Hellinger decisions into the four most
frequent rows and reused that shared C4 set. This is a post-hoc, K-informed
diagnostic, not a deployable Q-only selector. Exact
`KL(A_dense || A_GMA)` was evaluated on five layers, four heads, and three
query frames, totaling 216,000 query distributions.

| Selector | Mean KL | Change versus fixed-wide C4 |
|---|---:|---:|
| Fixed-wide C4 | 0.35143 | baseline |
| Original key-dependent Hellinger-Span C4 | 0.32735 | -6.85% |
| Post-hoc top-frequency shared C4 | 0.57907 | +64.78% |
| Q-only global C4, projection dimension 32 | 0.52542 | +49.51% |

The collapse of the shared-vote control shows that the useful signal is the
block-conditional variation in the seed set, not a small collection of
universally important query positions.

## Finite Q-only portfolio oracle

To estimate how much headroom remains without reading K at selection time, we
formed a finite portfolio from the existing analytic Q-only selectors. Each
candidate selector uses Q only. For every captured QKV case, however, the
oracle examines the true K-derived KL and chooses the best candidate after the
fact. It is therefore a non-deployable upper bound for this finite portfolio,
not a valid inference algorithm and not the optimum over all possible Q-only
functions.

Across 480 cases and 1,728,000 query distributions:

| Method | Mean KL | Improvement versus fixed-wide C4 |
|---|---:|---:|
| Fixed-wide C4 | 0.54296 | baseline |
| Per-case finite-portfolio oracle | 0.48034 | 11.53% |

The most frequent oracle winner was global anchor-plus-three at projection
dimension 32 (153/480 cases), followed by QR-cosine (81), row-proxy Hellinger
k-center (65), global C4 at dimension 32 (40), and L2 k-center (33). The broad
winner distribution means there is no single dominant Q-only heuristic in the
current portfolio.

## Implication

A fixed, globally precomputed C4 query set is not a faithful approximation to
Hellinger-Span. A promising key-free selector must cheaply predict
target-dependent seed variation from Q alone, or use a small structured Q-only
portfolio with a deployable Q-only routing criterion. The 11.53% oracle gain
quantifies finite-portfolio headroom, but it cannot be reported as an achieved
training-free method because the oracle selection itself uses K-derived error.

