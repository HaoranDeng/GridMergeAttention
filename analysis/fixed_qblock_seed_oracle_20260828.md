# Fixed Seed Queries per Q Block: Offline Oracle Diagnostic

## Question

Can each native Wan query block use one fixed set of four real query rows,
shared over all key frames and key rows, without giving up too much attention
quality?  This is the useful upper-bound diagnostic for a future selector that
predicts seed rows from Q alone.

The experiment defines one Q block as a fixed `(layer/call, head, query frame,
query column)`.  The offline oracle may inspect K to choose four row indices,
but the resulting C4 table is constant over all 945 key blocks.  It is therefore
an oracle control, not a deployable training-free GMA selector.

## Search

For each Q block, every one of the 45 query rows is represented by its
blockwise attention distributions on real captured K.  We solve all
`C(45,4) = 148,995` C4 combinations under two search distances:

- mean blockwise squared Hellinger distance;
- mean blockwise forward KL distance.

The search is repeated on two mostly disjoint, uniformly distributed sets of
64 key blocks and their union.  These six exact p-median proposals plus a
Q-only global-C4 proposal are then compared with the exact induced-GMA KL on
all 945 key blocks, independently for each Q block.  The winning proposal is
the final fixed C4 oracle table.

The representative evaluation uses captured calls 1, 10, 20, 30, and 40,
heads 0, 10, 20, and 30, and query frames 0, 10, and 20: 60 cases, 4,800 Q
blocks, and 216,000 evaluated query rows.

## Results

| Method | Fixed C4 per Q block | Reads K online | Mean KL | Ratio vs fixed-wide |
|---|---:|---:|---:|---:|
| Q-only global C4 D32 | yes | no | 0.536973 | 1.5280 |
| Offline K-aware fixed-C4 Hellinger | yes | offline only | 0.414575 | 1.1797 |
| Offline K-aware fixed-C4 refined oracle | yes | offline only | **0.391547** | **1.1142** |
| Production fixed-wide | no; changes with key row | no tensor K | 0.351427 | 1.0000 |
| Key-dependent Hellinger-Span | no; changes with key block | yes | 0.327348 | 0.9315 |

The refined oracle lowers KL by 27.08% relative to the naive Q-only global
C4 selector.  It beats fixed-wide in 31 of 60 cases, but its aggregate KL is
still 11.42% above fixed-wide and 19.61% above key-dependent Hellinger-Span.
The gap is strongly layer-dependent.  Calls 1, 10, 30, and 40 improve or remain
close to fixed-wide, whereas call 20 degrades sharply (0.9091 versus 0.5575).

## Where the oracle seeds lie

The oracle does not collapse to a small collection of row positions:

- effective row support: 44.35 out of 45;
- normalized row entropy: 0.9962;
- mean sorted seed positions: `[6.96, 17.40, 27.51, 37.24]`;
- mean C4 span: 30.29 rows;
- mean pairwise row distance: 16.83 rows;
- largest single-row mass: 2.80%.

The average positions are roughly evenly spread over height, but their
standard deviations are large: `[5.41, 6.90, 6.43, 4.91]`.  Thus a simple
fixed quantile pattern captures the coarse geometry but not the actual
per-block choices.

The two key-sample splits share only 2.116 of four rows on average.  Their mean
Jaccard similarity is 0.425 and only 16.77% of Q blocks recover exactly the
same C4 set.  The best fixed seeds are therefore moderately dependent on which
K blocks define the objective, even though the final seed table itself is
key-block independent.

## Conclusion

This direction is worth retaining as a bounded research branch, because an
offline fixed-C4 oracle recovers a meaningful 27% KL reduction over the naive
Q-only selector and wins on about half the cases.  It is not yet evidence that
a universal or cheaply Q-predicted seed rule can match current GMA: the oracle
still loses to production fixed-wide on aggregate, has a severe bad layer,
and its locations have nearly full row entropy.

The next useful test is not another hand-designed global position rule.  It is
to identify a cheap Q-only statistic that predicts the oracle's coarse
four-band allocation while adding a layer-aware safeguard for call-20-like
blocks.  Any such selector must be evaluated against this oracle table and
must remain fully training-free to qualify as GMA.
