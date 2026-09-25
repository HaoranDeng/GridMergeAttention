# RF0 seed exploration: completed video decisions

Status2026-09-05. Goal remains unmet: canonical4-step/20-video PSNR>12,
simple entropy-derived seed selection, complete attention no slower than
Hellinger C4. No default or paper change. This is not a speed promotion.

All video numbers below use the same canonical20 prompts, seed0, raw
distilled checkpoint and dense reference20260826. All new generation
runs audited3200 calls across all40 layers. C4+SQ1, BF16 and full compressed
route are retained. The numerical timing observations overlap CPU scoring
and other host work and are not a matched benchmark or SDPA speedup.

| Method | Video run | PSNR | Observed GCA ms/call | Workbook data row |
| --- | --- | ---: | ---: | ---: |
| Geometric Fisher32 RF0 | n | 11.6175390251 | 150.026927691 | 339 |
| Anchor-Fisher RF0 | ae | 11.6364084279 | 153.064542451 | 457 |
| Antipodal-Fisher RF0 | af | 11.5787310465 | 154.265625949 | 467 |
| Mass-Fisher RF0 | aj | 11.5461593427 | 166.290106153 | 494 |
| Joint-column Fisher RF0 | ap | 11.5780405498 | 152.872086740 | 526 |

The matched-reference historical Hellinger RF10 PSNR is11.5296243252;
Hellinger RF0 BF16 control g has11.4994072668. Anchor-Fisher's gain over
geometric Fisher32 is0.0188694028dB and its gap to12 is0.3635915721dB.
These are one deterministic benchmark, not a general superiority claim.

Anchor-Fisher's completed al metric suite reports SSIM0.3635055850386802
and LPIPS0.5050196406649955 across20 videos/1620 full-resolution frames.
Matched-reference historical Hellinger RF10 has SSIM0.3579599716779665
and LPIPS0.5059471754012285: all three quality metrics improve slightly,
but the PSNR target and no-slowdown requirement still remain unmet.

The observed GCA column times the entire attention replacement, including
selection, merge and route. The separate Wan self-attention-module time
also includes QKV/output projections and normalization/RoPE. For Anchor,
that is196.208070636ms/call, or174.633381829 excluding the first video;
GCA alone is142.669590348ms/call when excluding the first video. Both
scopes contain run instrumentation and must not be called matched speedups.

Full-metric evaluations are ag(n), al(ae), am(af), ar(aj), as(ap).
Primary artifacts and verified archives reside in same-named run folders
under C:/GMA/experiments. The sole exploratory workbook is C:/GMA/tmp.xlsx.
SSIM/LPIPS are published only after complete guarded20-video evaluation;
VBench and matched SDPA speedup remain unmeasured. Check each metric run's
completion.json before treating a partial results.json as final.

The late exact tangent-advantage GS screen aq gives MSE0.0014112103721
on the3 original all-frame L33 captures, worse than the matched reference.
It has no native/video follow-up. Neighbor-column prototype an and native
ao also worsened this local output check; the complete ap video confirms
they do not improve on current Fisher32/Anchor-Fisher. Do not rerun these
as if they were untested new directions.

## Scope choice before another method family

The most concrete next proposal is to score a fixed small set of query
frames jointly while keeping one shared real-row set and each frame's
unchanged actual-Q merge. This expands what the selector observes and is
not strictly RF0-only, even if its output table has one frame dimension.
The user has not yet authorized that change. Do not implement it merely
by keeping the dispatch name RF0. The information-limit note gives only
a generic counterexample to universal RF0 optimality, not a PSNR12 barrier.
Further strict-RF0 research is possible; these results do not prove that
additional query-frame information is necessary to meet the empirical goal.
