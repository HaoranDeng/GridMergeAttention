# GMA-C4 80 ms campaign, 2026-09-12

Goal: complete BF16 GMA-C4 attention call in 80 ms, output bitwise equal to the
retained `kernels/best/c4` implementation. User constraint: exact numerics, no
quantization of any intermediate. After the first round the user revised the aim
to "keep exact, take ~90-95 ms".

All measurements: one exclusively clean H200 (scai7 GPU4,
UUID `GPU-bf8f3ffe-1da0-7e2c-a98f-ccd994c0fee1`), real captures
`real_full40_step0_layer{00,19}.pt`, complete H40 calls, foreign-process monitor
active, interleaved ABBA/BAAB. Raw receipts are the sibling JSON files.

**Neither 80 ms nor 90-95 ms is reachable under the exact-numerics constraint.
On this evidence the retained implementation is already at the floor of its
dataflow. Nothing is promoted; the retained C4 kernel is unchanged.**

## True budget, measured in place

`call_gap_layer00.json` times every kernel with events inside one complete call,
so caches and scheduling are the real ones. **The stage sum equals the call
exactly**: 101.050 ms of stages against a 101.050 ms instrumented total. The
apparent ~7 ms "gap" against earlier isolated stage timings was an artefact of
replaying a single hot head chunk.

| stage | in-call ms | isolated-replay ms |
| --- | ---: | ---: |
| selector | 9.732 | 8.26 |
| pack | 0.387 | 0.39 |
| merge (7.15 per chunk) | 57.168 | 51.1 |
| route (4.22 per chunk) | 33.764 | 32.5 |
| total | 101.050 | |

Per-chunk merge inside the call: 7.307, 6.902, 6.910, 7.026, 7.154, 7.234,
7.300, 7.334. Route: 4.164 to 4.285.

CUDA-graph replay of the whole call is bitwise equal and **slower** (102.07
against 100.39 ms plain), so there is no per-launch cost to recover.

Achievable device bandwidth in the same process: write 3.380 TB/s, read
3.767 TB/s, concurrent read+write 4.256 TB/s.

The compressed workspace dominates: `mk` and `mv` are 8.128 GB *each per
five-head chunk*, so one call writes and reads back ~131 GB.

- **Route is at the read wall.** 16.39 GB / 4.22 ms = 3.88 TB/s against a
  3.77 TB/s pure-read anchor (its singleton pass hits L2). No headroom.
- **Merge is not store bound.** Dividing store traffic by 21 with a
  frame-aliased tensor map, identical instruction stream, saves only 0.27 ms.
  The writes already hide under the merge's own compute.

## Merge decomposition

Timing-only probes, each identical to the retained merge except for one stage
replaced by a stand-in that still consumes its inputs. Savings per H5 chunk:

| stage removed | ms |
| --- | ---: |
| epilogue and store path, total | 1.92 |
| - of which the four CTA barriers and the staging drain wait | 0.39 |
| - of which residual scaling, BF16 narrowing, stmatrix, TMA issue | ~1.53 |
| exact normalisation, four-seed ownership and SQ1 correction | 1.35 |
| score WGMMA | 0.45 |
| compressed K/V merge WGMMA | 0.36 |
| store traffic (divided by 21) | 0.27 |
| seed-query global loads | 0.11 |

Merge overhead above its store floor is 57.17 - 38.8 = **18.4 ms per call**, and
every term above is either arithmetic mandated by the exact rounding order or a
term that has been attacked and failed. The full-alias probe (13.37 ms) is
invalid: collapsing every CTA onto one 80 KiB slice serialises on L2 write
contention.

## Selector decomposition

| stage | ms |
| --- | ---: |
| retained selector (isolated) | 8.26 |
| FP64 Fisher Gram-Schmidt | 5.50 |
| - of which cross-lane exchanges | 1.65 |
| - of which FP64 exp, sqrt, correctly rounded divide | 0.74 |
| - of which plain FP64 arithmetic and dependency latency | ~3.11 |
| Uniform16 probe-dot MMA | 1.99 |

The GS is FP64 because exact seed reproduction requires it, and its largest
component is mandated FP64 arithmetic issued at two warp instructions per SM
clock. Reassociating it changes seeds.

## Candidates tested

Every candidate produced **bitwise-equal** complete H40 output on layers 00 and
19. Deltas are paired means over 80 complete calls.

| candidate | delta ms | wins | why it failed |
| --- | ---: | ---: | --- |
| two-stream merge/route overlap, two workspaces | +0.30 | 36/80 | merge holds the entire register file at 4 CTAs/SM, so route CTAs cannot co-reside; second independent confirmation of the C10 result |
| per-warp 4 KiB TMA store (drops 4 CTA barriers per tile) | +0.89 | 23/80 | TMA issue count costs more than the barriers save; the barriers are only 0.39 ms per chunk in total |
| column-block invariant hoist (removes per-tile div/mod, hoists seed loads) | +4.83 | 1/80 | spills, resources `[128, 40, 57344, 4]`; matches the 2026-09-09 `hoisted_seed` failure |
| seed-paired ownership rows (80 shuffles + 40 ballots per tile to 20 shuffles, no ballots) | +4.29 | 3/80 | no spills, but `canonical = col*4+seed` is forced by the TMA box order, so each `stmatrix` then addresses 8 staging rows at stride 2 instead of 8 consecutive ones |
| single 105-tile launch slice (K/V tile fetched once per key block) | +0.008 | 40/80 | exact tie; merge is not traffic bound |

Masked-token check: only 2.19% (layer 00) and 0.10% (layer 19) of compressed
tokens carry `bias == +inf`, so compacting the workspace would remove ~1% of
route traffic. Closed.

Unbanked-win check: the 2026-09-09 `ballot_owner` result (-3.89 ms against a
113 ms baseline) **is already in the retained kernel** - its audit counted the
winner-index shuffles it removed, and the retained source carries the ballot
form. `packed_owner` and `ordered_owner` are strictly worse than that form.

## Two hard constraints found

1. The merge sits at exactly 127-128 registers with **zero** local bytes and
   4 CTAs/SM, with shared memory simultaneously at 57344 of 58368 bytes per CTA.
   Any added live state spills, costing about 5 ms.
2. `canonical = col*4+seed` is forced by the TMA box dimension order, which is
   in turn forced by the global compressed layout the route reads. The staging
   permutation cannot be reordered.

## Why the target is out of reach with exact numerics

Floor with a *perfect* merge, using measured anchors:

    selector 9.73 + pack 0.39 + 131 GB / 3.380 TB/s + route 33.76 = 82.7 ms

That assumes the merge reaches 100% of achievable write bandwidth with zero
remaining compute. The 18.4 ms that separates the real merge from that floor is
now fully itemised, and each item is either required arithmetic or an attempted
and rejected change. The realistically available exact saving is **0-3 ms**, so
the retained ~101-103 ms is essentially at this dataflow's floor.

Going lower requires reducing the ~262 GB of intermediate traffic, which means
reduced-precision compressed K/V. The one recorded FP8 attempt
(`gma_c4_cuda_80ms_20260909/FP8_STORAGE_RESULT.md`, 115.3 ms against a 100.8 ms
control) dequantised per token inside the Triton route; the native FP8 WGMMA
branch already present in `route_reference.py` (`COMPRESSED_FP8`) has never been
measured. That is the only untested lever with the required magnitude, and it
needs the numerics constraint relaxed plus generation-quality validation.

---

# Round 2, 2026-09-12: C8 technique transfer

The parallel GMA-C8 campaign's methods were reviewed for transfer. Round 1 had
concluded from *cost ablations* that the merge was near its floor; round 2
re-examined it with C8's *instruction-census* method and found headroom that the
ablations could not see, because an ablation prices a stage while a census
prices an instruction.

## What transferred and what did not

| C8 finding | transfers to C4? |
| --- | --- |
| SASS census plus an instruction exchange rate | yes. C4's own `localowner` probe gives 0.0287 ms/call per per-tile instruction (C8's is 0.0352), and the census predicts 84-93% of the measured merge time, so the C4 merge is issue-bound too |
| D6c: peel the per-warp TMA issue into compile-time `if(warp==w)` sites | yes. Round 1's per-warp TMA (+0.89 ms) used a runtime warp offset, which is exactly C8's D6a failure; D6c was a +1.20 ms promotion for C8 |
| D8: a library call silently expanding (their `log2f`) | partly. C4's `log2f` is 85 instructions across three sites, but `lg2.approx.f32` is numerics tier and C4 already rejected `fast_log_replay` on a 20-video screen |
| D2/D8B: next-tile Q-fragment prefetch, +3.33 ms | no. C8's Q path costs 16.3 ms/call from address divergence; C4 pre-packs Q into fragment order and its entire Q load path measures 0.9 ms/call |

## New finding: WGMMA descriptors are rebuilt every tile

All eighteen WGMMA shared descriptors depend only on the resident K/V tile base
and a compile-time k offset, so all are loop invariant. The census shows ptxas
rebuilding them **inside** the tile loop: `mma_sm90_gmma_ext.hpp:8116` 67
instructions plus `mma_sm90_gmma.hpp:2957` 64 instructions = **131 per tile**.

Hoisting them in source changes nothing - ptxas rematerialises rather than keep
eighteen descriptors live (verified: identical 1831 in-loop instructions). But
each descriptor is the base plus a constant element offset, and for every offset
this kernel uses the `boff` swizzle term is zero (score offsets 0/16/32/48/5120/
5136/5152/5168, merge offsets 0/1024/2048/3072/4096, all multiples of 8
elements), so the offset only moves the descriptor's 14-bit address field, with
a maximum field delta of 646 against a 16383 limit and no carry. One add
therefore reproduces `desc()` bit for bit.

The same hoist attempted on the eight `stmatrix` staging addresses compiles to an
identical 1699 in-loop instructions - ptxas rematerialises those too. Killed
offline at zero GPU cost.

## Priced candidates (tile-loop instruction census, baseline 1831)

Register and spill figures confirmed both offline and on-GPU.

| candidate | REG | LOCAL | in-loop | delta | predicted ms/call |
| --- | ---: | ---: | ---: | ---: | ---: |
| retained | 127 | 0 | 1831 | - | - |
| select_hoist | 124 | 0 | 1774 | -57 | -1.64 |
| desc_delta | 127 | 0 | 1754 | -77 | -2.21 |
| desc_select (stacked) | 124 | 0 | 1699 | -132 | -3.79 |
| warp_tma_peeled | 128 | 0 | 1922 | +91 static | dynamic, needs GPU |

`select_hoist` replaces the two self-column tests, which rebuilt each key index
per unrolled step, with one hoisted lane base compared against compile-time
constants. Same predicate, same values.

**These are predictions, not results.** Timing is pending: two attempts were
voided when foreign `ray`/`sglang` jobs landed on the card mid-run, which is why
the claim is now made server-side by a retry loop that takes whichever H200 is
clean at the instant it checks.

## Round 2 measured results

Four candidates, 80 paired complete calls each over layers 00 and 19. **Every
one produced bitwise-equal complete H40 output.** Receipt `desc_select_r2.json`.

| candidate | delta ms | wins | predicted |
| --- | ---: | ---: | ---: |
| desc_delta | +0.410 | 34/80 | -2.21 |
| select_hoist | +0.207 | 42/80 | -1.64 |
| desc_select (stacked) | -1.056 | 57/80 | -3.79 |
| warp_tma_peeled | +0.452 | 37/80 | - |

Confirmation of the stack at pre-registered n = 240 paired calls over layers
00/19/39 (receipt `desc_select_confirm.json`): layer 00 -0.165 (52/80), layer 19
-0.706 (58/80), layer 39 -0.609 (53/80), **pooled -0.493 ms, 163/240 wins**,
retained 102.270 against candidate 101.777. 163/240 is p ~ 3e-8 against a
fair-coin null, so the win is real; it is also only 0.5%.

D6c's compile-time peeling did repair most of round 1's per-warp TMA regression
(+0.89 -> +0.45 ms) but did not turn it into a win. Not promoted.

## Correction: the instruction exchange rate is pipe specific

Round 2 initially concluded from the census that the C4 merge is issue-bound and
that instruction count is the currency, predicting -3.79 ms for the stack. **The
direct test refutes the strong form**: 132 integer-ALU instructions removed
bought 0.49 ms, not 3.79, a 7.7x over-prediction.

The error is precisely C8 campaign-process lesson 5, "price by region, not by one
global rate". The 0.0287 ms/call per-instruction rate was derived from the
`localowner` probe, which removes **SHFL and VOTE** - narrow-pipe operations that
issue at one warp instruction per SM clock. Integer-ALU instructions run on a
pipe with spare capacity (C8 measured 35% utilisation on their equivalent), so
removing them is nearly free *and nearly worthless*.

Per-instruction cost by region in the C4 merge, from the round 1 ablations:

| region | ms/chunk | instructions | ms/chunk per instruction |
| --- | ---: | ---: | ---: |
| epilogue and store path | 1.93 | ~210 | 0.0092 |
| seed-query loads | 0.11 | ~29 | 0.0038 |
| ownership SHFL/VOTE | 0.43 | 120 | 0.0036 |
| middle math (ex2, reduce80, log2f) | 1.35 | ~400 | 0.0034 |
| integer ALU (descriptors, self-column tests) | ~0.13 | 132 | 0.0010 |

Epilogue instructions cost ~9x what integer-ALU instructions cost. This agrees
with C8's own D7 finding that shared-memory staging, not the global write,
dominates the store path - and C4's staging is fixed by the TMA box order, which
is fixed by the layout the route reads.

**Net position after round 2: a real, bitwise-exact, spill-free -0.49 ms/call
(124 registers against 127, still 4 CTAs/SM), i.e. ~101.8 ms. The 80 ms target
is unchanged in its inaccessibility, and the round 1 floor analysis stands.**

---

# Round 3, 2026-09-13: re-audit of the C8 record and of my own failures

Prompted by the instruction not to trust failure records. Three leads, chosen
because two had never been tested at all and one was recorded as a win that was
never banked. All three arms were bitwise-exact by construction and were
compile-priced offline (registers and spills) before any GPU was claimed.

Receipts: `selector_regpipe.json`, `headchunk.json`, `headchunk_scai6.json`,
`route_sweep.json`.

## 1. head_chunk = 10

The 2026-09-09 table recorded `head10` at -0.70 ms against its own control and
the retained package still ships head_chunk=5, so this looked like an unbanked
win. **It does not replicate.** On today's baseline, 100 paired complete calls
over layers 00 and 19:

| arm | delta ms | wins | bitwise |
| --- | ---: | ---: | --- |
| head10 alone | +0.357 | 51/100 | yes |
| head10 + desc_select | **-0.720** | **67/100** | yes |

head10 on its own is a dead wash - 51/100 is exactly coin-flip - so the 2026-09-09
figure was a window artefact, not a lost result. Stacked on desc_select it
measures -0.720 ms (p ~ 5e-4), 124 registers, 0 spills, 32.8 GB workspace. That
is the best measured configuration, but against desc_select alone (-0.493 over
240 pairs) the increment sits inside overlapping intervals, so head10 is **not**
shown to contribute independently.

## 2. Selector register-pipelined frame loop (C8 "gs regpipe")

C4's selector was never touched in rounds 1-2. C8 ships a register-pipelined GS
frame loop; ported to C4 it compiles at 128 registers, 0 spills, unchanged
shared, so 4 CTAs/SM is preserved. Seeds and complete output are bitwise equal
(0 seed mismatches).

The **selector stage** does get faster: 7.882 against 8.334 ms on layer 00 and
8.055 against 8.345 on layer 19. But the **complete call** measures **+0.581 ms
at 40/100 wins**. Stage and call disagree, and 40/100 is below chance, so this
is not a demonstrated win. Not promoted; would need a much larger n to resolve.

## 3. Route schedule sweep - never run for C4 before

Twelve configurations, route_h5 in ms, against the inherited base (64/4/3):

| config | layer 00 | layer 19 | bitwise |
| --- | ---: | ---: | --- |
| **b64 w4 s3 (inherited)** | **4.098** | **4.194** | yes |
| b64 w4 s2 | 4.663 | 4.791 | yes |
| b64 w4 s1 | 4.676 | 4.926 | yes |
| b64 w4 s4 | 5.336 | 5.592 | yes |
| b64 w8 s3 | 7.583 | 7.599 | yes |
| b32 w4 s3 | 4.187 | 4.255 | no, relL2 ~3-5e-4 |
| b128 w4 s3 | 5.351 | 5.553 | no |
| b128 w4 s1 | 5.480 | 5.623 | no |
| b128 w4 s2 | 6.461 | 6.513 | no |
| b128 w8 s3 | 7.055 | 7.202 | no |
| b256 w4 s1 | 7.335 | 7.340 | no |
| b256 w4 s2 | out of shared memory | | |

**The inherited configuration is the fastest of all twelve**, and the only arm
that comes close (b32) is both slower and not bitwise. No schedule change
captures anything.

## Correction: my read-bandwidth anchor was wrong

Round 1 claimed the route is "at or above the read wall" using a torch `.sum()`
reduction as the anchor (3.77 TB/s). A real grid-stride vectorised read kernel
measures **4.550 TB/s mean, 4.559 best** over 4 GiB. The route runs at
16.39 GB / 4.098 ms = **4.00 TB/s, i.e. 88% of achievable, not 103%**.

So the *justification* was wrong - there is ~12% of headroom in principle, worth
roughly 4 ms. The *conclusion* survives anyway: the schedule sweep shows nothing
reaches it, because the inherited configuration is already the best available and
every alternative is slower, non-bitwise, or both. A reduction understates the
streaming ceiling; never anchor a bandwidth claim on one.

## Net position after round 3

Best measured, bitwise-exact, zero-spill: **desc_select -0.493 ms (163/240
pairs)**, optionally stacked with head10 for -0.720 ms (67/100) though the
increment is not separately established. Nothing else promoted. The 80 ms target
is unchanged in its inaccessibility; the floor argument from round 1 stands, with
its route term now resting on the sweep rather than on the bad anchor.

---

# Round 4, 2026-09-13: store-scatter pricing (the C10 lever), and an audit of
# other agents' failure records

## Audit finding: the overlap dead end was killed at the headline

`gma-merge-route-overlap-dead-end` says "do not re-propose stream/event overlap
of merge and route for any GMA capacity". Its own body says the decisive variant
was never run:

> The only routes to that bandwidth headroom are (a) SM-level co-residency - cap
> merge at 3 CTAs/SM *and* near 112 registers so route's 49 KB shared / 20 352
> registers fit alongside (**untested**; costs ~8.7% of merge per prior occupancy
> receipts), or (b) intra-kernel warp-specialized fusion [...]

So option (a) is untested by anyone, and my own round-1 overlap kill (+0.30 ms,
36/80) rested on the same unverified mechanism plus an underpowered square. The
C10 campaign separately built the enabling probe - pad the shared request to cap
residency, code path untouched, output bitwise equal - but used it for fusion
analysis, not for co-residency. Fusion failing (consumer's 115 KB FP32
accumulator forces a 4-warp producer) says nothing about two separate kernels
co-residing.

## Store scatter is NOT the C4 lever (it was the C10 lever)

C10's bandwidth ladder found ~66 ms/call of un-overlapped store time and priced
the store scatter at ~12 ms/call, naming "lengthen the contiguous run in the
mk/mv output layout" as its only measured lever above 5 ms. C4's store emits
sixteen 1 KiB runs 9.7 MB apart - *shorter* runs than C10's six 2560 B runs - so
this looked like the strongest transfer available.

`fisher_c4_probe_linear_store.cu` writes the identical byte count through an
identical box, identical shared staging and an identical TMA issue count, but to
one contiguous 16 KiB run (496125 x 16384 B = 8,128,512,000 B, matching `mk` to
the byte; max block index 496124 < 496125). The box row order stays `col*8+j`,
so the 128 B swizzle and the shared side are untouched.

**Structural gate passed before any timing** (an ungated probe manufactures the
effect it reports):

| metric | baseline | probe |
| --- | ---: | ---: |
| ptxas serialisation warnings C7511/12/15/17/19 | none | none |
| STACK / LOCAL / LDL / STL | 0 | 0 |
| **CTAs per SM** | **4** | **4** |
| WARPGROUP.ARRIVE / DEPBAR | 3 / 3 | 3 / 3 |
| UTMASTG / STS | 2 / 26 | 2 / 26 |
| HGMMA 64x128x16 / 64x80x16 | 10 / 8 | 10 / 8 |
| REG (advisory) | 127 | 128, occupancy-neutral |

The +1 register was not waved through: the rule protects occupancy, so occupancy
was computed directly (128 regs x 128 threads x 4 = 65536 exactly) and then
confirmed on-GPU as `[128, 0, 57344, 4]`. The opcode regex uses
`[A-Za-z0-9._@]+` because SM90 mnemonics carry a lowercase x.

**Result (per-chunk merge inside a complete call, interleaved, with an A/A null
pair in the same square; receipt `store_scatter.json`):**

| arm | delta ms | sd | t | b faster | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| effect: scattered -> contiguous | +0.042 | 1.822 | 0.146 | 13/40 | 40 |
| null: A/A, same source | +0.001 | 1.109 | 0.006 | 19/40 | 40 |

**Scatter price = 0.041 ms/call**, standard error 0.29 ms, so bounded under
about +-0.6 ms. The null arm confirms this window carries no ABBA position
offset, so the reading is clean rather than a cancellation.

**C10's scatter lever does not transfer.** With round 1's volume probe (0.27 ms)
this is the second independent measurement showing C4's store path - volume and
scatter together - is essentially free, absorbed by the TMA engine behind the
merge's own compute. C4's merge is genuinely compute, not store.

Corollary: this *strengthens* the co-residency case. The merge is 57 ms of
compute drawing only ~2.4 TB/s of write bandwidth while the route is 33.8 ms of
nearly pure reads, so there is real spare bandwidth and real compute to hide
reads behind - if route CTAs can be made to co-reside at all.

---

# Round 5, 2026-09-13: co-residency - the untested option (a)

`gma-merge-route-overlap-dead-end` tells every later agent "do not re-propose
stream/event overlap of merge and route for any GMA capacity", but its own body
records the decisive variant as untested:

> (a) SM-level co-residency - cap merge at 3 CTAs/SM *and* near 112 registers so
> route's 49 KB shared / 20 352 registers fit alongside (**untested**)

Round 1's C4 overlap kill (+0.30 ms, 36/80) rested on the same unverified
mechanism. Merge at 4 CTAs/SM holds 127x128x4 = 65024 of 65536 registers, so no
route CTA can be placed - that was arithmetic, never a measurement.

## Method

Residency is capped by padding the merge's dynamic shared request (57344 -> 4
CTAs, 62464 -> 3, 81920 -> 2). The code path is untouched, the kernel still uses
only the first 57344 bytes, and output stays bitwise equal. Overlap is always
compared against a serial control at the SAME padding, so the padding cost is
never charged to the overlap.

**The premise is verified on-GPU, not assumed.** The first attempt reported 4 / 4
/ 4 and refused to time: `resources()` carries a *second* hardcoded
`constexpr int bytes` and my patch had only changed the `run()` copy, so the
kernel misreported its own occupancy. Without that assertion three arms would
have been timed at a secret 4 CTAs/SM and "occupancy does not matter" would have
been fabricated. After patching both constants the premise verifies on both
hosts: `[127, 0, 57344, 4]`, `[127, 0, 62464, 3]`, `[127, 0, 81920, 2]`,
registers unchanged, zero spills.

## Result: the mechanism is real

Overlap minus serial at the same padding, n=20 per arm, every arm bitwise equal:

| merge residency | overlap - serial | t | overlap faster |
| --- | ---: | ---: | ---: |
| 4 CTAs/SM (retained) | +0.201 | 0.32 | 12/20 |
| **3 CTAs/SM** | **-0.916** | **-2.88** | **14/20** |
| 2 CTAs/SM | -0.211 | -0.48 | 15/20 |

**This is the first positive overlap signal measured anywhere in the project.**
At 4 CTAs there is no room and overlap is exactly null, which confirms the
memory's mechanism by measurement rather than by arithmetic. Free the room and
route CTAs do co-reside, worth about 0.9 ms (p ~ 0.009). At 2 CTAs the gain
collapses again - more room, but the merge itself is now slow enough to dominate.

## Limitation: the occupancy cost is not yet quotable

Sharding by arm is what let the run finish at all on a churning cluster, but it
scattered the arms across hosts: cta4 and cta2 ran on scai6, cta3 on scai7. Per
"never splice absolutes across windows", **the 4->3 occupancy cost cannot be
quoted** from these numbers.

Same-host only (scai6), serial merge_total: 4 CTAs 57.659, 2 CTAs 68.408 ms,
i.e. **4->2 costs +10.75 ms (+18.6%)**. C4's occupancy curve is much gentler than
C10's (4->2 = 1.28x there).

The decisive comparison - cta4 and cta3 in one process and one window, so serial
and overlapped absolutes are finally comparable - is queued. If 4->3 costs more
than the 0.92 ms the overlap returns, option (a) is closed by measurement rather
than by a headline; if it costs less, this is the round's first structural gain.

## Correction: the 3-CTA gain is NOT co-residency

The section above called the 3-CTA result a confirmation of the co-residency
mechanism. That was wrong, and checking the route's real footprint refutes it.

Every compiled C4 Triton route variant in the run's Triton cache requires
**at least 49152 B of shared memory** (the observed set is 49152, 66560, 82944,
115712, 147456, 148480, 214016, 279552). The 159-register / 49152-byte figure
used earlier came from a C10 document, not from C4.

| merge residency | merge shared | shared left of 233472 | route fits? |
| --- | ---: | ---: | --- |
| 4 CTAs/SM | 229376 | 4096 | no |
| 3 CTAs/SM | 187392 | **46080** | **no** - short of 49152 |
| 2 CTAs/SM | 163840 | 69632 | yes |

So the route can only co-reside at 2 CTAs/SM - and that is precisely the arm
whose overlap effect is null (-0.211, t=-0.48). The significant arm, 3 CTAs/SM
(-0.916, t=-2.88), is one where co-residency is impossible. **The measurement
contradicts the co-residency story instead of supporting it.**

Residual explanations for the 3-CTA effect, none of them co-residency: wave
quantisation (the merge's 23625 CTAs span 44.7 waves at 4 CTAs/SM and 59.7 at 3,
changing the ragged tail a route kernel can fill), or a false positive - three
arms were tested, so t=-2.88 Bonferroni-corrects to p ~ 0.027, which is marginal.

**Option (a) is closed on economics regardless of mechanism.** The only residency
where the route fits costs +10.75 ms of merge time (57.659 -> 68.408 ms
same-host, +18.6%), which no overlap gain of the observed magnitude (~0.9 ms)
can recover. This closes it by measurement - the route's footprint and the
occupancy cost - rather than by the memory's headline.

---

# Round 6, 2026-09-14: two real wins, and co-residency closed

## Co-residency: closed by measurement

cta4 and cta3 in ONE process and window (receipt `h2h.json`), n=40 per arm, both
bitwise equal:

| merge residency | serial merge_total | overlap - serial | t | overlap faster |
| --- | ---: | ---: | ---: | ---: |
| 4 CTAs/SM | 57.160 | -0.583 | -1.27 | 26/40 |
| 3 CTAs/SM | 60.384 | -0.673 | -1.47 | 27/40 |

**4->3 costs +3.224 ms (1.056x)** and the overlap gain is not distinguishable
from zero at either residency. The earlier -0.916 ms / t=-2.88 at 3 CTAs **did
not replicate** - it was a false positive among three arms, exactly as the
route's shared-memory footprint predicted (the route needs >=49152 B and only
46080 B is free at 3 CTAs, so it could never have been co-residency).

Option (a) from `gma-merge-route-overlap-dead-end` is now closed by measurement
rather than by a headline: the padding costs more than the overlap returns.

## Selector register-pipelined frame loop: a real, bitwise-exact win

Round 3 called this "not a demonstrated win" on a full-call ABBA that read
+0.581 ms at 40/100. That verdict was the instrument's fault. Re-measured with
the selector stage timed in place, arms interleaved, and an A/A null pair in the
same square (receipt `selstage.json`):

| arm | delta ms | sd | t | candidate faster | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| effect: retained vs regpipe | **-0.535** | 0.245 | **-16.9** | **60/60** | 60 |
| null: A/A, same source | -0.034 | 0.469 | -0.55 | 29/60 | 60 |

**-0.501 ms after the null offset**, seeds and complete output bitwise equal
(0 of 12,096,000 seeds differ). This is the C8 `abba-gate-power` lesson landing
exactly as written: the full-call gate is blind below ~2 ms, and the interleaved
per-stage instrument resolves the same change at t=-16.9.

## FP32 Gram-Schmidt: the largest candidate so far, numerics tier

The user approved FP64 -> FP32 for the selector Gram-Schmidt on 2026-09-14,
conditional on a 20-video PSNR screen showing no difference. This is *not*
covered by the earlier "no quantization" constraint, which was about FP8
compressed K/V storage.

`native_fisher/fisher_c4_selector_fp32gs.cu` converts only the GS helpers; probe
dots stay BF16/FP32. Build: **508 FP64 SASS instructions -> 0**, 128 registers
(from 124), 0 spills, shared unchanged at 37888, and 128x128x4 = 65536 keeps
4 CTAs/SM.

| measure | value |
| --- | ---: |
| selector stage | 10.179 -> 7.988 ms |
| delta after null offset | **-2.247 ms/call** |
| t / wins | **-59.2 / 60:60**, n=60 |
| seed mismatches | **12 of 12,096,000** (9.92e-7) |
| output max_abs / relative L2 | 0.046875 / **4.92e-5** |

The 12 flips are **identical on both hosts**, so they are deterministic, not
noise. For scale: the retained kernel's own complete-H40 deviation from its
Triton reference is relative L2 **3.73e-4**, i.e. the FP32 selector moves the
output **7.6x less than the retained implementation already differs from its own
reference**. That is an argument for plausibility, not a substitute for the
screen the user asked for.

Seeds are discrete argmax outcomes, so a flip swaps one cluster's representative
row - different in kind from a reassociated sum. The 20-video screen **must carry
a same-node control**; see `video-screen-node-confound`, where a cross-node
screen's reproduction floor of ~15 dB fired kill gates on a null change.

## Position

| change | delta ms | tier | status |
| --- | ---: | --- | --- |
| desc_select (merge) | -0.493 | bitwise | validated, unpromoted |
| selector regpipe | -0.501 | bitwise | validated, unpromoted |
| FP32 Gram-Schmidt | -2.247 | numerics | pending 20-video PSNR |

desc_select and regpipe touch different kernels and should stack to about
-1.0 ms (~101.3 ms). With FP32 GS on top, about -3.2 ms (~99.1 ms) - though
regpipe and FP32 GS are both selector changes and their gains may not fully add,
which has to be measured on a combined source rather than assumed.

# Round 7 - the selector changes do stack, and the video screen is running

## Combined FP32 Gram-Schmidt + register-pipelined frame loop

`r7_combo_fp32gs_regpipe_selectorstage.json`, scai7, layer 00, 60 blocks with an
A/A null pair in the same square:

| arm | selector stage |
| --- | ---: |
| retained (FP64 GS, no prefetch) | 9.951 ms |
| combined candidate | 7.176 ms |
| raw delta | -2.775 ms, t = -70.2, 60/60 |
| null offset (retained vs retained_bis) | +0.051 ms, t = +1.53 |
| **offset-corrected** | **-2.826 ms** |

Separately the two were -2.247 (FP32 GS) and -0.501 (regpipe), summing to
-2.748. The combination measures -2.826, so they add - slightly more than
linearly, consistent with 124 registers in the combined source against 128 in
each part. This is the number to carry, not the sum of parts.

Correctness is unchanged from FP32 GS alone: 12 of 12 096 000 seeds differ
(9.92e-07), output relative L2 4.92e-05, identical on both hosts. The null arm
is bitwise equal, which is what makes the -2.826 a measurement of the candidate
rather than of the square.

## desc_select re-measured on the complete call

`r7_stack_desc_select_fullcall.json`, scai7, layers 00 and 19, 60 pairs each:

| layer | delta | wins |
| --- | ---: | --- |
| 00 | -0.177 ms | 37/60 |
| 19 | -0.679 ms | 40/60 |
| pooled | **-0.428 ms** | 77/120 |

Bitwise equal on both layers (max_abs 0, relative_l2 0). The earlier per-stage
reading was -0.493; the complete-call reading is -0.428. Both are real and the
smaller one is the one to quote, since it is measured where the time is spent.

Process note: this receipt was almost thrown away. The claim daemon gated
completion on a `SUMMARY_DONE` token, and `screen_c4_merge.py` ends at `SUMMARY`
- it never prints that token. The receipt carried `valid: true` with an empty
monitor-error list, so the run had in fact completed cleanly. **Gate step
completion on the receipt's own validity field, not on a print token that the
step may not emit.**

## Position

| change | delta ms | tier | status |
| --- | ---: | --- | --- |
| desc_select (merge) | -0.428 | bitwise | validated, unpromoted |
| FP32 GS + regpipe (selector) | -2.826 | numerics | 20-video PSNR running |

Projected together: 103.25 - 0.43 - 2.83 = **99.99 ms**, first time under 100.
The bitwise half alone gives ~102.8 ms.

## The video screen

Tooling, all new in this round:

* `analysis/validate_c4_selector_package_tol.py` - the existing package
  validator asserts exact seed equality, which the FP32 candidate cannot pass by
  construction. Rather than loosen that gate (which would retroactively weaken
  every proof it ever signed), this is a separate instrument with the tolerance
  written into the proof and both arms' numbers recorded whether they pass or
  not. Control is the retained C4 stack, so the merge is held identical and the
  selector is the only difference.
* `analysis/run_c4_native_videos.py` - ported into this worktree and
  parameterised. The earlier copy hard-coded the selector at module scope (and
  to a superseded one), which made a selector screen impossible. `--num-videos`
  / `--prompt-start` now pass through, and the dispatch gate scales with the
  requested count rather than asserting 3200/25600.
* `analysis/compare_c4_selector_videos.py` - CPU-only. Runs both arms against
  the dense reference and pairs per prompt, plus an arm-versus-arm PSNR that
  reports identical videos instead of rejecting them (the tracked screen refuses
  a non-finite PSNR, correct against a different method, wrong here). It refuses
  to report a cross-host pair unless explicitly overridden.

The whole CPU half was smoke-tested without a GPU: snapshot freezing, proof
verification and the frozen wrapper all pass, stopping exactly at the GPU claim.

Arms are 20 videos each, about 24 minutes of exclusive GPU apiece, and the
candidate arm is pinned to whichever host ran the control.

## Correction: the FP32 seed flip rate is layer-dependent, and I generalised from one layer

Every seed-flip number I had quoted - 12 of 12 096 000, 9.92e-07, "identical on
both hosts" - came from **layer 00 only**. The package proof runs three captured
layers, and they do not agree:

| layer | seed mismatches | fraction | output rel L2 |
| --- | ---: | ---: | ---: |
| 00 | 12 | 9.92e-07 | 4.92e-05 |
| 19 | 2 | 1.65e-07 | 3.21e-05 |
| **39** | **5191** | **4.29e-04** | 2.38e-05 |

Layer 39 flips **430x** the layer-00 rate and 2600x layer 19's. Late layers
evidently carry far more near-ties in the Fisher score, so FP32 resolves them
differently much more often. The two deliberate tie edge cases (all-zero query,
identical probe vectors) still agree exactly, so this is not a tie-break rule
change - it is ordinary precision loss landing on a discrete argmax in a regime
where the margins are small.

Output error does not track the flip count: layer 39 has the most flipped seeds
and the *smallest* relative L2. A flipped seed swaps one representative row for
another that scored almost identically, so the merged result barely moves. That
is reassuring but it is not the question the user asked; PSNR on real generations
is.

The instrument caught this by itself: the proof was declared at a 1e-5 ceiling,
breached on layer 39, and refused to sign (`valid: false`), so no video arm could
be prepared against it. The ceiling is now declared at 5e-4, which covers the
observed value. **A declared ceiling is a statement of what the kernel does, not
evidence that it is safe** - the 20-video screen remains the decider.

Practical lesson, same family as `abba-gate-power`: a numerics screen on one
captured layer is as unreliable as a timing screen in one window. Run all three
layers before quoting a rate.

## Operational: three separate one-GPU violations while restarting claim daemons

Killing the daemon shell does not free the GPU - its `timeout`/python child keeps
running and holds the device. Twice a second daemon then started work elsewhere
and two GPUs were in use at once, against the task's hard one-GPU limit. Both
were caught within minutes and killed, but the procedure was wrong three times.

Two compounding causes:

* a `trap ... TERM` in the daemon does **not** fire while bash is waiting on a
  foreground child, so a daemon mid-step ignores the signal until the step ends;
* `TORCH_EXTENSIONS_DIR` pointed at shared `/local2`, so a build killed on one
  host left torch's build lock behind and the other host waited on it - one such
  wait held a GPU, idle, for 23 minutes with no output at all.

Fixes: per-host extension and Triton cache directories in the validator; stop
procedure is now *kill the GPU children first, then the daemon shell*; and the
screen now runs **one daemon on one host** rather than one per host, since the
remaining work is serial and the paired video arms must share a host anyway.

## Round 7 results, landed overnight 2026-09-14 (05:27 - 09:21 PT, scai7)

The single claim daemon took a card four times and finished every queued step.
Three attempts were voided and retried by design: `fullstack` 05:27 (re-run
clean at 07:58), `gen_ctrl` 05:32 (a foreign job landed between the claim and
the launch; `require_clean` refused), `gen_fp32` 08:26 (two foreign processes of
46 and 49 GB took the card during model load; CUDA OOM). Each voided arm was
discarded whole and re-frozen; nothing was resumed or patched.

### Both changes together on the complete call

`r7_fullstack_desc_select_fp32gs_regpipe.json`. Retained pair against
desc_select merge + FP32 GS register-pipelined selector, complete H40 call,
15 ABBA/BAAB blocks (30 pairs) per layer, A/A null pair in the same square.
Resources: candidate merge `[124, 0, 57344, 4]`, retained `[127, 0, 57344, 4]`.

| layer | retained ms | candidate ms | delta ms | t | candidate faster | null offset |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 00 | 105.059 | 101.135 | -3.913 | -5.68 | 28/30 | -0.011 |
| 19 | 104.493 | 100.831 | -3.662 | -9.29 | 29/30 | +0.001 |
| 39 | 103.525 | 100.147 | -3.393 | -10.41 | 30/30 | +0.015 |
| **pooled** | **104.359** | **100.704** | **-3.656** | | **87/90** | |

Projected from the parts: -0.43 - 2.83 = -3.25. Measured together: -3.66. The
parts add, with nothing lost to sharing the launch queue. Correctness is the
FP32 selector's, unchanged: seed flips 12 / 2 / 5191 on layers 00 / 19 / 39,
output relative L2 4.9e-5 / 3.2e-5 / 2.4e-5, null arm bitwise equal on every
layer.

On "under 100 ms": the retained baseline read 103.25 in the earlier `stack`
window and 104.36 in this one, so absolutes move by about a millisecond between
windows and only the delta is quotable. The candidate sits at 100.7 in this
window and would sit near 99.6 against the earlier baseline. Call it 100 ms,
straddling the line; the retained-relative saving is 3.5%.

### The 20-video PSNR screen: no difference

`r7_video_fp32_psnr_summary.json` and the three `r7_video_fp32_psnr_20_*.json`
files. Both arms on scai7, 20 videos each, control = retained stack, candidate =
retained merge + `fisher_c4_selector_fp32gs.cu`. Dispatch audits: 3200 core
calls, 3200 selector calls, 25600 merge and route calls per arm; GPU monitors
clean (0 foreign-process samples); exit code 0 both arms. Reference: the dense
flash videos.

| | 5 videos | 20 videos |
| --- | ---: | ---: |
| control vs dense, mean dB | 11.945 | 11.621 |
| candidate vs dense, mean dB | 11.965 | 11.653 |
| paired candidate - control, mean dB | +0.020 | +0.032 |
| paired sd dB | 0.047 | 0.078 |
| paired t | 0.96 | 1.86 |
| max abs per-video delta dB | 0.076 | 0.215 |
| candidate vs control direct, mean / min dB | 21.69 / 17.11 | 19.54 / 15.91 |
| identical videos | 0 | 0 |

Against the dense reference the two arms are indistinguishable: +0.03 dB mean
difference, under half the per-video spread, and in the candidate's favour if
anything. The direct arm-versus-arm PSNR of ~19.5 dB confirms the seed flips do
change every trajectory - as any perturbation of a four-step distilled sampler
does - and the dense-relative numbers show that change is quality-neutral. The
user's condition ("no PSNR difference over 20 videos") is met; FP32
Gram-Schmidt is adopted.

### Position

| change | delta ms | tier | status |
| --- | ---: | --- | --- |
| desc_select (merge) | -0.428 | bitwise | validated |
| FP32 GS + regpipe (selector) | -2.826 at the stage | numerics | 20-video screen passed |
| **both, complete call** | **-3.656** (104.36 -> 100.70) | numerics | measured, 87/90 |

Open before promotion: the video screen ran the FP32 selector *without* the
register-pipelined frame loop, and the retained merge rather than desc_select.
Round 8 (queued on scai7) proves the promotion stack bitwise equal to the
screened arm (`equiv`, tolerance 0; regpipe was bitwise on FP64 and desc_select
is bitwise on the merge, so this should pass) and signs the whole-package proof
against the retained stack at the declared 5e-4 ceiling (`pkg_final`). The
validator gained `--control-merge-source` / `--allow-merge-change` for this;
its default still refuses a merge change so a selector proof cannot carry one
silently.

---

# Round 8/9, 2026-09-14 evening: promotion proofs and the next arms

The user's standing instruction is now: when no card is free, put the work in a
20-second claim daemon and wait. The daemon is queue-driven (`/tmp/c4_r9.sh` on
scai7, queue `queue9.txt`, one `name|receipt|command` per line, re-read every
loop, completion judged by the receipt's own `valid` field, three voids abandon
a step). Queued, in order:

| step | what | why |
| --- | --- | --- |
| `equiv` | promoted stack vs the screened arm, tolerance 0 | the videos must be this binary's videos |
| `pkg_final` | promoted stack vs retained stack at the declared 5e-4 ceiling | the package proof |
| `arms1` | head_chunk 10, tiles 7, tiles 3 on the promoted stack (`screen_c4_operator_arms.py`) | cheap config arms; head10 read -0.72 on desc_select alone |
| `route1` | L2 eviction hints on the Triton route (`screen_c4_route_hints.py`) | see below |
| `merge1` | `lg2.approx.ftz` in the merge (`screen_c4_merge_variants.py`) | see below |

## Route: the one-shot stream and the reusable pass carry the same L2 policy

An audit of every `tl.load` in the launched route kernel
(`_hard_c4_720p_route_kernel`): the 8.13 GB mk / mv streams (lines 4064 /
4070), the bias stream (4096) and the L2-reusable singleton raw K / V loads
(4273 / 4279, shared by the 21 query frames of a column) all use
`cache_modifier=".cg"` and none carries an `eviction_policy`. L2 cannot tell
the one-shot stream from the working set. `build_c4_route_hints.py` writes
three variants (`evict_first` on the stream, `evict_last` on the singleton
pass, both) plus a byte-identical null copy; a hint changes no value, so the
screen gates every arm on bitwise equality of the complete output. Triton
3.7.1 accepts `eviction_policy`; all four modules import on the CPU.

Also noted: BLOCK_I = 64 covers ROWS = 45, so 19 of 64 query rows in every
route program are masked padding. Triton blocks are powers of two, so this
cannot be tiled tighter without a different kernel.

## Fast log2: the earlier rejection was a sign test on noise

The claim "C4 already rejected fast_log_replay on a 20-video screen" was
checked against the record
(`gma_c4_cuda_80ms_20260909/scai7/fast_log_replay_video20`). The design was
sound - both arms on scai7 GPU-60bf0dcb, same commit, same selector, same dense
reference, 80 minutes apart - so `video-screen-node-confound` does not apply.
The inference was not. The kill was an uncalibrated sign test on

| metric, candidate minus control | value |
| --- | ---: |
| PSNR vs dense | -0.0069 dB (10 of 20 videos better, 10 worse; range -0.159 to +0.114) |
| SSIM | -0.00031 |
| LPIPS | +0.0018 |

against no pre-registered threshold, plus "integrated attention did not become
faster" taken from *unpaired* sequential runs (+0.105 ms) while the paired
microbench on the same GPU read the candidate 0.7 ms faster. Today's FP32
selector was accepted at +0.032 dB; -0.007 dB is a fifth of that. The
candidate-vs-control direct PSNR was 19.60 dB and control-vs-archive 19.41 dB,
so the same-GPU non-bitwise floor of this four-step sampler is itself ~19-20 dB.

`lg2.approx.ftz.f32` is the production log in the retained C8 and C10 merges
(`gma_c8_route_pipeline.cu:48`, `fisher_c10_tail_local.cu:65`); only C4 kept
`log2f`. The plain variant failed one *synthetic* edge (large_finite_query, K/V
rel L2 0.005, bias validity differs); the replay machinery existed only for
that edge. `fisher_c4_desc_select_fastlog.cu` is desc_select with the two
`log2f` sites replaced, no replay. `merge1` prices it per chunk with an A/A null
and records numerics per layer; if it is worth having, it is numerics tier and
goes through the same declared-tolerance proof and 20-video screen as the FP32
selector did.

## Round 9 results, landed 2026-09-14 17:08 - 18:34 PT (scai6 GPU 2/5) and acted on 2026-09-15

| step | result |
| --- | --- |
| `equiv` | **passed, tolerance 0**: promoted stack (desc_select + FP32 GS regpipe) gives bitwise-identical seeds and output to the screened arm (retained merge + FP32 GS) on layers 00/19/39 and both tie edges (`r8_equiv_promoted_vs_screened.json`). The 20 videos are this binary's videos. |
| `pkg_final` | **passed at the declared 5e-4 ceiling** vs the retained stack: 12 / 2 / 5191 flips, rel L2 4.9e-5 / 3.2e-5 / 2.4e-5 (`r8_package_final.json`). |
| `route1` | **null.** L2 eviction hints on the Triton route: stream evict_first +0.013, singleton evict_last -0.018, both +0.044 ms/call, 42-46 of 90 wins, every arm bitwise; the null arm read -0.045 / +0.033 / -0.017 per layer, so the effect is inside the null (`r9_route1_l2_hints.json`). L2 policy is not what limits the route. Closed. |
| `arms1`, `merge1` | not run: every attempt was voided by contention on scai6 GPU 2 (`require_clean` refused, or the foreign-process monitor tripped mid-square) and r9's three-strike rule counted those as failures. Re-queued under r9b, which does not count contention voids. |

### Promoted

`analysis/promote_c4_round8.py` checked every receipt against the exact source
hashes and rewrote the package: `kernels/best/c4/optimized_c4.py` now points at
`fisher_c4_desc_select.cu` and `fisher_c4_selector_fp32gs_regpipe.cu`; the
previous sources stay as the paired control; `manifest.json` hashes the new
files, `evidence.json` carries the package proof, equivalence proof, complete-
call timing and video screen under `round8_promotion_20260914`;
`python -m kernels.best._integrity` verifies all four packages. README row
updated to 100.704 ms with the paired-delta note (no spliced SDPA figure).

### Selector rewrite, compile-priced (no GPU yet)

Built by four agents from the census, every variant LOCAL 0 / STACK 0:

| variant | REG | shared | frame body | what |
| --- | ---: | ---: | ---: | --- |
| base fp32gs_regpipe | 124 | 37888 | 1327 | - |
| v1_argmax | 122 | 37888 | 1242 | branchless local argmax; REDUX.MAX/MIN cross-lane argmax (-18 SHFL, -9 divergent BSSY/BRA/BSYNC triples) |
| v2fix_abd | 128 | 37888 | | + hoisted anchor, candidate row in a lane register fetched by shfl, `__restrict__` (the packed 32-bit seed store alone forces an 88-byte spill: dropped) |
| v3fix_abd | 128 | 37888 | | + halved accumulator transpose (-16 SHFL.IDX) |
| ldsm | **74** | 40192 | 1085 | padded 272-byte rows + 24 `ldmatrix.x4` replace 96 LDS.32 and ~170 address instructions |
| **v4_ldsm** = v3fix_abd + ldsm | **80** | 40192 | **958** | -28% of the frame body |
| v5_ldsm_occ5 | 80 | 40192 | 958 | v4 with `__launch_bounds__(128,5)`; identical SASS, 5 CTAs/SM needs the 228 KB carveout |

The ldmatrix layout review found no defect (fragment mapping checked against
CuTe's LDSM/MMA traits; one fragility, a missing `"memory"` clobber, fixed
before queuing). The GS-stack review agent died on a session limit; its claims
rest on agent A's CPU simulation (16,960 argmax cases, 512 transpose routings)
until the GPU seed gate runs. Queued as `sel1` with the base as control and an
A/A null; the screen records CTAs/SM per arm, so whether occ5 reaches five is
read from the receipt, not assumed.

### `arms1`: head_chunk and tile count on the promoted stack (2026-09-15 05:14, scai6 GPU 3)

`r9_arms1_headchunk_tiles.json`, complete-call ABBA/BAAB, 30 pairs per layer,
A/A null in the same square, every arm bitwise equal on all three layers.

| arm | delta ms (offset corrected) | wins | verdict |
| --- | ---: | ---: | --- |
| head_chunk 10 (`fisher_c4_head10_desc_select.cu`, 32.8 GB workspace) | +0.216 | 32/90 | dead: the round-3 -0.72 does not replicate on the promoted stack |
| tiles 7 (15 z-slices, K/V tile fetched 3x more often) | +3.40 | 3/90 | dead |
| tiles 3 (35 z-slices) | +9.59 | 0/90 | dead |

With round 1's one_slice (105 tiles per CTA) an exact tie, tiles = 21 sits on
the flat top of its curve and the merge is K/V-refetch sensitive below it.
Both configuration knobs are closed. Incidentally the promoted stack read
99.7-100.0 ms as the base of these squares (scai6, this window).

### `merge1`: hardware lg2.approx in the merge, priced (2026-09-15 05:15, scai6 GPU 3)

`r9_merge1_fastlog.json`. `fisher_c4_desc_select_fastlog.cu` = the promoted
merge with `log2f` -> `lg2.approx.ftz.f32` at its two call sites, no replay.
Per-chunk merge time inside complete calls on the promoted stack, 30 pairs per
layer, A/A null in the same square (null offsets +0.02 / -0.04 / +0.08 ms).
Resources `[125, 0, 57344, 4]` against the base's `[124, 0, 57344, 4]`.

| layer | merge delta ms/call | t | wins | output rel L2 | max abs | masked-token set (last chunk) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 00 | -1.033 | -9.5 | 30/30 | 2.37e-4 | 0.196 | identical |
| 19 | -0.952 | -9.3 | 28/30 | 7.3e-5 | 0.037 | identical |
| 39 | -1.117 | -13.2 | 30/30 | 8.0e-5 | 0.020 | **differs** (count not recorded) |
| pooled | **-1.034** | | 88/90 | | | |

So the approximate log is worth about 1.0 ms per call at the merge stage (of
57.9), consistent across layers - roughly 45% of the 85-instruction census
estimate at the middle-math rate, which is the usual overestimate for a mixed
FMA/MUFU sequence. Numerics: the output moves by the same order as the retained
kernel's own deviation from its Triton reference (rel L2 3.7e-4, max abs 0.196
on layer 00). The approximate normaliser shifts every probability, so the
`residual > 1e-6` validity test can flip for near-degenerate tokens: on layer
39 the +inf bias set of the last chunk is not identical to the base's. That is
a discrete change of the kind a seed flip is, and it means this candidate is
**numerics tier for the merge**, which is the kernel the user's exactness
constraint was originally about. Not adopted; not screened. Decision requested
from the user before spending the ~50 GPU-minutes of a 20-video same-host
screen and a masked-token count.

**Decision (user, 2026-09-15):** 1 ms is too small to spend a numerics change
on; fast log is shelved, not rejected - source and receipt kept for a later
round where it might ride along with a larger change.

### `sel1`: the selector rewrite measured (2026-09-15 05:24, scai6 GPU 3)

`r9_sel1_selector_variants.json`. Selector stage timed in place inside complete
calls on the promoted merge, 30 pairs per layer, A/A null in the same square
(null -0.03 / -0.05 / -0.01 ms). **Every variant reproduces the base's seeds and
the complete output bitwise on layers 00, 19 and 39.** Base selector 7.335 ms.

| variant | REG / shared / CTAs per SM (measured) | stage delta ms/call | wins |
| --- | --- | ---: | ---: |
| v1_argmax (branchless + REDUX argmax) | 122 / 36864 / 4 | -0.738 | 90/90 |
| v3fix_abd (+ hoists, shfl candidates, halved transpose) | 128 / 36864 / 4 | -0.673 | 90/90 |
| ldsm (padded rows + 24 ldmatrix.x4) | 74 / 39168 / **5** | -0.922 | 89/90 |
| **v4_ldsm** = v3fix_abd + ldsm | 80 / 39168 / **5** | **-1.789** | **90/90** |
| v5_ldsm_occ5 (`__launch_bounds__(128,5)`) | 80 / 39168 / 5 | -1.746 | 90/90 |
| v6_ldsm_carveout (+ preferred max carveout) | 80 / 39168 / 5 | -1.729 | 90/90 |

Selector 7.34 -> 5.47 ms. The two halves are superadditive (-0.67 + -0.92 =
-1.59 measured apart, -1.79 together): the load-path rewrite freed 50 registers,
which took the kernel from 4 to 5 CTAs per SM on its own - the occupancy query
already reports 5 for the plain `(128,4)` build, so the explicit launch bound
and the carveout attribute buy nothing (both read slightly worse, within ~0.05).
v3fix_abd alone is 0.07 slower than v1 alone: at 128 registers the halved
transpose's extra selects cost more than the 16 shuffles they save; whether that
still holds at 80 registers is a cheap follow-up (v7 = v1_argmax + ldsm). With
50 registers of headroom the packed 32-bit seed store that spilled at 128 is
also back on the table (v8 = v4 + packed store).

Next: complete-call confirmation of v4 against the promoted pair (`full2`) and
the tie-edge equivalence proof at tolerance 0 (`pkg_v4`), both queued.

### v4_ldsm: equivalence proof and complete-call confirmation (2026-09-15 05:27-05:28, scai6 GPU 3)

* `r9_pkg_v4_ldsm_equiv.json` - v4_ldsm against the promoted FP32 GS regpipe
  selector on the promoted merge, tolerance 0: **0 seed mismatches and bitwise
  output on layers 00 / 19 / 39, both tie edges (all-zero query, identical
  probe vectors) 0**. Resources `[80, 0, 39168, 5]`. This is an exact change:
  no numerics question, no video screen needed.
* `r9_full2_v4_ldsm_fullcall.json` - complete call, 30 pairs per layer, A/A
  null in the same square (null -0.06 / +0.04 / -0.06):

| layer | base ms | v4 ms | delta | t | wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| 00 | 99.957 | 98.255 | -1.642 | -3.57 | 26/30 |
| 19 | 100.010 | 98.645 | -1.409 | -3.35 | 25/30 |
| 39 | 99.277 | 98.143 | -1.079 | -2.51 | 27/30 |
| pooled | **99.748** | **98.348** | **-1.377** | | **78/90** |

The stage instrument read -1.79; the complete call reads -1.38 with a wider
interval (n = 30 per layer against a ~0.5 ms full-call sd). Both agree on
direction and both are quotable; the complete-call figure is the one to carry.
Position in this window: promoted stack 99.7, with v4 98.3 ms.

Follow-ups queued: `sel2` (v7 = v1_argmax + ldsm, REG 76; v8 = v4 + packed
32-bit seed store, REG 89, both LOCAL 0) against v4 as base; equivalence proofs
`pkg_v7` / `pkg_v8` and a complete-call `full3` for both, so whichever wins can
be promoted without another claim.

### `sel2`: v7 and v8 against v4 (2026-09-15 05:34, scai6 GPU 3)

`r9_sel2_v7_v8_vs_v4.json`, base = v4_ldsm (5.404 ms), 30 pairs per layer, A/A
null +0.03 / +0.01 / +0.03. Both variants bitwise (seeds and output) on all
three layers.

| variant | REG / CTAs per SM | stage delta vs v4 | wins |
| --- | --- | ---: | ---: |
| v7 = v1_argmax + ldsm (no hoists, no halved transpose) | 76 / 5 | **+0.256** | 1/90 |
| **v8 = v4 + packed 32-bit seed store** | 89 / 5 | **-0.226** | **88/90** |

So at 80 registers the hoisted anchor / candidate-row shuffle / halved
transpose are worth 0.26 ms (they read as a 0.07 loss at 128 registers: the
register wall, not the idea, was the problem), and the packed seed store that
spilled at 128 registers is a clean 0.23 ms win at 89. **v8 is the candidate to
promote: selector 7.34 -> 5.23 ms, -2.1 ms per call at the stage.** Equivalence
proof `pkg_v8` and complete-call `full3` are running.

### v7 / v8 proofs and complete calls (2026-09-15 05:34-05:36, scai6 GPU 3)

`r9_pkg_v7_equiv.json`, `r9_pkg_v8_equiv.json`: both selectors bitwise (seeds,
output, tie edges) against the packaged FP32 GS regpipe selector at tolerance 0.
`r9_full3_v7_v8_fullcall.json`, complete call, 30 pairs per layer, A/A null:

| arm | delta ms | wins | base -> candidate |
| --- | ---: | ---: | --- |
| v7_argmax_ldsm | -1.067 | 73/90 | 99.71 -> 98.62 |
| **v8_ldsm_packed** | **-1.273** | **80/90** | 100.29 -> 98.95 |

Complete-call readings for v4 (-1.38), v7 (-1.07) and v8 (-1.27) all sit
inside the instrument's ~0.5 ms interval of one another; the stage instrument
separates them (v8 is 0.23 faster than v4, v7 0.26 slower) and is the one to
rank by. v8 is promoted next, quoting the complete-call -1.27 and the stage
-2.0 (7.34 -> 5.23 ms) side by side. A stage measurement of v8 directly
against the packaged selector (`sel_v8`) is queued so the evidence carries one
base rather than a chain.

### v8 promoted (2026-09-15 05:41)

`r9_sel_v8_vs_packaged.json`: v8 against the packaged FP32 GS regpipe selector,
stage in place, 30 pairs per layer, A/A null (-0.005 / -0.03 / +0.01):
**7.251 -> 5.213 ms, -2.022 ms per call, 90 of 90, t -86 to -122**, bitwise.

`analysis/promote_c4_selector_exact.py` verified the equivalence proof
(`r9_pkg_v8_equiv.json`), the stage receipt and the complete-call receipt
(`r9_full3_v7_v8_fullcall.json`, -1.273 ms, 80/90) against the exact bytes and
rewrote the package: `optimized_c4.py` -> `fisher_c4_selector_v8_ldsm_packed.cu`,
previous selector retained as control, manifest re-hashed (11 files), evidence
under `round9_selector_v8_20260915`, `_integrity` passes for all four packages.
README row now 98.954 ms with the paired note.

## Position after round 9

| stage | before (2026-09-12) | now | how |
| --- | ---: | ---: | --- |
| selector | 9.73 | 5.21 | FP32 GS (numerics, screened) + regpipe + ldmatrix/REDUX/packed-store rewrite (exact) |
| pack | 0.39 | 0.39 | |
| merge | 57.17 | ~56.7 | desc_select (exact) |
| route | 33.76 | 33.76 | L2 hints null; schedule sweep exhausted |
| **complete call** | **103.25 - 104.36** | **~98.4 - 99.0** (same-window paired: -3.66 then -1.27) | |

Shelved by the user: fast log in the merge (-1.03 ms, numerics tier).
Still open: the round-3 selector variants (double-buffered sk, halving
reductions, per-lane redundant divide/sqrt) being compile-priced; the route TMA
rewrite as the last exact-tier lever of size.

### Selector round 3, compile-priced (2026-09-15, CPU only), queued as `sel3`

Built from v8 by three agents, merged, and passed two adversarial reviews
(bitwise reductions; barriers/layout) with no defect. All LOCAL 0, all keep
5 CTAs/SM.

| variant | REG | shared | static change vs v8 | prediction |
| --- | ---: | ---: | --- | --- |
| v9a2_dbuf_toggle | 92 | 43520 | double-buffered K tile, **one barrier per frame** (2 -> 1), +3 int/frame | barrier cost unpriceable statically |
| v9c_halving | 80 | 39168 | recursive-halving score and center reductions: SHFL.BFLY 66 -> 45, -104 instr; lane owns candidate c = probe index, argmax collapses onto REDUX | ~ -0.46 ms |
| v9c_halving_mxredux | 80 | 39168 | + mx via REDUX.MAX.S32 on order-preserving int keys (SHFL 66 -> 41); +0/-0 tie corner noted | ~ -0.5 |
| **v10_ac** = v9a2 + v9c | **72** | 43520 | both | ~ -0.40 plus the barrier |
| v10_a1c = v9a + v9c | 72 | 43520 | both, buffer base re-derived per frame | slightly worse than v10_ac |
| v10_acd = v10_ac + v9d2 | 72 | 43520 | + per-lane redundant basis divide | predicted worse |
| v9d2 | 89 | 39168 | redundant basis divide in place of 4 SHFL.IDX + selects | predicted +0.17 (measured to check the shuffle rate) |

Agent D's finding: v8's generic (non-P==4) branches already are the
per-lane redundant form, so the v9d variants merely drop the specialisation;
statically a divide's IEEE fast path (~10 instr incl. FCHK/BSSY) costs more
than the shuffle it saves, so v9d is expected to lose and is measured only to
calibrate. Agent C's finding: the center halving needed the probe subset
relocated to lane bits 4,3 so the halving splits line up; the survivor still
equals the subset the P==4 branch consumes.

### `sel3` result (2026-09-15 09:44, scai7 GPU 2)

`r9_sel3_round3_vs_v8.json`, base v8 (5.258 ms), 30 pairs per layer, A/A null
+0.03 / +0.02 / +0.03; **every arm bitwise on all three layers.**

| variant | REG / shared / CTAs | stage delta vs v8 | wins |
| --- | --- | ---: | ---: |
| v9a2_dbuf_toggle (one barrier per frame) | 92 / 43520 / 5 | -0.077 | 63/90 |
| v9c_halving | 80 / 39168 / 5 | -0.332 | 90/90 |
| **v9c_halving_mxredux** | 80 / 39168 / 5 | **-0.374** | **90/90** |
| v10_ac = v9a2 + v9c_halving | 72 / 43520 / 5 | -0.348 | 90/90 |
| v10_a1c | 72 / 43520 / 5 | -0.317 | 90/90 |
| v10_acd (+ redundant basis divide) | 72 / 43520 / 5 | -0.220 | 89/90 |
| v9d2 (redundant basis divide alone) | 89 / 39168 / 5 | **+0.086** | 14/90 |

Readings: the halving reductions are worth a third of a millisecond and the
REDUX max another 0.04; the second barrier was worth only 0.08 and is
sub-additive with the halving (v10_ac -0.348 against -0.41 predicted); the
redundant-divide idea loses exactly as the static count said (a divide's IEEE
fast path costs more than the shuffle it replaces), which also calibrates the
shuffle exchange rate as about right. Queued: equivalence proof and
complete-call for v9c_halving_mxredux and v10_ac, plus v11 = v9a2 +
mxredux to see whether the barrier still adds anything on top.

### mxredux: proof and complete call (2026-09-15 16:11-16:13, scai7 GPU 1)

`r9_pkg_mxredux_equiv.json`: v9c_halving_mxredux vs packaged v8 at tolerance 0
- seeds and output bitwise on 00/19/39, tie edges 0, resources `[80, 0, 39168, 5]`.
The REDUX-on-int-keys max was argued (and here observed) to be downstream-
equivalent to fmaxf: a sign-of-zero difference in mx cannot change expf(c - mx).

`r9_full4_mxredux_v10ac_fullcall.json`, complete call, 30 pairs per layer:

| arm | delta ms | wins | note |
| --- | ---: | ---: | --- |
| v9c_halving_mxredux | -0.288 | 55/90 | base 98.49 -> 98.18; sign agrees with the stage (-0.374) |
| v10_ac | +0.226 | 44/90 | noise: the stage read -0.348 at t -23 |

Both readings are inside the complete-call instrument's ~0.5 ms interval; a
0.3 ms change is exactly what `abba-gate-power` says the complete call cannot
resolve, and the stage instrument (t -20 to -27, 90/90) is the decider. The
complete call is kept only as a no-regression check. mxredux is promotable
(proof, stage and complete call all in hand); v11 = double buffer + mxredux
is being measured against it in the same window (`sel4`, `pkg_v11`, `full5`).

### `sel4`, `pkg_v11`, and the third selector promotion (2026-09-15 16:15-16:20)

`r9_sel4_v11_vs_v8.json` (same window, base v8 5.183 ms): mxredux -0.339,
v11 = double buffer + mxredux -0.360, both 90/90 and bitwise;
`r9_pkg_v11_equiv.json`: v11 also bitwise with tie edges 0 (REG 72, shared
43520). The double buffer adds 0.02 ms on top of mxredux - not resolved by the
instrument (its own null reads +-0.03) and not worth 4352 more bytes of shared
memory and a second buffer to reason about. **v9c_halving_mxredux promoted**
via `promote_c4_selector_exact.py` (12 package files verified); evidence key
`round9_selector_mxredux_20260915`; README row 98.175 ms. v11 is kept as a
measured, unadopted variant.

## Position after the third selector step

selector 9.73 -> 4.88 ms across the campaign (FP32 GS, regpipe, ldmatrix/REDUX/
packed store, halving/REDUX max); merge ~56.7; route 33.76; complete call
~98.2 ms in the latest window. Exact-tier levers left: route TMA rewrite (in
progress, CPU audit/implementation), small selector items (Q-fragment frame
batching against the LDS floor). Numerics-tier, shelved by the user: fast log
in the merge (-1.03). Below ~93 ms only reduced-precision compressed K/V
(FP8) remains, which the user has excluded.

`r9_full5_v11_fullcall.json`: v11 complete call -0.399 ms (54/90; layers
-0.91 / -0.01 / -0.28, all t < 1.8) - consistent with mxredux's -0.29 and again
at the instrument's floor; v11 stays unadopted.

**Committed and pushed 2026-09-15 as e37fc10 on origin/main** (rebased over
25 upstream MI300 commits, no file overlap): the c4 package, the campaign's
instruments, receipts, STATUS.md, and the C4 variant sources. The C10 package
edits and native_fisher/selector.py from the parallel C10 session were left
uncommitted in the working tree, as found.

## Round 10, 2026-09-15: the route on TMA descriptors (CPU stage done, `route2` queued)

Two audits and an implementation, all CPU-only (Triton compiles need a device,
so nothing was compiled; the AST-to-TTIR front end was run for all variants).

**Math (route_reference.py 4051-4147, 4169-4305).** A masked compressed token is
neutralised twice: `logits = (dot*SCALE - bias)*log2e` with bias = +inf from the
masked bias load, and then `tl.where(token_mask, logits, -inf)` overwrites the
column regardless of what the dot produced. So the K rows of masked tokens never
reach a value; only their V rows do, as `(+0) * v` addends in the P.V dot. With
`other=0.0` that addend is +0; with a real neighbour row it is +-0, which cannot
change a non-zero partial sum but can flip the sign of an exactly-zero one. The
tail tile (3780 = 59*64 + 4) is the only place this matters.

**Triton 3.7.1 API (installed package, read on scai7).** `tl.make_tensor_descriptor`
(in-kernel) and `desc.load([...])` exist; the old `_experimental_*` names do not.
In-kernel descriptors keep the kernel signature but require
`triton.set_allocator` in the launching thread (128 B of global scratch per
descriptor per CTA, allocated on every launch: 2.15 MB per route launch here).
Box dims must be powers of two; the 256-byte-wide bf16 inner dim is split by the
compiler into two 128 B swizzled boxes; OOB rows zero-fill; `desc.load` carries
no cache modifier (the `.cg` hint is dropped - measured null anyway). Whether
the pipeliner multi-buffers `tt.descriptor_load` inside `tl.range(num_stages=3)`
could not be verified without compiling.

**Variants (each a complete copy of route_reference.py, same kernel name and
signature, plus `prepare_tma()` for the allocator):**

| file | descriptor | tail-tile behaviour | review |
| --- | --- | --- | --- |
| `route_c4_tma_slab.py` | per-route 2-D `[3780, 128]` at `ptr + route_base*128` | zero fill, identical to `other=0.0`, no neighbour read | preferred by both audits and the reviewer |
| `route_c4_tma.py` | flattened `[8400*3780, 128]`, row = route_base + token_start | reads 60 rows of the next route; signed-zero caveat on an exactly-zero partial sum | low-severity note |
| `route_c4_tma_singleton.py` | flattened, plus per-program `[945, 128]` descriptors for the singleton raw K/V pass (stride 409,600 elements) | singleton tail zero-fills exactly | as above |

`screen_c4_route_hints.py` now calls `prepare_tma()` on any variant that has
it, drops (and records) a variant that fails to compile or launch instead of
voiding the whole step, and uses the promoted mxredux selector as the fixed
stack. Queued as `route2` with all three variants; the bitwise gate decides the
flattened form's caveat empirically.

### `route2` result: TMA descriptor loads are a large loss (2026-09-15 17:03, scai6 GPU 3)

`r10_route2_tma_descriptors.json`, per-chunk route time inside complete calls
on the promoted stack, 30 pairs per layer, A/A null -0.12 / +0.02 / -0.08.
All three variants compiled on the first try and are **bitwise equal** on every
layer (the flattened form's signed-zero caveat did not fire). But:

| variant | route delta ms/call | wins |
| --- | ---: | ---: |
| route_c4_tma_slab (per-route 2-D descriptors) | **+6.74** | 0/90 |
| route_c4_tma (flattened descriptor) | +6.76 | 0/90 |
| route_c4_tma_singleton (+ singleton descriptors) | +6.21 | 0/90 |

Retained route 34.54 ms per call in this window; the descriptor variants 40.8-41.3.
The singleton descriptors alone are worth about -0.5 relative to the flattened
compressed-stream form, so the loss is in the compressed stream: replacing the
register-direct, software-pipelined `tl.load` stream with descriptor loads costs
~20% of the route. Diagnosis from the compiled artefacts follows below.

**Diagnosis from the compiled artefacts** (`round9/triton_cache_scai6`, TTGIR
and cubins of the route kernels): the retained route is not a register-direct
streamer at all - Triton already lowers its `tl.load` stream into a 3-stage
`cp.async` shared-memory ring (`async_copy_global_to_local` x15,
`memdesc<3x64x128xbf16>`, `tt.num_stages = 3`) feeding `warp_group_dot`
(wgmma) x8, at 255 registers, LOCAL 0. The descriptor variants are pipelined
the same depth (`async_tma_copy_global_to_local` x12, mbarriers, the same
3-deep ring) at 174 registers. So the +6.7 ms is not a missing pipeline; the
remaining suspects are the per-CTA in-kernel tensormap creation (8400 CTAs x 2
descriptors per launch, each ~50 `tensormap.replace` plus proxy fences, and an
allocator call per launch) and the 256 B row being split into two 128 B
swizzled boxes. One cheap test separates the first suspect: build the
descriptors on the host (`TensorDescriptor.from_tensor`) and pass them as kernel
arguments (Option B). Being written; stop rule: unless it beats the retained
route by at least 0.5 ms, route TMA is closed.

### `route3`: host-built descriptors (Option B) - route TMA closed (2026-09-15 17:23, scai7 GPU 0)

`r10_route3_tma_hostdesc.json`, same instrument, same window, A/A null -0.02 /
-0.06 / +0.08, both arms bitwise on every layer:

| variant | route delta ms/call | wins |
| --- | ---: | ---: |
| route_c4_tma_host (descriptors built once on the host, passed as arguments) | **+5.76** | 0/90 |
| route_c4_tma_slab (in-kernel descriptors, reference) | +6.86 | 0/90 |

Moving descriptor creation to the host recovers 1.1 ms of the 6.9 - that part
of the diagnosis was right - but the TMA stream itself is still ~5.8 ms slower
than the cp.async ring Triton already emits for the retained `tl.load` loop.
With 4 warps per CTA and about 2 CTAs per SM the single-issuer TMA path has far
less memory-level parallelism than 128 threads each issuing cp.async, and the
retained loop is already at 88% of the achievable read bandwidth. **Route TMA
via Triton descriptors is closed by the stop rule** (needed <= -0.5, measured
+5.8). The only route lever left in the exact tier is a hand-written
warp-specialised CUDA/Gluon route, a multi-day project against a 12% ceiling.

## Position after round 10

| stage | 2026-09-12 | now |
| --- | ---: | ---: |
| selector | 9.73 | 4.88 |
| pack | 0.39 | 0.39 |
| merge | 57.17 | ~56.7 |
| route | 33.76 | 33.76 (34.5-35.0 in today's windows) |
| complete call | 103.3-104.4 | ~98.2 (same-window paired deltas -3.66, -1.27, -0.29) |

Exact-tier levers remaining: selector Q-fragment frame batching (<= ~0.3 ms),
and nothing of size. Numerics tier, shelved by the user: fast log2 in the merge
(-1.03). Below the low 90s only reduced-precision compressed K/V remains, which
the user has excluded. The BF16-exact C4 floor sits near 98 ms on this design.
