# Anchor32 with one guarded last-slot replacement

Exploratory candidate, 2026-09-06 UTC. It keeps the existing Anchor32
selection as an incumbent, and changes only its fourth real row. No paper
or production default is changed. RF0, C32/g45, Uniform16, C4+SQ1 and
current-frame BF16 merging with full compressed core attention remain fixed.

## Evaluate the partition after the proposed replacement

Let the original ordered selection be S=(r,s2,s3,s4), with prefix T=(r,s2,s3).
Evaluate the existing Uniform16 keys and explicitly observe the matched
query-column singleton. If that key is already sampled, retain16 distinct
bins; otherwise append it as a17th bin. Its measure is1. The other m=15 or16
bins each have positive measure79/m, giving total measure80. For each of the
32 real RF0 candidates, define p_i=softmax(z_i+log(mu)). No G256 importance
estimate is used.

For any ordered set S, compute owners from the complete normalized seed
profiles before extracting SQ. Lower seed slots win ownership ties.
Condition each seed profile on its resulting nonempty bulk cell to obtain
w_c. On this weighted support the actual restricted-family reverse-KL gap is

    D_i(S) = -log[p_i(SQ) + sum_c exp(w_c^T log(p_i) + H(w_c))].
    E(S) = mean_i D_i(S).

The entropy here is bin entropy: the positive measures are already present
in log(p_i). This is equivalent to a measure-entropy correction on the
unweighted logits. Empty cells contribute zero represented mass. This is
the disjoint compressed-family objective, not nearest whole-block tangent
coverage and not a quadratic span residual.

Propose exactly one challenger b: the unused candidate with maximum D_i(T).
Only the first three slots are excluded; b may equal s4. Compare E(T,b)
with E(T,s4), and replace s4 only if the former is strictly smaller. Ties
retain the incumbent. Challenger ties choose the lowest candidate slot.
All four output rows remain distinct, with the first three unchanged.

In ideal exact arithmetic, the resulting E cannot exceed the incumbent E
because the incumbent is one of the two choices. This is only a local
RF0/C32/quadrature statement. It does not bound the true full80 ownership,
the global objective over all key blocks, transfer to other query frames,
finite-precision merge moments, attention-output error or video PSNR.

## Relation to earlier attempts

The existing `partition_variational_selector.py` already evaluates all
candidate insertions. Its `gap_gain` screen used RF10 and the older FP8-score
merge; the bounded history audit did not locate a full20 RF0/BF16 run.
The old owner-preserving medoid instead holds ownership fixed while changing
representatives. Neither is new theory introduced here.

The present rule limits the trial search to one optional final replacement
of the best completed Anchor32 incumbent. The already completed SQ1-deficit
and Conditional-Jeffreys RF0 trials instead choose successive challengers
without retaining Anchor32 and evaluating both completed partitions. Their
full20 PSNR results,11.507225480428119 and11.506616749918376, do not establish
a result for this guarded rule.

## Prototype cost and verification boundary

One C3 and two C4 gap evaluations require leading cell-target work of
approximately11RP in addition to incumbent selection, with no R-by-R
candidate-pair matrix. The current prototype deliberately calls the existing
Anchor32 J4 selector first, then applies a separate J1 correction. That
correction repeats U16 QK and uses a32-column dot tile for16/17 valid bins.
Its actual overhead is therefore greater than the6.25% logical QK increment
that observing one extra key would require in a future fused implementation.
No fused implementation or speed benefit is claimed.

The independent reference has separate NumPy and Torch FP64 implementations.
All17 reference tests passed in an actual single-thread Torch CPU run with
CUDA disabled. They include a fractional Python-list measure test: weights
must be converted directly toFP64, not rounded through defaultFP32 first.
The native algebra/AST contracts also passed12 CPU tests; these do not prove
GPU compilation or numerical agreement.
The native guard must match all three per-target gap vectors, challenger and
acceptance decisions, exact final rows, original first-three rows and
audit-on/off behavior on independently generated BF16 inputs. Both singleton
cases and two key-frame/scale cases are included. These are numerical checks,
not evidence of better generated videos. A complete20 experiment must
establish quality before any engineering speed optimization is justified.

## First native check: not accepted

The immutable `e40646f` run `anchor_gap_guard_native_check_20260906bp`
finished on scai6 at `2026-09-06T01:04:25.847295+00:00` with worker exit 1
and `valid=false`. No foreign process was observed. All 21,600 final row
sets, original prefixes, membership/uniqueness checks and audit-on/off
outputs matched, and all 2,073,600 gap values passed the original fixed
`2e-10 + 2e-11 * abs(reference)` tolerance. Nevertheless, the seed-53,
two-key-frame case had exactly one internal decision-triple mismatch.
The seed-17 case had none. Final-row agreement does not waive this gate.

The failed run did not save the individual decision or bank inputs, so its
root cause is unresolved. A checker-only follow-up now saves at most eight
failed banks per case, including the actual/reference decision triples,
FP64 logits, measures, singleton index and complete gap vectors for CPU
replay. Strict decision equality and all tolerances are unchanged. The
parent now validates each guard record before printing a success message.
No native/reference algorithm change or guarded-rule video run has been
made on the strength of this failed check.

The next step is a fresh diagnostic check, followed by a cause-specific fix
and a full numerical gate before the 20-video experiment. As of the next
resource inspection, other same-account jobs already occupy two scai6
cards, and no clean scai7 card is available. No third scai6 card is launched
and other tasks are left untouched.

## Diagnostic reproduction and identity fix

When scai6 became idle on 2026-09-06, diagnostic run
`anchor_gap_guard_native_check_20260906br` reproduced the failure and ended
with `valid=false` at `2026-09-06T07:15:03.421920+00:00`. The preceding `bq`
attempt stopped during source freezing, before worker/GPU launch: checkout
preparation had not yet finished. It is a setup failure, not a numerical
or video result. The `br` run used the completed, clean `73223ef` checkout.

The saved mismatch is bank 13079, at key frame 1, row 36, column 59, head 1.
Its incumbent candidate slots are `[23,22,28,20]`. Both implementations
choose challenger 20, but native records `[20,1,20]` and the independent
reference records `[20,0,20]`, where the fields are challenger, accept and
old fourth slot. The proposed and incumbent ordered tuples are identical.
There is therefore no possible strict objective improvement, irrespective
of numerical reduction details. Audit-vector means recomputed by Torch
are not the native comparison scalars and are not used to infer those
scalars' exact difference.

The native, NumPy and Torch acceptance rules now explicitly require
`challenger != old_last` in addition to the unchanged strict mean-gap
comparison. Raw gap vectors and the comparison for distinct tuples are
unchanged. This enforces an exact identity rather than introducing a
tolerance or fitted threshold. The GPU checker independently requires zero
self-accepted banks and requires the acceptance bit to match whether the
actual rows changed, even if a reference implementation were to share the
same mistake. A fresh complete native check is still required before video
generation; the failed diagnostic is retained as failed.

The fix passed 19 actual CPU reference tests (12 NumPy and 7 Torch, zero
skips) in 0.122 seconds with CUDA disabled and one thread per numerical
library. Both backends exercise an artificial one-ULP lower trial value
for an identical tuple, while distinct tuples continue to accept a strict
one-ULP improvement and reject a one-ULP increase. The local broader suite
passed 59 checks and explicitly skipped seven unavailable local Torch
checks; it is not additional GPU evidence.

## Corrected native gate passed; video evaluation started

The fresh `60e9aa3` run `anchor_gap_guard_native_check_20260906bs`
passed at `2026-09-06T07:22:39.139987+00:00`: valid completion, worker exit 0,
no observed foreign process and unchanged source fingerprints. All 21,600
banks have exact final rows, decisions, prefix preservation and audit-on/off
agreement. Self-accepted banks and acceptance/actual-row-change mismatches
are both zero in both cases. All 2,073,600 FP64 gap values pass the original
fixed tolerance. Maximum absolute errors are 3.552713678800501e-14 for
seed 17 and 2.3092638912203256e-14 for seed 53.

The parent validation was also rerun locally on the copied results, and
all 49 frozen source files were independently hash-checked. The results
SHA256 is `793e31a27e8f47cb530067c51e1a17dedf648e8f96c50d73ae5fdbc2aa5e7368`.
This proves the declared synthetic numerical checks, not video quality or
speed. No failed run is relabeled as a pass.

The corrected frozen source is now used by
`wan14b_720p_4step_anchor_gap_guard_rf0_bf16score_extended_20260906bt`.
Its worker started at `2026-09-06T07:24:04.031209+00:00` on one clean scai6
H200. This is the canonical 20-video, four-step, all-40-layer RF0/BF16
experiment. Complete quality and runtime results are pending; there is
still no evidence that this candidate meets the joint goal.

## Complete 20-video result: no improvement

Run `wan14b_720p_4step_anchor_gap_guard_rf0_bf16score_extended_20260906bt`
completed validly at `2026-09-06T08:07:30.788615+00:00`, with worker exit 0,
504 five-second monitor samples and no observed foreign process. All 3,200
attention calls, all 40 layers and all 20 canonical videos were verified.

Full-resolution PSNR over 20 videos and 1,620 frames is
**11.61707096942693 dB**. This is 0.019337458439853 dB below Anchor32
(11.636408427866783), although above matched-reference historical Hellinger
RF10 (11.529624325204969). It does not reach 12 and is not promoted. A lower
local quadrature gap did not yield better complete-video PSNR in this run.

Instrumented complete GCA averages 527.5955111122131 ms/call, or
520.3545060609516 excluding the first video. The complete Wan wrapper
averages 564.0846064376831 ms/call. These include generation instrumentation
and concurrent CPU scoring. No matched SDPA denominator or matched Hellinger
timing was collected for this candidate, so no headline speedup is claimed.
SSIM, LPIPS and VBench remain unmeasured. The negative full20 result does not
justify further kernel optimization of this guard.

All 547 archived files, including 510 frozen sources and 20 videos, were
verified after local transfer. Archive SHA256:
`aea577c024ad1cf5f284d77542af7c9ebcfe699962e334a2c0c4704aa1d55c29`.
The independent RF0-only temporal phase-average candidate continues in its
own frozen run; neither the paper nor the production default is changed.
