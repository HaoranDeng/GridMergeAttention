# One globally frozen-route conditional-medoid pass

CPU-only reference and fixed real-QK pilot, 2026-09-06. No native selector,
video result, speed claim, production change or paper edit. This is a discrete
instantiation of the existing [fixed-beta result](seed_barycenter_limits.md),
not a new optimization principle. It follows the numerically stable SQ1
investigation; old subtractive-SQ1 output failures alone do not refute it.

## Exact supplied-quadrature objective

Inputs cover one RF0 query column: all physical rows `i=0..44`, every supplied
bank `b`, and distinct Uniform16 plus the actually observed singleton. The
singleton measure is 1; the other `m=15/16` bins each have measure `79/m`.
Incumbent and replacements are four distinct **real** rows from each bank's
C32 table. Preserve the incumbent first row and all bulk-bin owners; assign
owners with full normalized profiles before removing SQ. The old SQ owner
may change because SQ remains its own separate component.

Let `p_ib=softmax(z_ib+log(mu))`. For initial conditional weights `w_bc`, define

\[
t_{ibc}=\langle w_{bc},z_{ib}+\log\mu\rangle+H_{bin}(w_{bc}),\qquad
\beta^0_{ibc}=\frac{e^{t_{ibc}}}{\sum_{d,e}e^{t_{ide}}}.
\]

The denominator includes all banks and their four bulk cells plus SQ.
Bank-local normalizers cannot be discarded: the older helper's normalized
component logits must have `log Z_ib` restored before global normalization.
Freeze this beta for the **entire** pass. For slots 1,2,3, sequentially within
each bank, set `alpha_i=beta^0_ibc/sum_j beta^0_jbc` and

\[
\rho_{bc}=\operatorname{softmax}_{C_{bc}}
 (\sum_i\alpha_i\log p_{ib}),\qquad
s_{bc}'=\arg\min_{s\ \mathrm{feasible}}
  \mathrm{KL}(p_{sb}(\cdot\mid C_{bc})\|\rho_{bc}).
\]

The feasible set preserves bulk probe owners and four-row distinctness.
Keep the incumbent unless a candidate's score is strictly lower; other ties
follow the C32 table order. Empty cells keep their row. No fitted threshold,
G256 estimate, new synthetic query or additional frame participates.

For the disjoint family, fixing beta separates the variational objective by
cell. Each replacement exactly minimizes its discrete conditional subproblem;
the incumbent is feasible. Hence, in ideal arithmetic,

\[
E(W')=\min_\beta\Phi(W',\beta)
\le\Phi(W',\beta^0)\le\Phi(W,\beta^0)=E(W).
\]

Here `E` averages all45 targets over the **supplied global quadrature**. It
does not guarantee every target improves, full80 owners remain fixed, other
frames improve, or output/PSNR improves. The caller must independently verify
that all945 original banks and RF0 provenance were supplied.

## Control, arithmetic and evidence

`analysis/global_route_medoid_reference.py::refine_global_route_medoids`
accepts `[B,45,P]` natural logits, `[B,32]` candidate physical rows,
`[B,4]` incumbent physical rows, probe columns and singleton column.
`weighting='bank_local'` changes only the initial frozen beta normalization;
it has a bank-local surrogate guarantee, **not** a global-gap guarantee.
Unlike the previous local reference, neither mode refreshes beta mid-pass.
`global_quadrature_state` independently exposes the supplied model's route.

NumPy FP64 log-domain demand normalization never skips an active cell merely
because every absolute `exp(log_beta)` underflows. Per-step normalized KL
gain and log mean demand certify its surrogate improvement; log gains remain
available when absolute improvements underflow. Empty support, not a numerical
mass cutoff, determines inactivity. Raw gaps are not clipped. Finite-precision
rounding remains distinct from the ideal-arithmetic theorem.

The independent unit tests reconstruct effective attention and direct KL,
enumerate every feasible replacement, compare frozen-beta surrogate changes,
check global versus bank-local demand, query-wide/bank shifts, one-bank
identity, tiny demand, all45 targets, real-row mapping, SQ support and guards.
They are CPU algebra tests, not native or real-QKV evidence.

## Cost and proposed falsification only

The scalar work remains linear in bank count and probe size, but adds a
cross-bank reduction and three constrained coordinate updates after Anchor32.
Avoiding large saved logits likely requires two RF0 probe passes with45
targets. Two45x17 passes are about3x the old32x16 selector QK, adding about1.35
percentage points of dense QK work on21x45x80, before extra FP64 reductions.
This does not establish the required no-slower-than-Hellinger runtime.

A later bounded diagnostic may reuse existing original BF16 captures, keeping
all945 banks and checking both SQ support cases. Compare the two frozen-beta
rules, then evaluate full80 ideal gap and stable complete output on RF0 and
held-out frames. Failure to transfer the probe gain is grounds to stop, not to
sweep parameters. The completed CPU pilot below tests the ideal-arithmetic
part of this diagnostic. A native replay remains a separate test.

## Completed fixed real-QK CPU pilot

`sq1_global_route_medoid_cpu_20260906cf` completed with attached exit0 on
2026-09-06, frozen at `8e8b0ebab447b559da4b155eab5c9bed3983ee1e`.
It uses only the locked prompt1/step0/L33 BF16 capture, original heads0/30
and columns0/40/41 (six prespecified cases). Each case includes all945
banks, all45 RF0 targets, C32 real candidates and U16+observed SQ.
Only RF0 determines seeds. Diagnostic frames0/10/20 independently rebuild
full80 ownership, direct bulk conditionals and the full global route in
FP64; these are not native BF16 outputs or generated-video metrics.

Global weighting changes2090/17010 adjustable slots in1906/5670 banks.
The bank-local control changes1930 slots in1754 banks. The two final seed
tables differ in907 slots across847 banks. Mean global probe gap is
0.0126126223472 initially, 0.0107425026751 with global weighting, and
0.0111919392355 with bank-local weighting.

| Measured query frame | Anchor ideal output MSE | Global-beta medoid | Local-beta medoid |
| --- | ---: | ---: | ---: |
| 0 | 0.001225676893 | 0.000867262861 | 0.000848466783 |
| 10 | 0.000521795636 | 0.000504940247 | 0.000530865152 |
| 20 | 0.000702745051 | 0.000662847340 | 0.000665934510 |

All values use the same independently computed dense FP64 output. Equal
45x128 output counts make these six-case means equivalent to pooled MSE.
Global full80 KL gaps are 0.0196537593241 ->0.0180091697399 at RF0,
0.0109948666315 ->0.0111476035559 at frame10, and
0.0124843490488 ->0.0125129608471 at frame20. Thus lower average output
error at the later frames is not a claim that their KL objective improves.

**The aggregate gain is highly concentrated.** Head30/column0 accounts for
100.02%, 99.96% and99.53% of the net MSE decrease at the respective frames.
The global method improves13/18 individual cases, but only3/6 at RF0;
the head0 RF0 mean slightly worsens. This is a reason to verify actual
stable native arithmetic on the fixed seed tables before committing to a
full GPU selector or20-video run, not evidence of broad-head superiority.
No expanded sample, parameter sweep or native implementation was run here.

Independent saved-output MSE recomputation is exact. Full80 direct-KL
identity error is at most1.851e-13 and probability-mass error4.581e-13.
CPU diagnostic time94.591s is not attention latency. PSNR, SSIM, LPIPS,
VBench and matched SDPA/Hellinger speed are unmeasured. The complete JSON,
271-array NPZ and six frozen sources are SHA-verified under the same run
name in `C:/GMA/experiments`, with `local_archive_receipt.json`.
Result SHA: `6991f7a3b57754c4844dfdf3f4dca7862af9c12e1a6385f13984d08bcf44c5d1`.
Array SHA: `13387ea1e56ecf53385a51300f337f560a66db36ba47c5a2c101a8f95727fafd`.
