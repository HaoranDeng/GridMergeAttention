# Entropy-mass seed selection for Grid Compressed Attention

> **RETIRED C4+SQ1 RECORD:** this document describes historical 4+1 runs. It
> is not an active default or runnable method. C4 now means exactly four merged
> K/V representatives; a fifth singleton is forbidden.

**Historical default update, 2026-09-04:** the user selected original Mass-GS C32 / RF0
as the then-current GCA default. At that time `generate.py` used
`configs/token_merging_wan_720p_entropy_mass_span_rf0_fewstep.yaml` by default;
the measured normalizer precision and four-warp selector are unchanged.
The matched 20-video RF0 run completed with PSNR 11.540888, SSIM 0.354963,
and LPIPS 0.505832, versus Hellinger's 11.520876, 0.358055, and 0.509484.
All-video mean core attention latency is 142.071851 ms versus 136.912539 ms;
the paired first-video-excluded diagnostic is 137.157585 versus 136.703199 ms.
The result is mixed, not a demonstrated quality-and-speed improvement.
The following earlier RF10 measurements remain historical and are not evidence
for RF0. The paper update is authorized by the user's subsequent request.

**Protocol update, 2026-09-04:** the user has fixed all subsequent experiments
to RF0. The measurements below remain historical RF10 results, not RF0
evidence. See [the active RF0 protocol](rf0_seed_selection_protocol.md) for
the explicitly named configurations and matched comparison rules.

This is an implementation and research note for two training-free GCA seed
selectors. The main candidate replaces Hellinger-Span's heuristic response
weight with an estimate of the query's attention mass on the current block.
The alternative selects a query whose current compressed representation loses
the most estimated partition mass. Both retain C4 hard ownership, conditional
soft merging, entropy correction, and global attention over compressed tokens.

The required end result is **PSNR above 12 on the canonical four-step,
20-video comparison**, with complete attention no slower than Hellinger C4
RF10. The completed mass-span run reaches **11.610300592 dB**, so the PSNR
goal is not met. Its original implementation is 0.119% slower than RF10 in
the matched 40-head attention benchmark; the slightly faster optional
normalizer has not been validated on 20 generated videos. Completed RF10
reevaluation against the same dense reference gives a modest matched gain
of **0.080676266 dB** for mass-span, with lower LPIPS and slightly lower SSIM.
The original research-note task did not modify the paper repository.

## Scope of the measured operator

Both experimental configurations and the matched Hellinger RF10 baseline
retain the existing `_sq1` construction: four seed-conditioned bulk cells
plus one singleton at the query column in every key block. The selected key
is removed from its bulk cell; the remaining bulk distribution, merged K/V,
and entropy are renormalized. The singleton is the distribution assigning
weight one to that key and has entropy zero. Thus C4 means **four selected
seed queries**, while this implementation has up to **five disjoint cells
per block**, not four total representatives.

The route processes bulk and singleton entries using the same online maximum,
partition sum, and value accumulator, followed by one final normalization.
It does not compute a separately normalized dense-attention result or replace
selected query outputs with a rescue result. Reading the singleton directly
from original K/V is a storage choice equivalent to materializing one more
compressed representative. Every original key belongs to exactly one final
cell. The global variational identities below therefore still apply when
the singleton is included among the cells.

The selector itself scores the coarse four-seed probe construction before
this fixed singleton refinement. Its probe gap is consequently a surrogate
for the final five-cell gap. The benchmark holds the singleton construction
fixed to isolate seed selection. At the canonical grid size, the logical
compressed sequence has up to TH(4+1)=4,725 entries per query column: 3,780
materialized bulk slots and 945 implicitly addressed singleton slots.

These details are implemented by
[singleton dispatch and storage](../attention/singleton_c4_wan21_14b_720p_triton.py)
and the common [merge and route kernels](../attention/hard_c4_wan21_14b_720p_triton.py).
A paper description limited to at most four representatives per block would
not describe this measured configuration.

## Why block mass belongs in the variational objective

Fix one query frame, query column, and head. Index the queries in that group
by i, and the disjoint width-aligned key blocks by b. For scaled query-key
logits z_i, define the dense block and global partition functions

\[
Z_{ib}=\sum_{\ell\in b}e^{z_{i\ell}},
\qquad Z_i=\sum_b Z_{ib},
\qquad \rho_{ib}=\frac{Z_{ib}}{Z_i}.
\]

Thus rho_ib is the exact dense attention mass on block b. The conditional
dense distribution within that block is p_ib(ell) = exp(z_iell) / Z_ib.

For a fixed seed set, let w_bc be the normalized conditional distribution of
nonempty hard cell c in block b. Its entropy-corrected representative logit is

\[
\widetilde L_{ibc}
=\sum_{\ell\in b}w_{bc\ell}z_{i\ell}+\mathcal H(w_{bc}).
\]

The compressed block partition function and its local variational gap are

\[
\widetilde Z_{ib}=\sum_c e^{\widetilde L_{ibc}},
\qquad
\epsilon_{ib}=\log Z_{ib}-\log\widetilde Z_{ib}\geq0.
\]

Because the cells are disjoint, log(tilde Z_ib) is the exact maximum of the
entropy-regularized objective within the block's structured simplex. The local
gap is its minimum reverse KL to p_ib. This statement fixes the conditional
cell distributions; it does not say that the seeds constructing them are
optimal.

**Proposition: global gap and partition-mass deficit.** For the same fixed
cells, the global entropy-regularized objective gap satisfies

\[
\Delta_i
=-\log\sum_b\rho_{ib}e^{-\epsilon_{ib}}
=-\log(1-D_i),
\qquad
D_i=\sum_b\rho_{ib}(1-e^{-\epsilon_{ib}}).
\]

Here D_i is the fraction of the dense partition function lost by the
compressed operator, and 0 <= D_i < 1. In addition,

\[
0\leq\Delta_i\leq\sum_b\rho_{ib}\epsilon_{ib}.
\]

For uniformly small local gaps,

\[
\Delta_i
=\sum_b\rho_{ib}\epsilon_{ib}
-\tfrac12\operatorname{Var}_{b\sim\rho_i}(\epsilon_{ib})
+O(\max_b|\epsilon_{ib}|^3).
\]

*Proof.* Substitute tilde Z_ib = Z_ib exp(-epsilon_ib) into
Delta_i = log(Z_i) - log(sum_b tilde Z_ib), then divide by Z_i. Rewriting
the weighted sum as 1-D_i gives the second equality. Jensen's inequality
applied to exp(-x) gives the upper bound. Expanding the weighted exponential
and the logarithm gives the small-gap expression. The first-order term is
therefore the block-mass-weighted local gap. QED.

This identifies a missing factor in a purely block-local seed criterion.
Queries that barely attend to a block should have less influence on that
block's compression. A large local log-partition alone does not establish
large block mass: that query can have an even larger global partition.

## Candidate 1: mass-weighted Hellinger-Span

The uncentered square-root probability map has the local relation

\[
\operatorname{KL}(p+\delta\|p)
=2\|\sqrt{p+\delta}-\sqrt p\|_2^2+o(\|\delta\|_2^2),
\]

for strictly positive p. We retain the existing centered Hellinger features
on the local probe set,

\[
\phi_i=\sqrt{\widehat p_{ib}}
-\operatorname{mean}(\sqrt{\widehat p_{ib}})\mathbf1.
\]

The local geometry and the first-order global gap motivate scaling a feature
by sqrt(rho_ib). Scaling individual vectors does not change their directions,
so pivoted Gram-Schmidt on the scaled features selects the next row using

\[
s_t=\arg\max_{i\notin S_{t-1}}
\widehat\rho_{ib}\,
\|(I-\Pi_{t-1})\phi_i\|_2^2.
\]

This replaces the old exp(2 alpha z_i) importance factor with estimated block
mass and removes the importance coefficient alpha from the proposed method.
The centering, probe restriction, finite candidate set, and span criterion
remain approximations. The rule is not an exact minimizer of the global
reverse-KL gap, and the local KL expansion is not a global equivalence between
Hellinger-span error and attention-output error.

The first seed remains the row matched to the key block. The other three rows
are selected from the existing 32-row multiscale candidate bank. Reference
frame 10 supplies the router queries. The resulting indices are reused across
query frames; the merge always loads the selected **real current-frame
queries**, rather than reusing RF10 query vectors or synthetic centroids.

## Fixed local and global probes

Local probabilities use P=16 uniform key columns: 0, 5, ..., 75 in each
80-key block. For every RF10 query, a single global normalizer is computed
using G=P^2=256 keys. Their flat spatiotemporal indices are

\[
u_g=g\left\lfloor\frac{N}{G}\right\rfloor,
\quad g=0,\ldots,G-1,
\qquad N=THW.
\]

The relation G=P^2 is a fixed implementation budget, not a claim that 256
probes are optimal. Neither probe set depends on training or calibration data.
Writing ell_local and ell_global for their log-sum-exp scores, the usual
uniform quadrature estimates give

\[
\widehat\rho_{ib}
=\frac{W/P}{N/G}
\exp(\ell_{ib}^{\mathrm{local}}-\ell_i^{\mathrm{global}}).
\]

The common factor cancels when comparing candidate rows for the same block.
The kernel therefore uses the stable, proportional weight

\[
v_i=\exp(d_i-\max_j d_j),
\qquad d_i=\ell_{ib}^{\mathrm{local}}-\ell_i^{\mathrm{global}}.
\]

The global normalizer is reused across every key block for that query. The
estimated mass need not be an accurate absolute probability: deterministic
probes can miss large logits. Its role here is relative weighting across
candidate queries. The implementation stores log2-partitions and uses exp2;
this is algebraically equivalent to the natural-log expressions above.

## Candidate 2: choose the largest partition-mass deficit

The current hard cells can be evaluated on the same local probe scores.
After normalizing each selected seed within the probe keys it owns, define

\[
r_i(S)=\sum_c
\exp\left[
\sum_\ell w_{c\ell}(S)\log\widehat p_{ib\ell}
+\mathcal H(w_c(S))
\right]
=e^{-\widehat\epsilon_{ib}(S)}.
\]

This expression uses the actual hard partition and conditional merge weights
on the probes. The next seed is the unselected query maximizing

\[
s_t=\arg\max_{i\notin S_{t-1}}
\widehat\rho_{ib}\,[1-r_i(S_{t-1})].
\]

The score is the estimated contribution of that query-block pair to lost
partition mass. Unlike unbounded raw KL, the conditional deficit 1-r_i lies
between zero and one. The implementation clamps tiny negative deficits
caused by floating-point rounding to zero.

This is a greedy **underrepresented-query** rule. It does not enumerate the
new partition induced by each possible insertion or maximize the actual
marginal decrease of the full objective. Adding a seed can change ownership
and existing conditional weights, so the resulting feasible sets are not
necessarily nested. There is no monotonic-gap or optimal-seed guarantee.
The matched-row anchor, probe budgets, RF10 reuse, and final four-bulk-plus-one-
singleton operator are the same as in the mass-weighted span candidate.

## Work and implementation

For one head and one RF10 router frame, let R=32 denote the candidate budget
and L=TH the number of key blocks. Local candidate-probe QK costs
O(L W R P d), and the new shared global normalizer costs O(H W G d). Their
QK work ratio is G/(T R P), or 1/42 at T=21, R=32, P=16, G=256. This is
additional work relative to the local selector, not relative to the complete
attention operator. The normalizer workspace is O(HW) per head.

Mass-weighted GS retains O(C R P) scalar selection work per block and query
column. The deficit rule evaluates the currently active cells at each
insertion, giving O(C^2 R P) scalar work at fixed C=4. Neither candidate adds
seed-key products to the C4 merge or changes the compressed sequence length.
These counts do not prove equal latency; measured complete-attention timing
on the same GPU and QKV stream remains the acceptance test.

Relevant implementation and reproduction entry points:

- [Mass-weighted GS and shared global normalizer](../attention/entropy_mass_seed_triton.py).
- [Partition-mass-deficit selector](../attention/partition_mass_seed_triton.py).
- [Reference probe objectives](../analysis/partition_variational_selector.py).
- [Captured-QKV production-path screen](../analysis/screen_partition_population_seeds.py).
- [Parity and interleaved latency benchmark](../analysis/benchmark_entropy_mass_seeds.py).
- [Matched 40-head normalizer and complete-operator benchmark](../analysis/benchmark_mass_normalizer_precision.py).
- [Clean-GPU screening launcher](../analysis/run_variational_seed_screen.sh).
- [Mass-span four-step configuration](../configs/token_merging_wan_720p_entropy_mass_span_fewstep.yaml).
- [Partition-deficit four-step configuration](../configs/token_merging_wan_720p_entropy_partition_mass_fewstep.yaml).

## Evidence and remaining acceptance tests

### Fixed-QKV screening is not a video-quality guarantee

Current screening on four captured QKV tensors uses the complete production
C4 merge and route over the captured query frames. Relative to the existing
Hellinger C4 RF10 selector, mass-weighted GS reduces attention-output MSE by
approximately 15.4-17.6% across these captures. The partition-mass-deficit
candidate reduces it by approximately 16.9-19.9%. These are per-capture
screening ranges, not a pooled video-quality estimate or an achieved PSNR.

A fixed-QKV screen changes one attention operator while holding its inputs
fixed. A generated video repeatedly feeds the changed output through later
layers and denoising steps, producing different future QKV tensors. Attention
MSE is also measured in latent feature space, whereas video PSNR, SSIM, and
LPIPS compare decoded images. Lower capture MSE therefore does not establish
better final PSNR, and a few captured calls do not cover all model behavior.

### Completed mass-span 20-video evaluation

Run `wan14b_720p_4step_entropy_mass_span_extended_20260904b` uses canonical
prompt indices 0-19, seed 0, four denoising steps, and pure GCA in every
self-attention call. The prompt contract and paired prompt identities are
verified. Its 20 videos contain 81 frames each, for 1,620 evaluated frames.
The generation implementation uses the original global normalizer (BF16
inputs cast to FP32, TF32x3 dots), GS with four warps, and normalizer
block-Q 32 with eight warps.

Both methods below are evaluated on the same 20 prompt pairs and 1,620
frames against the same dense reference. Delta means mass-span minus RF10;
higher PSNR/SSIM and lower LPIPS are better.

| Metric | Matched Uniform16 RF10 | Mass-span | Delta |
| --- | ---: | ---: | ---: |
| Mean framewise PSNR (dB) | 11.529624325204969 | 11.610300591664856 | +0.080676266459883 |
| Mean framewise RGB SSIM | 0.3579599716779665 | 0.3576849621446217 | -0.000275009533345 |
| Mean framewise AlexNet LPIPS | 0.5059471754012285 | 0.5001551824219432 | -0.005791992979285 |

PSNR is averaged over individual frames; it is not PSNR computed from pooled
MSE. SSIM uses RGB frames with data range 255, and LPIPS uses full-resolution
frames normalized to [-1, 1]. The dense reference was locked before quality
evaluation to `wan14b_720p_4step_dense_flash_matched_extended_ema_20260826/flash/videos`.
The existing Uniform16 RF10 videos have now been reevaluated against exactly
this reference. These matched results show a small PSNR and LPIPS improvement,
not a uniform improvement in all three metrics and not achievement of the
PSNR-above-12 target. The earlier TS3
**+0.0755 dB** comparison used unmatched dense references and is invalid as
a matched improvement claim.

An exploratory paired bootstrap with 20,000 video-level resamples gives 95%
intervals for the matched deltas: PSNR [0.0146863494, 0.15204505] dB,
SSIM [-0.0035042964, 0.0029039941], and LPIPS [-0.0097037681, -0.0021105538].
These describe variation within the 20 prompt pairs at seed 0; they are not
adjusted for candidate/model selection and do not establish generalization
across other prompts, random seeds, or denoising schedules.

The checkpoint provenance audit establishes that the historical RF10 and
current candidate checkpoint files are byte-identical: both complete-file
SHA256 hashes are
`13fec11400e094785dd9dae25a928a3acfb3cfc61168ab1f388344f06bef4db7`.
Both archives are raw model state dictionaries, not dictionaries containing
`generator` or `generator_ema`. The loader therefore loads the same raw
weights regardless of the requested `use_ema` flag. An EMA label in a run
name or launcher request does not establish that EMA weights were loaded.
This checkpoint identity does not make different dense reference videos
interchangeable; the comparison above uses the completed same-reference RF10
reevaluation rather than historical RF10 scores from a different reference.

The canonical local artifacts are `quality.json`, `evaluation_protocol.json`,
`checkpoint_provenance_audit.json`, and `runtime.json` under
`C:\GMA\experiments\wan14b_720p_4step_entropy_mass_span_extended_20260904b`.
These historical RF10 video metrics belong to original Mass-GS, not the
partition-deficit candidate. Separate later RF0 video results are recorded in
[RF0 seed results](rf0_seed_screen_results.md).

### Generation runtime and matched attention timing

The completed generation run records 160 attention calls per video. Its
runtime summary is:

| Aggregation | Core attention (ms/call) | Full self-attention module (ms/call) | Inference (s/video) |
| --- | ---: | ---: | ---: |
| All 20 videos | 136.636023 | 163.630394 | 40.368650 |
| Steady 19, prompt indices 1-19 | 136.385336 | 159.446599 | 39.645924 |

Core attention includes seed selection, merge, and route. The full module
also includes model projections and RoPE. Inference time is the recorded
end-to-end inference interval; video-file export is timed separately and
is excluded. The steady summary excludes the first video, not unfavorable
later samples. These generation timings do not supply a same-run SDPA
denominator and must not be combined with a historical SDPA time to create
a headline speedup.

A separate interleaved H200 benchmark uses the same captured BF16 QKV tensor
of shape [1, 75600, 40, 128] for every method, with contiguous [B,N,H,D]
outputs. It measures complete attention operators including selection,
merge, and route, but excludes model projections and RoPE. SDPA is PyTorch
SDPA forced to the Flash Attention backend. The original mass-span result is
**128.84620565 ms/call**, versus RF10 **128.69337362 ms/call** and SDPA
**331.16525879 ms/call**, giving **2.57024x vs SDPA**. The RF10 difference is
+0.15283203 ms, or **+0.119%**. This very small difference does not establish
a meaningful slowdown, but the point estimate does not meet the strict
no-slower-than-RF10 criterion either. A single captured QKV benchmark is
not a matched whole-generation speed comparison or an all-call timing sweep.

The optional native-precision normalizer gives **128.527484 ms/call** in
that benchmark. This is an engineering variant, not the implementation
used to obtain the 20-video metrics above. Its seed choices are almost,
but not exactly, identical: rare near-tie changes remain. It has not been
validated through 20-video generation, so its timing cannot be paired with
the original implementation's quality as an achieved joint result. The raw
matched benchmark is `screen.json` under
`C:\GMA\experiments\entropy_mass_native40_benchmark_20260904b`.

**Acceptance status:** the canonical mass-span run is complete, but PSNR
is below 12; the requested objective remains unfinished. The completed
same-reference comparison shows a modest PSNR/LPIPS gain and slightly lower
SSIM. The optional faster normalizer has no 20-video quality validation.
The variational motivation and fixed-QKV MSE improvements do not override
these final-video and timing limitations.

### Follow-up screens on 2026-09-04

None of the following refinements qualifies for another 20-video run on
its current evidence. Owner-preserving medoid refinement changed fixed-QKV
MSE by -0.5577% / +0.2553% on the two tested captures relative to mass-GS.
The singleton-complement mask worsened both, by +1.2118% / +1.1518%.
Partition-deficit selection improved those capture errors, but its best
measured complete operator remained 5.3235% slower than matched RF10 even
after owner-log-probability reuse and launch optimization. SDPA was not
measured in that partition launch experiment. See the dated entries in
[negative results](negative_results.md) for exact settings and artifacts.

The follow-up [joint-frame mass-weighted selection](joint_frame_mass_seed_selection.md)
has now completed native benchmarking and 20-video validation. Its C16
version improved fixed-QKV output MSE and passed the matched speed screen,
but the generated videos reached only 11.467133951 dB PSNR, 0.347324362
SSIM, and 0.508273922 LPIPS. All three are worse than both same-reference
RF10 and single-frame mass-GS. Do not promote or repeat that unchanged
two-frame selector on the strength of its captured-QKV improvement.

### Original Mass-GS with fixed RF0: completed follow-up

The user-requested RF0 test now has all 20 canonical videos and validated
quality, using original C32/g45 Mass-GS, Uniform16/G256 and the original
normalizer. It reaches **11.540888084 PSNR, 0.354963445 SSIM and 0.505831757
LPIPS**. Against matched RF0 Hellinger, deltas are **+0.020012200 dB,
-0.003091391 and -0.003652336**, respectively. This is a small PSNR/LPIPS
gain with lower SSIM, not a uniform improvement and not PSNR >12.

Complete-operator time is **142.071851 ms/call all20**, with full module
165.419669 ms/call and inference 47.151744 s/video. The first video has high
recorded overhead; steady19 diagnostic values are 137.157585 ms,
160.354711 ms and 39.815906 s. Even the steady19 operator point estimate is
0.332% slower than RF0 Hellinger, so the joint quality/speed objective is not
established. There is no matched full-video SDPA denominator. See
[RF0 seed results](rf0_seed_screen_results.md#original-mass-gs-c32-at-rf0-completed-20-video-test)
for integrity checks, per-video wins and the two explicit timing scopes.

### Conditional-Jeffreys exploration, without default changes

The [conditional-Jeffreys note](conditional_jeffreys_seed_selection.md) derives
an exact fixed-partition symmetrized-KL identity and tests a partition-aware
pivot in a separate CPU RF0/L33 sample. Its small local proxy gain is not a
complete-attention or video result. Original Mass-GS remains the default;
the known all-frame failure is not explained or repaired by that screen.
