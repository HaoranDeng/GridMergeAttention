# RF0 layer-33 replication and mass-sketch diagnosis

Completed on 2026-09-04. These are external diagnostics, not new GCA methods,
video quality results, or speed benchmarks. The canonical 20-video partition
candidate failed the quality gate; see [the final RF0 comparison](rf0_seed_screen_results.md).
RF0, Uniform16, training-free operation and the pure all-layer rule remain fixed.
No paper files were modified.

## Audited inputs and replication

Artifacts under the canonical `C:\GMA\experiments` directory:

- `rf0_layer33_dense_capture_20260904n`: all 480 dense calls over three
  canonical prompts and four actual steps; captures at calls 33/73/113/153,
  heads 0/10/20/30. Twelve files preserve original BF16 Q/K/V and the same
  call's original full-40-head dense output sliced to those heads. Only dense
  output advances this external reference trajectory. All three exit codes
  are zero, purity is empty, and the actual checkpoint branch is raw.
- `rf0_layer33_output_diagnostic_20260904o/analysis.json`: twelve validated
  payloads, four unchanged existing RF0 operators, actual selector arguments
  and seed-table fingerprints, and output SSE with FP64 reductions.
- `rf0_layer33_mass_probe_alias_20260904p/diagnostic.json`: the fixed-sample
  mass-partition diagnostic below. It does not run a candidate operator.

All stages used only scai7 H200 GPU5, sequentially, with clean-GPU checks and
empty purity logs. All twelve local capture hashes match the remote evaluated
bytes. Bundle validation checks ordered calls, raw branch, actual warped
timesteps, native RMSNorm, forced Flash, canonical prompt manifests, source and
config hashes, tensor shape/dtype, and both inner/outer completion evidence.
The recorded checkpoint path/size/mtime match the earlier stream; no new full
checkpoint checksum is claimed.

The twelve new calls reproduce the corresponding calls in
`native_seed_trajectory_rf0_20260904g/screen.json`. Across 1,344 total/axis
metrics, the largest relative discrepancy is 1.58e-7, consistent with the old
FP32 versus new FP64 reductions. An explicit rtol 2e-6 check passes (atol 1e-6
for SSE/energy and 1e-12 for MSE). This is aggregate numerical replication,
not a claim of bitwise-equal QKV or complete output arrays.

| Method | L33 pooled MSE, 12 calls | Head30 MSE | Other three heads, pooled MSE |
| --- | ---: | ---: | ---: |
| Hellinger RF0 | 0.004618089040 | 0.018437667314 | 0.000011562949 |
| Mass-GS C32 | 0.021383104260 | 0.085498507764 | 0.000011303092 |
| Mass-GS C16 | 0.021395075470 | 0.085545752150 | 0.000011516576 |
| Partition C16 | 0.018183582170 | 0.072700242015 | 0.000011362222 |

All three candidates lose on every one of the twelve calls. Head30 accounts
for 99.81%–99.96% of the respective methods' SSE on these L33 captures. The
other three heads improve slightly in aggregate. Mass C32 and C16 are nearly
equally bad, so shrinking the candidate bank is not a sufficient explanation.
Head30 shares only about 1.24–1.26 of its four seeds with Hellinger, including
the obligatory common anchor. This establishes changed selection, not its
cause or whether a particular seed objective is appropriate.

## What the global probes actually measure

For N = 21 x 45 x 80 = 75,600 keys and G = 256 global probes, the implemented
indices are `j * floor(N/G) = 295*j`. Every probe has column divisible by five.
Let P0 be all 15,120 keys on those sixteen columns, and let B be an 80-key row
block. For a fixed RF0 query, define unnormalized exponential sums:

- Z: all N keys; Z0: all P0 keys; ZG: the actual 256 global probes;
- ZB: all 80 keys in B; ZB0: its sixteen Uniform16 local probes.

The true block mass and quadrature-corrected sketch are

\[
\rho=Z_B/Z,\qquad
\widehat\rho=(256/15120)\,Z_{B0}/Z_G.
\]

The common factor cancels when the existing selector normalizes importance
across candidates, but must be restored to test whether estimated mass exceeds
one. There is an exact decomposition, using natural logs:

\[
\log(\widehat\rho/\rho)
=\underbrace{\log\frac{Z_{B0}/Z_B}{Z_0/Z}}_{\text{local/global phase mismatch}}
-\underbrace{\log\frac{(15120/256)Z_G}{Z_0}}_{\text{subsampling error within P0}}.
\]

This separates a shared-phase effect, which can cancel in a ratio, from missing
mass inside that phase. A large common normalizer error alone need not change
selection; candidate-relative errors and normalized importance are therefore
also checked.

## Measured sketch error

Only the three step-0 L33 captures are used. The sample is fixed in advance:
query columns 0/1/20/21/40/41/60/61/75/76, key rows 0/11/22/33/44, key frames
0/10/20, and the unchanged C16/g16 candidate bank. This gives 370 distinct RF0
queries per head, 1,800 candidate banks, and 28,800 candidate/block observations.
It is a deterministic diagnostic sample, not independent random replicates.

All partitions use the same original BF16 QK values, FP32 non-TF32 dot products
and FP64 streaming logsumexp. The full key matrix is never saved. Scalar
partition caches permit independent re-computation. These are sums over the
specified subsets of those FP32 logits, not exact real-arithmetic values or
original Triton register dumps.

Let pi and pi_hat normalize rho and rho_hat across the same candidate bank;
TV = one half of their L1 distance.

| Diagnostic | Head30 | Other sampled heads |
| --- | ---: | ---: |
| Mean normalized-importance TV | 0.276413 | 0.010163 |
| Phase-mismatch-only TV | 0.009108 | 0.009283 |
| Within-phase-subsampling-only TV | 0.268719 | 0.003947 |
| Candidate-centered log-mass-error RMS | 8.051581 | 0.025098 |
| Estimated block masses exceeding one | 481/7200 (6.681%) | 0/21600 |

Head30's median/p90/max TV are 0.145970/0.768894/0.995469. Its mean log-mass
error is +14.2281, and the mean within-phase normalizer error is -14.3748.
The phase-only sensitivity is small compared with the within-phase-subsampling
sensitivity on this sample. The two TV columns are not additive error shares.
Merely spreading G256 probes over all eighty columns is therefore not an
established fix for the dominant observed problem.

The estimator is demonstrably distorted precisely where the output diagnostic
shows severe regression. This motivated a targeted causal check, not a claim
that fixing it would improve output MSE or video PSNR. The normalizer-only
oracle test is now complete and negative; see the completed test below.
It kept the same partition selector/local probes/merge/route and replaced only
the sampled normalizer with full-key and full-P0 sums. These remain external
diagnostics, not eligible production candidates or evidence of unchanged speed.
No dense layer/head/frame exception is introduced.

An independent recomputation from the twelve scalar caches checked all 1,800
banks and 32,610 record/summary values, with maximum discrepancy 1.60e-14.
The caches contain 146,520 finite partition values. Source/config hashes,
capture-input receipts and all twelve locally copied PT fingerprints match.
This audit reuses the saved dot-product partitions; it does not re-run QK
or add independent data. Head30's mean TV is 0.2921/0.2738/0.2633 on the three
prompts, so the observed distortion is not confined to one prompt.

### Cached probability-clipping check: negative

A separate read-only calculation from the same caches tested only the physical
bound `rho_clip = min(1, rho_hat)`, with the quadrature factor restored before
clipping. This is not a selector/output experiment. All 481 clipped candidate
observations belong to 247 head30 banks. Pointwise absolute mass error cannot
increase under this projection because the true mass lies in [0,1]; that
inequality holds on every checked observation. Nevertheless, normalized
importance TV worsens: head30 mean 0.276413 -> 0.372070 and median 0.145970 ->
0.476929. There are 87 improving, 160 worsening and 203 unchanged head30 banks;
all other heads are unchanged. Each of the three prompts worsens in mean TV.
Clipping can flatten useful relative differences when the absolute estimate
shares an overestimate. Do not infer better selection from the physical bound,
promote this clipping rule, or repeat it as a new candidate based on the scalar
inequality. No new GPU computation or fitted parameter was used.

## Completed normalizer-only oracle: negative

Source: `C:/GMA/experiments/rf0_layer33_exact_normalizer_oracle_20260904q/diagnostic.json`.
All three canonical step-0 L33 captures completed, with `valid=true`,
`complete=true`, exit code zero, and an empty purity log under exclusive
single-H200 execution. Each retains all 3,600 RF0 queries per saved head,
all 75,600 keys, and heads 0/10/20/30. The source/config/capture identity gates
passed. This is not a video run or a new eligible GCA candidate.

The frozen selector remains RF0/C16/g16/Uniform16/J4/G256, four warps,
`reuse_owner_logp=True`, fixed anchor plus three partition-deficit steps,
unchanged SQ1, merge and full compressed route. The original baseline runs
without any patch. A second, observation-only baseline call is required to
match its complete output bitwise; all three prompts passed. Actual selector
arguments are identical across the three normalizer settings. Only the imported
normalizer function is replaced for the two oracles, and the original G256
routine is not redundantly executed on those paths. Seed tables are the real
selector outputs, not substituted tables.

The two oracle denominators use FP32 non-TF32 QK dots and FP64 streaming LSE:
`log2(Z) - log2(75600/256)` and `log2(Z0) - log2(15120/256)`, converted to the
native FP32 normalizer layout. These common quadrature factors cancel across
candidates. The local numerator remains ZB0 from P16: the full-key oracle has
weights proportional to ZB0/Z, not the exact full-block mass ZB/Z; the phase0
oracle has weights proportional to ZB0/Z0. Thus this test isolates denominator
replacement, not complete correction of every mass or partition approximation.

The reference is each capture's saved same-call full-40-head dense output slice.
Pooled MSE is sum SSE / sum numel over 3 calls and 116,121,600 elements:

| Normalizer / reference method | Pooled MSE | Change vs partition G256 |
| --- | ---: | ---: |
| Original partition G256 | 0.02365373244428573 | 0% |
| All-75,600-key denominator oracle | 0.023629423168304367 | -0.102771% |
| All-15,120-phase0-key denominator oracle | 0.02377341492384659 | +0.505977% |
| Hellinger RF0, matched step0 scope from output run `o` | 0.005489318363816438 | Separate baseline |

The full-key denominator barely changes the error and remains 4.3046 times
the matched Hellinger MSE; phase0 is slightly worse than original partition.
The earlier estimator distortion is real, but correcting its denominator alone
is insufficient to repair the output failure. This result does not justify
implementing a better global-probe policy, promoting either oracle, or another
20-video candidate run on that rationale. It also does not prove that local
P16 sampling or the remaining selector objective is the cause. Attention latency
and SDPA speedup are explicitly null; none of these MSEs is video PSNR.

### Cached global peaks: displaced local alignment, not a fixed sink

The same full-key stream saved each query's global argmax key and true peak
softmax mass. For head30, the JSON reports the following values. The offset
counts were independently derived by read-only CPU inspection of its portable
`promptNNN_exact_normalizers.pt` caches, rather than a separate saved offset
summary artifact. Define query flat-token index i = row * 80 + column in RF0
and argmax all-key flat index j; offset -1 means j - i = -1, without wrapping
at row boundaries.

| Prompt | Offset -1 / 3600 | Same token / 3600 | Mean peak mass | Argmax hit by G256 / 3600 | Argmax in P0 / 3600 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 3542 | 43 | 0.9730355655445957 | 13 | 763 |
| 1 | 3544 | 47 | 0.964484757504574 | 12 | 763 |
| 2 | 3550 | 45 | 0.978074388683127 | 14 | 766 |

These head30 distributions are sharply peaked, predominantly on the preceding
flat token, not diagonal identity attention. There are 3,554/3,551/3,551 distinct
argmax keys respectively, and the most frequent individual key serves only two
queries in each prompt: this is not one global fixed attention sink. These are
RF0/head30/step0 observations only, not a routing exception, new policy, or proof
that a particular replacement will improve the full model.

## Artifact fingerprints

- Capture manifest: `5c001dcf0a58a04379575e8390a7546e8f8e1f57a97d605d3c2c6b4c56d2f630`.
- L33 output analysis: `e7d05b1419e75fb05b8440b653ea6dec0565c9430ab662007630110f3d86170b`.
- Mass-sketch analysis: `9e0e6c5329d85807388d5046139033950c489ef59133f4bf2a3b86d1e004c44d`.
- Normalizer-only oracle analysis: `b2e6f3b8430087d40fdfc610681048878f2cdfa126bb62cd6d1760ab7c9408f4`.
- Oracle script used by that run: `48c902b915494cc6981aa11d286251c215da4209cddae45778c8e534361a09f7`.
- Oracle portable cache, prompt 000: `8e20a87e75227dcd53defd0b71113fdee971c6f39ee9a50bcd2ec5f8752e9e50`.
- Oracle portable cache, prompt 001: `542a136890be58fcd0eae4a0b6d1dc977b8c41a29d8fc30d22b1b02e4726ae13`.
- Oracle portable cache, prompt 002: `9cbf48f6cb4c994144c213608a60c2bc7c1b1e6501f7d6de28a14971d7f655fa`.

The three local oracle-cache file hashes above match their receipts in the
oracle JSON. Each file is 4,355,135 bytes and includes raw FP64 partitions,
FP32 injected normalizers, actual seed tables, peak arrays, and source/input
provenance. The offset counts above are derived from these exact cached bytes.

The capture, output-analysis and mass-sketch scripts have 29 passing CPU tests
combined, both locally and on the remote Python environment with CUDA hidden.
Their completed GPU runs are separate evidence from those CPU tests. Mean
attention latency and SDPA speedup are intentionally unmeasured for these
sampled-head diagnostics.
