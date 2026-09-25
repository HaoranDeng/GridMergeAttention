# RF10 partition C16 native benchmark: historical fixed-QKV diagnostic

Status: completed, clean benchmark; **not an RF0 result and not a video-quality result**.
The user's current setting is fixed RF0. This archive preserves the earlier RF10 measurement without transferring its conclusions to RF0.

## Evidence and scope

Source: [screen.json](C:/GMA/experiments/partition_c16_native40_benchmark_20260904f/screen.json), with `exit_code=0` and an empty `purity.log`.
The two records are `call020_holdout` and `call100_holdout`: the same holdout prompt, layer 20, denoising steps 0 and 2.
Each output-error and timing measurement covers all 21 latent query frames and all 40 heads, using BF16 execution on an NVIDIA H200. The historical NPZ files store FP16 copies of QKV from a BF16 model run; the benchmark converts those copies back to BF16, rather than reproducing the original live tensors bitwise.
This is **two captured attention calls**, not two independent prompts, all 40 layers, all 4 steps, or a canonical 20-video evaluation.
No PSNR, SSIM, LPIPS, full-module latency, or video-generation runtime was measured in this benchmark.

The complete native operator includes seed selection, the original merge and route, and contiguous output; it excludes QKV projections and RoPE.
Timing uses interleaved CUDA events, four repeats of five calls each, and two warmups. Compilation, warmup and context setup are excluded; selection is recomputed every call.
Each reported speedup divides the **same capture's** forced-Flash SDPA time by the candidate time. SDPA is an external reference, not a dense rescue inside GCA.

## Frozen variants

All compressed variants use RF10, four real query-row seeds with the matched row first, and unchanged SQ1.
SQ1 is a disjoint singleton included in the same compressed softmax, not a separate dense-attention branch.
Hellinger uses C32/geometry45 and alpha 0.2; mass C32 uses the same candidate bank, while mass/partition C16 use C16/geometry16.
Local probes are Uniform16; mass/partition global normalization uses 256 probes with the original FP32 TF32x3 implementation. Native-low-precision and TF32-single normalizer optimizations are off.

Partition uses the probe-domain score `rho_i * (1 - exp(-gap_i))`; its gap is exact for the current hard partition on the 16 probes, **not** for all keys or query frames.
J2 and J4 are launch variants of this rule. Both quality and timing use the benchmark-only four-warp, owner-log-probability-reuse override (`num_stages=1`), rather than unmodified incoming production settings (eight warps).

## Full40-head error against matched SDPA

Lower MSE is better. Percentages are `(candidate MSE / mass C32 MSE - 1)`.

| Method | Call020 MSE | Call100 MSE | 020 vs mass C32 | 100 vs mass C32 |
|---|---:|---:|---:|---:|
| Hellinger C32 RF10 | 0.005058141425251961 | 0.009085572324693203 | +15.9939% | +11.7513% |
| Mass C32 RF10 | 0.004360696766525507 | 0.008130175061523914 | 0% | 0% |
| Mass C16 RF10 | 0.004396552219986916 | 0.008351536467671394 | +0.8222% | +2.7227% |
| Partition C16 J2 RF10 | 0.004427372012287378 | 0.008305924944579601 | +1.5290% | +2.1617% |
| Partition C16 J4 RF10 | 0.004427372012287378 | 0.008305925875902176 | +1.5290% | +2.1617% |
| Dense SDPA Flash | Reference | Reference | — | — |

SDPA self-error was not separately recorded; blank workbook MSE cells do not represent missing candidate measurements or invented zero measurements.
Partition improves MSE over Hellinger RF10 by about 12.47%/8.58%, but is slightly worse than mass C32 in both captures.
Against mass C16, partition is about 0.70% worse on call020 and 0.55% better on call100: there is no consistent two-capture quality win over that cheaper bank-matched control.

## Matched complete-operator timing

Times are mean milliseconds per call; speedups are dimensionless and specific to these frozen QKV captures.

| Method | Call020 ms | Call100 ms | 020 SDPA speedup | 100 SDPA speedup |
|---|---:|---:|---:|---:|
| Hellinger C32 RF10 | 128.154947 | 128.137109 | 2.582696x | 2.589674x |
| Mass C32 RF10 | 127.952490 | 128.213400 | 2.586783x | 2.588133x |
| Mass C16 RF10 | 122.408411 | 122.261697 | 2.703942x | 2.714123x |
| Partition C16 J2 RF10 | 137.547018 | 137.283331 | 2.406343x | 2.417142x |
| Partition C16 J4 RF10 | 125.076849 | 125.436716 | 2.646255x | 2.645424x |
| Dense SDPA Flash | 330.985272 | 331.833319 | 1x | 1x |

J4 is 2.40%/2.11% faster than RF10 in these captures; J2 is 7.33%/7.14% slower.
Mass C16 is the fastest compressed operator here. Partition J4 remains about 2.18%/2.60% slower than mass C16.
The observed repeat standard deviations range from 0.136 to 0.611 ms; this small benchmark does not establish a general latency distribution or statistical significance.

## Native checks and limits

All methods pass the recorded native/Torch seed gates on heads 0/10/20/30; dispatcher outputs equal outputs supplied with the corresponding native seed tables.
Full40 partition J2/J4 tables satisfy row bounds 0–44, distinct rows, fixed first anchor, and shape `[1,1,21,45,80,40,4]`.
J2/J4 same-set fractions are 0.999999702/0.999998689, not bitwise equality; their output MSE differences are 3.11693e-11/2.62570e-10.
Rare numerical seed ties therefore remain relevant: the launch variants are very close, not proven identical for all inputs.

Conclusion: J4 removes the observed J2 timing disadvantage, but partition C16 does not beat mass C32 on error in these two calls.
This is a bounded historical RF10 implementation diagnostic, not an RF0 recommendation, a generated-video quality improvement, or evidence that PSNR exceeds 12.
