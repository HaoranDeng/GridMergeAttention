# FLOP-Bounded Hellinger-Span Seed Selection

## Scope

This study changes only the four seed rows used by all-layer Hard C4. The
key-softmax assignment, entropy correction, merged K/V construction, and full
query-to-compressed-token attention remain unchanged. There is no hybrid or
dense-layer schedule.

Algorithm screening uses 48 real Wan2.1-14B 720p QKV points: four heads, three
query frames, call 20 and call 100, and train/holdout captures. Video validation
uses four matched Wan2.1-T2V-14B-StepDistill-CfgDistill samples at 1280x720,
81 frames, seed 0, and exactly four denoising steps.

## FLOP budget

For height `H=45`, width `W=80`, and `C=4`, the dominant Hard-C4 attention
cost relative to dense QK+AV is

```text
rho_C4 = 3C / (2H) + C / W = 0.183333.
```

A Hellinger selector with `R` candidate rows, `P` probes, `A` adaptive-probe
anchors, and probe sharing over `B_J` adjacent columns adds

```text
rho_selector = RP / (2HW) + A / (2H B_J).
```

The selected hardware-aligned setting uses `R=22`, `P=16`, `A=2`, and
`B_J=4`:

```text
rho_total = 0.237778 = 1.296970 * rho_C4
FLOPs/call = 27.832 TFLOPs for 40 heads
Hard-C4 FLOPs/call = 21.459 TFLOPs
Dense QK+AV FLOPs/call = 117.051 TFLOPs
```

This satisfies the requested 25-28 TFLOPs/call range. A 24-candidate,
12-probe alternative is nominally 27.442 TFLOPs and has slightly lower QKV
error, but a 12-wide reduction is padded to the 16-element Tensor Core
granularity in a practical kernel. The 22x16 setting therefore has the more
honest hardware cost model.

## Method

For candidate query row `r` and the 16 probe keys `K_P`, form

```text
p_r = softmax(q_r K_P^T / sqrt(d)),
h_r = sqrt(p_r) - mean(sqrt(p_r)).
```

The square-root map makes inner products correspond to Bhattacharyya affinity,
so Gram-Schmidt diversity is measured directly between attention
distributions. It removes both the V reads and the `p_r V_P` product used by
the original value-profile selector.

The 22 candidate rows combine a local neighborhood with globally spread rows
and one-dimensional farthest-point fill. The table is deterministic and
precomputed. Eight probes are selected by two anchor queries at row offsets
`(-1,+1)` and eight probes are uniform. Probe sets are shared by four adjacent
query columns. The matched key row is fixed as the first seed; the remaining
three seeds maximize importance-weighted residual Hellinger energy under
greedy Gram-Schmidt.

## Real-QKV results

| Selector | Candidates x probes | Rel-L2 | MSE | TFLOPs/call | Cost / Hard C4 |
|---|---:|---:|---:|---:|---:|
| Fixed Hard C4 | 4 x 0 | 0.177331 | 0.00677547 | 21.459 | 1.000x |
| Original value Weighted-GS | 45 x 16 | 0.134323 | 0.00388754 | 38.69 | 1.803x |
| Full Hellinger-GS | 45 x 16 | 0.128867 | 0.00357814 | 35.77 | 1.667x |
| Hellinger-GS multiscale | 28 x 16 | 0.131307 | 0.00371492 | 31.34 | 1.461x |
| Hellinger-GS multiscale, nominal | 24 x 12 | 0.138768 | 0.00414906 | 27.442 | 1.279x |
| **Hellinger-GS, hardware aligned** | **22 x 16** | **0.141214** | **0.00429659** | **27.832** | **1.297x** |

The production Triton selector exactly reproduced the Python selector on the
screened QKV points. Relative to fixed Hard C4, the selected setting reduces
rel-L2 by 20.37% and MSE by 36.58%. It gives up 5.13% rel-L2 to the much more
expensive original full-pool value selector.

## Four-step 14B video results

The four-video H200 comparison uses byte-identical StepDistill checkpoints,
the same four prompts, seed 0, 1280x720, 81 frames, all 40 layers, and 160
attention calls per video. The stored baseline and candidate runs use the same
H200 model but were not interleaved in one process.

| Method | PSNR | SSIM | LPIPS | Core attn ms/call | Full attn ms/call | Inference s/video |
|---|---:|---:|---:|---:|---:|---:|
| Dense SDPA (reference) | -- | -- | -- | 211.350 | 234.193 | 51.524 |
| Hard C4 | 9.8794 | 0.2953 | 0.6532 | 109.665 | 132.505 | 35.101 |
| Weighted-Span C4, pool30 | 10.5942 | 0.3502 | 0.5916 | 596.983 | 618.857 | 112.767 |
| **Hellinger-Span C4, 22x16** | **11.2214** | **0.3566** | **0.5628** | **650.966** | **673.130** | **121.501** |
| SVG-EAR all-layer | 11.4076 | 0.3480 | 0.5787 | 164.060 | 186.746 | 43.787 |

Against fixed Hard C4, Hellinger-Span improves PSNR by 1.342 dB, SSIM by
0.0613, and LPIPS by 0.0904. It also beats the original C4 Weighted-Span by
0.627 dB PSNR, 0.0063 SSIM, and 0.0289 LPIPS. Against SVG-EAR it has 0.186 dB
lower PSNR but 0.0086 higher SSIM and 0.0159 lower LPIPS.

These four videos are an algorithm-screening set, not a paper-level final
estimate. The quality ordering should be rechecked on at least the full
20-prompt matched set after the selector kernel is optimized.

## Runtime interpretation

The current kernel validates the algorithm but is not throughput-ready. Its
650.97 ms core attention latency is 3.08x slower than dense SDPA and 5.94x
slower than fixed Hard C4 despite the lower mathematical FLOP count. The
selector uses sequential reductions and materializes probe and seed tables;
the measured latency therefore reflects implementation inefficiency rather
than the 27.832-TFLOP arithmetic budget. It is also 9.04% slower than the
original value Weighted-Span kernel, so the current result supports a
quality-versus-FLOPs claim only, not a speed claim.

## Decision

The 22-candidate, 16-probe Hellinger selector is the current algorithmic
candidate: it meets the 25-28 TFLOPs budget and improves real video quality
over both fixed Hard C4 and the original C4 Weighted-Span. It is not yet a
speed candidate. A paper claim should separate the quality/FLOP result from a
future fused-kernel result and should not claim wall-clock acceleration from
the present implementation.
