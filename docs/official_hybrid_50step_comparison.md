# Fifty-step comparison requested on 2026-09-06

Four methods, each using 20 canonical MovieGenVideoBench prompts, seed 0,
81 frames at 1280x720, original Wan2.1-T2V-14B-Diffusers, 50 UniPC steps,
flow shift 5 and CFG 5. This is not the four-step distilled checkpoint.
The previous seed-medoid exploration is paused for this comparison.

## Baseline setting decision

The user explicitly confirmed the released executable settings after the
script-versus-paper differences were explained. SVG, SVG2/SAP and SVG-EAR
are therefore authorized to run with their original released hybrid schedule.
The launcher uses the released scripts at upstream commit
`f89aedaf169ac2ae5b186bda674e53c3dc08c476`:

| Method | Dense schedule | Remaining attention |
| --- | --- | --- |
| SVG | First 10/50 steps; layer 0 in the remaining steps | Pattern parameter 0.3, 64 profiling rows |
| SVG2 / SAP | Same | Q/K clusters 300/1000, top-p 0.9, minimum ratio 0.10, K-means iterations 50/2 |
| SVG-EAR | Same | Same CLI settings; upstream implementation actually hardcodes minimum ratio 0.05 |
| Best measured GCA | None | Anchor-Fisher RF0 C32/U16, C4+SQ1, stable bulk normalization, full compressed route |

Official source scripts:
[SVG](https://github.com/svg-project/Sparse-VideoGen/blob/f89aedaf169ac2ae5b186bda674e53c3dc08c476/scripts/wan/wan_t2v_720p_svg.sh),
[SVG2](https://github.com/svg-project/Sparse-VideoGen/blob/f89aedaf169ac2ae5b186bda674e53c3dc08c476/scripts/wan/wan_t2v_720p_sap.sh),
[EAR](https://github.com/svg-project/Sparse-VideoGen/blob/f89aedaf169ac2ae5b186bda674e53c3dc08c476/scripts/wan/wan_t2v_720p_ear.sh).

SVG2's paper text instead specifies 30% warmup and 100/500 clusters.
The EAR paper's reported Wan setup uses Wan2.2 and top-p 0.85, not this
released Wan2.1 command. These are not interchangeable reproduction labels.

## GCA identity and scope

The chosen method is the existing stable Anchor-Fisher RF0 variant whose
canonical four-step PSNR is 11.64718233856182. That result determines which
existing method is tested; it is not a prediction or result for 50 steps.
The selector, C4+SQ1 partition and full route are unchanged. The same pure
operator must run in every layer and call; no dense rescue, FP8, learned
component, or newly proposed medoid selector participates.

The generated stable kernel must retain SHA-256
`fa2a729139cdbfee1abba9265eedf463c7f68b0f7de1f4402d6d48c2fe3a8e68`.
Each complete attention has one selector and five eight-head stable merge
calls. There are 4,000 self-attention calls per video (50 steps, 40 layers,
two CFG passes), hence 80,000 complete calls and 400,000 stable merge calls.

## Measurement and reproducibility

- Use fresh outputs only, with canonical `000-00.mp4` through `019-00.mp4`.
  Verify prompts independently in the parent and before the model loads.
- Reset CPU/CUDA/Python/NumPy RNGs for every video, as in the official
  single-prompt invocation. Also clear the original SAP/EAR centroid caches.
- Record core attention and full self-attention module latency, end-to-end
  inference time, per-layer branch counts and actual selector/kernel calls.
- Released hybrid settings require 880 dense and 3,120 sparse calls per
  video. SVG2/EAR density logs cover sparse calls only. Report their measured
  sparse density and all-call density `0.22 + 0.78 * sparse_density` separately.
  Top-p is not density, and EAR compensation is not an exact retained block.
- All generation uses the same monarch_rt runtime. The SVG CUDA extension
  was rebuilt privately for CUDA 12.9 because the existing shared binary
  requires unavailable CUDA 13. Shared upstream sources/binaries are not
  overwritten. The upstream source worktree is pinned separately.
- Subsequent launches use a run-private FlashInfer compilation cache so the
  scai7 CUDA 13 toolchain cannot share JIT outputs with scai6 CUDA 12.9. The
  already-running GCA/EAR source snapshots and their caches are unchanged.
- Admit only a currently clean H200. Monitor exclusivity and account-wide
  caps (scai6 at most two devices, scai7 at most one); stop only the owned
  child on contamination or failure. Do not prequeue a busy device.

The existing canonical 50-step dense reference is the Aug16 original run,
with runtime SHA-256
`b7a44477bd71eec0ef6f6cac6d19a68f33fe776b4dea946670f1aefd4e146c77`.
Its 20 prompts, seed, checkpoint path, step/frame counts and CFG match.
Its old source/fast-preprocessing equivalence is not established, so quality
metrics must identify it as the historical same-protocol reference. Its
historical timing must not become a matched current SDPA speedup denominator.
The better-instrumented Aug26/Aug28 dense runs use different prompts and
must not be substituted.

PSNR, SSIM, LPIPS and VBench are pending. No 50-step quality or speed result
has been obtained by preparing this comparison. External baseline results
belong in the existing baseline workbook; new GCA exploratory results remain
in the exploratory workbook unless the user promotes them.
