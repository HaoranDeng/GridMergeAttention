# Fixed RF0 seed-selection experiments

> **RETIRED C4+SQ1 PROTOCOL:** the configurations below were 4+1 historical
> controls and have been removed. C4 is now exactly four merged K/V
> representatives and must never route a fifth singleton.

As requested on 2026-09-04, all new seed-selection candidates and their matched
Hellinger controls use RF0: latent query frame index 0 of 21. The shared table
contains row indices only. Merge still loads real queries from each current
frame, and route attends to the full compressed K/V set. This does not retain
dense first-frame attention or add a dense rescue path.

## Default and comparison configurations

The user selected **Mass-GS C32 / RF0** as the default on 2026-09-04.
At that time `generate.py` defaulted to the Mass-GS C32 configuration below.
This changes the entry-point choice, not the kernel, historical configurations,
or optional precision switches. Choosing a default does not imply that the
PSNR or no-slowdown research targets have been achieved.

Four-step configurations (Hellinger remains the matched control):

- Hellinger: `token_merging_wan_720p_hellinger_span_c32g45_a0p2_p16_j4_t1_rf0_sq1_fp8score_qm_skv_nl_ml2_hc8_fewstep.yaml`.
- Mass-GS C32: `token_merging_wan_720p_entropy_mass_span_rf0_fewstep.yaml`.
- Mass-GS C16: `token_merging_wan_720p_entropy_mass_span_c16_rf0_fewstep.yaml`.
- Partition-deficit C16: `token_merging_wan_720p_entropy_partition_mass_c16_rf0_fewstep.yaml`.

All use an explicit `_t1_rf0_u16_` mode, Uniform-16 probes, seed 0, the canonical
MovieGenVideoBench extended prompt prefix, and the same operator at every
self-attention call. The exact listed partition mode now requests the measured
4-warp/owner-log-probability-reuse launch in production. Other modes preserve
their historical settings. Eight CPU dispatch checks passed in an isolated
staging copy. Direct GPU parity also passed on both complete 40-head/21-frame
captures in `partition_rf0_production_parity_20260904k`: real seed tables and
complete BF16 outputs were bitwise equal to the measured override. Selector,
normalizer, candidate-bank and merge/route source hashes match the earlier
benchmark; only the intended dispatch port and benchmark metadata changed.
Completed diagnostics retain their original override metadata and are not
video timing measurements. Parity itself is not a latency benchmark.

Legacy modes without `_rf0` are deliberately unchanged: a single centered
router over 21 latent frames historically meant RF10. Old configs, scripts,
measurements, and result labels remain reproducible and are not RF0 evidence.
The fixed two-frame joint selector is excluded from the new RF0 candidate set.

`analysis/screen_native_seed_trajectory.py` evaluates the four frozen RF0
operators on all 160 calls of each canonical four-step dense trajectory. The
dense output alone advances this external diagnostic. It samples four heads
and all 21 query frames without saving full QKV. Its output MSE and sampled-head
latency do not substitute for 20-video PSNR/SSIM/LPIPS or complete 40-head timing.

The final target remains 20-video PSNR above 12 and no-slower-than-Hellinger
runtime. New matched comparisons must use RF0 on both sides; RF10 historical
results retain their original reference-frame label.

## Historical RF0 control audit

The existing run `wan14b_720p_4step_hellinger_p16_j4_rf0_20260829` under
`/local2/denghaoran/GridMergeAttention/experiments` contains `runtime.json`,
`results/quality.json`, and `videos/prompt_manifest.json`. Read-only checks
confirmed filenames `000-00.mp4` through `019-00.mp4`, 81 evaluated frames per
video, 160 attention calls per video (3,200 total), and matching canonical
prompt hashes and seed 0 against its recorded dense reference.

This is not the current Uniform-16 control. The historical selector in commit
`32b76df` used `top_probes=probes // 2` and `uniform_probes=probes // 2`, i.e.
8 adaptive plus 8 uniform probes for P16. Uniform-16 was introduced by
`dd4422c` and made the default by `d4ec118` on 2026-09-04. Updating today's
config does not change the probe policy used to produce those old videos.

The old quality reference is
`wan14b_720p_4step_first_frame_global_hellinger_opt_20260829/dense_flash/videos`,
not the current dense20260826 reference. No complete purity or exit-code
artifacts were found in that run directory. Other located RF0 runs use C45 or
two-stage selection and are also not matched controls. The required canonical
20-video Uniform-16 RF0 control has now completed in
`wan14b_720p_4step_hellinger_uniform16_rf0_extended_20260904i`, with the matched
Mass-GS C32 run in `wan14b_720p_4step_mass_gs_c32_rf0_extended_20260904r`.
Do not substitute the historical A8U8 RF0 or RF10 results for this pair.
