# RF0 partition-seed failure localization

Read-only artifact diagnosis, 2026-09-04. This note changes no selector,
production configuration, reference frame, or running experiment. The main
finding is concentrated regression, not uniform ineffectiveness at step 0.
This is attention-output error on a fixed external dense trajectory, **not
video PSNR, SSIM, LPIPS, or an end-to-end quality result**.

## Inputs and aggregation contract

Local artifact directory:
`C:/GMA/experiments/native_seed_trajectory_rf0_20260904g/`.

- `screen.json`, SHA256
  `c147b7496585e32d98bfa8805462fced74ceff9805ce57bbb521bb326c2e27ed`.
- `validated_summary.json`, SHA256
  `21c9c9c776d279e52c7c639d3b5ba5eeccfa8eb58c23f460d8b316b3e506e02c`.
  It records `valid=true`, `complete=true`, exit code 0, and zero purity
  violations. Overall and per-step sums below were independently recomputed
  from the 480 records and agree with this summary.
- Recorded screen source: `analysis/screen_native_seed_trajectory.py`, SHA256
  `cc2fdcdd3fd83823c31548d7248529a52c781a81358bb717413d22fff969b538`.
  The recorded original partition kernel SHA256 is
  `93dbfd0b5c25171fef696af8f7d9afdae7e95064212711be17999997dac95775`.

The source records three canonical prompt indices 0, 1, 2, four denoising
steps, 40 layers per step, all 21 query frames, and sampled original heads
`[0,10,20,30]`: `3 * 4 * 40 = 480` calls. Layer and step indices in this note
are zero-based. The model configuration is
`configs/dense_wan_14b_720p_fewstep.yaml`; the recorded loaded checkpoint branch
is `raw_state_dict`, with warped timesteps
`[1000,937.5,833.3333129882812,625]`. Candidates do not advance the model.

The comparison is `partition_c16` against `rf0` (Hellinger C32/G45). Both
explicitly use RF0 and Uniform16; partition uses C16/G16, G256, J4 and the
recorded four-warp/owner-logp-reuse override. SQ1 merge and full compressed
attention remain in scope. These are different seed-selection methods,
not different dense-reference outputs.

For any selected records, `MSE = sum(squared_error_sum) / sum(numel)`;
`delta_percent = 100 * (MSE_partition / MSE_rf0 - 1)` (negative is better).
No per-call or per-layer percentage changes are averaged to obtain a headline.

Every record has 38,707,200 elements (`1 * 75600 * 4 * 128`). Thus each step
has 4,644,864,000 elements and the total has 18,579,456,000. The source's
`frame_mse` contains 21 equal-size marginals, 1,843,200 elements each;
`head_mse` contains four equal-size marginals, 9,676,800 elements each. For
these marginals only, reconstruct `SSE = marginal_mse * marginal_numel`, then
sum SSE and counts. Their means reproduce each call's direct SSE/numel with
maximum relative discrepancy below `2e-7` (FP32 reduction rounding).
The JSON has no joint frame-by-head error array; separate frame and head
effects must not be treated as an observed joint intersection.

## 1. Step 0 loses its otherwise broad gains to one layer

| Scope | RF0 MSE | Partition MSE | Relative change | Improved calls |
| --- | ---: | ---: | ---: | ---: |
| All 480 calls | 0.00466980217 | 0.00435240985 | -6.79670% | 428/480 |
| Step 0 | 0.00455037047 | 0.00455019172 | -0.003928% | 99/120 |
| Step 1 | 0.00484371946 | 0.00459584857 | -5.11737% | 106/120 |
| Step 2 | 0.00487244211 | 0.00435549069 | -10.60970% | 112/120 |
| Step 3 | 0.00441267666 | 0.00390810841 | -11.43452% | 111/120 |

Step 0 accounts for 24.3606% of RF0's total SSE, so it is not a negligible
step. Steps 2 and 3 contribute 80.4619% of the overall net SSE reduction.
At step 0, 32/40 layer aggregates improve, but layer 33 increases SSE by
2,109,280.75 while the other 39 layers together reduce it by 2,110,111.04.
The net reduction is only 830.2871.

Layer 33's step-0 MSE is `0.00548931842 -> 0.02365373184` (+330.9047%). It
accounts for 96.6988% of the gross SSE increase across the eight worsening
layer aggregates. Its baseline share of step-0 SSE is 3.01586%, rising to
12.99601% of partition SSE. This repeats across all three prompts: the layer's
MSE increases by +265.04%, +364.67%, and +379.35%, respectively.

For scale, the largest improving layer contributions are layer 25
(-33.1805%, saving 1.74618 percentage points of total step-0 baseline SSE),
layer 19 (-36.6468%, saving 0.93310 points), and layer 23 (-20.8539%, saving
0.92575 points). These influence decompositions are not proposed layer masks.

## 2. The exceptional regression is concentrated in sampled head 30

| Original sampled head | Step-0 change | All-480-call change |
| --- | ---: | ---: |
| 0 | -9.19422% | -13.98238% |
| 10 | -10.92007% | -15.50217% |
| 20 | -4.95320% | -8.97486% |
| 30 | +25.41862% | +11.04889% |

At layer 33/head 30/step 0, MSE is `0.02188960152 -> 0.09454767654`
(+331.9296%), adding approximately 2,109,292.98 SSE. This accounts for
essentially the entire layer regression; the slight difference from the
all-head layer increment is offset by the other three heads, plus marginal
rounding. The same layer/head also worsens at steps 1, 2, and 3 by +362.35%,
+236.89%, and +237.29%. The issue persists after step 0; later steps simply
have larger compensating gains elsewhere.

This establishes concentration among the **four sampled heads only**. It does
not establish the corresponding distribution over all 40 model heads.

## 3. Reference-frame error is worse, but temporal reuse is not a sufficient explanation

At step 0, query frame 0 worsens by +25.52584%, adding 158,467.11 SSE; the
other 20 query frames together improve by -0.77649%, saving 159,297.40 SSE.
Their net again matches the approximately 830.29 step-0 SSE reduction.
Across all 480 calls, frame 0 worsens by +6.51435%, whereas the other 20
frames improve by -7.27184%.

However, the problematic layer 33 at step 0 worsens both on query frame 0
(+323.24%) and on the other 20 frames together (+331.21%). Therefore the
regression cannot be explained solely as a failure to reuse RF0-selected
row indices at other frames. This observation does not authorize changing RF0.

## 4. A larger candidate bank alone did not fix this case

Layer 33/head 30/step 0, all three prompts:

| Existing method | MSE |
| --- | ---: |
| RF0 Hellinger C32/G45 | 0.02188960 |
| Mass-span C32/G45 | 0.10538329 |
| Mass-span C16/G16 | 0.10541077 |
| Partition-deficit C16/G16 | 0.09454768 |

The two mass-span banks produce nearly the same large error here. Partition
deficit improves on them, but remains far worse than RF0 Hellinger. Thus
shrinking C32 to C16 is not, by itself, a sufficient explanation. These
outputs do not identify whether importance weighting, probe approximation,
the first seed, later greedy choices, or their interaction is responsible.

## 5. A bounded, unproven medoid hypothesis

The [first-seed medoid proposal](rf0_first_seed_medoid_proposal.md) provides
one testable initialization change: perhaps the mandatory row anchor poorly
represents the important candidate population in this difficult layer/head.
Freeing that first slot might improve the represented partition. The present
JSON contains no seed tables, logits, importance vectors, or cell diagnostics,
so it neither verifies this hypothesis nor predicts improvement.

A later diagnostic would need to compare anchor-to-medoid surrogate costs,
actual selected rows, and actual output SSE for this layer/head and all three
prompts. A lower one-cell reverse-KL bound does not guarantee lower C4 error
or better videos; the same mass weights could preserve the problematic bias.
Existing call020/call100 captures alone do not cover layer 33. Any candidate
must retain one training-free rule in every layer/call, RF0, and the unchanged
full compressed attention path. No hybrid, dense rescue, head exception,
frame exception, or exclusion of layer 33 is recommended or authorized.

## Bounded historical capture inventory

Read-only SSH inventory on 2026-09-04 was restricted to
`/local2/denghaoran/GridMergeAttention/experiments/learned_seed_capture_20260822`.
All eight `qkv.pt.npz` files were inspected via ZIP/NPY headers and small
metadata members, plus their adjacent capture logs. No large Q/K/V payload
was loaded, no pickle or Torch model was loaded, and no GPU was contacted.

All eight archives contain full 40-head Q/K/V arrays, stored FP16 (`<f2`)
with archive layout `[1,40,75600,128]`; metadata records original attention
layout `[1,75600,40,128]` and tag `wan_self_attn`.

| Directory under the authorized root | Call evidence | Archive mtime, UTC on 2026-08-22 |
| --- | --- | --- |
| `train` | Log: first attention; no `call_index` member | 07:49:27 |
| `holdout` | Log: first attention; no `call_index` member | 07:49:26 |
| `holdout2` | Log: first attention; no `call_index` member | 07:59:47 |
| `holdout3` | Log: first attention; no `call_index` member | 07:59:53 |
| `call020_train` | Metadata and log: call 20 | 08:05:27 |
| `call020_holdout` | Metadata and log: call 20 | 08:05:17 |
| `call100_train` | Metadata and log: call 100 | 08:11:24 |
| `call100_holdout` | Metadata and log: call 100 | 08:17:12 |

No `call033`, `call073`, `call113`, or `call153` capture was found, and no
four-head-only capture was present. Under the current zero-based 40-layer
mapping those four calls would cover layer 33 at steps 0--3; call 20/100
cover layer 20 at steps 0/2 instead. The old files lack an independent layer
index, so that historical step/layer mapping is conditional on the same
40-layer capture convention. The logs' "first attention" cases also do not
supply layer-33 coverage. A new diagnostic capture would be needed to inspect
layer-33 Q/K/V; none was generated or scheduled by this audit.

Only `train/runtime.json` and `holdout/runtime.json` supply readable model
configuration metadata: `configs/dense_wan_14b_720p_fewstep.yaml`, checkpoint
`/local2/denghaoran/GridMergeAttention/wan_models/Wan2.1-T2V-14B-StepDistill-CfgDistill/distill_model.pt`,
and 160 calls/video, with prompt indices 0 and 1 respectively. Their metadata
does not establish the loaded raw/EMA branch, checkpoint hash, or canonical
per-prompt hashes. This model identity must not be silently assigned to every
other archive. Archive mtimes are filesystem times, not proof of run start
time or checkpoint identity. The historical `train`/`holdout` names are only
directory labels; this inventory does not introduce a trained GCA component.

## Limits and reporting

The dense trajectory isolates same-QKV operator error but does not measure
error accumulation when GCA advances the model. Three prompts and four heads
are not a full-40-head or 20-video quality evaluation. This post-hoc
localization is diagnostic, not a significance test or a claim that the
identified mechanism is causal. Sampled-head timings in `screen.json`
include first-call compilation: matched full-attention `mean_attention_ms`
and `speedup_vs_sdpa` are unavailable for this stratified diagnosis and are
not borrowed from another benchmark. No new run or promotion follows from
the aggregation alone.
