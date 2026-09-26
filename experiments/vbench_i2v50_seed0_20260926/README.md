# VBench I2V50: Dense and GMA-C4 on one H200

The user requested a newly sampled set of 50 VBench I2V prompts on 2026-09-26,
then specified Dense followed by GMA-C4, using only one GPU.

## Frozen inputs

- Dataset: `vbench_i2v_50_stratified_seed0_v1`.
- Source: VBench `vbench2_beta_i2v/vbench2_i2v_full_info.json`, 1,118 records,
  SHA256 `8cf4d34dc11779a336c216a1a71d482927cfa24a1bf736ab5c92425609f129f8`.
- Population: 355 distinct images with a base prompt. Camera-motion variants
  are excluded so repeated scenes do not take multiple sample slots.
- Sampling: proportional image-type quotas by largest remainder, then
  `random.Random(0).sample` over sorted image names in sorted categories,
  followed by a shuffle of the 50 selected entries. Selection was frozen before
  generation, with no output inspection or replacements.
- Raw English prompts, no prompt extension. Generation seed equals the
  zero-based selected prompt index, 0 through 49, for both methods.
- Prompt file SHA256:
  `522277ae0ea5366a501dc1f879e2409c0667c737a4bb5cedfb206fa76917d632`.
- Matching images are downloaded from a single pinned revision of
  `dc-ai/vbench_i2v`, `crop/16-9`. Their SHA256, source URL, dimensions and
  revision are recorded in the input manifest.
- This is a separate input set. It does not replace the default I2V20 set or
  change either existing 50-video job. It is not asserted to be disjoint from
  previous inputs.

The text, image-name list and sampling manifest are retained in the code
repository under `prompts/i2v/vbench_i2v_50_stratified_seed0_v1.*` and locally
under `C:/GMA/prompts/i2v/`. The campaign's `inputs/` contains identical files.

## Execution

One exclusively idle H200: scai6 GPU2,
UUID `GPU-244e87b8-5a10-aadb-858a-f0eea028dbd1`.

1. Generate all 50 Dense videos.
2. Validate C4 on the same GPU with native QKV from the first new input:
   zero-based layers 0, 5, 10, 15, 20, 25, 30, 35, both CFG branches.
   Preserve the existing numerical and dense-speed engineering gate.
3. Generate all 50 GMA-C4 videos using Q80K90, exact capacity 4, all 40
   layers and all attention calls. Per-video audits require 4,000 C4 calls,
   100 calls per layer, and zero dense self-attention calls.
4. Evaluate completed C4 videos against their matching Dense references on
   CPU only: PSNR, SSIM and AlexNet LPIPS over all 81 frames at native output
   resolution. VBench scores remain pending.

Both methods use the existing frozen Wan2.1-I2V-14B-720P pipeline,
1280x720 output, 81 frames, 50 UniPC denoising steps, shift 5.0, CFG 5.0,
identical negative prompt and BF16. Configurations differ only in attention.
The source snapshot and all input identities are hashed and rechecked before
model loading. No training or calibration is performed.

The supervisor checks for foreign GPU processes every ten seconds. If one
appears, it marks timings invalid and stops only its own work. It never
borrows a second GPU or interferes with another user's process. A fresh run
is required; automatic unverified resume is deliberately disabled.

Remote root:
`/local2/denghaoran/GridMergeAttention/experiments/vbench_i2v50_seed0_20260926`.

Local result root: `C:/GMA/experiments/vbench_i2v50_seed0_20260926`.
`stage.json`, `monitor.jsonl`, per-method `videos/runtime.json`, purity records,
and `C4/quality_cpu/quality.json` record progress and results. GPU workers run
sequentially; the evaluator has `CUDA_VISIBLE_DEVICES=''`.

Report the 50-step FLOPs scope explicitly. Historical SVG-EAR and original
Q45K80C8 results use different inputs and must not be reported as matched
quality deltas for this new sample. Neither baseline is added to this run.

## Reproduce the selection

```text
python source/sample_inputs.py --metadata /path/to/vbench2_i2v_full_info.json --out /path/to/vbench_i2v_50_stratified_seed0_v1 --n 50 --seed 0
```

`prepare_remote.py` prepares the frozen native runtime and pinned images on
the shared experiment filesystem. `launch.py` launches the single-GPU
supervisor after clean-GPU and identity checks. Inspect an existing dispatch
instead of launching it again.
