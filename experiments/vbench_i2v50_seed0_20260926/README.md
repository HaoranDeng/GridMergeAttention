# VBench I2V50: Dense and GMA-C4

The user requested a newly sampled set of 50 VBench I2V prompts on 2026-09-26,
then specified Dense followed by GMA-C4, initially using only one GPU.
Later on 2026-09-26, the user explicitly requested one additional GPU.
The current execution uses two H200 GPUs: the existing Dense job continues on
GPU2, while a separately supervised C4 job runs on GPU4. Frozen inputs and
model code are unchanged. `parallel_plan.json` records this authorization and
supersedes the execution/GPU assignment in the preserved original plan.

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

Dense uses scai6 GPU2, UUID
`GPU-244e87b8-5a10-aadb-858a-f0eea028dbd1`.
C4 uses the additional exclusively idle scai6 GPU4, UUID
`GPU-923ad71f-2808-6bf7-701e-dc79b7aaa38f`.
The original Dense process is neither stopped nor restarted.

1. Continue generating all 50 Dense videos on GPU2.
2. Concurrently validate C4 on GPU4 with native QKV from the first new input:
   zero-based layers 0, 5, 10, 15, 20, 25, 30, 35, both CFG branches.
   Preserve the existing numerical and dense-speed engineering gate, measuring
   C4 and SDPA on the same GPU4.
3. After the gate passes, generate all 50 GMA-C4 videos on GPU4 using Q80K90, exact capacity 4, all 40
   layers and all attention calls. Per-video audits require 4,000 C4 calls,
   100 calls per layer, and zero dense self-attention calls.
4. The original supervisor owns CPU evaluation, starting after Dense50 finishes.
   Evaluate completed C4 videos against their matching Dense references on
   CPU only: PSNR, SSIM and AlexNet LPIPS over all 81 frames at native output
   resolution. VBench scores remain pending.

Both methods use the existing frozen Wan2.1-I2V-14B-720P pipeline,
1280x720 output, 81 frames, 50 UniPC denoising steps, shift 5.0, CFG 5.0,
identical negative prompt and BF16. Configurations differ only in attention.
The source snapshot and all input identities are hashed and rechecked before
model loading. No training or calibration is performed.

Each supervisor checks its assigned GPU for foreign processes every ten seconds.
If one appears, it marks the relevant timings invalid and stops only its own work.
C4 failure does not interrupt the independently supervised Dense process.
The original supervisor delegates its later gate/C4 stages to the new owner,
preventing duplicate C4 generation. Six CPU checks cover delegation, partial
gate state, source-plan identity, failure propagation and GPU assignment.
The task uses at most two GPUs and does not interfere with another user's process.
Unverified resume is disabled. Audited recovery can use `resume_parallel.py`
only after the original workers have exited, the assigned GPU is clean, the
same-GPU native gate is valid, and every retained video matches its input
manifest, output hash, runtime row and full-layer call audit. Generation then
skips this verified contiguous prefix while retaining the original indices
and seeds.

Remote root:
`/local2/denghaoran/GridMergeAttention/experiments/vbench_i2v50_seed0_20260926`.

Local result root: `C:/GMA/experiments/vbench_i2v50_seed0_20260926`.
`stage.json`, `monitor.jsonl`, per-method `videos/runtime.json`, purity records,
and `C4/quality_cpu/quality.json` record progress and results. `parallel/`
contains the additional GPU's preflight, gate/C4 logs, monitor and status.
GPU workers run concurrently; the evaluator has `CUDA_VISIBLE_DEVICES=''`.
Keep Dense video timings from GPU2 separate from C4 timings on GPU4. The C4
native gate provides a same-GPU SDPA speed comparison on GPU4.

Report the 50-step FLOPs scope explicitly. Historical SVG-EAR and original
Q45K80C8 results use different inputs and must not be reported as matched
quality deltas for this new sample. Neither baseline is added to this run.

## Reproduce the selection

```text
python source/sample_inputs.py --metadata /path/to/vbench2_i2v_full_info.json --out /path/to/vbench_i2v_50_stratified_seed0_v1 --n 50 --seed 0
```

`prepare_remote.py` prepares the frozen native runtime and pinned images on
the shared experiment filesystem. `launch.py` launches the single-GPU
supervisor after clean-GPU and identity checks. `launch_parallel.py` performs
the authorized expansion while preserving the active Dense process. Inspect an existing dispatch
instead of launching it again.

## Interrupted C4 and audited recovery

The first GPU4 attempt stopped when another compute process appeared. It saved
four videos, indices 000-003. The interrupted attempt's logs, GPU monitor,
failure and timing-invalid markers are preserved under the path recorded in
`resume_plan.json`. Every retained video was saved before the last clean GPU
check preceding the foreign process. Unsaved interrupted work has no reported
runtime result.

After GPU4 became exclusively idle again, the run resumed on that same GPU
from index 004, seed 4, with the same frozen inputs, kernels and configuration.
Dense's original process was preserved. The validated native gate was reused
on the same GPU after confirming the frozen code hashes. Eleven CPU checks
cover ownership, non-duplication and rejection of invalid resume identities.
