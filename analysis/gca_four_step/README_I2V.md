# Four-step I2V and the shared evaluation fleet

The user expanded the Q80K90Even C4/C6/C8/C10 campaign to both I2V and T2V,
20 videos per capacity and task, and explicitly authorized every available
compatible GPU for this campaign. This supersedes earlier concurrency limits
for this run. Exclusivity and foreign-process monitoring still apply.

## I2V protocol

`i2v.py` uses the existing frozen I2V implementation under
`/local2/denghaoran/GridMergeAttention/experiments/q80k90_i2v_c4_c6_c8_c10_20260924/snapshot`.
Its critical sources, the input images/prompts, and the native kernel sources
are checked against the new campaign manifest before loading the model.

- Checkpoint: `Wan2.1-I2V-14B-720P-StepDistill-CfgDistill/distill_model.safetensors`.
- Existing `p04i_dense.yaml`: four-step re-noising sampler, shift 5, baked-in
  guidance, one branch, 160 attention calls/video, all 40 layers, BF16.
- 1280x720, 81 frames, Q80/K90, exact C, no dense layers or singleton routes.
- Inputs: `vbench_i2v_20_sparkler_v1`; seed equals zero-based prompt index.
  The replacement of input019 was selected post hoc; this history remains.
- The existing config documents that the 720P I2V checkpoint adapts 480P
  distillation. Preserve that qualification when reporting these results.
- Generate 20 fresh matched four-step Dense references. Never compare this
  run with the historical 50-step Dense outputs or the old input019.
- Each generation worker passes 3 I2V prompts x 4 timesteps x the fixed layers
  0/5/10/15/20/25/30/35: 96 real-QKV cases, all 40 heads, native versus the
  previous v4 implementation and same-GPU Dense. The .01 L2 gate is an
  engineering check. Sampling seeds are reset before every video.

The new campaign root is
`/local2/denghaoran/GridMergeAttention/experiments/gca_i2v_q80k90_4step_20260926`.
Every method has immutable input identities, individual runtime/purity records,
and atomic per-index claim directories. A failed claim remains visible for
diagnosis; workers never silently overwrite or reclaim failed videos.

## Dispatch

Six H200 plans cover C4 (scai7 GPU0), C6 (scai6 GPU2 plus scai7 GPU3), C8
(scai6 GPU4 plus scai7 GPU7), and C10 (scai6 GPU5). The two former T2V workers
start I2V only after their own successful T2V completion and a new clean-GPU
preflight. Four workers divide the fresh Dense references into five inputs each.
Multiple workers of the same capacity claim distinct remaining indices.

`fleet.py` launches a frozen per-GPU plan, records the actual UUID/model and
process group, and stops only its own processes if a foreign process appears.
It never waits for a foreign GPU job to finish. One L40S became occupied between
the initial scan and launch; its preflight correctly refused to start.

## Evaluation

Sixteen additional workers successfully started on the available scai2 TITAN
Xp, scai3 A100, and scai4 A6000 cards. These GPUs cannot run the Hopper native
GCA kernel, so they evaluate outputs through `metrics.py` instead. The existing
VBench environment, PyTorch 2.7.1+cu118, also supports TITAN Xp; the generation
environment, PyTorch 2.8.0+cu128, does not. GPU types, versions and timings remain
explicit. Do not combine generation runtimes across different GPU models.

PSNR/SSIM/LPIPS use the frozen I2V evaluator, all81 decoded frames, original
resolution, and AlexNet LPIPS. The unchanged LPIPS package is copied into a
campaign-local vendor directory because the VBench environment lacks it;
package source and weight hashes are recorded. The matching `imageio_ffmpeg`
decoder is also copied and hashed locally. Each GPU model/environment checks
one full video against a frozen CPU result before publishing new I2V metrics;
the check is numerical validation, not fitting or calibration of GCA.
GPU metric execution is labelled
separately from the existing T2V CPU evaluator. Input and output hashes are
checked before and after each evaluation. No frames are sent to a language
model. Existing T2V quality rows are preserved.

VBench uses the existing six-dimension custom-input protocol: subject and
background consistency, motion smoothness, dynamic degree, aesthetic quality,
and imaging quality (the existing longer-side preprocessing). This is not the
full official leaderboard score. Each task/capacity and Dense reference has a
separate job per dimension, allowing all compatible GPUs to share the queue.
TITAN Xp takes the smaller consistency/aesthetic jobs and per-video quality.
Native VBench dimension-specific sampling/preprocessing is unchanged.

Each VBench subprocess has world size one. A private FileStore avoids local
TCP rendezvous collisions with unrelated server jobs. Initial port-conflict
failures were archived and retried; their result files were never promoted.

SVG-EAR and original Q45K80C8 remain external comparison rows with matched
four-step results pending. This campaign does not silently substitute their
historical 50-step measurements. No paper files are changed.
