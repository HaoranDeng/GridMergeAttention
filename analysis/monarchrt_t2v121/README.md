# MonarchRT121 T2V20

User request on2026-09-25: add T2V f_tied=1, h_reduce=2, w_reduce=1.
Generate20 videos with the historical Wan2.1-T2V-14B Diffusers pipeline:
1280x720,81 frames,50 UniPC steps, flow shift5, CFG5, constant seed0 for
every prompt. This differs from the I2V per-index seed policy. Use the
canonical MovieGenVideoBench_extended first20 prompts and matching historical
Dense references; freeze and verify every prompt and reference-video hash.

The native MonarchRT one-step operator runs in all40 layers and both CFG
branches,4000 calls per video. Keep q_init=ith, head_chunk=1,
query_outer_chunk=21. The requested change from historical122 is w_reduce2
to1. The original historical working-tree source is frozen with hashes;
its pre-existing dirty files are recorded, not silently presented as a clean
commit. Existing experiments and their source files are never modified.

Major-matmul FLOPs:38.0226502656 PFLOPs/video over50 steps;
supporting scope:9.5056625664 TFLOPs/call. This includes H45 padded to46.
Report measured core attention ms/call, full attention ms/call, inference
seconds/video, GPU model/UUID, full81-frame PSNR/SSIM/AlexLPIPS, and the existing
six-dimension custom-input VBench composite (not an official leaderboard score).

Before video generation, the native gate validates layers0,5,10,15,20,25,30,35
in both CFG branches of step0, against the same operator in FP32 with TF32 off.
Its0.02 relative-L2 threshold is an engineering check, not a user-specified
universal tolerance. Paired native and SDPA timings use the same exclusive GPU.
The gate is a diagnostic and does not insert Dense attention into generated
videos. External-baseline generation does not require beating SDPA speed.

Preparation is CPU-only and refuses to overwrite an existing frozen plan.
The launcher requires an explicitly selected, twice-checked idle H200;
historical matching runs peaked at about58 GiB. It checks old queue reservations,
monitors foreign GPU processes, and terminates only its own group if contention
appears. It never waits for or queues a future GPU. Do not re-use contaminated
timings as clean speed evidence. Failed attempts and partial videos remain
available for diagnosis. Resumption requires inspecting the prior attempt.

Canonical experiment directory:
`/local2/denghaoran/GridMergeAttention/experiments/monarchrt_t2v121_20260925`.
Local artifacts: `C:/GMA/experiments/monarchrt_t2v121_20260925`.
Final baseline metrics belong in the existing `C:/GMA/experiments.xlsx`.

Copy these Python files into the experiment's `source/` directory. Alternatively
set MONARCH121_ROOT to the experiment directory when using the repository copy.
Run `prepare.py` once with the established MonarchRT Python environment.
When an H200 is exclusively available, invoke `launch.py --gpu INDEX --attempt 1`
on scai6 or scai7. The detached supervisor runs gate, generation, and VBench,
with a separate CPU quality evaluator. Generation status is in video20/runtime.json;
each launch has preflight.json, active.json, completion.json, and monitor logs.

Initially all compatible H200 GPUs were busy. scai6 GPU4 subsequently became
exclusively idle and its old queue process had exited. Attempt1 started on
2026-09-25, passed all16 native FP32 checks, and entered20-video generation.
The scorer's missing helper import was frozen separately in evaluation_support;
the full scorer import now passes the CPU readiness check. Its initial failure
is preserved. `source/status.py` reports actual progress in latest_status.json.
