# SVG-EAR p=0.4, matched I2V20

This campaign supplies the missing fifth SVG-EAR point for the I2V trade-off curve. It uses the same frozen native Wan2.1-I2V-14B-720P implementation, configuration, Dense references and `vbench_i2v_20_sparkler_v1` inputs as the existing p=0.5–0.8 points. Only `SVG_TOP_P` changes to `0.4`.

- 20 videos; seeds 0–19 match prompt indices.
- 720×1280, 81 frames, 50 FlowUniPC steps, CFG5, shift5.
- Q/K centroids300/1000, minimum K ratio0.10, initial/subsequent iterations50/2.
- Exactly4000 audited self-attention calls per video. Actual density/iteration FLOPs and the existing figure's scheduled clustering upper bound are separate.
- Frozen CPU PSNR/SSIM/AlexLPIPS over all81 original-resolution frames; every input, Dense and generated video is hashed.
- All20 videos run on one exclusively idle H200. Preserve every runtime and exclude first-video warmup from the steady mean. Do not pool different GPUs or campaigns.

Input019 was selected post-hoc in the earlier experiment. Adopting this set as the default does not change its selection history.

## Scripts

The scripts reproduce this workspace's existing frozen campaign; its checkpoints, frozen source snapshots and private experiment artifacts must already exist. No model weights or videos are included here. `prepare.py` verifies these dependencies and fails on mismatches.

- `common.py`: atomic records and hashes.
- `prepare.py`: once-only protocol and input verification.
- `watch.py`: detached per-host watcher for scai6/scai7. A shared file lock elects one exclusive idle H200; polls every20seconds, checks CPU-side GPU reservations and verifies idle state twice. The loser exits after launch. A5-second contention guard can stop only its own child process group, preserving other work and completed artifacts. Waiting is bounded at7days and generation at24hours.
- `worker.py`: unchanged SVG-EAR inference with per-layer audits and full50-step cost traces.
- `evaluate.py`: single-instance streaming CPU evaluation.
- `deploy.py`: Windows-side once-only deployment and watcher startup through configured SSH.
- `sync.py`: verified incremental synchronization, compact status and a plot update packet only when all20 samples and uncontended generation are complete.

Scripts execute under an isolated experiment directory. The Windows coordination root is `C:/GMA/QKC scan/SVG_EAR_p04_I2V20_20260925`; the shared server root is `/local2/denghaoran/GridMergeAttention/experiments/svg_ear_p04_i2v20_20260925`. Neither a new launch nor automatic failure retry overwrites an existing experiment. Inspect launch receipts, current process identity, logs and committed hashes before any recovery.

The follow-up integration adds only this point, its provenance and any necessary figure caption change. Existing curve points and unrelated paper edits remain independently owned. A generated video or partial mean is not a completed point.
