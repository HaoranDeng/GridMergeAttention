# Native p=.4 timing with density logging disabled

Steady short-test mean: **88.493686 ms/core attention call**, **111.438971 ms/full attention call**.

One fixed input000, seed0, on exclusive scai6 H200 GPU2. Execute only the first four steps of the unchanged50-step I2V schedule. Step0 (including k-means initialization) is reported separately; steps1-3 provide240 steady calls across all40 layers and both CFG branches. Density logging and the extra iteration audit are disabled. The original native CUDA-event recorder measures core and full attention, including their original layout-conversion scopes. No stage replay instrumentation is installed.

| Step (zero-based) | Calls | Core ms | Full attention ms |
|---|---:|---:|---:|
| 0 (initialization) | 80 | 341.281443 | 371.195784 |
| 1 | 80 | 87.265115 | 110.185601 |
| 2 | 80 | 88.821791 | 111.667105 |
| 3 | 80 | 89.394153 | 112.464207 |

All320 call identities, source/config hashes and input identity were independently checked. No foreign GPU process was observed. The guard completed successfully and released the GPU. No video was decoded or exported.

This is a short prefix measurement, not a50-step/video or20-video mean. No new PSNR/SSIM/LPIPS or full50 FLOPs are inferred from it. The completed20-video quality and measured50-step arithmetic evidence remain unchanged. If used as the requested plot latency coordinate, its short-test timing scope must be explicit in the source and caption. Do not subtract a constant logging penalty from other points.
