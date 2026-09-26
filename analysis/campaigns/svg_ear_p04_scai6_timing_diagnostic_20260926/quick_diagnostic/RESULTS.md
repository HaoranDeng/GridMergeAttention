# p=0.4 three-step timing diagnosis

The measured p=.4 attention interval contains synchronous density-file I/O. A same-GPU, same-QKV replay establishes that this adds substantial latency without changing outputs. The previously reported110.50/110.30 ms values are audited end-to-end core intervals including this work; they are not clean operator-only estimates.

| Replay condition | Mean core ms | Mean wall ms |
|---|---:|---:|
| original_log | 98.858245 | 98.930804 |
| log_to_dev_null | 80.621707 | 80.692980 |
| no_density_log | 79.737065 | 79.826320 |
| no_log_no_audit | 80.083152 | 80.172884 |

Original file versus /dev/null difference: mean 18.236539 ms, median 14.351055 ms, range 8.472832-120.754456 ms; all 32/32 paired differences are positive. /dev/null retains density computation, GPU scalar synchronization and JSON formatting; it removes the original filesystem write destination. Disabling the Python iteration audit does not explain the large gap.

Protocol: one frozen sparkler_v1 input000, seed0, p=.4 only, scai6 GPU2 H200. Execute the first three steps of the original50-step schedule and stop before step4, without decoding or exporting a video. Profile240 real attention calls. The32 ablation states span steps1 and2 (zero-based), both CFG branches and layers0,5,10,15,20,25,30,35. Restore identical pre-call Q/K centroids for every replay; reverse replay order across sampled states. All128 replay outputs match the trajectory output exactly (max absolute difference0). First-step initialization is excluded from the ablation means. No foreign GPU process was observed.

The measured processor core excludes the native wrapper layout conversions. It includes diagnostic event instrumentation. These are short, early-step, sampled-layer diagnostics, not50-step or20-video averages, and no multiplication/extrapolation is used to replace reported FLOPs or paper points.

The T2V p=.5 script underlying the89.48 ms reference explicitly disables SVG_DENSITY_LOG and performs density accounting after the core end event. Its archived source hash was checked against the published provenance. Thus that timing scope differs from this p=.4 campaign. No p=.5 experiment was run during this diagnostic.

Current p=.4 logs reside on /local2, an NFS mount. Historical native I2V runner116 also enabled density logging but used h200_run/density_p50i on a different NFS mount, /xuanwu-tank/east. We did not replay the canceled p=.5 case or prove the exact historical per-write latency, so the full numerical difference to historical I2V p=.5 is not causally apportioned.

The corrective measurement protocol is to capture density/iteration evidence in memory and serialize it after the attention timing end (preferably at the video boundary), while retaining the full trace. That requires separately measured corrected timings; subtracting18.24 ms from all historical points would be unjustified. Frozen raw runs, quality values, paper curves and20-video means are preserved.

Evidence: diagnostic.json, VERIFICATION.json, completion.json, launch.json, preflight.json, monitor.jsonl and sync_hashes.json in this directory. The three-step guard has exited successfully and released its GPU.
