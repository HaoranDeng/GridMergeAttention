# SVG-EAR p=.4 timing diagnosis (2026-09-26)

This campaign completed five fixed I2V inputs on each of two exclusive scai6 H200s. The user canceled p=.5; its partial/queued cases are excluded. The original directory retains `p04_p05` only for provenance. Initially this was a timing-only task; the latest user follow-up then requested disabling density recording, testing briefly, and putting the corrected timing on the figure.

`RESULTS.md` and `INDEPENDENT_VERIFICATION.json` document the independently checked ten executions /40,000 calls. Steady core times (indices001-004, index000 warmup) were110.502409 ms on GPU2 and110.297450 ms on GPU4, with synchronous density-file logging included.

The subsequent user requested a short diagnosis within30 minutes. `quick_diagnostic.py` executes only the first three steps of the frozen50-step p=.4 schedule, profiles240 real calls, and replays32 QKV/centroid states with four logging/audit modes. It does not decode/export a video or run p=.5. `quick_diagnostic/RESULTS.md` and `VERIFICATION.json` show98.858245 ms with the original log destination,80.621707 ms with density computed but writes sent to `/dev/null`, and79.737065 ms with density recording disabled. All128 replay outputs match exactly. The mean paired file-destination overhead is18.236539 ms (median14.351055 ms); these short processor-core means exclude native layout conversions and are not full-video estimates.

The earlier T2V p=.5 source disabled per-call density-file logging and collected density after the core end event. Directly comparing its operator interval to this p=.4 interval was not a consistent timing protocol. Historical native I2V also logged density but used another NFS destination; its exact historical logging overhead was not measured here. No intrinsic reversal of top_p versus operator speed has been established.

## Files and execution scope

- `source/worker.py`, `common.py`, `gpu_checks.py`: executed campaign worker/audit support.
- `source/run_p04_only.py`: one-time historical adoption of GPU2 p=.4 and cancellation of the old p=.5 process; never rerun on completed jobs.
- `source/run.py`: telemetry utility only in this publication. The canceled crossover entry point is disabled. Its exact historical source is retained as non-executable `provenance/run_crossover.py.txt`, with the original source hash in the experiment artifacts.
- `sync.py`: synchronize structured records and hash-verified committed videos; does not launch workers or inspect media.
- `verify_results.py`: independently audit the completed50-step traces and inputs from the local result bundle.
- `quick_diagnostic.py`: bounded20-minute p=.4 diagnostic, clean-GPU preflight and process-session monitoring, restoring per-call centroids for matched replays. Hardcoded historical scai6 GPU2 and immutable campaign paths; not a general job launcher.
- `sync_quick.py`, `verify_quick.py`: structured diagnostic transfer and independent verification.

Raw videos/full traces remain under `C:/GMA/QKC scan/SVG_EAR_p04_p05_scai6_timing5_20260926` and `/local2/denghaoran/GridMergeAttention/experiments/svg_ear_p04_p05_scai6_timing5_20260926`. The native/SVG sources were hashed against the frozen prior p=.4 campaign; manifests, launch receipts and exclusivity monitors are retained. Five unique sparkler_v1 inputs000-004, seed=index, were repeated across cards; cards were never pooled.

The next full-trace timing protocol should retain density and iteration evidence in memory and serialize after timed attention, preferably at the video boundary. Frozen implementations and raw measurements remain unchanged; no constant is subtracted from historical runtimes.

## Requested clean short test and plot coordinate

`quick_no_density.py` subsequently disabled density logging and the extra iteration audit, then ran four steps of the original native50-step schedule on exclusive scai6 GPU2. The original core/full event recorder was used without diagnostic replay/stage instrumentation. Excluding the first initialization step, steps2-4 contain240 calls: core88.493686167 ms and full111.438971011 ms. Per-step core means were87.265115,88.821791 and89.394153 ms. This is a one-input short test, not a50-step/video or20-video mean.

`update_plot_latency.py` applies only that user-requested I2V p=.4 latency override and adds a caption disclosure. Quality and measured full50 arithmetic remain the existing20-video values. T2V, all other curve points, GCA user runtime coordinates, paper tables and body text are unchanged. `PLOT_LATENCY_VERIFICATION.json` checks these invariants. The source preserves106.275035326 ms as the prior logged campaign measurement.

Validation: independent ten-video hash/input/trace/FLOPs audit, independent three-step matched-QKV audit, and Python syntax compilation. No media was shown to a model. The retired p=.5 deploy/crossover launchers are deliberately not published as executable launchers.
