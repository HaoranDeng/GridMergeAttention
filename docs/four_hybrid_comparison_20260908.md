# Four-method hybrid comparison, 2026-09-08

The user explicitly requested a fresh SVG, SVG2, SVG-EAR and standard GMA-C4
comparison using the released hybrid architecture, with two GPUs. This run
overrides the usual all-layer GMA and one-device defaults for this comparison.

All four methods use Wan2.1-T2V-14B-Diffusers, 1280x720, 81 frames, 50 UniPC
steps, flow shift 5, CFG 5, BF16 QKV and seed 0. Each generates the first 20
canonical MovieGenVideoBench_extended prompts, with fresh outputs and a verified
prompt manifest. The three upstream scripts are pinned at
`f89aedaf169ac2ae5b186bda674e53c3dc08c476` (verified current upstream main).

All calls in steps 0 through 9 use dense attention. In steps 10 through 49,
layer 0 stays dense and layers 1 through 39 use the selected method. There are
880 dense calls and 3,120 replacement calls per video, including both CFG passes.
The launcher validates these actual counts and complete per-layer coverage.

| Method | Replacement settings |
| --- | --- |
| SVG | pattern parameter 0.3; 64 online profiling rows |
| SVG2/SAP | Q/K clusters 300/1000; top-p 0.9; min ratio 0.10; K-means 50/2 |
| SVG-EAR | same released CLI; upstream EAR implementation hardcodes min ratio 0.05 |
| Standard GMA-C4 | existing hard-C4 operator; four fixed seed queries; entropy bias; full compressed attention; head chunk 8 |

The GMA selector and merge operator are unchanged. This is the standard Hard C4
comparison, distinct from the later Anchor-Fisher RF0 C4+SQ1 exploration.

Two independent lanes use at most two H200s: EAR then SVG2; GMA-C4 then SVG.
Each method is followed by PSNR/SSIM/Alex-LPIPS on all 1,620 frames and the six
established custom-input VBench dimensions. These six dimensions are reported
individually, not as the official full VBench total. Every GPU stage checks
cleanliness immediately before launch and monitors foreign compute processes.
If a device is busy, its lane stops without waiting for the device.

The exact-prompt historical 2026-08-16 50-step dense videos are reused for quality.
Their earlier preprocessing equivalence to the current fast path is unproven,
so this limitation is recorded. Current attention/module/end-to-end timing is
reported independently; no matched current SDPA speedup is fabricated.
SVG2/EAR record population-weighted retained area on replacement calls and
effective density across all calls, with dense calls counted as density 1.

External baselines route to `C:\GMA\experiments.xlsx`. The new GMA hybrid
ablation routes to `C:\GMA\tmp.xlsx`. Local primary artifacts belong under
`C:\GMA\experiments\wan14b_720p_50step_four_hybrid_20260908`.
