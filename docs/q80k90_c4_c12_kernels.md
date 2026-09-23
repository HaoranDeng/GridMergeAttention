# Q80K90 exact C4/C6/C8/C10/C12

The experimentally optimized Hopper family lives in
[`kernels/experimental/q80k90_c4_c12`](../kernels/experimental/q80k90_c4_c12).
Use `recommended.make_q80k90(C)` with that directory on the Python module path.
The capacity-specific configuration retains a previous implementation whenever
the paired H200 measurements do not demonstrate an improvement. Existing
`q80k90_c6_c8` kernels remain available unchanged.

See the family README and gpu_evidence.json for matched old/new/dense timings,
384-case fixed-eight-layer numerical gates, logical full-50-step FLOPs, and
the SVG-EAR/original-Q45K80C8 quality comparisons. No videos were generated.
