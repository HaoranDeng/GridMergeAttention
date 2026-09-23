# Q80K90 exact C8 / C6: optimization revision 2

This is an experimental implementation awaiting GPU numerical and latency
validation. The requested complete-attention targets are **100 ms for C8** and
**120 ms for C6**, at BF16 shape `[1,75600,40,128]`. Neither target is currently
verified for this revision. The earlier 164/155 ms smoke timings belong to v1
and must not be attributed to this code.

## Changes

- Shared capacity-parameterized CUDA/Triton source for exact C6 and C8.
- Consume ownership immediately during merge weighting, removing the large
  live label array. The conservative C8 binary uses 157 registers/thread,
  compared with v1's 228; C6 uses 165, or 161 in the three-block fast variant.
- Optional direct BF16-pair output stores remove shared output staging and
  four CTA barriers per tile. The four-block build uses 128 registers/thread
  and 49,152 dynamic shared-memory bytes for both capacities, with zero stack
  and local-memory bytes reported by the compiler. Its runtime benefit is
  unmeasured; it may trade barriers for less efficient global stores.
- Fuse Q/K/V input permutations and lossless WGMMA Q packing into one Triton
  launch. Route directly into `[N,H,D]` output, removing its final transpose.
- Add an optional FP32 selector and local approximate log/reciprocal switches.
  FP64 selection remains available. Candidate pool, Uniform-16 probes, two
  forced intersections, RF0 index reuse, exact capacity, entropy correction,
  empty-cluster semantics, and full compressed attention remain the operator
  contract. No extra representative or dense exception is added.
- Cache reusable input and merge workspaces. Use a separate instance for each
  concurrent stream/call; returned outputs have independent storage. A
  `return_debug=True` seed tensor is workspace-backed and must be cloned if it
  needs to survive another call.

## Completed checks without GPUs

All compilation processes ran with `CUDA_VISIBLE_DEVICES` empty and asserted
that PyTorch had not initialized a CUDA context.

- Python syntax checks and CPU address/ownership checks passed. These cover
  the lossless input/Q-pack permutations, direct output ordering, both seed
  tile layouts, all 45 query groups, exact C6/C8 output coverage, ties, and
  adversarial inactive C6 slots.
- Ten native sm_90a variants compiled successfully: five configurations per
  capacity. The compiler reported zero stack/local-memory use for each.
- Twenty-one Triton variants compiled offline for sm_90, including layout
  fusion, FP64/FP32 selection, routing, and the independent merge reference.
- The C8 selector's offline resource report falls from 255 registers and an
  80-byte stack in FP64/G4 to 149 registers and a zero-byte stack in FP32/G4.
- A CPU precision probe used prompt 0, frozen timestep 0, all eight fixed
  layers, heads 0/10/20/30, three query groups, and four key groups. FP32 and
  FP64 selected identical indices in all **384 sampled cases per capacity**.
  CPU reduction and transcendental behavior do not establish GPU equality;
  the full numerical gate is still mandatory.

These checks establish compilability and address algebra, not CUDA runtime
correctness or measured speed. They are not a method-quality ranking. No
videos, video frames, or quality metrics were inspected or generated.

## GPU validation entry point

After two clean GPU indices are explicitly supplied, launch from this folder:

```sh
python launch_pair.py --c8-gpu C8_GPU --c6-gpu C6_GPU --root NEW_RUN_DIRECTORY
```

The launcher checks both GPUs, and each worker independently repeats its
preflight. It never waits for a GPU. Foreign compute PIDs invalidate and stop
the affected worker. The default sequence stays in one tracked CUDA process:

1. Check RF0 reuse, unique exact capacity, forced anchors, partial head chunks,
   bitwise layout/packing equivalence, empty-cluster KV/bias, constant values,
   the independent Triton merge, and every native store/math variant.
2. Tune on a real four-head capture repeated to 40 heads. Record complete-call
   and same-GPU dense timings in alternating order; keep numerical failures
   out of selection. These timings remain preliminary shape measurements.
3. Validate the selected configuration against the frozen mathematical
   operator: 3 prompts x 4 timesteps x layers 0,5,10,15,20,25,30,35 x 4 heads,
   giving 384 numerical cases across 96 captures. This does not alter the
   frozen scan protocol. Explicit engineering review thresholds are saved in
   the gate record; they are not a universal user-specified tolerance.
4. Run the separate real-40-head timing protocol on all eight layers and
   three prompts, with matched dense timing on the same GPU. Report target
   attainment only from that result. This is a four-timestep frozen
   StepDistill trajectory benchmark, not 50-step generation latency.

No videos are launched by this pipeline. PSNR, SSIM, LPIPS, VBench and matched
SVG-EAR/original-Q45K80C8 comparisons remain pending. If full-video FLOPs are
reported later, use the actual 50-step/4,000-call schedule; do not turn this
four-step timing protocol into a generation result.

`--phase invariants` and `--phase tune` provide shorter explicit runs. The
default `full` chain stops on failed numerical validation. A missed speed
target is recorded as `target_met: false`, with the measurements retained.

## CPU-only reproduction

```sh
python check_cpu.py
CUDA_VISIBLE_DEVICES= python compile_cpu.py --output compile_checks
CUDA_VISIBLE_DEVICES= python compile_triton_cpu.py --output triton_compile_checks
CUDA_VISIBLE_DEVICES= python probe_selector_cpu.py --output selector_cpu_probe.json
```

Use the existing `monarch_rt` environment and CUDA toolkit. The frozen QKV,
source model, and checkpoint defaults refer to the GMA lab paths; the GPU
runner exposes `--data`, `--repo`, and `--checkpoint` for explicit overrides.
