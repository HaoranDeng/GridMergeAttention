# Q80K90 exact C8 / C6: optimization revision 2

This experimental implementation passed GPU numerical validation. Measured
real-40-head complete calls average **139.518 ms for C8** and **127.274 ms for
C6** on exclusive H200 GPUs. Neither requested target (**100 ms for C8**,
**120 ms for C6**) was reached. The earlier 164/155 ms smoke timings belong
to v1 and must not be attributed to this code.

## Measured GPU validation (2026-09-23 UTC)

| Capacity | H200 GPU | Complete call | Same-GPU dense | Speedup vs dense | Target met |
|---|---:|---:|---:|---:|---|
| C8 | scai7 GPU0 | 139.518 ms | 333.598 ms | 2.391x | No |
| C6 | scai7 GPU4 | 127.274 ms | 323.245 ms | 2.540x | No |

Each result covers 3 prompts x 4 frozen StepDistill timesteps x fixed layers
0,5,10,15,20,25,30,35 x 40 independent heads: 96 cases, with three alternating
timing repetitions per case. Shape is BF16 `[1,75600,40,128]`. This measures
complete attention calls on a four-step trajectory, not 50-step generation
latency. All 96 cases were faster than their same-GPU dense measurements.
Clean preflight and 492/485 monitoring samples found no foreign compute PIDs.
The borrowed GPUs were returned to their original jobs after testing.

Fourteen configurations were tested per capacity. Both selected the settings
in `validated_h200_config.json`: FP32 selection, four selector groups, local
fast math, fused layouts, shared output staging, head_chunk=4, seed_tile=8,
native_tiles=63, route=128x64, min_blocks=2. Constructors keep their conservative
defaults; pass this config explicitly to reproduce the measured variant.

Each selected variant passed 384 frozen numerical cases across all eight
layers. C8/C6 mean relative L2 to the corresponding mathematical operator was
0.00170008/0.00169052; maxima were 0.00457633/0.00199903. Mean seed-change
fractions were 0.000006330/0.000003445. RF0 reuse, exact capacity, forced
intersections, empty clusters, partial head chunks, bitwise layout packing,
and independent Triton arithmetic checks passed. Engineering gates are
recorded in `gpu_evidence.json`; they are not user-defined universal tolerances.

| Method | Mean relative L2 to dense | Matched real40 timing | Video quality |
|---|---:|---|---|
| SVG-EAR top_p=0.85, historical fixed-eight-layer scan | 0.06526199 | Pending | Pending for this protocol |
| Original Q45K80C8, historical fixed-eight-layer scan | 0.08862509 | Pending | Pending for this protocol |
| Q80K90C8 v2 | 0.10388767 | 139.518 ms | Pending; no videos generated |
| Q80K90C6 v2 | 0.12410760 | 127.274 ms | Pending; no videos generated |

The baseline numerical rows use the same frozen eight-layer, 384-case scan.
Their different hardware does not support timing speedups: original C8 used
L40S, SVG-EAR used H200. C8/C6 mean error is 59.19%/90.17% above SVG-EAR and
17.22%/40.04% above original C8. These implementations therefore do not qualify
for the shortlist requiring error no worse than original C8. SVG-EAR p=0.85
numerical results must not be confused with p=0.8749 video results.

Calculated logical major-matmul work for the fixed **50-step, 4,000-call**
schedule is **112.641048576 PFLOPs (C8)** and **84.678967296 PFLOPs (C6)**.
This exact fixed-schedule count excludes softmax, reductions, layout work and
tensor-core padding; it is not a measured generated video. Per-call core work
is `6*40*128*C*945*840*90 + 4*40*128*75600*C*840`, with selector work
`2*40*128*16*840*45*32`, then 4,000 calls. Comparable 50-step baseline counts
remain pending. Historical original-C8/SVG-EAR four-step average figures of
43.47592704/48.75106610096 TFLOPs retain their old scope; the variable-density
SVG-EAR figure is not extrapolated to 50 steps.

## Changes

- Shared capacity-parameterized CUDA/Triton source for exact C6 and C8.
- Consume ownership immediately during merge weighting, removing the large
  live label array. The conservative C8 binary uses 157 registers/thread,
  compared with v1's 228; C6 uses 165, or 161 in the three-block fast variant.
- Optional direct BF16-pair output stores remove shared output staging and
  four CTA barriers per tile. The four-block build uses 128 registers/thread
  and 49,152 dynamic shared-memory bytes for both capacities, with zero stack
  and local-memory bytes reported by the compiler. The tested direct-store
  variants were slower in GPU tuning and were not selected.
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
  the separate full GPU numerical gate is reported above.

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
SVG-EAR/original-Q45K80C8 timing comparisons remain pending. Full-video FLOPs
use the fixed 50-step/4,000-call schedule above; this four-step timing protocol
does not establish generation latency.

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
