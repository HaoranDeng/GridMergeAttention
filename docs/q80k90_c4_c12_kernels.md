# Q80K90 exact C4/C6/C8/C10/C12

The experimentally optimized Hopper family lives in
[`kernels/experimental/q80k90_c4_c12`](../kernels/experimental/q80k90_c4_c12).
Use `recommended.make_q80k90(C)` with that directory on the Python module path.
The capacity-specific configuration retains a previous implementation whenever
the paired H200 measurements do not demonstrate an improvement. Existing
`q80k90_c6_c8` kernels remain available unchanged.

See the family README and gpu_evidence.json for matched old/new/dense timings,
384-case fixed-eight-layer numerical gates, logical full-50-step FLOPs, and
the SVG-EAR/original-Q45K80C8 quality comparisons. The original kernel-tuning
campaign did not generate videos; later native50 video runs have separate
timing records.

## C4 latency audit and input-layout fix, 2026-09-25

The following historical measurements use the same v4 merge/route sources
and configuration, on the same scai7 H200 GPU0
(`GPU-4eca131f-0ba3-759d-bcb5-0418d6b7c68e`). Both completed with valid
exclusive-GPU guards.

| Scope | C4 ms/call | Dense ms/call | Protocol |
|---|---:|---:|---|
| Standalone real-QKV benchmark | 61.391903 | 331.489285 | StepDistill, 3 prompts, 4 steps, fixed 8 layers, 3 alternating repetitions |
| Native pipeline integration gate | 67.286981 | 331.100773 | Base T2V, first native50 step, prompt000, fixed 8 layers, both CFG branches |
| Full native50 generation | 72.138407 | Not paired | Base T2V, all 4,000 calls/video, GPU0 videos000-009 |

The kernel source SHA256 values in all three records match: Python
`de3ee3179a981e186e852f22887e0a8bf50ce059c441139a2536ec76b905eeaa`
and CUDA merge
`0562072d8241c321b4e1627bdaf12062aa7f015b22931fbb6c5b40add7970d12`.
The independently supplied 53.9 ms plot coordinate is not one of these
measurements and cannot be used to diagnose the measured latency difference.

One confirmed source of extra work is layout materialization. The standalone
pipeline supplies token-major Q/K/V. The native Diffusers processor makes
head-major tensors, and its adapter transposes their views back to the
kernel's logical `[B,N,H,D]` shape. The old kernel then calls `contiguous()` on
all three before its fused packing kernel. These copies are inside the core
attention timer. The new packer uses each input's actual strides and reads
directly into exactly the same packed tensors. It changes no attention
arithmetic or compressed representatives. For BF16 `[1,75600,40,128]`, it
removes 2.322432 GB of temporary storage and 4.644864 GB of redundant
read/write traffic per call (decimal GB).

This does not prove that copies explain the entire 10.746504 ms difference.
Native first-step timing is already 5.895079 ms above the standalone mean;
full-generation timing is another 4.851426 ms higher. Their model/checkpoint,
QKV population and execution/measurement context also differ. Standalone
measurements warm and repeat the same QKV, alternating with Dense/previous
implementations and synchronizing each measured call. Generation collects
CUDA events asynchronously across the full pipeline. Even the three native
gate repetitions average 64.621, 63.352 and 73.888 ms. This order-associated
variation is not a causal decomposition. Historical GPU
monitoring did not record clocks or power, preventing attribution to those
factors. The first full video is 72.119 ms versus 72.141 ms for the remaining
nine: one-time compilation does not explain the persistent gap. Projection,
normalization and RoPE costs are measured separately in the approximately
98.885 ms full-module mean, not in the 72.138 ms core mean.

### Validation of the fix

`check_input_layouts.py --full-shape --reference-kernel <original-v4.py>
--output <result.json>` checks 16 shape/layout cases, including independent
strides for Q/K/V, nonzero storage offsets and tail heads. The packed Q, K,
V and native query representation agree exactly with independent PyTorch
oracles; the frozen original packer also passes the full-shape checks. Run
it under `run_exclusive.py` on a verified idle GPU.

An exclusively idle scai5 L40S GPU0 completed the tests on 2026-09-25. The
two orders (old then new, new then old) each use 12 timed calls after warm-up:

| Input storage | Original packing ms | Direct strided packing ms | Temporary allocation removed |
|---|---:|---:|---:|
| Token-major contiguous | 8.609 | 8.626 | 0 |
| Head-major views | 15.527 | 8.445 | 2,322,432,000 bytes |

These are isolated packing measurements, not full attention or video timings.
The unchanged merge/route receives exactly the same packed inputs. Raw
samples, source hashes, GPU identity and the successful exclusive guard are
in `kernels/experimental/q80k90_c4_c12/input_layout_validation_20260925.json`.

The initial audit had no exclusively idle H200; that initial L40S result did
not replace any historical H200 measurement. The authorized H200 follow-up
below is now complete.

### Completed H200 follow-up

On 2026-09-25, the user authorized stopping one Q80K90 job and then explicitly
cancelled its restoration. The scai7 GPU7 Q80K90C8 I2V job was stopped with all
ten committed videos and metrics preserved; its heartbeat was paused. The
test used that H200 (`GPU-7aa670f7-95a6-1e16-c28f-80664a449ad8`) after two clean
preflights. All 603 monitor samples were uncontended; the test exited normally
and released the GPU. No cancelled job was restored.

The old baseline is the original v4 immediately before `eb52068`, verified
against Git ignoring line endings. The new implementation is `eb52068`.
Configuration and merge/selection/routing arithmetic were identical.

| Scope | Old ms | Fixed ms | Reduction |
|---|---:|---:|---:|
| Native first-step input packing | 4.170104 | 1.312200 | 2.857904 ms |
| Native first-step complete attention | 65.086850 | 62.988589 | 2.098261 ms |
| Full50 mean complete attention/call | 71.193183 | 68.540536 | 2.652647 ms (3.73%) |
| Full50 mean self-attention module/call | 97.613703 | 94.978466 | 2.635237 ms (2.70%) |

Full-pipeline time through VAE decoding, excluding export, was
608.271085 versus 598.595445 seconds: 9.675639 seconds (1.59%) less. Both runs
performed exactly 4000 self-attention calls across all40 layers and50 steps.
Both decoded outputs have SHA256
`75beb9df61abdbe26c6420907e9d5c27766d850beaa604af905d6b434e54c811`.
All16 first-step cases (layers0,5,10,15,20,25,30,35, both CFG branches) also
matched bitwise. The first-step gate alternated old/new order for eight
repetitions per case; its paired Dense mean was326.963960ms.

The redundant copies are removed and measured attention latency improves,
but they do not explain the entire standalone-versus-generation difference.
The fixed full50 mean is still5.551947ms (8.81%) above its own repeated
first-step gate. Historical GPU0's61.391903ms uses a different QKV population
and is not a same-card control for this GPU7 run. This experiment does not
causally identify the remaining difference.

Limitations: one canonical T2V prompt000, seed0, one full50 run per method,
old then new; no reversed-order full-video replication. SM-clock phase means
were1666.4/1644.5MHz and power means690.18/690.01W; clocks were not fixed.
These observations do not establish frequency as the cause of the residual.
Peak allocated memory for the complete pipeline was essentially unchanged
(88.369516/88.369209GiB), despite eliminating the per-call temporary copies.
Logical full50 major-matmul cost remains56.716886016PFLOPs/video for both.
Matched full50 SVG-EAR and original Q45K80C8 timing/quality were not measured
in this audit; no speedup or quality ranking against them is inferred.

Evidence: `kernels/experimental/q80k90_c4_c12/h200_input_layout_validation_20260925.json`.
The reproducible driver is `kernels/experimental/q80k90_c4_c12/benchmark_native50_layout.py`;
run it with `--source` pointing to a frozen source bundle containing old
`reference_q80k90_kernel.py`, new `q80k90_kernel.py`, unchanged CUDA source and
validated configuration, under the exclusive GPU guard, with `--full50`.
Raw source snapshots, telemetry and monitor logs are archived separately
with SHA2561470c26d686230414466c6a13878d4852179dd6c9cf067c58395b33ca146624a.
