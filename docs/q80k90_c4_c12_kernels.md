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

Both scai6/scai7 had no exclusively idle H200 at the audit's final check. A
paired old/new H200 replay and full native50 measurement remain pending; no
new video generation was launched, no running experiment was changed, and
none of the historical H200 numbers above is replaced by the L40S result.
Record SM/memory clocks and power during that follow-up to distinguish
layout savings from execution-context effects.
