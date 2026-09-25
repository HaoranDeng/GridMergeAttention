# Q90K80 C4/C6/C8/C10 GCA kernels: launch guide

Native CUDA kernels for Grid Compressed Attention on Wan2.1-14B 720p
(21 x 45 x 80 = 75,600 tokens, 40 heads, D = 128): query groups of 90 queries,
key blocks of 80 keys, exactly C merged K/V representatives per (query group,
key block). No singleton or residual route. Kernel design, measurements and
closed experiments are in [q90k80_kernels.md](q90k80_kernels.md).

## Files

| File | Role |
|---|---|
| `analysis/q90k80_even_kernel.py` | `Q90K80Even(cap)` operator and the JIT builders |
| `analysis/q90k80_even_select.cu` | Fisher seed selector (FP32, frame loop) |
| `analysis/q90k80_even_merge.cu` | K/V merge (WGMMA, fused ownership argmax, TMA stores) |
| `analysis/q90k80_even_route.cu` | Attention over the merged K/V (transposed WGMMA, lazy-rescaling softmax) |
| `analysis/q90k80_even_layout.cu` | Fused input/output layout kernels |
| `analysis/q90k80_even_generate.py` | Runs `generate.py` with GCA as every self-attention call |
| `scripts/generate/run_q90k80_even_video.sh` | Launch script for video generation |
| `analysis/video_pair_psnr.py` | PSNR between paired video directories |
| `analysis/q90k80_even_bench.py` | Synthetic stage timings and Triton-reference check |
| `analysis/q90k80_even_real40.py` | 96-case real-QKV timing protocol |
| `analysis/q90k80_even_route_check.py` | Route accuracy against an FP64 softmax |

## Requirements

- NVIDIA H200 / H100 (sm_90a). The kernels are compiled at first use with
  `torch.utils.cpp_extension` (about 1-2 minutes per capacity) and cached in
  `$TORCH_EXTENSIONS_DIR`.
- CUTLASS headers from the installed `flashinfer` package
  (`<site-packages>/flashinfer/data/cutlass/include`), `-lcuda`.
- `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` (the SentencePiece /
  native-extension conflict; the launch script sets it).
- Wan2.1-T2V-14B StepDistill-CfgDistill checkpoint for the 4-step config.

If a build is interrupted, `torch` leaves a `lock` file in the extension
directory and the next run waits on it forever (log stuck at the build step,
GPU idle). Delete that extension's directory and rerun.

## Video generation

```bash
CAP=8 GPU=0 OUT=/path/to/out scripts/generate/run_q90k80_even_video.sh
```

The paired dense reference, with the same prompts and seed:

```bash
CAP=0 GPU=0 OUT=/path/to/out scripts/generate/run_q90k80_even_video.sh
```

PSNR of the GCA videos against dense:

```bash
python analysis/video_pair_psnr.py /path/to/out/c0 /path/to/out/c8
```

Environment variables of the launch script (defaults in brackets): `CAP`
(8; 4, 6, 8, 10, or 0 for dense), `GPU` (0), `HC` (8 heads per native call),
`CONFIG` (`configs/dense_wan_14b_720p_fewstep.yaml`, 4 steps), `CHECKPOINT`
(`~/wan_models/Wan2.1-T2V-14B-StepDistill-CfgDistill/distill_model.pt`),
`NUM_VIDEOS` (3), `SEED` (0), `OUT` (`videos/q90k80_even`), `PYTHON`
(`python`), `ROUTE_FLAGS` (empty; `-DROUTE_LAZY=0` selects the previous route
softmax). Any further arguments go to `generate.py` (for example
`--prompt_path`, `--prompt_start`). Outputs: `$OUT/c$CAP/*.mp4` and
`$OUT/runtime_c$CAP.json` (per-video mean attention time per call, the whole
self-attention module time, inference seconds; the summary's `steady_state_*`
fields drop the first video, which includes kernel loading and warm-up).

The wrapper replaces only self-attention calls of shape [1, 75600, 40, 128]
without masks or windows; cross-attention stays dense. It was run with the
4-step config only; the 50-step config needs its own checkpoint and was not
run with this wrapper.

## Operator use

```python
from analysis.q90k80_even_kernel import Q90K80Even
attn = Q90K80Even(8, merge_backend='cuda', head_chunk=8)   # C8
out = attn(q, k, v)   # q, k, v: [1, 75600, 40, 128] bf16, Wan token order; out has the same shape
```

Keep one instance per capacity: it caches its workspaces (merged K/V, bias,
layout buffers) across calls.

## Benchmarks

```bash
# Synthetic stage timings + check against the Triton reference path
python analysis/q90k80_even_bench.py --cap 8 --hc 8 --reps 30 --ref

# Route accuracy of two builds against FP64 on the same merged K/V
python analysis/q90k80_even_route_check.py --cap 8 --flags=-DROUTE_LAZY=0

# Real-QKV protocol (3 prompts x 4 steps x 8 layers, dense Flash SDPA paired per case)
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python analysis/q90k80_even_real40.py \
  --repo . --kernels . --root /path/to/out --capacities 4 6 8 10
```

## Measured (2026-09-25, scai7 GPU 7, H200, commit with this guide)

4-step video generation, 3 prompts (MovieGenVideoBench_extended 0-2), seed 0,
160 self-attention calls per video, steady state (videos 2 and 3):

| Method | Attention (ms/call) | Self-attention module (ms/call) | Inference (s/video) | Attention speedup | Inference speedup | PSNR vs dense (dB) |
|---|---:|---:|---:|---:|---:|---:|
| Dense SDPA (Flash) | 325.6 | 348.1 | 69.7 | 1.00x | 1.00x | - |
| GCA C8 | 96.5 | 119.4 | 33.5 | 3.37x | 2.08x | 10.19 (11.02 / 10.74 / 8.81) |
| GCA C8, previous route softmax | 103.0 | 125.8 | 34.4 | 3.16x | 2.03x | 10.17 (10.89 / 10.77 / 8.84) |

The lazy-rescaling route leaves video quality unchanged (the two C8 builds
agree with each other at 18.4 dB, a divergence level typical of re-running the
same model). The operator-only real-QKV protocol gives C8 93.3 ms per call, about 3 ms below the
in-pipeline 96.5 ms, which also has the neighbouring projections, FFNs and
cross-attention on the GPU.

Real-QKV protocol, all capacities (operator only, paired dense ~324 ms):

| Capacity | Attention (ms/call) | Speedup vs SDPA |
|---|---:|---:|
| C4 | 48.3 | 6.69x |
| C6 | 75.0 | 4.33x |
| C8 | 93.3 | 3.47x |
| C10 | 130.2 | 2.48x |

Videos and runtime JSONs of the table above:
`scai7:/local2/denghaoran/GridMergeAttention/q90k80_even_video4_20260925/`.
