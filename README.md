# GCA Experiments

Experiment scripts for Grid Compressed Attention (GCA). Run commands from the
repository root. Model weights and conditioning images are external assets.

## T2V: Q90K80 launcher

The launcher uses Wan2.1-14B at 720p with the **4-step StepDistill** configuration.
It supports C4/C6/C8/C10 and a paired dense reference. Use Linux, an H100/H200,
a CUDA-enabled PyTorch installation, the CUDA toolkit, and FlashInfer's CUTLASS
headers. See the [launch guide](docs/q90k80_even_launch.md) for environment details.

```bash
python -m pip install -r requirements.txt

export WAN_MODEL_ROOT=/path/to/wan_models
export CHECKPOINT="$WAN_MODEL_ROOT/Wan2.1-T2V-14B-StepDistill-CfgDistill/distill_model.pt"
export OUT=outputs/q90k80_even
export NUM_VIDEOS=20
export SEED=0

# GCA C8; use CAP=4, 6, 8, or 10.
CAP=8 GPU=0 bash scripts/generate/run_q90k80_even_video.sh

# Dense reference with the same inputs.
CAP=0 GPU=0 bash scripts/generate/run_q90k80_even_video.sh
```

The default prompts are in
[`prompts/MovieGenVideoBench_extended.txt`](prompts/MovieGenVideoBench_extended.txt).
Videos are written to `$OUT/c$CAP/`; timings are written to
`$OUT/runtime_c$CAP.json`. Additional arguments, such as `--prompt_path` and
`--prompt_start`, are forwarded to `generate.py`.

## 50-step experiment scripts

These archived campaigns retain their original server paths, GPU assignments,
and frozen runtime dependencies. Read the linked setup notes before using them
on another machine.

| Experiment | Entry point | Setup |
| --- | --- | --- |
| Q80K90 T2V, C4/C6/C8 | [`generate_family.py`](experiments/q80k90_c4_c6_c8_video20/generate_family.py) (`--root OUTPUT_DIR --gpu 0` or `--gpu 4`) | [T2V notes](experiments/q80k90_c4_c6_c8_video20/README.md) |
| Q80K90 I2V, original 20-input campaign | [`launch.py`](experiments/q80k90_i2v_20260924/campaigns/main20/source/launch.py) (`--gpu 0` or `--gpu 7`) | [I2V notes](experiments/q80k90_i2v_20260924/README.md) |

For new I2V runs, prepare the default prompts and matching image list:

```bash
python analysis/build_i2v_prompt_set.py \
  --out outputs/i2v/inputs/prompts \
  --image-dir /path/to/frozen/images
```

The complete input definitions are in [`prompts/i2v/`](prompts/i2v/). Use the same
prompt, conditioning image, and seed for every method and its dense reference.

## Kernel benchmark

```bash
python analysis/q90k80_even_bench.py --cap 8 --hc 8 --reps 30 --ref
```

See the [launch guide](docs/q90k80_even_launch.md) for the real-QKV benchmark and
additional options.
