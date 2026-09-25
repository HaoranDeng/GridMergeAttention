# Fisher-geometric C10, four-step generation

`fisher_geometric_c10_rf0` extends the geometric Fisher C8 operator to ten
distinct representatives: the matched real anchor and nine Gram-Schmidt
pivots. It uses Uniform-16 probes, C32/g45 candidates, RF0, BF16 merge scores,
the existing query-column singleton, and the same full compressed route in
every layer and all four denoising steps. No fitting or calibration is added.

The initial implementation uses sixteen physical merge slots. Six trailing
copies of the last seed lose leftmost ownership ties and produce empty cells
with zero vectors and positive-infinite bias. Only ten representatives own
keys. Timing includes the extra storage and route work and must not be
presented as an optimized native-C10 kernel result. A head chunk of four
limits peak memory on one H200.

The J4 revision reduces the merge column block from 16 to 4 and the merge
loop stages from 2 to 1, retaining four warps. On captured real layer0/step0
QKV, complete forty-head attention including selection and padding improved
from 2665.580 ms to 524.331 ms (5.084x). Full output matched the old version
exactly in this check. Matched Flash SDPA measured 326.384 ms, so C10 is still
slower than that dense baseline (0.622x speedup vs SDPA). These are three
steady CUDA-event samples on one exclusive H200, not full-video averages.
The bounded comparison is `analysis/benchmark_fisher_c10_tiles.py`; primary
results are in `experiments/gma_c10_tile_tuning_20260909/results.json` on scai7.

Before model loading, native validation checks 90 selected indices against
an independent Fisher/GS calculation, exact C4 prefix parity, distinct seeds,
full compressed attention against a separate ten-representative reference,
and tied-query/empty-cell behavior. The validation run on scai7 H200 produced
maximum full-route error 6.7631e-5 and empty-cell error 5.1866e-5.

Run with the canonical Wan2.1-T2V-14B-StepDistill-CfgDistill checkpoint:

```sh
WAN_MODEL_ROOT=/xuanwu-tank/east/zrl/MonarchRT/wan_models \
python analysis/launch_quality_first_videos.py \
  --gpu 1 --gpu-scope task --method fisher_geometric_c10_rf0 \
  --run-dir /local2/denghaoran/GridMergeAttention/experiments/NEW_RUN \
  --checkpoint /xuanwu-tank/east/zrl/MonarchRT/wan_models/Wan2.1-T2V-14B-StepDistill-CfgDistill/distill_model.pt
```

Task scope owns exactly one clean GPU and continuously checks it for foreign
processes, while allowing unrelated account jobs on other GPUs. The default
account scope retains the existing account-wide limits. Neither scope queues
or waits for a busy GPU.

The wrapper freezes source and prompt provenance, generates 20 canonical
720p/81-frame videos with seed 0, and audits 3,200 attention calls across all
40 layers. Each complete call has ten four-head merge/route dispatches.
Runtime includes the audit overhead. Quality results and any later speedup
must be measured separately; native validation alone establishes neither.
