# Q80K90 I2V experiments and default inputs

Future I2V20 runs use `vbench_i2v_20_sparkler_v1`, adopted by the user on
2026-09-24. Samples000-018 retain the previous stratified VBench inputs;
sample019 is **a person holding a sparkler in their hand**, with its matching
conditioning image and seed19. Every method and its dense reference must use
the same versioned population. Do not silently reuse old019 dense video or quality.

The adoption followed inspection of the original results. The replacement
comparison is a **post-hoc sensitivity analysis**; becoming the future default
does not change that historical status. The old20 manifest is retained under
`prompts/i2v/vbench_i2v_20_legacy.*`.

## Prepare future inputs

From the repository root:

```sh
python analysis/build_i2v_prompt_set.py --out /new/run/inputs/prompts --image-dir /path/to/frozen/images
```

This writes paired `.txt`, `.images.txt` and `.json` files. The checked-in aliases
`prompts/vbench_i2v_20.*` contain the same default. The versioned manifest includes
each prompt/image SHA256, official metadata indices, input source revision,
selection history and seed policy. Verify all conditioning bytes before a run;
do not replace a missing image with a different resize or recompression. No
images, videos, model weights, environment dumps or private restoration records
are committed here. Model and conditioning assets remain external dependencies.

The legacy selector is available only explicitly:

```sh
python analysis/build_i2v_prompt_set.py --selection legacy-stratified --out /new/run/legacy
```

Other legacy sample counts additionally need `--n` and the official `--full-info`
metadata. Do not change the frozen inputs of an already running or resumed job.

## Published experiment code

- `campaigns/main20/source`: original C4/C6/C8/C10 generation, same-GPU native
  gates, sharding, CPU metrics, VBench evaluation, public synchronization and
  supervised task restoration.
- `campaigns/replace019_trial/source`: the Dense/C4/C8/EARp.8 one-sample trial.
- `campaigns/replace019_allpoints/source`: all18 settings, atomic method claims,
  independent GPU guards, paired metrics, full50-step arithmetic and plotting.
- `plotting`: the shared T2V/I2V renderer and user-selected plotting latencies.

The existing GCA kernel code remains in its original repository locations; it is
not duplicated here. These are the experiment scripts actually used on the
servers, including their original site-specific plans and paths. The plans
record the frozen native runtime and hashes used by each experiment. Keep using
those existing runtime snapshots for historical reproduction; this directory
is not a replacement for the repository's generation pipeline.

The archived borrowing/restoration helpers address specific saved jobs. Do not
run them against unrelated jobs. The archived main20 plan retains the old019
for reproducing that historical run; future experiments use the new versioned
default described above.

## Protocol and accounting

Wan2.1-I2V-14B-720P; 720x1280;81 frames;50 FlowUniPC steps; shift5; CFG5;
unchanged negative prompt; raw English prompt text; base seed0 plus prompt index.
GCA uses Q80/K90 and exact C4/C6/C8/C10 in all40 layers and both CFG branches,
4000 attention calls per video. Require each exclusive GPU's native numerical
and dense speed gate before generation. No kernel changes were made for the
019 replacement.

CPU PSNR/SSIM/AlexLPIPS use every81 frame with matching Dense input identities.
For replacement20, combine original000-018 with the measured new019; never
invent an aggregate while an original20 point is incomplete. EAR/SVG2 record
all4000 densities and returned clustering iterations, retaining exact full50
FLOPs and the scheduled upper bound used by the historical comparison.

Keep original per-GPU steady timings separate. The user-selected I2V plotting
latencies are C6=83.6ms, C8=104.1ms, C10=133.8ms; they are **plot coordinates**,
not replacement cold-start measurements. Plots display TFLOPs/call derived from
the full50-step total. VBench is the existing six-dimension custom-input protocol
with longer preprocessing; the weighted normalized composite is not a
leaderboard score. Original Q45K80C8 matched I2V results remain pending.

Only complete revised20 points enter the current replacement curve, with
unfinished settings omitted. Preserve original figures, T2V data, raw timings
and post-hoc provenance. This publication includes code and input definitions;
the running campaigns' partial results are not presented as final results.
