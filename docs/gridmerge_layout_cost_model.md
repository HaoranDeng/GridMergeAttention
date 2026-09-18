# GridMerge Layout Cost Estimate

Static Wan 480p estimate for candidate GridMerge layouts. This is a feasibility
model, not a measured runtime report. It ignores cache reuse, occupancy,
mask overhead, and scalar softmax costs.

| Variant | q x k block | C | density | paper density | two-stage IO / flash | no-intermediate IO / flash | merge MAC / C4 | route MAC / C4 | total MAC / flash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `c1` | 30 x 52 | 1 | 0.01923 | 0.05256 | 0.291x | 0.126x | 0.25x | 0.25x | 0.069x |
| `c2` | 30 x 52 | 2 | 0.03846 | 0.10513 | 0.497x | 0.167x | 0.50x | 0.50x | 0.138x |
| `c3` | 30 x 52 | 3 | 0.05769 | 0.15769 | 0.702x | 0.208x | 0.75x | 0.75x | 0.208x |
| `c4` | 30 x 52 | 4 | 0.07692 | 0.21026 | 0.908x | 0.249x | 1.00x | 1.00x | 0.277x |
| `qc1_kr2_c4` | 30 x 104 | 4 | 0.03846 | 0.17179 | 0.497x | 0.167x | 1.00x | 0.50x | 0.238x |
| `qc1_kr2_c6` | 30 x 104 | 6 | 0.05769 | 0.25769 | 0.702x | 0.208x | 1.50x | 0.75x | 0.358x |
| `qc1_kr2_c7` | 30 x 104 | 7 | 0.06731 | 0.30064 | 0.805x | 0.229x | 1.75x | 0.88x | 0.417x |
| `qc1_kr2_c8` | 30 x 104 | 8 | 0.07692 | 0.34359 | 0.908x | 0.249x | 2.00x | 1.00x | 0.477x |
| `qc1_kr3_c5` | 30 x 156 | 5 | 0.03205 | 0.19872 | 0.428x | 0.154x | 1.25x | 0.42x | 0.282x |
| `qc1_kr3_c6` | 30 x 156 | 6 | 0.03846 | 0.23846 | 0.497x | 0.167x | 1.50x | 0.50x | 0.338x |
| `qc1_kr5_c6` | 30 x 260 | 6 | 0.02308 | 0.22308 | 0.332x | 0.135x | 1.50x | 0.30x | 0.323x |
| `qc1_kr5_c7` | 30 x 260 | 7 | 0.02692 | 0.26026 | 0.373x | 0.143x | 1.75x | 0.35x | 0.377x |
| `qc1_kr5_c8` | 30 x 260 | 8 | 0.03077 | 0.29744 | 0.414x | 0.151x | 2.00x | 0.40x | 0.431x |
| `qc1_kr5_c9` | 30 x 260 | 9 | 0.03462 | 0.33462 | 0.456x | 0.159x | 2.25x | 0.45x | 0.485x |
| `qc1_kr5_c10` | 30 x 260 | 10 | 0.03846 | 0.37179 | 0.497x | 0.167x | 2.50x | 0.50x | 0.538x |
| `qc1_kr5_c15` | 30 x 260 | 15 | 0.05769 | 0.55769 | 0.702x | 0.208x | 3.75x | 0.75x | 0.808x |
| `qc2_kr1_c6` | 60 x 52 | 6 | 0.11538 | 0.21538 | 0.702x | 0.208x | 0.75x | 1.50x | 0.265x |
| `qc2_kr1_c7` | 60 x 52 | 7 | 0.13462 | 0.25128 | 0.805x | 0.229x | 0.88x | 1.75x | 0.310x |
| `qc2_kr1_c8` | 60 x 52 | 8 | 0.15385 | 0.28718 | 0.908x | 0.249x | 1.00x | 2.00x | 0.354x |
| `qc4_kr1_c8` | 120 x 52 | 8 | 0.15385 | 0.22051 | 0.497x | 0.167x | 0.50x | 2.00x | 0.254x |
| `qc2_kr2_c8` | 60 x 104 | 8 | 0.07692 | 0.21026 | 0.497x | 0.167x | 1.00x | 1.00x | 0.277x |
| `qc2_kr3_c12` | 60 x 156 | 12 | 0.07692 | 0.27692 | 0.497x | 0.167x | 1.50x | 1.00x | 0.377x |
| `qc4_kr2_c16` | 120 x 104 | 16 | 0.15385 | 0.28718 | 0.497x | 0.167x | 1.00x | 2.00x | 0.354x |
| `qf1_kf3_c3` | 30 x 156 | 3 | 0.01923 | 0.11923 | 0.291x | 0.126x | 0.75x | 0.25x | 0.169x |
| `qf1_kf7_c4` | 30 x 364 | 4 | 0.01099 | 0.14432 | 0.203x | 0.109x | 1.00x | 0.14x | 0.211x |
| `qf1_kf7_c5` | 30 x 364 | 5 | 0.01374 | 0.18040 | 0.232x | 0.115x | 1.25x | 0.18x | 0.264x |
| `qf1_kf7_c7` | 30 x 364 | 7 | 0.01923 | 0.25256 | 0.291x | 0.126x | 1.75x | 0.25x | 0.369x |
| `qf1_kf7_c10` | 30 x 364 | 10 | 0.02747 | 0.36081 | 0.379x | 0.144x | 2.50x | 0.36x | 0.527x |

Interpretation:

- `two-stage IO / flash` includes current intermediate compressed K/V/bias store and load.
- `no-intermediate IO / flash` is an optimistic lower bound for a perfect fused design.
- `merge MAC / C4` and `route MAC / C4` show why some better-quality layouts are hard to speed up.

Main signal:

- `qc1_kr2_c8` has strong analysis-only quality, but it doubles merge MAC versus C4 and has no IO advantage in this model.
- `qc2_kr2_c8` keeps C4-like main MAC and lower idealized IO, but the tuned H200 kernel did not realize the speedup.
- Temporal-key grouping is quality-interesting at very low compressed-token density: `qf1_kf7_c7` keeps C1 density while reaching hard-C4-like error in PyTorch screening.
- The temporal-key tradeoff is merge-heavy. `qf1_kf7_c7` has only `0.25x` C4 route MAC, but `1.75x` C4 merge MAC; `qf1_kf7_c5` has `1.25x` C4 merge MAC and only `0.18x` route MAC.
- For the current two-stage implementation, temporal-key grouping is unlikely to reach the 3x-vs-Flash speed target unless the merge kernel is redesigned around the large 364-key block.

## Hard-C4 Fused Tile Constraint

`analysis/estimate_attention_io.py` also models an optimistic no-intermediate
hard-C4 kernel that computes compressed K/V and immediately consumes it, instead
of writing `merged_k`, `merged_v`, and `bias` to HBM and reading them back.

The tradeoff is sharp:

- Small query-column tiles reduce live state, but reload original K/V too many
  times. For example, `I_TILE=8,J_TILE=16` is only about `152` KiB live state,
  but its logical IO is still `1.965x` FlashAttention.
- Large query-column tiles reduce IO, but the live state becomes too large.
  `I_TILE=30,J_TILE=16` is only `0.494x` FlashAttention logical IO, but needs
  about `419` KiB of live tensors, well above a rough H200 one-block shared
  memory budget of `228` KiB.
- With the default candidate list, the only tile that is both below
  FlashAttention logical IO and within the approximate H200 shared-memory budget
  is `I_TILE=30,J_TILE=8`: `0.739x` FlashAttention IO with about `222` KiB live
  state (`97.5%` of the 228 KiB budget).

The estimator also includes a more conservative two-pass no-merged-KV option:
pass 1 computes only the second-stage softmax row max/sum, then pass 2
recomputes the merge and consumes V. After accounting for D-tiled output
accumulation honestly, this is not a clean win. If pass 2 uses a small
`D_TILE`, it must recompute the full-D route logits for every D chunk, repeating
seed-Q and K traffic. If it uses `D_TILE=128`, the live state again exceeds the
rough H200 shared-memory budget. With the default sweep, no two-pass tile is
both below FlashAttention logical IO and within the 228 KiB live-state budget.

This is only a feasibility filter, not a runtime prediction. The live-state
estimate mixes tensors that may live in registers, shared memory, or be
recomputed, and it ignores occupancy. But it makes the next kernel target
clear: the only current no-intermediate candidate is one-pass full-query-rows
`J_TILE=8`; avoid many tiny column tiles and avoid D-tiled two-pass unless a
new way is found to reuse route logits without storing another large
intermediate.

A synthetic H200 Triton probe in
`analysis/probe_hard_c4_fused_tile_feasibility.py` confirms that the padded
accumulator sizes for this direction compile. With `D=128`, logical query tiles
`30/60/120/240` pad to `32/64/128/256` rows and all ran on H200. The largest
case, `q_tile=240`, uses a padded fp32 accumulator of `256 x 128 = 128` KiB.
This does not prove the full one-pass hard-C4 kernel will be fast, but it
removes the first compile/live-state blocker for the `I_TILE=30,J_TILE=8`
target.
