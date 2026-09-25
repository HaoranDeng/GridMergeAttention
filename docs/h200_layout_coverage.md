# H200 Layout Coverage

Generated from local experiment JSON files. Tuned rows can be used for speed; analysis-only rows are quality screens.

- Parsed rows: 389
- Skipped non-JSON/empty files: 15
- Minimum calls in tables: 6

## Best Tuned Rows

Tuned speed rows require at least 90 recorded calls.

| Variant | Family | Experiment | Calls | Density | Block q x k | Rel-L2 | MSE | Mean ms | Speedup vs Flash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hard_kr2_c6 | wider_key_rows | metrics_full3 | 360 | 0.05769 | 30 x 104 | 0.06635 | 0.001281 | 27.030 | 0.54 |
| hard_qc2_kr2_c8 | query_columns_and_key_rows | metrics_full3 | 360 | 0.07692 | 60 x 104 | 0.07246 | 0.001528 | 23.555 | 0.62 |
| hard_c4_triton | original_row_query_col_keyrow | temporal_grouped_promising_full3_20260813 | 360 | 0.07692 | 30 x 52 | 0.07463 | 0.001621 | 15.347 | 1.03 |
| hard_c4_rt_triton | original_row_query_col_keyrow | metrics_20prompts | 2400 | 0.07692 | 30 x 52 | 0.07924 | 0.001880 | 18.659 | - |
| hard_c4 | original_row_query_col_keyrow | attention_metrics | 2400 | 0.07692 | 30 x 52 | 0.07926 | 0.001882 | 15.649 | - |
| fused_triton | original_row_query_col_keyrow | metrics | 2400 | 0.07692 | 30 x 52 | 0.07930 | 0.001883 | 157.207 | 0.49 |
| hard_kr3_c5 | wider_key_rows | metrics_full3 | 360 | 0.03205 | 30 x 156 | 0.08240 | 0.001977 | 27.844 | 0.52 |
| hard_kr2_c4 | wider_key_rows | hard_kr2_c4_realqkv_full3_20260813 | 360 | 0.03846 | 30 x 104 | 0.08585 | 0.002146 | 17.457 | 0.84 |
| hard_c3 | original_row_query_col_keyrow | metrics_full3 | 360 | 0.05769 | 30 x 52 | 0.08771 | 0.002239 | 11.573 | 1.26 |
| hard_c2 | original_row_query_col_keyrow | metrics_full3 | 360 | 0.03846 | 30 x 52 | 0.11824 | 0.004069 | 7.396 | 1.97 |
| hard_c2_rt_triton | original_row_query_col_keyrow | metrics_20prompts | 2400 | 0.03846 | 30 x 52 | 0.12368 | 0.004580 | 8.431 | - |
| monarch_rt_triton | original_row_query_col_keyrow | metrics_20prompts | 2400 | 0.01923 | 30 x 52 | 0.18039 | 0.009743 | 4.160 | - |

## Best Low-Density Rows

These are at or below C2 compressed-token density (`<= 0.04`).

| Variant | Family | Experiment | Calls | Density | Block q x k | Rel-L2 | MSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qc1_kr5_c10 | wider_key_rows | rowcol_wider_key_lowdensity_metrics | 360 | 0.03846 | 30 x 260 | 0.06135 | 0.001096 |
| qf1_kf7_c10 | temporal_frame_group | temporal_kf7_full3_20260813 | 360 | 0.02747 | 30 x 364 | 0.06305 | 0.001157 |
| qc1_kr5_c9 | wider_key_rows | rowcol_kr5_c8_c9_full3_20260813 | 360 | 0.03462 | 30 x 260 | 0.06491 | 0.001226 |
| qc1_kr5_c8 | wider_key_rows | rowcol_kr5_c8_c9_full3_20260813 | 360 | 0.03077 | 30 x 260 | 0.07041 | 0.001443 |
| qf1_kf3_c6 | temporal_frame_group | temporal_grouped_promising_full3_20260813 | 360 | 0.03846 | 30 x 156 | 0.07198 | 0.001508 |
| qc1_kr3_c6 | wider_key_rows | rowcol_wider_key_lowdensity_metrics | 360 | 0.03846 | 30 x 156 | 0.07393 | 0.001591 |
| qf1_kf7_c7 | temporal_frame_group | temporal_kf7_full3_20260813 | 360 | 0.01923 | 30 x 364 | 0.07408 | 0.001597 |
| qc1_kr5_c9 | wider_key_rows | rowcol_kr5_c6_c9_screen_20260813 | 8 | 0.03462 | 30 x 260 | 0.08195 | 0.000866 |
| hard_kr3_c5 | wider_key_rows | metrics_full3 | 360 | 0.03205 | 30 x 156 | 0.08240 | 0.001977 |
| qc1_kr3_c5 | wider_key_rows | rowcol_wider_key_lowdensity_metrics | 360 | 0.03205 | 30 x 156 | 0.08251 | 0.001982 |
| hard_kr2_c4 | wider_key_rows | hard_kr2_c4_realqkv_full3_20260813 | 360 | 0.03846 | 30 x 104 | 0.08585 | 0.002146 |
| hard_kr2_c4 | wider_key_rows | hard_kr2_c4_metrics | 360 | 0.03846 | 30 x 104 | 0.08585 | 0.002146 |
| qf1_kf3_c6 | temporal_frame_group | temporal_grouped_rowcol_screen_20260813 | 8 | 0.03846 | 30 x 156 | 0.08639 | 0.000962 |
| qc1_kr5_c8 | wider_key_rows | rowcol_kr5_c6_c9_screen_20260813 | 8 | 0.03077 | 30 x 260 | 0.08700 | 0.000976 |
| qf1_kf7_c5 | temporal_frame_group | temporal_kf7_full3_20260813 | 360 | 0.01374 | 30 x 364 | 0.08704 | 0.002205 |
| qf1_kf7_c7 | temporal_frame_group | temporal_grouped_rowcol_screen_20260813 | 8 | 0.01923 | 30 x 364 | 0.08846 | 0.001009 |
| qc1_kr3_c5 | wider_key_rows | rowcol_more_rows_screen_metrics | 6 | 0.03205 | 30 x 156 | 0.08977 | 0.000990 |
| qc1_kr3_c6 | wider_key_rows | rowcol_interleaved_keyrow_screen_20260813 | 8 | 0.03846 | 30 x 156 | 0.08992 | 0.001043 |
| hard_kr2_c4 | wider_key_rows | kr2_c4_realqkv_smoke_20260813 | 6 | 0.03846 | 30 x 104 | 0.09034 | 0.001003 |
| hard_kr2_c4 | wider_key_rows | kr2_c4_realqkv_smoke_metrics | 6 | 0.03846 | 30 x 104 | 0.09034 | 0.001003 |

## Best C4-Density-Or-Lower Rows

These are at or below hard-C4 compressed-token density (`<= 0.08`).

| Variant | Family | Experiment | Calls | Density | Block q x k | Rel-L2 | MSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qc1_kr5_c15 | wider_key_rows | rowcol_wider_key_lowdensity_metrics | 360 | 0.05769 | 30 x 260 | 0.04953 | 0.000714 |
| qc1_kr2_c8 | wider_key_rows | candidate_full3_metrics | 360 | 0.07692 | 30 x 104 | 0.05567 | 0.000902 |
| qf1_kf3_c12 | temporal_frame_group | temporal_grouped_rowcol_screen_20260813 | 8 | 0.07692 | 30 x 156 | 0.05858 | 0.000443 |
| qc1_kr2_c7 | wider_key_rows | candidate_full3_metrics | 360 | 0.06731 | 30 x 104 | 0.06093 | 0.001081 |
| qc1_kr5_c10 | wider_key_rows | rowcol_wider_key_lowdensity_metrics | 360 | 0.03846 | 30 x 260 | 0.06135 | 0.001096 |
| qf1_kf7_c10 | temporal_frame_group | temporal_kf7_full3_20260813 | 360 | 0.02747 | 30 x 364 | 0.06305 | 0.001157 |
| qc1_kr5_c9 | wider_key_rows | rowcol_kr5_c8_c9_full3_20260813 | 360 | 0.03462 | 30 x 260 | 0.06491 | 0.001226 |
| hard_kr2_c6 | wider_key_rows | metrics_full3 | 360 | 0.05769 | 30 x 104 | 0.06635 | 0.001281 |
| qf1_kf3_c9 | temporal_frame_group | temporal_grouped_rowcol_screen_20260813 | 8 | 0.05769 | 30 x 156 | 0.07033 | 0.000638 |
| qc1_kr5_c8 | wider_key_rows | rowcol_kr5_c8_c9_full3_20260813 | 360 | 0.03077 | 30 x 260 | 0.07041 | 0.001443 |
| qf1_kf3_c6 | temporal_frame_group | temporal_grouped_promising_full3_20260813 | 360 | 0.03846 | 30 x 156 | 0.07198 | 0.001508 |
| hard_qc2_kr2_c8 | query_columns_and_key_rows | metrics_full3 | 360 | 0.07692 | 60 x 104 | 0.07246 | 0.001528 |
| qc2_kr2_c8 | query_columns_and_key_rows | rowcol_qc2kr2_c8_c10_full3_20260813 | 360 | 0.07692 | 60 x 104 | 0.07253 | 0.001531 |
| hard_c4_rt_triton | original_row_query_col_keyrow | attention_metrics_full1 | 120 | 0.07692 | 30 x 52 | 0.07335 | 0.001632 |
| qc1_kr3_c6 | wider_key_rows | rowcol_wider_key_lowdensity_metrics | 360 | 0.03846 | 30 x 156 | 0.07393 | 0.001591 |
| qf1_kf7_c7 | temporal_frame_group | temporal_kf7_full3_20260813 | 360 | 0.01923 | 30 x 364 | 0.07408 | 0.001597 |
| hard_c4 | original_row_query_col_keyrow | realqkv_3prompt_metrics | 360 | 0.07692 | 30 x 52 | 0.07463 | 0.001620 |
| hard_c4 | original_row_query_col_keyrow | hard_kr2_c4_realqkv_full3_20260813 | 360 | 0.07692 | 30 x 52 | 0.07463 | 0.001621 |
| hard_c4 | original_row_query_col_keyrow | hard_c2_full3_metrics | 360 | 0.07692 | 30 x 52 | 0.07463 | 0.001621 |
| hard_c4 | original_row_query_col_keyrow | hard_c3_full3_metrics | 360 | 0.07692 | 30 x 52 | 0.07463 | 0.001621 |
| hard_c4 | original_row_query_col_keyrow | monarchrt_full3_metrics | 360 | 0.07692 | 30 x 52 | 0.07463 | 0.001621 |
| hard_c4 | original_row_query_col_keyrow | metrics_full3 | 360 | 0.07692 | 30 x 52 | 0.07463 | 0.001621 |
| hard_c4 | original_row_query_col_keyrow | metrics_full3 | 360 | 0.07692 | 30 x 52 | 0.07463 | 0.001621 |
| hard_c4 | original_row_query_col_keyrow | metrics_full3 | 360 | 0.07692 | 30 x 52 | 0.07463 | 0.001621 |
| hard_c4_triton | original_row_query_col_keyrow | axis_low_density_metrics | 360 | 0.07692 | 30 x 52 | 0.07463 | 0.001621 |

## Best By Family

| Family | Rows | Best Quality | Rel-L2 | Density | Best <=0.08 Density | Rel-L2 | Density |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: |
| interleaved_key_rows | 4 | qc1_kr2_c6_ginterleave | 0.08906 | 0.05769 | qc1_kr2_c6_ginterleave | 0.08906 | 0.05769 |
| kernel_prototype | 5 | hard_c4_pytorch | 0.07955 | - | - | - | - |
| key_patch | 8 | kp2x26_c6 | 0.07548 | 0.11538 | kp2x26_c4 | 0.09306 | 0.07692 |
| mixed_axis | 40 | qt_kw_c8 | 0.05709 | 0.15385 | qt_kw_c4 | 0.09236 | 0.07692 |
| original_row_query_col_keyrow | 110 | hard_c4_rt_triton | 0.07335 | 0.07692 | hard_c4_rt_triton | 0.07335 | 0.07692 |
| query_column_group | 19 | qc2_kr1_c8 | 0.05827 | 0.15385 | hard_qc2_c4 | 0.09603 | 0.07692 |
| query_columns_and_key_rows | 33 | qc2_kr2_c12 | 0.05495 | 0.11538 | hard_qc2_kr2_c8 | 0.07246 | 0.07692 |
| query_key_patch | 9 | qp15x2_kp1x52_c6 | 0.07207 | 0.11538 | qp30x1_kp1x52_c4 | 0.08769 | 0.07692 |
| same_axis | 6 | qt_kt_c3 | 0.74037 | 0.14286 | qw_kw_c4 | 0.76274 | 0.07692 |
| seed_selection | 8 | c4_prefilter12_greedy | 0.06028 | - | - | - | - |
| temporal_adaptive_c | 11 | ta_r1c4_far2 | 0.10727 | 0.04378 | ta_r1c4_far2 | 0.10727 | 0.04378 |
| temporal_frame_group | 28 | qf1_kf3_c12 | 0.05858 | 0.07692 | qf1_kf3_c12 | 0.05858 | 0.07692 |
| transpose_rowcol | 9 | qw_kh_c4 | 0.10390 | 0.13333 | qw_kh_c2 | 0.12803 | 0.06667 |
| wider_key_rows | 36 | qc1_kr5_c15 | 0.04953 | 0.05769 | qc1_kr5_c15 | 0.04953 | 0.05769 |

## Current Search Implication

- The original full query-column plus full key-row geometry remains the only tuned speed/quality frontier.
- Multi-row key blocks and temporal-key blocks give strong quality signals, but direct two-stage kernels are merge-bound.
- Query grouping, transpose, same-axis, key-column grouping, and 2D patch layouts have not recovered enough quality at matched density.
- The next useful speed work is kernel-level IO reduction for a quality-preserving layout, not another fixed geometric split with weaker alignment.
