"""Screen lightweight Hard-C4 algorithm variants on real Wan2.1-14B 720p QKV.

Every variant keeps four grid-local KV tokens, entropy routing bias, and full
second-stage attention. Experiments change only how the four groups are formed
and how their merge weights are computed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low "
    "quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen "
    "limbs, fused fingers, still picture, messy background, three legs, many "
    "people in the background, walking backwards"
)

ROWS = 45
COLS = 80
FRAMES = 21
SEEDS = 4
NEG_INF = torch.finfo(torch.float32).min


VARIANTS = {
    # Current algorithm.
    "baseline": {"seed": "center_right", "score": "key_softmax"},
    "paper_confidence_curvature_c4": {
        "seed": "center_right",
        "score": "key_softmax",
        "confidence_sharpen_strength": 0.25,
        "confidence_sharpen_threshold": 0.5,
        "curvature_bias_scale": 0.5,
    },
    # Seed-only changes.
    "seed_center_left": {"seed": "center_left", "score": "key_softmax"},
    "seed_right": {"seed": "right", "score": "key_softmax"},
    "seed_wide": {"seed": "wide", "score": "key_softmax"},
    "seed_wider_left": {"seed": "wider_left", "score": "key_softmax"},
    "seed_wider_balanced": {"seed": "wider_balanced", "score": "key_softmax"},
    "seed_wider_symmetric": {"seed": "wider_symmetric", "score": "key_softmax"},
    "seed_wider_right": {"seed": "wider_right", "score": "key_softmax"},
    # Wider layouts that preserve their spacing at the top and bottom rows.
    "seed_sym4_shift": {
        "seed": "offsets",
        "offsets": [-4, -1, 0, 4],
        "boundary": "shift",
        "score": "key_softmax",
    },
    "seed_sym5_shift": {
        "seed": "offsets",
        "offsets": [-5, -1, 0, 5],
        "boundary": "shift",
        "score": "key_softmax",
    },
    "seed_sym6_shift": {
        "seed": "offsets",
        "offsets": [-6, -1, 0, 6],
        "boundary": "shift",
        "score": "key_softmax",
    },
    "seed_spread4_shift": {
        "seed": "offsets",
        "offsets": [-4, -2, 0, 4],
        "boundary": "shift",
        "score": "key_softmax",
    },
    "seed_spread5_shift": {
        "seed": "offsets",
        "offsets": [-5, -2, 0, 5],
        "boundary": "shift",
        "score": "key_softmax",
    },
    # Local query averaging changes the seed representative, not the QK count.
    "seed_symmetric_pool3": {
        "seed": "wider_symmetric", "seed_pool": 3, "score": "key_softmax"
    },
    "seed_sym4_shift_pool3": {
        "seed": "offsets",
        "offsets": [-4, -1, 0, 4],
        "boundary": "shift",
        "seed_pool": 3,
        "score": "key_softmax",
    },
    "seed_balanced_pool3": {
        "seed": "wider_balanced", "seed_pool": 3, "score": "key_softmax"
    },
    # Assignment calibration around the best broad seed geometry.
    "sym_scoretemp_0p5": {
        "seed": "wider_symmetric", "score": "temperature", "temperature": 0.5
    },
    "sym_scoretemp_0p75": {
        "seed": "wider_symmetric", "score": "temperature", "temperature": 0.75
    },
    "sym_scoretemp_1p25": {
        "seed": "wider_symmetric", "score": "temperature", "temperature": 1.25
    },
    "sym_scoretemp_1p5": {
        "seed": "wider_symmetric", "score": "temperature", "temperature": 1.5
    },
    "sym_scoretemp_2p0": {
        "seed": "wider_symmetric", "score": "temperature", "temperature": 2.0
    },
    "sym_normalizer_0p5": {
        "seed": "wider_symmetric", "score": "normalizer_weight", "weight": 0.5
    },
    "sym_normalizer_0p75": {
        "seed": "wider_symmetric", "score": "normalizer_weight", "weight": 0.75
    },
    "sym_normalizer_1p25": {
        "seed": "wider_symmetric", "score": "normalizer_weight", "weight": 1.25
    },
    "sym_normalizer_1p5": {
        "seed": "wider_symmetric", "score": "normalizer_weight", "weight": 1.5
    },
    # The partition is unchanged; only the within-cluster merge weights change.
    "sym_merge_0p5": {
        "seed": "wider_symmetric", "score": "key_softmax", "merge_temperature": 0.5
    },
    "sym_merge_0p75": {
        "seed": "wider_symmetric", "score": "key_softmax", "merge_temperature": 0.75
    },
    "sym_merge_1p25": {
        "seed": "wider_symmetric", "score": "key_softmax", "merge_temperature": 1.25
    },
    "sym_merge_1p5": {
        "seed": "wider_symmetric", "score": "key_softmax", "merge_temperature": 1.5
    },
    "sym_merge_2p0": {
        "seed": "wider_symmetric", "score": "key_softmax", "merge_temperature": 2.0
    },
    # Fractional key partitions reuse the same four seed-key dot products.
    "wide_soft_0p1": {
        "seed": "wide", "score": "key_softmax", "route": "soft", "route_temperature": 0.1
    },
    "wide_soft_0p25": {
        "seed": "wide", "score": "key_softmax", "route": "soft", "route_temperature": 0.25
    },
    "wide_soft_0p5": {
        "seed": "wide", "score": "key_softmax", "route": "soft", "route_temperature": 0.5
    },
    "wide_soft_1p0": {
        "seed": "wide", "score": "key_softmax", "route": "soft", "route_temperature": 1.0
    },
    "wide_soft_0p1_no_responsibility_correction": {
        "seed": "wide",
        "score": "key_softmax",
        "route": "soft",
        "route_temperature": 0.1,
        "responsibility_correction": False,
    },
    "wide_soft_0p25_no_responsibility_correction": {
        "seed": "wide",
        "score": "key_softmax",
        "route": "soft",
        "route_temperature": 0.25,
        "responsibility_correction": False,
    },
    "wide_soft_0p5_no_responsibility_correction": {
        "seed": "wide",
        "score": "key_softmax",
        "route": "soft",
        "route_temperature": 0.5,
        "responsibility_correction": False,
    },
    "wide_soft_1p0_no_responsibility_correction": {
        "seed": "wide",
        "score": "key_softmax",
        "route": "soft",
        "route_temperature": 1.0,
        "responsibility_correction": False,
    },
    "wide_top2_0p1": {
        "seed": "wide", "score": "key_softmax", "route": "top2", "route_temperature": 0.1
    },
    "wide_top2_0p25": {
        "seed": "wide", "score": "key_softmax", "route": "top2", "route_temperature": 0.25
    },
    "wide_top2_0p5": {
        "seed": "wide", "score": "key_softmax", "route": "top2", "route_temperature": 0.5
    },
    "wide_top2_1p0": {
        "seed": "wide", "score": "key_softmax", "route": "top2", "route_temperature": 1.0
    },
    "wide_raw_hard": {"seed": "wide", "score": "raw"},
    "wide_cosine_hard": {"seed": "wide", "score": "cosine"},
    "wide_dualprobe_assign": {"seed": "wide", "score": "dual_probe"},
    "wide_dualprobe_mix": {
        "seed": "wide",
        "score": "dual_probe",
        "merge_from_assignment_scores": True,
    },
    # Multi-anchor merge rules reuse the same four seed-key logits. They keep
    # the hard partition and the full second-stage route unchanged.
    "wide_anchor_intercept_mean": {
        "seed": "wide", "score": "key_softmax", "bias_rule": "anchor_mean"
    },
    "wide_anchor_intercept_lse": {
        "seed": "wide", "score": "key_softmax", "bias_rule": "anchor_lse"
    },
    "wide_anchor_lse_entropy": {
        "seed": "wide", "score": "key_softmax", "merge_rule": "anchor_lse"
    },
    "wide_anchor_lse_intercept": {
        "seed": "wide",
        "score": "key_softmax",
        "merge_rule": "anchor_lse",
        "bias_rule": "anchor_mean",
    },
    "wide_anchor_mix_entropy": {
        "seed": "wide", "score": "key_softmax", "merge_rule": "anchor_mix"
    },
    "wide_anchor_mix_intercept": {
        "seed": "wide",
        "score": "key_softmax",
        "merge_rule": "anchor_mix",
        "bias_rule": "anchor_mean",
    },
    "wide_anchor_geom_intercept": {
        "seed": "wide",
        "score": "key_softmax",
        "merge_rule": "anchor_geom",
        "bias_rule": "anchor_mean",
    },
    "wide_margin_m1": {
        "seed": "wide", "score": "key_softmax", "assignment_margin_scale": -1.0
    },
    "wide_margin_m0p5": {
        "seed": "wide", "score": "key_softmax", "assignment_margin_scale": -0.5
    },
    "wide_margin_p0p5": {
        "seed": "wide", "score": "key_softmax", "assignment_margin_scale": 0.5
    },
    "wide_margin_p1": {
        "seed": "wide", "score": "key_softmax", "assignment_margin_scale": 1.0
    },
    # Q-space farthest-point seeds add no seed-key products. Candidate scoring
    # is shared across every key frame for a query frame/key row pair.
    "qfarthest9": {
        "seed": "q_farthest", "seed_candidates": 9, "score": "key_softmax"
    },
    "qfarthest13": {
        "seed": "q_farthest", "seed_candidates": 13, "score": "key_softmax"
    },
    "qfarthest7": {
        "seed": "q_farthest", "seed_candidates": 7, "score": "key_softmax"
    },
    "qfarthest11": {
        "seed": "q_farthest", "seed_candidates": 11, "score": "key_softmax"
    },
    "qfarthest15": {
        "seed": "q_farthest", "seed_candidates": 15, "score": "key_softmax"
    },
    "qfarthest17": {
        "seed": "q_farthest", "seed_candidates": 17, "score": "key_softmax"
    },
    "qfarthest21": {
        "seed": "q_farthest", "seed_candidates": 21, "score": "key_softmax"
    },
    "qfarthest13_cosine": {
        "seed": "q_farthest",
        "seed_candidates": 13,
        "seed_distance": "cosine",
        "score": "key_softmax",
    },
    "qfarthest15_cosine": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_distance": "cosine",
        "score": "key_softmax",
    },
    "qfarthest13_anchor_left": {
        "seed": "q_farthest",
        "seed_candidates": 13,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
    },
    "qfarthest13_anchor_right": {
        "seed": "q_farthest",
        "seed_candidates": 13,
        "seed_anchor_offsets": [1],
        "score": "key_softmax",
    },
    "qfarthest15_anchor_left": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
    },
    "qfarthest15_anchor_right": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [1],
        "score": "key_softmax",
    },
    "qfarthest15_anchor_neighbors": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1, 1],
        "score": "key_softmax",
    },
    "qfarthest11_anchor_left": {
        "seed": "q_farthest",
        "seed_candidates": 11,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
    },
    "qfarthest17_anchor_left": {
        "seed": "q_farthest",
        "seed_candidates": 17,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
    },
    "qfarthest19_anchor_left": {
        "seed": "q_farthest",
        "seed_candidates": 19,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
    },
    "qfarthest21_anchor_left": {
        "seed": "q_farthest",
        "seed_candidates": 21,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
    },
    "qfarthest15_anchor_left2": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-2],
        "score": "key_softmax",
    },
    "qfarthest15_anchor_left3": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-3],
        "score": "key_softmax",
    },
    "qfarthest15_anchor_left_pair": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-2, -1],
        "score": "key_softmax",
    },
    "qfarthest15_anchor_left_right_any": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "seed_selection_sides": ["right", "any"],
        "score": "key_softmax",
    },
    "qfarthest15_anchor_left_balanced": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "seed_selection_sides": ["left", "right"],
        "score": "key_softmax",
    },
    # One local query-space Lloyd update replaces each discrete seed by the
    # centroid of the nearby query rows that it represents. QK work stays C4.
    "qcentroid11": {
        "seed": "q_centroid",
        "seed_candidates": 11,
        "centroid_normalize": False,
        "centroid_steps": 1,
        "score": "key_softmax",
    },
    "qcentroid15": {
        "seed": "q_centroid",
        "seed_candidates": 15,
        "centroid_normalize": False,
        "centroid_steps": 1,
        "score": "key_softmax",
    },
    "qcentroid15_norm": {
        "seed": "q_centroid",
        "seed_candidates": 15,
        "centroid_normalize": True,
        "centroid_steps": 1,
        "score": "key_softmax",
    },
    "qcentroid15_norm_step2": {
        "seed": "q_centroid",
        "seed_candidates": 15,
        "centroid_normalize": True,
        "centroid_steps": 2,
        "score": "key_softmax",
    },
    "qcentroid21_norm": {
        "seed": "q_centroid",
        "seed_candidates": 21,
        "centroid_normalize": True,
        "centroid_steps": 1,
        "score": "key_softmax",
    },
    "qcentroid15_norm_bias_1p05": {
        "seed": "q_centroid",
        "seed_candidates": 15,
        "centroid_normalize": True,
        "centroid_steps": 1,
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qcover13": {
        "seed": "q_cover", "seed_candidates": 13, "score": "key_softmax"
    },
    "qcover15": {
        "seed": "q_cover", "seed_candidates": 15, "score": "key_softmax"
    },
    "qcover15_cosine": {
        "seed": "q_cover",
        "seed_candidates": 15,
        "seed_distance": "cosine",
        "score": "key_softmax",
    },
    "qproxy15_t1": {
        "seed": "q_key_proxy",
        "seed_candidates": 15,
        "proxy_temporal_groups": 1,
        "score": "key_softmax",
    },
    "qproxy15_t3": {
        "seed": "q_key_proxy",
        "seed_candidates": 15,
        "proxy_temporal_groups": 3,
        "score": "key_softmax",
    },
    "qproxy15_t3_anchor_left": {
        "seed": "q_key_proxy",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_groups": 3,
        "score": "key_softmax",
    },
    "qproxy15_t3_cosine": {
        "seed": "q_key_proxy",
        "seed_candidates": 15,
        "proxy_temporal_groups": 3,
        "proxy_cosine": True,
        "score": "key_softmax",
    },
    "qsample11_s16": {
        "seed": "q_sampled_key_cover",
        "seed_candidates": 11,
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "proxy_score": "key_softmax",
        "score": "key_softmax",
    },
    "qsample15_s8": {
        "seed": "q_sampled_key_cover",
        "seed_candidates": 15,
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 2,
        "proxy_score": "key_softmax",
        "score": "key_softmax",
    },
    "qsample15_s16": {
        "seed": "q_sampled_key_cover",
        "seed_candidates": 15,
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "proxy_score": "key_softmax",
        "score": "key_softmax",
    },
    "qsample15_s32": {
        "seed": "q_sampled_key_cover",
        "seed_candidates": 15,
        "proxy_temporal_samples": 8,
        "proxy_spatial_samples": 4,
        "proxy_score": "key_softmax",
        "score": "key_softmax",
    },
    "qsample21_s16": {
        "seed": "q_sampled_key_cover",
        "seed_candidates": 21,
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "proxy_score": "key_softmax",
        "score": "key_softmax",
    },
    "qsample15_s8_anchor_left": {
        "seed": "q_sampled_key_cover",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 2,
        "proxy_score": "key_softmax",
        "score": "key_softmax",
    },
    "qsample15_s16_anchor_left": {
        "seed": "q_sampled_key_cover",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "proxy_score": "key_softmax",
        "score": "key_softmax",
    },
    "qsample15_s32_anchor_left": {
        "seed": "q_sampled_key_cover",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 8,
        "proxy_spatial_samples": 4,
        "proxy_score": "key_softmax",
        "score": "key_softmax",
    },
    "qsample15_s16_raw": {
        "seed": "q_sampled_key_cover",
        "seed_candidates": 15,
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "proxy_score": "raw",
        "score": "key_softmax",
    },
    "qsample15_s8_raw": {
        "seed": "q_sampled_key_cover",
        "seed_candidates": 15,
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 2,
        "proxy_score": "raw",
        "score": "key_softmax",
    },
    "qsample15_s32_raw": {
        "seed": "q_sampled_key_cover",
        "seed_candidates": 15,
        "proxy_temporal_samples": 8,
        "proxy_spatial_samples": 4,
        "proxy_score": "raw",
        "score": "key_softmax",
    },
    "qsample17_s16_raw": {
        "seed": "q_sampled_key_cover",
        "seed_candidates": 17,
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "proxy_score": "raw",
        "score": "key_softmax",
    },
    "qsample15_s16_raw_bias_1p05": {
        "seed": "q_sampled_key_cover",
        "seed_candidates": 15,
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "proxy_score": "raw",
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qsample15_s16_raw_anchor_left": {
        "seed": "q_sampled_key_cover",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "proxy_score": "raw",
        "score": "key_softmax",
    },
    "qsample15_s16_raw_anchor_left_bias_1p05": {
        "seed": "q_sampled_key_cover",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "proxy_score": "raw",
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qresponse15_s16_raw": {
        "seed": "q_response_farthest",
        "seed_candidates": 15,
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "response_scale": "raw",
        "score": "key_softmax",
    },
    "qresponse15_s16_logprob": {
        "seed": "q_response_farthest",
        "seed_candidates": 15,
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "response_scale": "logprob",
        "score": "key_softmax",
    },
    "qresponse15_s16_prob": {
        "seed": "q_response_farthest",
        "seed_candidates": 15,
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "response_scale": "probability",
        "score": "key_softmax",
    },
    "qresponse15_s16_anchor_left_raw": {
        "seed": "q_response_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "response_scale": "raw",
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qresponse15_s16_anchor_left_logprob": {
        "seed": "q_response_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "response_scale": "logprob",
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qresponse15_s32_logprob": {
        "seed": "q_response_farthest",
        "seed_candidates": 15,
        "proxy_temporal_samples": 8,
        "proxy_spatial_samples": 4,
        "response_scale": "logprob",
        "score": "key_softmax",
    },
    "qresponse17_s32_logprob": {
        "seed": "q_response_farthest",
        "seed_candidates": 17,
        "proxy_temporal_samples": 8,
        "proxy_spatial_samples": 4,
        "response_scale": "logprob",
        "score": "key_softmax",
    },
    "qresponse_centroid_blend_0p1": {
        "seed": "q_response_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "response_scale": "logprob",
        "seed_centroid_blend": 0.1,
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qresponse_centroid_blend_0p25": {
        "seed": "q_response_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "response_scale": "logprob",
        "seed_centroid_blend": 0.25,
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qresponse_adaptive_centroid_0p25": {
        "seed": "q_response_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "response_scale": "logprob",
        "seed_centroid_blend": 0.25,
        "blend_adaptive_only": True,
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qresponse_adaptive_centroid_0p5": {
        "seed": "q_response_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "response_scale": "logprob",
        "seed_centroid_blend": 0.5,
        "blend_adaptive_only": True,
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qresponse_cover15_s16_raw": {
        "seed": "q_response_cover",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "response_scale": "raw",
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qresponse_cover15_s16_logprob": {
        "seed": "q_response_cover",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "response_scale": "logprob",
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qresponse_cover15_s16_prob": {
        "seed": "q_response_cover",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "response_scale": "probability",
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qresponse_span15_s16_raw": {
        "seed": "q_response_span",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "response_scale": "raw",
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qresponse_span15_s16_logprob": {
        "seed": "q_response_span",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "response_scale": "logprob",
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qresponse_span15_s16_prob": {
        "seed": "q_response_span",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "response_scale": "probability",
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qresponse_exact15_s16_raw": {
        "seed": "q_response_exact_cover",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "response_scale": "raw",
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qresponse_exact15_s16_logprob": {
        "seed": "q_response_exact_cover",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "response_scale": "logprob",
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qresponse_exact15_s16_prob": {
        "seed": "q_response_exact_cover",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "response_scale": "probability",
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qresponse_exact15_s32_logprob": {
        "seed": "q_response_exact_cover",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 8,
        "proxy_spatial_samples": 4,
        "response_scale": "logprob",
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qblock_response15_s4_farthest_raw": {
        "seed": "q_block_response",
        "seed_candidates": 15,
        "block_proxy_samples": 4,
        "block_response_method": "farthest",
        "response_scale": "raw",
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qblock_response15_s4_farthest_prob": {
        "seed": "q_block_response",
        "seed_candidates": 15,
        "block_proxy_samples": 4,
        "block_response_method": "farthest",
        "response_scale": "probability",
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qblock_response15_s4_span_raw": {
        "seed": "q_block_response",
        "seed_candidates": 15,
        "block_proxy_samples": 4,
        "block_response_method": "span",
        "response_scale": "raw",
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qblock_response15_s4_span_prob": {
        "seed": "q_block_response",
        "seed_candidates": 15,
        "block_proxy_samples": 4,
        "block_response_method": "span",
        "response_scale": "probability",
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qblock_response15_s4_span_prob_anchor_left": {
        "seed": "q_block_response",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "block_proxy_samples": 4,
        "block_response_method": "span",
        "response_scale": "probability",
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qblock_response15_s8_farthest_prob": {
        "seed": "q_block_response",
        "seed_candidates": 15,
        "block_proxy_samples": 8,
        "block_response_method": "farthest",
        "response_scale": "probability",
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qblock_response15_s8_span_prob": {
        "seed": "q_block_response",
        "seed_candidates": 15,
        "block_proxy_samples": 8,
        "block_response_method": "span",
        "response_scale": "probability",
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qblock_response15_s8_span_prob_anchor_left": {
        "seed": "q_block_response",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "block_proxy_samples": 8,
        "block_response_method": "span",
        "response_scale": "probability",
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qhybrid15_s16_pqq": {
        "seed": "q_hybrid",
        "seed_candidates": 15,
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "proxy_score": "raw",
        "hybrid_order": ["proxy", "farthest", "farthest"],
        "score": "key_softmax",
    },
    "qhybrid15_s16_qpq": {
        "seed": "q_hybrid",
        "seed_candidates": 15,
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "proxy_score": "raw",
        "hybrid_order": ["farthest", "proxy", "farthest"],
        "score": "key_softmax",
    },
    "qhybrid15_s16_qqp": {
        "seed": "q_hybrid",
        "seed_candidates": 15,
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "proxy_score": "raw",
        "hybrid_order": ["farthest", "farthest", "proxy"],
        "score": "key_softmax",
    },
    "qhybrid15_s16_anchor_left_pq": {
        "seed": "q_hybrid",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "proxy_score": "raw",
        "hybrid_order": ["proxy", "farthest"],
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qhybrid15_s16_anchor_left_qp": {
        "seed": "q_hybrid",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "proxy_score": "raw",
        "hybrid_order": ["farthest", "proxy"],
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qhybrid_vnorm_m2p0": {
        "seed": "q_hybrid",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "proxy_score": "raw",
        "hybrid_order": ["farthest", "proxy"],
        "score": "key_softmax",
        "bias_scale": 1.05,
        "merge_value_norm_scale": -2.0,
    },
    "qhybrid_vnorm_m1p0": {
        "seed": "q_hybrid",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "proxy_score": "raw",
        "hybrid_order": ["farthest", "proxy"],
        "score": "key_softmax",
        "bias_scale": 1.05,
        "merge_value_norm_scale": -1.0,
    },
    "qhybrid_vnorm_m0p5": {
        "seed": "q_hybrid",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "proxy_score": "raw",
        "hybrid_order": ["farthest", "proxy"],
        "score": "key_softmax",
        "bias_scale": 1.05,
        "merge_value_norm_scale": -0.5,
    },
    "qhybrid_vnorm_p0p5": {
        "seed": "q_hybrid",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "proxy_score": "raw",
        "hybrid_order": ["farthest", "proxy"],
        "score": "key_softmax",
        "bias_scale": 1.05,
        "merge_value_norm_scale": 0.5,
    },
    "qhybrid_vnorm_p1p0": {
        "seed": "q_hybrid",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "proxy_score": "raw",
        "hybrid_order": ["farthest", "proxy"],
        "score": "key_softmax",
        "bias_scale": 1.05,
        "merge_value_norm_scale": 1.0,
    },
    "qhybrid_vnorm_p2p0": {
        "seed": "q_hybrid",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "proxy_score": "raw",
        "hybrid_order": ["farthest", "proxy"],
        "score": "key_softmax",
        "bias_scale": 1.05,
        "merge_value_norm_scale": 2.0,
    },
    # Score-quantile C4 removes seed search entirely. One or two local query
    # probes order the 80 keys, which are split into four content-adaptive
    # groups. The same probe logits then provide the within-group weights.
    "qquantile1_count": {
        "seed": "score_quantile",
        "probe_offsets": [0],
        "partition": "equal_count",
        "score": "raw",
    },
    "qquantile1_count_bias_1p05": {
        "seed": "score_quantile",
        "probe_offsets": [0],
        "partition": "equal_count",
        "score": "raw",
        "bias_scale": 1.05,
    },
    "qquantile1_count_temp_0p75": {
        "seed": "score_quantile",
        "probe_offsets": [0],
        "partition": "equal_count",
        "score": "raw",
        "merge_temperature": 0.75,
    },
    "qquantile1_count_temp_1p25": {
        "seed": "score_quantile",
        "probe_offsets": [0],
        "partition": "equal_count",
        "score": "raw",
        "merge_temperature": 1.25,
    },
    "qquantile2_count_mean": {
        "seed": "score_quantile",
        "probe_offsets": [-1, 0],
        "probe_reduce": "mean",
        "partition": "equal_count",
        "score": "raw",
    },
    "qquantile2_count_max": {
        "seed": "score_quantile",
        "probe_offsets": [-1, 0],
        "probe_reduce": "max",
        "partition": "equal_count",
        "score": "raw",
    },
    "qquantile1_mass": {
        "seed": "score_quantile",
        "probe_offsets": [0],
        "partition": "equal_mass",
        "score": "raw",
    },
    "qquantile2_mass_mean": {
        "seed": "score_quantile",
        "probe_offsets": [-1, 0],
        "probe_reduce": "mean",
        "partition": "equal_mass",
        "score": "raw",
    },
    # Cluster the full four-probe response of each key. This reuses the same
    # seed-key logits as Hard C4 and adds only four-dimensional distances.
    "profile_raw_uniform": {
        "seed": "wide",
        "score": "key_softmax",
        "partition_rule": "profile",
        "profile_scale": "raw",
        "profile_merge": "uniform",
    },
    "profile_raw_mean": {
        "seed": "wide",
        "score": "key_softmax",
        "partition_rule": "profile",
        "profile_scale": "raw",
        "profile_merge": "mean",
    },
    "profile_raw_owner": {
        "seed": "wide",
        "score": "key_softmax",
        "partition_rule": "profile",
        "profile_scale": "raw",
        "profile_merge": "owner",
    },
    "profile_raw_mix_0p5": {
        "seed": "wide",
        "score": "key_softmax",
        "partition_rule": "profile",
        "profile_scale": "raw",
        "profile_merge": "mix",
        "profile_mix_temperature": 0.5,
    },
    "profile_raw_mix_1p0": {
        "seed": "wide",
        "score": "key_softmax",
        "partition_rule": "profile",
        "profile_scale": "raw",
        "profile_merge": "mix",
        "profile_mix_temperature": 1.0,
    },
    "profile_zscore_uniform": {
        "seed": "wide",
        "score": "key_softmax",
        "partition_rule": "profile",
        "profile_scale": "zscore",
        "profile_merge": "uniform",
    },
    "profile_zscore_mean": {
        "seed": "wide",
        "score": "key_softmax",
        "partition_rule": "profile",
        "profile_scale": "zscore",
        "profile_merge": "mean",
    },
    "profile_zscore_owner": {
        "seed": "wide",
        "score": "key_softmax",
        "partition_rule": "profile",
        "profile_scale": "zscore",
        "profile_merge": "owner",
    },
    "profile_zscore_mix_0p5": {
        "seed": "wide",
        "score": "key_softmax",
        "partition_rule": "profile",
        "profile_scale": "zscore",
        "profile_merge": "mix",
        "profile_mix_temperature": 0.5,
    },
    "profile_zscore_mix_1p0_lloyd1": {
        "seed": "wide",
        "score": "key_softmax",
        "partition_rule": "profile",
        "profile_scale": "zscore",
        "profile_lloyd_steps": 1,
        "profile_merge": "mix",
        "profile_mix_temperature": 1.0,
    },
    "qsample15_s16_anchor_left_bias_1p05": {
        "seed": "q_sampled_key_cover",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "proxy_temporal_samples": 4,
        "proxy_spatial_samples": 4,
        "proxy_score": "key_softmax",
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qfarthest15_anchor_left_merge_temp_0p9": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
        "merge_temperature": 0.9,
    },
    "qfarthest15_anchor_left_merge_temp_1p1": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
        "merge_temperature": 1.1,
    },
    "qfarthest15_anchor_left_center_blend_0p1": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
        "merge_anchor_blend": 0.1,
        "merge_anchor_source": "center",
    },
    "qfarthest15_anchor_left_center_blend_0p25": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
        "merge_anchor_blend": 0.25,
        "merge_anchor_source": "center",
    },
    "qfarthest15_anchor_left_pair_blend_0p1": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
        "merge_anchor_blend": 0.1,
        "merge_anchor_source": "pair",
    },
    "qfarthest15_anchor_left_pair_blend_0p25": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
        "merge_anchor_blend": 0.25,
        "merge_anchor_source": "pair",
    },
    "qfarthest15_anchor_left_all_blend_0p1": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
        "merge_anchor_blend": 0.1,
        "merge_anchor_source": "all",
    },
    "qfarthest15_anchor_left_bias_0p95": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
        "bias_scale": 0.95,
    },
    "qfarthest15_anchor_left_bias_1p05": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qfarthest15_anchor_left_bias_1p1": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
        "bias_scale": 1.1,
    },
    "qfarthest15_anchor_left_bias_1p025": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
        "bias_scale": 1.025,
    },
    "qfarthest15_anchor_left_bias_1p075": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
        "bias_scale": 1.075,
    },
    "qfarthest17_anchor_left_bias_1p05": {
        "seed": "q_farthest",
        "seed_candidates": 17,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qfarthest15_anchor_left_balanced_bias_1p05": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "seed_selection_sides": ["left", "right"],
        "score": "key_softmax",
        "bias_scale": 1.05,
    },
    "qfarthest15_anchor_left_adaptive_logit_0p1": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
        "adaptive_anchor_blend": 0.1,
        "adaptive_anchor_mode": "logit",
    },
    "qfarthest15_anchor_left_adaptive_logit_0p25": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
        "adaptive_anchor_blend": 0.25,
        "adaptive_anchor_mode": "logit",
    },
    "qfarthest15_anchor_left_adaptive_logit_0p5": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
        "adaptive_anchor_blend": 0.5,
        "adaptive_anchor_mode": "logit",
    },
    "qfarthest15_anchor_left_adaptive_prob_0p05": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
        "adaptive_anchor_blend": 0.05,
        "adaptive_anchor_mode": "probability",
    },
    "qfarthest15_anchor_left_adaptive_prob_0p1": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
        "adaptive_anchor_blend": 0.1,
        "adaptive_anchor_mode": "probability",
    },
    "qfarthest15_anchor_left_adaptive_prob_0p25": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
        "adaptive_anchor_blend": 0.25,
        "adaptive_anchor_mode": "probability",
    },
    "qfarthest15_anchor_left_adaptive_prob_0p5": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
        "adaptive_anchor_blend": 0.5,
        "adaptive_anchor_mode": "probability",
    },
    "qfarthest15_anchor_left_adaptive_prob_raw_0p1": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
        "adaptive_anchor_blend": 0.1,
        "adaptive_anchor_mode": "probability",
        "adaptive_anchor_mass": "raw",
    },
    "qfarthest15_anchor_left_adaptive_prob_raw_0p25": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "seed_anchor_offsets": [-1],
        "score": "key_softmax",
        "adaptive_anchor_blend": 0.25,
        "adaptive_anchor_mode": "probability",
        "adaptive_anchor_mass": "raw",
    },
    "qfarthest13_margin_p0p25": {
        "seed": "q_farthest",
        "seed_candidates": 13,
        "score": "key_softmax",
        "assignment_margin_scale": 0.25,
    },
    "qfarthest13_margin_p0p5": {
        "seed": "q_farthest",
        "seed_candidates": 13,
        "score": "key_softmax",
        "assignment_margin_scale": 0.5,
    },
    "qfarthest13_margin_p0p75": {
        "seed": "q_farthest",
        "seed_candidates": 13,
        "score": "key_softmax",
        "assignment_margin_scale": 0.75,
    },
    "qfarthest13_anchor_intercept": {
        "seed": "q_farthest",
        "seed_candidates": 13,
        "score": "key_softmax",
        "bias_rule": "anchor_mean",
    },
    "qfarthest13_anchor_lse_intercept": {
        "seed": "q_farthest",
        "seed_candidates": 13,
        "score": "key_softmax",
        "merge_rule": "anchor_lse",
        "bias_rule": "anchor_mean",
    },
    "qfarthest13_merge_0p75": {
        "seed": "q_farthest",
        "seed_candidates": 13,
        "score": "key_softmax",
        "merge_temperature": 0.75,
    },
    "qfarthest13_merge_1p25": {
        "seed": "q_farthest",
        "seed_candidates": 13,
        "score": "key_softmax",
        "merge_temperature": 1.25,
    },
    "qfarthest15_margin_p0p25": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "score": "key_softmax",
        "assignment_margin_scale": 0.25,
    },
    "qfarthest15_ownership_p0p25": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "score": "key_softmax",
        "ownership_scale": 0.25,
    },
    "qfarthest15_ownership_p0p5": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "score": "key_softmax",
        "ownership_scale": 0.5,
    },
    "qfarthest15_ownership_p1": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "score": "key_softmax",
        "ownership_scale": 1.0,
    },
    "qfarthest15_ownership_p2": {
        "seed": "q_farthest",
        "seed_candidates": 15,
        "score": "key_softmax",
        "ownership_scale": 2.0,
    },
    "wide_ownership_p0p25": {
        "seed": "wide", "score": "key_softmax", "ownership_scale": 0.25
    },
    "wide_ownership_p0p5": {
        "seed": "wide", "score": "key_softmax", "ownership_scale": 0.5
    },
    "wide_ownership_p1": {
        "seed": "wide", "score": "key_softmax", "ownership_scale": 1.0
    },
    "wide_ownership_p2": {
        "seed": "wide", "score": "key_softmax", "ownership_scale": 2.0
    },
    "wide_raw_soft_0p1": {
        "seed": "wide", "score": "raw", "route": "soft", "route_temperature": 0.1
    },
    "wide_raw_soft_0p25": {
        "seed": "wide", "score": "raw", "route": "soft", "route_temperature": 0.25
    },
    "wide_raw_soft_0p5": {
        "seed": "wide", "score": "raw", "route": "soft", "route_temperature": 0.5
    },
    "wide_raw_soft_1p0": {
        "seed": "wide", "score": "raw", "route": "soft", "route_temperature": 1.0
    },
    "wide_raw_top2_0p25": {
        "seed": "wide", "score": "raw", "route": "top2", "route_temperature": 0.25
    },
    "wide_raw_top2_0p5": {
        "seed": "wide", "score": "raw", "route": "top2", "route_temperature": 0.5
    },
    "wide_centered_soft_0p25": {
        "seed": "wide", "score": "mean_centered", "route": "soft", "route_temperature": 0.25
    },
    "wide_centered_soft_0p5": {
        "seed": "wide", "score": "mean_centered", "route": "soft", "route_temperature": 0.5
    },
    "wide_zscore_soft_0p25": {
        "seed": "wide", "score": "zscore", "route": "soft", "route_temperature": 0.25
    },
    "wide_zscore_soft_0p5": {
        "seed": "wide", "score": "zscore", "route": "soft", "route_temperature": 0.5
    },
    # Score-only changes. All reuse the existing seed-key dot products.
    "temperature_0p5": {"seed": "center_right", "score": "temperature", "temperature": 0.5},
    "temperature_0p75": {"seed": "center_right", "score": "temperature", "temperature": 0.75},
    "temperature_1p5": {"seed": "center_right", "score": "temperature", "temperature": 1.5},
    "temperature_2p0": {"seed": "center_right", "score": "temperature", "temperature": 2.0},
    "normalizer_0p5": {"seed": "center_right", "score": "normalizer_weight", "weight": 0.5},
    "normalizer_0p75": {"seed": "center_right", "score": "normalizer_weight", "weight": 0.75},
    "normalizer_1p25": {"seed": "center_right", "score": "normalizer_weight", "weight": 1.25},
    "normalizer_1p5": {"seed": "center_right", "score": "normalizer_weight", "weight": 1.5},
    "mean_centered": {"seed": "center_right", "score": "mean_centered"},
    "zscore": {"seed": "center_right", "score": "zscore"},
    "sinkhorn2": {"seed": "center_right", "score": "sinkhorn2"},
    "count_balance_0p5": {"seed": "center_right", "score": "count_balance", "strength": 0.5},
}

# Reuse the four existing seed-key score rows to estimate how strongly every
# seed represents each hard cluster.  Mixing those four within-cluster
# distributions adds no seed-key dot products and leaves the route stage
# unchanged.
for mass_mix in (0.4, 0.5, 0.6):
    mix_label = str(mass_mix).replace(".", "p")
    mass_config = {
        "adaptive_anchor_blend": mass_mix,
        "adaptive_anchor_mode": "probability",
        "adaptive_anchor_mass": "raw",
    }
    VARIANTS[f"hard_c4_massmix_{mix_label}"] = {
        **VARIANTS["baseline"],
        **mass_config,
    }
    VARIANTS[f"symmetric_c4_massmix_{mix_label}"] = {
        **VARIANTS["seed_wider_symmetric"],
        **mass_config,
    }
for uniform_mix in (0.1, 0.2, 0.3):
    mix_label = str(uniform_mix).replace(".", "p")
    uniform_config = {
        "fixed_anchor_blend": uniform_mix,
        "adaptive_anchor_mode": "probability",
    }
    VARIANTS[f"hard_c4_uniformmix_{mix_label}"] = {
        **VARIANTS["baseline"],
        **uniform_config,
    }
    VARIANTS[f"symmetric_c4_uniformmix_{mix_label}"] = {
        **VARIANTS["seed_wider_symmetric"],
        **uniform_config,
    }

for label, power in {
    "0p25": 0.25,
    "0p5": 0.5,
    "0p75": 0.75,
    "1p0": 1.0,
}.items():
    VARIANTS[f"qresponse_keynorm_{label}"] = {
        **VARIANTS["qresponse15_s16_anchor_left_logprob"],
        "merged_key_norm_power": power,
    }

for label, value in {
    "m2p0": -2.0,
    "m1p0": -1.0,
    "m0p5": -0.5,
    "p0p5": 0.5,
    "p1p0": 1.0,
    "p2p0": 2.0,
}.items():
    VARIANTS[f"qresponse_vnorm_{label}"] = {
        **VARIANTS["qresponse15_s16_anchor_left_logprob"],
        "merge_value_norm_scale": value,
    }

for candidates, samples, proxy_mode, response_scale in (
    (15, 8, "sample", "raw"),
    (15, 8, "sample", "logprob"),
    (15, 16, "sample", "probability"),
    (21, 8, "sample", "probability"),
    (21, 16, "sample", "probability"),
    (21, 32, "sample", "probability"),
    (31, 8, "sample", "probability"),
    (31, 16, "sample", "probability"),
    (45, 8, "sample", "probability"),
    (15, 8, "mean", "probability"),
    (21, 8, "mean", "probability"),
):
    scale_name = {"raw": "raw", "logprob": "logprob", "probability": "prob"}[
        response_scale
    ]
    name = f"qblock_response{candidates}_s{samples}_span_{scale_name}"
    if proxy_mode == "mean":
        name += "_mean"
    VARIANTS[name] = {
        "seed": "q_block_response",
        "seed_candidates": candidates,
        "block_proxy_samples": samples,
        "block_proxy_mode": proxy_mode,
        "block_response_method": "span",
        "response_scale": response_scale,
        "score": "key_softmax",
        "bias_scale": 1.05,
    }

for suffix, changes in {
    "free": {"force_center": False},
    "sqrtprob": {"response_scale": "sqrt_probability"},
    "temp0p5": {"response_temperature": 0.5},
    "temp2p0": {"response_temperature": 2.0},
    "topk": {"block_proxy_mode": "topk"},
    "mixed_topk": {"block_proxy_mode": "mixed_topk"},
    "topk_free": {"block_proxy_mode": "topk", "force_center": False},
    "mixed_topk_free": {
        "block_proxy_mode": "mixed_topk",
        "force_center": False,
    },
}.items():
    VARIANTS[f"qblock_response21_s16_span_prob_{suffix}"] = {
        **VARIANTS["qblock_response21_s16_span_prob"],
        **changes,
    }

for candidates, samples, proxy_mode in (
    (21, 16, "anchor_topk"),
    (21, 16, "anchor_mixed_topk"),
    (31, 16, "topk"),
    (31, 16, "anchor_topk"),
    (31, 16, "anchor_mixed_topk"),
    (45, 16, "topk"),
    (45, 16, "anchor_topk"),
    (45, 16, "anchor_mixed_topk"),
    (45, 32, "anchor_topk"),
):
    VARIANTS[
        f"qblock_response{candidates}_s{samples}_span_prob_{proxy_mode}"
    ] = {
        "seed": "q_block_response",
        "seed_candidates": candidates,
        "block_proxy_samples": samples,
        "block_proxy_mode": proxy_mode,
        "block_response_method": "span",
        "response_scale": "probability",
        "score": "key_softmax",
        "bias_scale": 1.05,
    }

VARIANTS["qblock_response45_s16_span_prob_anchor_topk_anchor_left"] = {
    **VARIANTS["qblock_response45_s16_span_prob_anchor_topk"],
    "seed_anchor_offsets": [-1],
}
VARIANTS["qblock_response21_s80_span_prob"] = {
    **VARIANTS["qblock_response21_s16_span_prob"],
    "block_proxy_samples": 80,
}

for candidates, response_scale, force_center, anchor_left in (
    (21, "probability", True, False),
    (31, "probability", True, False),
    (45, "probability", True, False),
    (45, "raw", True, False),
    (45, "logprob", True, False),
    (45, "probability", False, False),
    (45, "probability", True, True),
):
    suffix = f"qpilot{candidates}_span_{response_scale}"
    if not force_center:
        suffix += "_free"
    if anchor_left:
        suffix += "_anchor_left"
    VARIANTS[suffix] = {
        "seed": "q_pilot_refine",
        "seed_candidates": candidates,
        "pilot_response_scale": response_scale,
        "force_center": force_center,
        "seed_anchor_offsets": [-1] if anchor_left else [],
        "score": "key_softmax",
        "bias_scale": 1.05,
    }

for base_name in (
    "seed_wide",
    "qblock_response31_s16_span_prob_anchor_mixed_topk",
):
    for normalize, scales in (
        ("raw", (0.1, 0.25, 0.5, 1.0)),
        ("zscore", (0.025, 0.05, 0.1, 0.25)),
    ):
        for value in scales:
            label = str(value).replace(".", "p")
            VARIANTS[f"{base_name}_vreassign_{normalize}_{label}"] = {
                **VARIANTS[base_name],
                "value_reassign_scale": value,
                "value_reassign_normalize": normalize,
            }

for candidates, proxy_mode, feature, center in (
    (21, "anchor_mixed_topk", "value_output", "candidates"),
    (31, "anchor_mixed_topk", "value_output", "candidates"),
    (45, "anchor_mixed_topk", "value_output", "candidates"),
    (31, "anchor_topk", "value_output", "candidates"),
    (31, "anchor_mixed_topk", "prob_value", "candidates"),
    (31, "anchor_mixed_topk", "value_output", "features"),
):
    name = f"qblock_response{candidates}_s16_span_{feature}_{proxy_mode}_{center}"
    VARIANTS[name] = {
        "seed": "q_block_response",
        "seed_candidates": candidates,
        "block_proxy_samples": 16,
        "block_proxy_mode": proxy_mode,
        "block_response_method": "span",
        "response_scale": "probability",
        "response_feature": feature,
        "span_center": center,
        "score": "key_softmax",
        "bias_scale": 1.05,
    }

for metric in ("cosine", "sqdist"):
    for value in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        label = str(value).replace(".", "p")
        VARIANTS[f"qblock31_vreassign_{metric}_zscore_{label}"] = {
            **VARIANTS["qblock_response31_s16_span_prob_anchor_mixed_topk"],
            "value_reassign_scale": value,
            "value_reassign_normalize": "zscore",
            "value_reassign_metric": metric,
        }

for value in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4):
    label = str(value).replace(".", "p")
    VARIANTS[f"qblock31_vreassign_sqdist_fine_{label}"] = {
        **VARIANTS["qblock_response31_s16_span_prob_anchor_mixed_topk"],
        "value_reassign_scale": value,
        "value_reassign_normalize": "zscore",
        "value_reassign_metric": "sqdist",
    }

value_output_seed = "qblock_response31_s16_span_value_output_anchor_mixed_topk_features"
for value in (0.1, 0.2, 0.25, 0.3):
    label = str(value).replace(".", "p")
    VARIANTS[f"qblock31_value_output_vreassign_{label}"] = {
        **VARIANTS[value_output_seed],
        "value_reassign_scale": value,
        "value_reassign_normalize": "zscore",
        "value_reassign_metric": "sqdist",
    }

for value in (0.025, 0.05, 0.1, 0.2, 0.3):
    label = str(value).replace(".", "p")
    VARIANTS[f"qblock31_center_merge_{label}"] = {
        **VARIANTS["qblock_response31_s16_span_prob_anchor_mixed_topk"],
        "merge_anchor_blend": value,
        "merge_anchor_source": "center",
    }
    VARIANTS[f"qblock31_center_merge_vreassign_{label}"] = {
        **VARIANTS[f"qblock31_center_merge_{label}"],
        "value_reassign_scale": 0.25,
        "value_reassign_normalize": "zscore",
        "value_reassign_metric": "sqdist",
    }

for method, force_center, anchor_left in (
    ("farthest", True, False),
    ("farthest", False, False),
    ("farthest", True, True),
):
    suffix = method
    if not force_center:
        suffix += "_free"
    if anchor_left:
        suffix += "_anchor_left"
    VARIANTS[f"qblock31_value_output_{suffix}"] = {
        **VARIANTS[value_output_seed],
        "block_response_method": method,
        "force_center": force_center,
        "seed_anchor_offsets": [-1] if anchor_left else [],
    }

VARIANTS["qblock45_anchor_mixed_topk_anchor_left"] = {
    **VARIANTS["qblock_response45_s16_span_prob_anchor_mixed_topk"],
    "seed_anchor_offsets": [-1],
}
VARIANTS["qblock45_anchor_mixed_topk_free"] = {
    **VARIANTS["qblock_response45_s16_span_prob_anchor_mixed_topk"],
    "force_center": False,
}

for candidates, method, force_center in (
    (21, "span", True),
    (31, "span", True),
    (45, "span", True),
    (31, "farthest", True),
    (31, "farthest", False),
    (45, "farthest", True),
):
    suffix = f"qblock{candidates}_full_value_{method}"
    if not force_center:
        suffix += "_free"
    VARIANTS[suffix] = {
        "seed": "q_block_response",
        "seed_candidates": candidates,
        "block_proxy_samples": COLS,
        "block_proxy_mode": "sample",
        "block_response_method": method,
        "response_scale": "probability",
        "response_feature": "value_output",
        "span_center": "features",
        "force_center": force_center,
        "score": "key_softmax",
        "bias_scale": 1.05,
    }

for mode in ("probability", "logit"):
    for value in (0.25, 0.5, 0.75, 1.0):
        label = str(value).replace(".", "p")
        VARIANTS[f"qblock31_value_reassign_adaptive_{mode}_{label}"] = {
            **VARIANTS["qblock31_value_output_vreassign_0p2"],
            "adaptive_anchor_blend": value,
            "adaptive_anchor_mode": mode,
            "adaptive_anchor_mass": "raw",
        }

for mode, values in (
    ("logit", (0.1, 0.15, 0.2, 0.3, 0.35, 0.4)),
    ("probability", (0.1, 0.15, 0.2, 0.3, 0.35, 0.4)),
):
    for value in values:
        label = str(value).replace(".", "p")
        VARIANTS[f"qblock31_adaptive_fine_{mode}_{label}"] = {
            **VARIANTS["qblock31_value_output_vreassign_0p2"],
            "adaptive_anchor_blend": value,
            "adaptive_anchor_mode": mode,
            "adaptive_anchor_mass": "raw",
        }

for mode in ("probability", "logit"):
    for value in (0.15, 0.25, 0.5):
        label = str(value).replace(".", "p")
        VARIANTS[f"qblock31_adaptive_scoremass_{mode}_{label}"] = {
            **VARIANTS["qblock31_value_output_vreassign_0p2"],
            "adaptive_anchor_blend": value,
            "adaptive_anchor_mode": mode,
            "adaptive_anchor_mass": "score",
        }

for value in (1.0, 1.025, 1.075, 1.1):
    label = str(value).replace(".", "p")
    VARIANTS[f"qblock31_adaptive_logit0p25_bias_{label}"] = {
        **VARIANTS["qblock31_value_reassign_adaptive_logit_0p25"],
        "bias_scale": value,
    }

value_fit_base = "qblock31_adaptive_logit0p25_bias_1p075"
for ridge in (0.001, 0.01, 0.05, 0.1, 0.25, 1.0):
    ridge_label = str(ridge).replace(".", "p")
    VARIANTS[f"qblock31_value_fit_r{ridge_label}"] = {
        **VARIANTS[value_fit_base],
        "value_fit_ridge": ridge,
    }
for ridge_label, ridge in (("0p00001", 1e-5), ("0p0001", 1e-4), ("0p0005", 5e-4)):
    VARIANTS[f"qblock31_value_fit_r{ridge_label}"] = {
        **VARIANTS[value_fit_base],
        "value_fit_ridge": ridge,
    }
for scale_fit in (0.25, 0.5, 0.75):
    scale_label = str(scale_fit).replace(".", "p")
    VARIANTS[f"qblock31_value_fit_r0p05_s{scale_label}"] = {
        **VARIANTS[value_fit_base],
        "value_fit_ridge": 0.05,
        "value_fit_scale": scale_fit,
    }
for scale_fit in (1.25, 1.5):
    scale_label = str(scale_fit).replace(".", "p")
    VARIANTS[f"qblock31_value_fit_r0p001_s{scale_label}"] = {
        **VARIANTS[value_fit_base],
        "value_fit_ridge": 0.001,
        "value_fit_scale": scale_fit,
    }

for affinity_scale in (0.5, 1.0, 2.0, 4.0, 8.0):
    affinity_label = str(affinity_scale).replace(".", "p")
    VARIANTS[f"qblock31_value_affinity_{affinity_label}"] = {
        **VARIANTS[value_output_seed],
        "response_affinity_scale": affinity_scale,
    }
    VARIANTS[f"qblock31_value_affinity_adaptive_{affinity_label}"] = {
        **VARIANTS[value_fit_base],
        "response_affinity_scale": affinity_scale,
    }
    VARIANTS[f"qblock31_value_affinity_fit_{affinity_label}"] = {
        **VARIANTS[value_fit_base],
        "response_affinity_scale": affinity_scale,
        "value_fit_ridge": 0.001,
    }

for partition_mode in ("top2", "soft"):
    for temperature in (0.05, 0.1, 0.2, 0.5, 1.0):
        temperature_label = str(temperature).replace(".", "p")
        name = f"qblock31_{partition_mode}_{temperature_label}"
        VARIANTS[name] = {
            **VARIANTS[value_output_seed],
            "route": partition_mode,
            "route_temperature": temperature,
        }
        VARIANTS[f"{name}_fit"] = {
            **VARIANTS[name],
            "value_fit_ridge": 0.001,
        }
        raw_name = f"qblock31_raw_{partition_mode}_{temperature_label}"
        VARIANTS[raw_name] = {
            **VARIANTS[name],
            "score": "raw",
        }

for logit_scale in (0.5, 0.75, 1.1, 1.2, 1.25, 1.3, 1.5, 2.0):
    scale_label = str(logit_scale).replace(".", "p")
    name = f"qblock31_weight_scale_{scale_label}"
    VARIANTS[name] = {
        **VARIANTS[value_fit_base],
        "merge_logit_scale": logit_scale,
    }
    VARIANTS[f"{name}_fit"] = {
        **VARIANTS[name],
        "value_fit_ridge": 1e-5,
    }
for peak_mix in (0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
    peak_label = str(peak_mix).replace(".", "p")
    name = f"qblock31_peak_mix_{peak_label}"
    VARIANTS[name] = {
        **VARIANTS[value_fit_base],
        "peak_weight_mix": peak_mix,
    }
    VARIANTS[f"{name}_fit"] = {
        **VARIANTS[name],
        "value_fit_ridge": 1e-5,
    }

for topk_count in (4, 8, 12):
    VARIANTS[f"qblock31_topk_weight_{topk_count}_fit"] = {
        **VARIANTS[value_fit_base],
        "topk_weight_count": topk_count,
        "topk_weight_mix": 1.0,
        "value_fit_ridge": 1e-5,
    }
for topk_count in (4, 8):
    for topk_mix in (0.25, 0.5, 0.75):
        mix_label = str(topk_mix).replace(".", "p")
        VARIANTS[f"qblock31_topk_weight_{topk_count}_mix_{mix_label}_fit"] = {
            **VARIANTS[value_fit_base],
            "topk_weight_count": topk_count,
            "topk_weight_mix": topk_mix,
            "value_fit_ridge": 1e-5,
        }
for peak_scale in (0.25, 0.5, 0.75):
    scale_label = str(peak_scale).replace(".", "p")
    for mode in ("entropy", "diffuse"):
        VARIANTS[f"qblock31_peak_{mode}_{scale_label}_fit"] = {
            **VARIANTS[value_fit_base],
            "adaptive_peak_mode": mode,
            "adaptive_peak_scale": peak_scale,
            "value_fit_ridge": 1e-5,
        }

for candidates, proxy_samples, force_center in (
    (31, 16, False),
    (45, 8, True),
    (45, 8, False),
    (45, 12, True),
    (45, 12, False),
    (45, 16, True),
    (45, 16, False),
):
    center_label = "center" if force_center else "free"
    VARIANTS[
        f"qblock_seed_c{candidates}_s{proxy_samples}_{center_label}_peak_fit"
    ] = {
        **VARIANTS[value_fit_base],
        "seed_candidates": candidates,
        "block_proxy_samples": proxy_samples,
        "force_center": force_center,
        "peak_weight_mix": 0.35,
        "value_fit_ridge": 1e-5,
    }

for value_ridge in (1e-4, 1e-3, 1e-2):
    ridge_label = str(value_ridge).replace(".", "p")
    VARIANTS[f"qblock31_peak_0p35_value_r{ridge_label}"] = {
        **VARIANTS[value_fit_base],
        "peak_weight_mix": 0.35,
        "value_fit_ridge": value_ridge,
    }
for value_scale in (0.5, 0.75, 1.25):
    scale_label = str(value_scale).replace(".", "p")
    VARIANTS[f"qblock31_peak_0p35_value_scale_{scale_label}"] = {
        **VARIANTS[value_fit_base],
        "peak_weight_mix": 0.35,
        "value_fit_ridge": 1e-5,
        "value_fit_scale": value_scale,
    }
for logit_scale in (1.05, 1.1, 1.2):
    scale_label = str(logit_scale).replace(".", "p")
    VARIANTS[f"qblock31_peak_0p35_logit_{scale_label}_fit"] = {
        **VARIANTS[value_fit_base],
        "merge_logit_scale": logit_scale,
        "peak_weight_mix": 0.35,
        "value_fit_ridge": 1e-5,
    }
for bias_scale in (1.0, 1.025, 1.05, 1.1, 1.125):
    scale_label = str(bias_scale).replace(".", "p")
    VARIANTS[f"qblock31_peak_0p35_bias_{scale_label}_fit"] = {
        **VARIANTS[value_fit_base],
        "bias_scale": bias_scale,
        "peak_weight_mix": 0.35,
        "value_fit_ridge": 1e-5,
    }
for entropy_mix in (0.0, 0.25, 0.5, 0.75):
    mix_label = str(entropy_mix).replace(".", "p")
    VARIANTS[f"qblock31_peak_0p35_entropy_mix_{mix_label}_fit"] = {
        **VARIANTS[value_fit_base],
        "peak_weight_mix": 0.35,
        "entropy_bias_mix": entropy_mix,
        "value_fit_ridge": 1e-5,
    }

c45_peak_base = "qblock_seed_c45_s16_center_peak_fit"
for proxy_samples in (20, 24, 32):
    VARIANTS[f"qblock_seed_c45_s{proxy_samples}_center_peak_fit"] = {
        **VARIANTS[c45_peak_base],
        "block_proxy_samples": proxy_samples,
    }
for value_ridge in (1e-4, 1e-3):
    ridge_label = str(value_ridge).replace(".", "p")
    VARIANTS[f"qblock_seed_c45_s16_peak_value_r{ridge_label}"] = {
        **VARIANTS[c45_peak_base],
        "value_fit_ridge": value_ridge,
    }
for bias_scale in (1.05, 1.1):
    scale_label = str(bias_scale).replace(".", "p")
    VARIANTS[f"qblock_seed_c45_s16_peak_bias_{scale_label}_fit"] = {
        **VARIANTS[c45_peak_base],
        "bias_scale": bias_scale,
    }
VARIANTS["qblock_seed_c45_s16_anchor_topk_peak_fit"] = {
    **VARIANTS[c45_peak_base],
    "block_proxy_mode": "anchor_topk",
}

for candidates, proxy_samples in (
    (31, 16),
    (31, 24),
    (45, 16),
    (45, 24),
    (45, 32),
):
    VARIANTS[f"qblock_seed_c{candidates}_s{proxy_samples}_sample_peak_fit"] = {
        **VARIANTS[c45_peak_base],
        "seed_candidates": candidates,
        "block_proxy_samples": proxy_samples,
        "block_proxy_mode": "sample",
    }
for seed_pattern in ("contiguous", "wide"):
    VARIANTS[f"seed_{seed_pattern}_peak_fit"] = {
        **VARIANTS[value_fit_base],
        "seed": seed_pattern,
        "peak_weight_mix": 0.35,
        "value_fit_ridge": 1e-4,
    }

c45_s24_peak_base = "qblock_seed_c45_s24_center_peak_fit"
VARIANTS["qblock_seed_c45_s24_peak_bias_1p1_fit"] = {
    **VARIANTS[c45_s24_peak_base],
    "bias_scale": 1.1,
}
VARIANTS["qblock_seed_c45_s24_peak_entropy_0p5_fit"] = {
    **VARIANTS[c45_s24_peak_base],
    "entropy_bias_mix": 0.5,
}
VARIANTS["qblock_seed_c45_s24_peak_entropy_0p5_bias_1p1_fit"] = {
    **VARIANTS[c45_s24_peak_base],
    "entropy_bias_mix": 0.5,
    "bias_scale": 1.1,
}
VARIANTS["qblock31_low_flop_proxy_peak_fit"] = {
    **VARIANTS["qblock31_peak_mix_0p35_fit"],
    "low_flop_proxy_selection": True,
}
VARIANTS["qblock_seed_c45_s16_low_flop_proxy_peak_fit"] = {
    **VARIANTS[c45_peak_base],
    "low_flop_proxy_selection": True,
}
VARIANTS["qblock_seed_c45_s24_low_flop_proxy_peak_fit"] = {
    **VARIANTS[c45_s24_peak_base],
    "low_flop_proxy_selection": True,
}
VARIANTS["qblock31_low_flop_proxy_peak_entropy_0p5_fit"] = {
    **VARIANTS["qblock31_peak_0p35_entropy_mix_0p5_fit"],
    "low_flop_proxy_selection": True,
}
VARIANTS["qblock_seed_c45_s24_low_flop_peak_bias_1p1_fit"] = {
    **VARIANTS["qblock_seed_c45_s24_peak_bias_1p1_fit"],
    "low_flop_proxy_selection": True,
}
VARIANTS["qblock_seed_c45_s24_low_flop_peak_entropy_0p5_fit"] = {
    **VARIANTS["qblock_seed_c45_s24_peak_entropy_0p5_fit"],
    "low_flop_proxy_selection": True,
}
VARIANTS["qblock_seed_c45_s24_low_flop_peak_entropy_0p5_bias_1p1_fit"] = {
    **VARIANTS["qblock_seed_c45_s24_peak_entropy_0p5_bias_1p1_fit"],
    "low_flop_proxy_selection": True,
}
for top_count in (16, 20):
    VARIANTS[f"qblock_seed_c45_s24_low_flop_top{top_count}_peak_fit"] = {
        **VARIANTS["qblock_seed_c45_s24_low_flop_proxy_peak_fit"],
        "proxy_top_count": top_count,
    }
VARIANTS["qblock_seed_c45_s24_low_flop_all_top_peak_fit"] = {
    **VARIANTS["qblock_seed_c45_s24_low_flop_proxy_peak_fit"],
    "block_proxy_mode": "anchor_topk",
}
for label, offsets in (
    ("medium", [-4, -1, 1, 4]),
    ("wide", [-8, -3, 3, 8]),
    ("wider", [-12, -4, 4, 12]),
):
    VARIANTS[f"qblock_seed_c45_s24_low_flop_anchor_{label}_peak_fit"] = {
        **VARIANTS["qblock_seed_c45_s24_low_flop_proxy_peak_fit"],
        "proxy_anchor_offsets": offsets,
        "proxy_anchor_boundary": "shift",
    }
for label, indices in (
    ("all", [0, 1, 2, 3]),
    ("distinct", [0, 2, 3]),
):
    VARIANTS[f"qblock_seed_c45_s24_low_flop_anchor_fit_{label}"] = {
        **VARIANTS["qblock_seed_c45_s24_low_flop_peak_entropy_0p5_fit"],
        "value_fit_proxy_anchor_indices": indices,
    }
for anchor_weight in (0.25, 0.5, 0.75, 1.5, 2.0):
    weight_label = str(anchor_weight).replace(".", "p")
    VARIANTS[f"qblock_seed_c45_s24_low_flop_anchor_fit_w{weight_label}"] = {
        **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_all"],
        "value_fit_proxy_anchor_weight": anchor_weight,
    }
for centroid_source in ("seed", "proxy"):
    for centroid_weight in (0.25, 0.5, 1.0):
        weight_label = str(centroid_weight).replace(".", "p")
        VARIANTS[
            f"qblock_seed_c45_s24_low_flop_anchor_fit_w0p25_"
            f"{centroid_source}_centroid_w{weight_label}"
        ] = {
            **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_w0p25"],
            f"value_fit_{centroid_source}_centroid_weight": centroid_weight,
        }
VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_w0p25_both_centroids"] = {
    **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_w0p25"],
    "value_fit_seed_centroid_weight": 0.25,
    "value_fit_proxy_centroid_weight": 0.25,
}
seed_medoid_base = "qblock_seed_c45_s24_low_flop_anchor_fit_w0p25"
for mass_mix in (0.4, 0.5, 0.6):
    mix_label = str(mass_mix).replace(".", "p")
    VARIANTS[f"qblock_seed_c45_massmix_{mix_label}"] = {
        **VARIANTS[seed_medoid_base],
        "adaptive_anchor_blend": mass_mix,
        "adaptive_anchor_mode": "probability",
        "adaptive_anchor_mass": "raw",
    }
for uniform_mix in (0.1, 0.2, 0.3):
    mix_label = str(uniform_mix).replace(".", "p")
    VARIANTS[f"qblock_seed_c45_uniformmix_{mix_label}"] = {
        **VARIANTS[seed_medoid_base],
        "fixed_anchor_blend": uniform_mix,
        "adaptive_anchor_mode": "probability",
    }
VARIANTS["qblock_seed_c45_s24_low_flop_seed_medoid"] = {
    **VARIANTS[seed_medoid_base],
    "force_center": False,
    "initial_seed": "medoid",
}
VARIANTS["qblock_seed_c45_s24_low_flop_normalized_center"] = {
    **VARIANTS[seed_medoid_base],
    "normalize_seed_features": True,
}
VARIANTS["qblock_seed_c45_s24_low_flop_normalized_medoid"] = {
    **VARIANTS[seed_medoid_base],
    "force_center": False,
    "initial_seed": "medoid",
    "normalize_seed_features": True,
}
for neighbor_offset in (-1, -2):
    offset_label = f"m{abs(neighbor_offset)}"
    for neighbor_weight in (0.25, 0.5, 1.0):
        weight_label = str(neighbor_weight).replace(".", "p")
        VARIANTS[
            f"qblock_seed_c45_s24_low_flop_anchor_fit_w0p25_"
            f"neighbor_{offset_label}_w{weight_label}"
        ] = {
            **VARIANTS[seed_medoid_base],
            "return_seed_indices": True,
            "value_fit_neighbor_offsets": [neighbor_offset],
            "value_fit_neighbor_weight": neighbor_weight,
        }
for uniform_count in (2, 4):
    for uniform_weight in (0.1, 0.25, 0.5, 1.0):
        weight_label = str(uniform_weight).replace(".", "p")
        VARIANTS[
            f"qblock_seed_c45_s24_low_flop_anchor_fit_w0p25_"
            f"uniform{uniform_count}_w{weight_label}"
        ] = {
            **VARIANTS[seed_medoid_base],
            "value_fit_uniform_query_count": uniform_count,
            "value_fit_uniform_query_weight": uniform_weight,
        }
for proxy_mix in (0.05, 0.1, 0.2, 0.3, 0.5):
    mix_label = str(proxy_mix).replace(".", "p")
    VARIANTS[f"qblock_seed_c45_s24_low_flop_proxy_weight_mix_{mix_label}"] = {
        **VARIANTS[seed_medoid_base],
        "proxy_anchor_weight_mix": proxy_mix,
    }
    VARIANTS[f"qblock_seed_c45_s24_low_flop_proxy_value_mix_{mix_label}"] = {
        **VARIANTS[seed_medoid_base],
        "proxy_anchor_value_mix": proxy_mix,
    }
for normalizer_weight in (0.5, 0.75, 0.9, 1.1, 1.25, 1.5):
    weight_label = str(normalizer_weight).replace(".", "p")
    VARIANTS[f"qblock_seed_c45_s24_low_flop_normalizer_{weight_label}"] = {
        **VARIANTS[seed_medoid_base],
        "score": "normalizer_weight",
        "weight": normalizer_weight,
    }
VARIANTS["qblock_seed_c45_s24_low_flop_multiscale_c31_c45"] = {
    **VARIANTS[seed_medoid_base],
    "multiscale_local_candidates": 31,
}
VARIANTS["qblock_seed_c45_s24_low_flop_free"] = {
    **VARIANTS[seed_medoid_base],
    "force_center": False,
}
VARIANTS["qblock_seed_c45_s24_low_flop_anchor_or_free_pooled4"] = {
    **VARIANTS[seed_medoid_base],
    "multiscale_local_candidates": 45,
    "multiscale_alternative_force_center": False,
    "proxy_objective_value_dimensions": 4,
}
for pooled_dimensions in (4, 8, 16):
    VARIANTS[
        f"qblock_seed_c45_s24_low_flop_multiscale_pooled{pooled_dimensions}"
    ] = {
        **VARIANTS["qblock_seed_c45_s24_low_flop_multiscale_c31_c45"],
        "proxy_objective_value_dimensions": pooled_dimensions,
    }
for objective in ("attention_l2", "attention_kl"):
    VARIANTS[f"qblock_seed_c45_s24_low_flop_multiscale_{objective}"] = {
        **VARIANTS["qblock_seed_c45_s24_low_flop_multiscale_c31_c45"],
        "proxy_selection_objective": objective,
    }
for qk_dimensions, value_dimensions in (
    (16, 8),
    (32, 8),
    (32, 16),
    (64, 16),
    (64, 32),
):
    VARIANTS[
        f"qblock_seed_c45_full80_d{qk_dimensions}_v{value_dimensions}"
    ] = {
        **VARIANTS[seed_medoid_base],
        "block_proxy_samples": COLS,
        "block_proxy_mode": "dimension_sketch",
        "qk_sketch_dimensions": qk_dimensions,
        "response_value_dimensions": value_dimensions,
    }
for value_similarity_scale in (
    -0.1,
    -0.05,
    0.05,
    0.1,
    0.2,
    0.3,
    0.5,
    0.75,
    1.0,
):
    scale_label = (
        str(value_similarity_scale).replace("-", "m").replace(".", "p")
    )
    VARIANTS[f"qblock_seed_c45_merge_vsim_{scale_label}"] = {
        **VARIANTS[seed_medoid_base],
        "merge_value_similarity_scale": value_similarity_scale,
    }
for refine_steps in (1, 2):
    for value_similarity_scale in (0.1, 0.2, 0.3, 0.5):
        scale_label = str(value_similarity_scale).replace(".", "p")
        VARIANTS[
            f"qblock_seed_c45_merge_vrefine{refine_steps}_{scale_label}"
        ] = {
            **VARIANTS[seed_medoid_base],
            "merge_value_refine_steps": refine_steps,
            "merge_value_refine_scale": value_similarity_scale,
        }
VARIANTS["qblock_seed_c45_merge_vrefine1_0p2_logit0p98"] = {
    **VARIANTS[seed_medoid_base],
    "merge_logit_scale": 0.98,
    "merge_value_refine_steps": 1,
    "merge_value_refine_scale": 0.2,
}
for route_refine_mix in (0.25, 0.5, 0.75, 1.0):
    mix_label = str(route_refine_mix).replace(".", "p")
    VARIANTS[f"hard_c4_route_refine_{mix_label}"] = {
        **VARIANTS["baseline"],
        "merge_route_refine_mix": route_refine_mix,
    }
    VARIANTS[f"qblock_seed_c45_route_refine_{mix_label}"] = {
        **VARIANTS[seed_medoid_base],
        "merge_route_refine_mix": route_refine_mix,
    }
for value_route_refine_mix in (0.5, 0.75, 1.0):
    mix_label = str(value_route_refine_mix).replace(".", "p")
    VARIANTS[f"hard_c4_value_route_refine_{mix_label}"] = {
        **VARIANTS["baseline"],
        "value_route_refine_mix": value_route_refine_mix,
    }
    VARIANTS[f"qblock_seed_c45_value_route_refine_{mix_label}"] = {
        **VARIANTS[seed_medoid_base],
        "value_route_refine_mix": value_route_refine_mix,
    }
for seed_name, seed_base in (
    ("hard_c4", "baseline"),
    ("symmetric_c4", "seed_wider_symmetric"),
    ("sym4_c4", "seed_sym4_shift"),
):
    for value_fit_ridge in (1e-5, 1e-4, 1e-3, 1e-2, 5e-2):
        ridge_label = f"{value_fit_ridge:g}".replace("-", "m").replace(".", "p")
        VARIANTS[f"{seed_name}_seed_value_fit_r{ridge_label}"] = {
            **VARIANTS[seed_base],
            "value_fit_ridge": value_fit_ridge,
        }

# Paper-oriented response-cover ablation: select four real queries by how well
# their attention profiles span a small key sketch, then keep the merge rule
# limited to optional peak sharpening and seed-output value interpolation.
for proxy_samples in (16, 24):
    for response_feature in ("attention", "value_output"):
        feature_label = "attn" if response_feature == "attention" else "value"
        name = f"response_cover_s{proxy_samples}_{feature_label}"
        VARIANTS[name] = {
            "seed": "q_block_response",
            "seed_candidates": ROWS,
            "block_proxy_samples": proxy_samples,
            "block_proxy_mode": "anchor_mixed_topk",
            "block_response_method": "span",
            "response_scale": "probability",
            "response_feature": response_feature,
            "span_center": "features",
            "score": "key_softmax",
            "force_center": True,
            "low_flop_proxy_selection": True,
        }

response_cover_base = "response_cover_s24_value"
VARIANTS["response_cover_peak"] = {
    **VARIANTS[response_cover_base],
    "peak_weight_mix": 0.35,
}
VARIANTS["response_cover_value_fit"] = {
    **VARIANTS[response_cover_base],
    "value_fit_ridge": 1e-5,
}
VARIANTS["response_cover_peak_value_fit"] = {
    **VARIANTS["response_cover_peak"],
    "value_fit_ridge": 1e-5,
}
VARIANTS["response_cover_peak_anchor_fit"] = {
    **VARIANTS["response_cover_peak_value_fit"],
    "value_fit_proxy_anchor_indices": [0, 1, 2, 3],
    "value_fit_proxy_anchor_weight": 0.25,
}
VARIANTS["response_cover_adaptive"] = {
    **VARIANTS[response_cover_base],
    "adaptive_anchor_blend": 0.25,
    "adaptive_anchor_mode": "logit",
    "adaptive_anchor_mass": "raw",
}
VARIANTS["response_cover_adaptive_peak_anchor_fit"] = {
    **VARIANTS["response_cover_peak_anchor_fit"],
    "adaptive_anchor_blend": 0.25,
    "adaptive_anchor_mode": "logit",
    "adaptive_anchor_mass": "raw",
}

response_cover_c31_s16 = {
    **VARIANTS["response_cover_s16_value"],
    "seed_candidates": 31,
}
VARIANTS["response_cover_c31_s16_value"] = response_cover_c31_s16
for peak_weight_mix in (0.25, 0.3, 0.35, 0.4):
    peak_label = str(peak_weight_mix).replace(".", "p")
    VARIANTS[f"response_cover_c31_s16_peak_{peak_label}_fit"] = {
        **response_cover_c31_s16,
        "peak_weight_mix": peak_weight_mix,
        "value_fit_ridge": 1e-5,
    }
    VARIANTS[f"response_cover_c31_s16_peak_{peak_label}_anchor_fit"] = {
        **VARIANTS[f"response_cover_c31_s16_peak_{peak_label}_fit"],
        "value_fit_proxy_anchor_indices": [0, 1, 2, 3],
        "value_fit_proxy_anchor_weight": 0.25,
    }
for anchor_weight in (0.1, 0.5, 1.0):
    weight_label = str(anchor_weight).replace(".", "p")
    VARIANTS[f"response_cover_c31_s16_peak_0p3_anchor_w{weight_label}"] = {
        **VARIANTS["response_cover_c31_s16_peak_0p3_fit"],
        "value_fit_proxy_anchor_indices": [0, 1, 2, 3],
        "value_fit_proxy_anchor_weight": anchor_weight,
    }
for value_fit_ridge in (1e-4, 1e-3):
    ridge_label = f"{value_fit_ridge:g}".replace("-", "m").replace(".", "p")
    VARIANTS[f"response_cover_c31_s16_peak_0p3_anchor_fit_r{ridge_label}"] = {
        **VARIANTS["response_cover_c31_s16_peak_0p3_anchor_fit"],
        "value_fit_ridge": value_fit_ridge,
    }
for sketch_count in (8, 16, 31):
    for sketch_weight in (0.1, 0.25, 0.5):
        weight_label = str(sketch_weight).replace(".", "p")
        VARIANTS[
            f"response_cover_c31_s16_sketch{sketch_count}_w{weight_label}"
        ] = {
            **VARIANTS["response_cover_c31_s16_peak_0p3_anchor_fit"],
            "value_fit_sketch_query_count": sketch_count,
            "value_fit_sketch_weight": sketch_weight,
        }
for adaptive_base, adaptive_prefix in (
    (
        "response_cover_c31_s16_peak_0p3_anchor_fit",
        "response_cover_c31_s16",
    ),
    (
        "qblock_seed_c45_s24_low_flop_anchor_fit_w0p25",
        "response_cover_c45_s24",
    ),
):
    for adaptive_mode in ("entropy", "farthest"):
        for adaptive_weight in (0.05, 0.1, 0.25):
            weight_label = str(adaptive_weight).replace(".", "p")
            VARIANTS[
                f"{adaptive_prefix}_adaptive_{adaptive_mode}_w{weight_label}"
            ] = {
                **VARIANTS[adaptive_base],
                "value_fit_adaptive_query_count": 4,
                "value_fit_adaptive_query_mode": adaptive_mode,
                "value_fit_adaptive_query_weight": adaptive_weight,
            }
for adaptive_count, adaptive_weights in (
    (8, (0.05, 0.1)),
    (16, (0.025, 0.05)),
):
    for adaptive_weight in adaptive_weights:
        weight_label = str(adaptive_weight).replace(".", "p")
        VARIANTS[
            f"response_cover_c45_s24_adaptive_farthest_n{adaptive_count}_w{weight_label}"
        ] = {
            **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_w0p25"],
            "value_fit_adaptive_query_count": adaptive_count,
            "value_fit_adaptive_query_mode": "farthest",
            "value_fit_adaptive_query_weight": adaptive_weight,
        }
for value_unpeak_mix in (0.25, 0.5, 0.75, 1.0):
    mix_label = str(value_unpeak_mix).replace(".", "p")
    VARIANTS[f"response_cover_c31_s16_value_unpeak_{mix_label}"] = {
        **VARIANTS["response_cover_c31_s16_peak_0p3_anchor_fit"],
        "value_unpeak_mix": value_unpeak_mix,
    }
    VARIANTS[f"response_cover_c45_s24_value_unpeak_{mix_label}"] = {
        **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_w0p25"],
        "value_unpeak_mix": value_unpeak_mix,
    }
for logmass_use_proxy, logmass_prefix in (
    (False, "seed"),
    (True, "anchor"),
):
    for logmass_ridge in (0.1, 1.0):
        ridge_label = str(logmass_ridge).replace(".", "p")
        for logmass_scale in (0.1, 0.25, 0.5, 1.0):
            scale_label = str(logmass_scale).replace(".", "p")
            VARIANTS[
                f"response_cover_c45_s24_logmass_{logmass_prefix}_r{ridge_label}_s{scale_label}"
            ] = {
                **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_w0p25"],
                "logmass_key_fit_ridge": logmass_ridge,
                "logmass_key_fit_scale": logmass_scale,
                "logmass_key_fit_use_proxy": logmass_use_proxy,
            }
for prior_use_proxy, prior_prefix in (
    (False, "seed"),
    (True, "anchor"),
):
    for prior_reduce in ("logmeanexp", "mean"):
        for prior_mix in (0.05, 0.1, 0.2, 0.4):
            mix_label = str(prior_mix).replace(".", "p")
            VARIANTS[
                f"response_cover_c45_s24_value_prior_{prior_prefix}_{prior_reduce}_{mix_label}"
            ] = {
                **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_w0p25"],
                "postfit_value_prior_reduce": prior_reduce,
                "postfit_value_prior_mix": prior_mix,
                "postfit_value_prior_use_proxy": prior_use_proxy,
            }
for convex_steps in (1, 2, 4, 8):
    VARIANTS[f"response_cover_c45_s24_value_convex_{convex_steps}"] = {
        **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_w0p25"],
        "value_fit_convex_steps": convex_steps,
    }
for fw_steps, fw_step_scales in (
    (1, (0.1, 0.25, 0.5, 1.0)),
    (2, (0.25, 0.5, 1.0)),
    (4, (0.25, 0.5)),
):
    for fw_step_scale in fw_step_scales:
        scale_label = str(fw_step_scale).replace(".", "p")
        VARIANTS[
            f"response_cover_c45_s24_value_fw{fw_steps}_s{scale_label}"
        ] = {
            **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_w0p25"],
            "value_weight_fw_steps": fw_steps,
            "value_weight_fw_step_scale": fw_step_scale,
        }
for global_probe_count in (1, 2, 4):
    for global_probe_scale in (0.25, 0.5, 1.0):
        scale_label = str(global_probe_scale).replace(".", "p")
        VARIANTS[
            f"response_cover_c45_s24_global_value_probe{global_probe_count}_s{scale_label}"
        ] = {
            **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_w0p25"],
            "global_value_probe_count": global_probe_count,
            "global_value_probe_scale": global_probe_scale,
        }
for global_probe_count in (2, 4):
    for global_probe_scale in (0.25, 0.5, 1.0):
        scale_label = str(global_probe_scale).replace(".", "p")
        VARIANTS[
            f"response_cover_c45_s24_global_value_probe{global_probe_count}_"
            f"exactline_s{scale_label}"
        ] = {
            **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_w0p25"],
            "global_value_probe_count": global_probe_count,
            "global_value_probe_scale": global_probe_scale,
            "global_value_probe_exact_line_search": True,
        }
for probe_ridge in (1e-4, 1e-2):
    ridge_label = f"{probe_ridge:g}".replace("-", "m").replace(".", "p")
    for global_probe_scale in (0.25, 0.5, 1.0):
        scale_label = str(global_probe_scale).replace(".", "p")
        VARIANTS[
            f"response_cover_c45_s24_global_value_probe4_ls_r{ridge_label}_"
            f"s{scale_label}"
        ] = {
            **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_w0p25"],
            "global_value_probe_count": 4,
            "global_value_probe_scale": global_probe_scale,
            "global_value_probe_solver": "least_squares",
            "global_value_probe_ridge": probe_ridge,
        }
for probe_layout in ("interior", "endpoints", "midpoints", "segment_mean"):
    VARIANTS[
        f"response_cover_c45_s24_global_value_probe4_ls_{probe_layout}_s0p65"
    ] = {
        **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_w0p25"],
        "global_value_probe_count": 4,
        "global_value_probe_scale": 0.65,
        "global_value_probe_solver": "least_squares",
        "global_value_probe_ridge": 1e-4,
        "global_value_probe_layout": probe_layout,
    }
for global_probe_count in (4, 6, 8):
    for global_probe_scale in (0.5, 0.65):
        scale_label = str(global_probe_scale).replace(".", "p")
        VARIANTS[
            f"response_cover_c45_s24_global_value_probe{global_probe_count}_"
            f"ls_stream_s{scale_label}"
        ] = {
            **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_w0p25"],
            "global_value_probe_count": global_probe_count,
            "global_value_probe_scale": global_probe_scale,
            "global_value_probe_solver": "least_squares",
            "global_value_probe_ridge": 1e-4,
            "global_value_probe_chunk": 2,
        }
for global_probe_count in (10, 12, 16):
    VARIANTS[
        f"response_cover_c45_s24_global_value_probe{global_probe_count}_"
        "ls_stream_s0p65"
    ] = {
        **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_w0p25"],
        "global_value_probe_count": global_probe_count,
        "global_value_probe_scale": 0.65,
        "global_value_probe_solver": "least_squares",
        "global_value_probe_ridge": 1e-4,
        "global_value_probe_chunk": 2,
    }
for global_probe_count in (8, 12, 16):
    VARIANTS[
        f"response_cover_c45_s24_global_value_probe{global_probe_count}_"
        "ls_query_farthest_s0p65"
    ] = {
        **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_w0p25"],
        "global_value_probe_count": global_probe_count,
        "global_value_probe_scale": 0.65,
        "global_value_probe_solver": "least_squares",
        "global_value_probe_ridge": 1e-4,
        "global_value_probe_chunk": 2,
        "global_value_probe_layout": "query_farthest",
    }
for probe_layout in ("interior", "midpoints"):
    for global_probe_scale in (0.5, 0.65, 0.8, 1.0):
        scale_label = str(global_probe_scale).replace(".", "p")
        VARIANTS[
            f"response_cover_c45_s24_global_value_probe16_ls_{probe_layout}_"
            f"tune_s{scale_label}"
        ] = {
            **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_w0p25"],
            "global_value_probe_count": 16,
            "global_value_probe_scale": global_probe_scale,
            "global_value_probe_solver": "least_squares",
            "global_value_probe_ridge": 1e-4,
            "global_value_probe_chunk": 2,
            "global_value_probe_layout": probe_layout,
        }
for global_probe_scale in (0.8, 0.9, 1.0):
    scale_label = str(global_probe_scale).replace(".", "p")
    VARIANTS[
        f"response_cover_c45_s24_global_value_probe16_ls_query_farthest_"
        f"tune_s{scale_label}"
    ] = {
        **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_w0p25"],
        "global_value_probe_count": 16,
        "global_value_probe_scale": global_probe_scale,
        "global_value_probe_solver": "least_squares",
        "global_value_probe_ridge": 1e-4,
        "global_value_probe_chunk": 2,
        "global_value_probe_layout": "query_farthest",
    }
for global_probe_count in (8, 12, 16):
    VARIANTS[
        f"response_cover_c45_s24_global_value_probe{global_probe_count}_"
        "ls_query_pivoted_s0p8"
    ] = {
        **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_w0p25"],
        "global_value_probe_count": global_probe_count,
        "global_value_probe_scale": 0.8,
        "global_value_probe_solver": "least_squares",
        "global_value_probe_ridge": 1e-4,
        "global_value_probe_chunk": 2,
        "global_value_probe_layout": "query_pivoted",
    }
for probe_layout in ("query_pivoted", "route_pivoted"):
    for global_probe_scale in (0.8, 0.9, 1.0):
        scale_label = str(global_probe_scale).replace(".", "p")
        VARIANTS[
            f"response_cover_c45_s24_global_value_probe16_ls_{probe_layout}_"
            f"tune_s{scale_label}"
        ] = {
            **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_w0p25"],
            "global_value_probe_count": 16,
            "global_value_probe_scale": global_probe_scale,
            "global_value_probe_solver": "least_squares",
            "global_value_probe_ridge": 1e-4,
            "global_value_probe_chunk": 2,
            "global_value_probe_layout": probe_layout,
        }
for local_key_rows in (2, 4, 8, 12):
    for global_probe_scale in (0.8, 1.0):
        scale_label = str(global_probe_scale).replace(".", "p")
        VARIANTS[
            f"response_cover_c45_s24_global_value_probe16_ls_"
            f"local{local_key_rows}_s{scale_label}"
        ] = {
            **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_w0p25"],
            "global_value_probe_count": 16,
            "global_value_probe_scale": global_probe_scale,
            "global_value_probe_solver": "least_squares",
            "global_value_probe_ridge": 1e-4,
            "global_value_probe_layout": "midpoints",
            "global_value_local_key_rows": local_key_rows,
        }
for top_key_rows in (2, 4, 8, 12):
    for global_probe_scale in (0.8, 1.0):
        scale_label = str(global_probe_scale).replace(".", "p")
        VARIANTS[
            f"response_cover_c45_s24_global_value_probe16_ls_"
            f"toprow{top_key_rows}_s{scale_label}"
        ] = {
            **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_w0p25"],
            "global_value_probe_count": 16,
            "global_value_probe_scale": global_probe_scale,
            "global_value_probe_solver": "least_squares",
            "global_value_probe_ridge": 1e-4,
            "global_value_probe_layout": "midpoints",
            "global_value_top_key_rows": top_key_rows,
        }
for top_key_blocks in (42, 84, 168, 252):
    for global_probe_scale in (0.8, 1.0):
        scale_label = str(global_probe_scale).replace(".", "p")
        VARIANTS[
            f"response_cover_c45_s24_global_value_probe16_ls_"
            f"topblock{top_key_blocks}_s{scale_label}"
        ] = {
            **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_w0p25"],
            "global_value_probe_count": 16,
            "global_value_probe_scale": global_probe_scale,
            "global_value_probe_solver": "least_squares",
            "global_value_probe_ridge": 1e-4,
            "global_value_probe_layout": "midpoints",
            "global_value_top_key_blocks": top_key_blocks,
        }
for probe_count, top_key_blocks in (
    (8, 504),
    (12, 336),
    (24, 168),
    (32, 126),
    (40, 101),
    (45, 90),
):
    VARIANTS[
        f"response_cover_c45_s24_global_value_probe{probe_count}_ls_"
        f"topblock{top_key_blocks}_budget9p5_s0p8"
    ] = {
        **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_w0p25"],
        "global_value_probe_count": probe_count,
        "global_value_probe_scale": 0.8,
        "global_value_probe_solver": "least_squares",
        "global_value_probe_ridge": 1e-4,
        "global_value_probe_layout": "midpoints",
        "global_value_top_key_blocks": top_key_blocks,
    }
for top_key_blocks, budget_label in (
    (135, "14p3"),
    (180, "19p0"),
    (210, "22p2"),
    (225, "23p8"),
    (252, "26p7"),
):
    VARIANTS[
        f"response_cover_c45_s24_global_value_probe45_ls_"
        f"topblock{top_key_blocks}_budget{budget_label}_s0p8"
    ] = {
        **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_w0p25"],
        "global_value_probe_count": 45,
        "global_value_probe_scale": 0.8,
        "global_value_probe_solver": "least_squares",
        "global_value_probe_ridge": 1e-4,
        "global_value_probe_layout": "midpoints",
        "global_value_top_key_blocks": top_key_blocks,
    }
for score_mode in ("value_std", "key_std", "value_key_std"):
    VARIANTS[
        f"response_cover_c45_s24_global_value_probe45_ls_topblock90_"
        f"{score_mode}_budget9p5_s0p8"
    ] = {
        **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_w0p25"],
        "global_value_probe_count": 45,
        "global_value_probe_scale": 0.8,
        "global_value_probe_solver": "least_squares",
        "global_value_probe_ridge": 1e-4,
        "global_value_probe_layout": "midpoints",
        "global_value_top_key_blocks": 90,
        "global_value_top_block_score": score_mode,
    }
for seed_variant in (
    "qblock_seed_c45_s24_low_flop_free",
    "qblock_seed_c45_s24_low_flop_anchor_or_free_pooled4",
):
    VARIANTS[f"{seed_variant}_merge_vsim_0p2"] = {
        **VARIANTS[seed_variant],
        "merge_value_similarity_scale": 0.2,
    }
for query_mean_blend in (0.1, 0.25, 0.5, 0.75, 1.0):
    blend_label = str(query_mean_blend).replace(".", "p")
    VARIANTS[f"qblock_seed_c45_merge_qmean_{blend_label}"] = {
        **VARIANTS[seed_medoid_base],
        "merge_query_mean_blend": query_mean_blend,
    }
for query_mean_blend in (0.25, 0.5):
    blend_label = str(query_mean_blend).replace(".", "p")
    VARIANTS[f"qblock_seed_c45_merge_qmean_{blend_label}_vsim_0p5"] = {
        **VARIANTS[seed_medoid_base],
        "merge_query_mean_blend": query_mean_blend,
        "merge_value_similarity_scale": 0.5,
    }
    VARIANTS[
        f"qblock_seed_c45_anchor_or_free_qmean_{blend_label}_vsim_0p5"
    ] = {
        **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_or_free_pooled4"],
        "merge_query_mean_blend": query_mean_blend,
        "merge_value_similarity_scale": 0.5,
    }
VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_r0p0001"] = {
    **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_all"],
    "value_fit_ridge": 1e-4,
}
for key_ridge in (0.1, 1.0):
    ridge_label = str(key_ridge).replace(".", "p")
    for key_scale in (0.1, 0.25):
        scale_label = str(key_scale).replace(".", "p")
        VARIANTS[
            f"qblock_seed_c45_s24_low_flop_anchor_key_r{ridge_label}_s{scale_label}"
        ] = {
            **VARIANTS["qblock_seed_c45_s24_low_flop_anchor_fit_all"],
            "key_fit_ridge": key_ridge,
            "key_fit_scale": key_scale,
            "key_fit_proxy_anchors": True,
        }

overlap_tangent_base = {
    "seed": "q_block_response",
    "seed_candidates": 45,
    "block_proxy_samples": 24,
    "block_proxy_mode": "anchor_mixed_topk",
    "block_response_method": "span",
    "response_scale": "probability",
    "response_feature": "value_output",
    "span_center": "features",
    "score": "key_softmax",
    "low_flop_proxy_selection": True,
    "overlap_clusters": True,
}
VARIANTS["qblock_seed_c45_s24_overlap_tangent"] = overlap_tangent_base
VARIANTS["qblock_seed_c45_s24_overlap_tangent_fit"] = {
    **overlap_tangent_base,
    "value_fit_ridge": 1e-4,
}
for bias_scale in (1.0, 1.05, 1.1):
    scale_label = str(bias_scale).replace(".", "p")
    VARIANTS[f"qblock_seed_c45_s24_overlap_tangent_anchor_bias_{scale_label}"] = {
        **overlap_tangent_base,
        "bias_scale": bias_scale,
        "value_fit_ridge": 1e-4,
        "value_fit_proxy_anchor_indices": [0, 1, 2, 3],
    }
for peak_mix, logit_scale in (
    (0.2, 0.75),
    (0.2, 1.25),
    (0.3, 0.75),
    (0.3, 1.25),
    (0.4, 0.75),
    (0.4, 1.25),
):
    peak_label = str(peak_mix).replace(".", "p")
    scale_label = str(logit_scale).replace(".", "p")
    VARIANTS[f"qblock31_peak_{peak_label}_scale_{scale_label}_fit"] = {
        **VARIANTS[value_fit_base],
        "peak_weight_mix": peak_mix,
        "merge_logit_scale": logit_scale,
        "value_fit_ridge": 1e-5,
    }

for proxy_samples, proxy_mode, method in (
    (16, "anchor_mixed_topk", "span"),
    (16, "anchor_mixed_topk", "farthest"),
    (COLS, "sample", "span"),
    (COLS, "sample", "farthest"),
):
    proxy_label = "full" if proxy_samples == COLS else "s16"
    for lloyd_steps in (1, 2):
        name = f"qblock31_value_medoid_{proxy_label}_{method}_l{lloyd_steps}"
        VARIANTS[name] = {
            **VARIANTS[value_output_seed],
            "block_proxy_samples": proxy_samples,
            "block_proxy_mode": proxy_mode,
            "block_response_method": method,
            "block_response_lloyd_steps": lloyd_steps,
        }
        VARIANTS[f"{name}_merge"] = {
            **VARIANTS[name],
            "value_reassign_scale": 0.2,
            "value_reassign_normalize": "zscore",
            "value_reassign_metric": "sqdist",
            "adaptive_anchor_blend": 0.25,
            "adaptive_anchor_mode": "logit",
            "adaptive_anchor_mass": "raw",
            "bias_scale": 1.075,
            "value_fit_ridge": 1e-5,
        }

for offset_label, offsets in (
    ("m4", [-4]),
    ("m2", [-2]),
    ("m1", [-1]),
    ("p1", [1]),
    ("p2", [2]),
    ("p4", [4]),
    ("pm1", [-1, 1]),
    ("pm2", [-2, 2]),
    ("pm4", [-4, 4]),
):
    for use_peak in (False, True):
        peak_label = "_peak0p3" if use_peak else ""
        name = f"qblock31_neighbor_fit_{offset_label}{peak_label}"
        VARIANTS[name] = {
            **VARIANTS[value_fit_base],
            "return_seed_indices": True,
            "value_fit_neighbor_offsets": offsets,
            "value_fit_ridge": 1e-4,
        }
        if use_peak:
            VARIANTS[name]["peak_weight_mix"] = 0.3

for ridge in (0.001, 0.01, 0.1):
    ridge_label = str(ridge).replace(".", "p")
    for fit_scale in (0.25, 0.5, 1.0):
        scale_label = str(fit_scale).replace(".", "p")
        name = f"qblock31_bias_fit_r{ridge_label}_s{scale_label}"
        VARIANTS[name] = {
            **VARIANTS[value_fit_base],
            "bias_fit_ridge": ridge,
            "bias_fit_scale": fit_scale,
            "bias_fit_clamp": 2.0,
        }
        VARIANTS[f"{name}_value"] = {
            **VARIANTS[name],
            "peak_weight_mix": 0.35,
            "value_fit_ridge": 1e-5,
        }
VARIANTS["qblock31_bias_fit_r0p01_s0p5_steps2_value"] = {
    **VARIANTS["qblock31_bias_fit_r0p01_s0p5_value"],
    "bias_fit_steps": 2,
}

for ridge in (0.001, 0.01, 0.1):
    ridge_label = str(ridge).replace(".", "p")
    for fit_scale in (0.25, 0.5, 1.0):
        scale_label = str(fit_scale).replace(".", "p")
        name = f"qblock31_key_fit_r{ridge_label}_s{scale_label}"
        VARIANTS[name] = {
            **VARIANTS[value_fit_base],
            "peak_weight_mix": 0.35,
            "key_fit_ridge": ridge,
            "key_fit_scale": fit_scale,
            "value_fit_ridge": 1e-5,
        }
        VARIANTS[f"{name}_mass"] = {
            **VARIANTS[name],
            "bias_fit_ridge": 0.01,
            "bias_fit_scale": 0.5,
            "bias_fit_clamp": 2.0,
        }

for candidates, samples, proxy_mode, objective, force_center in (
    (15, 8, "sample", "mean", True),
    (21, 8, "sample", "mean", True),
    (21, 16, "sample", "mean", True),
    (21, 32, "sample", "mean", True),
    (31, 16, "sample", "mean", True),
    (21, 16, "sample", "max", True),
    (21, 16, "sample", "mean", False),
    (21, 16, "topk", "mean", True),
    (21, 16, "mixed_topk", "mean", True),
    (21, 16, "topk", "mean", False),
):
    name = f"qblock_tangent{candidates}_s{samples}_{proxy_mode}_{objective}"
    if not force_center:
        name += "_free"
    VARIANTS[name] = {
        "seed": "q_block_response",
        "seed_candidates": candidates,
        "block_proxy_samples": samples,
        "block_proxy_mode": proxy_mode,
        "block_response_method": "tangent",
        "tangent_objective": objective,
        "force_center": force_center,
        "response_scale": "raw",
        "score": "key_softmax",
        "bias_scale": 1.05,
    }

VARIANTS["qblock_tangent21_s16_sample_mean_anchor_left"] = {
    **VARIANTS["qblock_tangent21_s16_sample_mean"],
    "seed_anchor_offsets": [-1],
}

for suffix, changes in {
    "merge0p9": {"merge_temperature": 0.9},
    "merge1p1": {"merge_temperature": 1.1},
    "bias1p075": {"bias_scale": 1.075},
    "anchor_lse": {"merge_rule": "anchor_lse"},
    "anchor_geom": {"merge_rule": "anchor_geom"},
    "anchor_mix": {"merge_rule": "anchor_mix"},
    "ownership0p25": {"ownership_scale": 0.25},
    "value_temp0p75": {"value_temperature": 0.75},
    "value_temp1p25": {"value_temperature": 1.25},
}.items():
    VARIANTS[f"qblock21_{suffix}"] = {
        **VARIANTS["qblock_response21_s16_span_prob"],
        **changes,
    }

for label, value in {
    "0p75": 0.75,
    "0p9": 0.9,
    "1p1": 1.1,
    "1p25": 1.25,
}.items():
    VARIANTS[f"qblock_best_merge_{label}"] = {
        **VARIANTS["qblock_response15_s8_span_prob"],
        "merge_temperature": value,
    }

for label, value in {
    "1p0": 1.0,
    "1p025": 1.025,
    "1p075": 1.075,
    "1p1": 1.1,
}.items():
    VARIANTS[f"qblock_best_bias_{label}"] = {
        **VARIANTS["qblock_response15_s8_span_prob"],
        "bias_scale": value,
    }

# Focused sweeps use the best broad seed geometry and preserve C=4/QK cost.
for name, offsets in {
    "seed_offsets_m3_m1_0_p4": [-3, -1, 0, 4],
    "seed_offsets_m4_m2_0_p3": [-4, -2, 0, 3],
    "seed_offsets_m4_m1_0_p3": [-4, -1, 0, 3],
    "seed_offsets_m5_m2_0_p3": [-5, -2, 0, 3],
    "seed_offsets_m5_m1_0_p3": [-5, -1, 0, 3],
    "seed_offsets_m3_m2_0_p4": [-3, -2, 0, 4],
    "seed_offsets_m5_m2_0_p4": [-5, -2, 0, 4],
    "seed_offsets_m5_m1_0_p4": [-5, -1, 0, 4],
    "seed_offsets_m3_m1_0_p5": [-3, -1, 0, 5],
    "seed_offsets_m4_m1_0_p5": [-4, -1, 0, 5],
    "seed_offsets_m4_m2_0_p5": [-4, -2, 0, 5],
    "seed_offsets_m4_0_p1_p4": [-4, 0, 1, 4],
    "seed_offsets_m3_0_p1_p4": [-3, 0, 1, 4],
    "seed_offsets_m5_0_p1_p5": [-5, 0, 1, 5],
}.items():
    VARIANTS[name] = {"seed": "offsets", "offsets": offsets, "score": "key_softmax"}

for stride in (1, 2, 3, 4, 5, 7, 9, 11, 15):
    VARIANTS[f"seed_cyclic_stride_{stride}"] = {
        "seed": "offsets",
        "offsets": [0, stride, 2 * stride, 3 * stride],
        "boundary": "cyclic",
        "score": "key_softmax",
    }

VARIANTS["c4_wide_diagvar_0p1"] = {
    "seed": "wide",
    "score": "key_softmax",
    "query_diag_variance_scale": 0.1,
}

for label, value in {"0p25": 0.25, "0p5": 0.5, "1p0": 1.0, "2p0": 2.0}.items():
    VARIANTS[f"c4_wide_valuecov_{label}"] = {
        "seed": "wide",
        "score": "key_softmax",
        "query_value_covariance_scale": value,
    }
    VARIANTS[f"c4_wide_diagvar0p1_valuecov_{label}"] = {
        "seed": "wide",
        "score": "key_softmax",
        "query_diag_variance_scale": 0.1,
        "query_value_covariance_scale": value,
    }

for label, value in {
    "0p5": 0.5,
    "0p75": 0.75,
    "0p9": 0.9,
    "0p95": 0.95,
    "1p05": 1.05,
    "1p1": 1.1,
    "1p25": 1.25,
    "1p5": 1.5,
}.items():
    VARIANTS[f"sym_bias_{label}"] = {
        "seed": "wider_symmetric",
        "score": "key_softmax",
        "bias_scale": value,
    }

for label, value in {
    "m0p5": -0.5,
    "m0p25": -0.25,
    "m0p1": -0.1,
    "p0p1": 0.1,
    "p0p25": 0.25,
    "p0p5": 0.5,
}.items():
    VARIANTS[f"sym_count_{label}"] = {
        "seed": "wider_symmetric",
        "score": "key_softmax",
        "count_bias_scale": value,
    }

for label, value in {
    "0p75": 0.75,
    "0p85": 0.85,
    "0p9": 0.9,
    "0p95": 0.95,
    "1p05": 1.05,
    "1p1": 1.1,
    "1p15": 1.15,
    "1p25": 1.25,
}.items():
    VARIANTS[f"sym_attention_{label}"] = {
        "seed": "wider_symmetric",
        "score": "key_softmax",
        "attention_scale_factor": value,
    }

for label, value in {
    "0p5": 0.5,
    "0p75": 0.75,
    "1p25": 1.25,
    "1p5": 1.5,
    "2p0": 2.0,
    "4p0": 4.0,
}.items():
    VARIANTS[f"sym_value_temp_{label}"] = {
        "seed": "wider_symmetric",
        "score": "key_softmax",
        "value_temperature": value,
    }

for label, value in {
    "0p1": 0.1,
    "0p25": 0.25,
    "0p5": 0.5,
    "0p75": 0.75,
    "1p0": 1.0,
}.items():
    VARIANTS[f"sym_value_uniform_{label}"] = {
        "seed": "wider_symmetric",
        "score": "key_softmax",
        "value_uniform_mix": value,
    }

VARIANTS.update(
    {
        "sym_bias_1p075": {
            "seed": "wider_symmetric",
            "score": "key_softmax",
            "bias_scale": 1.075,
        },
        "best_geom_bias_1p05": {
            "seed": "offsets",
            "offsets": [-5, -1, 0, 3],
            "score": "key_softmax",
            "bias_scale": 1.05,
        },
        "best_geom_bias_1p075": {
            "seed": "offsets",
            "offsets": [-5, -1, 0, 3],
            "score": "key_softmax",
            "bias_scale": 1.075,
        },
    }
)

for mode in ("logit", "key"):
    for label, value in {
        "0p05": 0.05,
        "0p1": 0.1,
        "0p2": 0.2,
        "0p4": 0.4,
        "0p6": 0.6,
        "0p8": 0.8,
        "1p0": 1.0,
        "1p5": 1.5,
    }.items():
        VARIANTS[f"best_geom_{mode}var_{label}"] = {
            "seed": "offsets",
            "offsets": [-5, -1, 0, 3],
            "score": "key_softmax",
            "variance_bias_mode": mode,
            "variance_bias_scale": value,
        }
        VARIANTS[f"best_geom_bias1p05_{mode}var_{label}"] = {
            "seed": "offsets",
            "offsets": [-5, -1, 0, 3],
            "score": "key_softmax",
            "bias_scale": 1.05,
            "variance_bias_mode": mode,
            "variance_bias_scale": value,
        }


class StopSweep(Exception):
    pass


class ErrorStats:
    def __init__(self):
        self.calls = 0
        self.numel = 0
        self.sqerr = 0.0
        self.ref_sq = 0.0
        self.max_abs = 0.0
        self.cosine_error_sum = 0.0
        self.empty_clusters = 0
        self.total_clusters = 0
        self.cluster_count_sum = 0.0
        self.cluster_count_sq_sum = 0.0

    def add(self, output, dense, cluster_counts):
        output = output.float()
        dense = dense.float()
        difference = output - dense
        self.calls += 1
        self.numel += difference.numel()
        self.sqerr += float(difference.square().sum())
        self.ref_sq += float(dense.square().sum())
        self.max_abs = max(self.max_abs, float(difference.abs().max()))
        cosine = F.cosine_similarity(output, dense, dim=-1).mean()
        self.cosine_error_sum += 1.0 - float(cosine)
        counts = cluster_counts.float()
        self.empty_clusters += int((counts == 0).sum())
        self.total_clusters += counts.numel()
        self.cluster_count_sum += float(counts.sum())
        self.cluster_count_sq_sum += float(counts.square().sum())

    def summary(self):
        count_mean = self.cluster_count_sum / max(self.total_clusters, 1)
        count_var = self.cluster_count_sq_sum / max(self.total_clusters, 1) - count_mean**2
        return {
            "calls": self.calls,
            "rel_l2_vs_dense": math.sqrt(self.sqerr / max(self.ref_sq, 1e-20)),
            "mse_vs_dense": self.sqerr / max(self.numel, 1),
            "max_abs_vs_dense": self.max_abs,
            "cosine_error_vs_dense": self.cosine_error_sum / max(self.calls, 1),
            "empty_cluster_fraction": self.empty_clusters / max(self.total_clusters, 1),
            "cluster_count_cv": math.sqrt(max(count_var, 0.0)) / max(count_mean, 1e-20),
        }


def seed_rows(pattern, device, offsets=None, boundary="fallback", num_seeds=SEEDS):
    key_row = torch.arange(ROWS, device=device)
    if pattern == "contiguous":
        left = num_seeds // 2 - 1
        start = (key_row - left).clamp(0, ROWS - num_seeds)
        return start[:, None] + torch.arange(num_seeds, device=device)
    if pattern == "center_left":
        start = (key_row - 2).clamp(0, ROWS - SEEDS)
        return start[:, None] + torch.arange(SEEDS, device=device)
    if pattern == "center_right":
        start = (key_row - 1).clamp(0, ROWS - SEEDS)
        return start[:, None] + torch.arange(SEEDS, device=device)
    if pattern == "right":
        start = key_row.clamp(0, ROWS - SEEDS)
        return start[:, None] + torch.arange(SEEDS, device=device)
    if pattern == "wide":
        baseline_start = (key_row - 1).clamp(0, ROWS - SEEDS)
        baseline = baseline_start[:, None] + torch.arange(SEEDS, device=device)
        wide = key_row[:, None] + torch.tensor([-2, 0, 1, 3], device=device)
        interior = ((key_row >= 2) & (key_row <= ROWS - 4)).unsqueeze(-1)
        return torch.where(interior, wide, baseline)
    wider_offsets = {
        "wider_left": [-4, -2, 0, 2],
        "wider_balanced": [-3, -1, 0, 3],
        "wider_symmetric": [-4, -1, 0, 4],
        "wider_right": [-1, 0, 2, 4],
    }
    if pattern in wider_offsets:
        offsets = torch.tensor(wider_offsets[pattern], device=device)
        rows = key_row[:, None] + offsets
        fallback = (key_row - 1).clamp(0, ROWS - SEEDS)[:, None]
        fallback = fallback + torch.arange(SEEDS, device=device)
        return torch.where(((rows >= 0) & (rows < ROWS)).all(dim=-1, keepdim=True), rows, fallback)
    if pattern == "offsets":
        rows = key_row[:, None] + torch.tensor(offsets, device=device)
        if boundary == "cyclic":
            return rows.remainder(ROWS)
        if boundary == "shift":
            rows = rows + (-rows.amin(dim=-1, keepdim=True)).clamp_min(0)
            rows = rows - (rows.amax(dim=-1, keepdim=True) - (ROWS - 1)).clamp_min(0)
            return rows
        fallback = (key_row - 1).clamp(0, ROWS - SEEDS)[:, None]
        fallback = fallback + torch.arange(SEEDS, device=device)
        return torch.where(((rows >= 0) & (rows < ROWS)).all(dim=-1, keepdim=True), rows, fallback)
    raise ValueError(pattern)


def assignment_scores(logits, config):
    mode = config["score"]
    if mode == "raw":
        return logits
    if mode == "key_softmax":
        return torch.log_softmax(logits, dim=-1)
    if mode == "temperature":
        return torch.log_softmax(logits / config["temperature"], dim=-1)
    if mode == "normalizer_weight":
        return logits - config["weight"] * torch.logsumexp(logits, dim=-1, keepdim=True)
    centered = logits - logits.mean(dim=-1, keepdim=True)
    if mode == "mean_centered":
        return centered
    if mode == "zscore":
        return centered * torch.rsqrt(logits.var(dim=-1, keepdim=True, correction=0) + 1e-6)
    if mode == "sinkhorn2":
        score = torch.log_softmax(logits, dim=-1)
        score = score - torch.logsumexp(score, dim=-2, keepdim=True)
        return score - torch.logsumexp(score, dim=-1, keepdim=True)
    if mode == "count_balance":
        score = torch.log_softmax(logits, dim=-1)
        first = score.argmax(dim=-2)
        counts = F.one_hot(first, logits.size(-2)).sum(dim=-2)
        return score - config["strength"] * torch.log1p(counts).unsqueeze(-1)
    raise ValueError(mode)


def profile_cluster_assignment(logits, config):
    """Cluster keys by their four seed-query responses."""

    profiles = logits.transpose(-2, -1)
    if config["profile_scale"] == "zscore":
        profiles = profiles - profiles.mean(dim=-2, keepdim=True)
        profiles = profiles * torch.rsqrt(
            profiles.var(dim=-2, keepdim=True, correction=0) + 1e-6
        )

    profile_mean = profiles.mean(dim=-2, keepdim=True)
    distance_to_mean = (profiles - profile_mean).square().sum(dim=-1)
    center_index = distance_to_mean.argmin(dim=-1)
    centers = []
    minimum_distance = None
    for _ in range(SEEDS):
        center = torch.gather(
            profiles,
            -2,
            center_index[..., None, None].expand(
                *center_index.shape, 1, profiles.size(-1)
            ),
        ).squeeze(-2)
        centers.append(center)
        distance = (profiles - center.unsqueeze(-2)).square().sum(dim=-1)
        minimum_distance = (
            distance
            if minimum_distance is None
            else torch.minimum(minimum_distance, distance)
        )
        center_index = minimum_distance.argmax(dim=-1)

    centers = torch.stack(centers, dim=-2)
    distance = (profiles.unsqueeze(-2) - centers.unsqueeze(-3)).square().sum(dim=-1)
    cluster_id = distance.argmin(dim=-1)
    for _ in range(config.get("profile_lloyd_steps", 0)):
        membership = F.one_hot(cluster_id, SEEDS).to(profiles.dtype)
        count = membership.sum(dim=-2).clamp_min(1.0)
        centers = torch.einsum("...lc,...ld->...cd", membership, profiles)
        centers = centers / count.unsqueeze(-1)
        distance = (
            profiles.unsqueeze(-2) - centers.unsqueeze(-3)
        ).square().sum(dim=-1)
        cluster_id = distance.argmin(dim=-1)
    return F.one_hot(cluster_id, SEEDS).movedim(-1, -2).bool()


def block_response_seeds(query, key, value, scale, config):
    """Select four real query rows independently for every key block.

    A small set of uniformly spaced keys describes each (frame, row) block.
    Selection is tensorized over all blocks; only the fixed C=4 selection loop
    remains. The final merge still evaluates exactly four full seed-key rows.
    """

    candidate_count = config["seed_candidates"]
    key_row = torch.arange(ROWS, device=query.device)
    start = (key_row - candidate_count // 2).clamp(0, ROWS - candidate_count)
    candidate_rows = start[:, None] + torch.arange(
        candidate_count, device=query.device
    )
    candidates = query[:, :, candidate_rows, :, :, :]
    candidate_bank = rearrange(
        candidates, "b a k p j h d -> b a 1 k j h p d"
    ).expand(-1, -1, key.size(1), -1, -1, -1, -1, -1)

    proxy_count = config["block_proxy_samples"]
    center = (key_row - start)[None, None, None, :, None, None].expand(
        query.size(0),
        query.size(1),
        key.size(1),
        ROWS,
        query.size(3),
        query.size(4),
    )
    proxy_mode = config.get("block_proxy_mode", "sample")
    if proxy_mode == "dimension_sketch":
        anchor_rows = key_row[:, None] + torch.tensor(
            config.get("proxy_anchor_offsets", [-2, 0, 1, 3]),
            device=query.device,
        )
        valid = ((anchor_rows >= 0) & (anchor_rows < ROWS)).all(
            dim=-1, keepdim=True
        )
        fallback = (
            (key_row - 1).clamp(0, ROWS - SEEDS)[:, None]
            + torch.arange(SEEDS, device=query.device)
        )
        anchor_rows = torch.where(valid, anchor_rows, fallback)
        anchor_queries = query[:, :, anchor_rows, :, :, :]
        anchor_response = torch.einsum(
            "bakcjhd,bfklhd->bafkjhcl",
            anchor_queries.float(),
            key.float(),
        )
        sketch_dimensions = config["qk_sketch_dimensions"]
        sketch_index = torch.linspace(
            0, query.size(-1) - 1, sketch_dimensions, device=query.device
        ).round().long()
        responses = torch.einsum(
            "bakpjhd,bfklhd->bafkjhpl",
            candidates.index_select(-1, sketch_index).float(),
            key.index_select(-1, sketch_index).float(),
        )
        responses *= query.size(-1) / sketch_dimensions
        probe_index = torch.arange(key.size(3), device=query.device).view(
            *(1 for _ in responses.shape[:-2]), key.size(3)
        ).expand(*responses.shape[:-2], key.size(3))
    elif proxy_mode in {
        "topk",
        "mixed_topk",
        "anchor_topk",
        "anchor_mixed_topk",
    }:
        top_count = config.get(
            "proxy_top_count",
            proxy_count if not proxy_mode.endswith("mixed_topk") else proxy_count // 2,
        )
        if config.get("low_flop_proxy_selection", False):
            if proxy_mode.startswith("anchor_"):
                anchor_rows = key_row[:, None] + torch.tensor(
                    config.get("proxy_anchor_offsets", [-2, 0, 1, 3]),
                    device=query.device,
                )
                if config.get("proxy_anchor_boundary") == "shift":
                    anchor_rows = anchor_rows + (
                        -anchor_rows.amin(dim=-1, keepdim=True)
                    ).clamp_min(0)
                    anchor_rows = anchor_rows - (
                        anchor_rows.amax(dim=-1, keepdim=True) - (ROWS - 1)
                    ).clamp_min(0)
                else:
                    valid = ((anchor_rows >= 0) & (anchor_rows < ROWS)).all(
                        dim=-1, keepdim=True
                    )
                    fallback = (
                        (key_row - 1).clamp(0, ROWS - SEEDS)[:, None]
                        + torch.arange(SEEDS, device=query.device)
                    )
                    anchor_rows = torch.where(valid, anchor_rows, fallback)
                anchor_queries = query[:, :, anchor_rows, :, :, :]
                anchor_response = torch.einsum(
                    "bakcjhd,bfklhd->bafkjhcl",
                    anchor_queries.float(),
                    key.float(),
                )
                probe_importance = torch.log_softmax(
                    anchor_response * scale, dim=-1
                ).amax(dim=-2)
            else:
                center_queries = query[:, :, key_row, :, :, :]
                probe_importance = torch.einsum(
                    "bakjhd,bfklhd->bafkjhl",
                    center_queries.float(),
                    key.float(),
                )
            top_index = probe_importance.topk(top_count, dim=-1).indices
            probe_index = top_index
            if proxy_mode.endswith("mixed_topk"):
                uniform_count = proxy_count - top_count
                uniform_probe_index = torch.linspace(
                    0, key.size(3) - 1, uniform_count, device=key.device
                ).round().long()
                uniform_probe_index = uniform_probe_index.view(
                    *(1 for _ in top_index.shape[:-1]), uniform_count
                ).expand(*top_index.shape[:-1], uniform_count)
                probe_index = torch.cat((top_index, uniform_probe_index), dim=-1)
            expanded_key = rearrange(
                key.float(), "b f k l h d -> b 1 f k 1 h l d"
            ).expand(-1, query.size(1), -1, -1, query.size(3), -1, -1, -1)
            key_proxies = torch.gather(
                expanded_key,
                -2,
                probe_index[..., None].expand(*probe_index.shape, key.size(-1)),
            )
            responses = torch.einsum(
                "bakpjhd,bafkjhsd->bafkjhps",
                candidates.float(),
                key_proxies,
            )
        else:
            all_responses = torch.einsum(
                "bakpjhd,bfklhd->bafkjhpl", candidates.float(), key.float()
            )
            if proxy_mode.startswith("anchor_"):
                anchor_rows = key_row[:, None] + torch.tensor(
                    config.get("proxy_anchor_offsets", [-2, 0, 1, 3]),
                    device=query.device,
                )
                valid = ((anchor_rows >= 0) & (anchor_rows < ROWS)).all(
                    dim=-1, keepdim=True
                )
                fallback = (
                    (key_row - 1).clamp(0, ROWS - SEEDS)[:, None]
                    + torch.arange(SEEDS, device=query.device)
                )
                anchor_rows = torch.where(valid, anchor_rows, fallback)
                anchor_index = (anchor_rows - start[:, None])[
                    None, None, None, :, None, None, :
                ].expand(*center.shape, anchor_rows.size(-1))
                anchor_response = torch.gather(
                    all_responses,
                    -2,
                    anchor_index[..., None].expand(
                        *anchor_index.shape, all_responses.size(-1)
                    ),
                )
                probe_importance = torch.log_softmax(
                    anchor_response * scale, dim=-1
                ).amax(dim=-2)
            else:
                probe_importance = torch.gather(
                    all_responses,
                    -2,
                    center[..., None, None].expand(
                        *center.shape, 1, all_responses.size(-1)
                    ),
                ).squeeze(-2)
            top_index = probe_importance.topk(top_count, dim=-1).indices
            probe_index = top_index
            responses = torch.gather(
                all_responses,
                -1,
                top_index.unsqueeze(-2).expand(
                    *top_index.shape[:-1], candidate_count, top_count
                ),
            )
            if proxy_mode.endswith("mixed_topk"):
                uniform_count = proxy_count - top_count
                uniform_probe_index = torch.linspace(
                    0, key.size(3) - 1, uniform_count, device=key.device
                ).round().long()
                uniform_probe_index = uniform_probe_index.view(
                    *(1 for _ in top_index.shape[:-1]), uniform_count
                ).expand(*top_index.shape[:-1], uniform_count)
                probe_index = torch.cat((top_index, uniform_probe_index), dim=-1)
                uniform_index = uniform_probe_index.unsqueeze(-2).expand(
                    *uniform_probe_index.shape[:-1], candidate_count, uniform_count
                )
                responses = torch.cat(
                    (responses, torch.gather(all_responses, -1, uniform_index)), dim=-1
                )
    elif proxy_mode == "mean":
        key_proxies = key.reshape(
            *key.shape[:3], proxy_count, key.size(3) // proxy_count, *key.shape[4:]
        ).float().mean(dim=4)
        responses = torch.einsum(
            "bakpjhd,bfkshd->bafkjhps", candidates.float(), key_proxies
        )
    else:
        proxy_columns = torch.linspace(
            0, key.size(3) - 1, proxy_count, device=key.device
        ).round().long()
        key_proxies = key.index_select(3, proxy_columns).float()
        if config.get("block_proxy_temporal_mean", False):
            key_proxies = key_proxies.mean(dim=1, keepdim=True).expand(
                -1, key.size(1), -1, -1, -1, -1
            )
        responses = torch.einsum(
            "bakpjhd,bfkshd->bafkjhps", candidates.float(), key_proxies
        )
        probe_index = proxy_columns.view(
            *(1 for _ in responses.shape[:-2]), proxy_count
        ).expand(*responses.shape[:-2], proxy_count)
    responses *= scale / config.get("response_temperature", 1.0)
    proxy_logits = responses
    response_affinity = None
    if "response_affinity_scale" in config:
        response_affinity = torch.logsumexp(responses, dim=-1)
        response_affinity = response_affinity - response_affinity.mean(
            dim=-1, keepdim=True
        )
        response_affinity = response_affinity * torch.rsqrt(
            response_affinity.var(dim=-1, keepdim=True, correction=0) + 1e-6
        )
    tangent_gap = None
    if config["block_response_method"] == "tangent":
        probe_log_probability = torch.log_softmax(responses, dim=-1)
        probe_probability = probe_log_probability.exp()
        probe_entropy = -(probe_probability * probe_log_probability).sum(dim=-1)
        variational_value = torch.einsum(
            "...cs,...ps->...pc", probe_probability, responses
        ) + probe_entropy.unsqueeze(-2)
        tangent_gap = (
            torch.logsumexp(responses, dim=-1).unsqueeze(-1) - variational_value
        ).clamp_min_(0.0)
    response_feature = config.get("response_feature", "attention")
    value_response = None
    if response_feature != "attention":
        if proxy_mode == "mean":
            raise ValueError("Value-output seed features require explicit key probes.")
        feature_value = value.float()
        if "response_value_dimensions" in config:
            value_dimensions = config["response_value_dimensions"]
            feature_value = feature_value.reshape(
                *feature_value.shape[:-1],
                value_dimensions,
                feature_value.size(-1) // value_dimensions,
            ).mean(dim=-1)
        expanded_value = rearrange(
            feature_value, "b f k l h d -> b 1 f k 1 h l d"
        ).expand(
            -1,
            query.size(1),
            -1,
            -1,
            query.size(3),
            -1,
            -1,
            -1,
        )
        probe_value = torch.gather(
            expanded_value,
            -2,
            probe_index[..., None].expand(*probe_index.shape, feature_value.size(-1)),
        )
        probability_response = torch.softmax(responses, dim=-1)
        value_response = torch.einsum(
            "bafkjhps,bafkjhsd->bafkjhpd",
            probability_response,
            probe_value,
        )
        if response_feature == "value_output":
            responses = value_response
        else:
            responses = torch.cat(
                (
                    F.normalize(probability_response, dim=-1),
                    config.get("value_feature_scale", 1.0)
                    * F.normalize(value_response, dim=-1),
                ),
                dim=-1,
            )
    elif config["response_scale"] == "logprob":
        responses = torch.log_softmax(responses, dim=-1)
    elif config["response_scale"] == "probability":
        responses = torch.softmax(responses, dim=-1)
    elif config["response_scale"] == "sqrt_probability":
        responses = torch.softmax(responses, dim=-1).sqrt()
    if response_affinity is not None:
        responses = torch.cat(
            (
                responses,
                config["response_affinity_scale"]
                * response_affinity.unsqueeze(-1),
            ),
            dim=-1,
        )
    if config.get("normalize_seed_features", False):
        responses = F.normalize(responses, dim=-1)

    index_shape = responses.shape[:-2]
    if config.get("initial_seed") == "medoid":
        response_center = responses.mean(dim=-2, keepdim=True)
        initial_seed = (
            (responses - response_center).square().mean(dim=-1).argmin(dim=-1)
        )
        selected_indices = [initial_seed]
    else:
        selected_indices = [center] if config.get("force_center", True) else []
    for offset in config.get("seed_anchor_offsets", []):
        anchor_row = key_row + offset
        anchor_row = torch.where(
            (anchor_row >= 0) & (anchor_row < ROWS), anchor_row, key_row - offset
        )
        selected_indices.append(
            (anchor_row - start)[None, None, None, :, None, None].expand(index_shape)
        )

    used = torch.zeros_like(responses[..., 0], dtype=torch.bool)
    selected = []
    for index in selected_indices:
        selected.append(
            torch.gather(
                candidate_bank,
                -2,
                index[..., None, None].expand(*index.shape, 1, query.size(-1)),
            ).squeeze(-2)
        )
        used.scatter_(-1, index.unsqueeze(-1), True)

    if config["block_response_method"] == "tangent":
        minimum_gap = torch.full_like(tangent_gap[..., 0], torch.inf)
        for index in selected_indices:
            selected_gap = torch.gather(
                tangent_gap,
                -1,
                index[..., None, None].expand(
                    *index.shape, tangent_gap.size(-2), 1
                ),
            ).squeeze(-1)
            minimum_gap = torch.minimum(minimum_gap, selected_gap)
    elif config["block_response_method"] == "span":
        if config.get("span_center") == "candidates":
            residual = responses - responses.mean(dim=-2, keepdim=True)
        else:
            residual = responses - responses.mean(dim=-1, keepdim=True)
        for index in selected_indices:
            picked = torch.gather(
                residual,
                -2,
                index[..., None, None].expand(*index.shape, 1, residual.size(-1)),
            ).squeeze(-2)
            basis = F.normalize(picked, dim=-1)
            residual -= (
                (residual * basis.unsqueeze(-2)).sum(dim=-1, keepdim=True)
                * basis.unsqueeze(-2)
            )
    else:
        minimum_distance = None
        for index in selected_indices:
            picked = torch.gather(
                responses,
                -2,
                index[..., None, None].expand(*index.shape, 1, responses.size(-1)),
            ).squeeze(-2)
            distance = (responses - picked.unsqueeze(-2)).square().mean(dim=-1)
            minimum_distance = (
                distance
                if minimum_distance is None
                else torch.minimum(minimum_distance, distance)
            )
        if minimum_distance is None:
            feature_mean = responses.mean(dim=-2, keepdim=True)
            minimum_distance = (responses - feature_mean).square().mean(dim=-1)

    while len(selected) < SEEDS:
        if config["block_response_method"] == "tangent":
            covered_gap = torch.minimum(
                minimum_gap.unsqueeze(-1), tangent_gap
            )
            if config.get("tangent_objective", "mean") == "max":
                objective = covered_gap.amax(dim=-2)
            else:
                objective = covered_gap.mean(dim=-2)
            choice = objective.masked_fill(used, torch.inf).argmin(dim=-1)
        else:
            objective = (
                residual.square().sum(dim=-1)
                if config["block_response_method"] == "span"
                else minimum_distance
            )
            choice = objective.masked_fill(used, -1.0).argmax(dim=-1)
        selected.append(
            torch.gather(
                candidate_bank,
                -2,
                choice[..., None, None].expand(
                    *choice.shape, 1, query.size(-1)
                ),
            ).squeeze(-2)
        )
        selected_indices.append(choice)
        used.scatter_(-1, choice.unsqueeze(-1), True)
        if config["block_response_method"] == "tangent":
            selected_gap = torch.gather(
                tangent_gap,
                -1,
                choice[..., None, None].expand(
                    *choice.shape, tangent_gap.size(-2), 1
                ),
            ).squeeze(-1)
            minimum_gap = torch.minimum(minimum_gap, selected_gap)
        elif config["block_response_method"] == "span":
            picked = torch.gather(
                residual,
                -2,
                choice[..., None, None].expand(
                    *choice.shape, 1, residual.size(-1)
                ),
            ).squeeze(-2)
            basis = F.normalize(picked, dim=-1)
            residual -= (
                (residual * basis.unsqueeze(-2)).sum(dim=-1, keepdim=True)
                * basis.unsqueeze(-2)
            )
        else:
            picked = torch.gather(
                responses,
                -2,
                choice[..., None, None].expand(
                    *choice.shape, 1, responses.size(-1)
                ),
            ).squeeze(-2)
            distance = (responses - picked.unsqueeze(-2)).square().mean(dim=-1)
            minimum_distance = torch.minimum(minimum_distance, distance)

    if "multiscale_local_candidates" in config:
        local_count = config["multiscale_local_candidates"]
        local_start = (key_row - local_count // 2).clamp(0, ROWS - local_count)
        local_rows = start[:, None] + torch.arange(
            candidate_count, device=query.device
        )
        local_valid = (
            (local_rows >= local_start[:, None])
            & (local_rows < local_start[:, None] + local_count)
        )
        local_valid = local_valid[None, None, None, :, None, None].expand(
            *responses.shape[:-1]
        )

        alternative_force_center = config.get(
            "multiscale_alternative_force_center", True
        )
        local_indices = [center] if alternative_force_center else []
        local_used = torch.zeros_like(responses[..., 0], dtype=torch.bool)
        local_residual = responses - responses.mean(dim=-1, keepdim=True)
        if alternative_force_center:
            local_used.scatter_(-1, center.unsqueeze(-1), True)
            center_response = torch.gather(
                local_residual,
                -2,
                center[..., None, None].expand(
                    *center.shape, 1, local_residual.size(-1)
                ),
            ).squeeze(-2)
            center_basis = F.normalize(center_response, dim=-1)
            local_residual -= (
                (local_residual * center_basis.unsqueeze(-2)).sum(
                    dim=-1, keepdim=True
                )
                * center_basis.unsqueeze(-2)
            )
        while len(local_indices) < SEEDS:
            local_score = local_residual.square().sum(dim=-1)
            local_choice = local_score.masked_fill(
                local_used | ~local_valid, -1.0
            ).argmax(dim=-1)
            local_indices.append(local_choice)
            local_used.scatter_(-1, local_choice.unsqueeze(-1), True)
            picked = torch.gather(
                local_residual,
                -2,
                local_choice[..., None, None].expand(
                    *local_choice.shape, 1, local_residual.size(-1)
                ),
            ).squeeze(-2)
            basis = F.normalize(picked, dim=-1)
            local_residual -= (
                (local_residual * basis.unsqueeze(-2)).sum(dim=-1, keepdim=True)
                * basis.unsqueeze(-2)
            )

        global_indices = torch.stack(selected_indices, dim=-1)
        local_indices = torch.stack(local_indices, dim=-1)

        def proxy_output_error(indices):
            seed_proxy_logits = torch.gather(
                proxy_logits,
                -2,
                indices.unsqueeze(-1).expand(
                    *indices.shape, proxy_logits.size(-1)
                ),
            )
            seed_proxy_scores = torch.log_softmax(seed_proxy_logits, dim=-1)
            proxy_assignment = F.one_hot(
                seed_proxy_scores.argmax(dim=-2), SEEDS
            ).movedim(-1, -2).bool()
            proxy_counts = proxy_assignment.sum(dim=-1)
            base_log_weights = torch.log_softmax(
                seed_proxy_logits.masked_fill(~proxy_assignment, NEG_INF), dim=-1
            ).masked_fill(~proxy_assignment, 0.0)
            base_weights = base_log_weights.exp().masked_fill(
                ~proxy_assignment, 0.0
            )
            weights = base_weights
            if "peak_weight_mix" in config:
                peak_weights = F.one_hot(
                    weights.argmax(dim=-1), weights.size(-1)
                ).to(weights.dtype)
                peak_weights *= (proxy_counts > 0).unsqueeze(-1)
                weights = torch.lerp(
                    weights, peak_weights, config["peak_weight_mix"]
                )
            log_weights = weights.clamp_min(1e-30).log()
            base_bias = (base_weights * base_log_weights).sum(dim=-1)
            routing_bias = (weights * log_weights).sum(dim=-1)
            if "entropy_bias_mix" in config:
                routing_bias = torch.lerp(
                    base_bias, routing_bias, config["entropy_bias_mix"]
                )
            routing_bias.masked_fill_(proxy_counts == 0, float("inf"))
            compressed_logits = torch.einsum(
                "...ps,...cs->...pc", proxy_logits, weights
            )
            compressed_attention = torch.softmax(
                compressed_logits - routing_bias.unsqueeze(-2), dim=-1
            )
            objective = config.get("proxy_selection_objective", "value")
            if objective.startswith("attention_"):
                expanded_attention = torch.einsum(
                    "...pc,...cs->...ps", compressed_attention, weights
                )
                dense_attention = torch.softmax(proxy_logits, dim=-1)
                if objective == "attention_kl":
                    return (
                        dense_attention
                        * (
                            dense_attention.clamp_min(1e-30).log()
                            - expanded_attention.clamp_min(1e-30).log()
                        )
                    ).sum(dim=-1).mean(dim=-1)
                return (expanded_attention - dense_attention).square().mean(
                    dim=(-1, -2)
                )
            objective_probe_value = probe_value
            objective_value_response = value_response
            if "proxy_objective_value_dimensions" in config:
                pooled_dimensions = config["proxy_objective_value_dimensions"]
                channels_per_pool = value.size(-1) // pooled_dimensions
                objective_probe_value = objective_probe_value.reshape(
                    *objective_probe_value.shape[:-1],
                    pooled_dimensions,
                    channels_per_pool,
                ).mean(dim=-1)
                objective_value_response = objective_value_response.reshape(
                    *objective_value_response.shape[:-1],
                    pooled_dimensions,
                    channels_per_pool,
                ).mean(dim=-1)
            compressed_output = torch.einsum(
                "...pc,...cd->...pd",
                compressed_attention,
                torch.einsum(
                    "...cs,...sd->...cd", weights, objective_probe_value
                ),
            )
            return (compressed_output - objective_value_response).square().mean(
                dim=(-1, -2)
            )

        use_local = proxy_output_error(local_indices) < proxy_output_error(
            global_indices
        )
        selected_indices = [
            torch.where(
                use_local,
                local_indices[..., index],
                global_indices[..., index],
            )
            for index in range(SEEDS)
        ]
        selected = [
            torch.gather(
                candidate_bank,
                -2,
                index[..., None, None].expand(
                    *index.shape, 1, query.size(-1)
                ),
            ).squeeze(-2)
            for index in selected_indices
        ]

    lloyd_steps = config.get("block_response_lloyd_steps", 0)
    if lloyd_steps:
        indices = torch.stack(selected_indices, dim=-1)
        fixed_count = int(config.get("force_center", True)) + len(
            config.get("seed_anchor_offsets", [])
        )
        for _ in range(lloyd_steps):
            centers = torch.gather(
                responses,
                -2,
                indices.unsqueeze(-1).expand(*indices.shape, responses.size(-1)),
            )
            distance = (
                responses.unsqueeze(-2) - centers.unsqueeze(-3)
            ).square().mean(dim=-1)
            cluster_id = distance.argmin(dim=-1)
            membership = F.one_hot(cluster_id, SEEDS).to(responses.dtype)
            count = membership.sum(dim=-2)
            centroids = torch.einsum(
                "...pc,...pd->...cd", membership, responses
            ) / count.clamp_min(1).unsqueeze(-1)
            used = torch.zeros_like(responses[..., 0], dtype=torch.bool)
            updated = []
            for seed_index in range(SEEDS):
                previous = indices[..., seed_index]
                if seed_index < fixed_count:
                    choice = previous
                else:
                    candidate_distance = (
                        responses - centroids[..., seed_index, :].unsqueeze(-2)
                    ).square().mean(dim=-1)
                    eligible = (cluster_id == seed_index) & ~used
                    choice = candidate_distance.masked_fill(
                        ~eligible, torch.inf
                    ).argmin(dim=-1)
                    choice = torch.where(eligible.any(dim=-1), choice, previous)
                updated.append(choice)
                used.scatter_(-1, choice.unsqueeze(-1), True)
            indices = torch.stack(updated, dim=-1)
        selected_queries = torch.gather(
            candidate_bank,
            -2,
            indices.unsqueeze(-1).expand(*indices.shape, query.size(-1)),
        )
        selected_queries = rearrange(
            selected_queries, "b a f k j h c d -> b a f k c j h d"
        )
        if config.get("return_seed_indices", False):
            global_rows = indices + start[None, None, None, :, None, None, None]
            return selected_queries, global_rows
        return selected_queries

    selected_queries = torch.stack(selected, dim=4)
    if "value_fit_proxy_anchor_indices" in config:
        fit_indices = torch.tensor(
            config["value_fit_proxy_anchor_indices"], device=query.device
        )
        fit_anchor_queries = anchor_queries.index_select(3, fit_indices)
        fit_anchor_queries = fit_anchor_queries[:, :, None].expand(
            -1, -1, key.size(1), -1, -1, -1, -1, -1
        )
        fit_anchor_logits = anchor_response.index_select(-2, fit_indices)
        fit_anchor_logits = rearrange(
            fit_anchor_logits,
            "b a f k j h s l -> b h a f k j s l",
        ) * scale
        seed_indices = None
        if config.get("return_seed_indices", False):
            indices = torch.stack(selected_indices, dim=-1)
            seed_indices = indices + start[None, None, None, :, None, None, None]
        result = (
            selected_queries,
            seed_indices,
            fit_anchor_queries,
            fit_anchor_logits,
        )
        if "value_fit_sketch_query_count" in config:
            sketch_count = config["value_fit_sketch_query_count"]
            sketch_indices = torch.linspace(
                0, candidate_count - 1, sketch_count, device=query.device
            ).round().long()
            sketch_queries = candidate_bank.index_select(-2, sketch_indices)
            sketch_queries = rearrange(
                sketch_queries, "b a f k j h s d -> b a f k s j h d"
            )
            sketch_targets = value_response.index_select(-2, sketch_indices)
            sketch_targets = rearrange(
                sketch_targets, "b a f k j h s d -> b h a f k j s d"
            )
            result += (sketch_queries, sketch_targets)
        return result
    if "value_fit_sketch_query_count" in config:
        sketch_count = config["value_fit_sketch_query_count"]
        sketch_indices = torch.linspace(
            0, candidate_count - 1, sketch_count, device=query.device
        ).round().long()
        sketch_queries = candidate_bank.index_select(-2, sketch_indices)
        sketch_queries = rearrange(
            sketch_queries, "b a f k j h s d -> b a f k s j h d"
        )
        sketch_targets = value_response.index_select(-2, sketch_indices)
        sketch_targets = rearrange(
            sketch_targets, "b a f k j h s d -> b h a f k j s d"
        )
        seed_indices = None
        if config.get("return_seed_indices", False):
            indices = torch.stack(selected_indices, dim=-1)
            seed_indices = indices + start[None, None, None, :, None, None, None]
        return (
            selected_queries,
            seed_indices,
            None,
            None,
            sketch_queries,
            sketch_targets,
        )
    if config.get("return_seed_indices", False):
        indices = torch.stack(selected_indices, dim=-1)
        global_rows = indices + start[None, None, None, :, None, None, None]
        return selected_queries, global_rows
    return selected_queries


def pilot_refined_seeds(query, key, scale, config):
    """Use one ordinary C4 merge to choose a better set of four real queries.

    The pilot wide seeds compress each 80-key block to four keys. Candidate
    queries are then compared only with those four pilot keys, so searching all
    45 query rows costs 45 x 4 rather than 45 x 80 dot products per block.
    """

    rows = seed_rows("wide", query.device)
    pilot_seeds = query[:, :, rows, :, :, :]
    pilot_logits = torch.einsum(
        "bakcjhd,bfklhd->bhafkjcl", pilot_seeds, key
    ).float() * scale
    pilot_scores = torch.log_softmax(pilot_logits, dim=-1)
    pilot_assignment = torch.zeros_like(pilot_logits, dtype=torch.bool)
    pilot_assignment.scatter_(
        -2, pilot_scores.argmax(dim=-2, keepdim=True), True
    )
    pilot_log_weights = torch.log_softmax(
        pilot_logits.masked_fill(~pilot_assignment, NEG_INF), dim=-1
    ).masked_fill(~pilot_assignment, 0.0)
    pilot_weights = pilot_log_weights.exp().masked_fill(~pilot_assignment, 0.0)
    pilot_keys = torch.einsum(
        "bhafkjcl,bfklhd->bafjkchd", pilot_weights.to(key.dtype), key
    )

    candidate_count = config.get("seed_candidates", ROWS)
    key_row = torch.arange(ROWS, device=query.device)
    start = (key_row - candidate_count // 2).clamp(0, ROWS - candidate_count)
    candidate_rows = start[:, None] + torch.arange(
        candidate_count, device=query.device
    )
    candidates = query[:, :, candidate_rows, :, :, :]
    candidate_bank = rearrange(
        candidates, "b a k p j h d -> b a 1 k j h p d"
    ).expand(-1, -1, key.size(1), -1, -1, -1, -1, -1)
    responses = torch.einsum(
        "bakpjhd,bafjkchd->bafkjhpc", candidates.float(), pilot_keys.float()
    ) * scale
    if config.get("pilot_response_scale", "probability") == "probability":
        responses = torch.softmax(responses, dim=-1)
    elif config["pilot_response_scale"] == "logprob":
        responses = torch.log_softmax(responses, dim=-1)

    index_shape = responses.shape[:-2]
    selected_indices = []
    if config.get("force_center", True):
        selected_indices.append(
            (key_row - start)[None, None, None, :, None, None].expand(index_shape)
        )
    for offset in config.get("seed_anchor_offsets", []):
        anchor_row = key_row + offset
        anchor_row = torch.where(
            (anchor_row >= 0) & (anchor_row < ROWS), anchor_row, key_row - offset
        )
        selected_indices.append(
            (anchor_row - start)[None, None, None, :, None, None].expand(index_shape)
        )

    used = torch.zeros_like(responses[..., 0], dtype=torch.bool)
    selected = []
    residual = responses - responses.mean(dim=-1, keepdim=True)
    for index in selected_indices:
        selected.append(
            torch.gather(
                candidate_bank,
                -2,
                index[..., None, None].expand(*index.shape, 1, query.size(-1)),
            ).squeeze(-2)
        )
        used.scatter_(-1, index.unsqueeze(-1), True)
        picked = torch.gather(
            residual,
            -2,
            index[..., None, None].expand(*index.shape, 1, residual.size(-1)),
        ).squeeze(-2)
        basis = F.normalize(picked, dim=-1)
        residual -= (
            (residual * basis.unsqueeze(-2)).sum(dim=-1, keepdim=True)
            * basis.unsqueeze(-2)
        )

    while len(selected) < SEEDS:
        choice = residual.square().sum(dim=-1).masked_fill(used, -1.0).argmax(dim=-1)
        selected.append(
            torch.gather(
                candidate_bank,
                -2,
                choice[..., None, None].expand(
                    *choice.shape, 1, query.size(-1)
                ),
            ).squeeze(-2)
        )
        used.scatter_(-1, choice.unsqueeze(-1), True)
        picked = torch.gather(
            residual,
            -2,
            choice[..., None, None].expand(
                *choice.shape, 1, residual.size(-1)
            ),
        ).squeeze(-2)
        basis = F.normalize(picked, dim=-1)
        residual -= (
            (residual * basis.unsqueeze(-2)).sum(dim=-1, keepdim=True)
            * basis.unsqueeze(-2)
        )

    return torch.stack(selected, dim=4)


def hard_c4_sample(query, key, value, scale, config):
    """Run Hard C4 for sampled heads and one sampled query frame."""

    quantile_assignment = None
    selected_seed_rows = None
    value_fit_proxy_queries = None
    value_fit_proxy_logits = None
    value_fit_sketch_queries = None
    value_fit_sketch_targets = None
    route_refined_weights = None
    if config["seed"] == "score_quantile":
        key_row = torch.arange(ROWS, device=query.device)
        offsets = torch.tensor(config["probe_offsets"], device=query.device)
        probe_rows = key_row[:, None] + offsets
        probe_rows = torch.where(
            (probe_rows >= 0) & (probe_rows < ROWS),
            probe_rows,
            key_row[:, None] - offsets,
        )
        probes = query[:, :, probe_rows, :, :, :]
        probe_logits = torch.einsum(
            "bakpjhd,bfklhd->bhafkjpl", probes, key
        ).float()
        probe_logits *= scale
        if probe_logits.size(-2) == 1:
            partition_logits = probe_logits.squeeze(-2)
        elif config.get("probe_reduce", "mean") == "max":
            partition_logits = probe_logits.amax(dim=-2)
        else:
            partition_logits = probe_logits.mean(dim=-2)

        order = partition_logits.argsort(dim=-1)
        if config["partition"] == "equal_count":
            rank = torch.empty_like(order)
            rank.scatter_(
                -1,
                order,
                torch.arange(key.size(3), device=key.device).expand_as(order),
            )
            cluster_id = rank * SEEDS // key.size(3)
        else:
            sorted_logits = partition_logits.gather(-1, order)
            sorted_mass = torch.softmax(sorted_logits, dim=-1)
            midpoint_mass = sorted_mass.cumsum(dim=-1) - 0.5 * sorted_mass
            sorted_cluster = (midpoint_mass * SEEDS).long().clamp_max(SEEDS - 1)
            cluster_id = torch.empty_like(sorted_cluster)
            cluster_id.scatter_(-1, order, sorted_cluster)
        quantile_assignment = F.one_hot(cluster_id, SEEDS).movedim(-1, -2).bool()
        logits = partition_logits.unsqueeze(-2).expand(
            *partition_logits.shape[:-1], SEEDS, partition_logits.size(-1)
        )
        seeds = None
    elif config["seed"] == "q_block_response":
        seeds = block_response_seeds(query, key, value, scale, config)
        if isinstance(seeds, tuple):
            if len(seeds) == 2:
                seeds, selected_seed_rows = seeds
            elif len(seeds) == 4:
                (
                    seeds,
                    selected_seed_rows,
                    value_fit_proxy_queries,
                    value_fit_proxy_logits,
                ) = seeds
            else:
                (
                    seeds,
                    selected_seed_rows,
                    value_fit_proxy_queries,
                    value_fit_proxy_logits,
                    value_fit_sketch_queries,
                    value_fit_sketch_targets,
                ) = seeds
    elif config["seed"] == "q_pilot_refine":
        seeds = pilot_refined_seeds(query, key, scale, config)
    elif config["seed"] == "q_centroid":
        candidate_count = config["seed_candidates"]
        key_row = torch.arange(ROWS, device=query.device)
        start = (key_row - candidate_count // 2).clamp(
            0, ROWS - candidate_count
        )
        candidate_rows = start[:, None] + torch.arange(
            candidate_count, device=query.device
        )
        candidates = query[:, :, candidate_rows, :, :, :]
        rows = seed_rows("wide", query.device)
        seeds = query[:, :, rows, :, :, :]
        for _ in range(config["centroid_steps"]):
            distance = (
                candidates.float().unsqueeze(4) - seeds.float().unsqueeze(3)
            ).square().mean(dim=-1)
            cluster_id = distance.argmin(dim=4)
            membership = F.one_hot(cluster_id, SEEDS).movedim(-1, 4).float()
            count = membership.sum(dim=3)
            centroid = torch.einsum(
                "bakpcjh,bakpjhd->bakcjhd", membership, candidates.float()
            )
            centroid = centroid / count.clamp_min(1).unsqueeze(-1)
            if config["centroid_normalize"]:
                candidate_norm = candidates.float().norm(dim=-1)
                target_norm = torch.einsum(
                    "bakpcjh,bakpjh->bakcjh", membership, candidate_norm
                )
                target_norm = target_norm / count.clamp_min(1)
                centroid = centroid * (
                    target_norm / centroid.norm(dim=-1).clamp_min(1e-12)
                ).unsqueeze(-1)
            seeds = torch.where(
                (count > 0).unsqueeze(-1), centroid, seeds.float()
            ).to(query.dtype)
    elif config["seed"] in {
        "q_farthest",
        "q_cover",
        "q_key_proxy",
        "q_sampled_key_cover",
        "q_response_farthest",
        "q_response_cover",
        "q_response_exact_cover",
        "q_response_span",
        "q_hybrid",
    }:
        candidate_count = config["seed_candidates"]
        key_row = torch.arange(ROWS, device=query.device)
        start = (key_row - candidate_count // 2).clamp(0, ROWS - candidate_count)
        candidate_rows = start[:, None] + torch.arange(
            candidate_count, device=query.device
        )
        candidates = query[:, :, candidate_rows, :, :, :]
        center = key_row - start
        center = center[None, None, :, None, None, None].expand(
            query.size(0), query.size(1), ROWS, 1, COLS, query.size(4)
        )
        selected = torch.gather(
            candidates,
            3,
            center[..., None].expand(-1, -1, -1, -1, -1, -1, query.size(-1)),
        ).squeeze(3)
        seeds = [selected]
        used = torch.zeros_like(candidates[..., 0], dtype=torch.bool)
        used.scatter_(3, center, True)
        selected_indices = [center.squeeze(3)]

        for offset in config.get("seed_anchor_offsets", []):
            anchor_row = key_row + offset
            anchor_row = torch.where(
                (anchor_row >= 0) & (anchor_row < ROWS), anchor_row, key_row - offset
            )
            anchor = (anchor_row - start)[None, None, :, None, None, None].expand_as(center)
            chosen = torch.gather(
                candidates,
                3,
                anchor[..., None].expand(-1, -1, -1, -1, -1, -1, query.size(-1)),
            ).squeeze(3)
            seeds.append(chosen)
            selected_indices.append(anchor.squeeze(3))
            used.scatter_(3, anchor, True)

        distance_vectors = candidates.float()
        if config.get("seed_distance") == "cosine":
            distance_vectors = F.normalize(distance_vectors, dim=-1)

        pair_distance = None
        proxy_scores = None
        if config["seed"] == "q_farthest":
            selected_vector = torch.gather(
                distance_vectors,
                3,
                center[..., None].expand(-1, -1, -1, -1, -1, -1, query.size(-1)),
            ).squeeze(3)
            minimum_distance = (
                distance_vectors - selected_vector[:, :, :, None]
            ).square().mean(-1)
            for seed_index in selected_indices[1:]:
                chosen_vector = torch.gather(
                    distance_vectors,
                    3,
                    seed_index[:, :, :, None, :, :, None].expand(
                        -1, -1, -1, 1, -1, -1, query.size(-1)
                    ),
                ).squeeze(3)
                distance = (
                    distance_vectors - chosen_vector[:, :, :, None]
                ).square().mean(-1)
                minimum_distance = torch.minimum(minimum_distance, distance)
        elif config["seed"] == "q_cover":
            vectors = rearrange(distance_vectors, "b a k p j h d -> b a k j h p d")
            norm = vectors.square().mean(dim=-1)
            pair_distance = (
                norm[..., :, None]
                + norm[..., None, :]
                - 2.0
                * torch.einsum("...pd,...qd->...pq", vectors, vectors)
                / vectors.size(-1)
            ).clamp_min_(0.0)
            minimum_cover_distance = torch.full_like(pair_distance[..., 0, :], torch.inf)
            for seed_index in selected_indices:
                seed_distance = torch.gather(
                    pair_distance,
                    -2,
                    seed_index[..., None, None].expand(
                        *seed_index.shape, 1, candidate_count
                    ),
                ).squeeze(-2)
                minimum_cover_distance = torch.minimum(
                    minimum_cover_distance, seed_distance
                )
        else:
            if config["seed"] == "q_key_proxy":
                temporal_groups = config["proxy_temporal_groups"]
                spatial_groups = 4
                key_proxies = key.reshape(
                    key.size(0),
                    temporal_groups,
                    key.size(1) // temporal_groups,
                    ROWS,
                    spatial_groups,
                    COLS // spatial_groups,
                    key.size(4),
                    key.size(5),
                ).float().mean(dim=(2, 5))
                key_proxies = rearrange(
                    key_proxies, "b t k s h d -> b k (t s) h d"
                )
            else:
                temporal_indices = torch.linspace(
                    0,
                    key.size(1) - 1,
                    config["proxy_temporal_samples"],
                    device=key.device,
                ).round().long()
                spatial_indices = torch.linspace(
                    0,
                    COLS - 1,
                    config["proxy_spatial_samples"],
                    device=key.device,
                ).round().long()
                key_proxies = key.index_select(1, temporal_indices).index_select(
                    3, spatial_indices
                ).float()
                key_proxies = rearrange(
                    key_proxies, "b t k s h d -> b k (t s) h d"
                )
            proxy_candidates = candidates.float()
            if config.get("proxy_cosine", False):
                proxy_candidates = F.normalize(proxy_candidates, dim=-1)
                key_proxies = F.normalize(key_proxies, dim=-1)
            proxy_scores = torch.einsum(
                "bakpjhd,bkshd->bakjhps", proxy_candidates, key_proxies
            )
            if config.get("proxy_score") == "key_softmax":
                proxy_scores = torch.log_softmax(proxy_scores, dim=-1)
            if config["seed"] in {
                "q_response_farthest",
                "q_response_cover",
                "q_response_exact_cover",
                "q_response_span",
            }:
                response_vectors = proxy_scores * scale
                if config["response_scale"] == "logprob":
                    response_vectors = torch.log_softmax(response_vectors, dim=-1)
                elif config["response_scale"] == "probability":
                    response_vectors = torch.softmax(response_vectors, dim=-1)
                if config["seed"] in {
                    "q_response_cover",
                    "q_response_exact_cover",
                }:
                    response_pair_distance = (
                        response_vectors.unsqueeze(-2)
                        - response_vectors.unsqueeze(-3)
                    ).square().mean(dim=-1)
                    minimum_response_cover = torch.full_like(
                        response_pair_distance[..., 0, :], torch.inf
                    )
                    for seed_index in selected_indices:
                        selected_distance = torch.gather(
                            response_pair_distance,
                            -2,
                            seed_index[..., None, None].expand(
                                *seed_index.shape, 1, candidate_count
                            ),
                        ).squeeze(-2)
                        minimum_response_cover = torch.minimum(
                            minimum_response_cover, selected_distance
                        )
                elif config["seed"] == "q_response_span":
                    span_residual = response_vectors - response_vectors.mean(
                        dim=-1, keepdim=True
                    )
                    for seed_index in selected_indices:
                        selected_residual = torch.gather(
                            span_residual,
                            -2,
                            seed_index[..., None, None].expand(
                                *seed_index.shape, 1, span_residual.size(-1)
                            ),
                        ).squeeze(-2)
                        basis = F.normalize(selected_residual, dim=-1)
                        projection = (
                            span_residual * basis.unsqueeze(-2)
                        ).sum(dim=-1, keepdim=True)
                        span_residual = span_residual - projection * basis.unsqueeze(-2)
                else:
                    minimum_response_distance = None
                    for seed_index in selected_indices:
                        selected_response = torch.gather(
                            response_vectors,
                            -2,
                            seed_index[..., None, None].expand(
                                *seed_index.shape, 1, response_vectors.size(-1)
                            ),
                        ).squeeze(-2)
                        response_distance = (
                            response_vectors - selected_response.unsqueeze(-2)
                        ).square().mean(dim=-1)
                        minimum_response_distance = (
                            response_distance
                            if minimum_response_distance is None
                            else torch.minimum(
                                minimum_response_distance, response_distance
                            )
                        )
            else:
                best_proxy_score = torch.full_like(
                    proxy_scores[..., 0, :], -torch.inf
                )
                for seed_index in selected_indices:
                    seed_score = torch.gather(
                        proxy_scores,
                        -2,
                        seed_index[..., None, None].expand(
                            *seed_index.shape, 1, proxy_scores.size(-1)
                        ),
                    ).squeeze(-2)
                    best_proxy_score = torch.maximum(best_proxy_score, seed_score)
            if config["seed"] == "q_hybrid":
                selected_vector = torch.gather(
                    distance_vectors,
                    3,
                    center[..., None].expand(
                        -1, -1, -1, -1, -1, -1, query.size(-1)
                    ),
                ).squeeze(3)
                minimum_distance = (
                    distance_vectors - selected_vector[:, :, :, None]
                ).square().mean(-1)
                for seed_index in selected_indices[1:]:
                    chosen_vector = torch.gather(
                        distance_vectors,
                        3,
                        seed_index[:, :, :, None, :, :, None].expand(
                            -1, -1, -1, 1, -1, -1, query.size(-1)
                        ),
                    ).squeeze(3)
                    distance = (
                        distance_vectors - chosen_vector[:, :, :, None]
                    ).square().mean(-1)
                    minimum_distance = torch.minimum(minimum_distance, distance)

        if config["seed"] == "q_response_exact_cover":
            used_last = rearrange(used, "b a k p j h -> b a k j h p")
            pair_cost = torch.minimum(
                minimum_response_cover[..., None, None, :],
                torch.minimum(
                    response_pair_distance.unsqueeze(-2),
                    response_pair_distance.unsqueeze(-3),
                ),
            ).sum(dim=-1)
            available = ~used_last
            valid_pair = available.unsqueeze(-1) & available.unsqueeze(-2)
            valid_pair = valid_pair & ~torch.eye(
                candidate_count, device=query.device, dtype=torch.bool
            )
            pair_index = pair_cost.masked_fill(~valid_pair, torch.inf).flatten(
                -2
            ).argmin(dim=-1)
            exact_choices = (
                pair_index // candidate_count,
                pair_index % candidate_count,
            )
            for choice in exact_choices:
                chosen = torch.gather(
                    candidates,
                    3,
                    choice[:, :, :, None, :, :, None].expand(
                        -1, -1, -1, 1, -1, -1, query.size(-1)
                    ),
                ).squeeze(3)
                seeds.append(chosen)
                used.scatter_(3, choice[:, :, :, None], True)

        fixed_seed_count = len(seeds)
        while len(seeds) < SEEDS:
            used_last = rearrange(used, "b a k p j h -> b a k j h p")
            if config["seed"] == "q_farthest":
                blocked = used
                side_rules = config.get("seed_selection_sides", [])
                adaptive_step = len(seeds) - fixed_seed_count
                if adaptive_step < len(side_rules) and side_rules[adaptive_step] != "any":
                    offsets = candidate_rows - key_row[:, None]
                    if side_rules[adaptive_step] == "left":
                        allowed = offsets < 0
                    else:
                        allowed = offsets > 0
                    has_allowed = allowed.any(dim=-1, keepdim=True)
                    allowed = torch.where(has_allowed, allowed, torch.ones_like(allowed))
                    allowed = allowed[None, None, :, :, None, None].expand_as(used)
                    blocked = blocked | ~allowed
                choice = minimum_distance.masked_fill(blocked, -1.0).argmax(dim=3)
            elif config["seed"] == "q_cover":
                cover_cost = torch.minimum(
                    minimum_cover_distance.unsqueeze(-2), pair_distance
                ).sum(dim=-1)
                choice = cover_cost.masked_fill(used_last, torch.inf).argmin(dim=-1)
            elif config["seed"] == "q_hybrid":
                hybrid_step = len(seeds) - fixed_seed_count
                if config["hybrid_order"][hybrid_step] == "farthest":
                    choice = minimum_distance.masked_fill(used, -1.0).argmax(dim=3)
                else:
                    coverage = torch.maximum(
                        best_proxy_score.unsqueeze(-2), proxy_scores
                    ).sum(dim=-1)
                    choice = coverage.masked_fill(used_last, -torch.inf).argmax(dim=-1)
            elif config["seed"] == "q_response_farthest":
                choice = minimum_response_distance.masked_fill(
                    used_last, -1.0
                ).argmax(dim=-1)
            elif config["seed"] == "q_response_cover":
                cover_cost = torch.minimum(
                    minimum_response_cover.unsqueeze(-2), response_pair_distance
                ).sum(dim=-1)
                choice = cover_cost.masked_fill(used_last, torch.inf).argmin(dim=-1)
            elif config["seed"] == "q_response_span":
                residual_energy = span_residual.square().sum(dim=-1)
                choice = residual_energy.masked_fill(used_last, -1.0).argmax(dim=-1)
            else:
                coverage = torch.maximum(
                    best_proxy_score.unsqueeze(-2), proxy_scores
                ).sum(dim=-1)
                choice = coverage.masked_fill(used_last, -torch.inf).argmax(dim=-1)
            chosen = torch.gather(
                candidates,
                3,
                choice[:, :, :, None, :, :, None].expand(
                    -1, -1, -1, 1, -1, -1, query.size(-1)
                ),
            ).squeeze(3)
            seeds.append(chosen)
            used.scatter_(3, choice[:, :, :, None], True)
            if config["seed"] == "q_farthest":
                chosen_vector = torch.gather(
                    distance_vectors,
                    3,
                    choice[:, :, :, None, :, :, None].expand(
                        -1, -1, -1, 1, -1, -1, query.size(-1)
                    ),
                ).squeeze(3)
                distance = (
                    distance_vectors - chosen_vector[:, :, :, None]
                ).square().mean(-1)
                minimum_distance = torch.minimum(minimum_distance, distance)
            elif config["seed"] == "q_cover":
                chosen_distance = torch.gather(
                    pair_distance,
                    -2,
                    choice[..., None, None].expand(*choice.shape, 1, candidate_count),
                ).squeeze(-2)
                minimum_cover_distance = torch.minimum(
                    minimum_cover_distance, chosen_distance
                )
            elif config["seed"] == "q_response_farthest":
                chosen_response = torch.gather(
                    response_vectors,
                    -2,
                    choice[..., None, None].expand(
                        *choice.shape, 1, response_vectors.size(-1)
                    ),
                ).squeeze(-2)
                response_distance = (
                    response_vectors - chosen_response.unsqueeze(-2)
                ).square().mean(dim=-1)
                minimum_response_distance = torch.minimum(
                    minimum_response_distance, response_distance
                )
            elif config["seed"] == "q_response_cover":
                chosen_distance = torch.gather(
                    response_pair_distance,
                    -2,
                    choice[..., None, None].expand(
                        *choice.shape, 1, candidate_count
                    ),
                ).squeeze(-2)
                minimum_response_cover = torch.minimum(
                    minimum_response_cover, chosen_distance
                )
            elif config["seed"] == "q_response_span":
                selected_residual = torch.gather(
                    span_residual,
                    -2,
                    choice[..., None, None].expand(
                        *choice.shape, 1, span_residual.size(-1)
                    ),
                ).squeeze(-2)
                basis = F.normalize(selected_residual, dim=-1)
                projection = (span_residual * basis.unsqueeze(-2)).sum(
                    dim=-1, keepdim=True
                )
                span_residual = span_residual - projection * basis.unsqueeze(-2)
            else:
                chosen_score = torch.gather(
                    proxy_scores,
                    -2,
                    choice[..., None, None].expand(
                        *choice.shape, 1, proxy_scores.size(-1)
                    ),
                ).squeeze(-2)
                best_proxy_score = torch.maximum(best_proxy_score, chosen_score)
                if config["seed"] == "q_hybrid":
                    chosen_vector = torch.gather(
                        distance_vectors,
                        3,
                        choice[:, :, :, None, :, :, None].expand(
                            -1, -1, -1, 1, -1, -1, query.size(-1)
                        ),
                    ).squeeze(3)
                    distance = (
                        distance_vectors - chosen_vector[:, :, :, None]
                    ).square().mean(-1)
                    minimum_distance = torch.minimum(minimum_distance, distance)
        seeds = torch.stack(seeds, dim=3)
        if "seed_centroid_blend" in config:
            distance = (
                candidates.float().unsqueeze(4) - seeds.float().unsqueeze(3)
            ).square().mean(dim=-1)
            candidate_cluster = distance.argmin(dim=4)
            membership = F.one_hot(candidate_cluster, SEEDS).movedim(-1, 4).float()
            count = membership.sum(dim=3)
            centroid = torch.einsum(
                "bakpcjh,bakpjhd->bakcjhd", membership, candidates.float()
            )
            centroid = centroid / count.clamp_min(1).unsqueeze(-1)
            original = seeds.float()
            blended = torch.lerp(
                original, centroid, config["seed_centroid_blend"]
            )
            blended = blended * (
                original.norm(dim=-1) / blended.norm(dim=-1).clamp_min(1e-12)
            ).unsqueeze(-1)
            active = count > 0
            if config.get("blend_adaptive_only", False):
                seed_index = torch.arange(SEEDS, device=query.device)
                active = active & (seed_index >= fixed_seed_count)[
                    None, None, None, :, None, None
                ]
            seeds = torch.where(
                active.unsqueeze(-1), blended, original
            ).to(query.dtype)
    else:
        rows = seed_rows(
            config["seed"],
            query.device,
            config.get("offsets"),
            config.get("boundary", "fallback"),
            config.get("num_seeds", SEEDS),
        )
        seed_pool = config.get("seed_pool", 1)
        if seed_pool == 1:
            seeds = query[:, :, rows, :, :, :]
        else:
            radius = seed_pool // 2
            pool_rows = rows[..., None] + torch.arange(
                -radius, radius + 1, device=query.device
            )
            seeds = query[:, :, pool_rows.clamp(0, ROWS - 1), :, :, :].mean(dim=4)
    # q/seeds: [B,A,K,C,J,H,D], key/value: [B,F,K,L,H,D]
    if config["seed"] == "score_quantile":
        pass
    elif config["score"] == "dual_probe":
        direction = torch.tensor([-1, -1, 1, 1], device=query.device)
        second_rows = rows + direction
        second_rows = torch.where(
            (second_rows >= 0) & (second_rows < ROWS),
            second_rows,
            rows - direction,
        )
        probe_rows = torch.stack((rows, second_rows), dim=-1)
        probes = query[:, :, probe_rows, :, :, :]
        probe_logits = torch.einsum(
            "bakcrjhd,bfklhd->bhafkjcrl", probes, key
        ).float()
        probe_logits *= scale
        logits = probe_logits.select(-2, 0)
        probe_log_probability = torch.log_softmax(probe_logits, dim=-1)
        scores = torch.logsumexp(probe_log_probability, dim=-2) - math.log(2.0)
    else:
        if seeds.ndim == 8:
            logits = torch.einsum(
                "bafkcjhd,bfklhd->bhafkjcl", seeds, key
            ).float()
        else:
            logits = torch.einsum(
                "bakcjhd,bfklhd->bhafkjcl", seeds, key
            ).float()
        logits *= scale

    if config["seed"] == "score_quantile":
        scores = logits
    elif config["score"] == "cosine":
        seed_norm = rearrange(
            seeds.float().norm(dim=-1),
            "b a k c j h -> b h a 1 k j c 1",
        )
        key_norm = rearrange(
            key.float().norm(dim=-1),
            "b f k l h -> b h 1 f k 1 1 l",
        )
        scores = logits / (seed_norm * key_norm).clamp_min(1e-12)
    elif config["score"] != "dual_probe":
        scores = assignment_scores(logits, config)
    if config.get("merge_from_assignment_scores", False):
        merge_logits = scores
    else:
        merge_logits = (
            logits
            if "merge_temperature" not in config
            else logits / config["merge_temperature"]
        )
    route_mode = config.get("route", "hard")
    if route_mode == "hard":
        if config.get("partition_rule") == "profile":
            assignment = profile_cluster_assignment(logits, config)
        elif quantile_assignment is None:
            assignment = torch.zeros_like(logits, dtype=torch.bool)
            assignment.scatter_(
                dim=-2, index=scores.argmax(dim=-2, keepdim=True), value=True
            )
        else:
            assignment = quantile_assignment
        if config.get("overlap_clusters", False):
            assignment = torch.ones_like(logits, dtype=torch.bool)
        if "value_reassign_scale" in config and not config.get(
            "overlap_clusters", False
        ):
            provisional_log_weights = torch.log_softmax(
                logits.masked_fill(~assignment, NEG_INF), dim=-1
            ).masked_fill(~assignment, 0.0)
            provisional_weights = provisional_log_weights.exp().masked_fill(
                ~assignment, 0.0
            )
            provisional_value = torch.einsum(
                "bhafkjcl,bfklhd->bafjkchd",
                provisional_weights.to(value.dtype),
                value,
            )
            if config.get("value_reassign_metric", "cosine") == "sqdist":
                value_similarity = -(
                    rearrange(
                        provisional_value.float(),
                        "b a f j k c h d -> b a f k j c 1 h d",
                    )
                    - rearrange(
                        value.float(),
                        "b f k l h d -> b 1 f k 1 1 l h d",
                    )
                ).square().mean(dim=-1)
                value_similarity = rearrange(
                    value_similarity,
                    "b a f k j c l h -> b h a f k j c l",
                )
            else:
                value_similarity = torch.einsum(
                    "bafjkchd,bfklhd->bhafkjcl",
                    F.normalize(provisional_value.float(), dim=-1),
                    F.normalize(value.float(), dim=-1),
                )
            if config.get("value_reassign_normalize") == "zscore":
                value_similarity = value_similarity - value_similarity.mean(
                    dim=-2, keepdim=True
                )
                value_similarity = value_similarity * torch.rsqrt(
                    value_similarity.var(dim=-2, keepdim=True, correction=0) + 1e-6
                )
            reassignment_scores = (
                scores + config["value_reassign_scale"] * value_similarity
            )
            assignment.zero_().scatter_(
                -2, reassignment_scores.argmax(dim=-2, keepdim=True), True
            )
        cluster_counts = assignment.sum(dim=-1)
        adaptive_anchor_weights = None
        if "adaptive_anchor_blend" in config:
            assignment_by_target = assignment.unsqueeze(-2)
            mass_source = logits if config.get("adaptive_anchor_mass") == "raw" else scores
            cluster_mass = torch.logsumexp(
                mass_source.unsqueeze(-3).masked_fill(~assignment_by_target, NEG_INF),
                dim=-1,
            )
            mass_weights = torch.softmax(cluster_mass, dim=-1)
            owner_weights = torch.eye(
                logits.size(-2), device=logits.device, dtype=logits.dtype
            )
            adaptive_anchor_weights = torch.lerp(
                owner_weights,
                mass_weights,
                config["adaptive_anchor_blend"],
            )
        elif "fixed_anchor_blend" in config:
            assignment_by_target = assignment.unsqueeze(-2)
            owner_weights = torch.eye(
                logits.size(-2), device=logits.device, dtype=logits.dtype
            )
            uniform_weights = torch.full_like(
                owner_weights, 1.0 / logits.size(-2)
            )
            adaptive_anchor_weights = torch.lerp(
                owner_weights,
                uniform_weights,
                config["fixed_anchor_blend"],
            )
        merge_rule = config.get("merge_rule", "current")
        if merge_rule == "anchor_lse":
            merge_logits = torch.logsumexp(scores, dim=-2, keepdim=True).expand_as(logits)
        elif merge_rule == "anchor_geom":
            merge_logits = logits.mean(dim=-2, keepdim=True).expand_as(logits)
        elif "assignment_margin_scale" in config:
            top2 = scores.topk(2, dim=-2).values
            margin = top2.select(-2, 0) - top2.select(-2, 1)
            merge_logits = logits + config["assignment_margin_scale"] * margin.unsqueeze(-2)
        elif "ownership_scale" in config:
            log_ownership = scores - torch.logsumexp(scores, dim=-2, keepdim=True)
            merge_logits = logits + config["ownership_scale"] * log_ownership
        if "merge_anchor_blend" in config:
            source = config["merge_anchor_source"]
            if source == "center":
                anchor_logits = logits[..., :1, :]
            elif source == "pair":
                anchor_logits = logits[..., :2, :].mean(dim=-2, keepdim=True)
            else:
                anchor_logits = logits.mean(dim=-2, keepdim=True)
            merge_logits = torch.lerp(
                merge_logits, anchor_logits, config["merge_anchor_blend"]
            )
        if (
            adaptive_anchor_weights is not None
            and config["adaptive_anchor_mode"] == "logit"
        ):
            merge_logits = (
                adaptive_anchor_weights.unsqueeze(-1) * logits.unsqueeze(-3)
            ).sum(dim=-2)
        profile_merge = config.get("profile_merge")
        if profile_merge == "uniform":
            merge_logits = torch.zeros_like(logits)
        elif profile_merge == "mean":
            merge_logits = logits.mean(dim=-2, keepdim=True).expand_as(logits)
        elif profile_merge in {"owner", "mix"}:
            cluster_seed_score = torch.einsum(
                "...cl,...sl->...cs", assignment.to(scores.dtype), scores
            )
            cluster_seed_score = cluster_seed_score / cluster_counts.clamp_min(
                1
            ).unsqueeze(-1)
            if profile_merge == "owner":
                owner = cluster_seed_score.argmax(dim=-1)
                merge_logits = torch.gather(
                    logits,
                    -2,
                    owner.unsqueeze(-1).expand(*owner.shape, logits.size(-1)),
                )
            else:
                probe_weight = torch.softmax(
                    cluster_seed_score / config["profile_mix_temperature"], dim=-1
                )
                merge_logits = torch.einsum(
                    "...cs,...sl->...cl", probe_weight, logits
                )
        if "merge_value_norm_scale" in config:
            value_log_norm = rearrange(
                value.float().norm(dim=-1).clamp_min(1e-12).log(),
                "b f k l h -> b h 1 f k 1 1 l",
            )
            merge_logits = (
                merge_logits
                + config["merge_value_norm_scale"] * value_log_norm
            )
        if "merge_query_mean_blend" in config:
            mean_query_logits = torch.einsum(
                "bajhd,bfklhd->bhafkjl", query.float().mean(dim=2), key.float()
            )
            mean_query_logits *= scale
            merge_logits = torch.lerp(
                merge_logits,
                mean_query_logits.unsqueeze(-2),
                config["merge_query_mean_blend"],
            )
        if "merge_value_similarity_scale" in config:
            merge_logits = (
                merge_logits
                + config["merge_value_similarity_scale"] * value_similarity
            )
        if "merge_value_refine_scale" in config:
            base_merge_logits = merge_logits
            for _ in range(config["merge_value_refine_steps"]):
                refine_logits = merge_logits * config.get(
                    "merge_logit_scale", 1.0
                )
                refine_logits = refine_logits.masked_fill(~assignment, NEG_INF)
                refine_weights = torch.softmax(refine_logits, dim=-1).masked_fill(
                    ~assignment, 0.0
                )
                refine_value = torch.einsum(
                    "bhafkjcl,bfklhd->bafjkchd",
                    refine_weights.to(value.dtype),
                    value,
                )
                refine_similarity = torch.einsum(
                    "bafjkchd,bfklhd->bhafkjcl",
                    F.normalize(refine_value.float(), dim=-1),
                    F.normalize(value.float(), dim=-1),
                )
                merge_logits = (
                    base_merge_logits
                    + config["merge_value_refine_scale"] * refine_similarity
                )
        selected = merge_logits * config.get("merge_logit_scale", 1.0)
        if "confidence_sharpen_strength" in config:
            # Sharpen only blocks whose merge distributions are confident.
            base_selected = selected.masked_fill(~assignment, NEG_INF)
            base_log_weights = torch.log_softmax(
                base_selected, dim=-1
            ).masked_fill(~assignment, 0.0)
            base_weights = base_log_weights.exp().masked_fill(
                ~assignment, 0.0
            )
            max_entropy = (
                cluster_counts.clamp_min(1).float().log().clamp_min(1.0)
            )
            normalized_entropy = -(
                base_weights * base_log_weights
            ).sum(dim=-1) / max_entropy
            active_clusters = cluster_counts > 0
            block_entropy = (
                normalized_entropy * active_clusters
            ).sum(dim=-1) / active_clusters.sum(dim=-1).clamp_min(1)
            sharpen = (
                config["confidence_sharpen_threshold"] - block_entropy
            ).clamp_min(0.0)
            gamma = 1.0 + config["confidence_sharpen_strength"] * sharpen
            selected = selected * gamma.unsqueeze(-1).unsqueeze(-1)
        selected = selected.masked_fill(~assignment, NEG_INF)
        log_weights = torch.log_softmax(selected, dim=-1).masked_fill(~assignment, 0.0)
        weights = log_weights.exp().masked_fill(~assignment, 0.0)
        proxy_anchor_weights = None
        if (
            "proxy_anchor_weight_mix" in config
            or "proxy_anchor_value_mix" in config
        ):
            proxy_assignment = assignment.unsqueeze(-3)
            proxy_log_weights = torch.log_softmax(
                value_fit_proxy_logits.unsqueeze(-2).masked_fill(
                    ~proxy_assignment, NEG_INF
                ),
                dim=-1,
            ).masked_fill(~proxy_assignment, 0.0)
            proxy_anchor_weights = proxy_log_weights.exp().masked_fill(
                ~proxy_assignment, 0.0
            ).mean(dim=-3)
        if "proxy_anchor_weight_mix" in config:
            weights = torch.lerp(
                weights,
                proxy_anchor_weights,
                config["proxy_anchor_weight_mix"],
            )
            log_weights = weights.clamp_min(1e-30).log()
        if merge_rule == "anchor_mix":
            anchor_assignment = assignment.unsqueeze(-3)
            anchor_logits = logits.unsqueeze(-2).masked_fill(~anchor_assignment, NEG_INF)
            anchor_log_weights = torch.log_softmax(anchor_logits, dim=-1)
            anchor_weights = anchor_log_weights.exp().masked_fill(~anchor_assignment, 0.0)
            weights = anchor_weights.mean(dim=-3)
            log_weights = weights.clamp_min(1e-30).log()
        if (
            adaptive_anchor_weights is not None
            and config["adaptive_anchor_mode"] == "probability"
        ):
            anchor_logits = logits.unsqueeze(-3).masked_fill(
                ~assignment_by_target, NEG_INF
            )
            anchor_log_weights = torch.log_softmax(anchor_logits, dim=-1)
            anchor_log_weights.masked_fill_(~assignment_by_target, 0.0)
            anchor_weights = anchor_log_weights.exp().masked_fill(
                ~assignment_by_target, 0.0
            )
            weights = (
                adaptive_anchor_weights.unsqueeze(-1) * anchor_weights
            ).sum(dim=-2)
            log_weights = weights.clamp_min(1e-30).log()
        base_routing_bias = (weights * log_weights).sum(dim=-1)
        if "topk_weight_count" in config:
            threshold = selected.topk(config["topk_weight_count"], dim=-1).values[
                ..., -1:
            ]
            topk_assignment = assignment & (selected >= threshold)
            topk_log_weights = torch.log_softmax(
                selected.masked_fill(~topk_assignment, NEG_INF), dim=-1
            ).masked_fill(~topk_assignment, 0.0)
            topk_weights = topk_log_weights.exp().masked_fill(
                ~topk_assignment, 0.0
            )
            weights = torch.lerp(
                weights, topk_weights, config["topk_weight_mix"]
            )
            log_weights = weights.clamp_min(1e-30).log()
        pre_peak_weights = weights
        if "adaptive_peak_mode" in config:
            if config["adaptive_peak_mode"] == "entropy":
                entropy = -(weights * log_weights).sum(dim=-1)
                peak_mix = config["adaptive_peak_scale"] * entropy / cluster_counts.clamp_min(
                    2
                ).float().log()
            else:
                peak_mix = config["adaptive_peak_scale"] * (
                    1.0 - weights.amax(dim=-1)
                )
            peak_weights = F.one_hot(
                weights.argmax(dim=-1), weights.size(-1)
            ).to(weights.dtype)
            peak_weights = peak_weights * (cluster_counts > 0).unsqueeze(-1)
            weights = torch.lerp(weights, peak_weights, peak_mix.clamp(0, 1).unsqueeze(-1))
            log_weights = weights.clamp_min(1e-30).log()
        if "peak_weight_mix" in config:
            peak_weights = F.one_hot(
                weights.argmax(dim=-1), weights.size(-1)
            ).to(weights.dtype)
            peak_weights = peak_weights * (cluster_counts > 0).unsqueeze(-1)
            weights = torch.lerp(
                weights, peak_weights, config["peak_weight_mix"]
            )
            log_weights = weights.clamp_min(1e-30).log()
        if (
            "merge_route_refine_mix" in config
            or "value_route_refine_mix" in config
        ):
            provisional_key = torch.einsum(
                "bhafkjcl,bfklhd->bafjkchd", weights.to(key.dtype), key
            )
            provisional_bias = (weights * log_weights).sum(dim=-1)
            provisional_logits = torch.einsum(
                "baijhd,bafjkchd->bhafkjic",
                query.float(),
                provisional_key.float(),
            )
            provisional_logits = (
                provisional_logits * scale - provisional_bias.unsqueeze(-2)
            )
            provisional_probability = rearrange(
                provisional_logits,
                "b h a f k j i c -> b h a j i (f k c)",
            ).softmax(dim=-1)
            provisional_probability = rearrange(
                provisional_probability,
                "b h a j i (f k c) -> b h a f k j i c",
                f=key.size(1),
                k=ROWS,
                c=SEEDS,
            )
            route_mass = provisional_probability.sum(dim=-2).clamp_min(1e-12)
            representative_query = torch.einsum(
                "bhafkjic,baijhd->bafkjchd",
                provisional_probability,
                query.float(),
            )
            representative_query = representative_query / rearrange(
                route_mass,
                "b h a f k j c -> b a f k j c h 1",
            )
            refined_logits = torch.einsum(
                "bafkjchd,bfklhd->bhafkjcl",
                representative_query,
                key.float(),
            )
            refined_logits = (refined_logits * scale).masked_fill(
                ~assignment, NEG_INF
            )
            refined_weights = torch.softmax(refined_logits, dim=-1).masked_fill(
                ~assignment, 0.0
            )
            route_refined_weights = refined_weights
            if "merge_route_refine_mix" in config:
                weights = torch.lerp(
                    weights,
                    refined_weights,
                    config["merge_route_refine_mix"],
                )
                log_weights = weights.clamp_min(1e-30).log()
        routing_bias = (weights * log_weights).sum(dim=-1)
        if "entropy_bias_mix" in config:
            routing_bias = torch.lerp(
                base_routing_bias, routing_bias, config["entropy_bias_mix"]
            )
        bias_rule = config.get("bias_rule", "entropy")
        if bias_rule != "entropy":
            anchor_assignment = assignment.unsqueeze(-3)
            anchor_logits = logits.unsqueeze(-2).masked_fill(~anchor_assignment, NEG_INF)
            anchor_log_mass = torch.logsumexp(anchor_logits, dim=-1)
            anchor_expected_logit = (
                weights.unsqueeze(-3) * logits.unsqueeze(-2)
            ).sum(dim=-1)
            anchor_intercept = anchor_log_mass - anchor_expected_logit
            if bias_rule == "anchor_mean":
                intercept = anchor_intercept.mean(dim=-2)
            else:
                intercept = torch.logsumexp(anchor_intercept, dim=-2) - math.log(SEEDS)
            routing_bias = -intercept
        routing_bias.masked_fill_(cluster_counts == 0, float("inf"))
    else:
        route_scores = scores / config["route_temperature"]
        if route_mode == "top2":
            threshold = route_scores.topk(2, dim=-2).values.select(-2, 1).unsqueeze(-2)
            route_scores = route_scores.masked_fill(route_scores < threshold, NEG_INF)
        log_responsibility = torch.log_softmax(route_scores, dim=-2)
        responsibility = log_responsibility.exp()
        cluster_counts = responsibility.sum(dim=-1)
        active = cluster_counts > 0
        log_weight_numerator = merge_logits + log_responsibility
        log_normalizer = torch.logsumexp(log_weight_numerator, dim=-1, keepdim=True)
        log_weights = torch.where(
            active.unsqueeze(-1), log_weight_numerator - log_normalizer, 0.0
        )
        weights = torch.where(active.unsqueeze(-1), log_weights.exp(), 0.0)
        if config.get("responsibility_correction", True):
            # Fractional partition tangent: H(alpha) + E_alpha[log responsibility].
            routing_bias = (weights * (log_weights - log_responsibility)).sum(dim=-1)
        else:
            # Ablation: treat fractional copies as full cluster members.
            routing_bias = (weights * log_weights).sum(dim=-1)
        routing_bias.masked_fill_(~active, float("inf"))

    if "bias_scale" in config:
        routing_bias = routing_bias * config["bias_scale"]
    if "count_bias_scale" in config:
        count_bias = cluster_counts.clamp_min(1).float().log()
        routing_bias = routing_bias - config["count_bias_scale"] * count_bias
    if config.get("variance_bias_mode") == "logit":
        mean_logit = (weights * logits).sum(dim=-1, keepdim=True)
        logit_variance = (weights * (logits - mean_logit).square()).sum(dim=-1)
        routing_bias = routing_bias - config["variance_bias_scale"] * logit_variance

    value_weights = weights
    if "value_unpeak_mix" in config:
        value_weights = torch.lerp(
            value_weights, pre_peak_weights, config["value_unpeak_mix"]
        )
    if route_refined_weights is not None and "value_route_refine_mix" in config:
        value_weights = torch.lerp(
            value_weights,
            route_refined_weights,
            config["value_route_refine_mix"],
        )
    if "proxy_anchor_value_mix" in config:
        value_weights = torch.lerp(
            value_weights,
            proxy_anchor_weights,
            config["proxy_anchor_value_mix"],
        )
    if "value_temperature" in config:
        value_logits = (logits / config["value_temperature"]).masked_fill(
            ~assignment, NEG_INF
        )
        value_log_weights = torch.log_softmax(value_logits, dim=-1)
        value_log_weights.masked_fill_(~assignment, 0.0)
        value_weights = value_log_weights.exp().masked_fill(~assignment, 0.0)
    if "value_uniform_mix" in config:
        uniform_weights = assignment.to(weights.dtype)
        uniform_weights = uniform_weights / cluster_counts.clamp_min(1).unsqueeze(-1)
        value_weights = torch.lerp(weights, uniform_weights, config["value_uniform_mix"])

    value_prior_weights = None
    if "postfit_value_prior_mix" in config:
        prior_logits = logits
        if config.get("postfit_value_prior_use_proxy", False):
            prior_logits = torch.cat(
                (prior_logits, value_fit_proxy_logits), dim=-2
            )
        if config["postfit_value_prior_reduce"] == "mean":
            prior_score = prior_logits.mean(dim=-2)
        else:
            prior_score = torch.logsumexp(prior_logits, dim=-2)
            prior_score = prior_score - math.log(prior_logits.size(-2))
        prior_score = prior_score.unsqueeze(-2).expand_as(assignment)
        prior_log_weights = torch.log_softmax(
            prior_score.masked_fill(~assignment, NEG_INF), dim=-1
        ).masked_fill(~assignment, 0.0)
        value_prior_weights = prior_log_weights.exp().masked_fill(
            ~assignment, 0.0
        )

    merged_key = torch.einsum("bhafkjcl,bfklhd->bafjkchd", weights.to(key.dtype), key)
    merged_value = torch.einsum(
        "bhafkjcl,bfklhd->bafjkchd", value_weights.to(value.dtype), value
    )
    if "curvature_bias_scale" in config:
        # Diagonal second-order correction to the entropy intercept.
        key_second_moment = torch.einsum(
            "bhafkjcl,bfklhd->bafjkchd", weights, key.float().square()
        )
        key_variance = (
            key_second_moment - merged_key.float().square()
        ).clamp_min(0.0)
        if seeds.ndim == 8:
            seed_prototypes = rearrange(
                seeds.float(), "b a f k c j h d -> b a f j k c h d"
            )
        else:
            seed_prototypes = rearrange(
                seeds.float(), "b a k c j h d -> b a 1 j k c h d"
            ).expand(-1, -1, key.size(1), -1, -1, -1, -1, -1)
        query_mean = query.float().mean(dim=2)
        query_second_moment = query.float().square().mean(dim=2)
        query_mean = rearrange(
            query_mean, "b a j h d -> b a 1 j 1 1 h d"
        )
        query_second_moment = rearrange(
            query_second_moment, "b a j h d -> b a 1 j 1 1 h d"
        )
        query_delta_second_moment = (
            query_second_moment
            - 2.0 * seed_prototypes * query_mean
            + seed_prototypes.square()
        )
        curvature = 0.5 * (scale * scale) * (
            key_variance * query_delta_second_moment
        ).sum(dim=-1)
        routing_bias = routing_bias - config["curvature_bias_scale"] * rearrange(
            curvature, "b a f j k c h -> b h a f k j c"
        )
    if "merged_key_norm_power" in config:
        key_square_norm = key.float().square().sum(dim=-1)
        target_square_norm = torch.einsum(
            "bhafkjcl,bfklh->bafjkch", weights, key_square_norm
        )
        norm_ratio = target_square_norm.clamp_min(1e-20).sqrt()
        norm_ratio = norm_ratio / merged_key.float().norm(dim=-1).clamp_min(1e-12)
        norm_ratio = norm_ratio.pow(config["merged_key_norm_power"]).clamp_max(2.0)
        merged_key = (merged_key.float() * norm_ratio.unsqueeze(-1)).to(key.dtype)
    if config.get("variance_bias_mode") == "key":
        key_norm = key.float().square().sum(dim=-1)
        second_moment = torch.einsum("bhafkjcl,bfklh->bhafkjc", weights, key_norm)
        mean_norm = rearrange(
            merged_key.float().square().sum(dim=-1),
            "b a f j k c h -> b h a f k j c",
        )
        key_variance = (second_moment - mean_norm).clamp_min(0) / key.size(-1)
        routing_bias = routing_bias - config["variance_bias_scale"] * key_variance

    if "logmass_key_fit_ridge" in config:
        fit_queries = seeds
        fit_logits = logits
        if config.get("logmass_key_fit_use_proxy", False):
            fit_queries = torch.cat((fit_queries, value_fit_proxy_queries), dim=4)
            fit_logits = torch.cat((fit_logits, value_fit_proxy_logits), dim=-2)

        active = cluster_counts > 0
        fit_assignment = assignment.unsqueeze(-3)
        target_log_mass = torch.logsumexp(
            fit_logits.unsqueeze(-2).masked_fill(~fit_assignment, NEG_INF),
            dim=-1,
        ).masked_fill(~active.unsqueeze(-2), 0.0)
        safe_bias = routing_bias.masked_fill(~active, 0.0)
        current_log_mass = (
            torch.einsum(
                "bafksjhd,bafjkchd->bhafkjsc",
                fit_queries.float(),
                merged_key.float(),
            )
            * scale
            - safe_bias.unsqueeze(-2)
        )
        residual = (target_log_mass - current_log_mass).masked_fill(
            ~active.unsqueeze(-2), 0.0
        )

        fit_queries_h = rearrange(
            fit_queries.float(),
            "b a f k s j h d -> b h a f k j s d",
        )
        gram = torch.matmul(fit_queries_h, fit_queries_h.transpose(-2, -1))
        identity = torch.eye(
            fit_queries_h.size(-2), device=query.device, dtype=torch.float32
        )
        coefficients = torch.linalg.solve(
            gram + config["logmass_key_fit_ridge"] * identity,
            residual / scale,
        )
        key_delta = torch.matmul(
            coefficients.transpose(-2, -1), fit_queries_h
        )
        merged_key_h = rearrange(
            merged_key.float(),
            "b a f j k c h d -> b h a f k j c d",
        )
        merged_key_h += config["logmass_key_fit_scale"] * key_delta
        merged_key = rearrange(
            merged_key_h,
            "b h a f k j c d -> b a f j k c h d",
        ).to(key.dtype)

    if "key_fit_ridge" in config:
        key_fit_queries = seeds
        key_fit_logits = logits
        if config.get("key_fit_proxy_anchors", False):
            key_fit_queries = torch.cat(
                (key_fit_queries, value_fit_proxy_queries), dim=4
            )
            key_fit_logits = torch.cat(
                (key_fit_logits, value_fit_proxy_logits), dim=-2
            )
        compressed_seed_logits = torch.einsum(
            "bafksjhd,bafjkchd->bhafkjsc",
            key_fit_queries.float(),
            merged_key.float(),
        )
        compressed_seed_attention = torch.softmax(
            compressed_seed_logits * scale - routing_bias.unsqueeze(-2), dim=-1
        )
        dense_seed_key = torch.einsum(
            "bhafkjsl,bfklhd->bhafkjsd",
            torch.softmax(key_fit_logits, dim=-1),
            key.float(),
        )
        merged_key_h = rearrange(
            merged_key.float(),
            "b a f j k c h d -> b h a f k j c d",
        )
        residual = dense_seed_key - torch.matmul(
            compressed_seed_attention, merged_key_h
        )
        attention_t = compressed_seed_attention.transpose(-2, -1)
        gram = torch.matmul(attention_t, compressed_seed_attention)
        identity = torch.eye(SEEDS, device=gram.device, dtype=gram.dtype)
        delta = torch.linalg.solve(
            gram + config["key_fit_ridge"] * identity,
            torch.matmul(attention_t, residual),
        )
        merged_key_h = merged_key_h + config["key_fit_scale"] * delta
        merged_key = rearrange(
            merged_key_h,
            "b h a f k j c d -> b a f j k c h d",
        ).to(key.dtype)

    if "bias_fit_ridge" in config:
        if seeds.ndim == 8:
            compressed_seed_logits = torch.einsum(
                "bafksjhd,bafjkchd->bhafkjsc",
                seeds.float(),
                merged_key.float(),
            )
        else:
            compressed_seed_logits = torch.einsum(
                "baksjhd,bafjkchd->bhafkjsc",
                seeds.float(),
                merged_key.float(),
            )
        dense_log_mass = torch.logsumexp(logits, dim=-1)
        identity = torch.eye(SEEDS, device=query.device, dtype=torch.float32)
        for _ in range(config.get("bias_fit_steps", 1)):
            compressed_logits = (
                compressed_seed_logits * scale - routing_bias.unsqueeze(-2)
            )
            compressed_attention = torch.softmax(compressed_logits, dim=-1)
            mass_residual = dense_log_mass - torch.logsumexp(
                compressed_logits, dim=-1
            )
            attention_t = compressed_attention.transpose(-2, -1)
            gram = torch.matmul(attention_t, compressed_attention)
            delta = torch.linalg.solve(
                gram + config["bias_fit_ridge"] * identity,
                torch.matmul(attention_t, mass_residual.unsqueeze(-1)),
            ).squeeze(-1)
            delta = delta.clamp(
                -config["bias_fit_clamp"], config["bias_fit_clamp"]
            )
            routing_bias = routing_bias - config["bias_fit_scale"] * delta

    if "value_fit_ridge" in config:
        fit_queries = seeds
        fit_logits = logits
        fit_equation_weights = [
            torch.ones(SEEDS, device=query.device, dtype=torch.float32)
        ]
        if value_fit_proxy_queries is not None:
            fit_queries = torch.cat((fit_queries, value_fit_proxy_queries), dim=4)
            fit_logits = torch.cat((fit_logits, value_fit_proxy_logits), dim=-2)
            fit_equation_weights.append(
                torch.full(
                    (value_fit_proxy_queries.size(4),),
                    config.get("value_fit_proxy_anchor_weight", 1.0),
                    device=query.device,
                    dtype=torch.float32,
                )
            )
        if "value_fit_seed_centroid_weight" in config:
            fit_queries = torch.cat(
                (fit_queries, seeds.mean(dim=4, keepdim=True)), dim=4
            )
            fit_logits = torch.cat(
                (fit_logits, logits.mean(dim=-2, keepdim=True)), dim=-2
            )
            fit_equation_weights.append(
                torch.tensor(
                    [config["value_fit_seed_centroid_weight"]],
                    device=query.device,
                    dtype=torch.float32,
                )
            )
        if "value_fit_proxy_centroid_weight" in config:
            fit_queries = torch.cat(
                (fit_queries, value_fit_proxy_queries.mean(dim=4, keepdim=True)),
                dim=4,
            )
            fit_logits = torch.cat(
                (fit_logits, value_fit_proxy_logits.mean(dim=-2, keepdim=True)),
                dim=-2,
            )
            fit_equation_weights.append(
                torch.tensor(
                    [config["value_fit_proxy_centroid_weight"]],
                    device=query.device,
                    dtype=torch.float32,
                )
            )
        if "value_fit_neighbor_offsets" in config:
            seed_rows_for_fit = rearrange(
                selected_seed_rows,
                "b a f k j h c -> b a f k c j h",
            )
            query_bank = query[:, :, None, None].expand(
                -1, -1, key.size(1), ROWS, -1, -1, -1, -1
            )
            neighbor_queries = []
            neighbor_logits = []
            for offset in config["value_fit_neighbor_offsets"]:
                neighbor_rows = seed_rows_for_fit + offset
                neighbor_rows = torch.where(
                    (neighbor_rows >= 0) & (neighbor_rows < ROWS),
                    neighbor_rows,
                    seed_rows_for_fit - offset,
                )
                neighbor = torch.gather(
                    query_bank,
                    4,
                    neighbor_rows.unsqueeze(-1).expand(
                        *neighbor_rows.shape, query.size(-1)
                    ),
                )
                neighbor_queries.append(neighbor)
                neighbor_logits.append(
                    torch.einsum(
                        "bafkcjhd,bfklhd->bhafkjcl", neighbor, key
                    ).float()
                    * scale
                )
            fit_queries = torch.cat((fit_queries, *neighbor_queries), dim=4)
            fit_logits = torch.cat((fit_logits, *neighbor_logits), dim=-2)
            fit_equation_weights.append(
                torch.full(
                    (SEEDS * len(neighbor_queries),),
                    config.get("value_fit_neighbor_weight", 1.0),
                    device=query.device,
                    dtype=torch.float32,
                )
            )
        if "value_fit_uniform_query_count" in config:
            uniform_count = config["value_fit_uniform_query_count"]
            uniform_rows = torch.linspace(
                0,
                ROWS - 1,
                uniform_count + 2,
                device=query.device,
            )[1:-1].round().long()
            uniform_queries = query.index_select(2, uniform_rows)
            uniform_queries = rearrange(
                uniform_queries,
                "b a s j h d -> b a 1 1 s j h d",
            ).expand(-1, -1, key.size(1), ROWS, -1, -1, -1, -1)
            uniform_logits = torch.einsum(
                "bafksjhd,bfklhd->bhafkjsl", uniform_queries, key
            ).float() * scale
            fit_queries = torch.cat((fit_queries, uniform_queries), dim=4)
            fit_logits = torch.cat((fit_logits, uniform_logits), dim=-2)
            fit_equation_weights.append(
                torch.full(
                    (uniform_count,),
                    config["value_fit_uniform_query_weight"],
                    device=query.device,
                    dtype=torch.float32,
                )
            )
        if fit_queries.ndim == 8:
            compressed_seed_logits = torch.einsum(
                "bafksjhd,bafjkchd->bhafkjsc",
                fit_queries.float(),
                merged_key.float(),
            )
        else:
            compressed_seed_logits = torch.einsum(
                "baksjhd,bafjkchd->bhafkjsc",
                fit_queries.float(),
                merged_key.float(),
            )
        compressed_seed_attention = torch.softmax(
            compressed_seed_logits * scale - routing_bias.unsqueeze(-2), dim=-1
        )
        dense_seed_output = torch.einsum(
            "bhafkjsl,bfklhd->bhafkjsd",
            torch.softmax(fit_logits, dim=-1),
            value.float(),
        )
        if "value_fit_adaptive_query_count" in config:
            candidate_count = config.get("seed_candidates", ROWS)
            key_row = torch.arange(ROWS, device=query.device)
            start = (key_row - candidate_count // 2).clamp(
                0, ROWS - candidate_count
            )
            candidate_rows = start[:, None] + torch.arange(
                candidate_count, device=query.device
            )
            candidate_queries = query[:, :, candidate_rows, :, :, :]
            candidate_queries = rearrange(
                candidate_queries,
                "b a k p j h d -> b h a 1 k j p d",
            ).expand(-1, -1, -1, key.size(1), -1, -1, -1, -1)
            candidate_logits = torch.einsum(
                "bhafkjpd,bafjkchd->bhafkjpc",
                candidate_queries.float(),
                merged_key.float(),
            )
            candidate_attention = torch.softmax(
                candidate_logits * scale - routing_bias.unsqueeze(-2), dim=-1
            )
            adaptive_count = config["value_fit_adaptive_query_count"]
            if config["value_fit_adaptive_query_mode"] == "entropy":
                selected_index = (
                    -(
                        candidate_attention
                        * candidate_attention.clamp_min(1e-30).log()
                    ).sum(dim=-1)
                ).topk(adaptive_count, dim=-1).indices
            else:
                reference = compressed_seed_attention
                selected = []
                used = torch.zeros(
                    candidate_attention.shape[:-1],
                    device=query.device,
                    dtype=torch.bool,
                )
                for _ in range(adaptive_count):
                    distance = (
                        candidate_attention.unsqueeze(-2)
                        - reference.unsqueeze(-3)
                    ).square().sum(dim=-1)
                    score = distance.amin(dim=-1).masked_fill(used, -1.0)
                    selected_index_i = score.argmax(dim=-1)
                    selected.append(selected_index_i)
                    used.scatter_(-1, selected_index_i.unsqueeze(-1), True)
                    selected_attention = torch.gather(
                        candidate_attention,
                        -2,
                        selected_index_i[..., None, None].expand(
                            *selected_index_i.shape, 1, SEEDS
                        ),
                    )
                    reference = torch.cat((reference, selected_attention), dim=-2)
                selected_index = torch.stack(selected, dim=-1)
            adaptive_queries = torch.gather(
                candidate_queries,
                -2,
                selected_index[..., None].expand(
                    *selected_index.shape, query.size(-1)
                ),
            )
            adaptive_attention = torch.gather(
                candidate_attention,
                -2,
                selected_index[..., None].expand(
                    *selected_index.shape, SEEDS
                ),
            )
            adaptive_logits = torch.einsum(
                "bhafkjsd,bfklhd->bhafkjsl",
                adaptive_queries.float(),
                key.float(),
            ) * scale
            adaptive_output = torch.einsum(
                "bhafkjsl,bfklhd->bhafkjsd",
                torch.softmax(adaptive_logits, dim=-1),
                value.float(),
            )
            compressed_seed_attention = torch.cat(
                (compressed_seed_attention, adaptive_attention), dim=-2
            )
            dense_seed_output = torch.cat(
                (dense_seed_output, adaptive_output), dim=-2
            )
            fit_equation_weights.append(
                torch.full(
                    (adaptive_count,),
                    config["value_fit_adaptive_query_weight"],
                    device=query.device,
                    dtype=torch.float32,
                )
            )
        if value_fit_sketch_queries is not None:
            sketch_compressed_logits = torch.einsum(
                "bafksjhd,bafjkchd->bhafkjsc",
                value_fit_sketch_queries.float(),
                merged_key.float(),
            )
            sketch_compressed_attention = torch.softmax(
                sketch_compressed_logits * scale - routing_bias.unsqueeze(-2),
                dim=-1,
            )
            compressed_seed_attention = torch.cat(
                (compressed_seed_attention, sketch_compressed_attention), dim=-2
            )
            dense_seed_output = torch.cat(
                (dense_seed_output, value_fit_sketch_targets.float()), dim=-2
            )
            fit_equation_weights.append(
                torch.full(
                    (value_fit_sketch_queries.size(4),),
                    config.get("value_fit_sketch_weight", 0.25),
                    device=query.device,
                    dtype=torch.float32,
                )
            )
        merged_value_h = rearrange(
            merged_value.float(),
            "b a f j k c h d -> b h a f k j c d",
        )
        residual = dense_seed_output - torch.matmul(
            compressed_seed_attention, merged_value_h
        )
        fit_attention = compressed_seed_attention
        fit_residual = residual
        if len(fit_equation_weights) > 1:
            equation_weight = torch.cat(fit_equation_weights).sqrt()
            fit_attention = fit_attention * equation_weight[None, None, None, None, None, None, :, None]
            fit_residual = fit_residual * equation_weight[None, None, None, None, None, None, :, None]
        attention_t = fit_attention.transpose(-2, -1)
        gram = torch.matmul(attention_t, fit_attention)
        identity = torch.eye(SEEDS, device=gram.device, dtype=gram.dtype)
        delta = torch.linalg.solve(
            gram + config["value_fit_ridge"] * identity,
            torch.matmul(attention_t, fit_residual),
        )
        target_merged_value_h = (
            merged_value_h + config.get("value_fit_scale", 1.0) * delta
        )
        if "value_weight_fw_steps" in config:
            projected_value = merged_value_h
            fit_target = (
                torch.matmul(fit_attention, merged_value_h) + fit_residual
            )
            value_h = rearrange(
                value.float(),
                "b f k l h d -> b h 1 f k 1 1 l d",
            ).expand(
                -1,
                -1,
                query.size(1),
                -1,
                -1,
                query.size(3),
                SEEDS,
                -1,
                -1,
            )
            for _ in range(config["value_weight_fw_steps"]):
                output_residual = (
                    torch.matmul(fit_attention, projected_value) - fit_target
                )
                value_gradient = torch.matmul(
                    fit_attention.transpose(-2, -1), output_residual
                )
                atom_score = torch.einsum(
                    "bhafkjcd,bfklhd->bhafkjcl",
                    value_gradient,
                    value.float(),
                ).masked_fill(~assignment, torch.inf)
                atom_index = atom_score.argmin(dim=-1)
                atom_value = torch.gather(
                    value_h,
                    -2,
                    atom_index[..., None, None].expand(
                        *atom_index.shape, 1, value.size(-1)
                    ),
                ).squeeze(-2)
                direction = (atom_value - projected_value).masked_fill(
                    (cluster_counts == 0).unsqueeze(-1), 0.0
                )
                output_direction = torch.matmul(fit_attention, direction)
                step_size = -(
                    output_residual * output_direction
                ).sum(dim=(-2, -1)) / output_direction.square().sum(
                    dim=(-2, -1)
                ).clamp_min(1e-12)
                step_size = step_size.clamp(0.0, 1.0)
                step_size *= config["value_weight_fw_step_scale"]
                projected_value = (
                    projected_value + step_size[..., None, None] * direction
                )
            merged_value_h = projected_value
        elif "value_fit_convex_steps" in config:
            projected_value = merged_value_h
            value_h = rearrange(
                value.float(),
                "b f k l h d -> b h 1 f k 1 1 l d",
            ).expand(
                -1,
                -1,
                query.size(1),
                -1,
                -1,
                query.size(3),
                SEEDS,
                -1,
                -1,
            )
            for _ in range(config["value_fit_convex_steps"]):
                projection_residual = projected_value - target_merged_value_h
                atom_score = torch.einsum(
                    "bhafkjcd,bfklhd->bhafkjcl",
                    projection_residual,
                    value.float(),
                ).masked_fill(~assignment, torch.inf)
                atom_index = atom_score.argmin(dim=-1)
                atom_value = torch.gather(
                    value_h,
                    -2,
                    atom_index[..., None, None].expand(
                        *atom_index.shape, 1, value.size(-1)
                    ),
                ).squeeze(-2)
                direction = atom_value - projected_value
                step_size = -(
                    projection_residual * direction
                ).sum(dim=-1) / direction.square().sum(dim=-1).clamp_min(1e-12)
                step_size = step_size.clamp(0.0, 1.0)
                step_size = step_size.masked_fill(cluster_counts == 0, 0.0)
                projected_value = (
                    projected_value + step_size.unsqueeze(-1) * direction
                )
            merged_value_h = projected_value
        else:
            merged_value_h = target_merged_value_h
        merged_value = rearrange(
            merged_value_h,
            "b h a f k j c d -> b a f j k c h d",
        ).to(value.dtype)

    if value_prior_weights is not None:
        value_prior = torch.einsum(
            "bhafkjcl,bfklhd->bafjkchd",
            value_prior_weights.to(value.dtype),
            value,
        )
        merged_value = torch.lerp(
            merged_value,
            value_prior,
            config["postfit_value_prior_mix"],
        )

    if "global_value_probe_count" in config:
        probe_count = config["global_value_probe_count"]
        probe_layout = config.get("global_value_probe_layout", "interior")
        precomputed_probe_probability = None
        if probe_layout == "route_pivoted":
            full_compressed_logits = torch.einsum(
                "baijhd,bafjkchd->bhaijfkc",
                query.float(),
                merged_key.float(),
            ) * scale
            full_compressed_logits -= rearrange(
                routing_bias,
                "b h a f k j c -> b h a 1 j f k c",
            )
            full_compressed_probability = torch.softmax(
                full_compressed_logits.flatten(-3), dim=-1
            ).reshape_as(full_compressed_logits)
            route_profile = rearrange(
                full_compressed_probability,
                "b h a i j f k c -> b a j h i (f k c)",
            )
            selected_rows = [
                torch.full(
                    route_profile.shape[:-2],
                    query.size(2) // 2,
                    device=query.device,
                    dtype=torch.long,
                )
            ]
            residual_profile = route_profile
            for _ in range(probe_count):
                pivot_profile = torch.gather(
                    residual_profile,
                    -2,
                    selected_rows[-1][..., None, None].expand(
                        *selected_rows[-1].shape, 1, route_profile.size(-1)
                    ),
                ).squeeze(-2)
                pivot_profile = F.normalize(pivot_profile, dim=-1)
                residual_profile = residual_profile - torch.einsum(
                    "bajhim,bajhm->bajhi",
                    residual_profile,
                    pivot_profile,
                ).unsqueeze(-1) * pivot_profile.unsqueeze(-2)
                if len(selected_rows) < probe_count:
                    residual_norm = residual_profile.square().sum(dim=-1)
                    for selected_row in selected_rows:
                        residual_norm.scatter_(
                            -1, selected_row.unsqueeze(-1), -1.0
                        )
                    selected_rows.append(residual_norm.argmax(dim=-1))
            probe_rows = torch.stack(selected_rows, dim=-1)
            query_by_column = rearrange(
                query.float(),
                "b a i j h d -> b a j h i d",
            )
            probe_query = torch.gather(
                query_by_column,
                -2,
                probe_rows[..., None].expand(*probe_rows.shape, query.size(-1)),
            )
            probe_query = rearrange(
                probe_query,
                "b a j h s d -> b a s j h d",
            )
            probability_by_column = rearrange(
                full_compressed_probability,
                "b h a i j f k c -> b a j h i f k c",
            )
            precomputed_probe_probability = torch.gather(
                probability_by_column,
                4,
                probe_rows[..., None, None, None].expand(
                    *probe_rows.shape,
                    probability_by_column.size(-3),
                    probability_by_column.size(-2),
                    probability_by_column.size(-1),
                ),
            )
            precomputed_probe_probability = rearrange(
                precomputed_probe_probability,
                "b a j h s f k c -> b h a s j f k c",
            )
        elif probe_layout in {"query_farthest", "query_pivoted"}:
            query_by_column = rearrange(
                query.float(),
                "b a i j h d -> b a j h i d",
            )
            selected_rows = [
                torch.full(
                    query_by_column.shape[:-2],
                    query.size(2) // 2,
                    device=query.device,
                    dtype=torch.long,
                )
            ]
            if probe_layout == "query_farthest":
                query_square_norm = query_by_column.square().sum(dim=-1)
                minimum_distance = torch.full_like(
                    query_square_norm, float("inf")
                )
                for _ in range(1, probe_count):
                    selected_query = torch.gather(
                        query_by_column,
                        -2,
                        selected_rows[-1][..., None, None].expand(
                            *selected_rows[-1].shape, 1, query.size(-1)
                        ),
                    ).squeeze(-2)
                    distance = query_square_norm + selected_query.square().sum(
                        dim=-1, keepdim=True
                    )
                    distance -= 2 * torch.einsum(
                        "bajhid,bajhd->bajhi",
                        query_by_column,
                        selected_query,
                    )
                    minimum_distance = torch.minimum(minimum_distance, distance)
                    minimum_distance.scatter_(
                        -1, selected_rows[-1].unsqueeze(-1), -1.0
                    )
                    selected_rows.append(minimum_distance.argmax(dim=-1))
            else:
                residual_query = query_by_column.clone()
                for _ in range(probe_count):
                    pivot_query = torch.gather(
                        residual_query,
                        -2,
                        selected_rows[-1][..., None, None].expand(
                            *selected_rows[-1].shape, 1, query.size(-1)
                        ),
                    ).squeeze(-2)
                    pivot_query = F.normalize(pivot_query, dim=-1)
                    residual_query -= torch.einsum(
                        "bajhid,bajhd->bajhi",
                        residual_query,
                        pivot_query,
                    ).unsqueeze(-1) * pivot_query.unsqueeze(-2)
                    if len(selected_rows) < probe_count:
                        residual_norm = residual_query.square().sum(dim=-1)
                        for selected_row in selected_rows:
                            residual_norm.scatter_(
                                -1, selected_row.unsqueeze(-1), -1.0
                            )
                        selected_rows.append(residual_norm.argmax(dim=-1))
            probe_rows = torch.stack(selected_rows, dim=-1)
            probe_query = torch.gather(
                query_by_column,
                -2,
                probe_rows[..., None].expand(*probe_rows.shape, query.size(-1)),
            )
            probe_query = rearrange(
                probe_query,
                "b a j h s d -> b a s j h d",
            )
        elif probe_layout == "segment_mean":
            boundaries = [
                round(i * query.size(2) / probe_count)
                for i in range(probe_count + 1)
            ]
            probe_query = torch.stack(
                [
                    query[:, :, boundaries[i] : boundaries[i + 1]].float().mean(dim=2)
                    for i in range(probe_count)
                ],
                dim=2,
            )
        else:
            if probe_layout == "endpoints":
                probe_rows = torch.linspace(
                    0,
                    query.size(2) - 1,
                    probe_count,
                    device=query.device,
                )
            elif probe_layout == "midpoints":
                probe_rows = (
                    (torch.arange(probe_count, device=query.device) + 0.5)
                    * query.size(2)
                    / probe_count
                    - 0.5
                )
            else:
                probe_rows = torch.linspace(
                    0,
                    query.size(2) - 1,
                    probe_count + 2,
                    device=query.device,
                )[1:-1]
            probe_row_indices = probe_rows.round().long()
            probe_query = query.index_select(2, probe_row_indices).float()

        if precomputed_probe_probability is None:
            compressed_probe_logits = torch.einsum(
                "basjhd,bafjkchd->bhasjfkc",
                probe_query,
                merged_key.float(),
            ) * scale
            compressed_probe_logits -= rearrange(
                routing_bias,
                "b h a f k j c -> b h a 1 j f k c",
            )
            compressed_probe_probability = torch.softmax(
                compressed_probe_logits.flatten(-3), dim=-1
            ).reshape_as(compressed_probe_logits)
        else:
            compressed_probe_probability = precomputed_probe_probability

        dense_probe_outputs = []
        local_key_rows = config.get("global_value_local_key_rows")
        top_key_rows = config.get("global_value_top_key_rows")
        top_key_blocks = config.get("global_value_top_key_blocks")
        top_block_score_mode = config.get(
            "global_value_top_block_score", "mass"
        )
        top_block_value_std = None
        top_block_key_std = None
        if top_key_blocks is not None and top_block_score_mode != "mass":
            if top_block_score_mode in {"value_std", "value_key_std"}:
                value_square_norm = value.float().square().sum(dim=-1)
                value_second_moment = torch.einsum(
                    "bhafkjcl,bfklh->bhafkjc",
                    value_weights,
                    value_square_norm,
                )
                merged_value_square_norm = rearrange(
                    merged_value.float().square().sum(dim=-1),
                    "b a f j k c h -> b h a f k j c",
                )
                top_block_value_std = rearrange(
                    (value_second_moment - merged_value_square_norm)
                    .clamp_min(0.0)
                    .sqrt(),
                    "b h a f k j c -> b h a j f k c",
                )
            if top_block_score_mode in {"key_std", "value_key_std"}:
                key_square_norm = key.float().square().sum(dim=-1)
                key_second_moment = torch.einsum(
                    "bhafkjcl,bfklh->bhafkjc",
                    weights,
                    key_square_norm,
                )
                merged_key_square_norm = rearrange(
                    merged_key.float().square().sum(dim=-1),
                    "b a f j k c h -> b h a f k j c",
                )
                top_block_key_std = rearrange(
                    (key_second_moment - merged_key_square_norm)
                    .clamp_min(0.0)
                    .sqrt(),
                    "b h a f k j c -> b h a j f k c",
                )
        if (
            local_key_rows is None
            and top_key_rows is None
            and top_key_blocks is None
        ):
            for probe_query_chunk in probe_query.split(
                config.get("global_value_probe_chunk", probe_count), dim=2
            ):
                dense_probe_logits = torch.einsum(
                    "basjhd,bfklhd->bhasjfkl",
                    probe_query_chunk,
                    key.float(),
                ) * scale
                dense_probe_probability = torch.softmax(
                    dense_probe_logits.flatten(-3), dim=-1
                ).reshape_as(dense_probe_logits)
                dense_probe_outputs.append(
                    torch.einsum(
                        "bhasjfkl,bfklhd->bhasjd",
                        dense_probe_probability,
                        value.float(),
                    )
                )
            dense_probe_output = torch.cat(dense_probe_outputs, dim=3)
        elif local_key_rows is not None:
            key_row = torch.arange(ROWS, device=query.device)
            for probe_index, probe_row in enumerate(probe_row_indices):
                local_start = (probe_row - local_key_rows // 2).clamp(
                    0, ROWS - local_key_rows
                )
                local_rows = local_start + torch.arange(
                    local_key_rows, device=query.device
                )
                local_key = key.index_select(2, local_rows).float()
                local_value = value.index_select(2, local_rows).float()
                local_logits = torch.einsum(
                    "bajhd,bfwlhd->bhajfwl",
                    probe_query[:, :, probe_index],
                    local_key,
                ) * scale

                remote_logits = compressed_probe_logits[:, :, :, probe_index]
                keep_remote_row = torch.ones(
                    ROWS, device=query.device, dtype=torch.bool
                )
                keep_remote_row[local_rows] = False
                remote_logits = remote_logits.masked_fill(
                    ~keep_remote_row[None, None, None, None, None, :, None],
                    NEG_INF,
                )
                maximum = torch.maximum(
                    remote_logits.amax(dim=(-3, -2, -1), keepdim=True),
                    local_logits.amax(dim=(-3, -2, -1), keepdim=True),
                )
                remote_weight = torch.exp(remote_logits - maximum)
                local_weight = torch.exp(local_logits - maximum)
                normalizer = remote_weight.sum(dim=(-3, -2, -1))
                normalizer += local_weight.sum(dim=(-3, -2, -1))
                remote_output = torch.einsum(
                    "bhajfkc,bafjkchd->bhajd",
                    remote_weight,
                    merged_value.float(),
                )
                local_output = torch.einsum(
                    "bhajfwl,bfwlhd->bhajd",
                    local_weight,
                    local_value,
                )
                dense_probe_outputs.append(
                    (remote_output + local_output)
                    / normalizer.clamp_min(1e-30).unsqueeze(-1)
                )
            dense_probe_output = torch.stack(dense_probe_outputs, dim=3)
        else:
            for probe_index in range(probe_count):
                compressed_probability = compressed_probe_probability[
                    :, :, :, probe_index
                ]
                if top_key_rows is not None:
                    row_mass = compressed_probability.sum(dim=(-3, -1))
                    selected_rows = row_mass.topk(top_key_rows, dim=-1).indices
                    selected_row_mask = F.one_hot(
                        selected_rows, num_classes=ROWS
                    ).any(dim=-2)
                    selected_block_mask = selected_row_mask.unsqueeze(-2).expand(
                        *selected_row_mask.shape[:-1], key.size(1), ROWS
                    )
                else:
                    block_score = compressed_probability
                    if top_block_score_mode == "value_std":
                        block_score = block_score * top_block_value_std
                    elif top_block_score_mode == "key_std":
                        block_score = block_score * top_block_key_std
                    elif top_block_score_mode == "value_key_std":
                        value_scale = top_block_value_std.mean(
                            dim=(-3, -2, -1), keepdim=True
                        ).clamp_min(1e-12)
                        key_scale = top_block_key_std.mean(
                            dim=(-3, -2, -1), keepdim=True
                        ).clamp_min(1e-12)
                        block_score = block_score * (
                            top_block_value_std / value_scale
                            + top_block_key_std / key_scale
                        )
                    block_score = block_score.sum(dim=-1)
                    block_index = block_score.flatten(-2).topk(
                        top_key_blocks, dim=-1
                    ).indices
                    selected_block_mask = F.one_hot(
                        block_index, num_classes=key.size(1) * ROWS
                    ).any(dim=-2)
                    selected_block_mask = selected_block_mask.reshape(
                        *selected_block_mask.shape[:-1], key.size(1), ROWS
                    )

                exact_logits = torch.einsum(
                    "bajhd,bfklhd->bhajfkl",
                    probe_query[:, :, probe_index],
                    key.float(),
                ) * scale
                exact_logits = exact_logits.masked_fill(
                    ~selected_block_mask[..., None],
                    NEG_INF,
                )
                remote_logits = compressed_probe_logits[:, :, :, probe_index]
                remote_logits = remote_logits.masked_fill(
                    selected_block_mask[..., None],
                    NEG_INF,
                )
                maximum = torch.maximum(
                    remote_logits.amax(dim=(-3, -2, -1), keepdim=True),
                    exact_logits.amax(dim=(-3, -2, -1), keepdim=True),
                )
                remote_weight = torch.exp(remote_logits - maximum)
                exact_weight = torch.exp(exact_logits - maximum)
                normalizer = remote_weight.sum(dim=(-3, -2, -1))
                normalizer += exact_weight.sum(dim=(-3, -2, -1))
                remote_output = torch.einsum(
                    "bhajfkc,bafjkchd->bhajd",
                    remote_weight,
                    merged_value.float(),
                )
                exact_output = torch.einsum(
                    "bhajfkl,bfklhd->bhajd",
                    exact_weight,
                    value.float(),
                )
                dense_probe_outputs.append(
                    (remote_output + exact_output)
                    / normalizer.clamp_min(1e-30).unsqueeze(-1)
                )
            dense_probe_output = torch.stack(dense_probe_outputs, dim=3)

        compressed_probe_output = torch.einsum(
            "bhasjfkc,bafjkchd->bhasjd",
            compressed_probe_probability,
            merged_value.float(),
        )
        probe_residual = compressed_probe_output - dense_probe_output
        merged_value_h = rearrange(
            merged_value.float(),
            "b a f j k c h d -> b h a f k j c d",
        )
        if config.get("global_value_probe_solver") == "least_squares":
            probe_probability = rearrange(
                compressed_probe_probability,
                "b h a s j f k c -> b h a j s (f k c)",
            )
            probe_residual = rearrange(
                probe_residual,
                "b h a s j d -> b h a j s d",
            )
            gram = torch.einsum(
                "bhajsm,bhajtm->bhajst",
                probe_probability,
                probe_probability,
            )
            identity = torch.eye(
                probe_count, device=query.device, dtype=gram.dtype
            )
            ridge = config["global_value_probe_ridge"] * gram.diagonal(
                dim1=-2, dim2=-1
            ).mean(dim=-1, keepdim=True).clamp_min(1e-12)
            coefficients = torch.linalg.solve(
                gram + ridge.unsqueeze(-1) * identity,
                -probe_residual,
            )
            value_delta = torch.einsum(
                "bhajsm,bhajsd->bhajmd",
                probe_probability,
                coefficients,
            )
            value_delta = rearrange(
                value_delta,
                "b h a j (f k c) d -> b h a f k j c d",
                f=key.size(1),
                k=ROWS,
                c=SEEDS,
            )
            merged_value_h += config["global_value_probe_scale"] * value_delta
        else:
            value_direction = -torch.einsum(
                "bhasjfkc,bhasjd->bhafkjcd",
                compressed_probe_probability,
                probe_residual,
            )
            output_direction = torch.einsum(
                "bhasjfkc,bhafkjcd->bhasjd",
                compressed_probe_probability,
                value_direction,
            )
            step_size = -(
                probe_residual * output_direction
            ).sum(dim=(-3, -1)) / output_direction.square().sum(
                dim=(-3, -1)
            ).clamp_min(1e-12)
            if config.get("global_value_probe_exact_line_search", False):
                step_size = step_size.clamp_min(0.0)
            else:
                step_size = step_size.clamp(0.0, 1.0)
            step_size *= config["global_value_probe_scale"]
            merged_value_h += (
                step_size[:, :, :, None, None, :, None, None] * value_direction
            )
        merged_value = rearrange(
            merged_value_h,
            "b h a f k j c d -> b a f j k c h d",
        ).to(value.dtype)

    attention_correction = None
    value_cross_covariance = None
    needs_seed_token = (
        "query_diag_variance_scale" in config
        or "query_value_covariance_scale" in config
    )
    seed_for_token = None
    if needs_seed_token:
        seed_for_token = seeds[:, :, None].expand(
            -1, -1, key.size(1), -1, -1, -1, -1, -1
        )
        seed_for_token = rearrange(
            seed_for_token,
            "b a f k c j h d -> (b h a j) 1 1 (f k c) d",
        )
    if "query_diag_variance_scale" in config:
        second_moment = torch.einsum(
            "bhafkjcl,bfklhd->bafjkchd", weights, key.float().square()
        )
        diagonal_variance = (second_moment - merged_key.float().square()).clamp_min(0)
        variance_for_token = rearrange(
            diagonal_variance,
            "b a f j k c h d -> (b h a j) 1 1 (f k c) d",
        )
        query_for_token = rearrange(
            query, "b a i j h d -> (b h a j) 1 i 1 d"
        ).float()
        delta = query_for_token - seed_for_token.float()
        attention_correction = (
            config["query_diag_variance_scale"]
            * scale**2
            * (variance_for_token * delta.square()).sum(dim=-1)
        )
    if "query_value_covariance_scale" in config:
        value_key_moment = torch.einsum(
            "bhafkjcl,bfklhd->bafjkchd",
            weights,
            value.float() * key.float(),
        )
        value_cross_covariance = value_key_moment - (
            merged_value.float() * merged_key.float()
        )
        value_cross_covariance = rearrange(
            value_cross_covariance,
            "b a f j k c h d -> (b h a j) 1 (f k c) d",
        )

    q = rearrange(query, "b a i j h d -> (b h a j) 1 i d")
    k = rearrange(merged_key, "b a f j k c h d -> (b h a j) 1 (f k c) d")
    v = rearrange(merged_value, "b a f j k c h d -> (b h a j) 1 (f k c) d")
    bias = rearrange(routing_bias, "b h a f k j c -> (b h a j) 1 1 (f k c)")
    attention_bias = (-bias).to(q.dtype)
    if attention_correction is not None:
        attention_bias = attention_bias + attention_correction.to(q.dtype)
    if value_cross_covariance is None:
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            dropout_p=0.0,
            is_causal=False,
            scale=scale * config.get("attention_scale_factor", 1.0),
        )
    else:
        attention_logits = torch.matmul(q.float(), k.float().transpose(-2, -1))
        attention_logits = attention_logits * scale + attention_bias.float()
        attention_weights = torch.softmax(attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v.float())
        seed_for_token = seed_for_token.squeeze(2)
        weighted_cross = torch.matmul(attention_weights, value_cross_covariance)
        weighted_cross_seed = torch.matmul(
            attention_weights, value_cross_covariance * seed_for_token.float()
        )
        output = output + config["query_value_covariance_scale"] * scale * (
            q.float() * weighted_cross - weighted_cross_seed
        )
    output = rearrange(
        output.squeeze(1),
        "(b h a j) i d -> b a i j h d",
        b=query.size(0),
        h=query.size(4),
        a=query.size(1),
        j=query.size(3),
    )
    return output, cluster_counts


def parse_ints(value):
    return [int(item) for item in value.replace(",", " ").replace(":", " ").split()]


def measure_cuda_ms(function):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def install_probe(pipe, args, stats):
    layer_count = len(pipe.transformer.blocks)
    head_count = pipe.transformer.blocks[0].attn1.heads
    selected_layers = set(range(layer_count) if args.layers == "all" else parse_ints(args.layers))
    selected_heads = list(range(head_count) if args.heads == "all" else parse_ints(args.heads))
    active_variants = (
        list(VARIANTS)
        if args.variants == "all"
        else args.variants.replace(",", " ").replace(":", " ").split()
    )
    unknown = set(active_variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")
    if "baseline" not in active_variants:
        raise ValueError("The baseline variant is required for relative metrics.")
    if args.check_triton_wide and "seed_wide" not in active_variants:
        raise ValueError("Triton-wide validation requires the seed_wide reference.")
    baseline_triton = None
    if args.triton_baseline_module:
        spec = importlib.util.spec_from_file_location(
            "hard_c4_720p_triton_baseline", args.triton_baseline_module
        )
        baseline_triton = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline_triton)
    result_variants = list(active_variants)
    if args.check_triton_wide:
        result_variants.append("triton_wide")
    if baseline_triton is not None:
        result_variants.append("triton_baseline")
    total_calls = args.steps * len(pipe.transformer.blocks)
    state = {
        "call": 0,
        "evaluated": [],
        "layers": sorted(selected_layers),
        "heads": selected_heads,
        "variants": result_variants,
        "timings_ms": defaultdict(list),
    }

    class Processor:
        def __init__(self, layer):
            self.layer = layer

        def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            rotary_emb=None,
            timestep=None,
        ):
            del timestep
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states

            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)
            query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
            key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
            value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
            if rotary_emb is not None:
                dtype = torch.float64
                q_complex = torch.view_as_complex(query.to(dtype).unflatten(3, (-1, 2)))
                k_complex = torch.view_as_complex(key.to(dtype).unflatten(3, (-1, 2)))
                query = torch.view_as_real(q_complex * rotary_emb).flatten(3, 4).type_as(query)
                key = torch.view_as_real(k_complex * rotary_emb).flatten(3, 4).type_as(key)

            step = state["call"] // len(pipe.transformer.blocks)
            if self.layer in selected_layers:
                frame = (step * 7 + self.layer * 3) % FRAMES
                heads = torch.tensor(selected_heads, device=query.device)
                q_head = query.index_select(1, heads).reshape(
                    query.size(0), len(heads), FRAMES, ROWS, COLS, query.size(-1)
                )
                k_head = key.index_select(1, heads).reshape(
                    key.size(0), len(heads), FRAMES, ROWS, COLS, key.size(-1)
                )
                v_head = value.index_select(1, heads).reshape_as(k_head)
                q_block = q_head[:, :, frame : frame + 1].permute(0, 2, 3, 4, 1, 5)
                k_block = k_head.permute(0, 2, 3, 4, 1, 5)
                v_block = v_head.permute(0, 2, 3, 4, 1, 5)
                if args.time_triton:
                    dense_selected = None
                    timed_query = query if args.time_dense_all_heads else q_head
                    timed_key = key if args.time_dense_all_heads else k_head
                    timed_value = value if args.time_dense_all_heads else v_head

                    def run_dense():
                        nonlocal dense_selected
                        dense_selected = F.scaled_dot_product_attention(
                            timed_query,
                            timed_key,
                            timed_value,
                            dropout_p=0.0,
                            is_causal=False,
                        )

                    state["timings_ms"]["sdpa"].append(measure_cuda_ms(run_dense))
                    if args.time_dense_all_heads:
                        dense_selected = dense_selected.index_select(1, heads)
                    dense = dense_selected.unflatten(2, (FRAMES, ROWS, COLS))[:, :, frame]
                else:
                    dense = F.scaled_dot_product_attention(
                        q_head[:, :, frame].flatten(2, 3),
                        k_head.flatten(2, 4),
                        v_head.flatten(2, 4),
                        dropout_p=0.0,
                        is_causal=False,
                    )
                dense = dense.reshape(
                    dense.size(0), dense.size(1), ROWS, COLS, dense.size(-1)
                ).permute(0, 2, 3, 1, 4).unsqueeze(1)
                scale = query.size(-1) ** -0.5
                wide_counts = None
                wide_output = None
                stats_counts = {}
                timed_hard_output = (
                    torch.empty_like(query) if args.time_triton_all_heads_chunked else None
                )
                for name in active_variants:
                    config = VARIANTS[name]
                    output, counts = hard_c4_sample(q_block, k_block, v_block, scale, config)
                    stats[name].add(output, dense, counts)
                    stats_counts[name] = counts
                    if name == "seed_wide":
                        wide_counts = counts
                        wide_output = output
                if args.check_triton_wide:
                    from attention.hard_c4_wan21_14b_720p_triton import (
                        hard_c4_720p_attention_triton_head_major,
                    )

                    triton_output = torch.empty_like(q_head)
                    run_wide = lambda: hard_c4_720p_attention_triton_head_major(
                        q_head.flatten(2, 4),
                        k_head.flatten(2, 4),
                        v_head.flatten(2, 4),
                        output=triton_output.flatten(2, 4),
                    )
                    if args.time_triton:
                        if args.time_triton_all_heads_chunked:
                            run_wide()

                            def run_wide_all_heads():
                                for head_start in range(0, query.size(1), args.triton_head_chunk):
                                    head_end = head_start + args.triton_head_chunk
                                    hard_c4_720p_attention_triton_head_major(
                                        query[:, head_start:head_end],
                                        key[:, head_start:head_end],
                                        value[:, head_start:head_end],
                                        output=timed_hard_output[:, head_start:head_end],
                                    )

                            elapsed = measure_cuda_ms(run_wide_all_heads)
                        else:
                            elapsed = measure_cuda_ms(run_wide)
                        state["timings_ms"]["triton_wide"].append(elapsed)
                    else:
                        run_wide()
                    triton_frame = triton_output[:, :, frame].permute(0, 2, 3, 1, 4).unsqueeze(1)
                    stats["triton_wide"].add(triton_frame, dense, wide_counts)
                    stats["triton_wide_vs_pytorch_wide"].add(
                        triton_frame, wide_output, wide_counts
                    )
                if baseline_triton is not None:
                    baseline_output = torch.empty_like(q_head)
                    run_baseline = lambda: baseline_triton.hard_c4_720p_attention_triton_head_major(
                        q_head.flatten(2, 4),
                        k_head.flatten(2, 4),
                        v_head.flatten(2, 4),
                        output=baseline_output.flatten(2, 4),
                    )
                    if args.time_triton:
                        if args.time_triton_all_heads_chunked:
                            run_baseline()

                            def run_baseline_all_heads():
                                for head_start in range(0, query.size(1), args.triton_head_chunk):
                                    head_end = head_start + args.triton_head_chunk
                                    baseline_triton.hard_c4_720p_attention_triton_head_major(
                                        query[:, head_start:head_end],
                                        key[:, head_start:head_end],
                                        value[:, head_start:head_end],
                                        output=timed_hard_output[:, head_start:head_end],
                                    )

                            elapsed = measure_cuda_ms(run_baseline_all_heads)
                        else:
                            elapsed = measure_cuda_ms(run_baseline)
                        state["timings_ms"]["triton_baseline"].append(elapsed)
                    else:
                        run_baseline()
                    baseline_frame = (
                        baseline_output[:, :, frame].permute(0, 2, 3, 1, 4).unsqueeze(1)
                    )
                    stats["triton_baseline"].add(
                        baseline_frame, dense, stats_counts["baseline"]
                    )
                state["evaluated"].append(
                    {"call": state["call"], "step": step, "layer": self.layer, "query_frame": frame}
                )

            dense_full = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            state["call"] += 1
            if state["call"] == total_calls:
                raise StopSweep
            dense_full = dense_full.transpose(1, 2).flatten(2, 3).type_as(query)
            return attn.to_out[1](attn.to_out[0](dense_full))

    for layer, block in enumerate(pipe.transformer.blocks):
        block.attn1.set_processor(Processor(layer))
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--layers", default="0,13,26,39")
    parser.add_argument("--heads", default="0,20")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--check_triton_wide", action="store_true")
    parser.add_argument("--triton_baseline_module")
    parser.add_argument("--time_triton", action="store_true")
    parser.add_argument("--time_dense_all_heads", action="store_true")
    parser.add_argument("--time_triton_all_heads_chunked", action="store_true")
    parser.add_argument("--triton_head_chunk", type=int, default=8)
    parser.add_argument("--timing_head_multiplier", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    vae = AutoencoderKLWan.from_pretrained(
        args.model, subfolder="vae", torch_dtype=torch.float32, local_files_only=True
    )
    pipe = WanPipeline.from_pretrained(
        args.model,
        vae=vae,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe.scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=5.0,
    )
    pipe.to("cuda")

    stats = defaultdict(ErrorStats)
    state = install_probe(pipe, args, stats)
    prompts = [
        line.strip()
        for line in Path(args.prompt_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    prompt = prompts[args.prompt_index]
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    try:
        with torch.inference_mode():
            pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                height=720,
                width=1280,
                num_frames=81,
                guidance_scale=5.0,
                num_inference_steps=args.steps,
                generator=generator,
            )
    except StopSweep:
        pass
    torch.cuda.synchronize()

    results = {name: stats[name].summary() for name in state["variants"]}
    baseline = results["baseline"]
    for result in results.values():
        result["mse_ratio_vs_baseline"] = result["mse_vs_dense"] / baseline["mse_vs_dense"]
        result["rel_l2_ratio_vs_baseline"] = (
            result["rel_l2_vs_dense"] / baseline["rel_l2_vs_dense"]
        )
    compressed_tokens = {
        name: VARIANTS.get(name, {}).get("num_seeds", SEEDS) for name in state["variants"]
    }
    payload = {
        "device": torch.cuda.get_device_name(),
        "model": str(Path(args.model).resolve()),
        "steps": args.steps,
        "prompt_index": args.prompt_index,
        "layers": state["layers"],
        "heads": state["heads"],
        "evaluated_calls": state["evaluated"],
        "variants": results,
        "compressed_tokens_per_block": compressed_tokens,
        "variant_configs": {
            name: (VARIANTS[name] if name in VARIANTS else {"implementation": name})
            for name in state["variants"]
        },
        "invariants": {
            "grid_local": True,
            "entropy_routing_bias": True,
            "full_second_stage_attention": True,
            "extra_seed_key_dot_products": 0,
            "global_query_probes": {
                name: VARIANTS.get(name, {}).get("global_value_probe_count", 0)
                for name in state["variants"]
            },
            "local_exact_key_rows_per_probe": {
                name: VARIANTS.get(name, {}).get(
                    "global_value_local_key_rows", 0
                )
                for name in state["variants"]
            },
            "top_exact_key_rows_per_probe": {
                name: VARIANTS.get(name, {}).get(
                    "global_value_top_key_rows", 0
                )
                for name in state["variants"]
            },
            "top_exact_key_blocks_per_probe": {
                name: VARIANTS.get(name, {}).get(
                    "global_value_top_key_blocks", 0
                )
                for name in state["variants"]
            },
        },
    }
    if args.check_triton_wide:
        payload["triton_wide_vs_pytorch_wide"] = stats[
            "triton_wide_vs_pytorch_wide"
        ].summary()
    if args.time_triton:
        timings = {}
        for name, values in state["timings_ms"].items():
            measured = values[1:] if len(values) > 1 else values
            multiplier = args.timing_head_multiplier if name.startswith("triton_") else 1
            timings[name] = {
                "calls": len(measured),
                "mean_attention_ms": multiplier * sum(measured) / len(measured),
                "raw_head_chunk_ms": sum(measured) / len(measured),
                "head_chunk_multiplier": multiplier,
                "first_call_excluded_as_warmup": len(values) > 1,
            }
        sdpa_ms = timings["sdpa"]["mean_attention_ms"]
        for result in timings.values():
            result["speedup_vs_sdpa"] = sdpa_ms / result["mean_attention_ms"]
        payload["timings"] = timings
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
