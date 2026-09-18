"""CPU-only contracts for isolated exact-capacity RF0 quality screens.

C4 always means four complete merged representatives.  C8 always means eight.
Retired singleton/SQ1 capacity arms are intentionally absent from this module.
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType


_HELLINGER = "hellinger_span_alpha0p2_c32_p16_a2_j4_t1_rf0_u16_cg45"
_MASS = "entropy_mass_span_c32_p16_g256_j4_t1_rf0_u16_cg45"
_MASS_C10 = "entropy_mass_span_c32_p16_g256_j4_t1_rf0_u16_cg45_sq1_qm_skv_nl_ml1"
C10_METHODS = ("fisher_geometric_c10_rf0",)
BF16_MODES = MappingProxyType({
    "hellinger_rf0": _HELLINGER,
    "mass_rf0": _MASS,
    # There is deliberately no production Jeffreys parser mode.
    "jeffreys_rf0": _MASS,
})
JEFFREYS_OVERRIDE = (
    "attention.conditional_jeffreys_seed_triton.conditional_jeffreys_seed_rows_triton"
)
STABLE_SQ1_METHODS = ()
SPATIAL_TRANSPOSE_METHOD = "fisher_anchor_stable_sq1_rf0_hw_swap"
SPATIAL_TRANSPOSE_MODE = "fisher_anchor_rf0_c32_g45_u16_hw_swap"
TEMPORAL_KEY_METHOD = "fisher_anchor_stable_sq1_rf0_key_t_query_w"
TEMPORAL_KEY_MODE = "fisher_anchor_rf0_c32_g45_u16_key_t_query_w"
TEMPORAL_HEIGHT_METHOD = "fisher_anchor_stable_sq1_rf0_key_t_query_h"
TEMPORAL_HEIGHT_MODE = "fisher_anchor_rf0_c32_g45_u16_key_t_query_h"
TEMPORAL_METHODS = ()
AXIS_METHODS = ()
# Keep the frozen three-method diagnostic controls unchanged.
VIDEO_MODES = MappingProxyType(dict(BF16_MODES) | {
    "fisher_geometric_rf0": _MASS,
    "fisher_geometric_c10_rf0": _MASS_C10,
    "fisher_full45_rf0": _MASS.replace("_c32_", "_c45_"),
    "fisher_anchor_full45_rf0": _MASS.replace("_c32_", "_c45_"),
    "fisher_anchor_rf0": _MASS, "fisher_antipodal_rf0": _MASS,
    "fisher_mass_rf0": _MASS, "fisher_joint4_rf0": _MASS,
    "zero_fisher_rf0": _MASS,
    "mean_fisher_rf0": _MASS, "mean_anchor_fisher_rf0": _MASS,
    "barycentric_medoid_rf0": _MASS,
    # This isolated selector actually launches J1; never label it as J4.
    "fisher_expected_rf0": _MASS.replace("_j4_", "_j1_"),
    # Incumbent selection uses actual J4; the isolated correction is J1.
    "anchor_gap_guard_rf0": _MASS,
    # Only RF0 Q is read; analytic temporal phase averaging is selection-only.
    "phase_averaged_fisher_rf0": _MASS,
    # Explicit nondefault merge changes; original selectors/parser substrates.
})
_COMMON_ARGS = {
    "enable": True, "f_tied": 1, "h_reduce": 1, "w_reduce": 1,
    "layout": "hard_route", "query_outer_chunk": 21, "head_chunk": 8,
    "num_merge_tokens": 4, "seed": 0, "hard_route_backend": "hard_c4",
}
_OPTIONAL_DEFAULTS = {
    "enabled_layers": None,
    "exact_intra_frame_attention": False,
    "exact_neighbor_window_columns": 0,
    "exact_query_rows": 0,
    "first_frame_full_attention": False,
    "full_attention_anchor_time_count": 0,
    "full_attention_anchor_row_count": 0,
    "full_attention_anchor_width_count": 0,
    "full_attention_anchor_scale": 1.0,
    "full_attention_anchor_layers": None,
    "dense_head_schedule": None,
    "singleton_layer_schedule": None,
    "q_init": None,
}


def method_contract(method: str) -> dict:
    """Return a fresh serializable declaration, not proof of actual execution."""
    if method not in VIDEO_MODES:
        raise ValueError(f"Unsupported quality-first RF0 method: {method!r}")
    if method == TEMPORAL_HEIGHT_METHOD:
        declaration = method_contract(TEMPORAL_KEY_METHOD)
        declaration.update(
            method=method, seed_mode=VIDEO_MODES[method],
            change_from_historical_control="temporal_key_groups_with_spatial_H_queries_and_physical_frame0_seeds",
            selector_override="attention.temporal_key_height_gca_triton.fisher_seed_rows",
            core_override="attention.temporal_key_height_gca_triton.TemporalKeyHeightGCA",
            merge_override="attention.temporal_key_height_gca_triton.merge_and_full_route",
            candidate_geometry_meaning="fixed local/global budget, not remaining-axis extent; all45 height anchors",
            layout={"key_axis": "T", "query_axis": "H", "remaining_axis": "W",
                    "key_group_size": 21, "query_group_size": 45, "outer_size": 80,
                    "key_groups_per_outer": 45, "query_groups_per_outer": 21,
                    "compressed_slots_per_group": 5, "full_route": True},
            head_chunk=2,
            merge_queries="current Q[tq,selected_h,wq]; indices selected from physical Q[0,:,wq]",
            singleton="K[tq,hk,wk]; full route over all wk,hk",
        )
        return declaration
    if method == TEMPORAL_KEY_METHOD:
        declaration = method_contract(SPATIAL_TRANSPOSE_METHOD)
        declaration.update(
            method=method, seed_mode=VIDEO_MODES[method],
            change_from_historical_control="temporal_key_groups_with_spatial_W_queries_and_physical_frame0_seeds",
            selector_override="attention.temporal_key_gca_triton.fisher_seed_columns",
            core_override="attention.temporal_key_gca_triton.TemporalKeyGCA",
            merge_override="attention.temporal_key_gca_triton.merge_and_full_route",
            probe_indices=[i * 21 // 16 for i in range(16)],
            layout={"key_axis": "T", "query_axis": "W", "remaining_axis": "H",
                    "key_group_size": 21, "query_group_size": 80, "outer_size": 45,
                    "key_groups_per_outer": 80, "query_groups_per_outer": 21,
                    "compressed_slots_per_group": 5, "full_route": True},
            physical_reference_frame=0, head_chunk=4,
            bulk_normalization="direct stable softmax after full21 ownership and temporal SQ1 removal",
            merge_queries="current Q[tq,hq,selected_w]; indices selected from physical Q[0,hq,:]",
            singleton="K[tq,hk,wk]; full route over all hk,wk",
        )
        return declaration
    if method == SPATIAL_TRANSPOSE_METHOD:
        return {"method": method, "seed_mode": VIDEO_MODES[method], "dtype": "bf16",
                "change_from_historical_control": "swap_spatial_query_key_axes_with_same_anchor_fisher_rf0_and_stable_c4_sq1",
                "router_frame": 0, "bulk_representatives": 4, "intersection_singletons_per_block": 1,
                "all_layer_operator": True, "full_compressed_route": True, "direct_bulk_normalization": True,
                "selector_symbol": None,
                "selector_override": "attention.spatial_transpose_gca_triton.fisher_seed_columns",
                "core_override": "attention.spatial_transpose_gca_triton.SpatialTransposeGCA",
                "merge_override": "attention.spatial_transpose_gca_triton.merge_and_full_route",
                "candidate_count": 32, "candidate_geometry_count": 45,
                "candidate_geometry_meaning": "fixed local/global budget, not axis extent; all80 anchors",
                "probe_indices": [i * 45 // 16 for i in range(16)],
                "layout": {"key_axis": "H", "query_axis": "W", "time_axis": "T",
                           "key_group_size": 45, "query_group_size": 80,
                           "key_groups_per_frame": 80, "query_groups_per_frame": 45,
                           "compressed_slots_per_group": 5, "full_route": True},
                "bulk_normalization": "direct stable softmax after full45 ownership and intersection SQ1 removal",
                "bulk_validity": "structural nonempty valid support; padded keys excluded",
                "merge_queries": "four original current-frame real columns; seed indices selected using RF0 only",
                "generated_kernel_location": None,
                "precision_scope": "BF16 QK, FP32 stable normalization/entropy, BF16 weights/stored K/V; no FP8",
                "training_free": True, "dense_fallback": False}
    return {
        "method": method,
        "seed_mode": VIDEO_MODES[method],
        "dtype": "bf16",
        "change_from_historical_control": (
            method + "_selector_on_bf16_substrate"
            if method not in BF16_MODES else "remove_fp8score_only"
        ),
        "router_frame": 0,
        "bulk_representatives": 10 if method in C10_METHODS else 4,
        "query_column_singletons_per_block": 1 if method in C10_METHODS else 0,
        "complete_representatives_per_block": 11 if method in C10_METHODS else 4,
        "exact_capacity_contract": "C4 is exactly four merged K/V; singleton/SQ1/C+1 routing is forbidden",
        "all_layer_operator": True,
        "full_compressed_route": True,
        "direct_bulk_normalization": False,
        "selector_symbol": (
            "hellinger_span_seed_rows_triton" if method in ("hellinger_rf0", "hellinger_stable_sq1_rf0")
            else "entropy_mass_seed_rows_triton"
        ),
        "selector_override": {
            "jeffreys_rf0": JEFFREYS_OVERRIDE,
            "fisher_geometric_rf0": "attention.affine_hellinger_seed_triton.fisher_geometric_seed_rows_triton",
            "fisher_geometric_c10_rf0": "attention.affine_hellinger_seed_triton.fisher_geometric_c10_seed_rows_triton",
            "fisher_full45_rf0": "attention.affine_hellinger_seed_triton.fisher_geometric_seed_rows_triton",
            "fisher_anchor_rf0": "attention.affine_hellinger_seed_triton.fisher_anchor_seed_rows_triton",
            "fisher_anchor_stable_sq1_rf0": "attention.affine_hellinger_seed_triton.fisher_anchor_seed_rows_triton",
            "fisher_anchor_full45_rf0": "attention.affine_hellinger_seed_triton.fisher_anchor_seed_rows_triton",
            "fisher_antipodal_rf0": "attention.affine_hellinger_seed_triton.fisher_antipodal_seed_rows_triton",
            "fisher_mass_rf0": "attention.affine_hellinger_seed_triton.fisher_mass_seed_rows_triton",
            "fisher_joint4_rf0": "attention.affine_hellinger_seed_triton.fisher_joint4_seed_rows_triton",
            "zero_fisher_rf0": "attention.affine_hellinger_seed_triton.fisher_zero_seed_rows_triton",
            "mean_fisher_rf0": "attention.affine_hellinger_seed_triton.fisher_mean_seed_rows_triton",
            "mean_anchor_fisher_rf0": "attention.affine_hellinger_seed_triton.fisher_mean_anchor_seed_rows_triton",
            "barycentric_medoid_rf0": "attention.affine_hellinger_seed_triton.fisher_barycentric_medoid_seed_rows_triton",
            "fisher_expected_rf0": "attention.expected_fisher_seed_triton.fisher_expected_seed_rows_triton",
            "anchor_gap_guard_rf0": "attention.anchor_gap_guard_seed_triton.anchor_gap_guard_seed_rows_triton",
            "phase_averaged_fisher_rf0": "attention.phase_averaged_seed.phase_averaged_fisher_seed_rows",
        }.get(method),
        "precision_scope": (
            "BF16 merge-score operands and stored K/V; FP32 softmax/bias and "
            "accumulation. Not FP64/exact entropy merging."
        ),
    } | ({"effective_num_merge_tokens": 10, "physical_merge_slots": 16, "head_chunk": 4,
          "merge_block_j": 4, "merge_loop_stages": 1, "merge_num_warps": 4,
          "merge_override": "analysis.fisher_c10_video_adapter.FisherC10MergeAdapter",
          "capacity_override": "ten real seeds, six trailing duplicates with empty ownership; physical SINGLETON_C=16",
          "outer_dispatch_num_merge_tokens": 4,
          "candidate_count": 32, "candidate_geometry_count": 45,
          "seed_rule": "same geometric Fisher, matched real anchor plus nine GS pivots",
          "timing_caveat": "C10 correctness implementation scans sixteen physical slots including six zero-mass padded cells"}
         if method == "fisher_geometric_c10_rf0" else {}) | ({"synthetic_seed": {
        "zero_fisher_rf0": "analytic_zero_slot0_plus_three_real_rf0_rows",
        "mean_fisher_rf0": "current_frame_row_mean_slot0_plus_three_real_rf0_rows",
        "mean_anchor_fisher_rf0": "current_frame_row_mean_slot0_plus_matched_row_and_two_real_rf0_rows",
    }[method]} if method in ("zero_fisher_rf0", "mean_fisher_rf0", "mean_anchor_fisher_rf0") else {}) | (
        {"selection_reference": "RF0-only analytic positive temporal RoPE phase mean; FP64 transform then BF16",
         "merge_queries": "four original current-frame real rows; no synthetic merge query"}
        if method == "phase_averaged_fisher_rf0" else {}) | (
        {"change_from_historical_control": "explicit_stable_sq1_merge_with_unchanged_rf0_real_seed_selector",
         "direct_bulk_normalization": True,
         "legacy_direct_bulk_normalization_parser_flag": False,
         "merge_override": "analysis.stable_sq1_video_adapter.StableSQ1MergeAdapter",
         "bulk_normalization": "direct softmax on owned full80 keys after singleton removal; no residual subtraction/division",
         "bulk_validity": "structural nonempty support replaces residual > 1e-6 cutoff",
         "merge_queries": "four original current-frame real rows; unchanged selector and full C4+SQ1 route",
         "generated_kernel_location": "run_dir/stable_kernel.py outside frozen source; primary artifact",
         "precision_scope": "BF16 QK operands, FP32 bulk normalization/entropy, BF16 weights and stored K/V; not exact arithmetic"}
        if method in STABLE_SQ1_METHODS else {})


def quality_first_token_merging_args(method: str) -> dict:
    """Produce standalone all-layer arguments without modifying a config."""
    return {**_COMMON_ARGS, "seed_mode": method_contract(method)["seed_mode"],
            **({"head_chunk": method_contract(method)["head_chunk"]}
               if method in (*TEMPORAL_METHODS, *C10_METHODS) else {})}


def validate_token_merging_args(
    args: Mapping, method: str, *, dtype: str,
    selector_override: str | None = None,
) -> dict:
    """Fail closed for the exact common substrate, allowing neutral defaults.

The override argument records the caller's intended Jeffreys substitution;
it does not install or observe it. This checks operator arguments only, not
checkpoint/prompt provenance, generation completion, or metric validity.
Unknown options are rejected to avoid silently permitting future hybrids.
"""
    declaration = method_contract(method)
    if dtype != "bf16":
        raise ValueError("The matched quality-first substrate requires dtype='bf16'")
    if not isinstance(args, Mapping):
        raise ValueError("token_merging_args must be a mapping")
    expected = quality_first_token_merging_args(method)
    unknown = set(args) - set(expected) - set(_OPTIONAL_DEFAULTS)
    if unknown:
        raise ValueError(f"Unvalidated token_merging_args options: {sorted(unknown)}")
    for key, value in expected.items():
        # Avoid accepting booleans as integer settings or truthy strings.
        if key not in args or type(args[key]) is not type(value) or args[key] != value:
            raise ValueError(f"token_merging_args.{key} must equal {value!r}")
    for key, value in _OPTIONAL_DEFAULTS.items():
        if key not in args:
            continue
        actual = args[key]
        if key == "enabled_layers" and actual is not None:
            if not isinstance(actual, (list, tuple)) or any(type(i) is not int for i in actual):
                raise ValueError("enabled_layers must be None or every layer 0..39")
            if len(actual) != 40 or set(actual) != set(range(40)):
                raise ValueError("Quality-first candidates must replace all 40 layers")
        elif value is None:
            if actual is not None:
                raise ValueError(f"token_merging_args.{key} must remain None")
        elif type(actual) is not type(value) or actual != value:
            raise ValueError(f"token_merging_args.{key} must equal {value!r}")
    if selector_override != declaration["selector_override"]:
        raise ValueError("Selector override does not match the declared method")
    return declaration
