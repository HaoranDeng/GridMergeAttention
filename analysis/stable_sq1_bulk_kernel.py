"""Generate an OFF-PATH stable SQ1 merge control, never install or launch it.

The source is pinned to the reviewed production-function AST. Full-support
scores and ownership are retained; only subsequent conditioning is replaced.
The four bulk slots and indexed raw singleton are unchanged. Structural
nonempty support REPLACES the old residual > 1e-6 gate: this is an explicit
semantic difference, not merely a floating-point reorder. It requires its
own counterfactual in the witness diagnosis. This module has no GPU imports,
file writes, monkeypatch, wrapper integration, or automatic kernel loading.
CPU tests are not evidence of native GPU parity, quality, or speed.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import textwrap

SOURCE_KERNEL = "_hard_c4_720p_temporal_shared_merge_kernel"
GENERATED_KERNEL = "_experimental_stable_sq1_bulk_merge_kernel"
SOURCE_AST_SCHEMA = "sq1-semantic-ast-v1"
# Reviewed function in production file SHA256
# 6a62c06dd4ea1d6dc45b34be2e01548a5b8a65493597db9a1f1a4ede87df3f8f.
SOURCE_AST_SHA256 = "12a60f619680511675ddd2471039f0829aadee1bdf75378feec7c9ab9c161e9a"

# Fail closed. In particular, this is neither the old directbulk mode nor a
# general R>1, materialized-singleton, synthetic-query, or FP8 implementation.
REQUIRED_OPTIONS = {
    "Q_OUTER": 21, "ROUTER_FRAMES": 1, "K_OUTER": 21,
    "D_HEAD": 128, "ROWS": 45, "COLS": 80, "C": 4, "OUT_C": 4,
    "BLOCK_J": 16, "MERGE_FRAME_GROUP": 1, "MERGE_COL_GROUP": 1,
    "USE_ENTROPY_BIAS": True, "ROUTE_BIAS_SCALE": 1.0,
    "LOGIT_VARIANCE_BIAS_SCALE": 0.0,
    "STORE_CLUSTER_MASS": False, "FUSE_CLUSTER_MASS": False,
    "CORRECT_CLUSTER_VECTORS": False, "GLOBAL_MASS_TOP_R": 1,
    "GLOBAL_MASS_EXTRA_BLOCK": 1, "MATERIALIZE_GLOBAL_MASS": False,
    "GLOBAL_MASS_BIAS_ONLY": False, "GLOBAL_QUERY_COLUMN_SELECTOR": True,
    "GLOBAL_QUERY_COLUMN_SECONDARY_FRAME_STRIDE": 1,
    "GLOBAL_QUERY_COLUMN_PRIMARY_ROW_MOD4_OMIT2": False,
    "GLOBAL_MASS_DEFER_CORRECTION": False,
    "MEAN_CENTERED_ROUTING": False, "GAUSSIAN_PARTITION_ROUTING": False,
    "REUSE_PARTITION_EXP": False, "DIRECT_BULK_NORMALIZATION": False,
    "FP8_SCORE_DOT": False, "FP8_MERGE_K_DOT": False,
    "FP8_MERGE_V_DOT": False, "SCALED_FP8_KV": False,
    "SPLIT_KV_ACCUM": True, "STORE_PACKED_SINGLETON_KV": False,
    "SEED_ROWS_NATIVE_LAYOUT": True, "USE_TEMPORAL_ROW_MAP": False,
    "MEAN_SEED_ZERO": False,
}

# Scores retain the original log2 units. Center only on each owned bulk,
# AFTER the unchanged full-80 ownership operation and singleton exclusion.
# E[t] <= 0 and log2(Z) >= 0, so their difference does not cancel.
STABLE_BULK_SOURCE = """
cluster_ids = tl.arange(0, C)[None, :, None]
sq_columns = query_col_start + tl.arange(0, BLOCK_J)
bulk_main = (route_main[:, None, :] == cluster_ids) & (tl.arange(0, 64)[None, None, :] != sq_columns[:, None, None])
bulk_tail = (route_tail[:, None, :] == cluster_ids) & ((64 + tl.arange(0, 16))[None, None, :] != sq_columns[:, None, None])
bulk_count = tl.sum(tl.where(bulk_main, 1, 0), axis=2) + tl.sum(tl.where(bulk_tail, 1, 0), axis=2)
has_key = bulk_count > 0
cell_scores_main = tl.reshape(scores_main, (BLOCK_J, C, 64))
cell_scores_tail = tl.reshape(scores_tail, (BLOCK_J, C, 16))
bulk_max = tl.maximum(tl.max(tl.where(bulk_main, cell_scores_main, -float("inf")), axis=2), tl.max(tl.where(bulk_tail, cell_scores_tail, -float("inf")), axis=2))
safe_bulk_max = tl.where(has_key, bulk_max, 0.0)
centered_main = tl.where(bulk_main, cell_scores_main - safe_bulk_max[:, :, None], 0.0)
centered_tail = tl.where(bulk_tail, cell_scores_tail - safe_bulk_max[:, :, None], 0.0)
bulk_exp_main = tl.where(bulk_main, tl.exp2(centered_main), 0.0)
bulk_exp_tail = tl.where(bulk_tail, tl.exp2(centered_tail), 0.0)
bulk_sum = tl.sum(bulk_exp_main, axis=2) + tl.sum(bulk_exp_tail, axis=2)
safe_bulk_sum = tl.where(has_key, bulk_sum, 1.0)
weights_main = bulk_exp_main / safe_bulk_sum[:, :, None]
weights_tail = bulk_exp_tail / safe_bulk_sum[:, :, None]
centered_expectation = tl.sum(weights_main * centered_main, axis=2) + tl.sum(weights_tail * centered_tail, axis=2)
route_bias = (centered_expectation - tl.log2(safe_bulk_sum)) * 0.6931471805599453
route_bias = tl.where(has_key, route_bias, float("inf"))
"""


def _canonical_ast_value(value):
    """Version-independent semantic fields, not version-dependent ast.dump.

    CPython 3.12 added FunctionDef.type_params and 3.13 changed ast.dump's
    empty-list display default. Normalize only absent/empty type_params on
    the three definition nodes. Preserve EVERY other field, including empty
    lists and None (notably Constant(value=None)), all list order, scalar
    types, float bit values, nonempty type parameters, and unknown fields.
    Location attributes are not semantic fields and remain excluded.
    """
    if isinstance(value, ast.AST):
        fields = {}
        for name, item in ast.iter_fields(value):
            if (name == "type_params" and type(value).__name__ in
                    ("FunctionDef", "AsyncFunctionDef", "ClassDef")
                    and isinstance(item, list) and not item):
                continue
            fields[name] = _canonical_ast_value(item)
        return ["ast", type(value).__name__, fields]
    if isinstance(value, list):
        return ["list", [_canonical_ast_value(item) for item in value]]
    if value is None:
        return ["none"]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is int:
        return ["int", str(value)]
    if type(value) is float:
        return ["float", value.hex()]
    if type(value) is complex:
        return ["complex", value.real.hex(), value.imag.hex()]
    if type(value) is str:
        return ["str", value]
    if type(value) is bytes:
        return ["bytes", value.hex()]
    if value is Ellipsis:
        return ["ellipsis"]
    raise TypeError("unsupported semantic AST value: " + type(value).__name__)


def canonical_ast_fingerprint(node):
    """Fail-closed AST pin stable across the reviewed Python 3.10/3.13 parsers."""
    payload = {"schema": SOURCE_AST_SCHEMA, "tree": _canonical_ast_value(node)}
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_launch_contract(options):
    """Optional CPU-side validation; generated static_asserts also enforce it."""
    wrong = [key for key, value in REQUIRED_OPTIONS.items()
             if key not in options or options[key] != value]
    if wrong:
        raise ValueError("unsupported stable SQ1 options: " + ", ".join(wrong))


def _fold_condition(node):
    """Evaluate only scalar Boolean/compare arithmetic in known constexprs."""
    tree = copy.deepcopy(node)
    for item in ast.walk(tree):
        if isinstance(item, ast.Name) and item.id not in REQUIRED_OPTIONS:
            return None
    allowed = (ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.Compare,
               ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
               ast.Name, ast.Load, ast.Constant)
    if any(not isinstance(item, allowed) for item in ast.walk(tree)):
        return None
    return bool(eval(compile(ast.Expression(tree), "<constexpr>", "eval"),
                     {"__builtins__": {}}, REQUIRED_OPTIONS))


class _SelectReviewedBranches(ast.NodeTransformer):
    def visit_If(self, node):
        result = _fold_condition(node.test)
        if result is None:
            return self.generic_visit(node)
        body = []
        for statement in node.body if result else node.orelse:
            value = self.visit(statement)
            body.extend(value if isinstance(value, list) else [value])
        return body


def generate_stable_sq1_source(source):
    """Return standalone experimental Triton source; no import or execution.

    Accept the full production module text or inspect.getsource(function).
    Any arithmetic/signature drift fails the pinned AST check. A later native
    experiment must archive this returned source, validate arguments, compile
    it separately, and retain the unchanged route and seed table explicitly.
    """
    tree = ast.parse(textwrap.dedent(source))
    matches = [item for item in ast.walk(tree)
               if isinstance(item, ast.FunctionDef) and item.name == SOURCE_KERNEL]
    if len(matches) != 1:
        raise ValueError("expected exactly one production merge function")
    function = matches[0]
    fingerprint = canonical_ast_fingerprint(function)
    if fingerprint != SOURCE_AST_SHA256:
        raise ValueError("production merge AST changed; review before generating")

    # Replace one contiguous conditioning region, leaving all preceding QK,
    # full-support log-normalization, and winner/tie-break AST nodes intact.
    replacements = 0
    for node in list(ast.walk(function)):
        for attribute in ("body", "orelse"):
            body = getattr(node, attribute, None)
            if not isinstance(body, list):
                continue
            starts = [i for i, item in enumerate(body) if isinstance(item, ast.Assign)
                      and ast.unparse(item.targets[0]) == "cluster_ids"]
            if not starts:
                continue
            if len(starts) != 1:
                raise ValueError("ambiguous conditioning start")
            start = starts[0]
            ends = [i for i in range(start + 1, len(body)) if isinstance(body[i], ast.If)
                    and ast.unparse(body[i].test) == "STORE_CLUSTER_MASS or FUSE_CLUSTER_MASS"]
            if len(ends) != 1:
                raise ValueError("ambiguous conditioning end")
            body[start:ends[0]] = ast.parse(STABLE_BULK_SOURCE).body
            replacements += 1
    if replacements != 1:
        raise ValueError("conditioning anchor missing")
    function = _SelectReviewedBranches().visit(function)

    # Normalized bulk weights already feed the existing BF16 K/V dot. Remove
    # the old final /residual_scale and its setup; never divide by 1-alpha.
    removed = {"residual_scale": 0, "merged_key": 0, "merged_value": 0}

    class RemoveResidualDivision(ast.NodeTransformer):
        def visit_Assign(self, node):
            if ast.unparse(node.targets[0]) == "residual_scale":
                removed["residual_scale"] += 1
                return None
            return node

        def visit_AugAssign(self, node):
            name = ast.unparse(node.target)
            if name in ("merged_key", "merged_value") and isinstance(node.op, ast.Div):
                if ast.unparse(node.value) != "residual_scale[:, None]":
                    raise ValueError("unreviewed final division")
                removed[name] += 1
                return None
            return node

    function = RemoveResidualDivision().visit(function)
    if removed != {"residual_scale": 1, "merged_key": 1, "merged_value": 1}:
        raise ValueError("residual-division anchors changed")
    function.name = GENERATED_KERNEL
    function.decorator_list = [ast.parse("triton.jit", mode="eval").body]
    guards = ast.parse("\n".join(f"tl.static_assert({key} == {value!r})"
                                 for key, value in REQUIRED_OPTIONS.items())).body
    guards += ast.parse("\n".join(
        f"tl.static_assert({pointer}.dtype.element_ty == tl.bfloat16)"
        for pointer in ("q_ptr", "k_ptr", "v_ptr", "merged_k_ptr", "merged_v_ptr")
    )).body
    function.body = [function.body[0], *guards, *function.body[1:]]
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    result = "import triton\nimport triton.language as tl\n\n" + ast.unparse(module) + "\n"
    if any(name in {item.id for item in ast.walk(module) if isinstance(item, ast.Name)}
           for name in ("selected_cluster_weight", "safe_residual_mass", "residual_mass", "selected_probability", "bulk_expected")):
        raise ValueError("old residual arithmetic survived specialization")
    return result
