"""Isolated, instrumented 20-video BF16-score RF0 generation entry point.

The parent must own the clean single-H200 preflight and continuous purity
monitor. This wrapper never selects/queues GPUs. It freezes a small source
whitelist and a new run-local config, then executes frozen generate.py with
runpy. Production configs/defaults remain untouched. Metadata/counter checks
inside attention add some launch overhead; reported timing is instrumented.

Use --prepare-only before launch, then --execute-prepared with identical
method/checkpoint/run-dir and --parent-purity-managed. No resume is supported.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis.quality_first_mode_contract import (
    VIDEO_MODES, C10_METHODS, STABLE_SQ1_METHODS, SPATIAL_TRANSPOSE_METHOD, TEMPORAL_KEY_METHOD,
    TEMPORAL_HEIGHT_METHOD, TEMPORAL_METHODS, AXIS_METHODS,
    method_contract, quality_first_token_merging_args,
    validate_token_merging_args,
)

PROMPTS = "prompts/MovieGenVideoBench_extended.txt"
OWN_SOURCES = (
    "analysis/run_quality_first_videos.py", "analysis/quality_first_mode_contract.py",
    "analysis/fisher_c10_video_adapter.py",
)
MERGE_SYMBOL = "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks"
SHAPE = (1, 75600, 40, 128)
GRID = (1, 21, 45, 80, 40, 128)
SEED_SHAPE = (1, 1, 21, 45, 80, 40, 4)
MEAN_METHODS = ("mean_fisher_rf0", "mean_anchor_fisher_rf0")
SYNTHETIC_METHODS = ("zero_fisher_rf0", *MEAN_METHODS)
STABLE_SQ1_SOURCES = ("analysis/stable_sq1_bulk_kernel.py", "analysis/stable_sq1_video_adapter.py")
SPATIAL_SOURCES = ("attention/spatial_transpose_gca_triton.py", "attention/spatial_transpose_gca_reference.py",
                   "attention/hard_c4_wan21_14b_720p_triton.py", "attention/adaptive_seed_pytorch.py")
TEMPORAL_SOURCES = ("attention/temporal_key_gca_triton.py", "attention/temporal_key_gca_reference.py",
                    *SPATIAL_SOURCES[2:])
TEMPORAL_HEIGHT_SOURCES = ("attention/temporal_key_height_gca_triton.py", "attention/temporal_key_height_gca_reference.py",
                         *SPATIAL_SOURCES[2:])
MODEL_NAME = "Wan2.1-T2V-14B-StepDistill-CfgDistill"
CANONICAL_CHECKPOINT_BYTES = 28577333801
CAPTURE_ENVIRONMENT = (
    "TOKEN_MERGING_SAVE_FIRST_QKV_PATH", "TOKEN_MERGING_EXIT_AFTER_QKV_SAVE",
    "GMA_OUTPUT_CALIBRATION_CAPTURE_PATH",
)
ASSET_FILES = (
    "config.json", "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth",
    "google/umt5-xxl/tokenizer.json", "google/umt5-xxl/tokenizer_config.json",
    "google/umt5-xxl/spiece.model", "google/umt5-xxl/special_tokens_map.json",
)
HYBRID_FIELDS = (
    "exact_intra_frame_attention", "exact_neighbor_window_columns", "exact_query_rows",
    "first_frame_full_attention", "full_attention_anchor_time_count",
    "full_attention_anchor_row_count", "full_attention_anchor_width_count",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)


def checkpoint_identity(path):
    path = Path(path).resolve(strict=True)
    require(path.is_file(), "checkpoint must be an existing regular file")
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "identity_scope": "path/size/mtime; checkpoint contents are not SHA256-verified"}


def check_environment(environment=None):
    environment = os.environ if environment is None else environment
    for name in CAPTURE_ENVIRONMENT:
        require(not environment.get(name), f"capture/early-exit environment is forbidden: {name}")


def asset_preflight(checkpoint, environment=None):
    """Read-only asset check; hashes small config/tokenizer files, not weights."""
    environment = os.environ if environment is None else environment
    check_environment(environment)
    raw_root = environment.get("WAN_MODEL_ROOT", "")
    require(bool(raw_root) and Path(raw_root).is_absolute(), "WAN_MODEL_ROOT must be an explicit absolute asset root")
    model_root = Path(raw_root).resolve(strict=True)
    model_dir = model_root / MODEL_NAME
    require(Path(checkpoint).is_absolute(), "checkpoint must be absolute")
    identity = checkpoint_identity(checkpoint)
    require(Path(identity["path"]) == (model_dir / "distill_model.pt").resolve(strict=True),
            "use the matched canonical distill_model.pt, not Self-Forcing/another checkpoint")
    require(identity["bytes"] == CANONICAL_CHECKPOINT_BYTES, "canonical checkpoint size differs")
    records = []
    for name in ASSET_FILES:
        path = model_dir / name
        require(path.is_file() and path.stat().st_size > 0, f"required external asset is missing/empty: {path}")
        stat = path.stat()
        record = {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if path.suffix != ".pth":
            require(stat.st_size <= 32 * 1024 * 1024, f"unexpected large config/tokenizer asset: {path}")
            record["sha256"] = digest(path)
        else:
            record["identity_scope"] = "path/size/mtime only; weights are not SHA256-verified"
        records.append(record)
    return {"WAN_MODEL_ROOT": str(model_root), "model_dir": str(model_dir), "files": records,
            "checkpoint": identity}


def config_document(method):
    return {
        "image_or_video_shape": [1, 21, 16, 90, 160],
        "denoising_step_list": [1000, 750, 500, 250], "warp_denoising_step": True,
        "model_kwargs": {"model_name": "Wan2.1-T2V-14B-StepDistill-CfgDistill",
                         "load_pretrained_weights": False, "timestep_shift": 5.0},
        "token_merging_args": quality_first_token_merging_args(method),
    }


def source_names(root):
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    extensions = {".py", ".yaml", ".yml", ".json", ".jinja", ".jinja2"}
    names = {name for name in raw.decode("utf-8").split("\0") if name
             and Path(name).suffix.lower() in extensions
             and not name.startswith(("experiments/", "outputs/", "videos/"))}
    names.update((*OWN_SOURCES, PROMPTS))
    return sorted(names)


def prepare(run_dir, method, checkpoint, *, root=ROOT, num_videos=20, prompt_start=0):
    """Create a fresh CPU-only snapshot; no model import or GPU operation."""
    run_dir, root = Path(run_dir).resolve(), Path(root).resolve()
    require(not run_dir.exists(), "run-dir already exists; use a fresh stage or --execute-prepared")
    require(type(num_videos) is int and 1 <= num_videos <= 20, "num_videos must be 1..20")
    require(type(prompt_start) is int and 0 <= prompt_start < num_videos, "invalid prompt_start")
    assets = asset_preflight(checkpoint)
    identity = assets["checkpoint"]
    names = source_names(root)
    if method in STABLE_SQ1_METHODS:
        names = sorted(set(names) | set(STABLE_SQ1_SOURCES))
    if method in AXIS_METHODS:
        names = sorted(set(names) | set(axis_sources(method)))
    # Bound source copies and reject links escaping the source tree.
    for name in names:
        path = (root / name).resolve(strict=True)
        require(path.is_relative_to(root) and path.is_file(), f"unsafe source path: {name}")
        require(path.stat().st_size <= 32 * 1024 * 1024, f"unexpected large source resource: {name}")
    run_dir.mkdir(parents=True)
    frozen = run_dir / "source"
    records = []
    for name in names:
        src, dst = root / name, frozen / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        before = digest(src)
        shutil.copyfile(src, dst)
        require(before == digest(src) == digest(dst), f"source changed while freezing: {name}")
        records.append({"path": name, "sha256": before, "bytes": dst.stat().st_size})
    config_name = f"configs/quality_first_{method}_bf16_4step.json"
    (frozen / "configs").mkdir(parents=True, exist_ok=True)
    config = config_document(method)
    validate_token_merging_args(config["token_merging_args"], method, dtype="bf16",
                               selector_override=method_contract(method)["selector_override"])
    write_json(frozen / config_name, config, exclusive=True)  # JSON is valid YAML/OmegaConf.
    records.append({"path": config_name, "sha256": digest(frozen / config_name),
                    "bytes": (frozen / config_name).stat().st_size})
    provenance = {
        "method": method + "_bf16score", "declaration": method_contract(method),
        "remote_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "config": config_name, "files": records, "checkpoint": identity,
        "planned": {"router_frame": 0, "num_videos": num_videos, "prompt_start": prompt_start, "num_samples": 1, "seed": 0,
                    "frames": 81, "denoise_steps": 4, "attention_calls_per_video": 160},
        "snapshot_scope": "tracked Python/YAML/JSON/Jinja source; canonical prompt; explicit wrapper/helper",
        "external_assets": assets,
    }
    write_json(run_dir / "source_provenance.json", provenance, exclusive=True)
    subprocess.run([sys.executable, str(frozen / "analysis/verify_prompt_contract.py"),
                    "--prompt_path", str(frozen / PROMPTS), "--num_videos", str(num_videos), "--seed", "0",
                    "--num_samples", "1", "--manifest_out", str(run_dir / "prompt_preflight_manifest.json")],
                   cwd=frozen, check=True, stdout=subprocess.DEVNULL)
    return provenance


def verify_snapshot(run_dir, method, checkpoint):
    run_dir = Path(run_dir).resolve()
    provenance = json.loads((run_dir / "source_provenance.json").read_text(encoding="utf-8"))
    require(type(provenance["planned"]["num_videos"]) is int and 1 <= provenance["planned"]["num_videos"] <= 20, "invalid prepared sample count")
    planned_prompt_indices(provenance)
    require(provenance["declaration"] == method_contract(method), "prepared method differs")
    require(provenance["checkpoint"] == checkpoint_identity(checkpoint), "checkpoint metadata changed")
    require(provenance["external_assets"] == asset_preflight(checkpoint), "external model/tokenizer assets changed")
    frozen = run_dir / "source"
    seen = set()
    for record in provenance["files"]:
        path = (frozen / record["path"]).resolve(strict=True)
        require(path.is_relative_to(frozen) and record["path"] not in seen, "unsafe/duplicate frozen source")
        seen.add(record["path"])
        require(digest(path) == record["sha256"], f"frozen source changed: {record['path']}")
    require(set(OWN_SOURCES) | {PROMPTS, "generate.py", provenance["config"]} <= seen,
            "missing required frozen entry points")
    if method in STABLE_SQ1_METHODS:
        require(set(STABLE_SQ1_SOURCES) <= seen, "missing frozen stable SQ1 generator/adapter")
    if method in AXIS_METHODS:
        require(set(axis_sources(method)) <= seen, "missing frozen axis source")
    config = json.loads((frozen / provenance["config"]).read_text(encoding="utf-8"))
    require(config == config_document(method), "frozen configuration differs")
    return provenance


def tensor_metadata(tensor):
    """Host-side metadata only: no reductions, copies, item(), or CUDA sync."""
    return {"shape": tuple(tensor.shape), "stride": tuple(tensor.stride()),
            "storage_offset": tensor.storage_offset(), "data_ptr": tensor.data_ptr(),
            "dtype": str(tensor.dtype), "device": str(tensor.device)}


def grid_metadata(metadata, *, rf0=False):
    require(metadata["shape"] == SHAPE, "unexpected full-40-head QKV shape")
    batch, token, head, dim = metadata["stride"]
    require((token, head, dim) == (5120, 128, 1), "QKV token/head layout is not a metadata-only grid view")
    return metadata | {"shape": (1, 1 if rf0 else 21, 45, 80, 40, 128),
                       "stride": (batch, 3600 * token, 80 * token, token, head, dim)}


class DispatchAudit:
    """Fail-closed nested call accounting for one sequential 20-video worker."""
    def __init__(self, method, path=None, *, num_videos=20):
        require(type(num_videos) is int and 1 <= num_videos <= 20, "invalid sample count")
        self.num_videos = num_videos
        self.method, self.path = method, path
        self.declaration = method_contract(method)
        self.layers = [0] * 40
        self.calls = self.core_calls = self.selector_calls = self.merge_calls = self.videos = 0
        self.synthetic_seed_kernel_calls = 0
        self.mean_seed_constructions = 0
        self.stable_merge_kernel_calls = self.stable_merge_wrapper_calls = 0
        self.generated_stable_kernel = None
        self.spatial_core = None
        self.spatial_metadata = None
        self.spatial_route_calls = 0
        self.active_layer = None
        self.qkv = None
        self.seed_metadata = None
        self.identities = {}
        self.first_call_metadata = {}

    def enter_layer(self, module, grid_size):
        require(self.active_layer is None, "nested/concurrent self-attention is unsupported")
        layer = module.layer_idx
        require(type(layer) is int and layer == self.calls % 40, "missing/reordered transformer layer")
        require(tuple(grid_size) == (21, 45, 80), "wrong generation grid")
        expected = quality_first_token_merging_args(self.method)
        require(module.enable_token_merging is True, "dense self-attention layer detected")
        for key, value in expected.items():
            if key != "enable":
                require(getattr(module, "token_merging_" + key) == value, f"actual layer setting changed: {key}")
        for key in HYBRID_FIELDS:
            require(not getattr(module, "token_merging_" + key), f"hybrid layer setting: {key}")
        require(module.token_merging_dense_heads == (), "dense heads detected")
        require(module.token_merging_full_attention_anchor_scale == 1.0, "anchor scale changed")
        require(module.token_merging_q_init is None, "q_init override")
        for key in ("seed_query_proj", "gma_output_calibration_weights", "gma_query_calibration_parameters"):
            require(getattr(module, key) is None, f"trained/calibrated component: {key}")
        self.active_layer = layer
        self.qkv = self.seed_metadata = None

    def enter_core(self, args, kwargs):
        require(self.active_layer is not None and self.qkv is None, "missing/duplicate attention dispatch")
        require(tuple(args[3:8]) == (1, 1, 1, 45, 80), "grouping/layout changed")
        require(kwargs.get("seed_mode") == VIDEO_MODES[self.method], "wrong actual seed mode")
        require(kwargs.get("seed_query") is None and not kwargs.get("dense_heads"), "extra seed/dense path")
        self.qkv = [tensor_metadata(value) for value in args[:3]]
        for metadata in self.qkv:
            require(metadata["dtype"] == "torch.bfloat16" and metadata["device"].startswith("cuda"), "not CUDA BF16 QKV")
            grid_metadata(metadata)
        if self.core_calls == 0:
            self.first_call_metadata["core_qkv"] = self.qkv
        self.core_calls += 1

    def enter_selector(self, args, kwargs):
        require(self.qkv is not None and self.seed_metadata is None, "selector outside core or repeated")
        require(len(args) == 3 and args[2] == 128 ** -0.5, "selector scale/arity changed")
        require(tensor_metadata(args[0]) == grid_metadata(self.qkv[0], rf0=True), "selector is not actual RF0 Q view")
        require(tensor_metadata(args[1]) == grid_metadata(self.qkv[1]), "selector K storage changed")
        if self.selector_calls == 0:
            self.first_call_metadata["selector_qk"] = [tensor_metadata(value) for value in args[:2]]
        expected = {"candidate_count": 45 if self.method in ("fisher_full45_rf0", "fisher_anchor_full45_rf0") else 32,
                    "candidate_geometry_count": 45,
                    "block_j": 1 if self.method == "fisher_expected_rf0" else 4,
                    "gs_num_warps": 4}
        if self.declaration["selector_symbol"] == "hellinger_span_seed_rows_triton":
            expected.update(alpha=0.2, top_probes=0, uniform_probes=16, probe_anchor_count=2,
                            group_tile=16, selection_mode="gs", force_local_minus_one=False)
        else:
            expected.update(probe_count=16, global_probe_count=256)
        require(set(kwargs) == set(expected) | {"output_dtype"}, "unexpected selector options")
        require(all(kwargs[key] == value for key, value in expected.items()), "selector criterion/probes changed")
        require(str(kwargs["output_dtype"]) == "torch.uint8", "selector output dtype changed")
        self.selector_calls += 1

    def leave_selector(self, result):
        self.seed_metadata = tensor_metadata(result)
        expected_shape = SEED_SHAPE[:-1] + (self.declaration["bulk_representatives"],)
        require(self.seed_metadata["shape"] == expected_shape and self.seed_metadata["dtype"] == "torch.uint8",
                "native seed-table layout changed")
        if self.selector_calls == 1:
            self.first_call_metadata["seed_table"] = self.seed_metadata

    def enter_merge(self, args, kwargs):
        require(self.seed_metadata is not None and len(args) == 5, "merge without actual selector result")
        require(args[4] == 128 ** -0.5, "merge scale changed")
        for index in range(3):
            require(tensor_metadata(args[index]) == grid_metadata(self.qkv[index]), "merge QKV storage changed")
        require(tensor_metadata(args[3]) == self.seed_metadata, "merge did not consume observed seed table")
        expected = {
            "head_start": (self.merge_calls % axis_chunks(self.method)) * (40 // axis_chunks(self.method)),
            "head_count": 40 // axis_chunks(self.method), "top_r": 1,
            "selector": "query_column", "compressed_fp8_kv": False, "scaled_fp8_kv": False,
            "mean_centered_routing": False, "gaussian_partition_routing": False,
            "reuse_partition_exp": False, "direct_bulk_normalization": False,
            "fp8_score_dot": False, "fp8_merge_k_dot": False, "fp8_merge_v_dot": False,
            "query_frame_group": 21, "split_merge": False, "materialize_singleton": False,
            "query_column_major": True, "query_column_secondary_frame_stride": 1,
            "query_column_primary_row_mod4_omit2": False, "packed_singleton_kv": False,
            "seed_rows_native_layout": True, "merge_frame_group": 1, "merge_col_group": 1,
            "merge_loop_stages": 1 if self.method in C10_METHODS else 2,
            "merge_disable_licm": False, "split_kv_accum": True,
        }
        require(set(kwargs) == set(expected) | {"output"}, "unexpected merge/route override")
        require(all(kwargs[key] == value for key, value in expected.items()), "actual merge/route substrate changed")
        self.merge_calls += 1

    def leave_layer(self):
        next_call = self.calls + 1
        require((self.core_calls, self.selector_calls, self.merge_calls) == (next_call, next_call, axis_chunks(self.method) * next_call),
                "missing/extra core, selector, or head-chunk merge/route call")
        require(self.synthetic_seed_kernel_calls == (self.merge_calls if self.method in SYNTHETIC_METHODS else 0),
                "missing/unexpected analytic seed merge kernel")
        require(self.mean_seed_constructions == (next_call if self.method in MEAN_METHODS else 0),
                "require one current-frame mean construction per attention, no stale reuse")
        stable_expected = self.merge_calls if self.method in STABLE_SQ1_METHODS or self.method in AXIS_METHODS else 0
        require(self.stable_merge_kernel_calls == self.stable_merge_wrapper_calls == stable_expected,
                "missing/unexpected stable SQ1 kernel or merge wrapper call")
        if self.method in AXIS_METHODS:
            require(self.spatial_route_calls == self.merge_calls, "missing spatial full-route kernel")
        self.layers[self.active_layer] += 1
        self.calls = next_call
        self.active_layer = None
        self.qkv = self.seed_metadata = None

    def finish_video(self, timing):
        require(self.active_layer is None, "video ended inside attention")
        require(timing["attention_calls"] == timing["full_attention_calls"] == 160, "runtime call-count mismatch")
        self.videos += 1
        require(self.videos <= self.num_videos and self.calls == self.videos * 160 and self.layers == [self.videos * 4] * 40,
                "video did not execute all 40 layers at all four steps")
        self.save(False)

    def save(self, complete, error=None):
        if complete:
            require(self.videos == self.num_videos and self.calls == self.num_videos * 160 and self.layers == [self.num_videos * 4] * 40, "generation incomplete")
            require(self.active_layer is None and self.core_calls == self.selector_calls == self.calls
                    and self.merge_calls == self.calls * axis_chunks(self.method),
                    "incomplete full-layer selector/merge coverage")
            if self.method in STABLE_SQ1_METHODS:
                require(self.active_layer is None and self.core_calls == self.selector_calls == self.num_videos * 160
                        and self.merge_calls == self.stable_merge_kernel_calls == self.stable_merge_wrapper_calls == self.num_videos * 800,
                        "stable SQ1 requires all20/all40/all4/five-head-chunk coverage")
                verify_generated_stable_kernel(self)
            if self.method in AXIS_METHODS:
                require(self.active_layer is None and self.core_calls == self.selector_calls == self.num_videos * 160
                        and self.merge_calls == self.stable_merge_kernel_calls == self.stable_merge_wrapper_calls
                        == self.spatial_route_calls == self.num_videos * 160 * axis_chunks(self.method), "axis full20/all40/fourstep coverage mismatch")
                require(self.spatial_core is not None, "missing actual spatial core")
                self.spatial_core.verify_kernel()
        payload = {"complete": complete, "error": error, "declaration": self.declaration,
                   "videos_observed": self.videos, "self_attention_calls": self.calls,
                   "core_calls": self.core_calls, "selector_calls": self.selector_calls,
                   "merge_route_head_chunk_calls": self.merge_calls, "per_layer_calls": self.layers,
                   "synthetic_seed_kernel_calls": self.synthetic_seed_kernel_calls,
                   "mean_seed_constructions": self.mean_seed_constructions,
                   "callable_identities": self.identities,
                   "first_call_metadata": self.first_call_metadata,
                   "instrumentation": "host metadata/counters only; no added tensor reductions, CPU copies, or CUDA sync",
                   "timing_caveat": "core timing includes metadata/check launch overhead; parent GPU-purity verification required"}
        if self.method in STABLE_SQ1_METHODS:
            payload.update(stable_merge_kernel_calls=self.stable_merge_kernel_calls,
                           stable_merge_wrapper_calls=self.stable_merge_wrapper_calls,
                           generated_stable_kernel=self.generated_stable_kernel)
        if self.method in AXIS_METHODS:
            payload.update(stable_merge_kernel_calls=self.stable_merge_kernel_calls,
                           stable_merge_wrapper_calls=self.stable_merge_wrapper_calls,
                           spatial_route_kernel_calls=self.spatial_route_calls,
                           **{axis_payload_key(self.method): self.spatial_metadata})
        if self.path is not None:
            write_json(self.path, payload)
        return payload


def callable_identity(function, frozen):
    path = Path(inspect.getsourcefile(function)).resolve()
    require(path.is_relative_to(frozen), f"callable imported outside frozen source: {path}")
    return {"callable": function.__module__ + "." + function.__name__,
            "source": str(path.relative_to(frozen)), "sha256": digest(path)}


def verify_generated_stable_kernel(audit):
    receipt = audit.generated_stable_kernel
    require(isinstance(receipt, dict), "missing generated stable source receipt")
    path = Path(receipt["path"]).resolve(strict=True)
    require(path.name == "stable_kernel.py" and path.is_file(), "wrong stable source artifact")
    if audit.path is not None:
        require(path.parent == Path(audit.path).resolve().parent, "stable source must be a run-local primary artifact")
    require(path.stat().st_size == receipt["bytes"] and digest(path) == receipt["sha256"], "generated stable source changed")
    return receipt


def axis_chunks(method):
    if method in C10_METHODS:
        return 10
    return 20 if method == TEMPORAL_HEIGHT_METHOD else (10 if method == TEMPORAL_KEY_METHOD else 5)


def axis_sources(method):
    require(method in AXIS_METHODS, "unknown axis method")
    return {SPATIAL_TRANSPOSE_METHOD: SPATIAL_SOURCES, TEMPORAL_KEY_METHOD: TEMPORAL_SOURCES,
            TEMPORAL_HEIGHT_METHOD: TEMPORAL_HEIGHT_SOURCES}[method]


def axis_payload_key(method):
    return "temporal_operator" if method in TEMPORAL_METHODS else "spatial_operator"


def axis_module_name(method):
    return {SPATIAL_TRANSPOSE_METHOD: "attention.spatial_transpose_gca_triton",
            TEMPORAL_KEY_METHOD: "attention.temporal_key_gca_triton",
            TEMPORAL_HEIGHT_METHOD: "attention.temporal_key_height_gca_triton"}[method]


def verify_spatial_sources(receipt, frozen, method=SPATIAL_TRANSPOSE_METHOD):
    """Verify all four native-proof files against this immutable snapshot."""
    sources = axis_sources(method)
    require(isinstance(receipt, dict) and len(receipt) == len(sources), "missing axis source receipt")
    frozen = Path(frozen).resolve()
    verified = {}
    for path, sha256 in receipt.items():
        # Old validated absolute worktree paths may differ; exact attention/
        # filename and bytes, not the old checkout location, identify sources.
        relative = Path(*Path(path).parts[-2:]).as_posix()
        require(relative in sources and relative not in verified, "unexpected axis proof source")
        source = (frozen / relative).resolve(strict=True)
        require(source.is_relative_to(frozen) and digest(source) == sha256, "spatial native-proof source changed: " + relative)
        verified[relative] = sha256
    require(set(verified) == set(sources), "incomplete axis native-proof source set")
    return verified


def validate_spatial_proof(proof, frozen):
    require(isinstance(proof, dict) and proof.get("valid") is True and proof.get("passed") is True,
            "spatial native validation did not pass")
    verified = verify_spatial_sources(proof.get("source_receipt"), frozen)
    cases = proof.get("cases", {})
    require(set(cases) == {"random_two_frames", "zero_query_ties_empty_bulks"}, "spatial native cases incomplete")
    for name, case in cases.items():
        require(case.get("seed_cells") == 24 and case.get("seed_mismatches") == 0
                and case.get("merge_cells") == 48 and case.get("merge_max_abs", float("inf")) <= .01
                and case.get("bias_max_abs", float("inf")) <= 2e-5
                and case.get("route_query_vectors") == 640
                and case.get("route_max_abs", float("inf")) <= .02
                and case.get("route_rmse", float("inf")) <= .0025
                and case.get("all_outputs_finite") is True
                and case.get("tail_query_columns_checked") == list(range(64, 80))
                and case.get("query_rows_checked") == [0, 16, 32, 44], "spatial native case failed: " + name)
    require(cases["zero_query_ties_empty_bulks"].get("empty_bulk_count") == 2 * 45 * 2 * 80 * 3,
            "spatial native empty bulk witness missing")
    full = proof.get("full_shape_smoke", {})
    require(full.get("qkv_shape") == [1, 40, 75600, 128] and full.get("dtype") == "bfloat16"
            and full.get("finite_output") is True and full.get("route_kernel_calls") == 5
            and full.get("counts") == dict(core_calls=1, selector_calls=1, merge_wrapper_calls=5, stable_kernel_calls=5),
            "spatial native full-shape witness missing")
    return verified


def validate_axis_proof(proof, frozen, method):
    if method == SPATIAL_TRANSPOSE_METHOD:
        return validate_spatial_proof(proof, frozen)
    require(method in TEMPORAL_METHODS, "unknown native axis proof")
    require(isinstance(proof, dict) and proof.get("valid") is True and proof.get("passed") is True,
            "temporal native validation did not pass")
    verified = verify_spatial_sources(proof.get("source_receipt"), frozen, method)
    require(proof.get("physical_reference_frame") == 0, "native proof must use physical frame0")
    cases = proof.get("cases", {})
    height = method == TEMPORAL_HEIGHT_METHOD
    rows, chunks = (45 if height else 80), axis_chunks(method)
    random_case = "random_two_widths" if height else "random_two_heights"
    tail_field = "tail_query_heights_checked" if height else "tail_query_columns_checked"
    cross_field = "cross_width_route" if height else "cross_height_route"
    require(set(cases) == {random_case, "zero_query_ties_empty_bulks"}, "temporal native cases incomplete")
    for name, case in cases.items():
        require(case.get("seed_cells") == 4 * rows and case.get("seed_mismatches") == 0
                and case.get("score_max_rel", float("inf")) <= 2e-5
                and case.get("merge_cells") == 48 and case.get("merge_max_abs", float("inf")) <= .01
                and case.get("bias_max_abs", float("inf")) <= 2e-5
                and case.get("route_query_vectors") == 8 * rows
                and case.get("route_max_abs", float("inf")) <= .02
                and case.get("route_rmse", float("inf")) <= .0025
                and case.get("all_outputs_finite") is True
                and case.get(tail_field) == list(range(32 if height else 64, rows))
                and case.get("query_times_checked") == [0, 1, 16, 20]
                and case.get(cross_field) is True
                and case.get("rf0_nonzero_frame_perturbation_invariant") is True,
                "temporal native case failed: " + name)
    require(cases["zero_query_ties_empty_bulks"].get("empty_bulk_count") == 2 * 21 * 2 * rows * 3,
            "temporal empty-bulk witness missing")
    full = proof.get("full_shape_smoke", {})
    require(full.get("qkv_shape") == [1, 40, 75600, 128] and full.get("dtype") == "bfloat16"
            and full.get("finite_output") is True and full.get("route_kernel_calls") == chunks
            and full.get("counts") == dict(core_calls=1, selector_calls=1, merge_wrapper_calls=chunks, stable_kernel_calls=chunks),
            "temporal full-shape witness missing")
    return verified


def install_spatial_audit(stack, audit, wan, token, recorder, frozen):
    """Keep original all-layer forward/timing; replace only the inner C4 core."""
    require(audit.method in AXIS_METHODS, "axis installer called for another method")
    spatial = importlib.import_module(axis_module_name(audit.method))
    factory_name = {SPATIAL_TRANSPOSE_METHOD: "create_spatial_transpose_core",
                    TEMPORAL_KEY_METHOD: "create_temporal_key_core",
                    TEMPORAL_HEIGHT_METHOD: "create_temporal_key_height_core"}.get(audit.method)
    factory = getattr(spatial, factory_name)
    operator = factory()
    chunks = axis_chunks(audit.method)
    require(operator.seed_mode == VIDEO_MODES[audit.method], "wrong spatial seed mode")
    metadata = operator.metadata()
    require(metadata["layout"] == audit.declaration["layout"], "spatial layout changed")
    verify_spatial_sources(metadata["source_receipt"], frozen, audit.method)
    if audit.method in TEMPORAL_METHODS:
        require(metadata.get("physical_reference_frame") == 0 and metadata.get("head_chunk") == 40 // chunks,
                "temporal axis changed physical RF0 or safe head chunk")
    selector = (spatial.fisher_seed_rows if audit.method == TEMPORAL_HEIGHT_METHOD
                else spatial.fisher_seed_columns)
    originals = {"forward": wan.WanSelfAttention.forward, "core": wan.token_merging_attn,
                 "replaced_inner_core": token._hard_c4_attention,
                 "spatial_core": type(operator).__call__, "selector": selector,
                 "spatial_merge_route": spatial.merge_and_full_route}
    audit.identities = {name: callable_identity(function, frozen) for name, function in originals.items()}
    require(audit.identities["selector"]["callable"] == audit.declaration["selector_override"], "wrong spatial selector")
    audit.spatial_core, audit.spatial_metadata = operator, metadata

    def forward(module, x, grid_size, rope_cache):
        audit.enter_layer(module, grid_size)
        result = originals["forward"](module, x, grid_size, rope_cache)
        audit.leave_layer()
        return result

    def core(*args, **kwargs):
        audit.enter_core(args, kwargs)
        return originals["core"](*args, **kwargs)

    def spatial_inner(query, key, value, **kwargs):
        require(audit.active_layer is not None and audit.qkv is not None, "spatial inner core outside original timed attention")
        expected = dict(seed_query=None, h=45, w=80, h_reduce=1, w_reduce=1, sm_scale=128 ** -.5,
                        query_outer_chunk=21, head_chunk=40 // chunks, hard_route_backend="hard_c4",
                        exact_intra_frame_attention=False, exact_neighbor_window_columns=0, exact_query_rows=0,
                        first_frame_full_attention=False, full_attention_anchor_time_count=0,
                        full_attention_anchor_row_count=0, full_attention_anchor_width_count=0,
                        full_attention_anchor_scale=1.0, seed_mode=VIDEO_MODES[audit.method], dense_heads=())
        require(set(kwargs) == set(expected) | {"seed_table_cache", "cache_timestep"}
                and all(kwargs[name] == value for name, value in expected.items()), "spatial inner-core options changed")
        cache = kwargs["seed_table_cache"]
        require(cache is None or (isinstance(cache, dict) and not cache), "spatial RF0 forbids seed-cache reuse")
        for tensor, meta in zip((query, key, value), audit.qkv):
            require(tensor_metadata(tensor) == meta, "spatial core changed original current QKV")
        before = operator.counts()
        previous = audit.core_calls - 1
        require(before == dict(core_calls=previous, selector_calls=previous,
                               merge_wrapper_calls=chunks * previous, stable_kernel_calls=chunks * previous),
                "spatial core was missing/repeated or used outside this worker")
        # Token/head transposes and the actual spatial copies remain within
        # Wan's original complete-attention timer; RoPE is never reinterpreted.
        result = operator(*(tensor.transpose(1, 2) for tensor in (query, key, value)))
        result = result.transpose(1, 2).contiguous()
        after = operator.counts()
        require(after == dict(core_calls=audit.core_calls, selector_calls=audit.core_calls,
                              merge_wrapper_calls=chunks * audit.core_calls, stable_kernel_calls=chunks * audit.core_calls),
                "axis core did not execute one selector and all head chunks")
        audit.selector_calls = after["selector_calls"]
        audit.merge_calls = audit.stable_merge_wrapper_calls = after["merge_wrapper_calls"]
        audit.stable_merge_kernel_calls = after["stable_kernel_calls"]
        audit.spatial_metadata = operator.metadata()
        audit.spatial_route_calls = audit.spatial_metadata["route_kernel_calls"]
        require(audit.spatial_route_calls == chunks * audit.core_calls, "missing actual axis full routes")
        require(tensor_metadata(result)["shape"] == SHAPE and str(result.dtype) == "torch.bfloat16",
                "spatial core did not restore original token order/shape")
        if audit.core_calls == 1:
            audit.first_call_metadata[axis_payload_key(audit.method)] = audit.spatial_metadata
        return result

    finish = recorder.finish_video
    def finish_video():
        timing = finish()
        audit.finish_video(timing)
        return timing

    stack.enter_context(patch.object(wan.WanSelfAttention, "forward", forward))
    stack.enter_context(patch.object(wan, "token_merging_attn", core))
    stack.enter_context(patch.object(token, "_hard_c4_attention", spatial_inner))
    stack.enter_context(patch.object(recorder, "finish_video", finish_video))


def install_audit(stack, audit, wan, token, recorder, frozen):
    if audit.method in AXIS_METHODS:
        return install_spatial_audit(stack, audit, wan, token, recorder, frozen)
    originals = {"forward": wan.WanSelfAttention.forward, "core": wan.token_merging_attn,
                 "merge": getattr(token, MERGE_SYMBOL)}
    symbol = audit.declaration["selector_symbol"]
    if audit.declaration["selector_override"]:
        module_name, function_name = audit.declaration["selector_override"].rsplit(".", 1)
        selector = getattr(importlib.import_module(module_name), function_name)
    else:
        selector = getattr(token, symbol)
    originals["selector"] = selector
    for name, function in originals.items():
        audit.identities[name] = callable_identity(function, frozen)
    expected_selector = audit.declaration["selector_override"] or (
        "attention.adaptive_seed_triton.hellinger_span_seed_rows_triton" if symbol == "hellinger_span_seed_rows_triton"
        else "attention.entropy_mass_seed_triton.entropy_mass_seed_rows_triton"
    )
    require(audit.identities["selector"]["callable"] == expected_selector, "wrong loaded selector implementation")
    zero_adapter = None
    mean_adapter = None
    stable_adapter = None
    c8_adapter = None
    if audit.method in C10_METHODS:
        from analysis.fisher_c10_video_adapter import FisherC10MergeAdapter
        c8_adapter = FisherC10MergeAdapter()
        audit.first_call_metadata["intermediate_kv"] = {"scaled_fp8_kv": False, "dtype": "bfloat16"}
        audit.identities["capacity_adapter"] = callable_identity(FisherC10MergeAdapter.__call__, frozen)
    if audit.method == "zero_fisher_rf0":
        from analysis.zero_anchor_video_adapter import ZeroSeedMergeAdapter
        zero_adapter = ZeroSeedMergeAdapter(originals["merge"],
            importlib.import_module("attention.hard_c4_wan21_14b_720p_triton"))
        audit.identities["synthetic_merge_adapter"] = callable_identity(ZeroSeedMergeAdapter.__call__, frozen)
    elif audit.method in MEAN_METHODS:
        from analysis.mean_anchor_video_adapter import MeanSeedMergeAdapter
        mean_adapter = MeanSeedMergeAdapter(originals["merge"],
            importlib.import_module("attention.hard_c4_wan21_14b_720p_triton"))
        audit.identities["synthetic_merge_adapter"] = callable_identity(MeanSeedMergeAdapter.__call__, frozen)
    elif audit.method in STABLE_SQ1_METHODS:
        from analysis.stable_sq1_video_adapter import StableSQ1MergeAdapter, load_generated_kernel
        hard = importlib.import_module("attention.hard_c4_wan21_14b_720p_triton")
        frozen = Path(frozen).resolve()
        generated_path = frozen.parent / "stable_kernel.py"
        require(not generated_path.is_relative_to(frozen), "stable artifact must be outside frozen source")
        kernel, receipt = load_generated_kernel(getattr(hard, "_hard_c4_720p_temporal_shared_merge_kernel"), generated_path)
        audit.generated_stable_kernel = receipt | {"bytes": generated_path.stat().st_size}
        verify_generated_stable_kernel(audit)
        audit.identities["stable_merge_kernel"] = audit.generated_stable_kernel
        stable_adapter = StableSQ1MergeAdapter(originals["merge"], hard, kernel)
        audit.identities["stable_merge_adapter"] = callable_identity(StableSQ1MergeAdapter.__call__, frozen)
        audit.identities["stable_kernel_loader"] = callable_identity(load_generated_kernel, frozen)

    def forward(module, x, grid_size, rope_cache):
        audit.enter_layer(module, grid_size)
        result = originals["forward"](module, x, grid_size, rope_cache)
        audit.leave_layer()
        return result

    def core(*args, **kwargs):
        audit.enter_core(args, kwargs)
        if mean_adapter is not None:
            mean_adapter.begin_attention()
        result = originals["core"](*args, **kwargs)
        if mean_adapter is not None:
            mean_adapter.end_attention()
        return result

    def select(*args, **kwargs):
        audit.enter_selector(args, kwargs)
        result = selector(*args, **kwargs)
        audit.leave_selector(result)
        return result

    def merge(*args, **kwargs):
        audit.enter_merge(args, kwargs)
        if c8_adapter is not None:
            result = c8_adapter(*args, **kwargs)
            require(c8_adapter.calls == audit.merge_calls, "C8 adapter call missing")
            return result
        if stable_adapter is not None:
            result = stable_adapter(*args, **kwargs)
            audit.stable_merge_kernel_calls = stable_adapter.kernel_calls
            audit.stable_merge_wrapper_calls = stable_adapter.calls
            require(audit.stable_merge_kernel_calls == audit.stable_merge_wrapper_calls == audit.merge_calls,
                    "stable adapter did not execute exactly one kernel per merge")
            return result
        if zero_adapter is not None:
            result = zero_adapter(*args, **kwargs)
            audit.synthetic_seed_kernel_calls = zero_adapter.kernel_calls
            if audit.merge_calls == 1:
                audit.first_call_metadata["analytic_zero_seed"] = {
                    "slot":0, "construction":"new_zeros; no query values read",
                    "tensor":tensor_metadata(zero_adapter.zeros),
                    "real_seed_slots":[1,2,3], "query_values_for_real_slots":"current_frame_actual_Q",
                }
            return result
        if mean_adapter is not None:
            result = mean_adapter(*args, **kwargs)
            audit.synthetic_seed_kernel_calls = mean_adapter.kernel_calls
            audit.mean_seed_constructions = mean_adapter.mean_computations
            if audit.merge_calls == 1:
                audit.first_call_metadata["analytic_mean_seed"] = {
                    "slot":0, "construction":"all45 spatial query rows, FP32 average then BF16; once per attention",
                    "tensor":tensor_metadata(mean_adapter.means),
                    "real_seed_slots":[1,2,3], "query_values_for_real_slots":"current_frame_actual_Q",
                    "selector_query_frame":0, "mean_query_frames":"each current frame independently",
                }
            return result
        return originals["merge"](*args, **kwargs)

    finish = recorder.finish_video
    def finish_video():
        timing = finish()
        audit.finish_video(timing)
        return timing

    stack.enter_context(patch.object(wan.WanSelfAttention, "forward", forward))
    stack.enter_context(patch.object(wan, "token_merging_attn", core))
    stack.enter_context(patch.object(token, symbol, select))
    stack.enter_context(patch.object(token, MERGE_SYMBOL, merge))
    stack.enter_context(patch.object(recorder, "finish_video", finish_video))


def planned_prompt_indices(provenance):
    planned = provenance["planned"]
    start, end = planned.get("prompt_start", 0), planned["num_videos"]
    require(type(start) is int and type(end) is int and 0 <= start < end <= 20, "invalid planned prompt range")
    return list(range(start, end))


def generation_argv(run_dir, provenance, checkpoint):
    frozen = Path(run_dir) / "source"
    return [str(frozen / "generate.py"), "--config", str(frozen / provenance["config"]),
            "--checkpoint", str(Path(checkpoint).resolve()), "--prompt_path", str(frozen / PROMPTS),
            "--output_dir", str(Path(run_dir) / "videos"), "--runtime_json", str(Path(run_dir) / "runtime.json"),
            "--num_videos", str(provenance["planned"]["num_videos"]), "--num_samples", "1", "--batch_size", "1", "--seed", "0",
            "--prompt_start", str(provenance["planned"].get("prompt_start", 0)),
            "--dtype", "bf16", "--fps", "16", "--video_preset", "ultrafast", "--video_crf", "23"]


def execute(run_dir, method, checkpoint, *, parent_purity_managed, spatial_native_proof=None):
    require(parent_purity_managed, "parent-owned clean GPU preflight and continuous purity monitoring required")
    check_environment()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(bool(visible) and "," not in visible and visible != "-1", "expose exactly one parent-selected GPU")
    run_dir = Path(run_dir).resolve()
    provenance = verify_snapshot(run_dir, method, checkpoint)
    prompt_indices = planned_prompt_indices(provenance)
    num_videos = len(prompt_indices)
    native_path = None
    if method == SPATIAL_TRANSPOSE_METHOD:
        require(spatial_native_proof is not None and Path(spatial_native_proof).is_absolute(),
                "spatial transpose requires explicit prior native-proof path")
    if method in AXIS_METHODS and spatial_native_proof is not None:
        require(Path(spatial_native_proof).is_absolute(), "native proof path must be absolute")
        native_path = Path(spatial_native_proof).resolve(strict=True)
        proof_hash = digest(native_path)
        native_proof = json.loads(native_path.read_text(encoding="utf-8"))
        native_sources = validate_axis_proof(native_proof, run_dir / "source", method)
    elif method not in AXIS_METHODS:
        require(spatial_native_proof is None, "native spatial proof provided for another method")
    for name in OWN_SOURCES:
        require(digest(ROOT / name) == digest(run_dir / "source" / name),
                f"running wrapper/helper differs from prepared snapshot: {name}")
    for name in ("videos", "runtime.json", "dispatch_audit.json", "run_manifest.json", "generation.exitcode"):
        require(not (run_dir / name).exists(), f"refusing resume/overwrite: {name}")
    if method in STABLE_SQ1_METHODS:
        require(not (run_dir / "stable_kernel.py").exists(), "refusing generated stable source overwrite")
    if method in AXIS_METHODS:
        require(not (run_dir / "native_validation.json").exists(), "refusing native proof overwrite")
    if native_path is not None:
        shutil.copyfile(native_path, run_dir / "native_validation.json")
        require(digest(native_path) == digest(run_dir / "native_validation.json") == proof_hash,
                "native proof changed while copying")
    frozen = run_dir / "source"
    require(not any(name == package or name.startswith(package + ".") for name in sys.modules
                    for package in ("wan", "attention", "pipeline", "utils", "model")),
            "production modules were imported before selecting frozen source")
    manifest = {"method": method + "_bf16score", "seed_mode": VIDEO_MODES[method],
                "config": str(frozen / provenance["config"]), "config_sha256": digest(frozen / provenance["config"]),
                "git_commit": provenance["remote_base_commit"], "checkpoint": str(Path(checkpoint).resolve()),
                **provenance["planned"], "dtype": "bf16", "parent_purity_managed": True,
                "timing_valid": None, "declaration": method_contract(method)}
    if native_path is not None:
        manifest["native_validation"] = {"path": str(run_dir / "native_validation.json"), "sha256": proof_hash,
                                         "reused_from": str(native_path), "verified_sources": native_sources,
                                         "scope": "independent native correctness proof only; not prior video completion/quality"}
    write_json(run_dir / "run_manifest.json", manifest, exclusive=True)
    audit = DispatchAudit(method, run_dir / "dispatch_audit.json", num_videos=num_videos)
    old_cwd, old_path = Path.cwd(), list(sys.path)
    try:
        os.chdir(frozen)
        sys.path[:0] = [str(frozen), str(frozen / "analysis")]
        os.environ["WAN_MODEL_ROOT"] = provenance["external_assets"]["WAN_MODEL_ROOT"]
        os.environ["TOKEN_MERGING_FORCE_TORCH_RMSNORM"] = "1"
        if method in C10_METHODS:
            from analysis.fisher_c10_video_adapter import validate_native
            proof = validate_native()
            write_json(run_dir / "native_validation.json", proof, exclusive=True)
        if method in TEMPORAL_METHODS and native_path is None:
            # The parent already owns/monitors this single GPU. Validate new
            # kernels before constructing any model; generate.main resets
            # seed0 afterwards, so synthetic diagnostic RNG cannot leak in.
            module = importlib.import_module(axis_module_name(method))
            native_proof = module.validate_native(run_dir / "native_validation")
            native_sources = validate_axis_proof(native_proof, frozen, method)
            write_json(run_dir / "native_validation.json", native_proof, exclusive=True)
            proof_hash = digest(run_dir / "native_validation.json")
            manifest["native_validation"] = {
                "path": str(run_dir / "native_validation.json"), "sha256": proof_hash,
                "verified_sources": native_sources, "reused_from": None,
                "scope": "independent pre-model synthetic diagnostic; generation.main subsequently resets seed0",
            }
            write_json(run_dir / "run_manifest.json", manifest)
        namespace = runpy.run_path(str(frozen / "generate.py"), run_name="quality_first_generate")
        wan = importlib.import_module("wan.modules.model")
        token = importlib.import_module("attention.token_merging")
        recorder = importlib.import_module("attention.runtime_timing").ATTENTION_RUNTIME_RECORDER
        with ExitStack() as stack:
            install_audit(stack, audit, wan, token, recorder, frozen)
            if method in STABLE_SQ1_METHODS:
                manifest["generated_stable_kernel"] = verify_generated_stable_kernel(audit)
                write_json(run_dir / "run_manifest.json", manifest)
                audit.save(False)
            if method in AXIS_METHODS:
                manifest[axis_payload_key(method)] = audit.spatial_metadata
                write_json(run_dir / "run_manifest.json", manifest)
                audit.save(False)
            stack.enter_context(patch.object(sys, "argv", generation_argv(run_dir, provenance, checkpoint)))
            namespace["main"]()
        runtime = json.loads((run_dir / "runtime.json").read_text(encoding="utf-8"))
        require(runtime["checkpoint_weights"] == "raw_state_dict", "unexpected checkpoint branch")
        records = runtime["records"]
        require(len(records) == num_videos and [row["prompt_index"] for row in records] == prompt_indices, "runtime prompt coverage wrong")
        require(all(row["attention_calls"] == row["full_attention_calls"] == 160 for row in records), "runtime incomplete")
        names = [f"{index:03d}-00.mp4" for index in prompt_indices]
        require(sorted(path.name for path in (run_dir / "videos").glob("*.mp4")) == names, "video filenames incomplete")
        require(all((run_dir / "videos" / name).stat().st_size > 0 for name in names), "empty video")
        verify_snapshot(run_dir, method, checkpoint)
        manifest["checkpoint_weights"] = runtime["checkpoint_weights"]
        if method in STABLE_SQ1_METHODS:
            manifest["generated_stable_kernel"] = verify_generated_stable_kernel(audit)
            manifest["stable_merge_kernel_calls"] = audit.stable_merge_kernel_calls
            manifest["stable_merge_wrapper_calls"] = audit.stable_merge_wrapper_calls
            manifest["merge_route_head_chunk_calls"] = audit.merge_calls
        if method in AXIS_METHODS:
            audit.spatial_core.verify_kernel()
            require(digest(run_dir / "native_validation.json") == proof_hash, "run-local native proof changed")
            validate_axis_proof(json.loads((run_dir / "native_validation.json").read_text()), frozen, method)
            manifest[axis_payload_key(method)] = audit.spatial_metadata
        write_json(run_dir / "run_manifest.json", manifest)
        audit.save(True)
    except BaseException as error:
        audit.save(False, f"{type(error).__name__}: {error}")
        (run_dir / "generation.exitcode").write_text("1\n", encoding="utf-8")
        raise
    else:
        (run_dir / "generation.exitcode").write_text("0\n", encoding="utf-8")
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(VIDEO_MODES), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-videos", type=int, default=20)
    parser.add_argument("--prompt-start", type=int, default=0)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--execute-prepared", action="store_true")
    parser.add_argument("--parent-purity-managed", action="store_true")
    parser.add_argument("--spatial-native-proof", type=Path)
    args = parser.parse_args()
    if not args.execute_prepared:
        prepare(args.run_dir, args.method, args.checkpoint, num_videos=args.num_videos, prompt_start=args.prompt_start)
    if args.prepare_only:
        print(f"Prepared frozen source/config only: {args.run_dir.resolve()}")
        return
    execute(args.run_dir, args.method, args.checkpoint, parent_purity_managed=args.parent_purity_managed,
            spatial_native_proof=args.spatial_native_proof)


if __name__ == "__main__":
    main()
