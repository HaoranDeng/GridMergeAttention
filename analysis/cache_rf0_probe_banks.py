"""Export audited L33/step0 RF0 probe banks without CUDA or model execution.

The output is a portable NumPy diagnostic cache, not an attention benchmark.
Only the three already saved BF16 captures and exact-normalizer receipts are
accepted. Production modules are never imported; the candidate-table function
is read by AST from its fingerprinted source and executed on CPU.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HEADS = (0, 10, 20, 30)
QUERY_COLUMNS = (0, 1, 20, 21, 40, 41, 60, 61, 75, 76)
KEY_FRAMES, KEY_ROWS = (0, 10, 20), (0, 11, 22, 33, 44)
ROWS, COLS, DIM, N, SCALE = 45, 80, 128, 75600, 128 ** -0.5
METHODS = ("partition_g256", "partition_oracle_allkeys", "partition_oracle_phase0")
CAPTURE_TENSORS = ("q", "k", "v", "dense_output")
ORACLE_SHA = "b2e6f3b8430087d40fdfc610681048878f2cdfa126bb62cd6d1760ab7c9408f4"
MANIFEST_SHA = "5c001dcf0a58a04379575e8390a7546e8f8e1f57a97d605d3c2c6b4c56d2f630"
PROMPT_MANIFEST_SHA = "9e3dc8eb560d1d0991221ce9b3b5ecade5d8fb88c77b4bd6856960265f5f5ac7"
PROMPT_FILE_SHA = "50293c2717c8f2df5773154d4680c74e636ecb08eb4fe13cdce7357d0ab9d121"
FIRST_PROMPT_SHA = "d0bf2d0fa5fd5f0a08da801d40232bb9e960c84d5d4889c175a0da4e4cc0aa05"
CANDIDATE_SHA = "ab288d3a1d70013f912a944f912337633cfc6fe2228b9ee4649febcb94c42e22"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def verify_fingerprint(path, receipt, known_sha=None):
    actual = fingerprint(path)
    require(actual["bytes"] == receipt["bytes"], f"size mismatch: {path}")
    require(actual["sha256"] == receipt["sha256"], f"SHA256 mismatch: {path}")
    if known_sha is not None:
        require(actual["sha256"] == known_sha, f"not the audited input: {path}")
    return actual


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_new(path, data):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def bank_coordinates():
    """[600,4], columns head/query_column/key_frame/key_row; row is innermost."""
    return np.asarray([(h, c, f, r) for h in HEADS for c in QUERY_COLUMNS
                       for f in KEY_FRAMES for r in KEY_ROWS], dtype=np.int64)


def flat_indices(coordinates):
    coordinates = np.asarray(coordinates)
    require(coordinates.ndim == 2 and coordinates.shape[1] == 4
            and np.issubdtype(coordinates.dtype, np.integer), "invalid bank coordinates")
    h, c, f, r = coordinates.T
    require(np.isin(h, HEADS).all() and ((0 <= c) & (c < COLS)).all()
            and ((0 <= f) & (f < 21)).all() and ((0 <= r) & (r < ROWS)).all(),
            "bank coordinate out of range")
    query = np.arange(ROWS, dtype=np.int64)[None, :] * COLS + c[:, None]
    key = f[:, None] * (ROWS * COLS) + r[:, None] * COLS + np.arange(COLS)[None, :]
    return query, key


def normalizer_arrays(raw, native, coordinates):
    require(raw.shape == (4, 3600, 2) and raw.dtype == np.float64, "wrong raw normalizers")
    require(native.shape == (1, 1, 45, 80, 4) and native.dtype == np.float32,
            "wrong native G256 normalizers")
    require(np.isfinite(raw).all() and np.isfinite(native).all(), "nonfinite normalizers")
    query, _ = flat_indices(coordinates)
    slots = np.asarray([HEADS.index(int(head)) for head in coordinates[:, 0]])
    exact = raw[slots[:, None], query, 0]
    sampled = native[0, 0].reshape(3600, 4)[query, slots[:, None]].astype(np.float64) * math.log(2)
    return exact, sampled


def validate_prompt_manifest(prompt):
    require(prompt["prompt_file_sha256"] == PROMPT_FILE_SHA and prompt["seed"] == 0
            and prompt["num_samples"] == 1 and prompt["canonical_contract_enforced"] is True,
            "wrong prompt contract")
    require([row["index"] for row in prompt["prompts"]] == [0, 1, 2], "wrong prompt indices")
    for row in prompt["prompts"]:
        require(hashlib.sha256(row["text"].encode("utf-8")).hexdigest() == row["sha256"],
                "prompt text/hash mismatch")
    require(prompt["prompts"][0]["sha256"] == FIRST_PROMPT_SHA, "wrong first prompt")


def validate_inputs(capture_root, oracle_root):
    oracle_path = oracle_root / "diagnostic.json"
    oracle_identity = fingerprint(oracle_path)
    require(oracle_identity["sha256"] == ORACLE_SHA, "oracle JSON is not the audited completed artifact")
    oracle = read_json(oracle_path)
    require(oracle["valid"] is True and oracle["complete"] is True
            and oracle["capture_validation"]["valid"] is True, "incomplete or invalid oracle")
    require(len(oracle["records"]) == 3, "expected three oracle records")
    require((oracle_root / "exit_code").read_text().strip() == "0"
            and (oracle_root / "purity.log").stat().st_size == 0, "oracle did not finish cleanly")
    receipts = oracle["capture_validation"]["inputs"]
    identities = {"oracle": oracle_identity}
    for name, known in (("capture_manifest.json", MANIFEST_SHA),
                        ("prompt_manifest.json", PROMPT_MANIFEST_SHA),
                        ("prompt_preflight_manifest.json", PROMPT_MANIFEST_SHA)):
        identities[name] = verify_fingerprint(capture_root / name, receipts[name], known)
    for name, path in (("exit_code", capture_root / "exit_code"),
                       ("wrapper_exit_code", capture_root.parent / "exit_code"),
                       ("wrapper_worker_exit_code", capture_root.parent / "worker_exit_code"),
                       ("wrapper_purity.log", capture_root.parent / "purity.log")):
        identities[name] = verify_fingerprint(path, receipts[name])
    manifest, prompt = read_json(capture_root / "capture_manifest.json"), read_json(capture_root / "prompt_manifest.json")
    validate_prompt_manifest(prompt)
    require(manifest["complete"] is True and manifest["checkpoint_weights"] == "raw_state_dict"
            and manifest["total_self_attention_calls"] == 480 and manifest["total_captures"] == 12,
            "capture producer contract mismatch")
    selected = [item for item in manifest["captures"] if item["step_index"] == 0]
    require(len(selected) == 3, "expected three step0 captures")
    jobs = []
    for index, (item, row) in enumerate(zip(selected, oracle["records"])):
        print(f"verifying audited prompt {index} capture and normalizer bytes", flush=True)
        name = f"prompt{index:03d}_call033_layer33.pt"
        require(item == row["metadata"] and item["capture_file"] == name
                and item["prompt_index"] == index and item["call_index"] == 33
                and item["layer_index"] == 33 and item["heads"] == list(HEADS)
                and item["shape"] == [1, N, 4, DIM] and item["dtype"] == "torch.bfloat16"
                and item["actual_timestep"] == 1000.0 and item["scale"] == SCALE
                and item["checkpoint_weights"] == "raw_state_dict"
                and item["prompt"] == prompt["prompts"][index] and item["sample_seed"] == index,
                f"capture metadata mismatch for prompt {index}")
        receipt = oracle["capture_validation"]["capture_files"][name]
        require(receipt == row["input_file"], "oracle capture receipts disagree")
        capture_identity = verify_fingerprint(capture_root / name, receipt)
        cache_name = f"prompt{index:03d}_exact_normalizers.pt"
        cache_receipt = row["normalizer_and_seed_cache"]
        require(Path(cache_receipt["path"]).name == cache_name, "wrong oracle cache filename")
        cache_identity = verify_fingerprint(oracle_root / cache_name, cache_receipt)
        jobs.append({"metadata": item, "capture": capture_identity, "normalizers": cache_identity})
    return oracle, identities, jobs


def validate_cache_metadata(metadata, job, oracle):
    item = job["metadata"]
    require(metadata["capture_file"] == item["capture_file"]
            and metadata["capture_sha256"] == job["capture"]["sha256"]
            and metadata["prompt_index"] == item["prompt_index"]
            and metadata["step_index"] == 0 and metadata["router_frame"] == 0
            and metadata["heads"] == list(HEADS)
            and metadata["normalizer_layout"] == [1, 1, 45, 80, 4]
            and metadata["raw_natural_layout"] == [4, 3600, 2]
            and metadata["raw_natural_columns"] == ["allkeys", "phase0"]
            and metadata["eligible_candidate"] is False
            and metadata["normalizer_convention"] == oracle["normalizer_convention"]
            and metadata["evaluation_source_sha256"] == oracle["evaluation_provenance"]["source_sha256"],
            "oracle tensor-cache metadata mismatch")


def validate_capture_metadata(payload, item):
    expected = {key: value for key, value in item.items() if key not in ("bytes", "sha256")}
    require(set(payload) == set(CAPTURE_TENSORS) | {"metadata"},
            "capture tensor keys differ from q/k/v/dense_output/metadata")
    require(payload["metadata"] == expected, "capture tensor metadata mismatch")


def production_candidates(torch, oracle):
    path = ROOT / "attention" / "adaptive_seed_pytorch.py"
    identity = fingerprint(path)
    require(identity["sha256"] == CANDIDATE_SHA
            == oracle["evaluation_provenance"]["source_sha256"]["attention/adaptive_seed_pytorch.py"],
            "candidate-table source changed")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == "multiscale_candidate_table"]
    require(len(nodes) == 1, "missing production candidate-table function")
    scope = {"torch": torch, "_MULTISCALE_TABLE_CACHE": {}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    table, anchor = scope["multiscale_candidate_table"](45, 32, "cpu", geometry_count=45)
    table, anchor = table.numpy().astype(np.int64), anchor.numpy().astype(np.int64)
    require(table.shape == (45, 32) and anchor.shape == (45,)
            and np.array_equal(table[np.arange(45), anchor], np.arange(45)), "bad production bank/anchor")
    return table, anchor, identity


def export_prompt(torch, job, oracle, table, anchors, output_dir):
    # Repeat identity checks immediately before deserialization, with CPU-only loads.
    verify_fingerprint(job["capture"]["path"], job["capture"])
    verify_fingerprint(job["normalizers"]["path"], job["normalizers"])
    payload = torch.load(job["capture"]["path"], map_location="cpu", weights_only=True)
    cache = torch.load(job["normalizers"]["path"], map_location="cpu", weights_only=True)
    validate_capture_metadata(payload, job["metadata"])
    for name in CAPTURE_TENSORS:
        tensor = payload[name]
        require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
                and tensor.dtype == torch.bfloat16 and tuple(tensor.shape) == (1, N, 4, DIM)
                and tensor.is_contiguous() and bool(torch.isfinite(tensor).all()), f"invalid original {name}")
    validate_cache_metadata(cache["metadata"], job, oracle)
    raw = cache["raw_log_partitions_natural_fp64"]
    native = cache["normalizers_log2_g256_scale"]["partition_g256"]
    require(raw.device.type == native.device.type == "cpu", "non-CPU normalizer cache")
    coordinates = bank_coordinates()
    query_indices, key_indices = flat_indices(coordinates)
    exact_z, sampled_z = normalizer_arrays(raw.numpy(), native.numpy(), coordinates)
    count = len(coordinates)
    logits = np.empty((count, ROWS, COLS), dtype=np.float64)
    values = np.empty((count, COLS, DIM), dtype=np.float32)
    seed_arrays = {name: np.empty((count, 4), dtype=np.int64) for name in METHODS}
    for name in METHODS:
        seed = cache["seed_tables"][name]
        require(seed.device.type == "cpu" and seed.dtype == torch.uint8
                and tuple(seed.shape) == (1, 1, 21, 45, 80, 4, 4), "invalid saved seed table")
    with torch.inference_mode():
        for index, (head, column, frame, row) in enumerate(coordinates):
            slot = HEADS.index(int(head))
            query = payload["q"][0, query_indices[index], slot].float().contiguous()
            key = payload["k"][0, key_indices[index], slot].float().contiguous()
            scores = (query @ key.T) * SCALE
            logits[index] = scores.double().numpy()
            values[index] = payload["v"][0, key_indices[index], slot].float().numpy()
            for name in METHODS:
                seed_arrays[name][index] = cache["seed_tables"][name][0, 0, frame, row, column, slot].numpy()
            if (index + 1) % 150 == 0:
                print(f"prompt {job['metadata']['prompt_index']}: {index + 1}/600 CPU QK banks", flush=True)
    require(np.isfinite(logits).all() and np.isfinite(values).all(), "nonfinite exported values")
    for seeds in seed_arrays.values():
        require(((0 <= seeds) & (seeds < ROWS)).all()
                and (np.diff(np.sort(seeds, axis=1), axis=1) > 0).all()
                and np.array_equal(seeds[:, 0], coordinates[:, 3]), "invalid sampled anchored seeds")
    arrays = {"full_logits": logits, "local_values": values, "global_log_Z": exact_z,
              "global_log_Z_G256": sampled_z, "head": coordinates[:, 0],
              "query_column": coordinates[:, 1], "key_frame": coordinates[:, 2],
              "key_row": coordinates[:, 3], "query_token_indices": query_indices,
              "key_token_indices": key_indices, "candidate_rows": table[coordinates[:, 3]],
              "anchor": anchors[coordinates[:, 3]], "uniform16_columns": np.arange(0, COLS, 5, dtype=np.int64)}
    arrays.update({"seed_rows_" + name: seeds for name, seeds in seed_arrays.items()})
    metadata = {"prompt_index": job["metadata"]["prompt_index"], "banks": count,
                "capture": job["capture"], "normalizers": job["normalizers"],
                "scale": SCALE, "router_frame": 0, "candidate_count": 32, "candidate_geometry_count": 45,
                "anchor_semantics": "candidate slot; candidate_rows[bank,anchor[bank]] == key_row[bank]",
                "global_log_Z_G256_semantics": "ln(sum exp over original G256); native saved log2 value * ln(2), no N/G factor",
                "precision": "original BF16 Q/K -> CPU FP32 matmul and FP32 scale -> float64 storage; not GPU bitwise parity",
                "saved_seed_semantics": "External partition C16 controls from oracle; NOT Mass/Hellinger C32 seed tables"}
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    destination = output_dir / f"prompt{metadata['prompt_index']:03d}_banks.npz"
    with destination.open("xb") as handle:
        np.savez(handle, **arrays)
    with np.load(destination, allow_pickle=False) as copied:
        require(set(copied.files) == set(arrays), "NPZ round-trip keys changed")
        for key, value in arrays.items():
            require(np.array_equal(copied[key], value), f"NPZ round-trip changed {key}")
    return {"prompt_index": metadata["prompt_index"], "metadata": metadata,
            "cache": fingerprint(destination),
            "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                       for key, value in arrays.items()}}


def check_cpu_environment(environment, threads):
    require(environment.get("CUDA_VISIBLE_DEVICES") in ("", "-1"),
            "Set CUDA_VISIBLE_DEVICES='' or '-1' before this CPU-only exporter")
    require(type(threads) is int and 1 <= threads <= 4, "CPU threads must be between 1 and 4")


def run(args):
    check_cpu_environment(os.environ, args.num_threads)
    require(not args.output_dir.exists(), "output-dir already exists; refuse overwrite/resume")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"schema_version": 1, "valid": False, "complete": False, "device": "cpu",
              "eligible_candidate": False,
              "mean_attention_ms": None, "speedup_vs_sdpa": None, "records": [],
              "scope": "Fixed 1800-bank cache only; no candidate/model/video execution or attention latency",
              "bank_order": "head, query_column, key_frame, key_row (innermost)",
              "heads": list(HEADS), "query_columns": list(QUERY_COLUMNS),
              "key_frames": list(KEY_FRAMES), "key_rows": list(KEY_ROWS),
              "source": fingerprint(Path(__file__)), "num_threads": args.num_threads}
    write_json_new(args.output_dir / "started.json", report)
    try:
        oracle, identities, jobs = validate_inputs(args.capture_root, args.oracle_root)
        import torch
        require(not torch.cuda.is_initialized(), "CUDA was initialized before CPU export")
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(1)
        torch.set_float32_matmul_precision("highest")
        report.update(inputs=identities, torch_version=str(torch.__version__), numpy_version=np.__version__)
        table, anchor, source_identity = production_candidates(torch, oracle)
        report["candidate_table_source"] = source_identity
        for job in jobs:
            row = export_prompt(torch, job, oracle, table, anchor, args.output_dir)
            require(not torch.cuda.is_initialized(), "CPU exporter unexpectedly initialized CUDA")
            report["records"].append(row)
            print(f"exported prompt {job['metadata']['prompt_index']}: 600 CPU probe banks", flush=True)
        report.update(valid=True, complete=True, total_banks=1800,
                      export_wall_seconds=time.perf_counter() - started, cuda_initialized=False)
        write_json_new(args.output_dir / "manifest.json", report)
        return report
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error),
                      export_wall_seconds=time.perf_counter() - started)
        write_json_new(args.output_dir / "export_error.json", report)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-threads", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
