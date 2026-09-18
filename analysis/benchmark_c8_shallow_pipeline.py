"""Bounded, exclusively guarded exact-C8 paired-tile pipeline experiment.

CPU compilation is a separate preparatory mode. The run uses one reusable workspace,
bitwise gates on raw intermediate/output storage, and full H40 ABBA timing.
No videos are generated or inspected. Every complete call recomputes seeds.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CAPTURES = Path("/local2/denghaoran/GridMergeAttention/experiments/gma_c8_cuda_150ms_20260908")


def summary(values):
    return {"n": len(values), "mean_ms": statistics.fmean(values),
            "median_ms": statistics.median(values), "samples_ms": values}


def paired(rows):
    deltas = [r["control"] - r["candidate"] for r in rows]
    mean = statistics.fmean(deltas)
    sem = statistics.stdev(deltas) / len(deltas)**0.5 if len(deltas) > 1 else 0.
    return {"blocks": len(rows), "mean_saving_ms": mean,
            "sem_ms": sem, "ci95_approx_ms": [mean-1.96*sem, mean+1.96*sem],
            "wins": sum(x > 0 for x in deltas), "block_means": rows,
            "control_mean_ms": statistics.fmean(r["control"] for r in rows),
            "candidate_mean_ms": statistics.fmean(r["candidate"] for r in rows)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--gpu", type=int)
    p.add_argument("--blocks", type=int, default=16)
    p.add_argument("--layers", default="00,19,39")
    p.add_argument("--sources", nargs="*")
    args = p.parse_args()
    if args.output.exists():
        p.error("refusing to overwrite result")
    if args.blocks < 2:
        p.error("at least two prespecified ABBA blocks required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # nvcc and Triton temporary files must not fill scai7's small system disk.
    runtime_tmp = args.output.parent.resolve() / "runtime_tmp"
    runtime_tmp.mkdir(exist_ok=True)
    os.environ.update(TORCH_EXTENSIONS_DIR=str(args.output.parent / "torch_extensions"),
                      TRITON_CACHE_DIR=str(args.output.parent / "triton_cache"),
                      TORCH_CUDA_ARCH_LIST="9.0a", MAX_JOBS="2", OMP_NUM_THREADS="4",
                      TMPDIR=str(runtime_tmp))
    ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libstdc++.so.6", mode=ctypes.RTLD_GLOBAL)
    names = args.sources or ["fisher_c8_pair_serial_lb3_v0.cu",
                            "fisher_c8_pair_overlap_lb3_v0.cu",
                            "fisher_c8_pair_overlap_lb4_v0.cu"]
    paths = [ROOT / "native_fisher" / n for n in names]
    result = {"scope": "Fisher exact C8 H40: selector + pack + merge + full route",
              "capacity": 8, "singleton_count": 0, "baseline_commit": "57f269f",
              "sources": {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "prespecified_abba_blocks": args.blocks, "candidates": {}, "status": "preparing"}
    stop = threading.Event()
    monitor_errors, monitor_samples = [], []

    def save():
        result["monitor_errors"] = monitor_errors
        result["monitor_samples"] = monitor_samples
        temp = args.output.with_suffix(".partial.json")
        temp.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temp.replace(args.output)

    if args.compile_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from native_fisher.fused_c8_merge import extension
        from native_fisher.fused_c8_selector import extension as selector_extension
        from kernels.best.c8_r2 import gma_c8_r2 as retained
        for stem, driver in (("gma_c8_cuda_probe", False), ("b1_gs_regpipe_v0", False),
                             (retained.MERGE_SOURCE, True), ("gma_c8_route_pipeline", True)):
            retained._extension(stem, driver)
        selector_extension()
        for path in paths:
            try:
                mod = extension(path)
                resource = subprocess.run(["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", mod.__file__],
                                          text=True, capture_output=True, check=True).stdout
                result["candidates"][path.name] = {"compiled": True, "module": mod.__file__,
                                                    "static_resources": resource}
                print(json.dumps({"compiled": path.name}), flush=True)
            except Exception as e:
                result["candidates"][path.name] = {"compiled": False, "error": str(e)}
            save()
        result["status"] = "compiled"
        save()
        return

    if args.gpu is None:
        p.error("--gpu is required for execution")
    from analysis.launch_quality_first_videos import require_clean, gpu_processes, now
    initial = require_clean(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = initial["gpu"]["uuid"]
    result["initial_gpu"] = initial
    import torch
    from native_fisher.fused_c8_merge import extension
    from analysis.fisher_c8_exact import ExactFisherC8Attention
    from torch.nn.attention import sdpa_kernel, SDPBackend
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    # Claim the actual reusable workspace while a concurrent CPU compiler may
    # finish candidates. No GPU kernels are timed until every module is ready.
    result["prelaunch_gpu"] = require_clean(args.gpu, expected_uuid=initial["gpu"]["uuid"])

    def monitor():
        while not stop.is_set():
            try:
                rows = gpu_processes(initial["gpu"]["uuid"])
                monitor_samples.append({"utc": now(), "processes": rows})
                if any(row["pid"] != os.getpid() for row in rows):
                    monitor_errors.append({"utc": now(), "foreign_processes": rows})
            except Exception as e:
                monitor_errors.append({"utc": now(), "error": repr(e)})
            stop.wait(1)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    def clean():
        if monitor_errors:
            raise RuntimeError("GPU contamination or monitor failure; timing invalid")

    def timed(call):
        clean()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        clean()
        return start.elapsed_time(end)

    def raw_equal(a, b):
        dtype = torch.int16 if a.element_size() == 2 else torch.int32
        return bool(torch.equal(a.view(dtype), b.view(dtype)))

    def raw_mismatches(a, b):
        dtype = torch.int16 if a.element_size() == 2 else torch.int32
        av, bv = a.view(dtype).reshape(-1), b.view(dtype).reshape(-1)
        chunk = 1 << 26
        return sum(int((av[i:i+chunk] != bv[i:i+chunk]).sum())
                   for i in range(0, av.numel(), chunk))

    try:
        op = ExactFisherC8Attention()
        backend = op.backend
        retained = backend.merge
        # Reuse the actual validation buffers across every capture and timing
        # block. Stable allocation avoids repeatedly releasing/reclaiming 30GiB.
        refs = [torch.empty_like(x) for x in (backend.mk,backend.mv,backend.bias)]
        result["baseline"] = op.resources()
        result["status"] = "workspace_allocated_loading_candidates"
        save()
        print(json.dumps({"workspace_allocated_gpu":args.gpu,"baseline":result["baseline"]}),flush=True)
        modules = {path.name: extension(path) for path in paths}
        from analysis.c8_numerical_gate import NUMERICAL_SOURCES, compare_tensor
        for name, mod in modules.items():
            values = mod.resources()
            result["candidates"][name] = {"resources": dict(zip(
                ("registers", "local_bytes", "smem_bytes", "ctas_per_sm"), values)), "layers": {}}
            result['candidates'][name]['gate_contract'] = (
                'bounded_float_roundoff' if name in NUMERICAL_SOURCES else 'bitwise')
        result["status"] = "running"
        save()
        for layer in args.layers.split(","):
            clean()
            payload = torch.load(CAPTURES / f"real_full40_step0_layer{layer}.pt", map_location="cpu", weights_only=True)
            q, k, v = [payload[n].reshape(1,21,45,80,40,128).cuda() for n in ("q", "k", "v")]
            del payload
            output = torch.empty_like(q)
            reference = torch.empty_like(q)
            backend.merge = retained
            op(q, k, v, output=reference)
            seeds = op.last_seeds.clone()
            # Full H5 materialized buffers cover all query frames, columns and
            # key groups (including the unpaired last query frame).
            retained.run(backend.packed_q,k,v,seeds,backend.mk,backend.mv,backend.bias,
                         0,5,105,128**-0.5,backend.empty,0,1)
            for dst,src in zip(refs,(backend.mk,backend.mv,backend.bias)):
                dst.copy_(src)
            torch.cuda.synchronize()
            for name, mod in modules.items():
                row = result["candidates"][name]["layers"][layer] = {}
                clean()
                mod.run(backend.packed_q,k,v,seeds,backend.mk,backend.mv,backend.bias,
                        0,5,105,128**-0.5,backend.empty,0,1)
                row["intermediate_mismatches"] = {n: raw_mismatches(a,b) for n,a,b in
                    zip(("mk","mv","bias"),(backend.mk,backend.mv,backend.bias),refs)}
                if name in NUMERICAL_SOURCES:
                    row['numerical_errors'] = {n: compare_tensor(a,b,kind=n) for n,a,b in
                        zip(('mk','mv','bias'),(backend.mk,backend.mv,backend.bias),refs)}
                backend.merge = mod
                op(q,k,v,output=output)
                row["output_bitwise_equal"] = raw_equal(output,reference)
                row["seeds_equal"] = raw_equal(op.last_seeds.to(torch.int32),seeds.to(torch.int32))
                row["finite"] = bool(torch.isfinite(output).all())
                row["passed"] = (not any(row["intermediate_mismatches"].values()) and
                                 row["output_bitwise_equal"] and row["seeds_equal"] and row["finite"])
                if name in NUMERICAL_SOURCES:
                    row['numerical_errors']['output'] = compare_tensor(output,reference,kind='output')
                    row['passed'] = (row['seeds_equal'] and row['finite'] and
                                     all(x['passed'] for x in row['numerical_errors'].values()))
                print(json.dumps({"layer":layer,"candidate":name,"gate":row}),flush=True)
                save()
            def call(mod):
                backend.merge=mod
                op(q,k,v,output=output)

            for name,mod in modules.items():
                row=result["candidates"][name]["layers"][layer]
                if not row["passed"]:
                    continue
                for _ in range(2):
                    call(retained);call(mod)
                torch.cuda.synchronize()
                blocks=[]
                samples={"control":[],"candidate":[]}
                for block in range(args.blocks):
                    order=("control","candidate","candidate","control") if block%2==0 else (
                        "candidate","control","control","candidate")
                    local={"control":[],"candidate":[]}
                    for key in order:
                        ms=timed(lambda: call(retained if key=="control" else mod))
                        samples[key].append(ms);local[key].append(ms)
                    blocks.append({key:statistics.fmean(vals) for key,vals in local.items()})
                row["complete_call"]=paired(blocks)
                row["samples"]=samples
                # Merge timings are diagnostic and never substitute for H40.
                row["merge_h5_ms"]={}
                for key,module in (("control",retained),("candidate",mod)):
                    row["merge_h5_ms"][key]=summary([timed(lambda: module.run(
                        backend.packed_q,k,v,op.last_seeds,backend.mk,backend.mv,backend.bias,
                        0,5,105,128**-0.5,backend.empty,0,1)) for _ in range(6)])
                from analysis.c8_pipeline_stage_events import profile_stages
                row['stage_events'] = {}
                for key,module in (("control",retained),("candidate",mod)):
                    backend.merge = module
                    row['stage_events'][key] = profile_stages(op,q,k,v,output,clean)
                print(json.dumps({"layer":layer,"candidate":name,"timing":row["complete_call"]}),flush=True)
                save()
                if name in ('fisher_c8_clock_phases_v37.cu','fisher_c8_clock_predicated_v38.cu',
                            'fisher_c8_clock_lb3_v39.cu'):
                    from analysis.c8_clock_phase_diagnostic import profile_merge_clocks
                    row['clock_phase_diagnostic'] = profile_merge_clocks(
                        mod,backend,k,v,seeds,refs,clean)
                    save()

            flat=[x.reshape(1,75600,40,128).transpose(1,2) for x in (q,k,v)]
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                dense=lambda: torch.nn.functional.scaled_dot_product_attention(*flat)
                dense();dense()
                sdpa=summary([timed(dense) for _ in range(6)])
            result.setdefault("sdpa",{})[layer]=sdpa
            from analysis.c8_empty_cluster_diagnostic import empty_cluster_histogram
            result.setdefault('empty_cluster_diagnostic',{})[layer] = empty_cluster_histogram(
                backend,retained,q,k,v,op.last_seeds,clean)
            for name in modules:
                row=result["candidates"][name]["layers"][layer]
                if "complete_call" in row:
                    row["speedup_vs_sdpa"]=sdpa["mean_ms"]/row["complete_call"]["candidate_mean_ms"]
            del q,k,v,output,reference,seeds,flat,dense
            torch.cuda.empty_cache()
            save()
        clean()
        result["status"]="complete"
        result["valid_timing"]=True
    except BaseException as e:
        result.update(status="failed",error=repr(e),valid_timing=False)
        raise
    finally:
        stop.set()
        thread.join(5)
        if monitor_errors:
            result["valid_timing"]=False
        save()


if __name__ == "__main__":
    main()
