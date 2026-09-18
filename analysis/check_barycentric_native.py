"""Small synthetic BF16 native correctness test, never a quality benchmark.

No model, captured QKV, calibration or training. A clean H200 parent runs
one worker and monitors exclusivity. Exact integer-input seed-table equality
is required before direct-video testing. No timing or video claim follows.

Default barycentric keeps its original two inputs / three methods (six
records). Explicit expected_fisher checks two inputs / one method: 7,200
one-key-frame banks at scale .125 and 14,400 two-key-frame banks at the
actual 128**-.5 scale. Both native score-audit paths must return identical
rows. Every candidate score is checked at the ACTUAL native pivot prefix
against an independent FP64 Gram oracle, with fixed pre-launch tolerances.

Explicit anchor_gap_guard uses the same two shapes/scales and checks the
actual Anchor32 incumbent, three per-target partition gaps, the challenger,
the accept decision, and preservation of its first three real rows.

Explicit phase_averaged checks T1 identity and T21 virtual time averaging on
7,200 + 151,200 banks. Its independent per-time FP64 transform is rounded to
BF16, then checked bitwise except the recorded sign of zeros. Rows, input
preservation and the fixed gamma127 FP32 dot forward-error bound are gates;
every native row must still match the independent FP64 oracle exactly.
FP32 score audits are not compared as FP64 values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCES = (
    "analysis/check_barycentric_native.py", "analysis/mean_seed_pytorch.py",
    "analysis/barycentric_medoid_seed.py", "analysis/mean_anchor_video_adapter.py",
    "attention/affine_hellinger_seed_triton.py", "attention/adaptive_seed_pytorch.py",
    "analysis/launch_quality_first_videos.py", "analysis/run_quality_first_videos.py",
    "analysis/quality_first_mode_contract.py",
    "analysis/zero_anchor_video_adapter.py",
    "analysis/expected_fisher_seed.py", "attention/expected_fisher_seed_triton.py",
    "analysis/check_anchor_gap_guard_case.py", "analysis/anchor_gap_guard_seed.py",
    "analysis/check_phase_averaged_case.py", "analysis/anchor_fullgrid_seed.py",
    "attention/phase_averaged_seed.py",
)
FAMILIES = ("barycentric", "expected_fisher", "anchor_gap_guard", "phase_averaged")
BARYCENTRIC_METHODS = ("mean_fisher32", "mean_anchor_fisher32", "barycentric_medoid32")
EXPECTED_METHOD = "fisher_expected"
Q_SHAPE = (1, 1, 45, 80, 2, 128)
BANKS_PER_FRAME = 45 * 80 * 2
# Preset before any GPU execution; exact row equality remains a separate gate.
SCORE_ATOL = 2e-10
SCORE_RTOL = 2e-11


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def source_hashes():
    # Importing an attention submodule first executes attention/__init__.py,
    # which imports other attention modules. Freeze all local attention Python
    # sources as well as the explicit checker/reference/parent helper closure.
    names = set(SOURCES) | {str(path.relative_to(ROOT)).replace("\\", "/")
                            for path in (ROOT / "attention").rglob("*.py")}
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(names)}


def freeze_sources(output, hashes):
    for name, expected in hashes.items():
        data = (ROOT / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError("source changed while freezing")
        destination = output / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(data)
    verify_source_hashes(output, hashes)


def verify_source_hashes(output, hashes):
    if source_hashes() != hashes:
        raise RuntimeError("live source fingerprint changed")
    for name, expected in hashes.items():
        if hashlib.sha256((output / "sources" / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("frozen source fingerprint changed")


def family_cases(family):
    """(input seed, number of independent key frames, dot-product scale)."""
    if family == "barycentric":
        return ((17, 1, 0.125), (53, 1, 0.125))
    if family in ("expected_fisher", "anchor_gap_guard"):
        return ((17, 1, 0.125), (53, 2, 128 ** -0.5))
    if family == "phase_averaged":
        return ((17, 1, 0.125), (53, 21, 128 ** -0.5))
    raise ValueError("unknown native-check family")


def expected_records(family):
    methods = BARYCENTRIC_METHODS if family == "barycentric" else (
        EXPECTED_METHOD if family == "expected_fisher" else family,)
    return tuple((seed, frames, scale, method, frames * BANKS_PER_FRAME)
                 for seed, frames, scale in family_cases(family) for method in methods)


def decode_bank_ids(ids):
    """B=1 layout [key_frame,45,80,2]; works for integers/NumPy/Torch."""
    head = ids % 2
    column = (ids // 2) % 80
    row = (ids // (2 * 80)) % 45
    frame = ids // BANKS_PER_FRAME
    return frame, row, column, head


def gram_scores_at_native_prefix(gram, slots):
    """FP64 Gram oracle at native slots, not the oracle's preferred pivots.

    slots includes the matched anchor in column zero. Inputs are validated by
    the caller; this backend-neutral arithmetic is CPU-tested with NumPy and
    runs with Torch FP64 on-device in the diagnostic worker. No native helpers
    or native intermediate values other than the chosen row identities enter.
    """
    batch = list(range(gram.shape[0]))
    residual = gram
    scores = []
    for step in range(3):
        score = residual[:, range(32), range(32)].clip(0, None)
        scores.append(score)
        if step < 2:
            pivot = slots[:, step + 1]
            norm = score[batch, pivot]
            column = residual[batch, :, pivot]
            factor = (norm + 2e-12) / (norm + 1e-12) ** 2
            residual = residual - column[:, :, None] * column[:, None, :] * factor[:, None, None]
    return scores


def validate_completed_result(result, family, hashes):
    """Parent-side fail-closed validation, including exact case identities."""
    specs = expected_records(family)
    if (result.get("complete") is not True or result.get("family") != family
            or result.get("source_sha256") != hashes
            or result.get("source_sha256_after") != hashes
            or result.get("shape") != list(Q_SHAPE)
            or len(result.get("records", [])) != len(specs)):
        raise RuntimeError("incomplete or inconsistent native check")
    for record, (seed, frames, scale, method, banks) in zip(result["records"], specs):
        if (record.get("input_seed") != seed or record.get("key_frames") != frames
                or record.get("scale") != scale or record.get("method") != method
                or record.get("banks") != banks or record.get("group_agreement") != 1.0
                or record.get("mismatched_banks") != 0
                or not isinstance(record.get("native_sha256"), str)
                or len(record["native_sha256"]) != 64):
            raise RuntimeError("native row gate or case identity failed")
        if family == "expected_fisher":
            if (record.get("anchor_mismatches") != 0 or record.get("nonunique_banks") != 0
                    or record.get("invalid_candidate_banks") != 0
                    or record.get("score_audit_mismatched_banks") != 0
                    or record.get("score_audit_on_off_equal") is not True
                    or record.get("score_atol") != SCORE_ATOL or record.get("score_rtol") != SCORE_RTOL
                    or record.get("score_values_checked") != banks * 3 * 32
                    or record.get("score_comparison_passed") is not True
                    or not 0 <= record.get("max_score_tolerance_ratio", float("inf")) <= 1
                    or record.get("frame_banks") != [BANKS_PER_FRAME] * frames
                    or record.get("frame_mismatched_banks") != [0] * frames
                    or record.get("q_dtype") != "torch.bfloat16"
                    or record.get("k_dtype") != "torch.bfloat16"
                    or record.get("k_shape") != [1, frames, 45, 80, 2, 128]):
                raise RuntimeError("expected-Fisher structure or fixed FP64 score gate failed")
        elif family == "anchor_gap_guard":
            from analysis.check_anchor_gap_guard_case import validate_guard_record
            validate_guard_record(record, banks, frames)
        elif family == "phase_averaged":
            from analysis.check_phase_averaged_case import validate_phase_record
            validate_phase_record(record, banks, frames)


def check_expected_case(torch, seed, frames, scale):
    from analysis.expected_fisher_seed import expected_fisher_indices, expected_fisher_gram
    from attention.adaptive_seed_pytorch import multiscale_candidate_table
    from attention.expected_fisher_seed_triton import fisher_expected_seed_rows_triton

    torch.manual_seed(seed)
    # Independent integers populate every element/head/frame, converted directly
    # to original BF16. No captured arrays, head replication or BF16->FP8 path.
    q = torch.randint(-4, 5, Q_SHAPE, device="cuda").to(torch.bfloat16)
    k_shape = (1, frames, 45, 80, 2, 128)
    k = torch.randint(-4, 5, k_shape, device="cuda").to(torch.bfloat16)
    table, anchors = multiscale_candidate_table(45, 32, q.device, geometry_count=45)
    kwargs = dict(candidate_count=32, candidate_geometry_count=45, block_j=1,
                  probe_count=16, global_probe_count=256, gs_num_warps=4)
    expected_shape = (1, 1, frames, 45, 80, 2, 4)
    native = fisher_expected_seed_rows_triton(q, k, scale, **kwargs)
    score_audit = torch.full((1, 1, frames, 45, 80, 2, 3, 32), float("nan"),
                             dtype=torch.float64, device=q.device)
    audited = fisher_expected_seed_rows_triton(q, k, scale, score_audit=score_audit, **kwargs)
    if (tuple(native.shape) != expected_shape or tuple(audited.shape) != expected_shape
            or native.dtype != torch.uint8 or audited.dtype != torch.uint8):
        raise AssertionError("wrong native row-table shape or dtype")
    native, audited = native.reshape(-1, 4), audited.reshape(-1, 4)
    audit_equal = (native == audited).all(-1)
    actual_scores = score_audit.reshape(-1, 3, 32)
    banks = frames * BANKS_PER_FRAME
    expected = torch.empty_like(native)
    invalid = anchor_errors = nonunique = 0
    max_error = max_ratio = 0.0
    score_passed = True
    for start in range(0, banks, 256):
        ids = torch.arange(start, min(start + 256, banks), device=q.device)
        frame, row, col, head = decode_bank_ids(ids)
        real = table[row].long()
        qq = q[0, 0, real, col[:, None], head[:, None]].float()
        kk = k[0, frame[:, None], row[:, None], torch.arange(0, 80, 5, device=q.device),
               head[:, None]].float()
        # Integer products/sums are exactly representable in FP32. Promote the
        # accumulated dot BEFORE applying either .125 or actual attention scale.
        z = torch.bmm(qq, kk.transpose(1, 2)).double() * scale
        slots = expected_fisher_indices(z, anchors[row])
        expected[start:start + len(ids)] = real.gather(1, slots).to(torch.uint8)
        actual = native[start:start + len(ids)].long()
        membership = real[:, :, None] == actual[:, None, :]
        invalid += int((membership.sum(1) != 1).any(-1).sum())
        anchor_errors += int((actual[:, 0] != row).sum())
        nonunique += int((actual.sort(dim=-1).values.diff(dim=-1) == 0).any(-1).sum())
        if invalid:
            raise AssertionError("native output includes rows outside the C32 candidate bank")
        actual_slots = membership.long().argmax(dim=1)
        reference_scores = torch.stack(gram_scores_at_native_prefix(
            expected_fisher_gram(z, anchors[row]), actual_slots), dim=1)
        observed = actual_scores[start:start + len(ids)]
        if not bool(torch.isfinite(observed).all()) or not bool(torch.isfinite(reference_scores).all()):
            raise AssertionError("nonfinite or missing native/independent FP64 scores")
        error = (observed - reference_scores).abs()
        ratio = error / (SCORE_ATOL + SCORE_RTOL * reference_scores.abs())
        max_error = max(max_error, float(error.max()))
        max_ratio = max(max_ratio, float(ratio.max()))
        score_passed = score_passed and bool((ratio <= 1).all())
    equal = (native == expected).all(-1)
    return {"input_seed": seed, "key_frames": frames, "scale": scale,
        "method": EXPECTED_METHOD, "banks": banks, "q_dtype": str(q.dtype), "k_dtype": str(k.dtype),
        "k_shape": list(k.shape), "group_agreement": float(equal.float().mean()),
        "mismatched_banks": int((~equal).sum()), "anchor_mismatches": anchor_errors,
        "nonunique_banks": nonunique, "invalid_candidate_banks": invalid,
        "frame_banks": [BANKS_PER_FRAME] * frames,
        "frame_mismatched_banks": [int((~part).sum()) for part in equal.reshape(frames, -1)],
        "score_audit_on_off_equal": bool(audit_equal.all()),
        "score_audit_mismatched_banks": int((~audit_equal).sum()),
        "score_atol": SCORE_ATOL, "score_rtol": SCORE_RTOL,
        "score_values_checked": banks * 3 * 32, "score_comparison_passed": score_passed,
        "max_score_abs_error": max_error, "max_score_tolerance_ratio": max_ratio,
        "native_sha256": hashlib.sha256(native.cpu().numpy().tobytes()).hexdigest()}


def worker(args):
    if os.environ.get("BARYCENTRIC_CHECK_PARENT") != str(os.getppid()) or os.getppid() <= 1:
        raise RuntimeError("require monitoring parent before importing GPU libraries")
    startup = json.loads((args.output / "startup.json").read_text())
    hashes = startup["source_sha256"]
    if startup.get("family") != args.family:
        raise RuntimeError("worker family differs from monitoring parent")
    verify_source_hashes(args.output, hashes)
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    results = {"complete":False, "family":args.family,
               "scope":"synthetic BF16 integer QK, not video or speed evidence",
               "shape":list(Q_SHAPE), "scale":0.125 if args.family == "barycentric" else None,
               "records":[], "source_sha256":hashes}
    if args.family in ("expected_fisher", "anchor_gap_guard", "phase_averaged"):
        case_function = check_expected_case
        if args.family == "anchor_gap_guard":
            from analysis.check_anchor_gap_guard_case import check_guard_case
            case_function = check_guard_case
        elif args.family == "phase_averaged":
            from analysis.check_phase_averaged_case import check_phase_case
            case_function = check_phase_case
        with torch.inference_mode():
            for seed, frames, scale in family_cases(args.family):
                record = case_function(torch, seed, frames, scale)
                results["records"].append(record)
                write(args.output / "results.json", results)
                if args.family == "anchor_gap_guard":
                    from analysis.check_anchor_gap_guard_case import validate_guard_record
                    validate_guard_record(record, record["banks"], frames)
                elif args.family == "phase_averaged":
                    from analysis.check_phase_averaged_case import validate_phase_record
                    validate_phase_record(record, record["banks"], frames)
                    if record["mismatched_banks"]:
                        raise AssertionError(f"phase_averaged native row gate failed: {record}")
                    print(f"input_seed={seed}, key_frames={frames}: {record['banks']} exact row banks; "
                          "transform/input/FP32 forward-bound gates and FP32 audit on/off agree; "
                          "no score-value comparison", flush=True)
                    continue
                if (record["mismatched_banks"] or record["anchor_mismatches"]
                        or record["nonunique_banks"] or not record["score_audit_on_off_equal"]
                        or not record["score_comparison_passed"]):
                    raise AssertionError(f"{args.family} native gate failed: {record}")
                print(f"input_seed={seed}, key_frames={frames}: {record['banks']} exact row banks; "
                      "all fixed-tolerance FP64 scores and audit on/off agree", flush=True)
        verify_source_hashes(args.output, hashes)
        results["source_sha256_after"] = source_hashes()
        results["complete"] = True
        validate_completed_result(results, args.family, hashes)
        write(args.output / "results.json", results)
        return
    from analysis.mean_seed_pytorch import mean_seed_indices
    from analysis.barycentric_medoid_seed import medoid_indices
    from attention.adaptive_seed_pytorch import multiscale_candidate_table
    from attention.affine_hellinger_seed_triton import (
        fisher_mean_seed_rows_triton, fisher_mean_anchor_seed_rows_triton,
        fisher_barycentric_medoid_seed_rows_triton,
    )
    methods = {
        "mean_fisher32": fisher_mean_seed_rows_triton,
        "mean_anchor_fisher32": fisher_mean_anchor_seed_rows_triton,
        "barycentric_medoid32": fisher_barycentric_medoid_seed_rows_triton,
    }
    with torch.inference_mode():
        for seed, frames, scale in family_cases(args.family):
            torch.manual_seed(seed)
            q = torch.randint(-4,5,(1,1,45,80,2,128),device="cuda").to(torch.bfloat16)
            k = torch.randint(-4,5,q.shape,device="cuda").to(torch.bfloat16)
            means = q.mean(2,dtype=torch.float32).to(torch.bfloat16).contiguous()
            torch.testing.assert_close(means, q.float().mean(2).to(q.dtype), rtol=0, atol=0)
            table, anchors = multiscale_candidate_table(45,32,q.device,geometry_count=45)
            native = {name:fun(q,k,0.125).reshape(-1,4) for name,fun in methods.items()}
            expected = {name:torch.empty_like(value) for name,value in native.items()}
            for start in range(0,45*80*2,256):
                ids=torch.arange(start,min(start+256,45*80*2),device=q.device)
                head,col,row=ids%2,(ids//2)%80,ids//160
                real=table[row].long()
                qq=q[0,0,real,col[:,None],head[:,None]].float()
                kk=k[0,0,row[:,None],torch.arange(0,80,5,device=q.device),head[:,None]].float()
                z=torch.bmm(qq,kk.transpose(1,2)).double()*0.125
                mq=means[0,0,col,head].float()[:,None]
                mz=torch.bmm(mq,kk.transpose(1,2)).double()[:,0]*0.125
                for name in methods:
                    if name == "barycentric_medoid32":
                        slots=medoid_indices(z)
                        rows=real.gather(1,slots)
                    else:
                        slots=mean_seed_indices(z,mz,anchors[row],name)
                        rows=torch.cat((row[:,None],real.gather(1,slots)),dim=-1)
                    expected[name][start:start+len(ids)]=rows.to(torch.uint8)
            for name in methods:
                actual=native[name]
                equal=(actual==expected[name]).all(-1)
                record={"input_seed":seed,"key_frames":frames,"scale":scale,"method":name,"banks":len(actual),
                    "group_agreement":float(equal.float().mean()),
                    "mismatched_banks":int((~equal).sum()),
                    "native_sha256":hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest()}
                results["records"].append(record)
                write(args.output/"results.json",results)
                if not bool(equal.all()):
                    raise AssertionError(f"exact synthetic native parity failed: {record}")
            print(f"input_seed={seed}: all three native tables equal independent oracle",flush=True)
    verify_source_hashes(args.output, hashes)
    results["source_sha256_after"] = source_hashes()
    results["complete"]=True
    validate_completed_result(results, args.family, hashes)
    write(args.output/"results.json",results)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--family",choices=FAMILIES,default="barycentric")
    parser.add_argument("--gpu",type=int,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.worker:
        return worker(args)
    from analysis import launch_quality_first_videos as guard
    initial=guard.require_clean(args.gpu)
    if args.output.exists():
        raise ValueError("fresh output required")
    hashes=source_hashes()
    args.output.mkdir(parents=True)
    freeze_sources(args.output, hashes)
    write(args.output/"startup.json",{"initial":initial,"family":args.family,"source_sha256":hashes,
        "expected_records":len(expected_records(args.family)),
        "expected_banks":sum(spec[-1] for spec in expected_records(args.family))})
    env=guard.child_environment(args.output,initial["gpu"]["uuid"])
    env["BARYCENTRIC_CHECK_PARENT"]=str(os.getpid())
    process=None; failure=None; foreign=[]
    def interrupted(*_):
        raise KeyboardInterrupt("parent interrupted")
    signal.signal(signal.SIGTERM,interrupted)
    try:
        guard.require_clean(args.gpu,expected_uuid=initial["gpu"]["uuid"])
        verify_source_hashes(args.output, hashes)
        with (args.output/"worker.log").open("x") as log, (args.output/"gpu_monitor.jsonl").open("x") as monitor:
            process=subprocess.Popen([sys.executable,"-u",str(Path(__file__).resolve()),"--worker",*sys.argv[1:]],
                env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            write(args.output/"process.json",{"parent_pid":os.getpid(),"worker_pid":process.pid})
            while True:
                active=guard.gpu_processes(initial["gpu"]["uuid"])
                foreign=[item for item in active if item["pid"]!=process.pid]
                monitor.write(json.dumps({"utc":guard.now(),"processes":active,"foreign":foreign})+"\n")
                monitor.flush()
                if foreign: raise RuntimeError("foreign GPU process")
                if process.poll() is not None: break
                time.sleep(5)
            if process.returncode!=0: raise RuntimeError(f"worker exit {process.returncode}")
            verify_source_hashes(args.output, hashes)
            result=json.loads((args.output/"results.json").read_text())
            validate_completed_result(result, args.family, hashes)
    except BaseException as error:
        failure=f"{type(error).__name__}: {error}"
    finally:
        guard.stop_own_child(process)
    write(args.output/"completion.json",{"valid":failure is None,"failure":failure,"foreign":foreign,
        "family":args.family,"expected_records":len(expected_records(args.family)),
        "source_sha256":hashes,
        "worker_exit_code":process.returncode if process else None,"completed_utc":guard.now(),
        "mean_attention_ms":None,"speedup_vs_sdpa":None})
    if failure: raise RuntimeError(failure)


if __name__=="__main__": main()
