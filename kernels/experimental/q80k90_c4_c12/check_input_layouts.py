"""Validate strided input packing and time only that stage on an idle GPU.

This does not execute the Hopper merge or infer full-attention speedups.
Use run_exclusive.py for preflight and continuous contention monitoring.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

import torch

from q80k90_kernel import _pack_inputs_into


def make_inputs(t, h, layout):
    shape = (1, t * 3600, h, 128)
    tensors = []
    for i in range(3):
        kind = ("token", "head", "padded")[i] if layout == "mixed" else layout
        if kind == "head":
            tensor = torch.empty((1, h, t * 3600, 128), device="cuda",
                                 dtype=torch.bfloat16).transpose(1, 2)
        elif kind == "padded":
            storage = torch.empty((1, t * 3600 + 1, h + 1, 256),
                                  device="cuda", dtype=torch.bfloat16)
            tensor = storage[:, 1:, :h, 1::2]
        else:
            tensor = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
        tensor.copy_(torch.randint(-128, 128, shape, device="cuda",
                                   dtype=torch.int16))
        tensors.append(tensor)
    return tuple(tensors)


def buffers(t, h):
    return [torch.empty((h, t * 3600, 128), device="cuda", dtype=torch.bfloat16)
            for _ in range(4)]


def check(q, k, v, outputs, t, h):
    qp, kp, vp, packed = outputs
    assert torch.equal(qp, q[0].permute(1, 0, 2)), "query layout"
    # The merge's K blocks pair adjacent W positions across all 45 H rows.
    indices = torch.arange(t * 3600, device="cuda")
    frame, within = indices // 3600, indices % 3600
    source = (frame * 45 + within % 45) * 80 + (within // 90) * 2 + (within % 90) // 45
    for target, tensor in ((kp, k), (vp, v)):
        assert torch.equal(target, tensor[0].index_select(0, source).permute(1, 0, 2)), "KV layout"
    # Native query packing transposes a 4x4 matrix of BF16 pairs within each
    # 32-element slice. This reshape/transpose oracle is independent of the
    # Triton address calculation.
    expected = qp.reshape(h, t * 3600, 4, 4, 4, 2).transpose(-3, -2).reshape_as(qp)
    assert torch.equal(packed, expected), "native query packing"


def measure(fn, repeats=12):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    elapsed = []
    for _ in range(repeats):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        elapsed.append(begin.elapsed_time(end))
    return dict(mean_ms=statistics.mean(elapsed), median_ms=statistics.median(elapsed),
                samples_ms=elapsed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-shape", action="store_true")
    parser.add_argument("--reference-kernel", type=Path,
                        help="Optional unmodified v4 q80k90_kernel.py for timing the original packer")
    args = parser.parse_args()
    reference = None
    if args.reference_kernel:
        spec = importlib.util.spec_from_file_location("q80k90_layout_reference", args.reference_kernel)
        reference = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = reference
        spec.loader.exec_module(reference)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    shapes = [(1, 1), (1, 3), (2, 8)]
    if args.full_shape:
        shapes.append((21, 40))
    result = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  cuda=torch.version.cuda, scope="input packing only",
                  source_sha256=hashlib.sha256(Path(__file__).with_name("q80k90_kernel.py").read_bytes()).hexdigest(),
                  reference_sha256=(hashlib.sha256(args.reference_kernel.read_bytes()).hexdigest()
                                    if args.reference_kernel else None),
                  checks=[], timing={})
    for t, h in shapes:
        for layout in ("token", "head", "mixed", "padded"):
            inputs = make_inputs(t, h, layout)
            outputs = buffers(t, h)
            _pack_inputs_into(*inputs, *outputs, t, h)
            check(*inputs, outputs, t, h)
            result["checks"].append(dict(t=t, heads=h, layout=layout,
                                         strides=[list(x.stride()) for x in inputs], exact=True))
            if (t, h) == (21, 40) and layout in ("token", "head"):
                # Same values, output buffers, GPU and timing method. An
                # optional frozen source also checks the original packer.
                def direct():
                    _pack_inputs_into(*inputs, *outputs, t, h)

                def materialized():
                    contiguous = [x.contiguous() for x in inputs]
                    if reference is None:
                        _pack_inputs_into(*contiguous, *outputs, t, h)
                    else:
                        reference._pack_inputs[((inputs[0].numel()+1023)//1024,)](
                            *contiguous, *outputs, t, h, 1024)

                # Reverse order in the second round to expose drift.
                first = dict(materialized=measure(materialized), direct=measure(direct))
                second = dict(direct=measure(direct), materialized=measure(materialized))
                materialized()
                check(*inputs, outputs, t, h)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                initial = torch.cuda.memory_allocated()
                materialized()
                torch.cuda.synchronize()
                old_extra = torch.cuda.max_memory_allocated() - initial
                torch.cuda.reset_peak_memory_stats()
                initial = torch.cuda.memory_allocated()
                direct()
                torch.cuda.synchronize()
                new_extra = torch.cuda.max_memory_allocated() - initial
                result["timing"][layout] = dict(rounds=[first, second],
                    materialized_extra_bytes=old_extra, direct_extra_bytes=new_extra)
            del inputs, outputs
            gc.collect()
            torch.cuda.empty_cache()
    result["passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(passed=True, checks=len(result["checks"]), gpu=result["gpu"],
                         output=str(args.output))))


if __name__ == "__main__":
    with torch.inference_mode():
        main()
