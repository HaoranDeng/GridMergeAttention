"""CPU-only Triton front-end check for the native_fisher/route_c4_tma*.py variants.

Runs only the AST -> TTIR stage (Triton's code generator and semantic checks,
including tl.make_tensor_descriptor / desc.load validation) for
_hard_c4_720p_route_kernel with the operator's exact constexpr set on an
sm_90a target. No device, no optimisation pass, no ptxas, no cubin, no GPU.

Usage, from the repo root, with CUDA_VISIBLE_DEVICES= (empty):
    python native_fisher/route_c4_tma_frontend_check.py native_fisher/route_c4_tma.py [more paths]
"""
import sys

# native_fisher/operator.py shadows the stdlib module; drop the script directory
# that "python native_fisher/...py" prepends before anything else is imported.
sys.path[:] = [p for p in sys.path if not p.replace("\\", "/").rstrip("/").endswith("native_fisher")]

import importlib.util  # noqa: E402
import os  # noqa: E402

POINTERS = {
    "q_ptr": "*bf16", "k_ptr": "*bf16", "v_ptr": "*bf16",
    "merged_k_ptr": "*bf16", "merged_v_ptr": "*bf16",
    "merged_k_scale_ptr": "*fp32", "merged_v_scale_ptr": "*fp32",
    "bias_ptr": "*fp32", "prototype_distance_ptr": "*u8",
    "distortion_score_ptr": "*fp32", "out_ptr": "*bf16",
}
# kernels/best/c4/operator.py route() / analysis/screen_c4_route_hints.py route_with, head chunk 0.
CONSTEXPRS = dict(
    Q_OUTER=21, K_OUTER=21, TOTAL_HEADS=40, ACTIVE_HEADS=5, HEAD_START=0, D_HEAD=128,
    HEAD_MAJOR=False, OUTPUT_HEAD_MAJOR=False, ROWS=45, COLS=80, C=4, BLOCK_I=64,
    BLOCK_TOKENS=64, LOOP_STAGES=3, INPUT_PRECISION="ieee", EXACT_INTRA_FRAME=False,
    EXACT_NEIGHBOR_WINDOW_COLS=0, RECORD_DISTORTION=False, QUERY_OUTER_OFFSET=0,
    SINGLETON_TOP_R=1, SINGLETON_QUERY_COLUMN=True, SINGLETON_SECONDARY_FRAME_STRIDE=1,
    SINGLETON_PRIMARY_ROW_MOD4_OMIT2=False, QUERY_COLUMN_MAJOR=True,
    PACKED_SINGLETON_KV=False, SCALED_FP8_KV=False, COMPRESSED_FP8=False,
    SCALE=128 ** -0.5,
)
OPS = ["tt.make_tensor_descriptor", "tt.descriptor_load", "tt.load", "tt.dot", "scf.for",
       "tt.num_stages", "tt.store", "tt.trans"]


def load(path, index):
    spec = importlib.util.spec_from_file_location("route_c4_frontend_target_%d" % index, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frontend(fn):
    from triton.compiler.compiler import ASTSource, make_backend
    from triton.backends.compiler import GPUTarget
    from triton._C.libtriton import ir

    target = GPUTarget("cuda", 90, 32)
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": 4, "num_stages": 1})
    names = list(fn.arg_names)
    assert set(names) == set(POINTERS) | set(CONSTEXPRS), set(names) ^ (set(POINTERS) | set(CONSTEXPRS))
    signature = {n: (POINTERS[n] if n in POINTERS else "constexpr") for n in names}
    src = ASTSource(fn, signature, constexprs=CONSTEXPRS)
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()
    module = src.make_ir(target, options, codegen_fns, module_map, context)
    return str(module)


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "run with CUDA_VISIBLE_DEVICES= (empty)"
    for index, path in enumerate(sys.argv[1:]):
        module = load(path, index)
        fn = module._hard_c4_720p_route_kernel
        print("==", path, "kernel type:", type(fn).__name__,
              "constants:", module.HARD_720_ROUTE_BLOCK_I, module.HARD_720_INPUT_PRECISION,
              module.HARD_720_ROUTE_NUM_STAGES, "prepare_tma:", hasattr(module, "prepare_tma"))
        text = frontend(fn)
        print("   TTIR chars", len(text), {op: text.count(op) for op in OPS})
        for line in text.splitlines():
            s = line.strip()
            if "make_tensor_descriptor" in s or "descriptor_load" in s or "tt.num_stages" in s:
                print("   ", s[:230])
        if "--dump" in os.environ.get("ROUTE_C4_FRONTEND", ""):
            print(text)
    print("FRONTEND_OK")


if __name__ == "__main__":
    main()
