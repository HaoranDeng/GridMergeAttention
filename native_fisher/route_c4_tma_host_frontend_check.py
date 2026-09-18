"""CPU-only Triton front-end check for native_fisher/route_c4_tma_host.py.

Same AST -> TTIR stage as route_c4_tma_frontend_check.py (Triton's code
generator and semantic checks for _hard_c4_720p_route_kernel with the
operator's exact constexpr set on an sm_90a target; no device, no optimisation
pass, no ptxas, no cubin, no GPU), except that the two compressed-stream
arguments merged_k_ptr / merged_v_ptr are typed as host TMA descriptors. Their
signature string is not hand-written: it is produced by the same
triton.runtime.jit.mangle_type (-> libtriton native_specialize_impl) the JIT
applies at launch, run on a CPU bf16 tensor wrapped exactly as
make_descriptors() wraps the operator workspace, and asserted to be
'tensordesc<bf16[64, 128]>'.

Usage, from the repo root, with CUDA_VISIBLE_DEVICES= (empty):
    python native_fisher/route_c4_tma_host_frontend_check.py native_fisher/route_c4_tma_host.py
"""
import sys

# native_fisher/operator.py shadows the stdlib module; drop the script directory
# that "python native_fisher/...py" prepends before anything else is imported.
sys.path[:] = [p for p in sys.path if not p.replace("\\", "/").rstrip("/").endswith("native_fisher")]

import importlib.util  # noqa: E402
import os  # noqa: E402

POINTERS = {
    "q_ptr": "*bf16", "k_ptr": "*bf16", "v_ptr": "*bf16",
    "merged_k_scale_ptr": "*fp32", "merged_v_scale_ptr": "*fp32",
    "bias_ptr": "*fp32", "prototype_distance_ptr": "*u8",
    "distortion_score_ptr": "*fp32", "out_ptr": "*bf16",
}
DESCRIPTORS = ("merged_k_ptr", "merged_v_ptr")
EXPECTED_DESCRIPTOR_TYPE = "tensordesc<bf16[64, 128]>"
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
       "tt.num_stages", "tt.store", "tt.trans", "tensordesc<"]


def load(path, index):
    spec = importlib.util.spec_from_file_location("route_c4_host_frontend_target_%d" % index, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def descriptor_type(module):
    """The tensordesc<...> string the JIT would mangle make_descriptors' output to.

    Built on a small CPU bf16 tensor with the same trailing dim, dtype, view and
    block shape as the operator workspace; TensorDescriptor.from_tensor and
    mangle_type never touch a CUDA device for a CPU tensor.
    """
    import torch
    from triton.runtime.jit import mangle_type

    cpu_mk = torch.empty((4, 3780, 128), dtype=torch.bfloat16)
    cpu_mv = torch.empty_like(cpu_mk)
    k_desc, v_desc = module.make_descriptors(cpu_mk, cpu_mv)
    assert tuple(k_desc.shape) == (4 * 3780, 128) and tuple(k_desc.strides) == (128, 1), k_desc
    assert list(k_desc.block_shape) == [CONSTEXPRS["BLOCK_TOKENS"], CONSTEXPRS["D_HEAD"]], k_desc.block_shape
    types = (mangle_type(k_desc), mangle_type(v_desc), mangle_type(k_desc, specialize=True))
    assert set(types) == {EXPECTED_DESCRIPTOR_TYPE}, types
    assert not torch.cuda.is_initialized()
    return types[0]


def frontend(fn, desc_type):
    from triton.compiler.compiler import ASTSource, make_backend
    from triton.backends.compiler import GPUTarget
    from triton._C.libtriton import ir

    target = GPUTarget("cuda", 90, 32)
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": 4, "num_stages": 1})
    names = list(fn.arg_names)
    expected = set(POINTERS) | set(DESCRIPTORS) | set(CONSTEXPRS)
    assert set(names) == expected, set(names) ^ expected
    signature = {}
    for n in names:
        if n in DESCRIPTORS:
            signature[n] = desc_type
        elif n in POINTERS:
            signature[n] = POINTERS[n]
        else:
            signature[n] = "constexpr"
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
              "constants:", module.HARD_720_ROUTE_BLOCK_I, module.HARD_720_ROUTE_BLOCK_TOKENS,
              module.HARD_720_INPUT_PRECISION, module.HARD_720_ROUTE_NUM_STAGES,
              "prepare_tma:", hasattr(module, "prepare_tma"), "->", module.prepare_tma(),
              "make_descriptors:", hasattr(module, "make_descriptors"))
        desc_type = descriptor_type(module)
        print("   descriptor arg type from mangle_type:", repr(desc_type))
        text = frontend(fn, desc_type)
        counts = {op: text.count(op) for op in OPS}
        print("   TTIR chars", len(text), counts)
        assert counts["tt.make_tensor_descriptor"] == 0, "host variant must not build descriptors in-kernel"
        assert counts["tt.descriptor_load"] == 2, counts
        for line in text.splitlines():
            s = line.strip()
            if "make_tensor_descriptor" in s or "descriptor_load" in s or "tt.num_stages" in s \
                    or ("tt.func" in s and "tensordesc" in s):
                print("   ", s[:300])
        if "--dump" in os.environ.get("ROUTE_C4_FRONTEND", ""):
            print(text)
    print("FRONTEND_OK")


if __name__ == "__main__":
    main()
