"""GMA-C8 r2: exactly eight merged K/V, with no singleton route.

The package deliberately exposes one entry point and one CUDA merge kernel.
Historical alternatives remain in ``evidence.json`` for auditability, but
cannot be selected or compiled through this runtime package.
"""


def create(device="cuda"):
    """Create the screened numerics implementation with serial merge/route.

    The merge uses approximate base-2 exponentials, flat warp reductions under
    a guarded-softmax rescale, and a fused 32-bit packed argmax. The selector,
    probe, route, C=8 clustering, C32/CG45 geometry and selected seeds are
    unchanged from the retained C8 reference.
    """
    from .._integrity import verify

    verify(__file__)
    from .gma_c8_r2 import NativeC8R2

    return NativeC8R2(device=device)


__all__ = ["create"]
