"""Production GMA-C8 r2: exactly eight merged K/V, no singleton route."""


def create(device="cuda"):
    from .._integrity import verify

    verify(__file__)
    from .gma_c8_r2 import NativeC8R2

    return NativeC8R2(device=device)


__all__ = ["create"]
