"""BF16 native Hellinger C8: exactly eight merged K/V, no singleton."""


def create(device="cuda"):
    from .._integrity import verify
    verify(__file__)
    from .gma_c8_native import NativeC8
    return NativeC8(device=device)


__all__ = ["create"]
