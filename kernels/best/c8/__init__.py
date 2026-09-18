"""Default C8 dispatches to r2; retained sources implement slow_reference."""

from ..c8_r2 import create


def create_slow_reference(device="cuda"):
    """Construct the pre-r2 C8 reference explicitly, with source verification."""
    from .._integrity import verify
    verify(__file__)
    from .gma_c8_native import NativeC8
    return NativeC8(device=device)


__all__ = ["create", "create_slow_reference"]
