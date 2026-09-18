"""BF16 Fisher C10: retained long-tiles merge and register-distributed GS."""
from pathlib import Path


def create(device="cuda"):
    from .._integrity import verify
    verify(__file__)
    from .operator import NativeFisherAttention
    here = Path(__file__).resolve().parent
    return NativeFisherAttention(
        device=device, capacity=10, tiles=84,
        route_block=64, route_warps=4, route_stages=3,
        selector_backend="probe4", selector_block=2,
        merge_source=here / "fisher_c10_long_tiles_tree.cu",
        selector_source=here / "fisher_gs_register_distribute.cu",
    )


__all__ = ["create"]
