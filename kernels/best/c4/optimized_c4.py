"""Production exact-capacity C4 operator.

The operator emits exactly four merged K/V representatives per key block.
SQ1, singleton, raw-K/V, residual, and C+1 routing are forbidden.
"""
from pathlib import Path

from .operator import NativeFisherAttention


class OptimizedFisherC4Attention(NativeFisherAttention):
    def __init__(self, device="cuda"):
        here = Path(__file__).resolve().parent
        super().__init__(
            device=device,
            capacity=4,
            selector_backend="fused_cooperative_loop",
            selector_source=here / "fisher_c4_selector_v9c_halving_mxredux.cu",
            merge_source=here / "fisher_c4_desc_select.cu",
        )
