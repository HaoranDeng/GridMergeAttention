"""BF16 Fisher C4 with exactly four merged K/V representatives."""


def create(device="cuda"):
    from .._integrity import verify
    verify(__file__)
    from .optimized_c4 import OptimizedFisherC4Attention
    return OptimizedFisherC4Attention(device=device)


__all__ = ["create"]
