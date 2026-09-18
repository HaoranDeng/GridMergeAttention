"""C10 with E4M3 compressed K/V and the retained local-score merge."""


def create(device="cuda"):
    from ..._integrity import verify
    verify(__file__)
    from .c10_fp8_candidate import make_candidate
    return make_candidate(device=device)


__all__ = ["create"]
