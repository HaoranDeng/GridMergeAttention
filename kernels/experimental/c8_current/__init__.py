"""Opt-in Q45K80 exact C8 with current-frame selection (Hopper)."""


def create(device='cuda', heads=40):
    from .attention import CurrentFrameC8
    return CurrentFrameC8(device=device, heads=heads)


__all__ = ['create']
