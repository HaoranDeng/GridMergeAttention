"""Current experimental C10 FP8 candidate with local-score reuse."""
from pathlib import Path
from .fp8_c10 import retained_control,FP8C10Attention

def make_candidate(device='cuda'):
    """Native BF16 QKV, direct E4M3 intermediates, complete RF0/SQ1 route.

    Verified kernel timing: about 246-248 ms on the canonical H200 H40 captures.
    Complete output bytes match the previous FP8 candidate on three captures;
    the 20-pair generation evaluation belongs to that previous implementation.
    This is an explicit experiment entry point, not a production default.
    One instance owns reusable buffers and supports serial calls only.
    """
    op=FP8C10Attention(retained_control(device),'direct',False,
                      Path(__file__).with_name('fisher_c10_fp8_local_scores.cu'))
    op.via_float_route=True
    op.route_block=64;op.route_stages=1;op.route_warps=4
    return op
