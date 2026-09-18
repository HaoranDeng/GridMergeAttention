"""Exact-capacity C4 CUDA merge and cooperative Uniform16 Fisher selector.

The current operator emits exactly four merged K/V representatives per key
block. SQ1/singleton/C+1 routing is forbidden. Historical timings below were
measured before this semantic correction and must not be attributed to the
current exact-capacity operator until it is rebenchmarked.

Promoted 2026-09-14. The merge is `fisher_c4_desc_select.cu`: the retained
guarded-softmax merge with loop-invariant WGMMA descriptors derived from three
per-CTA bases and hoisted self-column tests. Output is bitwise equal to the
previous merge on layers 0/19/39. The selector is
`fisher_c4_selector_fp32gs_regpipe.cu`: FP32 Fisher Gram-Schmidt with a
register-pipelined frame loop. This is a declared numerics change - seeds
differ from the FP64 selector on at most 4.3e-4 of positions (layer 39; 1e-6
on layers 0/19), output relative L2 under 5e-5 - accepted by the user on the
condition of a 20-video PSNR screen, which showed +0.03 dB against the dense
reference (no difference). Complete paired calls: 100.70 ms against 104.36 ms
for the previous pair in the same window, -3.66 ms, 87 of 90 paired wins.
Earlier selector sources remain as audit controls; the retired SQ1 merge
source was deleted.

Promoted 2026-09-15 (round 9): the selector is now fisher_c4_selector_v8_ldsm_packed.cu, an exact rewrite of the FP32 GS register-pipelined selector: padded 272-byte shared rows with 24 ldmatrix.x4 fragment loads in place of 96 LDS.32 and their address arithmetic, a branchless local argmax with REDUX cross-lane argmax, hoisted anchor and candidate row, a halved accumulator transpose and one packed 32-bit seed store per frame. Seeds and complete output are bitwise equal to fisher_c4_selector_fp32gs_regpipe.cu on layers 0/19/39 and on the tie edge cases. Registers 124 -> 89, 4 -> 5 CTAs per SM. Selector stage 7.25 -> 5.23 ms (-2.02 ms per call, 90 of 90 paired wins); complete call -1.27 ms (100.29 -> 98.95 ms in the same window, 80 of 90 paired wins). The previous selector stays in this directory as the paired control.

Promoted 2026-09-15 (round 9, third selector step): fisher_c4_selector_v9c_halving_mxredux.cu replaces fisher_c4_selector_v8_ldsm_packed.cu. Exact rewrite of the Gram-Schmidt reductions: recursive-halving reductions for the per-step candidate scores (each probe lane ends with one candidate's score, so the local argmax collapses onto the REDUX cross-lane argmax) and for the 32-candidate centers (probe subset relocated to lane bits 4,3), with the center max taken by REDUX on order-preserving integer keys. Every added FP32 add pairs the same operands as the butterfly it replaces; seeds and complete output are bitwise equal to the previous selector on layers 0/19/39 and on the tie edge cases. SHFL.BFLY 66 -> 41 per frame, registers 89 -> 80, still 5 CTAs per SM. Selector stage 5.26 -> 4.88 ms (-0.374 ms per call, 90 of 90 paired wins, t -19 to -27); complete call -0.29 ms (98.49 -> 98.18 ms in the same window, 55 of 90 - at the complete-call instrument's resolution floor, kept as a no-regression check). The previous selector stays in this directory as the paired control.
"""
from pathlib import Path
from .operator import NativeFisherAttention

class OptimizedFisherC4Attention(NativeFisherAttention):
    def __init__(self,device='cuda'):
        super().__init__(device=device,capacity=4,selector_backend='fused_cooperative_loop',
                         selector_source=Path(__file__).with_name('fisher_c4_selector_v9c_halving_mxredux.cu'),
                         merge_source=Path(__file__).with_name('fisher_c4_desc_select.cu'))
