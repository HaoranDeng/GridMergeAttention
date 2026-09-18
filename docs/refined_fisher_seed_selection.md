# Bounded refinements of anchored Fisher-GS

2026-09-05; RF0, C32/g45, Uniform16, four real seeds and unchanged BF16
merge/SQ1/full route. No paper/default change or fitted parameter.

Fisher-GS greedily maximizes residual norms, hence builds a large volume
in the local entropy curvature geometry. Earlier pivots are not revisited.
The first proposal, fisher_maxvol1, makes one coordinate-exchange pass
over the three non-anchor seeds. Holding the other two directions fixed,
maximizing squared residual maximizes their three-direction Gram determinant
in nondegenerate exact arithmetic. This optimizes the volume surrogate,
not the hard-partition objective or video quality. Numerical zero norms use
the existing 1e-12 safeguard; ties retain the current seed.

A second, separate heuristic fisher_antipodal notes that opposite directions
are redundant for a linear span but can induce different hard key ownership.
After the first norm pivot, choose the most negatively aligned real row,
if one exists; otherwise keep the usual GS pivot. Then select the final
ordinary residual pivot. This is not a claimed optimum.

Both proposals retain the real matched anchor. Neither performs hard
partition lookahead, global mass estimation, query averaging, or extra
attention paths. Screen on the same three all-frame L33 QKV captures before
considering a complete20-video run.

Run refined_fisher_bf16_screen_20260905z completed cleanly. Max-volume
exchange gave pooled MSE 0.001434626265; antipodal selection 0.001342134317.
Neither beats Hellinger RF0 (0.001298445288) or Fisher32 (0.001257260255).
Neither advances to videos. Quality/timing beyond QKV MSE is unmeasured.
This was the initial single-GPU screen decision. Under the later expanded
GPU allowance, only antipodal is reopened for a bounded complete20-video
test after native verification; see fisher_seed_selection.md. Max-volume
exchange remains rejected. No new video outcome is implied.

## Signed reverse-KL features

The next bounded comparison keeps the original GS rule but replaces its
quadratic features with a nonlinear reverse-KL embedding:

    b_ij = sign(log p_ij - log rho_j)
           * sqrt(2 * (rho_j log(rho_j/p_ij) - rho_j + p_ij)).

Each expression in parentheses is a nonnegative scalar generalized KL.
The squared norm of b_i is exactly 2 KL(rho || p_i). Near rho it reduces
to the same Fisher feature, but it does not replace large excursions with
a quadratic. It is distinct from the rejected signed Jeffreys feature,
whose norm is the sum of both KL orientations.

kl_geometric_gs uses the paper's geometric barycenter rho and subtracts
b_anchor before GS. kl_anchor_gs sets rho=p_anchor, so b_anchor=0.
The exact norm identity is for b_i before translation/projection; it
does not establish exact pairwise KL distances or hard-partition bounds.
Both retain the original four real seeds, C32, RF0 and merge/route.

Run kl_gs_bf16_screen_20260905aa completed cleanly. Geometric KL-GS gave
pooled MSE 0.001632601567 and anchor KL-GS 0.001550669253. Both improve the
RF0-only slice but worsen the remaining20 frames and overall output.
Neither advances to videos. These results do not support a video claim.
