# Actual-tangent coverage for seed queries

Exploratory proposal, 2026-09-05. RF0, C4, real matched anchor, C32/g45,
Uniform16, BF16 merge and full compressed route remain fixed. No paper or
production default change. No trained or calibrated parameter.

## Motivation

Fisher-GS covers a linear span. The actual merge instead evaluates real
seed-conditioned tangents, not arbitrary linear combinations in that span.
On the local probe support, let F(z)=logsumexp(z), p_s=softmax(z_s), and

    T_s(z_i) = p_s^T z_i + H(p_s).
    d(i,s) = F(z_i)-T_s(z_i) = KL(p_s || p_i).
    A(i,s) = exp(-d(i,s)).

Each A(i,s) is the fraction of target i's partition sum represented by that
whole-block seed tangent. Fix the matched anchor and greedily add three real
queries using one of these prespecified rules:

- tangent_farthest: maximize min_{s in S} d(i,s).
- tangent_coverage: maximize sum_i max(A(i,s)-max_{u in S} A(i,u),0).
- fisher_nearest: farthest-point selection in the existing shared geometric
  Fisher metric, replacing GS span projection by nearest selected seed.

The primary proposal is tangent_coverage. Its set objective
sum_i max_{s in S} A(i,s) is monotone and has diminishing marginal gains,
since adding a seed can only raise each maximum and an existing higher
maximum leaves less room to improve. This statement concerns this surrogate,
not the final hard-partition variational objective or generated video quality.
The candidate matrix is only R by R (R=32) and uses existing local logits;
no global mass sketch or trial hard repartitioning is needed.

This differs from earlier cosine facility-location, raw-logit/key coverage,
and hard-partition deficit/lookahead selectors. It uses the exact directed
reverse-KL between whole-probe distributions, before any hard assignment.
After selection, the unchanged merge conditions each seed on its owned cell,
so the monotonicity of tangent coverage does not transfer to the GCA operator.
Uniform16 sampling, RF0 temporal reuse, and finite precision remain separate
approximations. In particular, farthest selection may overemphasize outliers.

## Verification and decision protocol

The independent scalar tests verify KL orientation, tangent fractions,
Fisher distances, shift invariance, greedy pivots, and distinct deterministic
tie filling. Screen all three rules on the existing three L33 captures,
all 21 query frames and heads 0/10/20/30, using complete BF16 GCA output.
Do not reject a video candidate based on a three-video prefix: use all 20
canonical videos for any final PSNR conclusion. Fixed-QKV MSE is only a
screen. Quality comes before implementation speed tuning; the final target
still requires PSNR above 12 and no slower complete attention than Hellinger.

## Results

Run tangent_coverage_bf16_screen_20260905x_retry1 completed cleanly.
Pooled complete-output MSE was 0.001554636055 for tangent_farthest,
0.001566768618 for tangent_coverage and 0.001577711612 for fisher_nearest.
All are worse than the matched Hellinger RF0 reference 0.001298445288 and
native Fisher32 0.001257260255. None advances to videos. Runtime, PSNR,
SSIM and LPIPS were not measured. The initial x attempt exited before
GPU work because its capture path omitted the captures subdirectory; its
failed completion record is retained, not treated as a numerical result.
