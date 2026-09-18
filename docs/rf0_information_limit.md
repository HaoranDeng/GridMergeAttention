# What strict RF0 selection can and cannot observe

This is a mathematical limitation, not an impossibility claim for PSNR12.
No candidate using other query frames is implemented or authorized here.

With shared row indices S, the actual entropy-regularized compression gap
is averaged over every query frame. Each frame loads its own Q at S,
constructs its own ownership and conditional distributions, then routes
all its queries through those representatives. Scoring only RF0 does not
directly observe those later conditional distributions.

A toy example rules out a universal optimality guarantee for RF0-only
selection. Let every RF0 candidate Q be zero, with more rows than seeds.
Use scalar bulk keys -1 and +1, and one separate SQ1 key0. For any fixed
selected set S, choose a later-frame row u outside S whose query becomes
L>0; all other queries remain0. Every selected seed is still0, so the two
bulk keys merge uniformly with Kbar0 and entropy log2. For query u,

    F(z_u) = log(exp(-L)+exp(L)+1),
    F_tilde(z_u;S) = log3.

The gap is positive. If u is selected alongside the zero anchor, its
normalized positive-key response exceeds the anchor's and its negative-key
response is smaller. The two bulk keys obtain different owners, yielding
two singleton bulk cells plus SQ1. Compression is then exact for every
query in this example. RF0 and K are identical for all choices of u, so a
selector observing only them cannot be optimal for every possible later Q.
This does not show that temporal information helps real videos, or that a
strict RF0 method cannot reach a particular empirical PSNR threshold.

A possible separately authorized next experiment would score a fixed
small set of query frames using concatenated per-frame Fisher features,
then retain one shared C4 row set and unchanged per-frame actual-Q merge.
This is not strictly RF0-only, even if the dispatch parameter still says
router_frame=0. The change is which Q tensors the selector may inspect.
Three scoring frames raise selector QK work from about0.677% to2.032% of
dense QK work; four give2.709%. These counts do not predict measured
attention latency, and no quality or no-slowdown guarantee follows.

## Bounded conditional-medoid pilot after the stable merge control

`sq1_conditional_medoid_cpu_20260906cd` examines nine predeclared banks
from the locked prompt1/step0/L33/head30 capture: query columns 0,40,79
crossed with key-frame/key-row pairs (0,0),(10,22),(20,44).
It is a CPU-only feasibility diagnostic, not global attention, a native
selector, an all-layer run, video quality, or a speed measurement.

The existing owner-preserving conditional-medoid reference makes one pass
after Anchor32. Optimization uses only RF0's C32 real candidate/target rows
and Uniform16 plus the observed singleton, with positive bin measures.
The real anchor is fixed. For measurement only, all 45 target queries in
each of frames 0,10,20 are evaluated against the full 80 keys. Each seed
set independently rebuilds full80 ownership, singleton removal, conditional
weights and the restricted route. FP64 arithmetic on original BF16 Q/K
isolates this local variational comparison from native merge rounding.
The selector prototype instead uses CPU FP32 products followed by FP64
Fisher/GS and is not claimed to reproduce native seed rows bitwise.

Three of nine banks change, involving four of 27 adjustable seed slots.
The mean probe gap decreases from 0.0535250949 to 0.0482651707;
none of the 27 individual coordinate steps increases its probe objective.
The full80 mean reverse-KL gaps over the same nine banks are:

| Query frame | Original Anchor32 | Refined | Difference |
| --- | ---: | ---: | ---: |
| 0 | 0.0923008767 | 0.0813310366 | -0.0109698402 |
| 10 | 0.0559633464 | 0.0640386766 | +0.0080753302 |
| 20 | 0.0682324342 | 0.0771410866 | +0.0089086523 |

The full80 bulk ownership changes at 11,138,90 of the 711 measured key
positions for these respective frames. Probe-owner preservation therefore
does not preserve the actual full-key cells or later-frame cells. Direct
reverse KL of the reconstructed 80-key distributions was also checked
against the reported restricted gaps.

This fixed sample shows useful RF0 replacements but inconsistent transfer
to the other measured frames. It does not justify immediate native/video
promotion of the refiner. It is not an impossibility result for strict RF0
or PSNR12, nor evidence that multi-frame selection will improve videos.
No extra banks, parameter search, GPU work, or paper changes were performed.
The attached CPU command exited 0; all 27 full80 cases have a direct-KL
identity error below 8.89e-16. Results and six source files are archived
under `C:/GMA/experiments/sq1_conditional_medoid_cpu_20260906cd`.
Result SHA-256:
`3954f233ce3a611bf7451aaf57345aedd39e02229c30543777697637b475602c`.
