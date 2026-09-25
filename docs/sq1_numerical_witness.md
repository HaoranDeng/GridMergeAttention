# Fixed-seed SQ1 numerical witness

This is an implementation diagnostic, not a new selector, a video-quality
result, or a timing benchmark. It follows the paper's requirement that each
bulk representative be a normalized conditional distribution after the
matched singleton is removed. The paper repository is unchanged.

The successful `20260906bz` diagnosis below attributes the severe row-9
witness error primarily to the conditioned K/V representation, with the
seed table, observed ownership/profile, and native validity mask fixed.
It does not establish an entropy-only explanation or a video-quality gain.

## Why diagnose before another seed sweep?

The completed `quality_first_bf16_20260905b` experiment holds the original
BF16 inputs and native seed rows fixed and compares native BF16 GCA with
its ideal FP64 construction. Reading its saved `prompt001_outputs.npz`
isolates a particularly informative witness: prompt 1, head 30, query frame
0, column 40, with every one of the 945 key blocks included.

| Query row | Mass ideal vs dense MSE | Mass native vs ideal MSE | Hellinger RF0 native vs ideal MSE |
| --- | ---: | ---: | ---: |
| 8 | 7.605187913664689e-9 | 0.20553614845752885 | 6.398914895498630e-6 |
| 9 | 3.344942160002497e-14 | 4.304488194843791 | 3.934208214347932e-6 |

Only these two of the 45 query rows have native-versus-ideal MSE above
0.01. Mass and Jeffreys have identical native outputs at both rows. The
ideal Mass representation at row 9 is already nearly exact, so a poor
ideal seed approximation does not explain this particular native error.
This does not establish that current Anchor32 has the same bottleneck.

## Prespecified attribution checks

First reproduce the saved native BF16 witness exactly, using its original
seed table, original tensor layout and current-frame real merge queries.
Do not infer a component failure if this replay fails. Then retain the
actual native merged K/V and negative-entropy bias, and compare them with
independently normalized FP64 conditionals on the same full key support.

Separate K/V and bias counterfactuals, keep the complete compressed route,
and inspect the dominant representatives for query rows 8 and 9. Numerical
reference owners and residual weights must not be labeled as an actual
native intermediate trace. Record any observed owner changes separately.

For a nonempty cell of size n, ideal negative entropy lies in
[-log(n), 0], and merged K/V lie in the convex hull of that cell. In the
default SQ1 path the kernel masks the singleton before its K/V dot product,
then divides by a subtractively computed residual mass. Its entropy update
also contains subtraction followed by division. The existing `directbulk`
control still subtracts singleton probability and its entropy moment; it
is not a cancellation-free reference.

No correction or stability-based seed penalty is justified merely by this
static risk. A successful attribution must demonstrate its effect on the
saved native witness. Any subsequent candidate still requires canonical
four-step, full-20-video PSNR strictly above 12 and matched complete-attention
timing no slower than Hellinger C4.

## First diagnostic attempt: replay passed, observer interface failed

`sq1_fixed_seed_witness_20260906bw` ended at
`2026-09-06T12:45:47.809246+00:00` with worker exit code 1 and
`valid=false`. The saved 45-by-128 native BF16 output for the witness was
reproduced bitwise (`gates.old_native_bitwise=true`). However, the generated
observer failed in Triton's argument binder with
`SyntaxError: non-default argument follows default argument`, before its
trace could be validated or any counterfactual completed. Consequently,
`counterfactuals` is empty: this attempt does **not** attribute the error to
merge normalization, entropy, ownership, or routing.

This is a failed diagnostic interface attempt, not a failed selector or a
video-quality/timing result. The original frozen diagnostic source has
SHA-256 `11403891da51e4379debc02af648ae7e51cd8b04a921258c80f98fde6cc7c53c`.
Its seven primary records and 43 frozen source files are archived under
`C:/GMA/experiments/sq1_fixed_seed_witness_20260906bw`; Python and Triton
caches are excluded. No inference correction follows from this partial
replay success alone.

## Second diagnostic attempt: replay passed, observer code generation failed

`sq1_fixed_seed_witness_20260906bx` ended at
`2026-09-06T12:54:16.874177+00:00`, again with worker exit code 1 and
`valid=false`. It reproduced the same saved native witness bitwise, but
the observer then failed during Triton code generation with
`NameError('witness_logits2_ptr is not defined')`. Giving the added
keyword-only pointers defaults resolved the earlier binder syntax error,
but did not make those pointers available to the code generator.

There is still no validated observer trace and `counterfactuals` remains
empty. This second interface failure does not identify a numerical cause
or justify an inference-path correction. Its seven primary records and
43 frozen source files are archived under
`C:/GMA/experiments/sq1_fixed_seed_witness_20260906bx`, with caches excluded;
the failed attempt is not counted as a selector, quality, or timing result.

## Third diagnostic attempt: replay passed, observer store shape failed

`sq1_fixed_seed_witness_20260906by` ended at
`2026-09-06T13:05:50.890668+00:00` with worker exit code 1,
`valid=false`, and no observed foreign GPU processes. The saved native
witness was again reproduced bitwise. The observer then failed to compile
because its `tl.store` pointer had shape `[1, 4, 64]` while the data had
shape `[16, 4, 64]`; the pointer's leading query-column dimension had not
been expanded to the data shape.

No observer trace was validated and no counterfactual was completed, so
the numerical cause remains unattributed. The primary records and 43
frozen source files are retained under
`C:/GMA/experiments/sq1_fixed_seed_witness_20260906by`, excluding caches
and the separately referenced capture arrays. This is a diagnostic
instrumentation failure, not evidence about candidate quality or speed.

## Successful diagnostic: K/V conditioning dominates the row-9 error

`sq1_fixed_seed_witness_20260906bz` completed at
`2026-09-06T13:13:14.732477+00:00`, with exit code 0, `valid=true`,
and no foreign process observed in 15 monitoring polls. All eight gates
passed: the old native output, observer output and K/V/bias buffers were
bitwise reproduced; the old ideal and dense FP64 outputs met their replay
tolerances; inputs and sources remained unchanged. Four counterfactuals
completed. The scope remains one fixed Mass-RF0 witness: prompt 1, call 33,
layer 33, step 0, original head 30, query frame 0 and column 40; all 945
key blocks are included, and only 45-by-128 output values are interpreted.

For the first three substitutions, both the observed native normalized
log profile and ownership were frozen, along with the native cell-validity
mask. Stable conditionals exclude the matched singleton, center logits on
the owned bulk maximum, and normalize directly in FP64. Their K/V are then
cast to the original BF16 buffer type and bias to FP32 before substitution
into the unchanged native full-compressed route. Thus these are diagnostic
buffer substitutions, not an already validated replacement merge kernel.

All entries below are MSE against the same **dense FP64** reference, rather
than the native-versus-ideal metric used in the earlier witness table.

| Native-route input buffers | Row 9 MSE | Row 8 MSE | All 45 rows MSE |
| --- | ---: | ---: | ---: |
| Original K/V and bias | 4.3044880974 | 0.2055806431 | 0.1003941358 |
| Original K/V; stable bias; native valid mask | 4.0314194905 | 0.0013341062 | 0.0897870881 |
| Stable K/V; original bias; native valid mask | 3.9179590700e-6 | 4.8116730989e-6 | 1.0321233622e-4 |
| Stable K/V and bias; native valid mask | 3.9179590700e-6 | 4.8116730989e-6 | 1.0321233622e-4 |
| Stable K/V and bias; structural nonempty mask | 3.9179590700e-6 | 4.8116730989e-6 | 1.0321233622e-4 |

Changing K/V while retaining the original bias removes the catastrophic
row-9 error; changing only bias does not. Bias does matter for row 8, so
this is not a claim that entropy arithmetic is universally harmless.
Re-routing the original K/V and bias in FP64 still gives row-9 MSE
`4.2999084548`, further locating this witness's dominant failure upstream
of the final route arithmetic. The three stable-K/V native-route outputs
are identical in the saved arrays, including when the 17 structurally
nonempty but originally dropped cells are restored.

### Dominant faulty representative

For query row 9, the dominant faulty bulk representative is block 234
(key frame 5, key row 9), cell 3, seed row 7, with 28 bulk keys. Its
observed singleton weight is `alpha=0.9999989867210388`. Subtraction gives
`r=1-alpha=1.0132789611816406e-6`, while directly summing the observed
remaining weights gives `1.1293706576193089e-6` (the stable-profile sum is
`1.1293705358096192e-6`). The BF16-rounded bulk weights divided by the
subtracted residual have effective total mass `tau=1.1137598584741528`,
not one. This is consistent with an inflated K/V representative rather
than a normalized convex combination.

In an FP64 inspection of the native buffers, that representative's logit
is `372.1136830`, exceeding its native-owner cell logsumexp `331.2683187`
by `40.8453643`. Its reconstructed attention weight is `0.9977328815`,
whereas the ideal representation receives only `7.9510912346e-16`.
Its bias is also inaccurate (`-3.4641475677` versus stable
`-0.7284595056`), but the isolated substitutions show that a bias-only
correction does not resolve the main row-9 failure.

There are 10 native-versus-true-FP64 owner mismatches among 75,600 keys;
the stable substitutions retain the native owners rather than silently
fixing these. The all-row counterfactual MSE remains `1.0321233622e-4`:
this is not complete numerical parity, a statement about every head, or
a result from canonical 20-video generation. No PSNR, SSIM, LPIPS, or
matched complete-attention speed claim follows from this diagnostic.

The 51 primary artifacts, including 43 frozen source files and the
21,458,136-byte intermediate array archive, are retained under
`C:/GMA/experiments/sq1_fixed_seed_witness_20260906bz`. Results SHA-256:
`96e7043c79ec6b8d2ccaa8ed9967fa132e5370987912ad6507f5d8ab8f1ad6e3`;
arrays SHA-256:
`d948c65c2935edac04639cdca37e8919059c3863424cc5c8e89dd8560f1230cc`.
Each copied file and extracted primary-archive member was SHA-verified;
CPU reads of the saved arrays independently reproduced the reported MSEs
and confirmed the identical stable-K/V outputs. Caches and separately
referenced capture inputs were not copied or rehashed during archiving.

## Native stable-kernel control

`stable_sq1_witness_20260906ca`, frozen at
`52d6fdbd2097d138dab34c0ffe48f9ae11523b74`, completed at
`2026-09-06T13:25:26.338842+00:00` with exit code 0, `valid=true`,
all 14 gates passed, and no foreign process observed in 15 monitoring
polls. Unlike the previous buffer-substitution diagnosis, this control
actually compiled and ran the isolated stable SQ1 merge kernel before
the unchanged native route. The scope is still the same fixed-real-seed,
one-head witness, with all 945 key blocks and 45 interpreted query rows.

The stable observer matched the stable kernel's output and K/V/bias
buffers bitwise. Its 302,400 QK scores and normalized log probabilities
were also bitwise identical to the previous kernel's recorded trace;
there were zero ownership changes among the 75,600 keys. Thus the gain
on this witness does not rely on a different selector or partition.
Bulk weights are computed directly after singleton exclusion, without
subtracting a singleton mass or dividing the final K/V by `1-alpha`.

| Complete native route following | Row 9 MSE vs dense FP64 | Row 8 MSE vs dense FP64 | All 45 rows MSE vs dense FP64 |
| --- | ---: | ---: | ---: |
| Original merge kernel | 4.3044880974 | 0.2055806431 | 0.1003941358 |
| Actual stable merge kernel | 3.9179590700e-6 | 4.8116730989e-6 | 1.2619336543e-4 |
| FP64 direct-bulk reference buffers, cast to native types | 3.9179590700e-6 | 4.8116730989e-6 | 1.0321233622e-4 |
| Stable-kernel buffers with the original valid mask restored | 3.9179590700e-6 | 4.8116730989e-6 | 1.2619336543e-4 |

The stable kernel removes the catastrophic row-8/9 errors, but is not
identical to the reference-buffer route over all rows. Retained BF16
merge-weight, dot-product and buffer-storage arithmetic still leaves
numerical error: maximum K/V coordinate differences from the direct
reference are `0.3864194911` and `0.0364481285`, respectively, while the
maximum finite bias difference is `2.54150236e-7`. This comparison does
not isolate every remaining rounding contribution. Improvements are not
uniform across rows; for example, row-3 MSE rises from `1.0063225492e-4`
to `5.8578563215e-4`, despite the much lower total witness MSE.

The structural mask retains 3,067 cells instead of 3,050, restoring 17
nonempty cells and dropping none. Restoring the original mask gives an
output bitwise identical to the stable-kernel output in the saved arrays;
therefore those extra cells do not explain this witness's improvement.
This validates an off-path native stability control on this witness,
not canonical 20-video quality, every head/layer, or matched latency.
No PSNR, SSIM, LPIPS, or speedup is reported here.

The 55 primary files, including 46 parent-recorded sources (45 of which
also appear in the worker's source manifest) and the 15,215,209-byte
`control_arrays.npz`, are archived under
`C:/GMA/experiments/stable_sq1_witness_20260906ca`. The parent-only source
is its launcher; its hash agrees with `startup.json`. Every file and
extracted archive member was SHA-verified. Independent CPU checks of the
saved arrays reproduced all seven output MSE records and confirmed the
prefix, ownership, seed, structural-mask and output-equality statements.
Results SHA-256:
`a54e2854a6252632cebed73f45c6c5159a725eb0af759d17cd37f2dfa1f1be02`;
arrays SHA-256:
`f95371c2db9ee23835a3aff5d562066f9eb2f71ba6672489de085a73a165c45b`.
Caches and separately referenced inputs were not copied.

## Canonical full-video test of the validated merge

The next experiment is
`wan14b_720p_4step_fisher_anchor_stable_sq1_rf0_bf16score_extended_20260906cb`,
with source frozen at `e023d797820da49223ebcab91f2e855607dfee1b`.
Its worker was launched on scai6 GPU 4 at
`2026-09-06T13:40:35.566796+00:00`. The generated merge source has SHA-256
`fa2a729139cdbfee1abba9265eedf463c7f68b0f7de1f4402d6d48c2fe3a8e68`,
identical to the source exercised by the successful native control above.

The candidate keeps the existing real-row Anchor-Fisher selector, RF0,
Uniform16 probes, C32 candidate table, and unchanged global compressed
route. Only the explicitly experimental stable SQ1 conditioning is added.
This is a numerical substrate test, not a newly demonstrated seed rule.
All 40 layers and four denoising steps must use this same candidate for
all 20 canonical prompts, with seed zero. Completion requires 3,200
complete attention calls and 16,000 actual stable merge-kernel calls.
The launcher enforces the scai6 account-wide limit of two GPU devices,
including unrelated same-account tasks, and monitors selected-device
exclusivity every five seconds.

Full-resolution CPU PSNR runs concurrently on the fixed set of all 20
videos and waits for valid generation completion. Prefix values are not
used to rank or stop this run. Generation timing includes audit and host
scoring overhead and is not the matched Hellinger speed comparison.
At launch, canonical PSNR above 12 and no slowdown versus Hellinger C4
both remain unproven. No default method or paper text is changed.

The canonical run completed at `2026-09-06T14:04:05.597373+00:00`
with actual parent/worker exit 0, all 20 videos, 3,200 complete attention
calls, 16,000 stable merge-kernel calls, and 80 calls in each of the 40
layers. Its 267 five-second monitor samples observed no foreign process;
the account-wide two-GPU limit was retained. The CPU scorer also exited
0 after all 20 videos and 1,620 original-resolution frames.

**Full-20 PSNR is 11.64718233856182**, versus the original Anchor32
11.636408427866783: **+0.010773910695036193 dB**, still below the strict
PSNR >12 target. Thus the large improvement on the fixed numerical witness
does not translate into a comparably large end-to-end video improvement.
This test changes merge normalization, not the seed rule; it is not
evidence for a new optimal selector. No further timing run is launched
on the strength of this quality result alone.

Instrumented complete GCA time is 148.07271816253663 ms/call
(last 19 videos: 141.93057145570455); full attention-module time is
185.10771906375885 ms/call. Mean inference is 50.421735192788766 s/video
(last 19: 41.89510263399662). These include generation instrumentation
and concurrent CPU scoring; matched `speedup_vs_sdpa` and no-slowdown
versus Hellinger remain unmeasured. SSIM, LPIPS and VBench are unmeasured.
The exploratory workbook records the completed values in method row 794.

PSNR result SHA-256:
`5f6108ce0651d8ffd75b6df0258b21c93b18eced6150a6fdf413893cdd453a34`;
generation completion SHA-256:
`334cf1cc2a510680153b192651c68b358337c80fc13d7a5e0b14e91022e69db7`;
runtime SHA-256:
`296bacfd53f507e296e11b76297db0c3844ca72435cd1edca7b5f6a5fc3963a4`.

## Prespecified stable Mass-GS video control

The explicit nondefault `mass_stable_sq1_rf0` combines the original
Mass-GS C32/g45, Uniform16/G256, J4 and RF0 selector with the same stable
merge source exercised by ca and cb. It changes no seed-selection rule,
production default, query-frame policy, checkpoint, or paper file.

This is a bounded numerical-substrate retest, not an unchanged rerun of a
negative selector: the validated ca witness used **Mass** seeds, whereas
the only completed stable full20 test, cb, used **Anchor-Fisher** seeds.
The original BF16 Mass full20 PSNR is 11.533208174; the small cb gain does
not determine how strongly Mass is affected by the confirmed merge error.
Conversely, ca's one-head numerical gain does not predict full-video PSNR.
Neither a newly optimal selector nor attainment of PSNR12 is asserted.

The fixed test is all 20 canonical prompts at seed0, four denoising steps,
all40 layers and all3200 attention calls, with original BF16 Q/K/V and
the unchanged full compressed route. Use a fresh `20260906ce` run; never
resume or substitute old outputs. Require 16000 actual stable merge calls
and the ca/cb generated-kernel SHA. Generation is admitted only on one
exclusively clean scai6 H200 within the account-wide two-device cap.
Full20 CPU PSNR may run concurrently, but its prefix values do not rank,
stop, or alter the candidate. Instrumented generation latency is not a
matched Hellinger or SDPA speed claim. Only complete quality determines
whether to proceed to a separate matched speed benchmark.

### Completed stable Mass-GS control

`wan14b_720p_4step_mass_stable_sq1_rf0_bf16score_extended_20260906ce`,
frozen at `7dde414b4b323937e9b95575eea6807c36a022cc`, completed on
2026-09-06 at 14:53:12 UTC with actual generation exit 0. All 20 videos,
3,200 complete attention calls, 16,000 stable merge calls and 80 calls
per layer were recorded. The original native Mass selector was used
without a selector override, and the generated stable merge source
matched the ca/cb SHA. All 234 selected-GPU monitor samples were free of
foreign compute processes; the account-wide two-device limit was kept.

**Full-20 PSNR is 11.5152300730207**, compared with original Mass BF16
11.533208173672342: **-0.017978100651642848 dB**. It is also below stable
Anchor-Fisher's 11.64718233856182 and the strict PSNR >12 target. The
fixed numerical-witness improvement therefore does not establish an
end-to-end quality gain for Mass. This control is not promoted and does
not justify a separate matched speed run.

Instrumented complete GCA time is 140.35556132793425 ms/call (last 19
videos: 134.19691713232743), and full attention-module time is
172.25451967716216 ms/call. Mean inference is 48.206996667245406 s/video
(last 19: 40.63632152964802). Generation audits and concurrent CPU scoring
were active; these are not matched SDPA or Hellinger timings. Matched
speedup, SSIM, LPIPS and VBench remain unmeasured.

The CPU scorer completed with actual exit 0 at 14:56:13 UTC. All 20 paired
prompt identities and all 1,620 full-resolution frame scores were checked.
The 20 generated videos, frozen sources and terminal evidence are archived
locally with `final_archive_receipt.json`; copied video hashes agree with
the inputs recorded by the scorer. Exploratory workbook row 818 contains
the completed PSNR and instrumented runtime.

PSNR SHA-256:
`8fea53c4f3ec6e73325b7a940f313ca9c5cca04017403f6745926093868d5156`;
generation completion SHA-256:
`39ec580488822f5711cf71627afac41a98a74fc3100dd9980e4e7edb2c97f1ec`;
runtime SHA-256:
`9106847985c34700d5865728923a73e961af3d3acda722fa135180365743a3fe`.
