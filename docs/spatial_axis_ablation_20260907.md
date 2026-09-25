# Spatial-axis ablation: key columns and query rows

The user requested a first H/W swap before exploring other ordered pairs of
the video T/H/W axes. This is a code/experiment change, not a paper edit.

## Mapping and preserved method

For the model's post-patch latent grid T=21, H=45, W=80:

| Setting | Query group | Key group | Block size (query x key) |
| --- | --- | --- | --- |
| Completed control | Q[tq, :, wq] | K[tk, hk, :] | 45 x 80 |
| Requested H/W swap | Q[tq, hq, :] | K[tk, :, wk] | 80 x 45 |

The swap is internal to the attention operator, after the original model's
RoPE. Q, K and V are permuted together and the output is inversely permuted.
The checkpoint, positional encoding, scheduler and output-token order do
not change. Every query still routes over every key frame and spatial key
group. A spatial intersection does not require tq=tk.

The method remains training-free, pure all-layer Anchor-Fisher RF0 with
four real seed queries, hard ownership, stable bulk normalization, entropy
correction, one singleton and full compressed attention. No dense rescue,
FP8, sparse second stage or new learned component is introduced.

In the swap, the retained anchor is width index wk. RF0 selects width
indices from Q[0,hq,:]; merge reads the corresponding actual current-frame
queries Q[tq,hq,:]. The singleton K[tk,hq,wk] is removed exactly once from
the bulk partition and included exactly once in full compressed routing.

Candidate count remains 32 and the multiscale geometry budget remains 45.
The latter is a candidate-table budget, not the query-axis length: the
transposed table covers all 80 possible anchors. Uniform16 probes are
floor(i*45/16), i=0,...,15, over valid key positions. Padded positions must
not participate in normalization, ownership, entropy or routing.

## What this first comparison can establish

Holding C4+SQ1 fixed changes the nominal compressed-token budget:

| Quantity | Control | H/W swap |
| --- | --- | --- |
| Keys per merge group | 80 | 45 |
| Query group size | 45 | 80 |
| Key groups per frame | 45 | 80 |
| Maximum representatives per query | 4725 | 8400 |
| Nominal representatives / original tokens | 6.25% | 11.11% |

These counts include four allocated bulk slots and one singleton per key
group. Empty bulk slots are masked; this is not a measurement of nonempty
clusters, attention-map sparsity, FLOPs or wall time. A quality difference
cannot be attributed to orientation alone at a matched representative
budget. Report quality and measured latency together before considering a
separate matched-budget experiment.

## Current experiment: four-step user override

On September 7 the user requested the existing four-step protocol instead
of waiting for 50-step generation. Use the actual
Wan2.1-T2V-14B-StepDistill-CfgDistill checkpoint (raw state dict), not the
original model with its step count shortened. Its denoising schedule is
[1000, 750, 500, 250], with the existing warped schedule and timestep shift
5. Preserve 1280x720, 81 frames, seed 0, the canonical 20 prompts and pure
all-layer execution: 40 layers x 4 steps = 160 calls/video, 3200 total.

The completed original-layout control is
`wan14b_720p_4step_fisher_anchor_stable_sq1_rf0_bf16score_extended_20260906cb`.
Its full20 PSNR is 11.64718233856182, SSIM is 0.36460963412710085 and
LPIPS-Alex is 0.5050692214090147, against its verified four-step dense
reference, not the 50-step reference. CPU evaluation uses all 1620 frames
at full resolution. Every per-video PSNR agrees with the earlier result;
the new aggregate differs only by 1.8e-15 from summation roundoff. The
evaluation JSON SHA-256 is
`59da9395667899fa83d5052fbc4ad53f51069275c8ff9ac71370b50f3ad8a158`.
Its historical timings are 148.072718 ms/core call, 185.107719 ms/full
self-attention call and 50.421735 s/video, including instrumentation and
concurrent CPU scoring; they are not a newly matched speed benchmark.

The new method identifier is `fisher_anchor_stable_sq1_rf0_hw_swap`.
The existing validated transpose kernels are unchanged; only the distilled
model integration changes. Reused native validation must match all four
kernel/reference source hashes and retain its original receipt. Record
the actual all-layer core/selector/merge/route coverage for the new run.

The four-step swap was launched on a clean scai7 GPU0 at 08:41 UTC from
frozen commit `c4e00be`, under
`wan14b_720p_4step_fisher_anchor_stable_sq1_rf0_hw_swap_bf16score_extended_20260907ck`.
All 67 focused CPU checks passed locally and in the remote environment.
The independently validated GPU operator proof is copied unchanged into
the new run; video generation and its actual dispatch coverage are new.

Generation completed validly at 09:10 UTC. The new run contains all20
canonical videos, 3200 core/selector calls, 16000 stable merge/full-route
chunks, and 80 calls in each of all40 layers. The five-second GPU monitor
observed no foreign process; both generation processes exited with code0
and GPU0 was released. All538 frozen source files and the five completion
artifact hashes were independently checked in the local archive.

| Four-step latency | All20 videos | Steady19 (excluding first compile) |
| --- | --- | --- |
| Core attention, ms/call | 177.42418244361878 | 171.86127177790593 |
| Full self-attention, ms/call | 220.53588673114777 | 203.45557108929282 |
| Inference, seconds/video | 55.476556692412125 | 46.66970388645208 |

The first video includes initialization/compilation. Model-loading time is
outside these inference measurements. The original-layout historical
control's steady19 inference time is 41.895103 seconds/video. Neither this
comparison nor the historical dense reference is a newly matched SDPA
timing benchmark. Three-metric CPU evaluation runs independently of the
generation GPU and validates all20 paired outputs before publication.

## Completed four-step quality comparison

Both methods use the same verified canonical20 prompts, seed0, checkpoint,
four-step schedule and historical four-step dense reference. All1620 frames
are evaluated at 1280x720, without resizing or subsampling.

| Metric | Original query H / key W | Swap query W / key H | Swap minus original |
| --- | --- | --- | --- |
| PSNR (higher better) | 11.64718233856182 | 11.30442199177352 | -0.34276034678830 |
| SSIM (higher better) | 0.36460963412710085 | 0.34862563584900447 | -0.01598399827809638 |
| LPIPS-Alex (lower better) | 0.5050692214090147 | 0.5160444784863496 | +0.01097525707733493 |

The swap improves PSNR on8/20 videos, SSIM on7/20 and LPIPS on9/20, but
all three full20 means deteriorate. It also has higher observed inference
and attention latency. Retain the original layout as the current choice;
do not promote the swapped layout or infer improvement from early prefixes.
The swap receives a larger nominal representative budget (11.11% versus
6.25%), so this is not a matched-budget orientation-only comparison.

The independent PSNR-only and full-suite per-video PSNR values agree
exactly; their aggregate difference is only 1.8e-15 summation roundoff.
The full-suite result SHA-256 is
`486a0368f412fda5103899b3eb1fc9b75f799c53204bfc1352141c67fa20b3e1`.
The separate PSNR-only result SHA-256 is
`a428aef46a5cb392d9358b5f24314a5911beca59073879c7048f6cab8023394b`.
Two cross-host streaming PSNR attempts encountered transient EEXIST while
reading the live runtime file; neither affected video generation. Their
partial records are preserved, and the final non-streaming20 evaluation
succeeded. The full-suite CPU driver reused the unchanged frozen evaluator
and validated completion hashes, all3200 calls and all40 paired input-file
hashes before publishing its final result. VBench and matched SDPA speedup
remain unmeasured.

The next axis candidate is query W / key T, using original frame0 query
values for seed selection. This requires an explicit temporal key
implementation; simply calling a spatial reference plane RF0 would change
the method. Its completed result is reported below.

### Temporal-key four-step candidate

The next explicit candidate is key T21 / query W80 / remaining H45.
Physical THW is permuted to HWT inside the timed attention call and restored
after full routing; positional embeddings are not changed. RF0 seeds are
selected from physical Q[0,hq,:], not a first-H reference plane. The compact
seed table is indexed by hq, hk and wk and reused across current query times;
merge weights still use current Q[tq,hq,selected_w]. C32/g45 selection uses
16 distinct temporal probes floor(i*21/16). Hard ownership normalizes over
all21 valid keys before removing SQ1 K[tq,hk,wk], followed by stable bulk
normalization, entropy bias and routing over every hk/wk key group.

Ten four-head chunks bound temporary memory. Each video must execute160
selectors/cores and1600 merge/route chunks, with all40 layers and all4 steps.
The new native GPU proof runs under the same exclusive-GPU guard before
model construction, checking physical RF0 invariance, all320 small-shape
seed cells, 48 merge cells, 640 routed queries, temporal and width tails,
empty bulks, and a full40-head production-shape smoke. A spatial-axis proof
cannot authorize this distinct operator. CPU mathematical/integration
tests pass75 cases on both Windows and scai7. The first guarded run uses
frozen commit5662c1a, directory
`wan14b_720p_4step_fisher_anchor_stable_sq1_rf0_key_t_query_w_bf16score_extended_20260907cl`.
Native H200 validation passed before model loading: both cases had zero
seed mismatches and zero checked BF16 merged-vector error; random-case
route max absolute error0.000198508 and RMSE0.0000358411. Full40-head smoke
executed one selector/core and ten merge/route chunks. The copied native
proof SHA-256 is
`18478ae1ffb5b53e4f3fb7abbe1768df543073fd0adb80a0fb2eeeaefaf03f0c`.
The run's canonical prompt manifest is verified. Generation subsequently
completed validly at 10:12 UTC: all20 videos, 3200 selector/core calls,
32000 stable merge/full-route chunks and 80 calls in each of all40 layers.
The five-second monitor recorded399 polls and no foreign process. The
worker exited with code0 and GPU0 was released before the next launch.
The local archive verifies all543 frozen source files, all20 video hashes
and all five completion artifact hashes; it includes the GPU monitor log.

CPU evaluation used the unchanged frozen evaluator and all1620 full-size
frames against the same canonical four-step dense reference. It validated
the generation completion, prompt identities and all40 paired input-file
hashes before publishing the final result. The initial streaming helper
attempt started before the output manifest existed and exited without
metrics; the subsequent manifest-verified launch completed successfully.
Video generation was not restarted.

| Full20 metric | Original query H / key W | Query W / key T | Temporal minus original |
| --- | --- | --- | --- |
| PSNR (higher better) | 11.647182338561818 | 11.9300639891927 | +0.282881650630882 |
| SSIM (higher better) | 0.36460963412710085 | 0.37999994600483616 | +0.01539031187773531 |
| LPIPS-Alex (lower better) | 0.5050692214090147 | 0.48337167063612996 | -0.02169755077288475 |

The temporal-key candidate improves PSNR on12/20 videos, SSIM on15/20 and
LPIPS on13/20, with no ties. This is a quality/cost tradeoff, not a speed
improvement: its observed all20 full-attention latency is2.312 times the
historical original-layout control's latency. That is not an SDPA speedup.

| Query W / key T latency | All20 videos | Steady19 (excluding first compile) |
| --- | --- | --- |
| Core attention, ms/call | 384.55299830436707 | 384.5413955989637 |
| Full self-attention, ms/call | 427.90902460098266 | 415.8877028716238 |
| Inference, seconds/video | 88.83444789103233 | 80.64572962030377 |

There are160 attention calls/video. Inference time excludes model loading
and video export; the first video includes remaining initialization and
compilation. Matched same-run SDPA speedup and VBench are not measured.
The full20 quality result, not a favorable early prefix, is authoritative:
PSNR remains below12. The quality JSON SHA-256 is
`a65b9b64ae7ebaa79db39847a650cfcd8b56c6c3fc4cab8bb82d54961a87c658`;
the generation completion SHA-256 is
`dac1dbe5257662c410f256ed8b1ecc0474df4343480d727d1188e5cb005021a7`.
Final metrics and timings are recorded in the existing `C:\GMA\tmp.xlsx`
Sheet1 row832. The existing original and swapped-layout rows are preserved.

This is not a matched compression-budget comparison: its nominal
representatives/original-keys ratio is5/21 (23.81%), versus5/45 (11.11%) for
the H-key swap and5/80 (6.25%) for the original W-key layout. Any quality
change must be interpreted together with this cost difference. No default
method, paper text or frozen prior experiment is changed.

The paired temporal-key candidate uses query H45 / key T21 / remaining W80,
physical Q[0,:,wq] for selection, and an explicit WHT layout restored to THW
after routing. Both temporal-key choices cover all3600 key groups and18000
nominal representatives/query (5/21), so this comparison fixes compressed
token budget while varying the query axis. It does not fix every operation
count: the H-query variant has1680 query groups versus945 for W-query.
Its compact RF0 seed table is46,080,000 bytes. Twenty2-head chunks keep the
merged KV+bias scratch at24,966,144,000 bytes, below the W-query4-head
scratch of28,086,912,000 bytes. The H-query run was not launched or queued
while the W-query20-video experiment owned scai7's one allowed device.

The H-query implementation is isolated in new native/reference files. Its
required pre-model native proof covers180 seed cells,48 merge cells and360
routed query vectors per case, height tail32..44, query times0/1/16/20,
cross-width routing, RF0 invariance, empty bulks and one full40-head smoke
with20 merge/route chunks. Combined CPU math/integration tests pass103
cases locally and on scai7. After the W-query run completed and GPU0 was
confirmed idle, the H-query run was launched at 10:14 UTC from frozen
commit `f756fc5b11cfcb76cae6735d752f907d72a3a8eb`, under
`wan14b_720p_4step_fisher_anchor_stable_sq1_rf0_key_t_query_h_bf16score_extended_20260907cm`.
Its fresh guarded H200 native proof passed before model loading: zero
seed mismatches, zero checked BF16 merged-vector error, random-case route
maximum absolute error0.0002607532940182844 and RMSE0.000047361928924978635.
Both RF0 perturbation checks and the full40-head/20-chunk smoke passed.
The native proof SHA-256 is
`24b130ff2c836e579b000a446bc00ea0156b391a5c21292b369ddb64cd737796`.
The H-query run completed on2026-09-07 at11:01:56 UTC. All20 videos and
1620 full-resolution paired frames are verified. PSNR is12.269718463618283,
SSIM0.39526291963542953 and LPIPS-Alex0.460790252244031. Mean complete
attention is583.7196122741699 ms/call (core540.2735017967224), with
113.76935034664348 inference seconds/video. Steady19 complete attention is
571.5548059965435 ms/call (core540.2579553704513), with105.57204325859875
inference seconds/video. Matched SDPA speedup and VBench are unmeasured.

Relative to the equal-token-budget key-T/query-W run, the mean quality
deltas are+0.3396544744 PSNR,+0.0152629736 SSIM and-0.0225814184 LPIPS;
the H-query variant wins12/14/15 of20 paired videos respectively. Relative
to original key-W/query-H, deltas are+0.6225361251,+0.0306532855 and
-0.0442789692, with15/16/17 wins. Both comparisons have no metric ties.
This is the best three-metric mean among these tested RF0 axis layouts,
not an equal-budget improvement over the original5/80 compression and
not a latency improvement. Query-T layouts require a different seed policy
because physical RF0 supplies only one in-line query; they remain untested.

All546 frozen source hashes,20 local candidate hashes and40 actual remote
candidate/reference video hashes match. Five completion artifact pins,
all40 layers/3200 attention calls/64000 merge-route chunks and534 clean
GPU monitor samples are verified. The completion SHA-256 is
`9a9e13545932d9708410db260c42ef2c9412779f8f70c8d2698fc95b807275f5`;
the final quality SHA-256 is
`1fc84f55aef617bea01738fee7f21c308106d5cf66c94dd6e0b473ef9828a66f`.
The existing tmp.xlsx row833 contains final quality and runtime. Videos
and primary evidence are archived under the canonical local experiments
directory. Neither production defaults nor the paper have been changed.

## Historical 50-step protocol (superseded for this ablation)

Use original Wan2.1-T2V-14B-Diffusers, 1280x720, 81 frames, 50 UniPC steps,
CFG 5, seed 0 and the same 20 canonical MovieGenVideoBench prompts. Reset
all RNGs per video as in the released single-prompt scripts. Replace all
40 layers and both CFG calls at all 50 steps: 4000 complete calls/video.

The completed control is
`wan14b_720p_50step_official_hybrid_20260906ch/gca`, with valid completion,
20 videos, 80000 attention calls and no foreign GPU process. Its mean core
attention is 149.1948386 ms and mean pipeline time is 914.6911973 s/video.
These are not 4-step results. Matched current SDPA speedup is unavailable.

The control's completed 20-video/1620-frame quality evaluation gives PSNR
14.452523752367563, SSIM 0.5217805809165689 and LPIPS-Alex
0.45597886406345134. The paired videos are 1280x720 with 81 decoded frames
each, without resizing or frame subsampling. The evaluation JSON SHA-256 is
`1749b3f65eeb1f4f7d46764db33b829cfe299b81d9385af8382ecab8c163e7ff`.

The swapped implementation must pass GPU differential checks before model
loading, including query positions 64..79, the last partial query-group
tile, all valid key positions, singleton exclusion, empty bulks and
cross-frame full routing. Record source pins, canonical manifests, actual
operator-call coverage and exclusive-GPU monitoring. Do not alter running
source snapshots or start jobs beyond scai6's two/scai7's one-device caps.

The first swap run was launched from frozen code `063d5fc` on scai7 GPU0,
under `wan14b_720p_50step_hw_swap_20260907cj/transpose`. Its native checks
passed before model loading: zero checked seed mismatches, zero checked
BF16 merged-vector error, maximum random-case route absolute error
0.00013551388005577508, and successful finite full40-head/21-frame output
with one selector and five merge/route chunks. These synthetic correctness
checks are not video-quality or performance measurements. At the user's
four-step override, the 50-step parent and its child were terminated on
2026-09-07 at 08:29 UTC. No complete video was produced. The completion
receipt correctly records an invalid, user-cancelled run (worker exit -15);
logs and the separately completed native checks are retained. No partial
50-step output is a quality or runtime result.

PSNR, SSIM and LPIPS use the verified canonical historical 50-step dense
reference, explicitly labeled historical same-protocol rather than bitwise
equivalent preprocessing. VBench and matched SDPA speedup remain unavailable
until separately measured. Exploratory results belong in the existing
`C:\GMA\tmp.xlsx`; preserve its other rows and structure.

Other ordered axis choices remain future ablations. When T becomes the
query or key axis, a fixed remaining-axis plane is not necessarily frame0;
the reference-query policy must be defined explicitly before reusing RF0
terminology. Do not silently change that policy in this H/W-only test.
