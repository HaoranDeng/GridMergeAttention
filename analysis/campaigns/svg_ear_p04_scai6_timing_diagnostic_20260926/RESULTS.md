# scai6 SVG-EAR p=0.4 timing diagnosis

Completed ten executions of five fixed sparkler_v1 inputs on two exclusive H200s. Each card used indices000-004 and seed=index. Index000 is warmup; steady values use001-004 separately per card. No p=.5 result is included.

| Card | Core ms/call | Full attention ms/call | Inference s/video |
|---|---:|---:|---:|
| scai6 GPU2 | 110.502409 | 132.842529 | 817.929604 |
| scai6 GPU4 | 110.297450 | 132.550069 | 815.874812 |

Core and full attention are CUDA-event scopes; inference is the generation pipeline wall time, excluding model load and file export. Four thousand attention calls per video. These are measured timings with the campaign audit/logging enabled.

## Per video

| Card | Index | Role | Core ms | Full ms | Inference s | Full50 PFLOPs |
|---|---:|---|---:|---:|---:|---:|
| GPU2 | 000 | warmup | 127.304628 | 149.886458 | 1169.919278 | 75.082028 |
| GPU2 | 001 | steady | 110.245023 | 132.625563 | 817.476799 | 74.024301 |
| GPU2 | 002 | steady | 110.318947 | 132.635138 | 816.958951 | 74.315727 |
| GPU2 | 003 | steady | 110.201444 | 132.549365 | 816.868055 | 73.695201 |
| GPU2 | 004 | steady | 111.244223 | 133.560050 | 820.414610 | 75.787620 |
| GPU4 | 000 | warmup | 128.516141 | 151.052674 | 1169.831821 | 75.075275 |
| GPU4 | 001 | steady | 110.255713 | 132.516828 | 816.090176 | 74.024301 |
| GPU4 | 002 | steady | 110.078204 | 132.315276 | 814.854061 | 74.315727 |
| GPU4 | 003 | steady | 110.038308 | 132.318183 | 815.066744 | 73.695201 |
| GPU4 | 004 | steady | 110.817574 | 133.049989 | 817.488267 | 75.793569 |

## Comparison and limits

Prior scai7 GPU7 p=.4, on the same steady input subset001-004, measured 106.592765 ms core, 128.558711 ms full attention and 797.504747 s inference. Its original19-video steady core mean was106.275035 ms. Compare equal input subsets descriptively; this is not a randomized cross-GPU experiment.

GPU2 is 3.668% slower in core time than scai7's matched four-input subset. Busy telemetry (568 samples): SM clock mean 1774.2 MHz, range 1500-1980; memory clock 3201 MHz; temperature mean 69.9 C, max 76 C; sampled power mean 639.7 W, limit 700 W.

GPU4 is 3.476% slower in core time than scai7's matched four-input subset. Busy telemetry (567 samples): SM clock mean 1784.5 MHz, range 1455-1980; memory clock 3201 MHz; temperature mean 56.4 C, max 61 C; sampled power mean 634.1 W, limit 700 W.

No foreign compute processes were observed during the monitored run intervals. Launch checks found clean cards. The large temperature difference between GPU2 and GPU4 accompanied nearly equal timings; this alone cannot identify the cause. There is no newly measured p=.5 control, so this test does not prove that lowering top_p itself slows the implementation. No same-run Dense reference was requested, so no SDPA speedup is inferred.

## Verification and provenance

Independently checked 59 synchronized file hashes, all10 video hashes without decoding, all40,000 trace rows, all50 steps x2 CFG branches x40 layers, seeds, input hashes, source/config identities and full50 FLOPs sums. Source and the first five input identities match the previous scai7 campaign. Actual and scheduled upper FLOPs are stored per video; short diagnostic steps are not extrapolated to full-video FLOPs.

The input set retains its documented post-hoc sparkler019 replacement history; this test only uses000-004. Canceled GPU4 p=.5 partial work and never-started GPU2 p=.5 are excluded. No paper figure, table, equation, text or20-video aggregate was changed. Raw artifacts are in results/, and the independent machine-readable audit is INDEPENDENT_VERIFICATION.json.
