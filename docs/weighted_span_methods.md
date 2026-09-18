# Weighted-Span Seed Selection: What Each Method Does

Companion to `docs/weighted_span_frontier.csv` (the numbers) and the
`research/importance-weighted-seed-span` branch (the code). This explains the
mechanism of every method that appears in the frontier table, plus the
rejected alternatives recorded in `docs/negative_results.md`.

All methods below change only **how the four seed query rows are chosen** in
hard-C4 GridMerge. Everything downstream is held fixed and identical across
every arm: key-softmax assignment, masked-softmax K/V merge, entropy routing
bias, and full second-stage attention over all compressed tokens. Density is
therefore identical (4 merged tokens per 52-key row, 7.7%) for every
GridMerge row in the table, which is what makes their rel-L2 directly
comparable.

## Background: the hard-C4 pipeline in one paragraph

Wan self-attention tokens form a (frame, height, width) grid: 21 x 30 x 52 at
480p. A block pairs one query column-strip (30 query rows at a fixed frame and
width) with one key row-strip (52 keys at a fixed frame and height); the
strips tile the full attention matrix exactly, so nothing is dropped. Inside a
block, four query rows are elected as *seeds*; each key joins the seed whose
key-normalized score ranks it highest; each cluster is averaged (softmax
weights from its own seed) into one merged K and one merged V carrying an
entropy bias equal to the cluster's log effective size; every query then
attends densely to all 2,520 merged tokens. Seed election is the only step the
methods below vary.

## Baseline

### Key-row C4 (fixed local seeds, `local_with_ith`)

The shipped rule and the baseline every delta is quoted against. Pure index
arithmetic, no data read: for key row `k`, `start = clamp(k-1, 0, 26)` and the
seeds are rows `start..start+3`. The index-matched row `k` is always inside
the window. Free at kernel time; rel-L2 0.09742 on the capture harness.
Code: `attention/hard_route_pytorch.py::cached_local_c4_seed_rows` and the
inline `clamp` in `attention/hard_c4_wan21_1_3b_480p_triton.py`.

## The accepted method, layer by layer

The production selector composes four mechanisms. Each was measured
separately; the ablation rows in the frontier table correspond to switching
them off one at a time.

### 1. Adaptive span selection over value-output response profiles

Instead of taking the four rows nearest the key row, score candidate query
rows on real data and pick four that are *diverse in what they would
retrieve*:

- **Candidate pool**: a window of query rows around the key row (30 = every
  row at 480p; pool width was the single largest quality lever, -28.7% from
  5 to 30 at alpha=0).
- **Probe keys**: scoring every candidate against all 52 keys is too
  expensive, so 16 probe columns stand in for the row: 8 chosen by top-k over
  four fixed anchor rows' normalized responses (anchors at offsets
  -2/0/+1/+3 from the key row, each scored against all keys; probes are the
  columns any anchor ranks highest) plus 8 uniformly spaced. Data-dependent
  probes are worth 13% over fixed uniform columns.
- **Response profile**: each candidate's softmax over the probe logits,
  multiplied into the probe *values* (pooled 128 -> 32 dims, +0.14% cost) --
  i.e. what that query row would actually retrieve. Ranking candidates by
  retrieval behaviour beats ranking by raw logits.
- **Greedy span (Gram-Schmidt)**: the index-matched row is always seed 0
  (removing this anchor hurts). The remaining three picks greedily maximize
  the residual norm of a candidate's profile after projecting out the
  already-chosen seeds' directions -- i.e. pick whoever says the most that
  the chosen seeds have not already said. Pure diversity; no importance term.

This alone (the `span alpha=0 pool30` row): rel-L2 0.07299, -25.1% vs fixed.
Code: `attention/adaptive_seed_pytorch.py::_adaptive_selection`.

### 2. The importance weight (alpha)

The branch's algorithmic contribution. Each candidate's importance is its log
attention mass over the probes (`logsumexp` of its probe logits), z-scored
across the pool so one alpha transfers across blocks. Every candidate's
residual is scaled by `exp(alpha * z)` before the greedy pick, so the
objective becomes (importance-weight x leftover novelty)^2 -- greedy MAP for a
determinantal point process with quality scores. Because every Gram-Schmidt
basis is normalized, the scaling commutes with the projections and can be
applied once to the pick objective; in the kernel it is one multiply.

alpha=0.3 is the swept optimum (bracketed on two prompts); the effect grows
with pool width (-0.1% at pool 5, -8.3% at pool 30 -- measuring alpha at a
narrow pool silences it, see the guardrail entry in negative_results.md).
With alpha: rel-L2 0.06644 (PyTorch, per-column probes), -31.8% vs fixed --
the best quality measured at this density.

### 3. Probe sharing (BLOCK_J)

A kernel-efficiency compromise with a measured quality price. The Triton
selector shares one probe set across a tile of BLOCK_J adjacent query
columns so the candidate x probe product stays one tensor-core `tl.dot`;
per-column probes (the PyTorch reference) would need a per-column gather
loop. Cost: +6.7% at 8 columns, +2% at 4, +0.4% at 2. Production is
BLOCK_J=2 (rel-L2 0.06670).

### 4. The split probe / Gram-Schmidt kernels

Not a method change -- the same selection computed as two kernels instead of
one, which is what makes BLOCK_J=2 affordable. The probe kernel batches 16
column groups per program: the key tile loads once instead of per group, the
anchor scoring for all groups is a single tensor-core matmul, and the eight
top-k rounds advance for all 16 groups in lockstep. The Gram-Schmidt kernel
then runs per group without the key tile or top-k state, so its BLOCK_J=2
candidate tile fits comfortably. Handoff is 16 int32 probe indices per
group. 2.4x faster than the fused kernel at equal output quality; see
"fused vs split" in negative_results.md for why fusion lost here.
Code: `_span_probe_kernel` / `_span_gs_kernel` in
`attention/adaptive_seed_triton.py`; the merge and route kernels consume the
resulting seed table unchanged.

### Upstream reference: `query_response_span15_s8`

The pre-existing adaptive selector on main (Wan 14B 720p path), included in
the table as the `pool15/probes8 alpha=0` proxy row. Same span idea with
three differences: probability-vector residuals rather than value-output
profiles, two fixed anchors (center and left) with only two greedy picks,
15 candidates x 8 fixed uniform probes, and no importance weight. The proxy
row reimplements its budget in our harness (16 probes, so an approximation):
rel-L2 0.07704.

## Rejected alternatives (mechanisms, briefly)

Full entries with numbers in `docs/negative_results.md`.

- **Cheap seeds (importance-only)**: skip the span machinery; take the anchor
  plus the highest-mass candidates a fixed gap apart. Recovers only a quarter
  of adaptivity's benefit (0.09032) -- the expensive diversity computation is
  where most of the value lives.
- **Tightness reassignment**: after normal assignment, one Lloyd step moving
  each key to the nearest merged key, targeting the second-order error term
  (within-cluster key covariance). Loses (0.07393): moving a key off its
  max-likelihood seed loses first-order routed mass that outweighs the
  second-order gain. Hybrid interpolations degrade monotonically.
- **Capacity caps**: forbid any cluster from exceeding N keys, to prevent
  degenerate swallowing. A null: cap 30 never binds (bit-identical), cap 20
  binds and costs +0.7%.
- **MMR**: `lambda * importance - (1-lambda) * max-similarity-to-chosen`.
  Monotone in lambda, no interior optimum; even pure diversity is +3.0%
  worse than span. Dominated everywhere.
- **Facility location**: mass-weighted coverage of the candidate rows
  themselves (every query row should have a similar chosen seed). Has a real
  interior mass-temperature optimum, but its best point is +4.5% worse than
  span -- cosine coverage carries no implicit importance, while span's
  residual norm already does.
- **Dimension-pooled scoring**: pool the Q/K head dim for candidate scoring
  the way values are pooled. Destructive at any width (+12.9% at 64 dims):
  similarity is a discriminative signal and pooling collapses exactly the
  directional detail that separates candidates.

## External baselines: SVG2 and SVG-EAR

Different family: block-sparse attention over k-means clusters, not
compression. Both cluster queries into `qc` and keys into `kc` k-means
clusters per head (centroids persist across denoising steps and refine with
2 iterations per step), permute tokens so clusters are contiguous, then pick
which (query-cluster x key-cluster) blocks to compute.

- **SVG2 / SAP** selects blocks by centroid-vs-centroid attention under a
  top-p mass budget with a minimum-blocks floor, and computes the selected
  blocks exactly; unselected blocks are skipped.
- **SVG-EAR** ("error-aware") refines the budget estimate by weighting
  centroid attention with key-cluster sizes, and instead of skipping
  unselected blocks approximates them with their K/V centroids -- an exact
  token / centroid mixture, so no attention mass is fully discarded.

Their knob is `top_p` (block mass coverage); the hardcoded
`min_kc_ratio=0.05` at the call site floors density near 10-12%, which is
why EAR cannot reach GridMerge's 7.7% operating point. Measured on our
tensors in the live pipeline: SVG2 0.07581 at 33% density (dominated by our
production point on both axes); EAR 0.03512 at 36% down to 0.05810 at its
~12% floor -- ahead on quality wherever it can operate, at 1.6-4.7x the
density.

## Measurement harnesses

- **capture-16**: rel-L2 vs dense on a fixed real-QKV capture -- prompt 0,
  Wan2.1-1.3B base 480p, steps 0/6/13/19 x layers 0/10/20/29, heads 0/6,
  2 of 21 query frames. Every GridMerge arm saw identical inputs, so
  rankings are clean; absolute numbers are single-prompt screening values
  (the method-level findings -- adaptivity, the alpha optimum, the pool
  curve -- were separately replicated on 2-3 prompts in the 80-call sweeps).
- **pipeline**: rel-L2 computed inline in the live pipeline (dense on the
  same inputs each call), steps 0-1, all layers and heads. Used for
  SVG2/EAR because their processors carry cross-call k-means state that a
  standalone harness fails to reproduce. Cross-harness deltas are
  indicative, not exact.
- **Speed**: A100, 480p, 21 frames, 12 heads chunked by 4, min-of-N CUDA
  events through `token_merging_attn`. H200 numbers do not exist yet; 720p
  numbers on A100 are indicative only.
