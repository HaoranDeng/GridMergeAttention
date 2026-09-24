# GCA Related Work Notes (2026-09-24)

This note records a focused, non-exhaustive prior-art search for Grid
Compressed Attention (GCA). It is intended as a starting point for a paper's
related-work section, not as a patentability opinion or a claim that every
relevant publication has been found.

## Current GCA operator

For each video-grid key block, GCA:

1. selects exactly `C` seed queries from real model queries;
2. lets every seed define a key-normalized distribution;
3. hard-assigns every key to one seed by comparing those normalized scores;
4. softmax-merges K and V inside each hard cluster;
5. represents each cluster by `(merged K, merged V, entropy bias)`; and
6. runs full attention from every original query to every compressed route.

For C4 and C8, `C` is the complete capacity. There is no additional raw,
singleton, SQ1, or residual K/V route. The maintained path is training-free
for pretrained video diffusion transformers and is implemented with
shape-specific CUDA/Triton kernels.

The broad idea "use a small number of query-like representatives to compress
attention" is not new. The most defensible novelty, if supported by a fuller
literature review and experiments, is the combination of real-query seed
selection, key-softmax hard ownership, exact capacity, video-grid structure,
training-free replacement, and production kernels.

## Closest operator-level work

### Multipole Semantic Attention (MuSe)

- Rupert Mitchell and Kristian Kersting, *Multipole Semantic Attention: A
  Fast Approximation of Softmax Attention for Pretraining*, arXiv:2509.10406,
  first posted 2025. [Paper](https://arxiv.org/html/2509.10406)
- MuSe clusters queries and keys separately. A query-cluster centroid induces
  exponential weights over keys, from which MuSe constructs weighted K and V
  summaries plus their log attention mass. Fine queries then attend to those
  summaries, with optional exact retrieval.
- This is the closest prior work to the complete GCA summary operator: it has
  query-conditioned, softmax-weighted merged K, merged V, mass correction, and
  a second attention stage.
- GCA differs by using real query seeds rather than K-means query centroids,
  assigning every key to exactly one seed through cross-seed hard competition,
  enforcing exact `C` representatives per video-grid block, and using the
  compressed routes directly without MuSe's retrieval hierarchy.

### HiLS-Attention

- Xiang Hu et al., *Hierarchical Sparse Attention Done Right: Toward Infinite
  Context Modeling*, arXiv:2607.02980, 2026.
  [Paper](https://arxiv.org/html/2607.02980v1)
- A landmark query induces `p_j = softmax(q_landmark^T k_j)` inside a chunk.
  HiLS forms a compressed key `k' = sum_j p_j k_j` and entropy bias
  `H(p) = -sum_j p_j log p_j`, scoring a chunk by `q^T k' + H(p)`.
- This is essentially the same entropy-calibrated compressed-key equation used
  by GCA. HiLS also derives it as a first-order Taylor approximation to the
  chunk log-sum-exp, so that derivation should not be claimed as new here.
- HiLS is nevertheless a different end-to-end operator: it is a trainable,
  causal LLM chunk-retrieval method. It uses the summary to select chunks and
  then performs fine-grained attention within retrieved chunks. GCA instead
  hard-partitions and merges both K and V, then lets every original query
  attend directly to all fixed-capacity representatives.

### Set Transformer

- Juho Lee et al., *Set Transformer: A Framework for Attention-based
  Permutation-Invariant Neural Networks*, ICML 2019.
  [Paper](https://proceedings.mlr.press/v97/lee19d.html)
- Its induced set attention block first lets a small learned inducing set read
  all input elements, then lets all elements read the induced representation.
  This established the general two-stage `inducing queries -> summaries ->
  original queries` pattern and linear-in-sequence complexity for fixed
  inducing capacity.
- GCA uses input-dependent real queries, hard key ownership, explicit K/V
  moments, entropy correction, and grid-local exact capacity rather than
  learned global inducing points.

## Video-generation baselines

### MonarchRT

- Krish Agarwal et al., *MonarchRT: Efficient Attention for Real-Time Video
  Generation*, arXiv:2602.12271, 2026.
  [Paper](https://arxiv.org/abs/2602.12271)
- MonarchRT and GCA address the same application-level problem: accelerating
  large 3D self-attention in few-step and causal video diffusion, including
  Self-Forcing, with custom kernels and end-to-end video-quality evaluation.
- The mathematical mechanisms are different. MonarchRT parameterizes the
  attention matrix with spatiotemporally aligned tiled Monarch factors. GCA
  instead compresses the K/V support with query-induced hard clusters and
  subsequently performs full attention over the compressed support.
- Consequently MonarchRT is a primary experimental/system baseline, but it is
  not the closest prior art for GCA's merged-K/V equations.

### AdaCluster

- Haoyue Tan et al., *AdaCluster: Adaptive Query-Key Clustering for Sparse
  Attention in Video Generation*, arXiv:2604.18348, 2026.
  [Paper](https://arxiv.org/abs/2604.18348)
- AdaCluster is a training-free video-DiT method that clusters queries and keys
  separately and uses adaptive sparse attention. It is close in task, use of
  query/key geometry, and target models such as HunyuanVideo and Wan.
- It selects sparse interactions rather than replacing every key block by
  exactly `C` entropy-calibrated merged K/V representatives. It should still be
  included as a direct video-generation comparison.

## Token and KV merging

### Token Merging (ToMe)

- Daniel Bolya et al., *Token Merging: Your ViT But Faster*, ICLR 2023.
  [Paper](https://arxiv.org/html/2210.09461v3)
- ToMe matches similar tokens, merges them by weighted averaging, and adds
  `log(cluster size)` to later attention logits (proportional attention) so a
  merged token retains the mass of the tokens it represents.
- GCA does not progressively shorten the residual token stream. It merges only
  K/V inside an attention call, keeps every query/output token, uses
  query-induced hard routing, and replaces the fixed `log(size)` correction
  with the entropy of the within-cluster softmax. For uniform weights, GCA's
  entropy correction reduces to the same `log(cluster size)` principle.

### KVMerger

- Zheng Wang et al., *Model Tells You Where to Merge: Adaptive KV Cache
  Merging for LLMs on Long-Context Tasks*, arXiv:2407.08454, 2024.
  [Paper](https://arxiv.org/abs/2407.08454)
- KVMerger identifies similar KV-cache states using key-space clustering and
  combines them with Gaussian-kernel weights.
- It targets persistent autoregressive KV-cache memory and clusters in key
  space. GCA recomputes query-conditioned grid-local summaries for an
  attention call and preserves an exact fixed route count.

### WeightedKV

- Jian Yuan et al., *WeightedKV: Attention Scores Weighted Key-Value Cache
  Merging for Large Language Models*, arXiv:2503.01330, 2025.
  [Paper](https://arxiv.org/abs/2503.01330)
- WeightedKV uses attention importance to retain anchor keys and merge values
  from less important cache entries.
- GCA jointly constructs new merged keys and values for every hard cluster and
  uses an entropy-calibrated compressed attention stage; it does not retain an
  anchor/raw-key side path.

## Clustering, routing, and landmark approximations

### Fast Transformers with Clustered Attention

- Apoorv Vyas, Angelos Katharopoulos, and Francois Fleuret, *Fast
  Transformers with Clustered Attention*, NeurIPS 2020.
  [Paper](https://arxiv.org/abs/2007.04825)
- This method clusters queries, computes attention for query centroids, and can
  refine selected query-key interactions exactly.
- It demonstrates query prototypes for efficient attention, but primarily
  shares attention computations across similar queries. GCA instead uses seed
  queries to partition and merge K/V while retaining all original queries in
  the second stage.

### Routing Transformer

- Aurko Roy et al., *Efficient Content-Based Sparse Attention with Routing
  Transformers*, TACL 2021 / arXiv:2003.05997.
  [Paper](https://arxiv.org/abs/2003.05997)
- Routing Transformer uses online K-means to place tokens into content-based
  buckets and restrict attention to compatible buckets.
- It establishes content-dependent clustering/routing as prior art, but drops
  cross-bucket interactions. GCA preserves a full second-stage attention over
  all compressed routes.

### Nystromformer

- Yunyang Xiong et al., *Nystromformer: A Nystrom-Based Algorithm for
  Approximating Self-Attention*, AAAI 2021.
  [Paper](https://arxiv.org/abs/2102.03902)
- Nystromformer uses a small set of landmarks to construct a low-rank
  approximation of the attention matrix.
- GCA's seeds are routing/expansion queries rather than Nystrom landmarks, and
  GCA explicitly forms hard-cluster K/V moments with entropy mass correction.

## Positioning summary

The closest work depends on the comparison axis:

| Comparison axis | Closest work | Main GCA distinction |
| --- | --- | --- |
| Video generation and systems | MonarchRT, AdaCluster | Exact-capacity query-induced K/V aggregation rather than matrix factorization or sparse selection |
| Complete compressed-K/V operator | MuSe | Real-query seeds, normalized hard ownership, video-grid blocks, no retrieval hierarchy |
| Compressed key and entropy bias | HiLS | Merge V and directly route all queries over all representatives; training-free video setting |
| Two-stage inducing architecture | Set Transformer | Input-dependent real seeds and hard clusters rather than learned inducing points |
| Token merging and mass preservation | ToMe | Per-attention K/V merging, entropy correction, unchanged query/output length |
| Content-based routing | Clustered Attention, Routing Transformer | Full attention over compressed routes rather than shared-Q computation or sparse buckets |

For a paper, avoid broad claims such as "the first query-based K/V
compression" or "the first entropy-corrected compressed key." A narrower and
more supportable positioning is:

> GCA is a training-free, exact-capacity, query-induced hard K/V aggregation
> method for grid-structured video diffusion attention, together with
> shape-specialized kernels and end-to-end video validation.

Whether even this combined claim is first requires a broader scholarly and
patent search, plus a version-by-version comparison of the closest papers.
