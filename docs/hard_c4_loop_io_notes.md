# Hard C4 Loop and IO Notes

This note records why the current one-pass hard-C4 prototype is not a speed
path, even though it avoids writing intermediate compressed K/V and routing
bias to HBM.

Default Wan 480p block-layout numbers used here:

```text
query_outer = key_outer = 21
rows = 30
cols = 52
heads = 12
dim = 128
sequence length = 21 * 30 * 52 = 32760
```

## FlashAttention Reference

The local FlashAttention-style Triton test kernel uses:

```text
BLOCK_M = 128 query tokens
BLOCK_N = 64 key/value tokens
```

For one head:

```text
query blocks = ceil(32760 / 128) = 256
key blocks per query block = ceil(32760 / 64) = 512
total block-loop iterations per head = 256 * 512 = 131072
```

The loop count is large, but each iteration is a regular tensor-core
`128 x 64 x 128` attention tile with good reuse:

```text
each Q token is loaded once by its query program
each K/V token is loaded once per query block, about 256 times per head
```

This kind of loop can be fast because the inner work is dense and regular.

## Maintained Two-Stage Hard C4

The maintained 480p fast path is
`attention/hard_c4_wan21_1_3b_480p_triton.py`.

Stage 1, merge:

```text
programs per head = query_outer * key_outer * rows = 21 * 21 * 30 = 13230
each program handles one (query_outer, key_outer, key_row) and all 52 columns
```

For one original K/V token, stage 1 reloads it once for each query outer block:

```text
merge-stage K/V reloads per original K/V token ~= 21
```

Stage 2, full compressed attention:

```text
programs per head = query_outer * cols = 21 * 52 = 1092
loop iterations per program = key_outer * ceil(rows / 32) = 21
```

This writes and rereads intermediate compressed tensors, but it preserves the
important reuse: one merge program serves all 52 query columns for that block.

## One-Pass J1 Prototype

The J1 prototype in `attention/hard_c4_onepass_triton.py` handles one query
column per program. For one head:

```text
programs per head = query_outer * cols = 21 * 52 = 1092
loop iterations per program = key_outer * rows = 21 * 30 = 630
K/V reloads per original K/V token ~= query_outer * cols = 1092
```

It removes intermediate HBM traffic, but destroys the stage-1 sharing across
query columns. This is why the synthetic H200 result was far slower than the
two-stage kernel.

## One-Pass J8 Prototype

The J8 prototype tries to recover some column sharing:

```text
J_TILE = 8
I_TILE = 4 in the best tested real-QKV version
programs per head = query_outer * ceil(cols / 8) * ceil(rows / 4)
                  = 21 * 7 * 8
                  = 1176
loop iterations per program = key_outer * rows = 630
K/V reloads per original K/V token ~= 1176
```

J8 avoids the NaN issue after the empty-token online-softmax fix, but it still
measured `324.71` ms/call on the first-three-call real Wan-QKV smoke, versus
`15.35` ms/call for maintained hard C4 and `14.53` ms/call for FlashAttention.

The core issue is not the presence of a `for` loop by itself. The problem is
the combination of:

```text
many serial key-row iterations inside each program
large live accumulators, especially [I_TILE, J_TILE, dim]
hard-route temporary scores and masks
much higher K/V reload count than both FlashAttention and two-stage hard C4
```

## Practical Rule

Loops are acceptable when they stream dense tiles with good data reuse, as in
FlashAttention. They are not acceptable when they move reuse from the grid level
into one oversized program and force K/V to be reread hundreds or thousands of
times.

For hard C4, a useful future fused kernel must satisfy both:

```text
avoid or reduce intermediate merged-K/V/bias HBM traffic
keep K/V reloads close to the maintained two-stage merge path, not the J1/J8 one-pass path
```

The current J1/J8 one-pass prototypes are therefore research/debugging code,
not production speed paths.
