#pragma once
#include <cuda_bf16.h>
#include <cstdint>

// Warp-local primitives for the experimental merge/route fusion. These are
// not used by the retained attention operator until full-call validation.
namespace fisher_warp {
using bf16 = __nv_bfloat16;

__device__ __forceinline__ uint32_t pair(bf16 x, bf16 y) {
  union { __nv_bfloat162 value; uint32_t bits; } p;
  p.value = __halves2bfloat162(x, y);
  return p.bits;
}

__device__ __forceinline__ void mma(const uint32_t (&a)[4],
    const uint32_t (&b)[2], float (&c)[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]));
}

// K/V buffer uses the same 128-byte swizzle as the retained merge kernel.
__device__ __forceinline__ int raw_offset(int k, int d) {
  int offset = k * 64 + (d & 63) + (d >> 6) * 5120;
  return offset ^ (((offset >> 6) & 7) << 3);
}

__device__ __forceinline__ void score_b(const bf16* raw, int key_panel,
    int dim_panel, int lane, uint32_t (&b)[2]) {
  int key = key_panel + (lane >> 2), d = dim_panel + 2 * (lane & 3);
  b[0] = pair(raw[raw_offset(key, d)], raw[raw_offset(key, d + 1)]);
  b[1] = pair(raw[raw_offset(key, d + 8)], raw[raw_offset(key, d + 9)]);
}

__device__ __forceinline__ void merge_b(const bf16* raw, int key_panel,
    int dim_panel, int lane, uint32_t (&b)[2]) {
  int key = key_panel + 2 * (lane & 3), d = dim_panel + (lane >> 2);
  b[0] = pair(raw[raw_offset(key, d)], raw[raw_offset(key + 1, d)]);
  b[1] = pair(raw[raw_offset(key + 8, d)], raw[raw_offset(key + 9, d)]);
}
}  // namespace fisher_warp
