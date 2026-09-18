#pragma once
#include <cstdint>

#ifdef __CUDACC__
#define GMA_OWNER_INLINE __host__ __device__ __forceinline__
#else
#define GMA_OWNER_INLINE inline
#endif

namespace gma_owner {
// Only call encode for non-NaN scores. The original strict > reduction
// ignores NaNs, initializes to -infinity/owner 0, and keeps the first tie.
GMA_OWNER_INLINE bool is_nan(uint32_t bits) {
    return (bits & 0x7fffffffu) > 0x7f800000u;
}
GMA_OWNER_INLINE uint64_t encode(uint32_t bits, unsigned owner) {
    const bool zero = (bits & 0x7fffffffu) == 0;
    const uint32_t canonical = zero ? 0u : bits;
    const uint32_t ordered = (canonical & 0x80000000u)
        ? ~canonical : canonical ^ 0x80000000u;
    // Owner priority dominates the sign bit; both zeros compare equal,
    // but decoding preserves the original sign of the winning zero.
    const uint32_t tie = ((10u - owner) << 1) | (zero ? bits >> 31 : 0u);
    return (uint64_t(ordered) << 32) | tie;
}
GMA_OWNER_INLINE unsigned owner(uint64_t key) {
    return 10u - (uint32_t(key) >> 1);
}
GMA_OWNER_INLINE uint32_t score_bits(uint64_t key) {
    const uint32_t ordered = uint32_t(key >> 32);
    uint32_t bits = (ordered & 0x80000000u)
        ? ordered ^ 0x80000000u : ~ordered;
    if (bits == 0u) bits |= uint32_t(key & 1u) << 31;
    return bits;
}
}
#undef GMA_OWNER_INLINE
