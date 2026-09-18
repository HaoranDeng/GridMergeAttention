// CPU-only semantic gate for the exact header used by the CUDA candidate.
// This does not validate GPU synchronization, output, occupancy or timing.
#include "../native_fisher/owner_key.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>

static float as_float(uint32_t bits) {
    float result;
    std::memcpy(&result, &bits, sizeof result);
    return result;
}
int main() {
    std::mt19937 rng(20260909);
    uint64_t cases = 0;
    auto check = [&](const std::array<uint32_t, 10>& scores) {
        uint32_t best = 0xff800000u;
        unsigned owner = 0;
        for (unsigned i = 0; i < 10; ++i) {
            if (as_float(scores[i]) > as_float(best)) {
                best = scores[i]; owner = i;
            }
        }
        std::array<unsigned, 10> order = {0,1,2,3,4,5,6,7,8,9};
        std::shuffle(order.begin(), order.end(), rng);
        uint64_t key = gma_owner::encode(0xff800000u, 0);
        for (unsigned i : order) {
            if (!gma_owner::is_nan(scores[i]))
                key = std::max(key, gma_owner::encode(scores[i], i));
        }
        if (gma_owner::score_bits(key) != best || gma_owner::owner(key) != owner)
            throw std::runtime_error("owner/score bit mismatch");
        ++cases;
    };
    const uint32_t edge[] = {0u,0x80000000u,1u,0x80000001u,
        0x007fffffu,0x807fffffu,0x00800000u,0x80800000u,
        0x3f800000u,0xbf800000u,0x7f7fffffu,0xff7fffffu,
        0x7f800000u,0xff800000u,0x7fc00000u,0xffc00000u,0x7f800001u};
    std::array<uint32_t,10> scores;
    for (auto a : edge) for (auto b : edge) for (auto c : edge) {
        scores.fill(0xff800000u);
        scores[0]=a; scores[4]=b; scores[9]=c; check(scores);
    }
    for (auto a : edge) {
        scores.fill(a); check(scores);
        for (unsigned i=0; i<10; ++i) for (unsigned j=i+1; j<10; ++j) {
            scores.fill(0xffc00000u); scores[i]=a; scores[j]=a; check(scores);
            scores[i]=0u; scores[j]=0x80000000u; check(scores);
            scores[i]=0x80000000u; scores[j]=0u; check(scores);
        }
    }
    for (unsigned trial=0; trial<250000; ++trial) {
        for (auto& bits : scores) bits=rng();
        if (trial%3==0) scores[trial%10]=scores[(trial+3)%10];
        check(scores);
    }
    std::cout << "{\"valid\":true,\"scope\":\"CPU score/owner encoding only\","
              << "\"cases\":" << cases << ",\"mismatches\":0,"
              << "\"random_seed\":20260909,\"gpu_validated\":false}\n";
}
