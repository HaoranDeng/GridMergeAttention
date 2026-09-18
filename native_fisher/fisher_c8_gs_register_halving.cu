// Exact Fisher C8 Gram--Schmidt with Hellinger-style recursive halving.
//
// The Fisher Gram--Schmidt arithmetic is deliberately FP32.  FP64 remains a
// comparison-only path in fisher_gs_register_distribute.cu.  The P=4 layout
// only moves values that a
// lane ultimately consumes:
//   * probe centers use 4 shuffles instead of four independent 3-shuffle
//     all-reductions;
//   * each lane owns one candidate score after two halving levels;
//   * warp-wide FP32 maxima and leftmost argmax use Hopper REDUX operations;
//   * eight uint8 seeds are committed with one aligned uint64 store.
// No singleton/raw/residual route is present: this kernel returns exactly C8.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>

__device__ __forceinline__ float register_select(
    float previous, float value, bool take) {
  float out;
  asm("{ .reg .pred p; setp.ne.u32 p,%3,0; selp.f32 %0,%2,%1,p; }"
      : "=f"(out) : "f"(previous), "f"(value), "r"(int(take)));
  return out;
}

template<int N>
__device__ __forceinline__ float sum_registers(float (&values)[N]) {
  #pragma unroll
  for (int distance = N / 2; distance; distance /= 2) {
    #pragma unroll
    for (int index = 0; index < distance; ++index) {
      values[index] = __fadd_rn(values[index], values[index + distance]);
    }
  }
  return values[0];
}

__device__ __forceinline__ float halve_add(
    float tag0, float tag1, bool mine, int width) {
  float keep = mine ? tag1 : tag0;
  float send = mine ? tag0 : tag1;
  return __fadd_rn(
      keep, __shfl_xor_sync(0xffffffffu, send, width));
}

template<int N>
__device__ __forceinline__ float sum_probe16(float (&values)[N]) {
  static_assert(N == 4, "P4 exact Fisher C8 expects four local probes");
  float total = sum_registers(values);
  total = __fadd_rn(total, __shfl_xor_sync(0xffffffffu, total, 2));
  total = __fadd_rn(total, __shfl_xor_sync(0xffffffffu, total, 1));
  return total;
}

__global__ void fisher_c8_gs(
    const float* logits,
    const int* candidates,
    const int* anchors,
    uint8_t* seeds,
    int groups,
    float scale) {
  constexpr int P = 4;
  constexpr int CS = 8;
  constexpr int PR = 4;
  const int lane = threadIdx.x & 31;
  const int probe = lane & 3;
  const int slot = lane >> 2;
  const int group = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (group >= groups) return;

  const int key_row = (group / (80 * 40)) % 45;
  const int anchor = anchors[key_row];
  const int candidate_row = candidates[key_row * 32 + lane];

  float z[P][PR];
  #pragma unroll
  for (int p = 0; p < PR; ++p) {
    #pragma unroll
    for (int candidate = 0; candidate < P; ++candidate) {
      const int64_t offset =
          (int64_t(group) * 32 + candidate * CS + slot) * 16
          + p * P + probe;
      z[candidate][p] = __fmul_rn(logits[offset], scale);
    }
  }

  // Preserve the original candidate tree, then recursively halve the four
  // center tags while applying the original xor-16/xor-8/xor-4 slot tree.
  float partial[PR];
  #pragma unroll
  for (int p = 0; p < PR; ++p) {
    float terms[P];
    #pragma unroll
    for (int candidate = 0; candidate < P; ++candidate) {
      terms[candidate] = z[candidate][p];
    }
    partial[p] = sum_registers(terms);
  }
  const bool high_tag = ((lane >> 4) & 1) != 0;
  const bool low_tag = ((lane >> 3) & 1) != 0;
  float high_even = halve_add(partial[0], partial[2], high_tag, 16);
  float high_odd = halve_add(partial[1], partial[3], high_tag, 16);
  float selected_center = halve_add(high_even, high_odd, low_tag, 8);
  selected_center = __fadd_rn(
      selected_center,
      __shfl_xor_sync(0xffffffffu, selected_center, 4));
  selected_center = __fmul_rn(selected_center, 1.0f / 32);

  int maximum_key = __float_as_int(selected_center);
  maximum_key ^= (maximum_key >> 31) & 0x7fffffff;
  maximum_key = __reduce_max_sync(0xffffffffu, maximum_key);
  maximum_key ^= (maximum_key >> 31) & 0x7fffffff;
  const float maximum = __int_as_float(maximum_key);

  float exponential = expf(__fadd_rn(selected_center, -maximum));
  float denominator = exponential;
  denominator = __fadd_rn(
      denominator, __shfl_xor_sync(0xffffffffu, denominator, 16));
  denominator = __fadd_rn(
      denominator, __shfl_xor_sync(0xffffffffu, denominator, 8));
  denominator = __fadd_rn(
      denominator, __shfl_xor_sync(0xffffffffu, denominator, 2));
  denominator = __fadd_rn(
      denominator, __shfl_xor_sync(0xffffffffu, denominator, 1));
  const float normalized = __fdiv_rn(exponential, denominator);
  const float owned_root = sqrtf(normalized);

  float rho[PR], root[PR], anchor_z[PR];
  #pragma unroll
  for (int p = 0; p < PR; ++p) {
    const int source_lane = p * 8 + probe;
    rho[p] = __shfl_sync(0xffffffffu, normalized, source_lane);
    root[p] = __shfl_sync(0xffffffffu, owned_root, source_lane);
    float value = 0;
    #pragma unroll
    for (int candidate = 0; candidate < P; ++candidate) {
      value = register_select(
          value, z[candidate][p], candidate == anchor / CS);
    }
    anchor_z[p] = __shfl_sync(
        0xffffffffu, value, (anchor % CS) * P + probe);
  }

  float residual[P][PR];
  #pragma unroll
  for (int candidate = 0; candidate < P; ++candidate) {
    float delta[PR], terms[PR];
    #pragma unroll
    for (int p = 0; p < PR; ++p) {
      delta[p] = __fadd_rn(z[candidate][p], -anchor_z[p]);
      terms[p] = __fmul_rn(delta[p], rho[p]);
    }
    float mean = sum_probe16(terms);
    #pragma unroll
    for (int p = 0; p < PR; ++p) {
      residual[candidate][p] = __fmul_rn(
          __fadd_rn(delta[p], -mean), root[p]);
    }
  }

  unsigned used = 1u << anchor;
  uint64_t packed = uint64_t(
      __shfl_sync(0xffffffffu, candidate_row, anchor) & 0xff);
  const bool score_high = ((lane >> 1) & 1) != 0;
  const bool score_low = (lane & 1) != 0;
  const int own_index = probe * CS + slot;

  #pragma unroll
  for (int step = 1; step < 8; ++step) {
    float score_parts[P];
    #pragma unroll
    for (int candidate = 0; candidate < P; ++candidate) {
      float terms[PR];
      #pragma unroll
      for (int p = 0; p < PR; ++p) {
        terms[p] = __fmul_rn(
            residual[candidate][p], residual[candidate][p]);
      }
      score_parts[candidate] = sum_registers(terms);
    }
    float score_high_even = halve_add(
        score_parts[0], score_parts[2], score_high, 2);
    float score_high_odd = halve_add(
        score_parts[1], score_parts[3], score_high, 2);
    float score = halve_add(
        score_high_even, score_high_odd, score_low, 1);

    const bool valid = !(used & (1u << own_index)) && score == score;
    const unsigned score_key = valid ? __float_as_uint(score) : 0u;
    const unsigned best_key = __reduce_max_sync(0xffffffffu, score_key);
    const int picked = int(__reduce_min_sync(
        0xffffffffu,
        score_key == best_key && valid ? unsigned(own_index) : 32u));
    const float best = __uint_as_float(best_key);

    used |= 1u << picked;
    packed |= uint64_t(
        __shfl_sync(0xffffffffu, candidate_row, picked) & 0xff)
        << (8 * step);

    if (step < 7) {
      const float norm = sqrtf(__fadd_rn(best, 1.e-12f));
      float basis[PR];
      #pragma unroll
      for (int p = 0; p < PR; ++p) {
        float value = 0;
        #pragma unroll
        for (int candidate = 0; candidate < P; ++candidate) {
          value = register_select(
              value,
              residual[candidate][p],
              candidate == picked / CS);
        }
        basis[p] = __shfl_sync(
            0xffffffffu, value, (picked % CS) * P + probe);
      }

      const int subset = (lane >> 3) & 3;
      float owned_basis = 0;
      #pragma unroll
      for (int p = 0; p < PR; ++p) {
        owned_basis = register_select(
            owned_basis, basis[p], subset == p);
      }
      owned_basis = __fdiv_rn(owned_basis, norm);
      #pragma unroll
      for (int p = 0; p < PR; ++p) {
        basis[p] = __shfl_sync(
            0xffffffffu, owned_basis, p * 8 + probe);
      }

      #pragma unroll
      for (int candidate = 0; candidate < P; ++candidate) {
        float terms[PR];
        #pragma unroll
        for (int p = 0; p < PR; ++p) {
          terms[p] = __fmul_rn(residual[candidate][p], basis[p]);
        }
        const float projection = sum_probe16(terms);
        #pragma unroll
        for (int p = 0; p < PR; ++p) {
          residual[candidate][p] = __fmaf_rn(
              -projection, basis[p], residual[candidate][p]);
        }
      }
    }
  }

  if (lane == 0) {
    *reinterpret_cast<uint64_t*>(seeds + int64_t(group) * 8) = packed;
  }
}

void run(
    torch::Tensor logits,
    torch::Tensor candidates,
    torch::Tensor anchors,
    torch::Tensor seeds,
    float scale,
    int probe_lanes) {
  TORCH_CHECK(
      logits.is_cuda() && logits.scalar_type() == at::kFloat
          && logits.is_contiguous(),
      "Contiguous CUDA FP32 logits required");
  TORCH_CHECK(
      candidates.scalar_type() == at::kInt && anchors.scalar_type() == at::kInt,
      "Int32 metadata required");
  TORCH_CHECK(
      seeds.scalar_type() == at::kByte && seeds.is_contiguous(),
      "Contiguous uint8 seeds required");
  constexpr int groups = 21 * 45 * 80 * 40;
  TORCH_CHECK(
      logits.numel() == int64_t(groups) * 32 * 16
          && seeds.numel() == int64_t(groups) * 8,
      "Exact Fisher C8 shapes required");
  TORCH_CHECK(probe_lanes == 4, "Exact Fisher C8 halving requires P4");
  auto stream = at::cuda::getCurrentCUDAStream();
  fisher_c8_gs<<<(groups + 3) / 4, 128, 0, stream>>>(
      logits.data_ptr<float>(),
      candidates.data_ptr<int>(),
      anchors.data_ptr<int>(),
      seeds.data_ptr<uint8_t>(),
      groups,
      scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("run", &run);
}
