// Kimi-K3 attention-residual bank partials, computed ahead of the read (ROCm, wave64).
//
// The bank rows only change at block-write layers, so most of a read's softmax mix is known
// before its stream is: for each token row t and score vector cw (score norm weight * score
// projection weight),
//     m = max_r s_r,  sum = sum_r exp(s_r - m),  acc = sum_r exp(s_r - m) * bank[t, r]   (fp32)
// over the bank rows r < nvb, s_r = <bank[t, r], cw> / rms(bank[t, r]). A read then folds in
// only the stream, as one more softmax candidate (read_hip.cuh, AttnResRead<0>). The route sort
// kernel (kimi_k3/route_sort.cuh) runs attn_res_partials() in its spare workgroups for the next
// layer's two reads.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstdint>

#ifndef USE_ROCM
#error "kimi_k3/attn_res/bank_partial_hip.cuh is a wave64 (ROCm) kernel"
#endif

namespace sglang {
namespace k3_attn_res_read {

inline constexpr int kHidden = 7168;
inline constexpr int kVec = 8;  // bf16 per thread: one 16-byte vector of the row
inline constexpr int kThreads = kHidden / kVec;
inline constexpr int kWave = 64;
inline constexpr int kWaves = kThreads / kWave;
inline constexpr int kMaxNvb = 8;

SGL_DEVICE float bf16_lo(uint32_t w) {
  return __uint_as_float(w << 16);
}

SGL_DEVICE float bf16_hi(uint32_t w) {
  return __uint_as_float(w & 0xffff0000u);
}

// Round to nearest even, as torch and Triton convert fp32 to bf16.
SGL_DEVICE uint32_t bf16_bits(float f) {
  const uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;
  return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}

SGL_DEVICE uint32_t pack_bf16(float lo, float hi) {
  return bf16_bits(lo) | (bf16_bits(hi) << 16);
}

SGL_DEVICE void unpack8(const uint4& v, float (&f)[kVec]) {
  f[0] = bf16_lo(v.x), f[1] = bf16_hi(v.x), f[2] = bf16_lo(v.y), f[3] = bf16_hi(v.y);
  f[4] = bf16_lo(v.z), f[5] = bf16_hi(v.z), f[6] = bf16_lo(v.w), f[7] = bf16_hi(v.w);
}

SGL_DEVICE uint4 pack8(const float (&f)[kVec]) {
  return make_uint4(pack_bf16(f[0], f[1]), pack_bf16(f[2], f[3]), pack_bf16(f[4], f[5]), pack_bf16(f[6], f[7]));
}

template <int kCtrl>
SGL_DEVICE float dpp(float v) {
  return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), kCtrl, 0xf, 0xf, false));
}

// Sum over the wave, uniform in every lane: DPP butterflies inside each 16-lane row
// (quad_perm [1,0,3,2], [2,3,0,1], row_half_mirror, row_mirror), then the four row sums
// via readlane. __shfl_xor would run every step through the LDS crossbar.
SGL_DEVICE float wave_sum(float v) {
  v += dpp<0xb1>(v);
  v += dpp<0x4e>(v);
  v += dpp<0x141>(v);
  v += dpp<0x140>(v);
  const int bits = __float_as_int(v);
  return (__int_as_float(__builtin_amdgcn_readlane(bits, 0)) + __int_as_float(__builtin_amdgcn_readlane(bits, 16))) +
         (__int_as_float(__builtin_amdgcn_readlane(bits, 32)) + __int_as_float(__builtin_amdgcn_readlane(bits, 48)));
}

SGL_DEVICE void load_fp32x8(const float4* src, float (&f)[kVec]) {
  const float4 a = src[0], b = src[1];
  f[0] = a.x, f[1] = a.y, f[2] = a.z, f[3] = a.w;
  f[4] = b.x, f[5] = b.y, f[6] = b.z, f[7] = b.w;
}

struct PartialParams {
  const bf16_t* __restrict__ bank;  // [T, NB, H] bf16
  int64_t bank_stride_t;            // elements
  int64_t bank_stride_b;
  const float4* cw[2];  // [H] fp32 each; cw[1] null: one score vector
  float4* acc[2];       // [T, H] fp32 each
  float2* ms[2];        // [T] (m, sum) each
  float score_eps;
  int32_t nvb;
  int32_t rows;  // 0: no partials
};

struct PartialShared {
  float part[kWaves][3 * kMaxNvb];  // per bank row: sum of squares, dot with cw[0], with cw[1]
  float weight[2][kMaxNvb];
  float2 ms[2];
};

// One workgroup of at least kThreads threads per token row t.
SGL_DEVICE void attn_res_partials(const PartialParams& p, PartialShared& s, int t) {
  const int tid = threadIdx.x;
  const int wave = tid / kWave;
  const int lane = tid % kWave;
  const bool active = tid < kThreads;
  const int nvb = p.nvb;
  const int sides = p.cw[1] != nullptr ? 2 : 1;
  uint4 bank[kMaxNvb];
  float cw[2][kVec] = {};
  // Every row loaded without a branch (rows past nvb repeat the last one, unused), so all
  // the loads are in flight at once; idle threads read the row's last vector.
  const bf16_t* row = p.bank + t * p.bank_stride_t + (active ? tid : kThreads - 1) * kVec;
#pragma unroll
  for (int r = 0; r < kMaxNvb; ++r)
    bank[r] = *reinterpret_cast<const uint4*>(row + (r < nvb ? r : nvb - 1) * p.bank_stride_b);
  const int cw_vec = 2 * (active ? tid : kThreads - 1);
  load_fp32x8(p.cw[0] + cw_vec, cw[0]);
  load_fp32x8(p.cw[sides - 1] + cw_vec, cw[1]);
  // Constant row indices only (no break): a runtime index would put bank[] in scratch memory.
#pragma unroll
  for (int r = 0; r < kMaxNvb; ++r) {
    if (r < nvb) {
      float f[kVec];
      unpack8(bank[r], f);
      float sq = 0.f, d0 = 0.f, d1 = 0.f;
#pragma unroll
      for (int e = 0; e < kVec; ++e)
        sq += f[e] * f[e], d0 += f[e] * cw[0][e], d1 += f[e] * cw[1][e];
      sq = wave_sum(sq);
      d0 = wave_sum(d0);
      d1 = wave_sum(d1);
      if (lane == 0 && wave < kWaves)
        s.part[wave][3 * r] = sq, s.part[wave][3 * r + 1] = d0, s.part[wave][3 * r + 2] = d1;
    }
  }
  __syncthreads();
  // Wave 0 scores, one 16-lane group per score vector and one lane per bank row.
  static_assert(kMaxNvb <= 16, "bank rows fit one 16-lane group");
  if (wave == 0) {
    const int side = lane / 16;
    const int r = lane % 16;
    const bool live = side < sides && r < nvb;
    float score = -INFINITY;
    if (live) {
      float sq = 0.f, dot = 0.f;
#pragma unroll
      for (int w = 0; w < kWaves; ++w)
        sq += s.part[w][3 * r], dot += s.part[w][3 * r + 1 + side];
      score = dot / sqrtf(sq / kHidden + p.score_eps);
    }
    float m = score;
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1)
      m = fmaxf(m, __shfl_xor(m, offset, kWave));
    const float e = live ? __expf(score - m) : 0.f;
    float sum = e;
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1)
      sum += __shfl_xor(sum, offset, kWave);
    if (live) s.weight[side][r] = e;
    if (r == 0 && side < sides) s.ms[side] = make_float2(m, sum);
  }
  __syncthreads();
  if (active) {
    for (int side = 0; side < sides; ++side) {
      float acc[kVec] = {};
#pragma unroll
      for (int r = 0; r < kMaxNvb; ++r) {
        if (r < nvb) {
          float f[kVec];
          unpack8(bank[r], f);
          const float w = s.weight[side][r];
#pragma unroll
          for (int e = 0; e < kVec; ++e)
            acc[e] += w * f[e];
        }
      }
      float4* dst = p.acc[side] + (static_cast<int64_t>(t) * kThreads + tid) * 2;
      dst[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
      dst[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
    }
  }
  if (tid < sides) p.ms[tid][t] = s.ms[tid];
}

}  // namespace k3_attn_res_read
}  // namespace sglang
