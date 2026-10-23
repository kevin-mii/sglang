// gfx950 fp8 MLA prefill forward, v3: Q bf16 (cast to fp8 in the kernel), K/V fp8 e4m3fn with per-tensor dequant
// scales, O bf16 (and optional LSE). Q/K head dim 192, V/O 128. C API and torch bindings as mla_fp8.hip (v1).
//
// From moonmath-ai/amd-kernels' gfx950 attention_mla.hip (MIT, see LICENSE.moonmath): 8 waves per CU, K/V moved
// by DMA into swizzled LDS rings, lazy-rescale frames, and the host planner, work list, KV splits and merge
// (attention_plan.hpp). The fp8 parts are v1's ([32x32], [p8], [log2], [frame], [rsum], [lds8] in mla_fp8.hip).
// What v3 changes, tagged:
//   [self]   On gfx950 a wave streaming MFMAs starves the other wave on its SIMD of VALU issue, so v1's groups
//            (one wave's MFMA block beside the other's softmax) serialize. But a wave's own VALU runs beside its
//            own in-flight MFMAs. So every wave pipelines itself, tile by tile:
//              phase X(t): QK(t + 1) into the other S buffer   ||  exps of S(t) to fp8 P(t)
//              phase Y(t): row sums + PV(t) over V(t)          ||  lane max of S(t + 1), then its frame test
//            Measured: ~6 VALU per 32x32x64 MFMA keep the MFMA pipe ~80% busy with 2 waves per SIMD.
//   [pair]   Groups now only split the DMA (A: K, B: V). One barrier per pair of tiles (t = 2p, 2p + 1), at the
//            pair's start: K and V rings 4 deep; pair p reads K(t + 1), K(t + 2), V(t), V(t + 1) and loads
//            K(t + 3), K(t + 4), V(t + 2), V(t + 3) into the buffers pair p - 1 read. A wave's skew spreads over
//            two tiles, and half the barriers.

#include <hip/hip_runtime.h>
#include "attention_plan.hpp"

namespace attn::mla8 {

// Experiment switches (profiling only): no exps, every DMA rereads tile 0, no QK/PV MFMAs.
#ifndef MLA8_NO_EXP
#define MLA8_NO_EXP 0
#endif
#ifndef MLA8_SAME_TILE
#define MLA8_SAME_TILE 0
#endif
#ifndef MLA8_NO_MFMA
#define MLA8_NO_MFMA 0
#endif
// MLA8_TIMING: per-phase cycle totals over all waves (s_memtime), read with mla_fp8_timing().
#ifndef MLA8_TIMING
#define MLA8_TIMING 0
#endif
#ifndef MLA8_NO_KREAD
#define MLA8_NO_KREAD 0
#endif
#ifndef MLA8_NO_VREAD
#define MLA8_NO_VREAD 0
#endif
__device__ unsigned long long timing_acc[8];

using i32x2_t  = __attribute__((ext_vector_type(2))) int;
using i32x4_t  = __attribute__((ext_vector_type(4))) int;
using i32x8_t  = __attribute__((ext_vector_type(8))) int;
using f32x16_t = __attribute__((ext_vector_type(16))) float;
using u8       = uint8_t;

constexpr int Dqk = 192, Dv = 128;
constexpr int Threads = BlockSize, GroupWaves = NumWaves / 2;   // 8 waves; group A moves K, group B V
constexpr int TileRows = 32;                       // one 32-row q tile per wave
constexpr int KSteps = Dqk / 64, KvMTiles = KvTile / 32, DMTiles = Dv / 32;
constexpr int KFrags = KvMTiles * KSteps;          // K fragments of a tile: (mt, ks)
constexpr float FrameTop = 7.0f, RescaleAt = 8.5f, Fp8Max = 448.0f;

// LDS [lds8] [dma2]: K tiles of 12 KB (3 ks blocks of 4 KB), V tiles of 8 KB (4 d blocks of 2 KB), 3 deep each.
constexpr int KTileBytes = KvTile * Dqk, VTileBytes = KvTile * Dv;
constexpr int KBufs = 4, VBufs = 4, KAhead = 3, VAhead = 2;   // [pair] rings 4 deep
constexpr int KRegion = KBufs * KTileBytes, VRegion = VBufs * VTileBytes;
constexpr int LdsBytes = KRegion + VRegion;
static_assert(LdsBytes <= 163840, "gfx950 allows 160 KB");
constexpr int k_buf(int t) { return (unsigned)t % KBufs * KTileBytes; }
constexpr int v_buf(int t) { return KRegion + (unsigned)t % VBufs * VTileBytes; }
constexpr int KDmas = KTileBytes / 1024 / GroupWaves, VDmas = VTileBytes / 1024 / GroupWaves;   // 3, 2

// ---- instruction wrappers ------------------------------------------------------------------------
__device__ __forceinline__ f32x16_t mfma32(i32x8_t a, i32x8_t b, f32x16_t c)
{
    return __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, b, c, 0, 0, 0, 0, 0, 0);
}
// A and B times 2^(scale - 127); VGPR scales only (SGPR scales miscompile on this ROCm).
__device__ __forceinline__ f32x16_t mfma32_scaled(i32x8_t a, i32x8_t b, f32x16_t c, int scale_a, int scale_b)
{
    return __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, b, c, 0, 0, 0, scale_a, 0, scale_b);
}
using lds_u8_ptr = __attribute__((address_space(3))) u8*;
__device__ __forceinline__ i32x4_t ds_read_b128(const u8* lds_base, uint32_t off)
{
    return *reinterpret_cast<const __attribute__((address_space(3))) i32x4_t*>((lds_u8_ptr)lds_base + off);
}
__device__ __forceinline__ i32x2_t ds_read_tr8(const u8* lds_base, uint32_t off)
{
    return __builtin_amdgcn_ds_read_tr8_b64_v2i32(
        (__attribute__((address_space(3))) i32x2_t*)((lds_u8_ptr)lds_base + off));
}
__device__ __forceinline__ int cvt_pk_fp8_lo(float a, float b) { return __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false); }
__device__ __forceinline__ int cvt_pk_fp8_hi(float a, float b, int old)
{
    return __builtin_amdgcn_cvt_pk_fp8_f32(a, b, old, true);
}
__device__ __forceinline__ float clamp_fp8(float x) { return __builtin_amdgcn_fmed3f(x, -Fp8Max, Fp8Max); }
// The max of x over lanes L and L ^ 32; s_nop 1 covers the VALU write -> permlane read hazard.
__device__ __forceinline__ float max_xor32(float x)
{
    float t;
    asm volatile("v_mov_b32 %1, %0\n\ts_nop 1\n\tv_permlane32_swap_b32 %0, %1 bound_ctrl:1" : "+v"(x), "=&v"(t));
    return fmaxf(x, t);
}


// ---- the kernel ----------------------------------------------------------------------------------
template <bool Split>
__global__ __launch_bounds__(Threads, 1) void fwd(Args a)
{
    __shared__ __attribute__((aligned(16))) u8 lds[LdsBytes];
    const int lane = threadIdx.x & (WaveSize - 1);
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / WaveSize);
    const int h = lane >> 5, r = lane & 31;        // MFMA lane view: half h, row/column r
    const bool group_a = wave < GroupWaves;
    const int gw = wave % GroupWaves;

    const WaveWork work = wave_work<Split>(a, wave, TileRows);
    if (work.sq == 0) return;

    // ==== Q: bf16 rows to fp8 B fragments [log2] =================================================
    __shared__ LseRows lse_stash[NumWaves];
    if constexpr (Split) {
        const LseRows lse = lse_rows(a, work);
        if (lane == 0) lse_stash[wave] = lse;
    }
    const float q_mul_full = a.scale_log2e * a.k_scale;
    const int q_exp = max(-100, min(100, (int)__builtin_rintf(-__builtin_log2f(q_mul_full))));
    const float q_mul = __builtin_ldexpf(q_mul_full, q_exp);       // in [0.7, 1.4]
    int scale_k = 127, scale_q = 127 - q_exp;                        // QK's E8M0 scales, VGPRs
    pin_after(scale_k), pin_after(scale_q);
    i32x8_t q_frag[KSteps];                        // lane (h, r): Q[row0 + r][64 ks + 32 h + s]
    {
        const int q = work.row0 + r;
        const bool valid = (unsigned)q < (unsigned)work.sq;
        const bf16_t* row = row_at(qo_rows(a.q, a.q_str, work), a.q_str, valid ? q : 0);
        for (int ks = 0; ks < KSteps; ++ks)
            for (int u = 0; u < 4; ++u) {
                bf16x8_t x = *reinterpret_cast<const bf16x8_t*>(row + 64 * ks + 32 * h + 8 * u);
                if (!valid) x = bf16x8_t{};
                float f[8];
                for (int e = 0; e < 8; ++e) f[e] = clamp_fp8((float)x[e] * q_mul);
                q_frag[ks][2 * u] = cvt_pk_fp8_hi(f[2], f[3], cvt_pk_fp8_lo(f[0], f[1]));
                q_frag[ks][2 * u + 1] = cvt_pk_fp8_hi(f[6], f[7], cvt_pk_fp8_lo(f[4], f[5]));
            }
    }

    // ==== K/V DMA [lds8] [dma2] ===================================================================
    // A wave gw, instruction i (ks = i): K rows 16 gw + L / 4, logical chunk (L % 4) ^ (L / 16), into
    //   k_buf + 4096 i + 1024 gw (+16 L).
    // B wave gw (d block gw), instruction i: V row 32 i + 8 (L / 16) + ((L / 2 % 8) ^ 4 (L / 16 % 2)),
    //   d 32 gw + 16 (L % 2), into v_buf + 2048 gw + 1024 i (+16 L).
    const uint32_t k_row_bytes = (uint32_t)a.k_str.token, v_row_bytes = (uint32_t)a.v_str.token;
    const u8* const dma_base = group_a ? (const u8*)a.k + offset(a.k_str, work.b, (uint32_t)work.kv_row0, work.kv_head)
                                       : (const u8*)a.v + offset(a.v_str, work.b, (uint32_t)work.kv_row0, work.kv_head);
    const uint32_t row_bytes = group_a ? k_row_bytes : v_row_bytes;
    const uint32_t dma_lane = group_a ? (16 * gw + lane / 4) * k_row_bytes + 16 * ((lane % 4) ^ (lane >> 4))
                                      : (8 * (lane >> 4) + (((lane >> 1) & 7) ^ (4 * ((lane >> 4) & 1)))) * v_row_bytes +
                                            32 * gw + 16 * (lane & 1);
    const size_t num_rows = work.kv_len - (size_t)work.tile_lo * KvTile;
    const int rsrc_tiles = UINT32_MAX / (KvTile * max(k_row_bytes, v_row_bytes)) - 1;
    int rsrc_t0 = 0;
    auto rsrc_at = [&](int t) INLINE {
        return kv_rsrc_at((const bf16_t*)dma_base, (size_t)t * KvTile, num_rows, row_bytes, group_a ? Dqk : Dv);
    };
    __amdgpu_buffer_rsrc_t dma_rsrc = rsrc_at(0);
    // Group A: K(t); group B: V(t). readfirstlane: the asm needs SGPR soffsets.
    int next_move = rsrc_tiles;                     // the first tile whose offset needs a new resource
    auto dma_tile = [&](auto ga, int t) INLINE {
        if (unlikely(t >= next_move)) {            // rare: the resource moves to tile t
            rsrc_t0 = t, next_move = t + rsrc_tiles;
            dma_rsrc = rsrc_at(t);
        }
        const uint32_t off = MLA8_SAME_TILE ? 0u : __builtin_amdgcn_readfirstlane((uint32_t)(t - rsrc_t0) * KvTile * row_bytes);
        if constexpr (decltype(ga)::value) {
            for (int i = 0; i < KDmas; ++i)
                buffer_load_dwordx4_lds(dma_rsrc, (lds_ptr_t)(lds + k_buf(t) + 4096 * i + 1024 * gw), dma_lane,
                                        off + 64 * i);
        } else {
            for (int i = 0; i < VDmas; ++i)
                buffer_load_dwordx4_lds(dma_rsrc, (lds_ptr_t)(lds + v_buf(t) + 2048 * gw + 1024 * i), dma_lane,
                                        __builtin_amdgcn_readfirstlane(off + 32 * i * v_row_bytes));
        }
    };

    // ==== LDS fragments [lds8] ====================================================================
    const uint32_t k_read0 = r * 64 + 16 * ((2 * h) ^ ((r >> 2) & 3));
    const uint32_t k_read1 = r * 64 + 16 * ((2 * h + 1) ^ ((r >> 2) & 3));
    const int vi = lane & 15, q16 = (lane >> 4) & 1;
    const uint32_t v_read = (vi / 8) * 256 + 32 * ((4 * h + (vi / 2) % 4) ^ (4 * (vi >> 3))) + 16 * q16 + 8 * (vi % 2);
    auto k_frag = [&](int t, int f) INLINE {       // f = KSteps mt + ks
        if (MLA8_NO_KREAD) { int x = lane + f + t; pin_after(x); return i32x8_t{} + (x & 0x3f3f3f3f); }
        const uint32_t o = k_buf(t) + 4096 * (f % KSteps) + 2048 * (f / KSteps);
        const i32x4_t x = ds_read_b128(lds, o + k_read0), y = ds_read_b128(lds, o + k_read1);
        return i32x8_t{x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]};
    };
    auto v_frag = [&](int t, int dt) INLINE {
        if (MLA8_NO_VREAD) { int x = lane + dt + t; pin_after(x); return i32x8_t{} + (x & 0x3f3f3f3f); }
        const uint32_t o = v_buf(t) + v_read + 2048 * dt;
        const i32x2_t c0 = ds_read_tr8(lds, o), c1 = ds_read_tr8(lds, o + 512);
        const i32x2_t c2 = ds_read_tr8(lds, o + 1024), c3 = ds_read_tr8(lds, o + 1536);
        return i32x8_t{c0[0], c0[1], c1[0], c1[1], c2[0], c2[1], c3[0], c3[1]};
    };

    // ==== state ===================================================================================
    f32x16_t o_acc[DMTiles] = {};                  // O^T[d = 32 dt + 8 (i / 4) + 4 h + i % 4][q = r]
    f32x16_t s_acc[2][KvMTiles];                   // S(t) and S(t + 1), alternating [self]
    f32x16_t row_sum = {};                         // [rsum]
    i32x8_t p_frag = {};                           // [p8]
    float seed = 0.0f;                             // -frame (log2 units)
    f32x16_t seed16 = {};                          // seed as QK's first C operand; changes only with the frame
    i32x8_t ones;
    for (int w = 0; w < 8; ++w) ones[w] = 0x38383838;   // fp8 e4m3 1.0

    auto mask_scores = [&](int sb, int t) INLINE {
        int kv_limit = work.kv_len - 1;
        if (a.causal) kv_limit = min(kv_limit, work.row0 + r + work.kv_shift);
        kv_limit -= (work.tile_lo + t) * KvTile + 4 * h;
        for (int mt = 0; mt < KvMTiles; ++mt)
            for (int i = 0; i < 16; ++i)
                s_acc[sb][mt][i] = 32 * mt + 8 * (i / 4) + i % 4 > kv_limit ? MaskScore : s_acc[sb][mt][i];
    };
    auto shift_frame = [&](int sb, float shift) INLINE {
        seed -= shift;
        seed16 = f32x16_t{} + seed;
        for (int mt = 0; mt < KvMTiles; ++mt)
            for (int i = 0; i < 16; ++i) s_acc[sb][mt][i] -= shift;
    };
    // Frames of S buffer sb from its lane max: the first tile sets them; later ones move lazily (rare).
    auto first_frame = [&](int sb, float mx) INLINE {
        const float m = max_xor32(mx);
        shift_frame(sb, m > NegInf ? m - FrameTop : 0.0f);
    };
    auto lazy_frame = [&](int sb, float mx) INLINE {
        if (unlikely(any_lane(mx > RescaleAt))) {
            const float shift = fmaxf(max_xor32(mx) - FrameTop, 0.0f);
            shift_frame(sb, shift);
            const float rs = __builtin_amdgcn_exp2f(-shift);
            for (int dt = 0; dt < DMTiles; ++dt) o_acc[dt] *= rs;
            row_sum *= rs;
        }
    };

    // ==== the two phases [self] ===================================================================
    // Phase X(t): QK(t + 1) into S[N] (K(t + 1) fragments read 2 MFMAs ahead), beside the exps of S[C] to P(t);
    // then V(t)'s fragments. Phase Y(t): row sums + PV(t), beside S[N]'s lane max (returned).
    // The VALU results are pinned at the phase's end, so they stay in its block.
    i32x8_t v_frags[DMTiles];
    auto phase_x = [&](auto c_c, int t) INLINE {
        constexpr int C = decltype(c_c)::value, N = 1 - C;
        constexpr int words[7] = {0, 1, 2, 4, 5, 6, 8};  // P words per MFMA (8 in all)
        i32x8_t kf[KFrags];
        kf[0] = k_frag(t + 1, 0), kf[1] = k_frag(t + 1, 1);
#pragma unroll
        for (int j = 0; j < KFrags; ++j) {
            for (int w = words[j]; w < words[j + 1]; ++w) {
                const int mt = w / 4, i = 4 * (w % 4);
                auto e = [](float x) INLINE { return MLA8_NO_EXP ? x : __builtin_amdgcn_exp2f(x); };
                const float p0 = e(s_acc[C][mt][i]), p1 = e(s_acc[C][mt][i + 1]), p2 = e(s_acc[C][mt][i + 2]), p3 = e(s_acc[C][mt][i + 3]);
                p_frag[w] = cvt_pk_fp8_hi(p2, p3, __builtin_amdgcn_cvt_pk_fp8_f32(p0, p1, p_frag[w], false));
            }
            const int mt = j / KSteps, ks = j % KSteps;
            if (!MLA8_NO_MFMA) s_acc[N][mt] = mfma32_scaled(kf[j], q_frag[ks], ks == 0 ? seed16 : s_acc[N][mt], scale_k, scale_q);
            else s_acc[N][mt] = ks == 0 ? seed16 + (float)kf[j][0] : s_acc[N][mt];
            if (j + 2 < KFrags) kf[j + 2] = k_frag(t + 1, j + 2);
            else v_frags[2 * (j + 2 - KFrags)] = v_frag(t, 2 * (j + 2 - KFrags)),
                 v_frags[2 * (j + 2 - KFrags) + 1] = v_frag(t, 2 * (j + 2 - KFrags) + 1);
        }
        pin_before(p_frag);
        for (int j = 0; j < KFrags; ++j) {
            __builtin_amdgcn_sched_group_barrier(SchedTrans, 5, 0);
            __builtin_amdgcn_sched_group_barrier(SchedValu, 3, 0);
            __builtin_amdgcn_sched_group_barrier(SchedMfma, 1, 0);
            if (j < KFrags - 2) __builtin_amdgcn_sched_group_barrier(SchedDsRead, 2, 0);
            else __builtin_amdgcn_sched_group_barrier(SchedDsRead, 8, 0);
        }
    };
    auto phase_y = [&](auto c_c) INLINE {
        constexpr int N = 1 - decltype(c_c)::value;
        constexpr int kmax[6] = {0, 4, 7, 10, 13, 16};   // max3 pairs per MFMA (16 in all)
        float m = MaskScore;
#pragma unroll
        for (int j = 0; j < 1 + DMTiles; ++j) {
            if (j == 0) row_sum = mfma32(ones, p_frag, row_sum);
            else if (!MLA8_NO_MFMA) o_acc[j - 1] = mfma32(v_frags[j - 1], p_frag, o_acc[j - 1]);
            else o_acc[j - 1][0] += (float)v_frags[j - 1][0];
            for (int k = kmax[j]; k < kmax[j + 1]; ++k)
                m = max3(m, s_acc[N][k / 8][2 * (k % 8)], s_acc[N][k / 8][2 * (k % 8) + 1]);
        }
        pin_before(m);
        for (int j = 0; j < 1 + DMTiles; ++j) {
            __builtin_amdgcn_sched_group_barrier(SchedMfma, 1, 0);
            __builtin_amdgcn_sched_group_barrier(SchedValu, 4, 0);
        }
        return m;
    };

    // ==== the main loop [self] [pair] =============================================================
    auto main_loop = [&](auto ga) INLINE {
        constexpr bool A = decltype(ga)::value;
        constexpr int Ahead = A ? KAhead : VAhead;
        auto barrier = []() INLINE {
            s_waitcnt_vmcnt<0>();                  // the last pair's DMAs
            sched_fence();
            s_barrier();
            sched_fence();
        };
        for (int t = 0; t < Ahead; ++t) dma_tile(ga, t);   // prologue: K(0 .. 2), V(0 .. 1)
        barrier();
        const int live = work.num_live;
        if (live > 0) {                            // S(0) and its frames (seed16 is 0)
            for (int mt = 0; mt < KvMTiles; ++mt)
                for (int ks = 0; ks < KSteps; ++ks)
                    s_acc[0][mt] = mfma32_scaled(k_frag(0, KSteps * mt + ks), q_frag[ks], ks == 0 ? seed16 : s_acc[0][mt],
                                                 scale_k, scale_q);
            if (unlikely(0 >= work.num_unmasked)) mask_scores(0, 0);
            float mx = MaskScore;
            for (int k = 0; k < 16; ++k) mx = max3(mx, s_acc[0][k / 8][2 * (k % 8)], s_acc[0][k / 8][2 * (k % 8) + 1]);
            first_frame(0, mx);
        }
        // Tile t: S(t) in buffer C, S(t + 1) into 1 - C; t even uses C = 0.
        auto now = []() INLINE { return MLA8_TIMING ? __builtin_amdgcn_s_memtime() : 0ull; };
        unsigned long long tm[5] = {};
        auto tile = [&](auto c_c, int t) INLINE {
            constexpr int C = decltype(c_c)::value;
            const auto t1 = now();
            phase_x(c_c, t);
            const auto t2 = now();
            if (unlikely(t + 1 >= work.num_unmasked)) mask_scores(1 - C, t + 1);
            const auto t3 = now();
            const float mx = phase_y(c_c);
            const auto t4 = now();
            lazy_frame(1 - C, mx);
            const auto t5 = now();
            if (MLA8_TIMING) tm[1] += t2 - t1, tm[2] += t3 - t2, tm[3] += t4 - t3, tm[4] += t5 - t4;
        };
        auto pair_start = [&](int t) INLINE {      // barrier, then this pair's DMA: tiles t + Ahead, t + Ahead + 1
            const auto t0 = now();
            barrier();
            if (MLA8_TIMING) tm[0] += now() - t0;
            dma_tile(ga, t + Ahead);
            dma_tile(ga, t + Ahead + 1);
        };
        using I0 = std::integral_constant<int, 0>;
        using I1 = std::integral_constant<int, 1>;
        int t = 0;
        for (; t + 1 < live; t += 2) {
            pair_start(t);
            tile(I0{}, t);
            tile(I1{}, t + 1);
        }
        if (t < live) {                            // a last pair with one live tile
            pair_start(t);
            tile(I0{}, t);
            t += 2;
        }
        for (; t < work.num_tiles; t += 2) pair_start(t);   // pairs every row of this wave masks: DMA only
        if (MLA8_TIMING && lane == 0) {
            for (int i = 0; i < 5; ++i) atomicAdd(&timing_acc[i], tm[i]);
            atomicAdd(&timing_acc[5], (unsigned long long)live);
        }
    };
    if (group_a) main_loop(std::true_type{});
    else         main_loop(std::false_type{});

    // ==== epilogue ================================================================================
    s_waitcnt_vmcnt<0>();
    const float l = row_sum[0];
    const int q = work.row0 + r;
    if (work.slot >= 0) {                          // a part of a split q block: fp32 partials for merge_parts
        if ((unsigned)q >= (unsigned)work.sq) return;
        const size_t row = (size_t)work.slot * BlockM + wave * TileRows + r;
        float* o_row = a.part_o + row * Dv;
        for (int dt = 0; dt < DMTiles; ++dt)
            for (int j = 0; j < 4; ++j) {
                f32x4_t x;
                for (int e = 0; e < 4; ++e) x[e] = o_acc[dt][4 * j + e] * a.v_scale;
                *reinterpret_cast<f32x4_t*>(o_row + 32 * dt + 8 * j + 4 * h) = x;
            }
        if (h == 0) a.part_ml[row] = float2{-seed, l};
        return;
    }
    if constexpr (Split) {
        const LseRows lse = lse_stash[wave];
        if (h == 0) store_lse(lse, work, q, -seed, l);
    }
    if ((unsigned)q >= (unsigned)work.sq) return;
    const float inv = (l > 0.0f ? __builtin_amdgcn_rcpf(l) : 0.0f) * a.v_scale;
    bf16_t* o_row = row_at(qo_rows(a.o, a.o_str, work), a.o_str, q);
    for (int dt = 0; dt < DMTiles; ++dt)
        for (int j = 0; j < 4; ++j) {
            bf16x4_t x;
            for (int e = 0; e < 4; ++e) x[e] = (bf16_t)(o_acc[dt][4 * j + e] * inv);
            *reinterpret_cast<bf16x4_t*>(o_row + 32 * dt + 8 * j + 4 * h) = x;
        }
}

// Explicit instantiations: clang 23 leaves fwd<true> undefined on the host when only the Kernels table names it.
template __global__ void fwd<false>(Args);
template __global__ void fwd<true>(Args);

}  // namespace attn::mla8

// ---- the C API (as mla_fp8.hip) ----------------------------------------------------------------
using namespace attn;

static const host::Kernels kernels{{{mla8::fwd<false>, mla8::fwd<true>}, {mla8::fwd<false>, mla8::fwd<true>}},
                                   1.0f, {1.0f, 1.0f}, mla8::Threads};

static Args fp8_args(const host::Tensors& t, int H, int HKv, int causal, float scale, float k_scale, float v_scale)
{
    Args a = host::make_args(t, H, HKv, causal, scale);
    a.k_scale = k_scale, a.v_scale = v_scale;
    return a;
}

extern "C" int mla_fp8_varlen_fwd(const void* q, const void* k, const void* v, void* o, float* lse,
                                  const int64_t* strides, const int* cu_q, const int* cu_k, int num_seqs, int H,
                                  int HKv, int causal, float scale, float k_scale, float v_scale,
                                  hipStream_t stream)
{
    const host::Tensors t{q, k, v, o, lse, strides};
    if (num_seqs < 0 || !host::valid_heads(H, HKv) || !host::valid_call(t, scale)) return -1;
    return host::launch_varlen<mla8::Dqk, mla8::Dv>(kernels, fp8_args(t, H, HKv, causal, scale, k_scale, v_scale),
                                                    cu_q, cu_k, num_seqs, stream);
}

extern "C" int mla_fp8_varlen_fwd_dev(const void* q, const void* k, const void* v, void* o, float* lse,
                                      const int64_t* strides, const int* cu_q, const int* cu_k,
                                      const int* host_cu_q, const int* host_cu_k, int num_seqs, int max_q,
                                      int max_k, int total_q, int total_k, int H, int HKv, int causal,
                                      float scale, float k_scale, float v_scale, hipStream_t stream)
{
    const host::Tensors t{q, k, v, o, lse, strides};
    if (num_seqs < 0 || !host::valid_heads(H, HKv) || !host::valid_call(t, scale)) return -1;
    const host::VarlenCall c{cu_q, cu_k, host_cu_q, host_cu_k, num_seqs, max_q, max_k, total_q, total_k};
    return host::launch_varlen_dev<mla8::Dqk, mla8::Dv>(
        kernels, fp8_args(t, H, HKv, causal, scale, k_scale, v_scale), c, stream);
}

extern "C" void* mla_fp8_plan_new() { return new host::VarlenPlan(); }
// MLA8_TIMING builds: copy out and clear the per-phase cycle totals (barrier, X, mask, Y, frame, tiles).
extern "C" int mla_fp8_timing(unsigned long long* out)
{
    if (hipMemcpyFromSymbol(out, HIP_SYMBOL(mla8::timing_acc), sizeof(unsigned long long) * 8) != hipSuccess) return -1;
    const unsigned long long zero[8] = {};
    return hipMemcpyToSymbol(HIP_SYMBOL(mla8::timing_acc), zero, sizeof(zero)) == hipSuccess ? 0 : -1;
}
extern "C" void mla_fp8_plan_free(void* p)
{
    if (!p) return;
    {
        std::lock_guard<std::mutex> lock(host::state_mutex);
        host::release(*(host::VarlenPlan*)p);
    }
    delete (host::VarlenPlan*)p;
}
extern "C" int mla_fp8_plan(void* plan, const int* cu_q, const int* cu_k, const int* host_cu_q,
                            const int* host_cu_k, int num_seqs, int max_q, int max_k, int total_q, int total_k,
                            int H, int HKv, int causal, hipStream_t stream)
{
    const host::VarlenCall c{cu_q, cu_k, host_cu_q, host_cu_k, num_seqs, max_q, max_k, total_q, total_k};
    return host::api_varlen_plan<mla8::Dqk, mla8::Dv>(plan, kernels, c, H, HKv, causal, stream);
}
extern "C" int mla_fp8_run(const void* plan, const void* q, const void* k, const void* v, void* o, float* lse,
                           const int64_t* strides, float scale, float k_scale, float v_scale, hipStream_t stream)
{
    const host::Tensors t{q, k, v, o, lse, strides};
    if (!plan || !host::valid_call(t, scale)) return -1;
    const host::VarlenPlan& h = *(const host::VarlenPlan*)plan;
    if (!h.ready) return -1;
    return host::run_varlen<mla8::Dqk, mla8::Dv>(h, kernels, fp8_args(t, h.H, h.HKv, h.causal, scale, k_scale,
                                                                      v_scale), stream);
}
