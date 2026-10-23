// gfx950 fp8 MLA prefill forward, v2: Q bf16 (cast to fp8 in the kernel), K/V fp8 e4m3fn with per-tensor dequant
// scales, O bf16 (and optional LSE). Q/K head dim 192, V/O 128. C API and torch bindings as mla_fp8.hip (v1).
//
// From moonmath-ai/amd-kernels' gfx950 attention_mla.hip (MIT, see LICENSE.moonmath): K/V move by DMA into LDS
// rings, lazy-rescale frames, and the host planner, work list, KV splits and merge (attention_plan.hpp). The fp8
// parts are v1's ([32x32], [p8], [log2], [frame], [rsum], [lds8] in mla_fp8.hip). What v2 changes, tagged:
//   [1w]     One wave per SIMD (4 waves, 512 registers each), two 32-row q tiles per wave (a, b): a CTA still
//            covers 8 chunks of 32 rows. On gfx950 a wave issuing back-to-back MFMAs leaves the other wave on
//            its SIMD almost no VALU issue, but a wave's own VALU runs beside its own in-flight MFMAs. So each
//            wave overlaps one tile's softmax with the other tile's MFMAs:
//              step A(t): softmax a(t)  ||  row sums + PV b(t - 1), QK b(t)
//              step B(t): softmax b(t)  ||  row sums + PV a(t),     QK a(t + 1)
//            in two MFMA groups each: (row sums, PV) beside the row max and frame test, then QK beside the exps.
//   [reuse]  V(t) fragments are read once (in A(t)) for PV a(t) and PV b(t), K(t + 1) once (in B(t)) for
//            QK a(t + 1) and QK b(t + 1): half v1's LDS reads per row.
//   [dma2]   Every wave moves its share of K and V; DMA runs 2 tiles ahead (K and V rings 3 deep), one barrier per
//            tile at the start of A(t): K(t + 3) and V(t + 2) go into the buffers of K(t) and V(t - 1), read in
//            A(t - 1) and B(t - 1).

#include <hip/hip_runtime.h>
#include "attention_plan.hpp"

namespace attn::mla8 {

using i32x2_t  = __attribute__((ext_vector_type(2))) int;
using i32x4_t  = __attribute__((ext_vector_type(4))) int;
using i32x8_t  = __attribute__((ext_vector_type(8))) int;
using f32x16_t = __attribute__((ext_vector_type(16))) float;
using u8       = uint8_t;

constexpr int Dqk = 192, Dv = 128;
constexpr int Waves = 4, Threads = Waves * WaveSize;   // [1w]
constexpr int TileRows = 32, Tiles = 2;            // q tiles per wave: chunks 2 wave and 2 wave + 1
static_assert(Waves * Tiles == NumWaves, "a CTA covers the planner's 8 chunks");
constexpr int KSteps = Dqk / 64, KvMTiles = KvTile / 32, DMTiles = Dv / 32;
constexpr int KFrags = KvMTiles * KSteps;          // K fragments of a tile: (mt, ks)
constexpr float FrameTop = 7.0f, RescaleAt = 8.5f, Fp8Max = 448.0f;

// LDS [lds8] [dma2]: K tiles of 12 KB (3 ks blocks of 4 KB), V tiles of 8 KB (4 d blocks of 2 KB), 3 deep each.
constexpr int KTileBytes = KvTile * Dqk, VTileBytes = KvTile * Dv;
constexpr int KBufs = 3, VBufs = 3, KAhead = 3, VAhead = 2;
constexpr int KRegion = KBufs * KTileBytes, VRegion = VBufs * VTileBytes;
constexpr int LdsBytes = KRegion + VRegion;
static_assert(LdsBytes <= 163840, "gfx950 allows 160 KB");
constexpr int k_buf(int t) { return (unsigned)t % KBufs * KTileBytes; }
constexpr int v_buf(int t) { return KRegion + (unsigned)t % VBufs * VTileBytes; }
constexpr int KDmas = KTileBytes / 1024 / Waves, VDmas = VTileBytes / 1024 / Waves, Dmas = KDmas + VDmas;

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

    // Chunks 2 wave (tile a) and 2 wave + 1 (tile b): same sequence, KV head and KV tiles.
    const WaveWork work[Tiles] = {wave_work<Split>(a, Tiles * wave, TileRows),
                                  wave_work<Split>(a, Tiles * wave + 1, TileRows)};
    if (work[0].sq == 0) return;
    const WaveWork& kv = work[0];
    const int live = max(work[0].num_live, work[1].num_live);   // tiles this wave computes

    // ==== Q: bf16 rows to fp8 B fragments [log2] =================================================
    __shared__ LseRows lse_stash[NumWaves];
    if constexpr (Split)
        for (int c = 0; c < Tiles; ++c) {
            const LseRows lse = lse_rows(a, work[c]);
            if (lane == 0) lse_stash[Tiles * wave + c] = lse;
        }
    const float q_mul_full = a.scale_log2e * a.k_scale;
    const int q_exp = max(-100, min(100, (int)__builtin_rintf(-__builtin_log2f(q_mul_full))));
    const float q_mul = __builtin_ldexpf(q_mul_full, q_exp);       // in [0.7, 1.4]
    int scale_k = 127, scale_q = 127 - q_exp;                        // QK's E8M0 scales, VGPRs
    pin_after(scale_k), pin_after(scale_q);
    i32x8_t q_frag[Tiles][KSteps];                 // lane (h, r): Q[row0 + r][64 ks + 32 h + s]
    for (int c = 0; c < Tiles; ++c) {
        const int q = work[c].row0 + r;
        const bool valid = (unsigned)q < (unsigned)work[c].sq;
        const bf16_t* row = row_at(qo_rows(a.q, a.q_str, work[c]), a.q_str, valid ? q : 0);
        for (int ks = 0; ks < KSteps; ++ks)
            for (int u = 0; u < 4; ++u) {
                bf16x8_t x = *reinterpret_cast<const bf16x8_t*>(row + 64 * ks + 32 * h + 8 * u);
                if (!valid) x = bf16x8_t{};
                float f[8];
                for (int e = 0; e < 8; ++e) f[e] = clamp_fp8((float)x[e] * q_mul);
                q_frag[c][ks][2 * u] = cvt_pk_fp8_hi(f[2], f[3], cvt_pk_fp8_lo(f[0], f[1]));
                q_frag[c][ks][2 * u + 1] = cvt_pk_fp8_hi(f[6], f[7], cvt_pk_fp8_lo(f[4], f[5]));
            }
    }

    // ==== K/V DMA [lds8] [dma2] ===================================================================
    // K: wave w, instruction i (ks = i): rows 16 w + L / 4, logical chunk (L % 4) ^ (L / 16), into
    //   k_buf + 4096 i + 1024 w (+16 L).
    // V: wave w (d block w), instruction i: row 32 i + 8 (L / 16) + ((L / 2 % 8) ^ 4 (L / 16 % 2)),
    //   d 32 w + 16 (L % 2), into v_buf + 2048 w + 1024 i (+16 L).
    const uint32_t k_row_bytes = (uint32_t)a.k_str.token, v_row_bytes = (uint32_t)a.v_str.token;
    const u8* const k_base = (const u8*)a.k + offset(a.k_str, kv.b, (uint32_t)kv.kv_row0, kv.kv_head);
    const u8* const v_base = (const u8*)a.v + offset(a.v_str, kv.b, (uint32_t)kv.kv_row0, kv.kv_head);
    const uint32_t k_lane = (16 * wave + lane / 4) * k_row_bytes + 16 * ((lane % 4) ^ (lane >> 4));
    const uint32_t v_lane = (8 * (lane >> 4) + (((lane >> 1) & 7) ^ (4 * ((lane >> 4) & 1)))) * v_row_bytes +
                            32 * wave + 16 * (lane & 1);
    const size_t num_rows = kv.kv_len - (size_t)kv.tile_lo * KvTile;
    const int rsrc_tiles = UINT32_MAX / (KvTile * max(k_row_bytes, v_row_bytes)) - 1;
    int rsrc_t0 = 0;
    auto rsrc_at = [&](const u8* base, uint32_t row_bytes, int d, int t) INLINE {
        return kv_rsrc_at((const bf16_t*)base, (size_t)t * KvTile, num_rows, row_bytes, d);
    };
    __amdgpu_buffer_rsrc_t k_rsrc = rsrc_at(k_base, k_row_bytes, Dqk, 0), v_rsrc = rsrc_at(v_base, v_row_bytes, Dv, 0);
    // K(t + KAhead) and V(t + VAhead) [dma2]; resources move when a tile's offset outgrows 32 bits.
    auto dma = [&](int t) INLINE {
        const int tk = t + KAhead, tv = t + VAhead;
        if (tk - rsrc_t0 >= rsrc_tiles) {          // rare: both resources move to tile tv
            rsrc_t0 = tv;
            k_rsrc = rsrc_at(k_base, k_row_bytes, Dqk, tv), v_rsrc = rsrc_at(v_base, v_row_bytes, Dv, tv);
        }
        // readfirstlane: the asm needs SGPR soffsets, and under VGPR pressure the compiler may hold these in VGPRs.
        const uint32_t k_off = __builtin_amdgcn_readfirstlane((uint32_t)(tk - rsrc_t0) * KvTile * k_row_bytes);
        const uint32_t v_off = __builtin_amdgcn_readfirstlane((uint32_t)(tv - rsrc_t0) * KvTile * v_row_bytes);
        for (int i = 0; i < KDmas; ++i)
            buffer_load_dwordx4_lds(k_rsrc, (lds_ptr_t)(lds + k_buf(tk) + 4096 * i + 1024 * wave), k_lane,
                                    k_off + 64 * i);
        for (int i = 0; i < VDmas; ++i)
            buffer_load_dwordx4_lds(v_rsrc, (lds_ptr_t)(lds + v_buf(tv) + 2048 * wave + 1024 * i), v_lane,
                                    v_off + 32 * i * v_row_bytes);
    };
    // Prologue: K(0 .. KAhead - 1), V(0 .. VAhead - 1).
    for (int t = 0; t < KAhead; ++t)
        for (int i = 0; i < KDmas; ++i)
            buffer_load_dwordx4_lds(k_rsrc, (lds_ptr_t)(lds + k_buf(t) + 4096 * i + 1024 * wave), k_lane,
                                    __builtin_amdgcn_readfirstlane((uint32_t)t * KvTile * k_row_bytes + 64 * i));
    for (int t = 0; t < VAhead; ++t)
        for (int i = 0; i < VDmas; ++i)
            buffer_load_dwordx4_lds(v_rsrc, (lds_ptr_t)(lds + v_buf(t) + 2048 * wave + 1024 * i), v_lane,
                                    __builtin_amdgcn_readfirstlane((uint32_t)t * KvTile * v_row_bytes + 32 * i * v_row_bytes));

    // ==== LDS fragments [lds8] ====================================================================
    const uint32_t k_read0 = r * 64 + 16 * ((2 * h) ^ ((r >> 2) & 3));
    const uint32_t k_read1 = r * 64 + 16 * ((2 * h + 1) ^ ((r >> 2) & 3));
    const int vi = lane & 15, q16 = (lane >> 4) & 1;
    const uint32_t v_read = (vi / 8) * 256 + 32 * ((4 * h + (vi / 2) % 4) ^ (4 * (vi >> 3))) + 16 * q16 + 8 * (vi % 2);
    auto read_k = [&](int t, i32x8_t (&f)[KFrags]) INLINE {
        const uint32_t base = k_buf(t);
        for (int mt = 0; mt < KvMTiles; ++mt)
            for (int ks = 0; ks < KSteps; ++ks) {
                const uint32_t o = base + 4096 * ks + 2048 * mt;
                const i32x4_t x = ds_read_b128(lds, o + k_read0), y = ds_read_b128(lds, o + k_read1);
                f[KSteps * mt + ks] = i32x8_t{x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]};
            }
    };
    auto read_v = [&](int t, i32x8_t (&f)[DMTiles]) INLINE {
        const uint32_t base = v_buf(t) + v_read;
        for (int dt = 0; dt < DMTiles; ++dt) {
            const uint32_t o = base + 2048 * dt;
            const i32x2_t c0 = ds_read_tr8(lds, o), c1 = ds_read_tr8(lds, o + 512);
            const i32x2_t c2 = ds_read_tr8(lds, o + 1024), c3 = ds_read_tr8(lds, o + 1536);
            f[dt] = i32x8_t{c0[0], c0[1], c1[0], c1[1], c2[0], c2[1], c3[0], c3[1]};
        }
    };

    // ==== state ===================================================================================
    f32x16_t o_acc[Tiles][DMTiles] = {};           // O^T[d = 32 dt + 8 (i / 4) + 4 h + i % 4][q = r]
    f32x16_t s_acc[Tiles][KvMTiles];               // S^T[kv = 32 mt + 8 (i / 4) + 4 h + i % 4][q = r]
    f32x16_t row_sum[Tiles] = {};                  // [rsum]
    i32x8_t p_frag[Tiles] = {};                    // [p8]
    float seed[Tiles] = {0.0f, 0.0f};              // -frame (log2 units)
    i32x8_t ones;
    for (int w = 0; w < 8; ++w) ones[w] = 0x38383838;   // fp8 e4m3 1.0
    i32x8_t k_frag[KFrags], v_frag[DMTiles];

    // ==== softmax of tile c, in two phases [1w] ===================================================
    auto mask_scores = [&](int c, int kv0) INLINE {
        int kv_limit = work[c].kv_len - 1;
        if (a.causal) kv_limit = min(kv_limit, work[c].row0 + r + work[c].kv_shift);
        kv_limit -= kv0 + 4 * h;
        for (int mt = 0; mt < KvMTiles; ++mt)
            for (int i = 0; i < 16; ++i)
                s_acc[c][mt][i] = 32 * mt + 8 * (i / 4) + i % 4 > kv_limit ? MaskScore : s_acc[c][mt][i];
    };
    auto lane_max = [&](int c) INLINE {
        float m = MaskScore;
        for (int mt = 0; mt < KvMTiles; ++mt)
            for (int i = 0; i < 16; i += 2) m = max3(m, s_acc[c][mt][i], s_acc[c][mt][i + 1]);
        return m;
    };
    auto shift_frame = [&](int c, float shift) INLINE {
        seed[c] -= shift;
        for (int mt = 0; mt < KvMTiles; ++mt)
            for (int i = 0; i < 16; ++i) s_acc[c][mt][i] -= shift;
    };
    // Phase 1's tail: frames from the lane max (rare work out of line).
    auto frames = [&](int c, int t, float mx) INLINE {
        if (t == 0) {                              // the first tile sets every frame (O and row sums are 0)
            const float m = max_xor32(mx);
            shift_frame(c, m > NegInf ? m - FrameTop : 0.0f);
        } else if (unlikely(any_lane(mx > RescaleAt))) {
            const float shift = fmaxf(max_xor32(mx) - FrameTop, 0.0f);
            shift_frame(c, shift);
            const float rs = __builtin_amdgcn_exp2f(-shift);
            for (int dt = 0; dt < DMTiles; ++dt) o_acc[c][dt] *= rs;
            row_sum[c] *= rs;
        }
    };
    // Phase 2: P = exp2(S) to fp8.
    auto exps = [&](int c) INLINE {
        for (int w = 0; w < 8; ++w) {
            const int mt = w / 4, i = 4 * (w % 4);
            const float p0 = __builtin_amdgcn_exp2f(s_acc[c][mt][i]), p1 = __builtin_amdgcn_exp2f(s_acc[c][mt][i + 1]);
            const float p2 = __builtin_amdgcn_exp2f(s_acc[c][mt][i + 2]), p3 = __builtin_amdgcn_exp2f(s_acc[c][mt][i + 3]);
            p_frag[c][w] = cvt_pk_fp8_hi(p2, p3, cvt_pk_fp8_lo(p0, p1));
        }
    };

    // ==== MFMA groups [1w] =========================================================================
    // A group is a run of fenced pieces, one MFMA each plus its share of the other tile's VALU and of the LDS
    // reads: a wave stalls on an MFMA's issue while its pipe is busy, so the VALU must sit between MFMAs.
    // Branches stay outside groups. Tile indices are compile-time (integral_constant).
    using I0 = std::integral_constant<int, 0>;
    using I1 = std::integral_constant<int, 1>;
    auto pv_final = [&](int c) INLINE {            // (row sums, PV) of tile c, no VALU beside
        row_sum[c] = mfma32(ones, p_frag[c], row_sum[c]);
        for (int dt = 0; dt < DMTiles; ++dt) o_acc[c][dt] = mfma32(v_frag[dt], p_frag[c], o_acc[c][dt]);
    };
    auto read_k_frag = [&](int t, int f) INLINE {
        const uint32_t o = k_buf(t) + 4096 * (f % KSteps) + 2048 * (f / KSteps);
        const i32x4_t x = ds_read_b128(lds, o + k_read0), y = ds_read_b128(lds, o + k_read1);
        k_frag[f] = i32x8_t{x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]};
    };
    auto read_v_frag = [&](int t, int dt) INLINE {
        const uint32_t o = v_buf(t) + v_read + 2048 * dt;
        const i32x2_t c0 = ds_read_tr8(lds, o), c1 = ds_read_tr8(lds, o + 512);
        const i32x2_t c2 = ds_read_tr8(lds, o + 1024), c3 = ds_read_tr8(lds, o + 1536);
        v_frag[dt] = i32x8_t{c0[0], c0[1], c1[0], c1[1], c2[0], c2[1], c3[0], c3[1]};
    };
    // Group 1: (row sums, PV) of tile C; beside: tile C2's lane max (returned); K(tk) fragments if ReadK.
    auto group1 = [&](auto c_c, auto c2_c, auto read_k_c, int tk) INLINE {
        constexpr int C = decltype(c_c)::value, C2 = decltype(c2_c)::value;
        constexpr int kmax[6] = {0, 4, 7, 10, 13, 16};   // max3 pairs per piece (16 in all)
        constexpr int kfrag[6] = {0, 1, 2, 4, 5, 6};     // K fragments per piece (6 in all)
        float m = MaskScore;
#pragma unroll
        for (int j = 0; j < 1 + DMTiles; ++j) {
            if (j == 0) row_sum[C] = mfma32(ones, p_frag[C], row_sum[C]);
            else o_acc[C][j - 1] = mfma32(v_frag[j - 1], p_frag[C], o_acc[C][j - 1]);
            for (int k = kmax[j]; k < kmax[j + 1]; ++k)
                m = max3(m, s_acc[C2][k / 8][2 * (k % 8)], s_acc[C2][k / 8][2 * (k % 8) + 1]);
            if constexpr (decltype(read_k_c)::value)
                for (int f = kfrag[j]; f < kfrag[j + 1]; ++f) read_k_frag(tk, f);
        }
        pin_before(m);                             // the max stays in this block (no sinking past a branch)
        for (int j = 0; j < 1 + DMTiles; ++j) {
            __builtin_amdgcn_sched_group_barrier(SchedMfma, 1, 0);
            __builtin_amdgcn_sched_group_barrier(SchedValu, 3, 0);
            if constexpr (decltype(read_k_c)::value) __builtin_amdgcn_sched_group_barrier(SchedDsRead, 3, 0);
        }
        return m;
    };
    // Group 2: QK of tile C; beside: tile C2's exps to fp8 P; V(tv) fragments if ReadV.
    auto group2 = [&](auto c_c, auto c2_c, auto read_v_c, int tv) INLINE {
        constexpr int C = decltype(c_c)::value, C2 = decltype(c2_c)::value;
        constexpr int words[7] = {0, 1, 2, 4, 5, 6, 8};  // P words per piece (8 in all)
        const f32x16_t seed16 = f32x16_t{} + seed[C];
#pragma unroll
        for (int j = 0; j < KFrags; ++j) {
            const int mt = j / KSteps, ks = j % KSteps;
            s_acc[C][mt] = mfma32_scaled(k_frag[j], q_frag[C][ks], ks == 0 ? seed16 : s_acc[C][mt], scale_k, scale_q);
            for (int w = words[j]; w < words[j + 1]; ++w) {
                const int mt2 = w / 4, i = 4 * (w % 4);
                const float p0 = __builtin_amdgcn_exp2f(s_acc[C2][mt2][i]), p1 = __builtin_amdgcn_exp2f(s_acc[C2][mt2][i + 1]);
                const float p2 = __builtin_amdgcn_exp2f(s_acc[C2][mt2][i + 2]), p3 = __builtin_amdgcn_exp2f(s_acc[C2][mt2][i + 3]);
                p_frag[C2][w] = cvt_pk_fp8_hi(p2, p3, cvt_pk_fp8_lo(p0, p1));
            }
            if constexpr (decltype(read_v_c)::value)
                if (j < DMTiles) read_v_frag(tv, j);
        }
        pin_before(p_frag[C2]);                    // P stays in this block (no sinking past a branch)
        for (int j = 0; j < KFrags; ++j) {
            __builtin_amdgcn_sched_group_barrier(SchedMfma, 1, 0);
            __builtin_amdgcn_sched_group_barrier(SchedTrans, 5, 0);
            __builtin_amdgcn_sched_group_barrier(SchedValu, 3, 0);
            if constexpr (decltype(read_v_c)::value) __builtin_amdgcn_sched_group_barrier(SchedDsRead, 3, 0);
        }
    };
    auto qk = [&](int c) INLINE {
        const f32x16_t seed16 = f32x16_t{} + seed[c];
        for (int mt = 0; mt < KvMTiles; ++mt)
            for (int ks = 0; ks < KSteps; ++ks)
                s_acc[c][mt] = mfma32_scaled(k_frag[KSteps * mt + ks], q_frag[c][ks], ks == 0 ? seed16 : s_acc[c][mt],
                                             scale_k, scale_q);
    };
    auto barrier = [&](bool first) INLINE {
        if (first) s_waitcnt_vmcnt<0>();
        else s_waitcnt_vmcnt<Dmas>();              // the last tile's DMAs may stay in flight [dma2]
        sched_fence();
        s_barrier();
        sched_fence();
    };

    // ==== the main loop [1w] ======================================================================
    // Single-buffered fragments: V(t) is read during A(t)'s QK group, after PV b(t - 1) took V(t - 1); K(t + 1)
    // during B(t)'s PV group, after QK b(t) took K(t). v_frag starts at 0 (P b is 0 before tile 0); B(live - 1)
    // reads K(live) and computes an unused S a(live).
    for (int dt = 0; dt < DMTiles; ++dt) v_frag[dt] = i32x8_t{};
    barrier(true);                                 // K(0 .. 2), V(0 .. 1) visible
    if (live > 0) {
        read_k(0, k_frag);
        qk(0);                                     // S a(0)
    }
    for (int t = 0; t < live; ++t) {
        // ---- step A(t): softmax a(t) || (row sums, PV) b(t - 1), QK b(t)
        barrier(false);
        dma(t);
        if (unlikely(t >= work[0].num_unmasked)) mask_scores(0, (kv.tile_lo + t) * KvTile);
        float mx = group1(I1{}, I0{}, std::false_type{}, 0);
        frames(0, t, mx);
        group2(I1{}, I0{}, std::true_type{}, t);
        // ---- step B(t): softmax b(t) || (row sums, PV) a(t), QK a(t + 1)
        if (unlikely(t >= work[1].num_unmasked)) mask_scores(1, (kv.tile_lo + t) * KvTile);
        mx = group1(I0{}, I1{}, std::true_type{}, t + 1);
        frames(1, t, mx);
        group2(I0{}, I1{}, std::false_type{}, 0);
    }
    if (live > 0) pv_final(1);                     // the final (row sums, PV) b(live - 1)
    for (int t = live; t < kv.num_tiles; ++t) {    // tiles every row of this wave masks: DMA only
        barrier(false);
        dma(t);
    }

    // ==== epilogue ================================================================================
    s_waitcnt_vmcnt<0>();
    for (int c = 0; c < Tiles; ++c) {
        const WaveWork& w = work[c];
        const float l = row_sum[c][0];
        const int q = w.row0 + r;
        if (w.slot >= 0) {                         // a part of a split q block: fp32 partials for merge_parts
            if ((unsigned)q >= (unsigned)w.sq) continue;
            const size_t row = (size_t)w.slot * BlockM + (Tiles * wave + c) * TileRows + r;
            float* o_row = a.part_o + row * Dv;
            for (int dt = 0; dt < DMTiles; ++dt)
                for (int j = 0; j < 4; ++j) {
                    f32x4_t x;
                    for (int e = 0; e < 4; ++e) x[e] = o_acc[c][dt][4 * j + e] * a.v_scale;
                    *reinterpret_cast<f32x4_t*>(o_row + 32 * dt + 8 * j + 4 * h) = x;
                }
            if (h == 0) a.part_ml[row] = float2{-seed[c], l};
            continue;
        }
        if constexpr (Split) {
            const LseRows lse = lse_stash[Tiles * wave + c];
            if (h == 0) store_lse(lse, w, q, -seed[c], l);
        }
        if ((unsigned)q >= (unsigned)w.sq) continue;
        const float inv = (l > 0.0f ? __builtin_amdgcn_rcpf(l) : 0.0f) * a.v_scale;
        bf16_t* o_row = row_at(qo_rows(a.o, a.o_str, w), a.o_str, q);
        for (int dt = 0; dt < DMTiles; ++dt)
            for (int j = 0; j < 4; ++j) {
                bf16x4_t x;
                for (int e = 0; e < 4; ++e) x[e] = (bf16_t)(o_acc[c][dt][4 * j + e] * inv);
                *reinterpret_cast<bf16x4_t*>(o_row + 32 * dt + 8 * j + 4 * h) = x;
            }
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
