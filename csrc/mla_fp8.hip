// gfx950 fp8 MLA prefill forward: Q bf16 (cast to fp8 in the kernel, softmax scale folded in), K/V fp8 e4m3fn
// with per-tensor dequant scales, O bf16 (and optional LSE). Q/K head dim 192 (128 nope + 64 rope), V/O 128.
//
// The structure is moonmath-ai/amd-kernels' gfx950 attention_mla.hip (MIT, see LICENSE.moonmath): one CTA of
// 8 waves per CU; groups A (waves 0-3) and B (4-7) alternate between an MFMA block and a softmax, two barriers
// per KV tile; A's DMA moves K, B's V, into LDS rings (K 2 deep, V 3 deep); the host planner, work list,
// KV splits and merge are moonmath's (attention_plan.hpp). What changes for fp8, tagged below:
//   [32x32]  v_mfma_f32_32x32x64_f8f6f4 (2x the bf16 rate): one 32-row q tile per wave. QK S^T = K Q^T runs 3
//            k-steps of 64 d over 2 kv M-tiles; PV O^T = V^T P^T one k-step over 4 d M-tiles. 192 = 3 x 64,
//            so no padding.
//   [p8]     P is S^T's C registers packed to fp8 in register order: byte s of lane 32h + r holds
//            kv = 32 (s / 16) + 8 (s % 16 / 4) + 4h + s % 4. V^T fragments follow the same kv order
//            (ds_read_b64_tr_b8), so P needs no lane shuffles.
//   [log2]   S comes out of the MFMA in log2 units minus the frame (the C seed): no per-score multiply. Q is
//            multiplied by q_mul = softmax_scale * log2(e) * k_scale before its fp8 cast, as m 2^-e with m in
//            [0.7, 1.4]: Q takes m (so it keeps fp8's normal range) and QK's E8M0 B scale takes 2^-e.
//   [frame]  fp8 P keeps 3 mantissa bits over 2^-9 .. 448: a frame puts the row max at P = 2^FrameTop and
//            moves (up only) when a score passes 2^RescaleAt.
//   [rsum]   Row sums by one more MFMA whose A is all ones: C[m][q] = sum over kv of P[q][kv].
//   [lds8]   Swizzled LDS tiles for 16-B K reads and 8-B transposed V reads, conflict free:
//              K tile [ks][kv][4 x 16 B], chunk c of row kv at c ^ (kv / 4 % 4)
//              V tile [d / 32][kv / 8][8 x 32 B], row kv at slot kv % 8 ^ 4 (kv / 8 % 2)

#include <hip/hip_runtime.h>
#include "attention_plan.hpp"

namespace attn::mla8 {

// Experiment switches (profiling only): MLA8_NO_EXP skips the exps, MLA8_SAME_TILE makes every DMA reread tile 0,
// MLA8_NO_MFMA skips QK and PV.
#ifndef MLA8_NO_EXP
#define MLA8_NO_EXP 0
#endif
#ifndef MLA8_SAME_TILE
#define MLA8_SAME_TILE 0
#endif
#ifndef MLA8_NO_MFMA
#define MLA8_NO_MFMA 0
#endif
// MLA8_VALU_RSUM: row sums by VALU adds over the fp32 P instead of the all-ones MFMA (one MFMA fewer per tile).
#ifndef MLA8_VALU_RSUM
#define MLA8_VALU_RSUM 1
#endif

using i32x2_t  = __attribute__((ext_vector_type(2))) int;
using i32x4_t  = __attribute__((ext_vector_type(4))) int;
using i32x8_t  = __attribute__((ext_vector_type(8))) int;
using f32x16_t = __attribute__((ext_vector_type(16))) float;
using u8       = uint8_t;

constexpr int Dqk = 192, Dv = 128;
constexpr int WaveRows = 32;                       // one 32-row q tile per wave [32x32]
constexpr int KSteps = Dqk / 64;                   // QK k-steps
constexpr int KvMTiles = KvTile / 32;              // QK M-tiles (32 kv each)
constexpr int DMTiles = Dv / 32;                   // PV M-tiles (32 d each)
constexpr float FrameTop = 7.0f;                   // [frame] a moved frame puts the row max at P = 2^7
constexpr float RescaleAt = 8.5f;                  // and moves again past P = 2^8.5 (fp8 e4m3 max: 448)
constexpr float Fp8Max = 448.0f;

// LDS bytes [lds8]: K tiles of 12 KB (3 ks blocks of 4 KB), V tiles of 8 KB (4 d blocks of 2 KB).
constexpr int KTileBytes = KvTile * Dqk, VTileBytes = KvTile * Dv;
constexpr int KBufs = 2, VBufs = 3;
constexpr int KRegion = KBufs * KTileBytes, VRegion = VBufs * VTileBytes;
constexpr int LdsBytes = KRegion + VRegion;
static_assert(LdsBytes <= 163840, "gfx950 allows 160 KB");
constexpr int k_buf(int t) { return (unsigned)t % KBufs * KTileBytes; }
constexpr int v_buf(int t) { return KRegion + (unsigned)t % VBufs * VTileBytes; }
constexpr int KDmas = KTileBytes / 1024 / 4, VDmas = VTileBytes / 1024 / 4;   // per wave per tile: 3, 2

// ---- instruction wrappers ------------------------------------------------------------------------
// v_mfma_f32_32x32x64_f8f6f4, fp8 e4m3 both sides. Constant zero scales select the unscaled opcode.
__device__ __forceinline__ f32x16_t mfma32(i32x8_t a, i32x8_t b, f32x16_t c)
{
    return __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, b, c, 0, 0, 0, 0, 0, 0);
}
// v_mfma_scale_f32_32x32x64_f8f6f4: A and B times 2^(scale - 127) (E8M0 in byte 0). The scales must be VGPRs:
// with SGPR scales this ROCm's compiler emits an instruction that gives wrong results.
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
// Two floats to two fp8 e4m3: the low half of a new word, or the high half of old.
__device__ __forceinline__ int cvt_pk_fp8_lo(float a, float b) { return __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false); }
__device__ __forceinline__ int cvt_pk_fp8_hi(float a, float b, int old)
{
    return __builtin_amdgcn_cvt_pk_fp8_f32(a, b, old, true);
}
__device__ __forceinline__ float clamp_fp8(float x) { return __builtin_amdgcn_fmed3f(x, -Fp8Max, Fp8Max); }

// The max of x over lanes L and L ^ 32.
__device__ __forceinline__ float max_xor32(float x)
{
    float t;
    // s_nop 1: the VALU write -> permlane read hazard (x and the copy t).
    asm volatile("v_mov_b32 %1, %0\n\ts_nop 1\n\tv_permlane32_swap_b32 %0, %1 bound_ctrl:1" : "+v"(x), "=&v"(t));
    return fmaxf(x, t);
}

// The sum of x over lanes L and L ^ 32.
__device__ __forceinline__ float sum_xor32(float x)
{
    float t;
    asm volatile("v_mov_b32 %1, %0\n\ts_nop 1\n\tv_permlane32_swap_b32 %0, %1 bound_ctrl:1" : "+v"(x), "=&v"(t));
    return x + t;
}

// ---- the kernel ----------------------------------------------------------------------------------
template <bool Split>
__global__ __launch_bounds__(BlockSize, 1) void fwd(Args a)
{
    __shared__ __attribute__((aligned(16))) u8 lds[LdsBytes];
    const int lane = threadIdx.x & (WaveSize - 1);
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / WaveSize);
    const int h = lane >> 5, r = lane & 31;        // MFMA lane view: half h, row/column r
    const bool group_a = wave < NumWaves / 2;
    const int gw = wave % (NumWaves / 2);

    const WaveWork work = wave_work<Split>(a, wave, WaveRows);
    if (work.sq == 0) return;

    // ==== Q: bf16 rows to fp8 B fragments [log2] =================================================
    // q_frag[ks] of lane (h, r): Q[row0 + r][64 ks + 32 h + s], s = 0..31.
    const bf16_t* const q_rows = qo_rows(a.q, a.q_str, work);
    bf16_t* const o_rows = qo_rows(a.o, a.o_str, work);
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
    i32x8_t q_frag[KSteps];
    {
        const int q = work.row0 + r;
        const bool valid = (unsigned)q < (unsigned)work.sq;
        const bf16_t* row = row_at(q_rows, a.q_str, valid ? q : 0);
        for (int ks = 0; ks < KSteps; ++ks)
            for (int u = 0; u < 4; ++u) {
                bf16x8_t x = *reinterpret_cast<const bf16x8_t*>(row + 64 * ks + 32 * h + 8 * u);
                if (!valid) x = bf16x8_t{};
                float f[8];
                for (int e = 0; e < 8; ++e) f[e] = clamp_fp8((float)x[e] * q_mul);
                const int lo = cvt_pk_fp8_hi(f[2], f[3], cvt_pk_fp8_lo(f[0], f[1]));
                const int hi = cvt_pk_fp8_hi(f[6], f[7], cvt_pk_fp8_lo(f[4], f[5]));
                q_frag[ks][2 * u] = lo, q_frag[ks][2 * u + 1] = hi;
            }
    }

    // ==== K/V DMA [lds8] ==========================================================================
    // A wave gw, instruction i (ks = i): K rows 16 gw + L / 4, logical chunk (L % 4) ^ (L / 16), into
    //   k_buf + 4096 i + 1024 gw (+16 L).
    // B wave gw (d block gw), instruction i: V row 32 i + 8 (L / 16) + ((L / 2 % 8) ^ 4 (L / 16 % 2)),
    //   d 32 gw + 16 (L % 2), into v_buf + 2048 gw + 1024 i (+16 L).
    const uint32_t k_row_bytes = (uint32_t)a.k_str.token, v_row_bytes = (uint32_t)a.v_str.token;
    const u8* const k_base = (const u8*)a.k + offset(a.k_str, work.b, (uint32_t)work.kv_row0, work.kv_head);
    const u8* const v_base = (const u8*)a.v + offset(a.v_str, work.b, (uint32_t)work.kv_row0, work.kv_head);
    uint32_t dma_lane;
    if (group_a) {
        dma_lane = (16 * gw + lane / 4) * k_row_bytes + 16 * ((lane % 4) ^ (lane >> 4));
    } else {
        const int kv = 8 * (lane >> 4) + (((lane >> 1) & 7) ^ (4 * ((lane >> 4) & 1)));
        dma_lane = kv * v_row_bytes + 32 * gw + 16 * (lane & 1);
    }
    const u8* const dma_rows = group_a ? k_base : v_base;
    const uint32_t row_bytes = group_a ? k_row_bytes : v_row_bytes, tile_bytes = KvTile * row_bytes;
    const size_t num_rows = work.kv_len - (size_t)work.tile_lo * KvTile;
    auto dma_rsrc_at = [&](int t) INLINE {
        return kv_rsrc_at((const bf16_t*)dma_rows, (size_t)t * KvTile, num_rows, row_bytes,
                          group_a ? Dqk : Dv);
    };
    const int rsrc_tiles = UINT32_MAX / (KvTile * max(k_row_bytes, v_row_bytes)) - 1;
    int rsrc_t0 = 0;
    __amdgpu_buffer_rsrc_t dma_rsrc = dma_rsrc_at(0);
    auto move_rsrc = [&](int t) INLINE {
        if (t - rsrc_t0 >= rsrc_tiles) {
            rsrc_t0 = t;
            dma_rsrc = dma_rsrc_at(t);
        }
    };
    // Tile t's DMA (A: K(t), B: V(t)), its instructions' soffsets.
    auto dma_tile = [&](int t) INLINE {
        const uint32_t off = (uint32_t)(t - rsrc_t0) * tile_bytes;
        if (group_a) {
            for (int i = 0; i < KDmas; ++i)
                buffer_load_dwordx4_lds(dma_rsrc, (lds_ptr_t)(lds + k_buf(t) + 4096 * i + 1024 * gw), dma_lane,
                                        off + 64 * i);
        } else {
            for (int i = 0; i < VDmas; ++i)
                buffer_load_dwordx4_lds(dma_rsrc, (lds_ptr_t)(lds + v_buf(t) + 2048 * gw + 1024 * i), dma_lane,
                                        off + 32 * i * v_row_bytes);
        }
    };
    dma_tile(0);

    // ==== LDS read addresses [lds8] ===============================================================
    // K fragment (mt, ks), A of QK: row 32 mt + r, d 64 ks + 32 h + 16 j + 0..15 (j = 0, 1).
    const uint32_t k_read0 = r * 64 + 16 * ((2 * h) ^ ((r >> 2) & 3));
    const uint32_t k_read1 = r * 64 + 16 * ((2 * h + 1) ^ ((r >> 2) & 3));
    // V^T fragment (dt), A of PV: read c (bytes 8c .. 8c + 7) of lane 32 h + 16 q16 + i reads
    // V[16 c + 8 (i / 8) + 4 h + (i / 2) % 4][32 dt + 16 q16 + 8 (i % 2) .. + 7]; ds_read_b64_tr_b8 hands
    // lane i the column 32 dt + 16 q16 + i of 8 such rows.
    const int vi = lane & 15, q16 = (lane >> 4) & 1;
    const uint32_t v_read = (vi / 8) * 256 + 32 * ((4 * h + (vi / 2) % 4) ^ (4 * (vi >> 3))) + 16 * q16 + 8 * (vi % 2);
    auto k_frag = [&](int t, int mt, int ks) INLINE {
        const uint32_t base = k_buf(t) + 4096 * ks + 2048 * mt;
        const i32x4_t x = ds_read_b128(lds, base + k_read0), y = ds_read_b128(lds, base + k_read1);
        return i32x8_t{x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]};
    };
    auto v_frag = [&](int t, int dt) INLINE {
        const uint32_t base = v_buf(t) + 2048 * dt + v_read;
        const i32x2_t c0 = ds_read_tr8(lds, base), c1 = ds_read_tr8(lds, base + 512);
        const i32x2_t c2 = ds_read_tr8(lds, base + 1024), c3 = ds_read_tr8(lds, base + 1536);
        return i32x8_t{c0[0], c0[1], c1[0], c1[1], c2[0], c2[1], c3[0], c3[1]};
    };

    // ==== state ===================================================================================
    f32x16_t o_acc[DMTiles] = {};                  // O^T[d = 32 dt + 8 (i / 4) + 4 h + i % 4][q = r]
    f32x16_t s_acc[KvMTiles];                      // S^T[kv = 32 mt + 8 (i / 4) + 4 h + i % 4][q = r]
    f32x16_t row_sum = {};                         // [rsum] every element: row r's sum of P
    float rsum = 0.0f;                             // MLA8_VALU_RSUM: this lane's half of row r's sum
    i32x8_t p_frag = {};                           // [p8]
    float seed = 0.0f;                             // -frame (log2 units), row r's
    f32x16_t seed16 = {};                          // seed as QK's first C operand; changes only with the frame
    i32x8_t ones;
    for (int w = 0; w < 8; ++w) ones[w] = 0x38383838;   // fp8 e4m3 1.0

    // ==== softmax =================================================================================
    // Keys past the sequence or past a causal row's last key: S = MaskScore.
    auto mask_scores = [&](int kv0) INLINE {
        int kv_limit = work.kv_len - 1;
        if (a.causal) kv_limit = min(kv_limit, work.row0 + r + work.kv_shift);
        kv_limit -= kv0 + 4 * h;
        for (int mt = 0; mt < KvMTiles; ++mt)
            for (int i = 0; i < 16; ++i)
                s_acc[mt][i] = 32 * mt + 8 * (i / 4) + i % 4 > kv_limit ? MaskScore : s_acc[mt][i];
    };
    auto lane_max = [&]() INLINE {
        float m = MaskScore;
        for (int mt = 0; mt < KvMTiles; ++mt)
            for (int i = 0; i < 16; i += 2) m = max3(m, s_acc[mt][i], s_acc[mt][i + 1]);
        return m;
    };
    auto shift_frame = [&](float shift) INLINE {
        seed -= shift;
        seed16 = f32x16_t{} + seed;
        for (int mt = 0; mt < KvMTiles; ++mt)
            for (int i = 0; i < 16; ++i) s_acc[mt][i] -= shift;
    };
    auto softmax = [&](int t) INLINE {
        if (t >= work.num_unmasked) mask_scores((work.tile_lo + t) * KvTile);
        const float mx = lane_max();
        if (t == 0) {                              // the first tile sets every frame (O and row sums are 0)
            const float m = max_xor32(mx);
            shift_frame(m > NegInf ? m - FrameTop : 0.0f);
        } else if (unlikely(any_lane(mx > RescaleAt))) {
            const float shift = fmaxf(max_xor32(mx) - FrameTop, 0.0f);
            shift_frame(shift);
            const float rs = __builtin_amdgcn_exp2f(-shift);
            for (int dt = 0; dt < DMTiles; ++dt) o_acc[dt] *= rs;
            row_sum *= rs;
            rsum *= rs;
        }
        for (int w = 0; w < 8; ++w) {
            const int mt = w / 4, i = 4 * (w % 4);
            auto e = [](float x) INLINE { return MLA8_NO_EXP ? x : __builtin_amdgcn_exp2f(x); };
            const float p0 = e(s_acc[mt][i]), p1 = e(s_acc[mt][i + 1]), p2 = e(s_acc[mt][i + 2]), p3 = e(s_acc[mt][i + 3]);
            p_frag[w] = cvt_pk_fp8_hi(p2, p3, __builtin_amdgcn_cvt_pk_fp8_f32(p0, p1, p_frag[w], false));
            if (MLA8_VALU_RSUM) rsum += (p0 + p1) + (p2 + p3);
        }
    };

    // ==== the MFMA block ==========================================================================
    // Block t, 11 steps: row sums of P(t - 1) [rsum]; PV of P(t - 1) over V(t - 1), d tiles 0..3 (block 0: P = 0
    // over V(0)); QK of K(t), (mt, ks) in order. A step's fragment is read Lead steps ahead; steps 1 and 2's
    // (V fragments) before the block: A after its softmax, B in its previous block's last two steps.
    constexpr int Steps = 1 + DMTiles + KvMTiles * KSteps, Lead = 2;
    auto step_frag = [&](int j, int t) INLINE {     // j >= 1
        return j <= DMTiles ? v_frag(t > 0 ? t - 1 : 0, j - 1)
                            : k_frag(t, (j - 1 - DMTiles) / KSteps, (j - 1 - DMTiles) % KSteps);
    };
    constexpr int StepReads[2] = {4, 2};            // LDS reads of a V / K fragment
    i32x8_t first_frags[Lead];                      // steps 1 .. Lead of the next block
    auto tile_barrier = []() INLINE {
        s_waitcnt_vmcnt<0>();
        sched_fence();
        s_barrier();
        sched_fence();
    };
    auto block_barrier = []() INLINE {
        sched_fence();
        s_barrier();
        sched_fence();
    };
    // Tile t's DMA instruction i (A: K(t), B: V(t)).
    auto dma_inst = [&](auto ga, int t, int i) INLINE {
        const uint32_t off = MLA8_SAME_TILE ? 0u : (uint32_t)(t - rsrc_t0) * tile_bytes;
        if constexpr (decltype(ga)::value)
            buffer_load_dwordx4_lds(dma_rsrc, (lds_ptr_t)(lds + k_buf(t) + 4096 * i + 1024 * gw), dma_lane,
                                    off + 64 * i);
        else
            buffer_load_dwordx4_lds(dma_rsrc, (lds_ptr_t)(lds + v_buf(t) + 2048 * gw + 1024 * i), dma_lane,
                                    off + 32 * i * v_row_bytes);
    };
    auto mfma_block = [&](auto ga, int t, auto qk_c) INLINE {
        constexpr bool A = decltype(ga)::value, qk = decltype(qk_c)::value;
        constexpr int Dmas = A ? KDmas : VDmas;
        move_rsrc(t + 1);
        tile_barrier();
        s_setprio<0>();
        i32x8_t frag[Steps];
        for (int j = 1; j <= Lead; ++j) frag[j] = first_frags[j - 1];
#pragma unroll
        for (int j = 0; j < Steps; ++j) {
            sched_fence();
            const int ahead = j + Lead;
            int reads = 0;
            if (ahead < Steps && (qk || ahead <= DMTiles)) {
                frag[ahead] = step_frag(ahead, t);
                reads = StepReads[ahead > DMTiles];
            } else if (!A && ahead >= Steps && ahead - Steps + 1 <= Lead) {   // B: the next block's first
                first_frags[ahead - Steps] = v_frag(t, ahead - Steps);
                reads = StepReads[0];
            }
            if (MLA8_NO_MFMA && j > 0) {}
            else if (j == 0) { if (!MLA8_VALU_RSUM) row_sum = mfma32(ones, p_frag, row_sum); }
            else if (j <= DMTiles) o_acc[j - 1] = mfma32(frag[j], p_frag, o_acc[j - 1]);
            else if (qk) {
                const int mt = (j - 1 - DMTiles) / KSteps, ks = (j - 1 - DMTiles) % KSteps;
                s_acc[mt] = mfma32_scaled(frag[j], q_frag[ks], ks == 0 ? seed16 : s_acc[mt], scale_k, scale_q);
            }
            if (j % 2 == 1 && j / 2 < Dmas) dma_inst(ga, t + 1, j / 2);
            __builtin_amdgcn_sched_group_barrier(SchedMfma, 1, 0);
            for (int n = 0; n < reads; ++n) __builtin_amdgcn_sched_group_barrier(SchedDsRead, 1, 0);
        }
        sched_fence();
        block_barrier();
        s_setprio<1>();
    };
    auto dma_only_block = [&](auto ga, int t) INLINE {
        move_rsrc(t + 1);
        tile_barrier();
        for (int i = 0; i < (decltype(ga)::value ? KDmas : VDmas); ++i) dma_inst(ga, t + 1, i);
        block_barrier();
    };

    // ==== the main loop (barriers as attention_mla.hip) ===========================================
    s_waitcnt_vmcnt<0>();
    s_barrier();                                   // tile 0 visible to every wave
    auto main_loop = [&](auto ga) INLINE {
        constexpr bool A = decltype(ga)::value;
        for (int j = 0; j < Lead; ++j) first_frags[j] = v_frag(0, j);
        if constexpr (!A) s_barrier();             // B's blocks trail by one window
        for (int t = 0; t < work.num_live; ++t) {
            mfma_block(ga, t, std::true_type{});
            softmax(t);
            if constexpr (A)                       // block t + 1's first fragments: V(t), visible since
                for (int j = 0; j < Lead; ++j) first_frags[j] = v_frag(t, j);   // B's barrier of block t
        }
        mfma_block(ga, work.num_live, std::false_type{});   // the final PV
        for (int t = work.num_live + 1; t <= work.num_tiles; ++t) dma_only_block(ga, t);
        if constexpr (A) s_barrier();              // pairs with B's last block barrier
    };
    if (group_a) main_loop(std::true_type{});
    else         main_loop(std::false_type{});

    // ==== epilogue ================================================================================
    const float l = MLA8_VALU_RSUM ? sum_xor32(rsum) : row_sum[0];
    const int q = work.row0 + r;
    if (work.slot >= 0) {                          // a part of a split q block: fp32 partials for merge_parts
        if ((unsigned)q >= (unsigned)work.sq) return;
        const size_t row = (size_t)work.slot * BlockM + wave * WaveRows + r;
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
    bf16_t* o_row = row_at(o_rows, a.o_str, q);
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

// ---- the C API ---------------------------------------------------------------------------------
using namespace attn;

static const host::Kernels kernels{{{mla8::fwd<false>, mla8::fwd<true>}, {mla8::fwd<false>, mla8::fwd<true>}},
                                   1.0f, {1.0f, 1.0f}};

static Args fp8_args(const host::Tensors& t, int H, int HKv, int causal, float scale, float k_scale, float v_scale)
{
    Args a = host::make_args(t, H, HKv, causal, scale);
    a.k_scale = k_scale, a.v_scale = v_scale;
    return a;
}

// Varlen, host cu_seqlens (planned on the host, cached per stream).
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

// Varlen, device cu_seqlens (host copies optional): planned per call (dev mode under capture).
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

// Plans: one per forward pass, run by every layer.
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
