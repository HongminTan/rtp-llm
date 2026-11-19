/// \file topk_v3.cuh
/// \brief DeepSeek V4 (DSA indexer) top-k implementation classes.
///
/// This header holds ONLY the device-side implementation classes + helpers; the
/// `__global__` kernels and the host dispatcher live in topk_v3.cu.
///
/// Design notes:
///  - top-k (`topk`) is a *runtime* value (<= kMaxTopK = 2048), never a
///    compile-time constant.
///  - topk_v3 returns raw column indices in each row's valid prefix.
///  - the host selects a kernel by batch size, K and maximum sequence length.
///    Device code uses the individual row lengths.
///  - the standalone launcher instantiates 2/4/8-CTA cluster variants.
///
/// The FP16 histogram locates the Kth score. Oversized mixed bins are narrowed
/// by FP32 radix digits before candidate collection and exact selection.

#pragma once

#include "topk_v3_compat.cuh"

#include <cooperative_groups.h>
#include <cstdint>
#include <limits>

namespace device::topk {

namespace cg = cooperative_groups;

/// sgl_kernel names the warp size `kWarpThreads`; alias it locally as `kWarpSize`.
inline constexpr uint32_t kWarpSize = kWarpThreads;

// ---------------------------------------------------------------------------
// Shared-memory storage sized/aligned for several impl `Smem` types
// ---------------------------------------------------------------------------

/// Compile-time max over a non-empty pack (avoids an <algorithm> dependency).
template<typename T>
constexpr T ct_max(T a) {
    return a;
}
template<typename T, typename... Ts>
constexpr T ct_max(T a, Ts... rest) {
    const T m = ct_max(rest...);
    return a > m ? a : m;
}

/// Shared storage sized and aligned for any of the supplied kernel implementations.
template<typename... Smems>
struct MaxSmem {
    static constexpr size_t kSize  = ct_max(sizeof(Smems)...);
    static constexpr size_t kAlign = ct_max(alignof(Smems)...);
    alignas(kAlign) uint8_t storage[kSize];
};

// ---------------------------------------------------------------------------
// Order-preserving float -> integer key extraction
// ---------------------------------------------------------------------------

SGL_DEVICE uint32_t extract_exact_bin(float x) {
    uint32_t bits = __float_as_uint(x);
    if ((bits & 0x7fffffffu) == 0)
        bits = 0;  // Both FP32 zeros have the same rank.
    if ((bits & 0x7fffffffu) > 0x7f800000u)
        return 0xffffffffu;
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

SGL_DEVICE bool is_nan(float x) {
    return (__float_as_uint(x) & 0x7fffffffu) > 0x7f800000u;
}

template<uint32_t kBits>
SGL_DEVICE uint32_t extract_coarse_bin(float x) {
    static_assert(kBits >= 8 && kBits < 15);  // Leave bins outside the finite FP16 key band.
    constexpr uint32_t end       = 1u << kBits;
    const uint32_t     raw       = __float_as_uint(x);
    const uint32_t     magnitude = raw & 0x7fffffffu;
    // Reserve exact ranks outside the finite bins: -Inf, +Inf, then NaN.
    // Finite FP32 inputs keep their coarse bin even when conversion overflows.
    if (magnitude >= 0x7f800000u)
        return magnitude > 0x7f800000u ? end - 1 : ((raw >> 31) ? 0 : end - 2);
    const uint16_t bits = magnitude == 0 ? 0u : __half_as_ushort(cast<fp16_t>(x));
    const uint16_t key  = (bits & 0x8000) ? ~bits : bits | 0x8000;
    return key >> (16 - kBits);
}

// Smallest fp32 value `v` for which `extract_coarse_bin<kBits>(v) >= bin`, i.e. the
// lower fp32 boundary of coarse bin `bin`. Because `extract_coarse_bin` is monotonic
// non-decreasing in its argument, the collect pass can classify an element with two
// fp32 comparisons against these boundaries instead of recomputing the fp16 bin --
// removing the F2F conversion and bit-twiddle from the (compute-bound) second pass.
// The last two bins hold +Inf and NaNs. Callers classify NaNs explicitly.
template<uint32_t kBits>
SGL_DEVICE float coarse_bin_lower_bound(uint32_t bin) {
    const uint32_t  key = bin << (16 - kBits);
    constexpr float inf = std::numeric_limits<float>::infinity();
    if (key < 0x0400u)
        return bin == 0 ? -inf : std::numeric_limits<float>::lowest();
    if (key > 0xfc00u)
        return bin < (1u << kBits) - 1 ? inf : std::numeric_limits<float>::quiet_NaN();

    const auto half_bits = [](uint32_t ordered) -> uint16_t {
        return (ordered & 0x8000u) ? static_cast<uint16_t>(ordered ^ 0x8000u) : static_cast<uint16_t>(~ordered);
    };
    const uint16_t upper = half_bits(key);
    // Use the virtual next FP16 magnitude at the finite-range endpoints.
    const float left     = key == 0x0400u ? -65536.0f : __half2float(__ushort_as_half(half_bits(key - 1)));
    const float right    = key == 0xfc00u ? 65536.0f : __half2float(__ushort_as_half(upper));
    const float midpoint = 0.5f * (left + right);
    // Round-to-nearest-even assigns the midpoint to the neighbor with an even
    // significand. This also puts +65520 in the overflow bin and -65520 below
    // the first finite negative bin. At the signed-zero transition upper is +0.
    return (upper & 1u) ? nextafterf(midpoint, inf) : midpoint;
}

// Exact keys of all threshold-bin candidates, including those beyond the tie buffer.
// The coarse key is monotone in the exact key, so this interval excludes every other bin.
struct BoundaryKeyRange {
    uint32_t lo;
    uint32_t hi;

    SGL_DEVICE void include(float value) {
        const uint32_t key = extract_exact_bin(value);
        lo                 = min(lo, key);
        hi                 = max(hi, key);
    }

    SGL_DEVICE void merge_warp_into(BoundaryKeyRange* row) const {
        const uint32_t warp_lo = __reduce_min_sync(0xffffffffu, lo);
        const uint32_t warp_hi = __reduce_max_sync(0xffffffffu, hi);
        if (threadIdx.x % kWarpSize == 0 && warp_lo <= warp_hi) {
            atomicMin(&row->lo, warp_lo);
            atomicMax(&row->hi, warp_hi);
        }
    }
};

SGL_DEVICE uint32_t warp_inclusive_sum(uint32_t lane_id, uint32_t val) {
#pragma unroll
    for (uint32_t offset = 1; offset < 32; offset *= 2) {
        uint32_t n = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (lane_id >= offset)
            val += n;
    }
    return val;
}

SGL_DEVICE uint32_t warp_sum_bool(bool pred, uint32_t mask = 0xFFFFFFFF) {
    return __popc(__ballot_sync(mask, pred));
}

struct alignas(8) TieValue {
    float                            value;
    uint32_t                         idx;
    inline static constexpr TieValue invalid() {
        // A padding element must sort below every legal FP32 value.  -FLT_MAX is
        // greater than -inf and therefore corrupts ranks when -inf is selected.
        return TieValue{-std::numeric_limits<float>::infinity(), 0xFFFFFFFFu};
    }
};

// ---------------------------------------------------------------------------
// Per-row input and output
// ---------------------------------------------------------------------------

SGL_DEVICE int32_t page_to_indices(const int32_t* __restrict__ page_table, uint32_t i, uint32_t page_bits) {
    const uint32_t mask = (1u << page_bits) - 1u;
    return (page_table[i >> page_bits] << page_bits) | (i & mask);
}

/// topk_v3 emits raw column indices from the first seq_len scores in this row.
struct TopKProblem {
    const float* __restrict__ in;
    int32_t* __restrict__ out;      // [topk]
    int32_t* __restrict__ raw_out;  // optional raw (pre-transform) indices [topk]; nullptr if unused
    const int32_t* __restrict__ page_table;
    uint32_t topk;
    uint32_t seq_len;
    uint32_t page_bits;

    SGL_DEVICE void emit(uint32_t pos, uint32_t raw_idx) const {
        out[pos] = static_cast<int32_t>(raw_idx);
    }
    SGL_DEVICE void transform_output(uint32_t t, int32_t raw) const {
        if (raw_out != nullptr)
            raw_out[t] = raw;
        out[t] = raw < 0 ? -1 : page_to_indices(page_table, raw, page_bits);
    }
};

// ---------------------------------------------------------------------------
// Shared configuration + tie handling (exact radix select on the threshold bin)
// ---------------------------------------------------------------------------

struct TopKConfig {
    static constexpr uint32_t kMaxTopK   = 2048;
    static constexpr uint32_t kBlockSize = 1024;
    static constexpr uint32_t kOccupancy = 2;
    static constexpr uint32_t kNumWarps  = kBlockSize / kWarpSize;
    // kMaxNumTie must be >= kMaxTopK: the collect pass keeps at most kMaxNumTie
    // threshold-bin candidates, and up to `topk` output slots may have to be
    // filled from them (above_count can be 0, e.g. heavily tied or all-inf
    // scores). A smaller cap leaves slots that handle_tie can only pad, and
    // padded slots inside the first min(seq_len, topk) entries are dereferenced
    // by downstream sparse attention.
    static constexpr uint32_t kMaxNumTie = 2048;
    static constexpr uint32_t kRadixSize = 1 << 8;
    static constexpr uint32_t kTopKItems = (kMaxTopK + kBlockSize - 1) / kBlockSize;
    // tie candidates owned per thread in the strided handle_tie loops
    static constexpr uint32_t kTieItems = kMaxNumTie / kBlockSize;
    static_assert(kMaxNumTie >= kMaxTopK && kMaxNumTie % kBlockSize == 0 && kBlockSize % kNumWarps == 0);

    struct TieHandleSmem {
        struct alignas(16) MatchBin {
            uint32_t bin;
            uint32_t above_count;
            uint32_t equal_count;
            uint32_t _pad = 0;
        };
        alignas(128) uint32_t counter;
        alignas(128) uint32_t counter_final;
        // Deterministic-scatter scratch: block totals land here (shared) because
        // radix_tie_select consumes them from thread 0, not from the thread that
        // the prefix helper designates as the total writer.
        uint32_t total_scratch[2];
        MatchBin match;
        uint32_t warp_sum[kNumWarps];
        uint32_t histogram[2][kRadixSize];
    };

    /// Deterministic block-wide exclusive prefix over one uint32 per thread,
    /// in thread order. Replaces atomicAdd-assigned scatter positions: those
    /// followed SM arrival order, so the emitted output order (and, at the
    /// top-k boundary, tie membership) differed run to run for identical
    /// inputs. The op contract only promises an unspecified order, so pinning
    /// it is a tightening, not a break. ``warp_sum`` needs kNumWarps entries
    /// of scratch; the block total is stored through ``total``. Every thread
    /// must reach the call (it barriers).
    SGL_DEVICE static uint32_t block_exclusive_prefix(uint32_t val, uint32_t* warp_sum, uint32_t* total) {
        static_assert(kNumWarps == kWarpSize, "prefix assumes warp count == warp size");
        const uint32_t tx       = threadIdx.x;
        const uint32_t lane_id  = tx % kWarpSize;
        const uint32_t warp_id  = tx / kWarpSize;
        const uint32_t warp_inc = warp_inclusive_sum(lane_id, val);
        if (lane_id == kWarpSize - 1)
            warp_sum[warp_id] = warp_inc;
        __syncthreads();
        if (tx < kWarpSize) {
            warp_sum[lane_id] = warp_inclusive_sum(lane_id, warp_sum[lane_id]);
        }
        __syncthreads();
        const uint32_t prefix = (warp_id == 0 ? 0u : warp_sum[warp_id - 1]) + (warp_inc - val);
        if (tx == kBlockSize - 1)
            *total = prefix + val;
        __syncthreads();
        return prefix;
    }

    /// Resolve the threshold bin's ties exactly. `base` is the number of strictly
    /// "above" elements already emitted (final output starts at slot `base`);
    /// `topk` here is the number of remaining slots to fill (== global_topk - base).
    SGL_DEVICE static void handle_tie(  //
        const TieValue*    tie_buffer,
        const TopKProblem& problem,
        const uint32_t     base,
        const uint32_t     num_ties,
        const uint32_t     topk,
        TieHandleSmem*     smem) {
        constexpr auto is_greater = [](const TieValue& a, const TieValue& b) {
            return (a.value > b.value) || (a.value == b.value && a.idx < b.idx);
        };
        const auto tx      = threadIdx.x;
        const auto lane_id = tx % kWarpSize;
        const auto warp_id = tx / kWarpSize;
        static_assert(kNumWarps == kWarpSize);

        if (num_ties <= topk) {
            for (uint32_t t = tx; t < num_ties; t += kBlockSize) {
                problem.emit(base + t, tie_buffer[t].idx);
            }
            for (uint32_t t = num_ties + tx; t < topk; t += kBlockSize) {
                problem.emit(base + t, base + t);
            }
        } else if (num_ties <= kWarpSize) {
            if (lane_id >= num_ties || warp_id >= num_ties)
                return;  // some threads are idle
            /// NOTE: use long long to avoid mask overflow when num_tie == 32
            const uint32_t mask   = (1ull << num_ties) - 1u;
            const auto     tie    = tie_buffer[lane_id];
            const auto     target = tie_buffer[warp_id];
            const auto     rank   = warp_sum_bool(is_greater(tie, target), mask);
            if (lane_id == 0 && rank < topk)
                problem.emit(base + rank, target.idx);
        } else if (num_ties <= kWarpSize * 2) {
            // 64 x 64 topk implementation: each thread takes 2 elements
            const auto warp_id_0 = warp_id;
            const auto warp_id_1 = warp_id + kWarpSize;
            const auto lane_id_1 = lane_id + kWarpSize;
            const auto invalid   = TieValue::invalid();
            const auto tie_0     = tie_buffer[lane_id];
            const auto tie_1     = lane_id_1 < num_ties ? tie_buffer[lane_id_1] : invalid;
            const auto target_0  = tie_buffer[warp_id_0];
            const auto target_1  = tie_buffer[warp_id_1];
            if (true) {  // NOTE: warp_id_0 <= kNumWarps < num_ties
                const auto rank_0 = warp_sum_bool(is_greater(tie_0, target_0));
                const auto rank_1 = warp_sum_bool(is_greater(tie_1, target_0));
                const auto rank   = rank_0 + rank_1;
                if (lane_id == 0 && rank < topk)
                    problem.emit(base + rank, target_0.idx);
            }
            if (warp_id_1 < num_ties) {
                const auto rank_0 = warp_sum_bool(is_greater(tie_0, target_1));
                const auto rank_1 = warp_sum_bool(is_greater(tie_1, target_1));
                const auto rank   = rank_0 + rank_1;
                if (lane_id == 0 && rank < topk)
                    problem.emit(base + rank, target_1.idx);
            }
        } else if (num_ties <= kWarpSize * 4) {
            // 128 x 128 topk implementation: each thread takes 4 elements and does local sort + merge
            const auto     invalid = TieValue::invalid();
            const TieValue tie[]   = {
                tie_buffer[lane_id + 0 * kWarpSize],
                tie_buffer[lane_id + 1 * kWarpSize],
                lane_id + 2 * kWarpSize < num_ties ? tie_buffer[lane_id + 2 * kWarpSize] : invalid,
                lane_id + 3 * kWarpSize < num_ties ? tie_buffer[lane_id + 3 * kWarpSize] : invalid,
            };
            const TieValue target[] = {
                tie_buffer[warp_id + 0 * kWarpSize],
                tie_buffer[warp_id + 1 * kWarpSize],
                tie_buffer[warp_id + 2 * kWarpSize],
                tie_buffer[warp_id + 3 * kWarpSize],
            };
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (i >= 2 && warp_id + i * kWarpSize >= num_ties)
                    break;
                uint32_t rank = 0;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    rank += warp_sum_bool(is_greater(tie[j], target[i]));
                }
                if (lane_id == 0 && rank < topk)
                    problem.emit(base + rank, target[i].idx);
            }
        } else if (num_ties <= kBlockSize) {
            // Common case: one candidate per thread.
            radix_tie_select<1>(tie_buffer, problem, base, num_ties, topk, smem);
        } else {
            // Rare overflow case (kBlockSize < num_ties <= kMaxNumTie), kept out of
            // the common path so it alone pays the multi-item register cost.
            radix_tie_select<kTieItems>(tie_buffer, problem, base, num_ties, topk, smem);
        }
    }

    /// Exact radix select over the tie candidates: each thread owns kItems
    /// strided elements (inactive beyond num_ties). Requires
    /// num_ties <= kItems * kBlockSize.
    template<uint32_t kItems>
    SGL_DEVICE static void radix_tie_select(  //
        const TieValue*    tie_buffer,
        const TopKProblem& problem,
        const uint32_t     base,
        const uint32_t     num_ties,
        const uint32_t     topk,
        TieHandleSmem*     smem) {
        const auto tx      = threadIdx.x;
        const auto lane_id = tx % kWarpSize;
        const auto warp_id = tx / kWarpSize;

        bool     active[kItems];
        uint32_t key[kItems];
        uint32_t idx[kItems];
        uint32_t write_pos[kItems];
#pragma unroll
        for (uint32_t i = 0; i < kItems; ++i) {
            const auto t   = tx + i * kBlockSize;
            active[i]      = t < num_ties;
            const auto tie = active[i] ? tie_buffer[t] : TieValue::invalid();
            key[i]         = extract_exact_bin(tie.value);
            idx[i]         = tie.idx;
            write_pos[i]   = topk;
        }
        uint32_t topk_remain = topk;
        if (tx < kRadixSize)
            smem->histogram[0][tx] = 0;
        if (tx == kRadixSize)
            smem->counter = smem->counter_final = 0;
        __syncthreads();
        uint32_t total_active = num_ties;

#pragma unroll
        for (int round = 0; round < 4; round++) {
            const uint32_t shift     = 24 - round * 8;
            const auto     hist_idx  = round % 2;
            const auto     histogram = smem->histogram[hist_idx];

#pragma unroll
            for (uint32_t i = 0; i < kItems; ++i) {
                if (active[i])
                    atomicAdd(&histogram[(key[i] >> shift) & 0xFFu], 1);
            }
            if (round < 3 && tx < kRadixSize) {
                smem->histogram[hist_idx ^ 1][tx] = 0;
            }
            __syncthreads();

            uint32_t hist_val = 0;
            uint32_t warp_inc = 0;
            if (tx < kRadixSize) {
                hist_val = histogram[tx];
                warp_inc = warp_inclusive_sum(lane_id, hist_val);
                if (lane_id == kWarpSize - 1)
                    smem->warp_sum[warp_id] = warp_inc;
            }
            __syncthreads();
            if (tx < kRadixSize) {
                const auto inter  = warp::reduce_sum(lane_id < warp_id ? smem->warp_sum[lane_id] : 0);
                const auto prefix = inter + warp_inc;       // inclusive prefix through this bin
                const auto above  = total_active - prefix;  // elements in bins ABOVE this one
                // 3. Find threshold bin
                if (above < topk_remain && above + hist_val >= topk_remain) {
                    smem->match = {tx, above, hist_val};
                }
            }
            __syncthreads();

            const auto [threshold_bin, above_count, equal_count, __] = smem->match;
            if (round < 3)
                total_active = equal_count;
            topk_remain -= above_count;

            // 4. Deterministic scatter: count this round's leavers per thread,
            // take a thread-order block prefix, then assign positions after the
            // running bases. The atomicAdd version ordered both the above-bin
            // leavers and the final-round ties by SM arrival, so the output order
            // and tie membership varied run to run for identical inputs.
            uint32_t n_leave = 0;
            uint32_t n_final = 0;
            uint32_t leave_slot[kItems];
            uint32_t final_slot[kItems];
#pragma unroll
            for (uint32_t i = 0; i < kItems; ++i) {
                leave_slot[i] = 0xFFFFFFFFu;
                final_slot[i] = 0xFFFFFFFFu;
                if (!active[i])
                    continue;
                const uint32_t bin = (key[i] >> shift) & 0xFFu;
                if (bin > threshold_bin) {
                    leave_slot[i] = n_leave++;
                    active[i]     = false;
                } else if (bin < threshold_bin) {
                    active[i] = false;
                } else if (round == 3) {
                    final_slot[i] = n_final++;
                }
            }
            __syncthreads();
            const uint32_t base_leave = block_exclusive_prefix(n_leave, smem->warp_sum, &smem->total_scratch[0]);
            const uint32_t base_final = block_exclusive_prefix(n_final, smem->warp_sum, &smem->total_scratch[1]);
#pragma unroll
            for (uint32_t i = 0; i < kItems; ++i) {
                if (leave_slot[i] != 0xFFFFFFFFu) {
                    write_pos[i] = smem->counter + base_leave + leave_slot[i];
                } else if (final_slot[i] != 0xFFFFFFFFu) {
                    write_pos[i] = topk - topk_remain + smem->counter_final + base_final + final_slot[i];
                }
                // bin == threshold && round < 3: stay active for next round
            }
            // Barrier before the running-base update: every thread must be done
            // reading smem->counter / smem->counter_final for this round.
            __syncthreads();
            if (tx == 0) {
                smem->counter += smem->total_scratch[0];
                smem->counter_final += smem->total_scratch[1];
            }
            __syncthreads();

            if (round == 3 || topk_remain == 0)
                break;
        }

#pragma unroll
        for (uint32_t i = 0; i < kItems; ++i) {
            if (write_pos[i] < topk)
                problem.emit(base + write_pos[i], idx[i]);
        }
    }
};

// ---------------------------------------------------------------------------
// Radix base: histogram storage + input iteration + threshold-bin search
// ---------------------------------------------------------------------------

template<uint32_t kHistBits_>
struct TopKRadixBase: TopKConfig {
    static constexpr uint32_t kVecSize  = 4;
    static constexpr uint32_t kHistBits = kHistBits_;
    static constexpr uint32_t kHistSize = 1 << kHistBits;
    using vec_t                         = AlignedVector<float, kVecSize>;

    struct Smem {
        using kHistVec = AlignedVector<uint32_t, kHistSize / kBlockSize>;
        alignas(128) uint32_t count_eq;
        alignas(128) uint32_t count_gt;
        uint32_t         threshold_bin;
        uint32_t         bin_count;
        uint32_t         above_bin_count;
        BoundaryKeyRange boundary_range;
        uint32_t         warp_sum[kNumWarps];
        // All histogram passes finish before collection fills tie.values.
        // handle_tie then uses tie.handle as scratch. Overlaying these phases keeps the
        // kMaxNumTie-candidate buffer from growing the block's shared-memory
        // footprint. tie.handle and tie.values are live TOGETHER, so they sit
        // side by side inside the overlay, not in a union with each other.
        union {
            uint32_t histogram[kHistSize];
            kHistVec hist_vecs[kBlockSize];
            struct {
                TieHandleSmem handle;
                TieValue      values[kMaxNumTie];
            } tie;
        };
    };

protected:
    template<bool kAlignedInput, typename F>
    SGL_DEVICE static void for_each_input(const float* __restrict__ in, uint32_t seq_len, F&& fn) {
        const auto     tx       = threadIdx.x;
        const uint32_t num_full = seq_len / kVecSize;  // fully-in-bounds vectors

        vec_t    next_vec;
        uint32_t vi = tx;
        if (vi < num_full) {
            if constexpr (kAlignedInput) {
                next_vec.load(in, vi);
            } else {
                next_vec.load_unaligned(in, vi);
            }
        }
        while (vi < num_full) {
            const auto cur  = next_vec;
            const auto base = vi * kVecSize;
            vi += kBlockSize;
            if (vi < num_full) {
                if constexpr (kAlignedInput) {
                    next_vec.load(in, vi);
                } else {
                    next_vec.load_unaligned(in, vi);
                }
            }
#pragma unroll
            for (uint32_t j = 0; j < kVecSize; ++j) {
                fn(cur[j], base + j);
            }
        }

        // Tail: at most one partial vector, `rem` in [0, kVecSize).
        static_assert(kVecSize <= kBlockSize);  // ensure tail correctness
        const uint32_t tail_start = num_full * kVecSize;
        if (tx < seq_len - tail_start) {
            const auto idx = tail_start + tx;
            fn(in[idx], idx);
        }
    }

    // The histogram has already proved that this bin overflows the tie buffer.
    // Each CTA scans its own chunk, including candidates that will not be staged.
    template<bool kAlignedInput>
    SGL_DEVICE static void
    measure_boundary_range(const float* in, uint32_t length, uint32_t bin, BoundaryKeyRange* row_range) {
        BoundaryKeyRange range{0xffffffffu, 0u};
        for_each_input<kAlignedInput>(in, length, [&](float value, uint32_t) {
            if (extract_coarse_bin<kHistBits>(value) == bin)
                range.include(value);
        });
        range.merge_warp_into(row_range);
    }

    SGL_DEVICE static void finish_selection(const TopKProblem& problem, Smem* smem) {
        const auto base = smem->count_gt;
        // An exact fit or a single winning rank may already have filled every slot.
        if (base >= problem.topk)
            return;
        const auto remaining = problem.topk - base;
        const auto count     = smem->count_eq;
        // Refinement stops above capacity only when every boundary key is identical.
        handle_tie(
            smem->tie.values, problem, base, count > kMaxNumTie ? remaining : count, remaining, &smem->tie.handle);
    }

    SGL_DEVICE static void find_threshold(const uint32_t topk, const uint32_t seq_len, Smem* smem) {
        const auto         tx     = threadIdx.x;
        constexpr uint32_t kItems = kHistSize / kBlockSize;
        uint32_t           orig[kItems];
        const auto         hist_vec      = smem->hist_vecs[tx];
        uint32_t           tmp_local_sum = 0;

#pragma unroll
        for (uint32_t i = 0; i < kItems; ++i) {
            orig[i] = hist_vec[i];
            tmp_local_sum += orig[i];
        }

        const auto lane_id  = tx % kWarpSize;
        const auto warp_id  = tx / kWarpSize;
        const auto warp_inc = warp_inclusive_sum(lane_id, tmp_local_sum);
        const auto warp_exc = warp_inc - tmp_local_sum;
        if (lane_id == kWarpSize - 1)
            smem->warp_sum[warp_id] = warp_inc;

        __syncthreads();

        const auto tmp = smem->warp_sum[lane_id];
        // Exactly one bin satisfies: above < K && above + count >= K
        uint32_t prefix_sum = warp::reduce_sum(lane_id < warp_id ? tmp : 0);
        prefix_sum += warp_exc;
#pragma unroll
        for (uint32_t i = 0; i < kItems; ++i) {
            prefix_sum += orig[i];
            const auto above = seq_len - prefix_sum;
            if (above < topk && above + orig[i] >= topk) {
                const uint32_t bin    = tx * kItems + i;
                smem->threshold_bin   = bin;
                smem->bin_count       = orig[i];
                smem->above_bin_count = above;
            }
        }
        __syncthreads();
    }

    template<uint32_t kClusterSize>
    SGL_DEVICE static void reduce_histogram(Smem* smem) {
        if constexpr (kClusterSize > 1) {
            // Each rank reduces a partition of bins and publishes the totals to every peer.
            static_assert(kHistSize == kBlockSize);
            const auto cluster = cg::this_cluster();
            cluster.sync();
            const auto bin   = blockIdx.y * (kHistSize / kClusterSize) + threadIdx.x / kClusterSize;
            const auto peer  = cluster.map_shared_rank(&smem->histogram[bin], threadIdx.x % kClusterSize);
            const auto count = *peer;
            *peer            = warp::reduce_sum<kClusterSize>(count);
            cluster.sync();
        }
    }

    struct Bounds {
        float lo;
        float hi;
    };

    template<uint32_t kClusterSize, bool kAlignedInput>
    __device__ __noinline__ static Bounds
    refine_boundary(const float* in, uint32_t length, uint32_t need, Smem* smem, BoundaryKeyRange* row_range) {
        const auto bin   = smem->threshold_bin;
        auto       count = smem->bin_count;
        measure_boundary_range<kAlignedInput>(in, length, bin, row_range);
        if constexpr (kClusterSize > 1) {
            cg::this_cluster().sync();
        } else {
            __syncthreads();
        }
        auto range = *row_range;

        // Each pass consumes up to kHistBits of the remaining exact key. All CTAs
        // participate, even if the Kth value is buried in one rank's input chunk.
        while (range.lo < range.hi && need < count && count > kMaxNumTie) {
            const uint32_t          shift  = max(0, 32 - __clz(range.hi ^ range.lo) - int(kHistBits));
            const uint32_t          suffix = (1u << shift) - 1;
            const uint32_t          prefix = range.lo & ~(((kHistSize - 1) << shift) | suffix);
            typename Smem::kHistVec zeros;
            zeros.fill(0);
            smem->hist_vecs[threadIdx.x] = zeros;
            __syncthreads();
            for_each_input<kAlignedInput>(in, length, [&](float value, uint32_t) {
                const auto key = extract_exact_bin(value);
                if (key >= range.lo && key <= range.hi)
                    atomicAdd(&smem->histogram[(key >> shift) & (kHistSize - 1)], 1);
            });
            __syncthreads();
            reduce_histogram<kClusterSize>(smem);
            find_threshold(need, count, smem);
            const uint32_t start = prefix | (smem->threshold_bin << shift);
            range.lo             = max(range.lo, start);
            range.hi             = min(range.hi, start | suffix);
            need -= smem->above_bin_count;
            count = smem->bin_count;
        }

        // Use the *next exact key* for the exclusive upper bound. Its inverse
        // also maps the key between the smallest negative subnormal and +0 to zero.
        const auto from_key = [](uint32_t key) {
            return __uint_as_float(key & 0x80000000u ? key ^ 0x80000000u : ~key);
        };
        const float lo = from_key(range.lo);
        return {lo, need == count ? lo : from_key(range.hi + 1)};
    }

    template<uint32_t kClusterSize, bool kAlignedInput>
    SGL_DEVICE static Bounds
    selection_bounds(const float* in, uint32_t length, uint32_t topk, Smem* smem, BoundaryKeyRange* row_range) {
        const auto bin         = smem->threshold_bin;
        const auto count       = smem->bin_count;
        const auto need        = topk - smem->above_bin_count;
        const bool single_rank = bin == 0 || bin >= kHistSize - 2;
        if (need == count || count <= kMaxNumTie || single_rank) {
            const float lo = coarse_bin_lower_bound<kHistBits>(bin);
            // If no higher score exists, any K members of a single rank can be
            // emitted directly. Otherwise reserve the prefix for all higher ranks.
            const bool direct = need == count || (single_rank && need == topk);
            return {lo, direct ? lo : coarse_bin_lower_bound<kHistBits>(bin + 1)};
        }
        return refine_boundary<kClusterSize, kAlignedInput>(in, length, need, smem, row_range);
    }
};

// ---------------------------------------------------------------------------
// Register path: scores stay resident across histogram and collection;
// oversized mixed bins can require extra refinement reads. Templated on kLocalVecs so the caller picks the smallest
// covering kernel -- a larger kLocalVecs raises kMaxSeqLen but its fixed-unrolled loop wastes work on shorter
// sequences.
// ---------------------------------------------------------------------------

template<uint32_t kLocalVecs_>
struct TopKRegister: TopKRadixBase<12> {
    static constexpr uint32_t kLocalVecs = kLocalVecs_;
    static constexpr uint32_t kMaxSeqLen = kBlockSize * kVecSize * kLocalVecs;
    using Smem                           = typename TopKRadixBase<12>::Smem;

    template<bool kUsePDL, bool kAlignedInput>
    SGL_DEVICE static void forward(const TopKProblem problem, void* _smem) {
        const auto tx   = threadIdx.x;
        const auto smem = static_cast<Smem*>(_smem);

        {
            Smem::kHistVec hist_vec;
            hist_vec.fill(0);
            smem->hist_vecs[tx] = hist_vec;
        }
        if (tx == 0) {
            smem->boundary_range = {0xffffffffu, 0u};
            smem->count_eq       = 0;
            smem->count_gt       = 0;
        }

        __syncthreads();
        PDLWaitPrimary<kUsePDL>();

        // A vector `vi` is fully in bounds iff vi < num_full; only full vectors are
        // vector-loaded (16B aligned, never straddling seq_len). The <kVecSize tail is
        // a scalar remainder on the LAST lanes (which own the fewest full vectors, so
        // it overlaps the busy lanes' extra vector). The full path has no per-element
        // bounds check, keeping register pressure low enough to hold all vectors.
        const uint32_t num_full   = problem.seq_len / kVecSize;
        const uint32_t tail_start = num_full * kVecSize;
        const uint32_t tail       = problem.seq_len - tail_start;

        // Phase 1: load full vectors + build histogram
        vec_t local_vecs[kLocalVecs];
#pragma unroll
        for (uint32_t i = 0; i < kLocalVecs; ++i) {
            const auto vi = tx + kBlockSize * i;
            if (vi >= num_full)
                break;
            if constexpr (kAlignedInput) {
                local_vecs[i].load(problem.in, vi);
            } else {
                local_vecs[i].load_unaligned(problem.in, vi);
            }
        }
#pragma unroll
        for (uint32_t i = 0; i < kLocalVecs; ++i) {
            const auto vi = tx + kBlockSize * i;
            if (vi >= num_full)
                break;
#pragma unroll
            for (uint32_t j = 0; j < kVecSize; ++j)
                atomicAdd(&smem->histogram[extract_coarse_bin<kHistBits>(local_vecs[i][j])], 1);
        }
        if (tx >= kBlockSize - tail) {
            const uint32_t idx = tail_start + tx - (kBlockSize - tail);
            atomicAdd(&smem->histogram[extract_coarse_bin<kHistBits>(problem.in[idx])], 1);
        }
        __syncthreads();

        // Phase 2: Find the threshold bin
        find_threshold(problem.topk, problem.seq_len, smem);

        const auto topk = problem.topk;
        const auto bounds =
            selection_bounds<1, kAlignedInput>(problem.in, problem.seq_len, topk, smem, &smem->boundary_range);
        const float v_lo = bounds.lo;
        const float v_hi = bounds.hi;
        // Deterministic positions: count per thread, take a thread-order block
        // prefix, then emit/append in the same iteration order. The atomicAdd
        // version ordered both the above-threshold emissions and the tie-buffer
        // fill by SM arrival, so run-to-run outputs differed for identical
        // inputs (matters for the V4.1 decode sparse-attention consumer, whose
        // accumulation order follows these slots).
        uint32_t my_gt = 0;
        uint32_t my_eq = 0;
#pragma unroll
        for (uint32_t i = 0; i < kLocalVecs; ++i) {
            const auto vi = tx + kBlockSize * i;
            if (vi >= num_full)
                break;
#pragma unroll
            for (uint32_t j = 0; j < kVecSize; ++j) {
                const auto val = local_vecs[i][j];
                if (is_nan(val) || val >= v_hi)
                    ++my_gt;
                else if (val >= v_lo)
                    ++my_eq;
            }
        }
        if (tx >= kBlockSize - tail) {
            const auto val = problem.in[tail_start + tx - (kBlockSize - tail)];
            if (is_nan(val) || val >= v_hi)
                ++my_gt;
            else if (val >= v_lo)
                ++my_eq;
        }
        __syncthreads();  // warp_sum scratch is dead past find_threshold
        const auto base_gt = block_exclusive_prefix(my_gt, smem->warp_sum, &smem->count_gt);
        const auto base_eq = block_exclusive_prefix(my_eq, smem->warp_sum, &smem->count_eq);
        uint32_t   run_gt  = 0;
        uint32_t   run_eq  = 0;
        const auto collect = [&](float val, uint32_t idx) {
            if (is_nan(val) || val >= v_hi) {
                const auto pos = base_gt + run_gt++;
                if (pos < topk) [[likely]]
                    problem.emit(pos, idx);
            } else if (val >= v_lo) {
                const auto pos = base_eq + run_eq++;
                if (pos < kMaxNumTie) [[likely]]
                    smem->tie.values[pos] = {val, idx};
            }
        };
#pragma unroll
        for (uint32_t i = 0; i < kLocalVecs; ++i) {
            const auto vi   = tx + kBlockSize * i;
            const auto base = vi * kVecSize;
            if (vi >= num_full)
                break;
#pragma unroll
            for (uint32_t j = 0; j < kVecSize; ++j)
                collect(local_vecs[i][j], base + j);
        }
        if (tx >= kBlockSize - tail) {
            const uint32_t idx = tail_start + tx - (kBlockSize - tail);
            collect(problem.in[idx], idx);
        }

        // Phase 4: Handle ties.
        __syncthreads();
        finish_selection(problem, smem);
    }
};

// ---------------------------------------------------------------------------
// Single-CTA streaming path: reread scores for histogram, counting and output.
// Oversized mixed bins require additional refinement scans.
// ---------------------------------------------------------------------------

struct TopKStreaming: TopKRegister<2> {
public:
    static constexpr uint32_t kMaxSeqLen = std::numeric_limits<uint32_t>::max();

    template<bool kUsePDL, bool kAlignedInput>
    SGL_DEVICE static void forward(const TopKProblem problem, void* _smem) {
        const auto tx   = threadIdx.x;
        const auto smem = static_cast<Smem*>(_smem);

        {
            Smem::kHistVec hist_vec;
            hist_vec.fill(0);
            smem->hist_vecs[tx] = hist_vec;
        }
        if (tx == 0) {
            smem->boundary_range = {0xffffffffu, 0u};
            smem->count_eq       = 0;
            smem->count_gt       = 0;
        }
        __syncthreads();
        PDLWaitPrimary<kUsePDL>();

        // Phase 1: Load and build histogram
        for_each_input<kAlignedInput>(problem.in, problem.seq_len, [&](float val, uint32_t) {
            const auto bin = extract_coarse_bin<kHistBits>(val);
            atomicAdd(&smem->histogram[bin], 1);
        });
        __syncthreads();

        // Phase 2: Find the threshold bin
        find_threshold(problem.topk, problem.seq_len, smem);

        const auto topk = problem.topk;
        const auto bounds =
            selection_bounds<1, kAlignedInput>(problem.in, problem.seq_len, topk, smem, &smem->boundary_range);
        const float v_lo = bounds.lo;
        const float v_hi = bounds.hi;
        // Deterministic positions: one extra counting sweep, a thread-order
        // block prefix, then emit/append in the same iteration order. The
        // atomicAdd version ordered emissions and tie-buffer fill by SM arrival
        // (run-to-run output order and tie membership varied).
        uint32_t my_gt = 0;
        uint32_t my_eq = 0;
        for_each_input<kAlignedInput>(problem.in, problem.seq_len, [&](float val, uint32_t) {
            if (is_nan(val) || val >= v_hi)
                ++my_gt;
            else if (val >= v_lo)
                ++my_eq;
        });
        __syncthreads();  // warp_sum scratch is dead past find_threshold
        const auto base_gt = block_exclusive_prefix(my_gt, smem->warp_sum, &smem->count_gt);
        const auto base_eq = block_exclusive_prefix(my_eq, smem->warp_sum, &smem->count_eq);
        uint32_t   run_gt  = 0;
        uint32_t   run_eq  = 0;
        for_each_input<kAlignedInput>(problem.in, problem.seq_len, [&](float val, uint32_t idx) {
            if (is_nan(val) || val >= v_hi) {
                const auto pos = base_gt + run_gt++;
                if (pos < topk) [[likely]] {
                    problem.emit(pos, idx);
                }
            } else if (val >= v_lo) {
                const auto pos = base_eq + run_eq++;
                if (pos < kMaxNumTie) [[likely]] {
                    smem->tie.values[pos] = {val, idx};
                }
            }
        });

        __syncthreads();
        finish_selection(problem, smem);
    }
};

// ---------------------------------------------------------------------------
// Cluster path: very long seq_len, small batch. `kClusterSize` blocks cooperate
// on one batch element via distributed shared memory (one cluster per element).
// ---------------------------------------------------------------------------

template<uint32_t kClusterSize_>
struct TopKCluster: TopKRadixBase<10> {
public:
    static constexpr uint32_t kClusterSize = kClusterSize_;
    static constexpr uint32_t kMaxSeqLen   = std::numeric_limits<uint32_t>::max();
    using Base                             = TopKRadixBase<10>;
    struct Smem: Base::Smem {
        using kHistVec = Base::Smem::kHistVec;
        uint32_t start_eq_local, start_gt_local;
        // Fixed per-rank count slots (rank r writes [r]) so cross-rank staging
        // offsets are rank-order prefix sums instead of atomicAdd arrival order.
        uint32_t peer_counts[8][2];
        int32_t  tmp_out[kMaxTopK];
        static_assert(kClusterSize <= 8, "peer count slots sized for cluster 8");
    };

    // Process one batch element per cluster. There is no trailing barrier, so
    // the kernel performs cluster.sync() after each forward(). Writes raw indices to out.
    template<bool kUsePDL, bool kAlignedInput>
    SGL_DEVICE static void forward(TopKProblem problem, void* _smem) {
        forward_impl<kUsePDL, kAlignedInput>(problem, _smem);
    }

private:
    template<bool kUsePDL, bool kAlignedInput>
    SGL_DEVICE static void forward_impl(TopKProblem problem, void* _smem) {
        const auto tx         = threadIdx.x;
        const auto smem       = static_cast<Smem*>(_smem);
        const auto cluster    = cg::this_cluster();
        const auto this_rank  = blockIdx.y;
        const bool is_primary = (this_rank == 0);

        constexpr uint32_t kAlignElems   = kWarpSize * kVecSize;
        const uint32_t     chunk_size    = div_ceil(problem.seq_len, kClusterSize * kAlignElems) * kAlignElems;
        const uint32_t     chunk_start   = min(this_rank * chunk_size, problem.seq_len);
        const uint32_t     chunk_finish  = min(chunk_start + chunk_size, problem.seq_len);
        const uint32_t     local_seq_len = chunk_finish - chunk_start;
        problem.in += chunk_start;

        PDLWaitPrimary<kUsePDL>();
        typename Smem::kHistVec hist_vec;
        hist_vec.fill(0);
        smem->hist_vecs[tx] = hist_vec;
        if (tx == 0) {
            smem->boundary_range = {0xffffffffu, 0u};
            smem->count_eq       = 0;
            smem->count_gt       = 0;
        }
        __syncthreads();

        // Phase 1: Load and build histogram over this rank's contiguous chunk.
        for_each_input<kAlignedInput>(problem.in, local_seq_len, [&](float val, uint32_t) {
            const auto bin = extract_coarse_bin<kHistBits>(val);
            atomicAdd(&smem->histogram[bin], 1);
        });
        __syncthreads();

        reduce_histogram<kClusterSize>(smem);
        find_threshold(problem.topk, problem.seq_len, smem);

        const auto topk   = problem.topk;
        const auto smem_0 = cluster.map_shared_rank(smem, 0);
        const auto bounds = selection_bounds<kClusterSize, kAlignedInput>(
            problem.in, local_seq_len, topk, smem, &smem_0->boundary_range);
        const float v_lo = bounds.lo;
        const float v_hi = bounds.hi;
        // Deterministic per-rank positions: counting sweep, thread-order block
        // prefix, then emit/append in the same iteration order. Replaces the
        // atomicAdd arrival order so outputs are reproducible run to run.
        uint32_t my_gt = 0;
        uint32_t my_eq = 0;
        for_each_input<kAlignedInput>(problem.in, local_seq_len, [&](float val, uint32_t) {
            if (is_nan(val) || val >= v_hi)
                ++my_gt;
            else if (val >= v_lo)
                ++my_eq;
        });
        __syncthreads();  // warp_sum scratch is dead past find_threshold
        const auto base_gt = block_exclusive_prefix(my_gt, smem->warp_sum, &smem->count_gt);
        const auto base_eq = block_exclusive_prefix(my_eq, smem->warp_sum, &smem->count_eq);
        uint32_t   run_gt  = 0;
        uint32_t   run_eq  = 0;
        // Keep the scatter destinations separate. Merging the cluster-mapped
        // output and CTA-local tmp_out pointers drops non-primary stores with
        // CUDA 13.1+ on Hopper (sgl-project/sglang#34167).
        if (is_primary) {
            for_each_input<kAlignedInput>(problem.in, local_seq_len, [&](float val, uint32_t local_idx) {
                const auto idx = chunk_start + local_idx;
                if (is_nan(val) || val >= v_hi) {
                    const auto pos = base_gt + run_gt++;
                    if (pos < topk) [[likely]] {
                        problem.out[pos] = idx;
                    }
                } else if (val >= v_lo) {
                    const auto pos = base_eq + run_eq++;
                    if (pos < kMaxNumTie) [[likely]] {
                        smem->tie.values[pos] = {val, idx};
                    }
                }
            });
        } else {
            for_each_input<kAlignedInput>(problem.in, local_seq_len, [&](float val, uint32_t local_idx) {
                const auto idx = chunk_start + local_idx;
                if (is_nan(val) || val >= v_hi) {
                    const auto pos = base_gt + run_gt++;
                    if (pos < topk) [[likely]] {
                        smem->tmp_out[pos] = idx;
                    }
                } else if (val >= v_lo) {
                    const auto pos = base_eq + run_eq++;
                    if (pos < kMaxNumTie) [[likely]] {
                        smem->tie.values[pos] = {val, idx};
                    }
                }
            });
        }

        // Phase 3.5: deterministic cross-rank staging for non-primary blocks.
        // Fixed per-rank count slots plus rank-order prefix sums replace the
        // previous atomicAdd offsets, whose arrival order varied run to run.
        uint32_t start_write       = 0;
        uint32_t num_write         = 0;
        uint32_t local_above_count = 0;
        uint32_t local_equal_count = 0;
        if (!is_primary) {
            __syncthreads();
            local_above_count = smem->count_gt;
            local_equal_count = smem->count_eq;
            // Keep the count separate from the bounded copy: rank 0 uses the
            // global count to recognize a uniform bin larger than the buffer.
            if (tx == 0) {
                smem_0->peer_counts[this_rank][0] = local_above_count;
                smem_0->peer_counts[this_rank][1] = local_equal_count;
            }
        }
        cluster.sync();
        if (!is_primary) {
            // Rank 0's own local totals are still in smem_0->count_gt/count_eq
            // here (untouched until the fold below), so offsets are pure
            // functions of the per-rank counts.
            if (tx == 0) {
                uint32_t s_gt = smem_0->count_gt;
                uint32_t s_eq = smem_0->count_eq;
                for (uint32_t q = 1; q < this_rank; ++q) {
                    s_gt += smem_0->peer_counts[q][0];
                    s_eq += smem_0->peer_counts[q][1];
                }
                smem->start_gt_local = s_gt;
                smem->start_eq_local = s_eq;
            }
            __syncthreads();
            const auto start_eq_local = smem->start_eq_local;
#pragma unroll
            for (uint32_t i = 0; i < kTieItems; ++i) {
                const auto t = tx + i * kBlockSize;
                if (t < local_equal_count && start_eq_local + t < kMaxNumTie) {
                    smem_0->tie.values[start_eq_local + t] = smem->tie.values[t];
                }
            }
            start_write = smem->start_gt_local;
            num_write   = local_above_count;
        }

        cluster.sync();
        if (!is_primary) {
#pragma unroll
            for (uint32_t i = 0; i < kTopKItems; ++i) {
                if (const auto t = tx + i * kBlockSize; t < num_write && start_write + t < topk) {
                    problem.emit(start_write + t, smem->tmp_out[t]);
                }
            }
        } else {
            // Fold the stable per-rank slots into the global totals for phase 4.
            if (tx == 0) {
                for (uint32_t q = 1; q < kClusterSize; ++q) {
                    smem->count_gt += smem->peer_counts[q][0];
                    smem->count_eq += smem->peer_counts[q][1];
                }
            }
            __syncthreads();
            finish_selection(problem, smem);
        }
    }
};

}  // namespace device::topk
