// One token per row of logits, on the GPU: greedy argmax, or temperature sampling via the
// Gumbel-max trick (argmax of logit / T + Gumbel noise is an exact draw from softmax(logit / T)).
// Noise comes from a counter-based hash of (row key, token id), so a row's draw depends only on
// its own key: reproducible per request and independent of batch composition.

#include <metal_stdlib>
using namespace metal;

inline uint lowbias32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

inline void keep_best(thread float& best, thread uint& best_i, float s, uint i) {
    if (s > best || (s == best && i < best_i)) {
        best = s;
        best_i = i;
    }
}

// params[row] = (key, bits of 1/T); 1/T == 0 means greedy.
kernel void sample_rows(
    device const float* logits [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant uint& vocab [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    const uint key = params[row * 2];
    const float inv_t = as_type<float>(params[row * 2 + 1]);
    device const float* l = logits + (ulong)row * vocab;

    float best = -INFINITY;
    uint best_i = 0xffffffffu;
    for (uint v = tid; v < vocab; v += tg_size) {
        float s = l[v];
        if (inv_t > 0.0f) {
            const uint h = lowbias32(key ^ (v * 0x9e3779b9u));
            const float u = (float(h >> 8) + 0.5f) * (1.0f / 16777216.0f);
            s = s * inv_t - precise::log(-precise::log(u));
        }
        keep_best(best, best_i, s, v);
    }
    for (ushort off = 16; off > 0; off >>= 1) {
        keep_best(best, best_i, simd_shuffle_down(best, off), simd_shuffle_down(best_i, off));
    }
    threadgroup float tg_best[32];
    threadgroup uint tg_idx[32];
    if (lane == 0) {
        tg_best[sg] = best;
        tg_idx[sg] = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const uint groups = tg_size / 32;
        best = lane < groups ? tg_best[lane] : -INFINITY;
        best_i = lane < groups ? tg_idx[lane] : 0xffffffffu;
        for (ushort off = 16; off > 0; off >>= 1) {
            keep_best(best, best_i, simd_shuffle_down(best, off), simd_shuffle_down(best_i, off));
        }
        if (lane == 0) {
            out[row] = best_i;
        }
    }
}
