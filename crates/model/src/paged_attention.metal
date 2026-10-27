// Decode-time paged attention: one query token per sequence, keys and values read straight
// out of the block pool through the sequence's block table.
//
// Grid: one threadgroup per (kv head, sequence). Each threadgroup serves all query heads that
// share the kv head (GQA), so every K/V row is loaded once per group. NSG simdgroups split the
// context positions; within a simdgroup each lane owns head_dim/32 dimensions. Scores use an
// online softmax, so context length is not limited by threadgroup memory, and the simdgroups'
// partial results are merged at the end.

#include <metal_stdlib>
using namespace metal;

#define NSG 8
#define MAX_REP 4
#define MAX_DPL 4

template <typename T>
inline void paged_attention_decode_impl(
    device const T* q,
    device const T* k_cache,
    device const T* v_cache,
    device const uint* block_tables,
    device const uint* ctx_lens,
    device T* out,
    constant uint& n_heads,
    constant uint& n_kv,
    constant uint& head_dim,
    constant uint& block_size,
    constant uint& max_blocks,
    constant float& scale,
    threadgroup float* tg_m,
    threadgroup float* tg_l,
    threadgroup float* tg_acc,
    uint2 tgid,
    uint sg,
    uint lane)
{
    const uint kvh = tgid.x;
    const uint seq = tgid.y;
    const uint n_rep = n_heads / n_kv;
    const uint dpl = head_dim / 32;
    const uint ctx = ctx_lens[seq];

    float qv[MAX_REP][MAX_DPL];
    float acc[MAX_REP][MAX_DPL];
    float m[MAX_REP];
    float l[MAX_REP];
    for (uint r = 0; r < n_rep; r++) {
        const uint head = kvh * n_rep + r;
        for (uint d = 0; d < dpl; d++) {
            qv[r][d] = float(q[(seq * n_heads + head) * head_dim + lane * dpl + d]) * scale;
            acc[r][d] = 0.0f;
        }
        m[r] = -INFINITY;
        l[r] = 0.0f;
    }

    device const uint* table = block_tables + seq * max_blocks;
    for (uint t = sg; t < ctx; t += NSG) {
        const uint slot = table[t / block_size] * block_size + t % block_size;
        const uint base = (slot * n_kv + kvh) * head_dim + lane * dpl;
        float kk[MAX_DPL];
        float vv[MAX_DPL];
        for (uint d = 0; d < dpl; d++) {
            kk[d] = float(k_cache[base + d]);
            vv[d] = float(v_cache[base + d]);
        }
        for (uint r = 0; r < n_rep; r++) {
            float partial = 0.0f;
            for (uint d = 0; d < dpl; d++) {
                partial += qv[r][d] * kk[d];
            }
            const float s = simd_sum(partial);
            const float m_new = max(m[r], s);
            const float corr = exp(m[r] - m_new);
            const float p = exp(s - m_new);
            l[r] = l[r] * corr + p;
            for (uint d = 0; d < dpl; d++) {
                acc[r][d] = acc[r][d] * corr + p * vv[d];
            }
            m[r] = m_new;
        }
    }

    for (uint r = 0; r < n_rep; r++) {
        if (lane == 0) {
            tg_m[sg * MAX_REP + r] = m[r];
            tg_l[sg * MAX_REP + r] = l[r];
        }
        for (uint d = 0; d < dpl; d++) {
            tg_acc[(sg * MAX_REP + r) * 128 + lane * dpl + d] = acc[r][d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        for (uint r = 0; r < n_rep; r++) {
            float big = -INFINITY;
            for (uint g = 0; g < NSG; g++) {
                big = max(big, tg_m[g * MAX_REP + r]);
            }
            float total = 0.0f;
            for (uint g = 0; g < NSG; g++) {
                total += tg_l[g * MAX_REP + r] * exp(tg_m[g * MAX_REP + r] - big);
            }
            const uint head = kvh * n_rep + r;
            for (uint d = 0; d < dpl; d++) {
                float a = 0.0f;
                for (uint g = 0; g < NSG; g++) {
                    a += tg_acc[(g * MAX_REP + r) * 128 + lane * dpl + d] * exp(tg_m[g * MAX_REP + r] - big);
                }
                out[(seq * n_heads + head) * head_dim + lane * dpl + d] = T(a / total);
            }
        }
    }
}

#define INSTANTIATE(NAME, T)                                                           \
kernel void NAME(                                                                      \
    device const T* q [[buffer(0)]],                                                   \
    device const T* k_cache [[buffer(1)]],                                             \
    device const T* v_cache [[buffer(2)]],                                             \
    device const uint* block_tables [[buffer(3)]],                                     \
    device const uint* ctx_lens [[buffer(4)]],                                         \
    device T* out [[buffer(5)]],                                                       \
    constant uint& n_heads [[buffer(6)]],                                              \
    constant uint& n_kv [[buffer(7)]],                                                 \
    constant uint& head_dim [[buffer(8)]],                                             \
    constant uint& block_size [[buffer(9)]],                                           \
    constant uint& max_blocks [[buffer(10)]],                                          \
    constant float& scale [[buffer(11)]],                                              \
    uint2 tgid [[threadgroup_position_in_grid]],                                       \
    uint sg [[simdgroup_index_in_threadgroup]],                                        \
    uint lane [[thread_index_in_simdgroup]])                                           \
{                                                                                      \
    threadgroup float tg_m[NSG * MAX_REP];                                             \
    threadgroup float tg_l[NSG * MAX_REP];                                             \
    threadgroup float tg_acc[NSG * MAX_REP * 128];                                     \
    paged_attention_decode_impl<T>(q, k_cache, v_cache, block_tables, ctx_lens, out,   \
        n_heads, n_kv, head_dim, block_size, max_blocks, scale, tg_m, tg_l, tg_acc,    \
        tgid, sg, lane);                                                               \
}

INSTANTIATE(paged_attention_decode_f32, float)
INSTANTIATE(paged_attention_decode_f16, half)
