// Decode-time paged attention: one query token per sequence, keys and values read straight
// out of the block pool through the sequence's block table.
//
// A threadgroup handles one (kv head, sequence) pair over a range of context positions. It
// serves all query heads that share the kv head (GQA), so every K/V row is loaded once per
// group. NSG simdgroups take blocks of 32 positions in turn. Within a block each lane scores
// its own position against every query head, the softmax statistics are combined once per
// block, and then each lane accumulates its head_dim/32 output dimensions over the block's V
// rows. Scores use an online softmax, so context length is not limited by threadgroup memory,
// and the simdgroups' partial results are merged at the end.
//
// Short contexts run `paged_attention_decode`: one threadgroup per pair covers the whole
// context. Long contexts with few sequences would leave most of the GPU idle that way (a
// single sequence is only n_kv threadgroups), so they split the context into partitions
// (flash-decoding): `paged_attention_decode_split` runs one threadgroup per (pair, partition)
// and writes each partition's max score, softmax denominator and unnormalized output, and
// `paged_attention_reduce` merges the partitions.

#include <metal_stdlib>
using namespace metal;

#define NSG 8
#define MAX_REP 4
#define MAX_DPL 4
#define MAX_HEAD_DIM 128

// Eight consecutive K elements as two float4s, in one 16-byte load for half.
inline void load8(device const half* p, thread float4& a, thread float4& b) {
    const uint4 raw = *(device const uint4*)p;
    a = float4(as_type<half4>(raw.xy));
    b = float4(as_type<half4>(raw.zw));
}

inline void load8(device const float* p, thread float4& a, thread float4& b) {
    a = *(device const float4*)p;
    b = *(device const float4*)(p + 4);
}

// Online-softmax state for every query head of a GQA group: max score, denominator relative
// to it, and unnormalized output (this lane's dimensions).
struct Partial {
    float m[MAX_REP];
    float l[MAX_REP];
    float acc[MAX_REP][MAX_DPL];
};

// Attends context positions [start, end) of one (kv head, sequence) pair and merges the
// simdgroups' states; every thread returns the merged state for its lane's dimensions. An
// empty range gives m = -inf, l = 0, acc = 0.
template <typename T>
inline Partial attend_range(
    device const T* q,
    device const T* k_cache,
    device const T* v_cache,
    device const uint* table,
    uint seq,
    uint kvh,
    uint n_heads,
    uint n_kv,
    uint head_dim,
    uint block_size,
    float scale,
    uint start,
    uint end,
    threadgroup float* tg_q,
    threadgroup float* tg_m,
    threadgroup float* tg_l,
    threadgroup float* tg_acc,
    uint sg,
    uint lane)
{
    const uint n_rep = n_heads / n_kv;
    const uint dpl = head_dim / 32;

    // The group's query heads, pre-scaled, shared by every lane.
    for (uint i = sg * 32 + lane; i < n_rep * head_dim; i += NSG * 32) {
        tg_q[i] = float(q[(seq * n_heads + kvh * n_rep) * head_dim + i]) * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    Partial p;
    for (uint r = 0; r < n_rep; r++) {
        for (uint d = 0; d < dpl; d++) {
            p.acc[r][d] = 0.0f;
        }
        p.m[r] = -INFINITY;
        p.l[r] = 0.0f;
    }

    for (uint t0 = start + sg * 32; t0 < end; t0 += NSG * 32) {
        // Scores: lane j takes position t0 + j and dots its whole K row with each query head.
        const uint t = t0 + lane;
        const bool valid = t < end;
        const uint slot = valid ? table[t / block_size] * block_size + t % block_size : 0;
        float s[MAX_REP];
        for (uint r = 0; r < n_rep; r++) {
            s[r] = -INFINITY;
        }
        if (valid) {
            device const T* krow = k_cache + (slot * n_kv + kvh) * head_dim;
            float dots[MAX_REP] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint i = 0; i < head_dim / 8; i++) {
                float4 ka, kb;
                load8(krow + i * 8, ka, kb);
                for (uint r = 0; r < n_rep; r++) {
                    threadgroup const float4* q4 = (threadgroup const float4*)(tg_q + r * head_dim);
                    dots[r] += dot(q4[2 * i], ka) + dot(q4[2 * i + 1], kb);
                }
            }
            for (uint r = 0; r < n_rep; r++) {
                s[r] = dots[r];
            }
        }
        // One online-softmax update for the whole block.
        float e[MAX_REP];
        for (uint r = 0; r < n_rep; r++) {
            const float m_new = max(p.m[r], simd_max(s[r]));
            const float corr = exp(p.m[r] - m_new);
            e[r] = valid ? exp(s[r] - m_new) : 0.0f;
            p.l[r] = p.l[r] * corr + simd_sum(e[r]);
            for (uint d = 0; d < dpl; d++) {
                p.acc[r][d] *= corr;
            }
            p.m[r] = m_new;
        }
        // Values: every lane adds its dimensions of each position's V row, weighted.
        const uint n = min(32u, end - t0);
        for (uint j = 0; j < n; j++) {
            device const T* vrow =
                v_cache + (simd_broadcast(slot, ushort(j)) * n_kv + kvh) * head_dim + lane * dpl;
            float vv[MAX_DPL];
            for (uint d = 0; d < dpl; d++) {
                vv[d] = float(vrow[d]);
            }
            for (uint r = 0; r < n_rep; r++) {
                const float w = simd_broadcast(e[r], ushort(j));
                for (uint d = 0; d < dpl; d++) {
                    p.acc[r][d] += w * vv[d];
                }
            }
        }
    }

    for (uint r = 0; r < n_rep; r++) {
        if (lane == 0) {
            tg_m[sg * MAX_REP + r] = p.m[r];
            tg_l[sg * MAX_REP + r] = p.l[r];
        }
        for (uint d = 0; d < dpl; d++) {
            tg_acc[(sg * MAX_REP + r) * MAX_HEAD_DIM + lane * dpl + d] = p.acc[r][d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    Partial out;
    for (uint r = 0; r < n_rep; r++) {
        float big = -INFINITY;
        for (uint g = 0; g < NSG; g++) {
            big = max(big, tg_m[g * MAX_REP + r]);
        }
        // Simdgroups that saw no positions have l = 0 and m = -inf; weight them 0 rather
        // than exp(-inf - -inf).
        float w[NSG];
        float total = 0.0f;
        for (uint g = 0; g < NSG; g++) {
            w[g] = tg_l[g * MAX_REP + r] > 0.0f ? exp(tg_m[g * MAX_REP + r] - big) : 0.0f;
            total += tg_l[g * MAX_REP + r] * w[g];
        }
        out.m[r] = big;
        out.l[r] = total;
        for (uint d = 0; d < dpl; d++) {
            float a = 0.0f;
            for (uint g = 0; g < NSG; g++) {
                a += tg_acc[(g * MAX_REP + r) * MAX_HEAD_DIM + lane * dpl + d] * w[g];
            }
            out.acc[r][d] = a;
        }
    }
    return out;
}

#define DECODE_ARGS(T)                                                                 \
    device const T* q [[buffer(0)]],                                                   \
    device const T* k_cache [[buffer(1)]],                                             \
    device const T* v_cache [[buffer(2)]],                                             \
    device const uint* block_tables [[buffer(3)]],                                     \
    device const uint* ctx_lens [[buffer(4)]],                                         \
    constant uint& n_heads [[buffer(6)]],                                              \
    constant uint& n_kv [[buffer(7)]],                                                 \
    constant uint& head_dim [[buffer(8)]],                                             \
    constant uint& block_size [[buffer(9)]],                                           \
    constant uint& max_blocks [[buffer(10)]],                                          \
    constant float& scale [[buffer(11)]],                                              \
    uint3 tgid [[threadgroup_position_in_grid]],                                       \
    uint sg [[simdgroup_index_in_threadgroup]],                                        \
    uint lane [[thread_index_in_simdgroup]]

#define THREADGROUP_SCRATCH                                                            \
    threadgroup float tg_q[MAX_REP * MAX_HEAD_DIM];                                    \
    threadgroup float tg_m[NSG * MAX_REP];                                             \
    threadgroup float tg_l[NSG * MAX_REP];                                             \
    threadgroup float tg_acc[NSG * MAX_REP * MAX_HEAD_DIM]

// Grid (n_kv, sequences): the whole context in one threadgroup, normalized output.
#define INSTANTIATE_DECODE(NAME, T)                                                    \
kernel void NAME(DECODE_ARGS(T), device T* out [[buffer(5)]])                          \
{                                                                                      \
    THREADGROUP_SCRATCH;                                                               \
    const uint kvh = tgid.x, seq = tgid.y;                                             \
    const Partial p = attend_range<T>(q, k_cache, v_cache,                             \
        block_tables + seq * max_blocks, seq, kvh, n_heads, n_kv, head_dim,            \
        block_size, scale, 0, ctx_lens[seq], tg_q, tg_m, tg_l, tg_acc, sg, lane);            \
    if (sg == 0) {                                                                     \
        const uint n_rep = n_heads / n_kv, dpl = head_dim / 32;                        \
        for (uint r = 0; r < n_rep; r++) {                                             \
            const uint head = kvh * n_rep + r;                                         \
            for (uint d = 0; d < dpl; d++) {                                           \
                out[(seq * n_heads + head) * head_dim + lane * dpl + d] =              \
                    T(p.acc[r][d] / p.l[r]);                                           \
            }                                                                          \
        }                                                                              \
    }                                                                                  \
}

// Grid (n_kv, sequences, partitions of `part` positions). Writes each partition's (m, l) to
// `ml` and unnormalized output to `acc`, indexed by ((seq * n_heads + head) * splits + split).
#define INSTANTIATE_SPLIT(NAME, T)                                                     \
kernel void NAME(DECODE_ARGS(T),                                                       \
    device float* ml [[buffer(5)]],                                                    \
    device float* acc [[buffer(12)]],                                                  \
    constant uint& part [[buffer(13)]],                                                \
    constant uint& splits [[buffer(14)]])                                              \
{                                                                                      \
    THREADGROUP_SCRATCH;                                                               \
    const uint kvh = tgid.x, seq = tgid.y, split = tgid.z;                             \
    const uint ctx = ctx_lens[seq];                                                    \
    const uint start = min(split * part, ctx);                                         \
    const uint end = min(start + part, ctx);                                           \
    const Partial p = attend_range<T>(q, k_cache, v_cache,                             \
        block_tables + seq * max_blocks, seq, kvh, n_heads, n_kv, head_dim,            \
        block_size, scale, start, end, tg_q, tg_m, tg_l, tg_acc, sg, lane);                  \
    if (sg == 0) {                                                                     \
        const uint n_rep = n_heads / n_kv, dpl = head_dim / 32;                        \
        for (uint r = 0; r < n_rep; r++) {                                             \
            const uint idx = (seq * n_heads + kvh * n_rep + r) * splits + split;       \
            if (lane == 0) {                                                           \
                ml[idx * 2] = p.m[r];                                                  \
                ml[idx * 2 + 1] = p.l[r];                                              \
            }                                                                          \
            for (uint d = 0; d < dpl; d++) {                                           \
                acc[idx * head_dim + lane * dpl + d] = p.acc[r][d];                    \
            }                                                                          \
        }                                                                              \
    }                                                                                  \
}

// Grid (n_heads, sequences), head_dim threads: merges the partitions of one query head.
#define INSTANTIATE_REDUCE(NAME, T)                                                    \
kernel void NAME(                                                                      \
    device const float* ml [[buffer(0)]],                                              \
    device const float* acc [[buffer(1)]],                                             \
    device T* out [[buffer(2)]],                                                       \
    constant uint& n_heads [[buffer(3)]],                                              \
    constant uint& head_dim [[buffer(4)]],                                             \
    constant uint& splits [[buffer(5)]],                                               \
    uint2 tgid [[threadgroup_position_in_grid]],                                       \
    uint d [[thread_index_in_threadgroup]])                                            \
{                                                                                      \
    const uint row = tgid.y * n_heads + tgid.x;                                        \
    float big = -INFINITY;                                                             \
    for (uint s = 0; s < splits; s++) {                                                \
        if (ml[(row * splits + s) * 2 + 1] > 0.0f) {                                   \
            big = max(big, ml[(row * splits + s) * 2]);                                \
        }                                                                              \
    }                                                                                  \
    float total = 0.0f, a = 0.0f;                                                      \
    for (uint s = 0; s < splits; s++) {                                                \
        const float l = ml[(row * splits + s) * 2 + 1];                                \
        if (l > 0.0f) {                                                                \
            const float w = exp(ml[(row * splits + s) * 2] - big);                     \
            total += l * w;                                                            \
            a += acc[(row * splits + s) * head_dim + d] * w;                           \
        }                                                                              \
    }                                                                                  \
    out[row * head_dim + d] = T(a / total);                                            \
}

INSTANTIATE_DECODE(paged_attention_decode_f32, float)
INSTANTIATE_DECODE(paged_attention_decode_f16, half)
INSTANTIATE_SPLIT(paged_attention_decode_split_f32, float)
INSTANTIATE_SPLIT(paged_attention_decode_split_f16, half)
INSTANTIATE_REDUCE(paged_attention_reduce_f32, float)
INSTANTIATE_REDUCE(paged_attention_reduce_f16, half)
