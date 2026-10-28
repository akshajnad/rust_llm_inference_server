// Small-batch matrix-vector product for Q8_0 weights: y[r, n] = sum_k W[n, k] * x[r, k] for
// R <= 8 input rows. Every weight block is read once and applied to all R rows, so decode with
// a few sequences costs about one weight pass instead of one per row.
//
// A threadgroup of NSG simdgroups owns NSG * NR output rows. K is processed in tiles of TK
// values: the threadgroup first stages the R input slices of the tile in threadgroup memory
// (shared by all its output rows), then lane l of each simdgroup handles slice (l % 4) of
// block (l / 4) of the tile for each of the simdgroup's NR rows.

#include <metal_stdlib>
using namespace metal;

#define QK8_0 32
#define NR 4
#define NSG 8
#define MAX_R 8
#define TK 256

typedef struct {
    half d;
    char qs[QK8_0];
} block_q8_0;

template <uint R>
kernel void q8_0_gemv_rows(
    device const block_q8_0* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float xs[R * TK];
    const uint nb = K / QK8_0;
    const uint n0 = (tg * NSG + sg) * NR;
    const uint tid = sg * 32 + lane;
    const uint sub = lane % 4;
    const uint blk_in_tile = lane / 4;

    float acc[NR][R];
    for (uint i = 0; i < NR; i++) {
        for (uint r = 0; r < R; r++) {
            acc[i][r] = 0.0f;
        }
    }

    for (uint kt = 0; kt < K; kt += TK) {
        for (uint i = tid; i < R * TK; i += NSG * 32) {
            const uint r = i / TK;
            const uint c = kt + i % TK;
            xs[i] = c < K ? x[r * K + c] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint b = kt / QK8_0 + blk_in_tile;
        if (b < nb) {
            const uint off = blk_in_tile * QK8_0 + sub * 8;
            float4 xa[R];
            float4 xb[R];
            for (uint r = 0; r < R; r++) {
                threadgroup const float4* xr = (threadgroup const float4*)(xs + r * TK + off);
                xa[r] = xr[0];
                xb[r] = xr[1];
            }
            for (uint i = 0; i < NR; i++) {
                const uint n = n0 + i;
                if (n >= N) {
                    break;
                }
                device const block_q8_0& blk = w[n * nb + b];
                const float d = float(blk.d);
                device const packed_char4* q = (device const packed_char4*)(blk.qs + sub * 8);
                const float4 wa = float4(char4(q[0])) * d;
                const float4 wb = float4(char4(q[1])) * d;
                for (uint r = 0; r < R; r++) {
                    acc[i][r] += dot(wa, xa[r]) + dot(wb, xb[r]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < NR; i++) {
        const uint n = n0 + i;
        for (uint r = 0; r < R; r++) {
            const float s = simd_sum(acc[i][r]);
            if (lane == 0 && n < N) {
                y[r * N + n] = s;
            }
        }
    }
}

#define INSTANTIATE(R) \
    template [[host_name("q8_0_gemv_rows_" #R)]] kernel void q8_0_gemv_rows<R>( \
        device const block_q8_0*, device const float*, device float*, constant uint&, \
        constant uint&, uint, uint, uint);

INSTANTIATE(2)
INSTANTIATE(3)
INSTANTIATE(4)
INSTANTIATE(5)
INSTANTIATE(6)
INSTANTIATE(7)
INSTANTIATE(8)
