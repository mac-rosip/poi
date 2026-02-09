// =============================================================================
// secp256k1_iterate_bitcoin.cu -- Bitcoin iterate kernel (compressed pubkey)
// =============================================================================
//
// Same hot loop as secp256k1_iterate.cu but with Bitcoin-specific hashing:
//   1. Loads current point P from global memory
//   2. Adds generator G: P_{n+1} = P_n + G
//   3. Builds COMPRESSED public key (33 bytes: 0x02/0x03 || x_big_endian)
//   4. HASH160 = RIPEMD160(SHA256(compressed_pubkey)) -> 20 bytes
//   5. Stores 20-byte address hash (same output format as ETH/TRX)
//   6. Updates point in global memory
//
// Dependencies: secp256k1_ops.cuh, sha256.cuh, ripemd160.cuh
// =============================================================================

#include "secp256k1_ops.cuh"
#include "../common/ripemd160.cuh"
#include <cstdint>

// =============================================================================
// Kernel: secp256k1_iterate_bitcoin
//
// Each thread iterates its point by adding G, hashes the compressed pubkey
// with HASH160, and stores the 20-byte address hash.
//
// Inputs/Outputs:
//   points_x[]    -- x coordinates (read/write)
//   points_y[]    -- y coordinates (read/write)
//   hash_out[]    -- output: 20-byte address hash per thread
//   num_points    -- total number of threads/points
//   iterations    -- number of iterations per kernel launch
// =============================================================================
__global__ void secp256k1_iterate_bitcoin(
    mp_number* __restrict__ points_x,
    mp_number* __restrict__ points_y,
    uint8_t* __restrict__ hash_out,
    uint32_t num_points,
    uint32_t iterations
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    // Load current point
    secp256k1_point P;
    mp_copy(P.x, points_x[idx]);
    mp_copy(P.y, points_y[idx]);

    // Generator point
    secp256k1_point G;
    mp_copy(G.x, SECP256K1_GX);
    mp_copy(G.y, SECP256K1_GY);

    for (uint32_t iter = 0; iter < iterations; ++iter) {
        // Add G: P = P + G
        secp256k1_point next;
        point_add(next, P, G);
        point_copy(P, next);

        // Build COMPRESSED public key (33 bytes)
        // Format: [0x02 if y even, 0x03 if y odd] || x_big_endian
        uint8_t pubkey[33];

        // Prefix byte: parity of y (LSB of first word in little-endian)
        pubkey[0] = (P.y.d[0] & 1) ? 0x03 : 0x02;

        // x coordinate: big-endian from mp_number (little-endian words)
        #pragma unroll
        for (int w = 0; w < MP_WORDS; ++w) {
            int byte_pos = 1 + (MP_WORDS - 1 - w) * 4;
            pubkey[byte_pos]     = (P.x.d[w] >> 24) & 0xFF;
            pubkey[byte_pos + 1] = (P.x.d[w] >> 16) & 0xFF;
            pubkey[byte_pos + 2] = (P.x.d[w] >> 8) & 0xFF;
            pubkey[byte_pos + 3] = P.x.d[w] & 0xFF;
        }

        // HASH160: SHA-256 -> RIPEMD-160
        uint8_t hash[20];
        device_hash160_33(hash, pubkey);

        // Store 20-byte result (same format as ETH/TRX)
        uint8_t* out = hash_out + (size_t)idx * 20;
        #pragma unroll
        for (int i = 0; i < 20; ++i) {
            out[i] = hash[i];
        }
    }

    // Store updated point
    mp_copy(points_x[idx], P.x);
    mp_copy(points_y[idx], P.y);
}

// =============================================================================
// Host-callable wrapper
// =============================================================================
extern "C" void secp256k1_iterate_bitcoin_launch(
    mp_number* d_points_x,
    mp_number* d_points_y,
    uint8_t* d_hash_out,
    uint32_t num_points,
    uint32_t iterations,
    cudaStream_t stream
) {
    const uint32_t block_size = 256;
    const uint32_t grid_size = (num_points + block_size - 1) / block_size;

    secp256k1_iterate_bitcoin<<<grid_size, block_size, 0, stream>>>(
        d_points_x, d_points_y, d_hash_out, num_points, iterations
    );
}
