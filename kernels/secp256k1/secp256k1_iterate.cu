// =============================================================================
// secp256k1_iterate.cu — Iterative point addition + address hash computation
// =============================================================================
//
// The iterate kernel is the hot inner loop of the vanity address miner.
// Each thread:
//   1. Loads its current point (x, y) from global memory
//   2. Adds the generator G to get the next point: P_{n+1} = P_n + G
//   3. Computes the uncompressed public key (64 bytes: x || y)
//   4. Hashes with Keccak-256 → 32 bytes
//   5. Stores the last 20 bytes as the address hash (for ETH/TRX)
//   6. Updates the point in global memory
//
// The address hash is written to a shared results buffer for scoring.
//
// Dependencies: secp256k1_ops.cuh, keccak.cuh
// =============================================================================

#include "secp256k1_ops.cuh"
#include "../common/keccak.cuh"
#include <cstdint>

// =============================================================================
// Kernel: secp256k1_iterate
//
// Each thread iterates its point by adding G, hashes the new pubkey,
// and stores the 20-byte address hash.
//
// Inputs/Outputs:
//   points_x[]    — x coordinates (read/write)
//   points_y[]    — y coordinates (read/write)
//   hash_out[]    — output: 20-byte address hash per thread (Keccak last 20 bytes)
//   num_points    — total number of threads/points
//   iterations    — number of iterations per kernel launch
// =============================================================================
__global__ void secp256k1_iterate(
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

        // Build uncompressed public key (64 bytes: x || y, big-endian)
        uint8_t pubkey[64];

        // x coordinate: big-endian from mp_number (little-endian words)
        #pragma unroll
        for (int w = 0; w < MP_WORDS; ++w) {
            int byte_pos = (MP_WORDS - 1 - w) * 4;
            pubkey[byte_pos]     = (P.x.d[w] >> 24) & 0xFF;
            pubkey[byte_pos + 1] = (P.x.d[w] >> 16) & 0xFF;
            pubkey[byte_pos + 2] = (P.x.d[w] >> 8) & 0xFF;
            pubkey[byte_pos + 3] = P.x.d[w] & 0xFF;
        }

        // y coordinate
        #pragma unroll
        for (int w = 0; w < MP_WORDS; ++w) {
            int byte_pos = 32 + (MP_WORDS - 1 - w) * 4;
            pubkey[byte_pos]     = (P.y.d[w] >> 24) & 0xFF;
            pubkey[byte_pos + 1] = (P.y.d[w] >> 16) & 0xFF;
            pubkey[byte_pos + 2] = (P.y.d[w] >> 8) & 0xFF;
            pubkey[byte_pos + 3] = P.y.d[w] & 0xFF;
        }

        // Keccak-256 of the 64-byte public key
        ethash_hash hash;
        keccak256_64(hash, pubkey);

        // Store last 20 bytes as address hash (for ETH/TRX)
        // Keccak output bytes [12..31] = last 20 bytes
        uint8_t* out = hash_out + (size_t)idx * 20;
        #pragma unroll
        for (int i = 0; i < 20; ++i) {
            out[i] = hash.bytes[12 + i];
        }
    }

    // Store updated point
    mp_copy(points_x[idx], P.x);
    mp_copy(points_y[idx], P.y);
}

// =============================================================================
// Host-callable wrapper
// =============================================================================
extern "C" void secp256k1_iterate_launch(
    mp_number* d_points_x,
    mp_number* d_points_y,
    uint8_t* d_hash_out,
    uint32_t num_points,
    uint32_t iterations,
    cudaStream_t stream
) {
    const uint32_t block_size = 256;
    const uint32_t grid_size = (num_points + block_size - 1) / block_size;

    secp256k1_iterate<<<grid_size, block_size, 0, stream>>>(
        d_points_x, d_points_y, d_hash_out, num_points, iterations
    );
}
