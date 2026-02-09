// =============================================================================
// ed25519_iterate.cu — Iterative Ed25519 point addition for address mining
// =============================================================================
//
// Each thread holds a compressed public key (32 bytes). On each iteration:
//   1. Add the base point B to the current extended point
//   2. Compress the result to 32 bytes
//   3. Store the compressed pubkey for scoring
//
// For Solana, the 32-byte compressed pubkey IS the address (after Base58).
//
// Dependencies: ge25519.cuh, ed25519_precomp.cuh
// =============================================================================

#include "ge25519.cuh"
#include "ed25519_precomp.cuh"
#include <cstdint>

// Base point B in Niels form (for fast mixed addition)
// This is precomp[0] — already available in the precomp table.

// =============================================================================
// Kernel: ed25519_iterate
//
// Each thread iterates its point by adding B (base point).
//
// We keep points in extended coordinates between iterations to avoid
// repeated compression/decompression. Only compress for the final output.
//
// Inputs/Outputs:
//   points_X[], points_Y[], points_Z[], points_T[] — extended coords (read/write)
//   pubkeys[]   — 32-byte compressed output per thread
//   num_points  — total number of threads/points
//   iterations  — number of iterations per kernel launch
// =============================================================================
__global__ void ed25519_iterate(
    fe25519* __restrict__ points_X,
    fe25519* __restrict__ points_Y,
    fe25519* __restrict__ points_Z,
    fe25519* __restrict__ points_T,
    uint8_t* __restrict__ pubkeys,
    uint32_t num_points,
    uint32_t iterations
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    // Load current point in extended coordinates
    ge25519_p3 P;
    fe25519_copy(P.X, points_X[idx]);
    fe25519_copy(P.Y, points_Y[idx]);
    fe25519_copy(P.Z, points_Z[idx]);
    fe25519_copy(P.T, points_T[idx]);

    // Base point in Niels form = precomp[0]
    // d_ed25519_precomp[0] holds B in Niels form

    for (uint32_t iter = 0; iter < iterations; ++iter) {
        // P = P + B (mixed addition with Niels-form base point)
        ge25519_p3 next;
        ge25519_niels_add(next, P, d_ed25519_precomp[0]);
        ge25519_copy(P, next);
    }

    // Store updated extended coordinates
    fe25519_copy(points_X[idx], P.X);
    fe25519_copy(points_Y[idx], P.Y);
    fe25519_copy(points_Z[idx], P.Z);
    fe25519_copy(points_T[idx], P.T);

    // Compress to 32-byte pubkey for scoring
    uint8_t* pubkey_out = pubkeys + (size_t)idx * 32;
    ge25519_compress(pubkey_out, P);
}

// =============================================================================
// Host-callable wrapper
// =============================================================================
extern "C" void ed25519_iterate_launch(
    fe25519* d_points_X,
    fe25519* d_points_Y,
    fe25519* d_points_Z,
    fe25519* d_points_T,
    uint8_t* d_pubkeys,
    uint32_t num_points,
    uint32_t iterations,
    cudaStream_t stream
) {
    const uint32_t block_size = 256;
    const uint32_t grid_size = (num_points + block_size - 1) / block_size;

    ed25519_iterate<<<grid_size, block_size, 0, stream>>>(
        d_points_X, d_points_Y, d_points_Z, d_points_T,
        d_pubkeys, num_points, iterations
    );
}
