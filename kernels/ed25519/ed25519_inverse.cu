// =============================================================================
// ed25519_inverse.cu — Batch modular inverse for Ed25519 (Algorithm 2.11)
// =============================================================================
//
// Same algorithm as secp256k1_inverse.cu but over the Ed25519 field
// (mod 2^255 - 19) using fe25519 arithmetic.
//
// Converts N points from extended (X,Y,Z,T) to affine (x,y) by
// batch-computing Z^(-1) using Montgomery's trick.
//
// Dependencies: ge25519.cuh (fe25519_mul, fe25519_invert)
// =============================================================================

#include "ge25519.cuh"
#include <cstdint>

// =============================================================================
// Kernel: ed25519_batch_inverse
//
// Inputs/Outputs:
//   points_X[] — X coordinates, updated to X * Z_inv
//   points_Y[] — Y coordinates, updated to Y * Z_inv
//   points_Z[] — Z coordinates, set to 1 after inverse
//   num_points — number of points
// =============================================================================
__global__ void ed25519_batch_inverse(
    fe25519* __restrict__ points_X,
    fe25519* __restrict__ points_Y,
    fe25519* __restrict__ points_Z,
    uint32_t num_points
) {
    extern __shared__ fe25519 shared_products[];

    const uint32_t tid = threadIdx.x;
    const uint32_t chunk_size = blockDim.x;
    const uint32_t chunk_start = blockIdx.x * chunk_size;

    if (chunk_start >= num_points) return;

    const uint32_t chunk_end = min(chunk_start + chunk_size, num_points);
    const uint32_t local_count = chunk_end - chunk_start;

    if (tid >= local_count) return;

    const uint32_t global_idx = chunk_start + tid;

    // Step 1: Load Z values into shared memory
    fe25519_copy(shared_products[tid], points_Z[global_idx]);

    __syncthreads();

    // Serial forward pass (thread 0)
    if (tid == 0) {
        for (uint32_t i = 1; i < local_count; ++i) {
            fe25519 prod;
            fe25519_mul(prod, shared_products[i - 1], shared_products[i]);
            fe25519_copy(shared_products[i], prod);
        }
    }

    __syncthreads();

    // Step 2: Single inversion of total product (thread 0)
    fe25519 total_inv;
    if (tid == 0) {
        fe25519_invert(total_inv, shared_products[local_count - 1]);
    }

    __syncthreads();

    // Step 3: Backward pass (thread 0)
    if (tid == 0) {
        for (int i = (int)local_count - 1; i >= 1; --i) {
            // z_inv[i] = total_inv * products[i-1]
            fe25519 z_inv;
            fe25519_mul(z_inv, total_inv, shared_products[i - 1]);

            // Update total_inv
            fe25519 orig_z;
            fe25519_copy(orig_z, points_Z[chunk_start + i]);
            fe25519_mul(total_inv, total_inv, orig_z);

            // Apply: X = X * z_inv, Y = Y * z_inv
            fe25519 new_X, new_Y;
            fe25519_mul(new_X, points_X[chunk_start + i], z_inv);
            fe25519_mul(new_Y, points_Y[chunk_start + i], z_inv);
            fe25519_copy(points_X[chunk_start + i], new_X);
            fe25519_copy(points_Y[chunk_start + i], new_Y);
            fe25519_set_one(points_Z[chunk_start + i]);
        }

        // First element
        fe25519 new_X, new_Y;
        fe25519_mul(new_X, points_X[chunk_start], total_inv);
        fe25519_mul(new_Y, points_Y[chunk_start], total_inv);
        fe25519_copy(points_X[chunk_start], new_X);
        fe25519_copy(points_Y[chunk_start], new_Y);
        fe25519_set_one(points_Z[chunk_start]);
    }
}

// =============================================================================
// Host-callable wrapper
// =============================================================================
extern "C" void ed25519_batch_inverse_launch(
    fe25519* d_points_X,
    fe25519* d_points_Y,
    fe25519* d_points_Z,
    uint32_t num_points,
    cudaStream_t stream
) {
    const uint32_t chunk_size = 256;
    const uint32_t num_blocks = (num_points + chunk_size - 1) / chunk_size;
    const size_t shared_mem = chunk_size * sizeof(fe25519);

    ed25519_batch_inverse<<<num_blocks, chunk_size, shared_mem, stream>>>(
        d_points_X, d_points_Y, d_points_Z, num_points
    );
}
