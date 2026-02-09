// =============================================================================
// secp256k1_inverse.cu — Batch modular inverse (Algorithm 2.11)
// =============================================================================
//
// Converts N points from Jacobian/projective to affine coordinates
// using Montgomery's trick (Algorithm 2.11):
//
//   Given Z[0..N-1], compute Z[i]^(-1) for all i using:
//     1 inversion + 3*(N-1) multiplications
//
// Steps:
//   1. Forward pass: products[i] = Z[0] * Z[1] * ... * Z[i]
//   2. Single inversion: inv = products[N-1]^(-1)
//   3. Backward pass: recover individual inverses
//      Z[i]^(-1) = inv * products[i-1]
//      inv = inv * Z[i]
//
// This kernel operates on the Z coordinates stored alongside
// the iterate kernel's point arrays. After batch inverse, points
// are in affine form (Z = 1).
//
// Dependencies: secp256k1_ops.cuh (mp_mod_mul, mp_mod_inverse)
// =============================================================================

#include "secp256k1_ops.cuh"
#include <cstdint>

// =============================================================================
// Kernel: secp256k1_batch_inverse
//
// Operates on a contiguous array of Z-coordinates (mp_number).
// Converts all points to affine by computing Z^(-1) for each,
// then multiplying X and Y by Z^(-1).
//
// Since we currently use affine coordinates (Z is implicit 1),
// this kernel is structured for the iterate kernel's Jacobian
// accumulation mode where Z != 1 after additions.
//
// Inputs/Outputs:
//   points_x[] — x coordinates, updated to x * z_inv
//   points_y[] — y coordinates, updated to y * z_inv
//   points_z[] — z coordinates, set to 1 after inverse
//   num_points — number of points
// =============================================================================
__global__ void secp256k1_batch_inverse(
    mp_number* __restrict__ points_x,
    mp_number* __restrict__ points_y,
    mp_number* __restrict__ points_z,
    uint32_t num_points
) {
    // This kernel uses a single block with shared memory for the products array.
    // For large N, it processes in chunks that fit in shared memory.

    extern __shared__ mp_number shared_products[];

    const uint32_t tid = threadIdx.x;
    const uint32_t chunk_size = blockDim.x;
    const uint32_t chunk_start = blockIdx.x * chunk_size;

    if (chunk_start >= num_points) return;

    const uint32_t chunk_end = min(chunk_start + chunk_size, num_points);
    const uint32_t local_count = chunk_end - chunk_start;

    if (tid >= local_count) return;

    const uint32_t global_idx = chunk_start + tid;

    // Step 1: Forward pass — compute running products
    // products[0] = Z[0]
    // products[i] = products[i-1] * Z[i]
    mp_number my_z;
    mp_copy(my_z, points_z[global_idx]);
    mp_copy(shared_products[tid], my_z);

    __syncthreads();

    // Serial forward pass (done by thread 0 in each block)
    if (tid == 0) {
        for (uint32_t i = 1; i < local_count; ++i) {
            mp_mod_mul(shared_products[i], shared_products[i - 1], shared_products[i]);
        }
    }

    __syncthreads();

    // Step 2: Single inversion of the total product
    // (done by thread 0)
    mp_number total_inv;
    if (tid == 0) {
        mp_mod_inverse(total_inv, shared_products[local_count - 1]);
    }

    __syncthreads();

    // Step 3: Backward pass — recover individual inverses
    // (done by thread 0, serial)
    if (tid == 0) {
        for (int i = (int)local_count - 1; i >= 1; --i) {
            mp_number z_inv;
            mp_mod_mul(z_inv, total_inv, shared_products[i - 1]);

            // Update total_inv for next iteration
            mp_number orig_z;
            mp_copy(orig_z, points_z[chunk_start + i]);
            mp_mod_mul(total_inv, total_inv, orig_z);

            // Apply inverse: x = x * z_inv, y = y * z_inv
            mp_number new_x, new_y;
            mp_mod_mul(new_x, points_x[chunk_start + i], z_inv);
            mp_mod_mul(new_y, points_y[chunk_start + i], z_inv);
            mp_copy(points_x[chunk_start + i], new_x);
            mp_copy(points_y[chunk_start + i], new_y);
            mp_set_ui(points_z[chunk_start + i], 1);
        }

        // First element: inv = total_inv (which has accumulated all Z[1..N-1])
        mp_number new_x, new_y;
        mp_mod_mul(new_x, points_x[chunk_start], total_inv);
        mp_mod_mul(new_y, points_y[chunk_start], total_inv);
        mp_copy(points_x[chunk_start], new_x);
        mp_copy(points_y[chunk_start], new_y);
        mp_set_ui(points_z[chunk_start], 1);
    }
}

// =============================================================================
// Host-callable wrapper
// =============================================================================
extern "C" void secp256k1_batch_inverse_launch(
    mp_number* d_points_x,
    mp_number* d_points_y,
    mp_number* d_points_z,
    uint32_t num_points,
    cudaStream_t stream
) {
    // Process in chunks, one block per chunk
    const uint32_t chunk_size = 256;
    const uint32_t num_blocks = (num_points + chunk_size - 1) / chunk_size;
    const size_t shared_mem = chunk_size * sizeof(mp_number);

    secp256k1_batch_inverse<<<num_blocks, chunk_size, shared_mem, stream>>>(
        d_points_x, d_points_y, d_points_z, num_points
    );
}
