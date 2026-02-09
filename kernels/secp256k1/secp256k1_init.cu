// =============================================================================
// secp256k1_init.cu — Initialize secp256k1 points from random seeds
// =============================================================================
//
// Each GPU thread gets a unique random seed (256-bit). The init kernel
// computes the corresponding public key point: privkey * G, using the
// precomputed table of generator multiples.
//
// The result is stored as an affine point (x, y) in global memory arrays.
// These points serve as the starting state for the iterate kernel.
//
// Algorithm:
//   For each bit i of the private key (seed):
//     if bit i is set, accumulate precomp[i] via point_add
//
// Dependencies: secp256k1_precomp.cuh, secp256k1_ops.cuh
// =============================================================================

#include "secp256k1_precomp.cuh"
#include "secp256k1_ops.cuh"
#include <cstdint>

// Define the precomp table in constant memory
__device__ __constant__ secp256k1_point d_secp256k1_precomp[SECP256K1_PRECOMP_SIZE];

// =============================================================================
// Kernel: secp256k1_init
//
// Inputs:
//   seeds[]    — array of 256-bit random seeds (one per thread), used as private keys
//   num_points — total number of points to initialize
//
// Outputs:
//   points_x[] — x coordinates of computed public key points
//   points_y[] — y coordinates of computed public key points
//   priv_keys[] — copy of private keys (for result reporting)
// =============================================================================
__global__ void secp256k1_init(
    const mp_number* __restrict__ seeds,
    mp_number* __restrict__ points_x,
    mp_number* __restrict__ points_y,
    mp_number* __restrict__ priv_keys,
    uint32_t num_points
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    // Load private key from seed
    mp_number privkey;
    mp_copy(privkey, seeds[idx]);

    // Store private key for later result reporting
    mp_copy(priv_keys[idx], privkey);

    // Compute pubkey = privkey * G using precomputed table
    // Binary method: for each bit of privkey, if set, add precomp[bit_index]
    secp256k1_point acc;
    point_set_infinity(acc);

    #pragma unroll
    for (int bit = 0; bit < 256; ++bit) {
        int word_idx = bit / 32;
        int bit_idx = bit % 32;

        if ((privkey.d[word_idx] >> bit_idx) & 1) {
            secp256k1_point sum;
            point_add(sum, acc, d_secp256k1_precomp[bit]);
            point_copy(acc, sum);
        }
    }

    // Store the computed affine point
    mp_copy(points_x[idx], acc.x);
    mp_copy(points_y[idx], acc.y);
}

// =============================================================================
// Host function: generate precomputed table and copy to device
//
// Computes the table on GPU using a single thread:
//   precomp[0] = G
//   precomp[i] = 2 * precomp[i-1]  (point doubling)
//
// Then copies result to the __constant__ symbol d_secp256k1_precomp.
// =============================================================================
__global__ void secp256k1_generate_precomp(secp256k1_point* table) {
    // Single thread kernel
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Entry 0: generator point G
    mp_copy(table[0].x, SECP256K1_GX);
    mp_copy(table[0].y, SECP256K1_GY);

    // Entries 1..255: repeated doubling
    for (int i = 1; i < SECP256K1_PRECOMP_SIZE; ++i) {
        point_double(table[i], table[i - 1]);
    }
}

// Host-callable function to initialize the precomputed table
extern "C" void secp256k1_init_precomp() {
    // Allocate temporary device memory for the table
    secp256k1_point* d_table;
    cudaMalloc(&d_table, sizeof(secp256k1_point) * SECP256K1_PRECOMP_SIZE);

    // Generate table on GPU (single thread)
    secp256k1_generate_precomp<<<1, 1>>>(d_table);
    cudaDeviceSynchronize();

    // Copy to constant memory
    cudaMemcpyToSymbol(d_secp256k1_precomp, d_table,
                       sizeof(secp256k1_point) * SECP256K1_PRECOMP_SIZE);

    cudaFree(d_table);
}
