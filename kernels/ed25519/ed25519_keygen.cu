// =============================================================================
// ed25519_keygen.cu — Ed25519 full keypair generation on GPU
// =============================================================================
//
// Each thread generates a complete Ed25519 keypair from a random seed:
//
//   1. seed (32 bytes, random)
//   2. h = SHA-512(seed) → 64 bytes
//   3. Clamp h[0:32]:
//      - h[0]  &= 248     (clear low 3 bits)
//      - h[31] &= 127     (clear bit 255)
//      - h[31] |= 64      (set bit 254)
//   4. scalar = clamped h[0:32] as little-endian 256-bit integer
//   5. pubkey = scalar * B (base point), using precomputed table
//   6. Compress pubkey to 32 bytes (y-coordinate + sign bit)
//
// The compressed pubkey IS the Solana address (after Base58 encoding).
//
// Dependencies: ge25519.cuh, sha512.cuh, ed25519_precomp.cuh
// =============================================================================

#include "ge25519.cuh"
#include "sha512.cuh"
#include <cstdint>

// Define the precomp table in constant memory
// Note: This is the actual definition; ed25519_precomp.cuh has the extern declaration
#define ED25519_PRECOMP_SIZE 256
__device__ __constant__ ge25519_niels d_ed25519_precomp[ED25519_PRECOMP_SIZE];

// =============================================================================
// Device helper: scalar multiplication using precomputed Niels table
//
// For each bit i of the scalar, if set, add precomp[i] (Niels form)
// to the accumulator (extended form).
// =============================================================================
__device__ __forceinline__ void ed25519_scalarmult_base(
    ge25519_p3 &result,
    const uint8_t scalar[32]
) {
    ge25519_set_identity(result);

    for (int bit = 0; bit < 256; ++bit) {
        int byte_idx = bit / 8;
        int bit_idx = bit % 8;

        if ((scalar[byte_idx] >> bit_idx) & 1) {
            ge25519_p3 sum;
            ge25519_niels_add(sum, result, d_ed25519_precomp[bit]);
            ge25519_copy(result, sum);
        }
    }
}

// =============================================================================
// Kernel: ed25519_keygen
//
// Inputs:
//   seeds[]     — 32-byte random seeds, one per thread
//   num_keys    — total number of keys to generate
//
// Outputs:
//   pubkeys[]   — 32-byte compressed public keys (= Solana address bytes)
//   priv_seeds[] — copy of seeds for result reporting
// =============================================================================
__global__ void ed25519_keygen(
    const uint8_t* __restrict__ seeds,
    uint8_t* __restrict__ pubkeys,
    uint8_t* __restrict__ priv_seeds,
    uint32_t num_keys
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_keys) return;

    const uint8_t* seed = seeds + (size_t)idx * 32;
    uint8_t* pubkey_out = pubkeys + (size_t)idx * 32;
    uint8_t* priv_out = priv_seeds + (size_t)idx * 32;

    // Copy seed to private key output
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        priv_out[i] = seed[i];
    }

    // Step 1: SHA-512(seed)
    uint8_t h[64];
    device_sha512_32(h, seed);

    // Step 2: Clamp the scalar
    h[0]  &= 248;    // Clear low 3 bits
    h[31] &= 127;    // Clear bit 255
    h[31] |= 64;     // Set bit 254

    // Step 3: Scalar multiplication: pubkey = scalar * B
    ge25519_p3 pubpoint;
    ed25519_scalarmult_base(pubpoint, h);

    // Step 4: Compress to 32 bytes
    ge25519_compress(pubkey_out, pubpoint);
}

// =============================================================================
// Host function: generate Ed25519 precomputed table and copy to device
//
// Computes the Niels-form table on GPU:
//   precomp[0] = B (base point) in Niels form
//   precomp[i] = 2 * precomp[i-1] in Niels form
// =============================================================================
__global__ void ed25519_generate_precomp(ge25519_niels* table) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Start with base point B in extended coordinates
    ge25519_p3 current;
    fe25519_copy(current.X, GE25519_BX);
    fe25519_copy(current.Y, GE25519_BY);
    fe25519_set_one(current.Z);
    fe25519_mul(current.T, current.X, current.Y);

    for (int i = 0; i < ED25519_PRECOMP_SIZE; ++i) {
        // Convert current point to Niels form: (Y+X, Y-X, 2*d*T)
        fe25519_add(table[i].yPlusX, current.Y, current.X);
        fe25519_sub(table[i].yMinusX, current.Y, current.X);
        fe25519_mul(table[i].xy2d, current.T, GE25519_2D);

        // Double for next entry
        ge25519_p3 doubled;
        ge25519_double(doubled, current);
        ge25519_copy(current, doubled);
    }
}

extern "C" void ed25519_init_precomp() {
    ge25519_niels* d_table;
    cudaMalloc(&d_table, sizeof(ge25519_niels) * ED25519_PRECOMP_SIZE);

    ed25519_generate_precomp<<<1, 1>>>(d_table);
    cudaDeviceSynchronize();

    cudaMemcpyToSymbol(d_ed25519_precomp, d_table,
                       sizeof(ge25519_niels) * ED25519_PRECOMP_SIZE);

    cudaFree(d_table);
}

// =============================================================================
// Host-callable wrapper
// =============================================================================
extern "C" void ed25519_keygen_launch(
    const uint8_t* d_seeds,
    uint8_t* d_pubkeys,
    uint8_t* d_priv_seeds,
    uint32_t num_keys,
    cudaStream_t stream
) {
    const uint32_t block_size = 256;
    const uint32_t grid_size = (num_keys + block_size - 1) / block_size;

    ed25519_keygen<<<grid_size, block_size, 0, stream>>>(
        d_seeds, d_pubkeys, d_priv_seeds, num_keys
    );
}
