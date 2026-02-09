#pragma once

// =============================================================================
// keccak.cuh — Keccak-f[1600] permutation and Keccak-256 for Ethereum
// =============================================================================
//
// Ethereum uses Keccak-256 with 0x01 padding (not SHA-3's 0x06).
// This is a CUDA implementation optimized for GPU.
//
// Structs:
// - ethash_hash: union for 32-byte hash output
//
// Functions:
// - sha3_keccakf: Keccak-f[1600] permutation
// - keccak256: General Keccak-256 hash
// - keccak256_64: Optimized for 64-byte input (pubkey)
// - keccak256_64_q: Variant with uint64_t input
//
// =============================================================================

#include <cstdint>

// State size for Keccak-f[1600]
#define KECCAK_STATE_SIZE 25
#define KECCAK_ROUNDS 24

// Keccak-256 output size
#define KECCAK_256_HASH_SIZE 32

// Union for 32-byte hash (Ethereum style)
union ethash_hash {
    uint8_t bytes[KECCAK_256_HASH_SIZE];
    uint32_t words[KECCAK_256_HASH_SIZE / 4];
    uint64_t qwords[KECCAK_256_HASH_SIZE / 8];
};

// Keccak round constants
__device__ __constant__ uint64_t keccak_round_constants[KECCAK_ROUNDS] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// Rotation offsets for rho step
__device__ __constant__ int keccak_rho_offsets[25] = {
    0, 1, 62, 28, 27,
    36, 44, 6, 55, 20,
    3, 10, 43, 25, 39,
    41, 45, 15, 21, 8,
    18, 2, 61, 56, 14
};

// =============================================================================
// sha3_keccakf — Keccak-f[1600] permutation
// =============================================================================
__device__ __forceinline__ void sha3_keccakf(uint64_t state[KECCAK_STATE_SIZE]) {
    #pragma unroll
    for (int round = 0; round < KECCAK_ROUNDS; ++round) {
        // Theta step
        uint64_t C[5], D[5];
        #pragma unroll
        for (int x = 0; x < 5; ++x) {
            C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        #pragma unroll
        for (int x = 0; x < 5; ++x) {
            D[x] = C[(x + 4) % 5] ^ ((C[(x + 1) % 5] << 1) | (C[(x + 1) % 5] >> 63));
        }
        #pragma unroll
        for (int x = 0; x < 5; ++x) {
            #pragma unroll
            for (int y = 0; y < 5; ++y) {
                state[x + 5 * y] ^= D[x];
            }
        }

        // Rho and Pi steps
        uint64_t temp = state[1];
        #pragma unroll
        for (int i = 0; i < 24; ++i) {
            int j = keccak_pi[i];
            uint64_t temp2 = state[j];
            state[j] = (temp << keccak_rho_offsets[j]) | (temp >> (64 - keccak_rho_offsets[j]));
            temp = temp2;
        }

        // Chi step
        #pragma unroll
        for (int y = 0; y < 5; ++y) {
            uint64_t T[5];
            #pragma unroll
            for (int x = 0; x < 5; ++x) {
                T[x] = state[x + 5 * y];
            }
            #pragma unroll
            for (int x = 0; x < 5; ++x) {
                state[x + 5 * y] = T[x] ^ ((~T[(x + 1) % 5]) & T[(x + 2) % 5]);
            }
        }

        // Iota step
        state[0] ^= keccak_round_constants[round];
    }
}

// Pi permutation for rho/pi
__device__ __constant__ int keccak_pi[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

// =============================================================================
// keccak256 — General Keccak-256 hash (Ethereum variant with 0x01 padding)
// =============================================================================
__device__ __forceinline__ void keccak256(ethash_hash &hash, const uint8_t *input, size_t input_size) {
    uint64_t state[KECCAK_STATE_SIZE] = {0};

    // Absorb input
    size_t offset = 0;
    while (input_size >= 136) {  // 136 bytes = 1088 bits, rate for Keccak-256
        #pragma unroll
        for (int i = 0; i < 17; ++i) {  // 17 * 8 = 136 bytes
            uint64_t word = 0;
            for (int j = 0; j < 8; ++j) {
                word |= ((uint64_t)input[offset + i * 8 + j]) << (8 * j);
            }
            state[i] ^= word;
        }
        sha3_keccakf(state);
        offset += 136;
        input_size -= 136;
    }

    // Pad remaining input with Ethereum padding (0x01)
    uint8_t buffer[136] = {0};
    memcpy(buffer, input + offset, input_size);
    buffer[input_size] = 0x01;  // Ethereum padding
    buffer[135] |= 0x80;  // End bit

    #pragma unroll
    for (int i = 0; i < 17; ++i) {
        uint64_t word = 0;
        for (int j = 0; j < 8; ++j) {
            word |= ((uint64_t)buffer[i * 8 + j]) << (8 * j);
        }
        state[i] ^= word;
    }
    sha3_keccakf(state);

    // Squeeze 32 bytes
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        hash.qwords[i] = state[i];
    }
}

// =============================================================================
// keccak256_64 — Optimized Keccak-256 for 64-byte input (e.g., pubkey)
// =============================================================================
__device__ __forceinline__ void keccak256_64(ethash_hash &hash, const uint8_t input[64]) {
    uint64_t state[KECCAK_STATE_SIZE] = {0};

    // Absorb 64 bytes directly
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint64_t word = 0;
        for (int j = 0; j < 8; ++j) {
            word |= ((uint64_t)input[i * 8 + j]) << (8 * j);
        }
        state[i] ^= word;
    }

    // Padding: 0x01 at position 64, 0x80 at end
    state[8] ^= 0x01ULL;
    state[16] ^= 0x8000000000000000ULL;

    sha3_keccakf(state);

    // Squeeze
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        hash.qwords[i] = state[i];
    }
}

// =============================================================================
// keccak256_64_q — Variant with uint64_t input array
// =============================================================================
__device__ __forceinline__ void keccak256_64_q(ethash_hash &hash, const uint64_t input[8]) {
    uint64_t state[KECCAK_STATE_SIZE] = {0};

    // Absorb 8 uint64_t
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        state[i] ^= input[i];
    }

    // Padding
    state[8] ^= 0x01ULL;
    state[16] ^= 0x8000000000000000ULL;

    sha3_keccakf(state);

    // Squeeze
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        hash.qwords[i] = state[i];
    }
}