#pragma once

// =============================================================================
// sha256.cuh -- GPU SHA-256 for Bitcoin address derivation
// =============================================================================
//
// Optimized for Bitcoin: device_sha256_33() for 33-byte compressed pubkey input.
// Single-block SHA-256 computation (33 + 9 padding = 42 < 64 bytes).
//
// Port of src/crypto/sha256.cpp to __device__ __forceinline__ functions.
//
// =============================================================================

#include <cstdint>

// SHA-256 round constants
__device__ __constant__ uint32_t sha256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// Initial hash values
__device__ __constant__ uint32_t sha256_H0[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

// Rotate right 32-bit
__device__ __forceinline__ uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

// SHA-256 sigma functions
__device__ __forceinline__ uint32_t sha256_sigma0(uint32_t x) {
    return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22);
}

__device__ __forceinline__ uint32_t sha256_sigma1(uint32_t x) {
    return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25);
}

__device__ __forceinline__ uint32_t sha256_gamma0(uint32_t x) {
    return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t sha256_gamma1(uint32_t x) {
    return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10);
}

// SHA-256 compression function (GPU version)
__device__ __forceinline__ void device_sha256_compress(uint32_t H[8], const uint8_t block[64]) {
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;

    // Prepare message schedule
    #pragma unroll
    for (int t = 0; t < 16; ++t) {
        W[t] = ((uint32_t)block[t*4] << 24) | ((uint32_t)block[t*4+1] << 16) |
               ((uint32_t)block[t*4+2] << 8) | (uint32_t)block[t*4+3];
    }
    #pragma unroll
    for (int t = 16; t < 64; ++t) {
        W[t] = sha256_gamma1(W[t-2]) + W[t-7] + sha256_gamma0(W[t-15]) + W[t-16];
    }

    // Initialize working variables
    a = H[0]; b = H[1]; c = H[2]; d = H[3];
    e = H[4]; f = H[5]; g = H[6]; h = H[7];

    // Main loop
    #pragma unroll
    for (int t = 0; t < 64; ++t) {
        uint32_t T1 = h + sha256_sigma1(e) + ((e & f) ^ (~e & g)) + sha256_K[t] + W[t];
        uint32_t T2 = sha256_sigma0(a) + ((a & b) ^ (a & c) ^ (b & c));
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }

    // Add to hash
    H[0] += a; H[1] += b; H[2] += c; H[3] += d;
    H[4] += e; H[5] += f; H[6] += g; H[7] += h;
}

// =============================================================================
// Optimized SHA-256 for 33-byte input (compressed pubkey)
//
// 33 bytes + 1 byte 0x80 + 22 bytes zero + 8 bytes length = 64 bytes (1 block)
// =============================================================================
__device__ __forceinline__ void device_sha256_33(uint8_t output[32], const uint8_t input[33]) {
    uint32_t H[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        H[i] = sha256_H0[i];
    }

    // Single block: 33 bytes input + padding
    uint8_t block[64];

    // Copy input
    #pragma unroll
    for (int i = 0; i < 33; ++i) {
        block[i] = input[i];
    }

    // Padding: 0x80 after input
    block[33] = 0x80;

    // Zero pad bytes 34-55
    #pragma unroll
    for (int i = 34; i < 56; ++i) {
        block[i] = 0;
    }

    // Length: 33 * 8 = 264 bits = 0x108 as big-endian uint64
    block[56] = 0x00; block[57] = 0x00; block[58] = 0x00; block[59] = 0x00;
    block[60] = 0x00; block[61] = 0x00; block[62] = 0x01; block[63] = 0x08;

    device_sha256_compress(H, block);

    // Convert H[8] to 32 bytes big-endian
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        output[i*4]     = (H[i] >> 24) & 0xFF;
        output[i*4 + 1] = (H[i] >> 16) & 0xFF;
        output[i*4 + 2] = (H[i] >> 8)  & 0xFF;
        output[i*4 + 3] =  H[i]        & 0xFF;
    }
}

// =============================================================================
// Optimized SHA-256 for 32-byte input (used by RIPEMD-160 pipeline)
//
// 32 bytes + 1 byte 0x80 + 23 bytes zero + 8 bytes length = 64 bytes (1 block)
// =============================================================================
__device__ __forceinline__ void device_sha256_32(uint8_t output[32], const uint8_t input[32]) {
    uint32_t H[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        H[i] = sha256_H0[i];
    }

    uint8_t block[64];

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        block[i] = input[i];
    }

    block[32] = 0x80;

    #pragma unroll
    for (int i = 33; i < 56; ++i) {
        block[i] = 0;
    }

    // Length: 32 * 8 = 256 bits = 0x100 as big-endian uint64
    block[56] = 0x00; block[57] = 0x00; block[58] = 0x00; block[59] = 0x00;
    block[60] = 0x00; block[61] = 0x00; block[62] = 0x01; block[63] = 0x00;

    device_sha256_compress(H, block);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        output[i*4]     = (H[i] >> 24) & 0xFF;
        output[i*4 + 1] = (H[i] >> 16) & 0xFF;
        output[i*4 + 2] = (H[i] >> 8)  & 0xFF;
        output[i*4 + 3] =  H[i]        & 0xFF;
    }
}
