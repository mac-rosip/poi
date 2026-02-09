#pragma once

// =============================================================================
// ripemd160.cuh -- GPU RIPEMD-160 for Bitcoin address derivation
// =============================================================================
//
// Implements RIPEMD-160 hash function as __device__ __forceinline__ functions.
// Optimized for Bitcoin: device_ripemd160_32() for 32-byte SHA-256 output.
// Also provides device_hash160_33() = RIPEMD160(SHA256(33-byte compressed pubkey)).
//
// RIPEMD-160 produces a 20-byte (160-bit) hash. It processes data in 64-byte
// blocks using two parallel computation streams (left and right) of 80 steps
// each, organized into 5 rounds of 16 steps.
//
// =============================================================================

#include <cstdint>
#include "sha256.cuh"

// ---- RIPEMD-160 Constants ----

// Initial hash values
__device__ __constant__ uint32_t ripemd160_H0[5] = {
    0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0
};

// Left-stream round constants
__device__ __constant__ uint32_t ripemd160_KL[5] = {
    0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E
};

// Right-stream round constants
__device__ __constant__ uint32_t ripemd160_KR[5] = {
    0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000
};

// Left-stream message word selection
__device__ __constant__ uint8_t ripemd160_RL[80] = {
    // Round 1
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
    // Round 2
     7,  4, 13,  1, 10,  6, 15,  3, 12,  0,  9,  5,  2, 14, 11,  8,
    // Round 3
     3, 10, 14,  4,  9, 15,  8,  1,  2,  7,  0,  6, 13, 11,  5, 12,
    // Round 4
     1,  9, 11, 10,  0,  8, 12,  4, 13,  3,  7, 15, 14,  5,  6,  2,
    // Round 5
     4,  0,  5,  9,  7, 12,  2, 10, 14,  1,  3,  8, 11,  6, 15, 13
};

// Right-stream message word selection
__device__ __constant__ uint8_t ripemd160_RR[80] = {
    // Round 1
     5, 14,  7,  0,  9,  2, 11,  4, 13,  6, 15,  8,  1, 10,  3, 12,
    // Round 2
     6, 11,  3,  7,  0, 13,  5, 10, 14, 15,  8, 12,  4,  9,  1,  2,
    // Round 3
    15,  5,  1,  3,  7, 14,  6,  9, 11,  8, 12,  2, 10,  0,  4, 13,
    // Round 4
     8,  6,  4,  1,  3, 11, 15,  0,  5, 12,  2, 13,  9,  7, 10, 14,
    // Round 5
    12, 15, 10,  4,  1,  5,  8,  7,  6,  2, 13, 14,  0,  3,  9, 11
};

// Left-stream rotation amounts
__device__ __constant__ uint8_t ripemd160_SL[80] = {
    // Round 1
    11, 14, 15, 12,  5,  8,  7,  9, 11, 13, 14, 15,  6,  7,  9,  8,
    // Round 2
     7,  6,  8, 13, 11,  9,  7, 15,  7, 12, 15,  9, 11,  7, 13, 12,
    // Round 3
    11, 13,  6,  7, 14,  9, 13, 15, 14,  8, 13,  6,  5, 12,  7,  5,
    // Round 4
    11, 12, 14, 15, 14, 15,  9,  8,  9, 14,  5,  6,  8,  6,  5, 12,
    // Round 5
     9, 15,  5, 11,  6,  8, 13, 12,  5, 12, 13, 14, 11,  8,  5,  6
};

// Right-stream rotation amounts
__device__ __constant__ uint8_t ripemd160_SR[80] = {
    // Round 1
     8,  9,  9, 11, 13, 15, 15,  5,  7,  7,  8, 11, 14, 14, 12,  6,
    // Round 2
     9, 13, 15,  7, 12,  8,  9, 11,  7,  7, 12,  7,  6, 15, 13, 11,
    // Round 3
     9,  7, 15, 11,  8,  6,  6, 14, 12, 13,  5, 14, 13, 13,  7,  5,
    // Round 4
    15,  5,  8, 11, 14, 14,  6, 14,  6,  9, 12,  9, 12,  5, 15,  8,
    // Round 5
     8,  5, 12,  9, 12,  5, 14,  6,  8, 13,  6,  5, 15, 13, 11, 11
};

// Rotate left 32-bit
__device__ __forceinline__ uint32_t rotl32(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

// RIPEMD-160 boolean functions (one per round)
__device__ __forceinline__ uint32_t ripemd160_f(int round, uint32_t x, uint32_t y, uint32_t z) {
    switch (round) {
        case 0: return x ^ y ^ z;
        case 1: return (x & y) | (~x & z);
        case 2: return (x | ~y) ^ z;
        case 3: return (x & z) | (y & ~z);
        case 4: return x ^ (y | ~z);
        default: return 0;
    }
}

// =============================================================================
// RIPEMD-160 compression function (single 64-byte block)
// =============================================================================
__device__ __forceinline__ void device_ripemd160_compress(uint32_t H[5], const uint8_t block[64]) {
    // Parse message block into 16 little-endian 32-bit words
    uint32_t X[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        X[i] = (uint32_t)block[i*4]        | ((uint32_t)block[i*4+1] << 8) |
               ((uint32_t)block[i*4+2] << 16) | ((uint32_t)block[i*4+3] << 24);
    }

    // Left stream
    uint32_t al = H[0], bl = H[1], cl = H[2], dl = H[3], el = H[4];
    // Right stream
    uint32_t ar = H[0], br = H[1], cr = H[2], dr = H[3], er = H[4];

    // 80 steps (5 rounds of 16)
    for (int j = 0; j < 80; ++j) {
        int round = j / 16;

        // Left stream
        uint32_t tl = al + ripemd160_f(round, bl, cl, dl) +
                      X[ripemd160_RL[j]] + ripemd160_KL[round];
        tl = rotl32(tl, ripemd160_SL[j]) + el;
        al = el; el = dl; dl = rotl32(cl, 10); cl = bl; bl = tl;

        // Right stream (reversed round order for f: 4,3,2,1,0)
        uint32_t tr = ar + ripemd160_f(4 - round, br, cr, dr) +
                      X[ripemd160_RR[j]] + ripemd160_KR[round];
        tr = rotl32(tr, ripemd160_SR[j]) + er;
        ar = er; er = dr; dr = rotl32(cr, 10); cr = br; br = tr;
    }

    // Final addition
    uint32_t t = H[1] + cl + dr;
    H[1] = H[2] + dl + er;
    H[2] = H[3] + el + ar;
    H[3] = H[4] + al + br;
    H[4] = H[0] + bl + cr;
    H[0] = t;
}

// =============================================================================
// Optimized RIPEMD-160 for 32-byte input (SHA-256 output)
//
// 32 bytes + 1 byte 0x80 + 23 bytes zero + 8 bytes length = 64 bytes (1 block)
// Note: RIPEMD-160 uses little-endian length encoding (unlike SHA-256)
// =============================================================================
__device__ __forceinline__ void device_ripemd160_32(uint8_t output[20], const uint8_t input[32]) {
    uint32_t H[5];
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        H[i] = ripemd160_H0[i];
    }

    uint8_t block[64];

    // Copy input
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        block[i] = input[i];
    }

    // Padding
    block[32] = 0x80;

    #pragma unroll
    for (int i = 33; i < 56; ++i) {
        block[i] = 0;
    }

    // Length: 32 * 8 = 256 bits as little-endian uint64
    block[56] = 0x00; block[57] = 0x01; block[58] = 0x00; block[59] = 0x00;
    block[60] = 0x00; block[61] = 0x00; block[62] = 0x00; block[63] = 0x00;

    device_ripemd160_compress(H, block);

    // Convert H[5] to 20 bytes little-endian
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        output[i*4]     =  H[i]        & 0xFF;
        output[i*4 + 1] = (H[i] >> 8)  & 0xFF;
        output[i*4 + 2] = (H[i] >> 16) & 0xFF;
        output[i*4 + 3] = (H[i] >> 24) & 0xFF;
    }
}

// =============================================================================
// HASH160 = RIPEMD160(SHA256(data)) -- standard Bitcoin address hash
//
// Optimized for 33-byte compressed public key input.
// =============================================================================
__device__ __forceinline__ void device_hash160_33(uint8_t output[20], const uint8_t input[33]) {
    uint8_t sha_out[32];
    device_sha256_33(sha_out, input);
    device_ripemd160_32(output, sha_out);
}
