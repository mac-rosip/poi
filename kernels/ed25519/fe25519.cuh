#pragma once

// =============================================================================
// fe25519.cuh — Field element arithmetic for Ed25519 (radix-2^51)
// =============================================================================
//
// Ed25519 field: p = 2^255 - 19
// Representation: 5 x int64_t limbs, each < 2^51
//
// Functions: fe25519_add, fe25519_sub, fe25519_mul, fe25519_sqr, etc.
//
// =============================================================================

#include <cstdint>

#define FE25519_LIMBS 5
#define FE25519_MASK 0x7FFFFFFFFFFFFULL  // 2^51 - 1

struct fe25519 {
    int64_t v[FE25519_LIMBS];
};

// Field modulus: 2^255 - 19
__device__ __constant__ int64_t FE25519_MOD[FE25519_LIMBS] = {
    0x7ffffffffffed,  // 2^51 - 19
    0x7ffffffffffff,  // 2^51 - 1
    0x7ffffffffffff,
    0x7ffffffffffff,
    0x7ffffffffffff
};

// Zero
__device__ __constant__ fe25519 FE25519_ZERO = {{0, 0, 0, 0, 0}};

// One
__device__ __constant__ fe25519 FE25519_ONE = {{1, 0, 0, 0, 0}};

// Carry propagation
__device__ __forceinline__ void fe25519_carry(fe25519 &r) {
    int64_t carry = r.v[0] >> 51;
    r.v[0] &= FE25519_MASK;
    r.v[1] += carry;
    carry = r.v[1] >> 51;
    r.v[1] &= FE25519_MASK;
    r.v[2] += carry;
    carry = r.v[2] >> 51;
    r.v[2] &= FE25519_MASK;
    r.v[3] += carry;
    carry = r.v[3] >> 51;
    r.v[3] &= FE25519_MASK;
    r.v[4] += carry;
    carry = r.v[4] >> 51;
    r.v[4] &= FE25519_MASK;
    r.v[0] += 19 * carry;
}

// Reduce modulo p
__device__ __forceinline__ void fe25519_reduce(fe25519 &r) {
    // Simple reduction: subtract p if >= p
    int64_t borrow = 0;
    for (int i = 0; i < FE25519_LIMBS; ++i) {
        int64_t diff = r.v[i] - FE25519_MOD[i] - borrow;
        if (diff < 0) {
            r.v[i] = diff + (1LL << 51);
            borrow = 1;
        } else {
            r.v[i] = diff;
            borrow = 0;
        }
    }
    if (borrow == 0) {
        // r was >= p, keep the subtracted result
    } else {
        // r was < p, restore
        for (int i = 0; i < FE25519_LIMBS; ++i) {
            r.v[i] += FE25519_MOD[i];
        }
    }
}

// Add two field elements
__device__ __forceinline__ void fe25519_add(fe25519 &r, const fe25519 &a, const fe25519 &b) {
    for (int i = 0; i < FE25519_LIMBS; ++i) {
        r.v[i] = a.v[i] + b.v[i];
    }
    fe25519_carry(r);
}

// Subtract two field elements
__device__ __forceinline__ void fe25519_sub(fe25519 &r, const fe25519 &a, const fe25519 &b) {
    for (int i = 0; i < FE25519_LIMBS; ++i) {
        r.v[i] = a.v[i] - b.v[i];
    }
    fe25519_carry(r);
}

// Multiply two field elements
__device__ __forceinline__ void fe25519_mul(fe25519 &r, const fe25519 &a, const fe25519 &b) {
    int64_t t[2*FE25519_LIMBS] = {0};

    // Schoolbook multiplication
    for (int i = 0; i < FE25519_LIMBS; ++i) {
        for (int j = 0; j < FE25519_LIMBS; ++j) {
            t[i+j] += a.v[i] * b.v[j];
        }
    }

    // Reduce modulo 2^255 - 19
    // Multiply by 19 and add to lower limbs
    for (int i = 0; i < FE25519_LIMBS; ++i) {
        int64_t carry = t[i] >> 51;
        t[i] &= FE25519_MASK;
        t[i+1] += carry;
    }

    // Handle overflow
    int64_t overflow = t[FE25519_LIMBS] + 19 * (t[FE25519_LIMBS+1] + 19 * (t[FE25519_LIMBS+2] + 19 * (t[FE25519_LIMBS+3] + 19 * t[FE25519_LIMBS+4])));

    for (int i = 0; i < FE25519_LIMBS; ++i) {
        r.v[i] = t[i] + 19 * overflow;
    }

    fe25519_carry(r);
    fe25519_reduce(r);
}

// Square a field element
__device__ __forceinline__ void fe25519_sqr(fe25519 &r, const fe25519 &a) {
    fe25519_mul(r, a, a);
}

// Negate a field element
__device__ __forceinline__ void fe25519_neg(fe25519 &r, const fe25519 &a) {
    for (int i = 0; i < FE25519_LIMBS; ++i) {
        r.v[i] = -a.v[i];
    }
    fe25519_carry(r);
}

// Copy
__device__ __forceinline__ void fe25519_copy(fe25519 &dst, const fe25519 &src) {
    for (int i = 0; i < FE25519_LIMBS; ++i) {
        dst.v[i] = src.v[i];
    }
}

// Set to zero
__device__ __forceinline__ void fe25519_set_zero(fe25519 &r) {
    fe25519_copy(r, FE25519_ZERO);
}

// Set to one
__device__ __forceinline__ void fe25519_set_one(fe25519 &r) {
    fe25519_copy(r, FE25519_ONE);
}

// Check if zero
__device__ __forceinline__ int fe25519_is_zero(const fe25519 &a) {
    int64_t acc = 0;
    for (int i = 0; i < FE25519_LIMBS; ++i) {
        acc |= a.v[i];
    }
    return acc == 0;
}

// Check if equal
__device__ __forceinline__ int fe25519_equal(const fe25519 &a, const fe25519 &b) {
    fe25519 diff;
    fe25519_sub(diff, a, b);
    return fe25519_is_zero(diff);
}