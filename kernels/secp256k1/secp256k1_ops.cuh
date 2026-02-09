#pragma once

// =============================================================================
// secp256k1_ops.cuh — Secp256k1 elliptic curve point operations (CUDA)
// =============================================================================
//
// Curve: y^2 = x^3 + 7 over F_p
// p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
//
// Point representation: affine coordinates (x, y) using mp_number (8x uint32_t)
// Point at infinity: represented with x = 0, y = 0
//
// Functions:
//   point_add      — P + Q (general addition, P != Q)
//   point_double   — 2P
//   point_add_mixed — P (Jacobian) + Q (affine) [for precomp table usage]
//   point_is_infinity — check for point at infinity
//   point_copy     — copy point
//   point_set_infinity — set point to infinity
//
// Dependencies: mp_uint256.cuh
// =============================================================================

#include "../common/mp_uint256.cuh"

// secp256k1 generator point G
// Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
// Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
__device__ __constant__ mp_number SECP256K1_GX = {{
    0x16f81798, // d[0] — LSW
    0x59f2815b,
    0x2dce28d9,
    0x029bfcdb,
    0xce870b07,
    0x55a06295,
    0xf9dcbbac,
    0x79be667e  // d[7] — MSW
}};

__device__ __constant__ mp_number SECP256K1_GY = {{
    0xfb10d4b8, // d[0]
    0x9c47d08f,
    0xa6855419,
    0xfd17b448,
    0x0e1108a8,
    0x5da4fbfc,
    0x26a3c465,
    0x483ada77  // d[7]
}};

// Constant b = 7
__device__ __constant__ mp_number SECP256K1_B = {{
    0x00000007, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
}};

// Constant 2
__device__ __constant__ mp_number MP_TWO = {{
    0x00000002, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
}};

// Constant 3
__device__ __constant__ mp_number MP_THREE = {{
    0x00000003, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
}};

// =============================================================================
// Point at infinity check
// Convention: point at infinity has x = 0, y = 0
// =============================================================================
__device__ __forceinline__ int point_is_infinity(const secp256k1_point &P) {
    return mp_is_zero(P.x) && mp_is_zero(P.y);
}

// =============================================================================
// Copy a point
// =============================================================================
__device__ __forceinline__ void point_copy(secp256k1_point &dst, const secp256k1_point &src) {
    mp_copy(dst.x, src.x);
    mp_copy(dst.y, src.y);
}

// =============================================================================
// Set point to infinity
// =============================================================================
__device__ __forceinline__ void point_set_infinity(secp256k1_point &P) {
    mp_copy(P.x, MP_ZERO);
    mp_copy(P.y, MP_ZERO);
}

// =============================================================================
// point_double — Compute R = 2P
//
// For secp256k1 (a = 0):
//   lambda = (3 * x^2) / (2 * y)        [since a = 0]
//   x_r = lambda^2 - 2 * x
//   y_r = lambda * (x - x_r) - y
//
// If P is infinity or y = 0, returns infinity.
// =============================================================================
__device__ __forceinline__ void point_double(secp256k1_point &R, const secp256k1_point &P) {
    // Check for special cases
    if (point_is_infinity(P) || mp_is_zero(P.y)) {
        point_set_infinity(R);
        return;
    }

    mp_number s, m, t;

    // m = 3 * x^2  (since a = 0 for secp256k1)
    mp_mod_sqr(t, P.x);            // t = x^2
    mp_mod_add(m, t, t);           // m = 2 * x^2
    mp_mod_add(m, m, t);           // m = 3 * x^2

    // s = 2 * y
    mp_mod_add(s, P.y, P.y);       // s = 2y

    // lambda = m * s^(-1) = m / (2y)
    mp_number s_inv;
    mp_mod_inverse(s_inv, s);      // s_inv = (2y)^(-1)

    mp_number lambda;
    mp_mod_mul(lambda, m, s_inv);  // lambda = 3x^2 / (2y)

    // x_r = lambda^2 - 2 * x
    mp_number lambda_sq;
    mp_mod_sqr(lambda_sq, lambda);       // lambda^2
    mp_number two_x;
    mp_mod_add(two_x, P.x, P.x);        // 2x
    mp_mod_sub(R.x, lambda_sq, two_x);   // x_r = lambda^2 - 2x

    // y_r = lambda * (x - x_r) - y
    mp_number dx;
    mp_mod_sub(dx, P.x, R.x);           // x - x_r
    mp_mod_mul(R.y, lambda, dx);         // lambda * (x - x_r)
    mp_mod_sub(R.y, R.y, P.y);          // - y
}

// =============================================================================
// point_add — Compute R = P + Q  (P != Q, neither is infinity)
//
//   lambda = (y_q - y_p) / (x_q - x_p)
//   x_r = lambda^2 - x_p - x_q
//   y_r = lambda * (x_p - x_r) - y_p
//
// Handles identity cases:
//   - P = infinity → R = Q
//   - Q = infinity → R = P
//   - P = Q → delegate to point_double
//   - P = -Q (x equal, y negated) → R = infinity
// =============================================================================
__device__ __forceinline__ void point_add(secp256k1_point &R, const secp256k1_point &P, const secp256k1_point &Q) {
    // Identity cases
    if (point_is_infinity(P)) {
        point_copy(R, Q);
        return;
    }
    if (point_is_infinity(Q)) {
        point_copy(R, P);
        return;
    }

    // Check if P == Q
    if (mp_cmp(P.x, Q.x) == 0) {
        if (mp_cmp(P.y, Q.y) == 0) {
            // P == Q → double
            point_double(R, P);
            return;
        } else {
            // P == -Q → infinity
            point_set_infinity(R);
            return;
        }
    }

    // General case: P != Q
    mp_number dy, dx;
    mp_mod_sub(dy, Q.y, P.y);     // dy = y_q - y_p
    mp_mod_sub(dx, Q.x, P.x);     // dx = x_q - x_p

    mp_number dx_inv;
    mp_mod_inverse(dx_inv, dx);    // dx^(-1)

    mp_number lambda;
    mp_mod_mul(lambda, dy, dx_inv); // lambda = dy / dx

    // x_r = lambda^2 - x_p - x_q
    mp_number lambda_sq;
    mp_mod_sqr(lambda_sq, lambda);
    mp_mod_sub(R.x, lambda_sq, P.x);
    mp_mod_sub(R.x, R.x, Q.x);

    // y_r = lambda * (x_p - x_r) - y_p
    mp_number diff;
    mp_mod_sub(diff, P.x, R.x);
    mp_mod_mul(R.y, lambda, diff);
    mp_mod_sub(R.y, R.y, P.y);
}

// =============================================================================
// Specialized subtractions for common constants (used in iteration kernels)
// =============================================================================

// sub_gx: r = a - Gx (mod p)
__device__ __forceinline__ void mp_sub_gx(mp_number &r, const mp_number &a) {
    mp_mod_sub(r, a, SECP256K1_GX);
}

// sub_gy: r = a - Gy (mod p)
__device__ __forceinline__ void mp_sub_gy(mp_number &r, const mp_number &a) {
    mp_mod_sub(r, a, SECP256K1_GY);
}

// =============================================================================
// point_negate — Negate a point: -P = (x, -y mod p)
// =============================================================================
__device__ __forceinline__ void point_negate(secp256k1_point &R, const secp256k1_point &P) {
    mp_copy(R.x, P.x);
    mp_mod_sub(R.y, MP_ZERO, P.y);  // R.y = p - P.y
}

// =============================================================================
// point_equal — Check if two points are equal
// =============================================================================
__device__ __forceinline__ int point_equal(const secp256k1_point &P, const secp256k1_point &Q) {
    return (mp_cmp(P.x, Q.x) == 0) && (mp_cmp(P.y, Q.y) == 0);
}

// =============================================================================
// point_scalar_mul — Compute R = k * P using double-and-add
//
// This is a basic implementation. The actual keygen kernels use precomputed
// tables for much faster fixed-base multiplication.
// =============================================================================
__device__ __forceinline__ void point_scalar_mul(secp256k1_point &R, const mp_number &k, const secp256k1_point &P) {
    point_set_infinity(R);
    secp256k1_point temp;
    point_copy(temp, P);

    for (int i = 0; i < 256; ++i) {
        int word_idx = i / 32;
        int bit_idx = i % 32;
        if ((k.d[word_idx] >> bit_idx) & 1) {
            secp256k1_point sum;
            point_add(sum, R, temp);
            point_copy(R, sum);
        }
        secp256k1_point doubled;
        point_double(doubled, temp);
        point_copy(temp, doubled);
    }
}
