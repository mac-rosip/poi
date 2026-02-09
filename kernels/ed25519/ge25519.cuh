#pragma once

// =============================================================================
// ge25519.cuh — Ed25519 group element operations (CUDA)
// =============================================================================
//
// Curve: -x^2 + y^2 = 1 + d*x^2*y^2  (twisted Edwards)
// d = -121665/121666
// Base point B = (Bx, By)
//
// Coordinate systems:
//   Extended:  (X, Y, Z, T) where x = X/Z, y = Y/Z, T = X*Y/Z
//   Niels:     (Y+X, Y-X, 2*d*T) — precomputed form for fast mixed add
//   Affine:    (x, y) — used for final output / compression
//
// Functions:
//   ge25519_add            — P + Q in extended coordinates
//   ge25519_double         — 2P in extended coordinates
//   ge25519_niels_add      — P (extended) + Q (Niels) mixed addition
//   ge25519_compress       — extended → 32-byte compressed point
//   ge25519_set_identity   — set to identity (0, 1, 1, 0)
//   ge25519_copy           — copy extended point
//
// Dependencies: fe25519.cuh
// =============================================================================

#include "fe25519.cuh"

// Ed25519 extended point: (X, Y, Z, T) where x=X/Z, y=Y/Z, T=XY/Z
struct ge25519_p3 {
    fe25519 X;
    fe25519 Y;
    fe25519 Z;
    fe25519 T;
};

// Niels form for precomputed points: (Y+X, Y-X, 2*d*T)
struct ge25519_niels {
    fe25519 yPlusX;
    fe25519 yMinusX;
    fe25519 xy2d;
};

// Projective point (used as intermediate in doubling): (X, Y, Z)
struct ge25519_p2 {
    fe25519 X;
    fe25519 Y;
    fe25519 Z;
};

// Completed point (used as intermediate): (X, Y, Z, T) different semantics
struct ge25519_p1p1 {
    fe25519 X;
    fe25519 Y;
    fe25519 Z;
    fe25519 T;
};

// -----------------------------------------------------------------------------
// Curve constant d = -121665/121666 mod p
// d = 37095705934669439343138083508754565189542113879843219016388785533085940283555
// In radix-2^51 limbs:
// -----------------------------------------------------------------------------
__device__ __constant__ fe25519 GE25519_D = {{
    0x00034dca135978a3LL,
    0x0001a8283b156ebdLL,
    0x0005e7a26001c029LL,
    0x000739c663a03cbbLL,
    0x00052036cee2b6ffLL
}};

// 2*d
__device__ __constant__ fe25519 GE25519_2D = {{
    0x00069b9426b2f159LL,
    0x00035050762ad5daLL,
    0x0003cf44c0038052LL,
    0x0006738cc7407977LL,
    0x0002406d9dc56dffLL
}};

// Base point B
// Bx = 15112221349535807912866137220509078750507884956996801397370759645944942038944
// By = 46316835694926478169428394003475163141307993866256225615783033890098355764399
__device__ __constant__ fe25519 GE25519_BX = {{
    0x00062d608f25d51aLL,
    0x000412a4b4f6592aLL,
    0x00075b7171a4b31dLL,
    0x0001ff60ad35050dLL,
    0x00009c30d5392af3LL
}};

__device__ __constant__ fe25519 GE25519_BY = {{
    0x0006666666666658LL,
    0x0004ccccccccccccLL,
    0x0001999999999999LL,
    0x0003333333333333LL,
    0x0006666666666666LL
}};

// =============================================================================
// Identity point: (0, 1, 1, 0)
// =============================================================================
__device__ __forceinline__ void ge25519_set_identity(ge25519_p3 &P) {
    fe25519_set_zero(P.X);
    fe25519_set_one(P.Y);
    fe25519_set_one(P.Z);
    fe25519_set_zero(P.T);
}

// =============================================================================
// Copy an extended point
// =============================================================================
__device__ __forceinline__ void ge25519_copy(ge25519_p3 &dst, const ge25519_p3 &src) {
    fe25519_copy(dst.X, src.X);
    fe25519_copy(dst.Y, src.Y);
    fe25519_copy(dst.Z, src.Z);
    fe25519_copy(dst.T, src.T);
}

// =============================================================================
// Convert p1p1 (completed) → p3 (extended)
//   X = X_c * T_c
//   Y = Y_c * Z_c
//   Z = Z_c * T_c
//   T = X_c * Y_c
// =============================================================================
__device__ __forceinline__ void ge25519_p1p1_to_p3(ge25519_p3 &R, const ge25519_p1p1 &P) {
    fe25519_mul(R.X, P.X, P.T);
    fe25519_mul(R.Y, P.Y, P.Z);
    fe25519_mul(R.Z, P.Z, P.T);
    fe25519_mul(R.T, P.X, P.Y);
}

// =============================================================================
// Convert p1p1 (completed) → p2 (projective, drops T)
//   X = X_c * T_c
//   Y = Y_c * Z_c
//   Z = Z_c * T_c
// =============================================================================
__device__ __forceinline__ void ge25519_p1p1_to_p2(ge25519_p2 &R, const ge25519_p1p1 &P) {
    fe25519_mul(R.X, P.X, P.T);
    fe25519_mul(R.Y, P.Y, P.Z);
    fe25519_mul(R.Z, P.Z, P.T);
}

// =============================================================================
// ge25519_double — Compute R = 2P
//
// Algorithm from "Twisted Edwards Curves Revisited" (HWCD08)
// Input:  P in extended coordinates (X, Y, Z, T)
// Output: R in p1p1 (completed) form, then convert to extended
//
// A = X^2
// B = Y^2
// C = 2*Z^2
// D = -A   (= a*A where a = -1 for Ed25519)
// E = (X+Y)^2 - A - B
// G = D + B
// F = G - C
// H = D - B
// X_r = E * F
// Y_r = G * H
// T_r = E * H
// Z_r = F * G
// =============================================================================
__device__ __forceinline__ void ge25519_double(ge25519_p3 &R, const ge25519_p3 &P) {
    ge25519_p1p1 t;
    fe25519 A, B, C, D, E, F, G, H;
    fe25519 xpy;

    fe25519_sqr(A, P.X);                   // A = X^2
    fe25519_sqr(B, P.Y);                   // B = Y^2
    fe25519_sqr(C, P.Z);                   // C0 = Z^2
    fe25519_add(C, C, C);                  // C = 2*Z^2
    fe25519_neg(D, A);                     // D = -A  (a = -1)
    fe25519_add(xpy, P.X, P.Y);           // xpy = X + Y
    fe25519_sqr(E, xpy);                   // E = (X+Y)^2
    fe25519_sub(E, E, A);                  // E -= A
    fe25519_sub(E, E, B);                  // E -= B  → E = (X+Y)^2 - A - B
    fe25519_add(G, D, B);                  // G = D + B
    fe25519_sub(F, G, C);                  // F = G - C
    fe25519_sub(H, D, B);                  // H = D - B

    t.X = E;
    t.Y = G;
    t.Z = F;
    t.T = H;

    // Convert to p3: X=E*H, Y=G*F, Z=F*H (wait, let's use the completed form)
    // p1p1 → p3: R.X = t.X * t.T, R.Y = t.Y * t.Z, R.Z = t.Z * t.T, R.T = t.X * t.Y
    fe25519_mul(R.X, E, H);    // X_r = E * H ... wait
    // Actually the standard formulas give:
    // X3 = E*F, Y3 = G*H, Z3 = F*G, T3 = E*H
    fe25519_mul(R.X, E, F);
    fe25519_mul(R.Y, G, H);
    fe25519_mul(R.Z, F, G);
    fe25519_mul(R.T, E, H);
}

// =============================================================================
// ge25519_add — Compute R = P + Q (both in extended coordinates)
//
// Algorithm: Unified addition from HWCD08
// Input:  P, Q in extended coordinates
// Output: R in extended coordinates
//
// A = X1 * X2
// B = Y1 * Y2
// C = T1 * 2d * T2
// D = Z1 * 2 * Z2  (= 2 * Z1 * Z2)
// E = (X1+Y1)*(X2+Y2) - A - B
// F = D - C
// G = D + C
// H = B + A   (note: B - a*A = B + A since a = -1)
// X3 = E * F
// Y3 = G * H
// T3 = E * H
// Z3 = F * G
// =============================================================================
__device__ __forceinline__ void ge25519_add(ge25519_p3 &R, const ge25519_p3 &P, const ge25519_p3 &Q) {
    fe25519 A, B, C, D, E, F, G, H;
    fe25519 temp;

    fe25519_mul(A, P.X, Q.X);              // A = X1 * X2
    fe25519_mul(B, P.Y, Q.Y);              // B = Y1 * Y2
    fe25519_mul(C, P.T, Q.T);              // C' = T1 * T2
    fe25519_mul(C, C, GE25519_2D);          // C = 2d * T1 * T2
    fe25519_mul(D, P.Z, Q.Z);              // D' = Z1 * Z2
    fe25519_add(D, D, D);                  // D = 2 * Z1 * Z2

    // E = (X1+Y1)*(X2+Y2) - A - B
    fe25519_add(E, P.X, P.Y);
    fe25519_add(temp, Q.X, Q.Y);
    fe25519_mul(E, E, temp);
    fe25519_sub(E, E, A);
    fe25519_sub(E, E, B);

    fe25519_sub(F, D, C);                  // F = D - C
    fe25519_add(G, D, C);                  // G = D + C
    fe25519_add(H, B, A);                  // H = B + A  (since a = -1: B - aA = B + A)

    fe25519_mul(R.X, E, F);                // X3 = E * F
    fe25519_mul(R.Y, G, H);                // Y3 = G * H
    fe25519_mul(R.Z, F, G);                // Z3 = F * G
    fe25519_mul(R.T, E, H);                // T3 = E * H
}

// =============================================================================
// ge25519_niels_add — Mixed addition: P (extended) + Q (Niels form)
//
// Niels form: (Y+X, Y-X, 2dT)
// This avoids 1 multiplication compared to full extended addition.
//
// A = (Y1 - X1) * (Y2 - X2)    →  (Y1-X1) * q.yMinusX
// B = (Y1 + X1) * (Y2 + X2)    →  (Y1+X1) * q.yPlusX
// C = T1 * 2dT2                →  T1 * q.xy2d
// D = 2 * Z1  (Z2 is implicitly 1 for affine precomp)
// E = B - A
// F = D - C
// G = D + C
// H = B + A
// X3 = E * F
// Y3 = G * H
// T3 = E * H
// Z3 = F * G
// =============================================================================
__device__ __forceinline__ void ge25519_niels_add(ge25519_p3 &R, const ge25519_p3 &P, const ge25519_niels &Q) {
    fe25519 A, B, C, D, E, F, G, H;
    fe25519 ymx, ypx;

    fe25519_sub(ymx, P.Y, P.X);            // Y1 - X1
    fe25519_add(ypx, P.Y, P.X);            // Y1 + X1
    fe25519_mul(A, ymx, Q.yMinusX);         // A = (Y1-X1) * (Y2-X2)
    fe25519_mul(B, ypx, Q.yPlusX);          // B = (Y1+X1) * (Y2+X2)
    fe25519_mul(C, P.T, Q.xy2d);            // C = T1 * 2dT2
    fe25519_add(D, P.Z, P.Z);              // D = 2 * Z1

    fe25519_sub(E, B, A);                  // E = B - A
    fe25519_sub(F, D, C);                  // F = D - C
    fe25519_add(G, D, C);                  // G = D + C
    fe25519_add(H, B, A);                  // H = B + A

    fe25519_mul(R.X, E, F);                // X3 = E * F
    fe25519_mul(R.Y, G, H);                // Y3 = G * H
    fe25519_mul(R.Z, F, G);                // Z3 = F * G
    fe25519_mul(R.T, E, H);                // T3 = E * H
}

// =============================================================================
// ge25519_niels_sub — Mixed subtraction: P - Q (Niels)
//
// Same as niels_add but swap yPlusX and yMinusX, and negate xy2d.
// =============================================================================
__device__ __forceinline__ void ge25519_niels_sub(ge25519_p3 &R, const ge25519_p3 &P, const ge25519_niels &Q) {
    fe25519 A, B, C, D, E, F, G, H;
    fe25519 ymx, ypx;

    fe25519_sub(ymx, P.Y, P.X);
    fe25519_add(ypx, P.Y, P.X);
    fe25519_mul(A, ymx, Q.yPlusX);          // Swapped: use yPlusX
    fe25519_mul(B, ypx, Q.yMinusX);         // Swapped: use yMinusX
    fe25519_mul(C, P.T, Q.xy2d);
    fe25519_neg(C, C);                      // Negate C
    fe25519_add(D, P.Z, P.Z);

    fe25519_sub(E, B, A);
    fe25519_sub(F, D, C);
    fe25519_add(G, D, C);
    fe25519_add(H, B, A);

    fe25519_mul(R.X, E, F);
    fe25519_mul(R.Y, G, H);
    fe25519_mul(R.Z, F, G);
    fe25519_mul(R.T, E, H);
}

// =============================================================================
// ge25519_compress — Compress extended point to 32 bytes
//
// Ed25519 encoding: little-endian y coordinate with sign bit of x in bit 255.
//   1. Convert from extended to affine: x = X/Z, y = Y/Z
//   2. Encode y as 32 bytes little-endian
//   3. Set high bit of byte 31 to the low bit of x
// =============================================================================
__device__ __forceinline__ void ge25519_compress(uint8_t out[32], const ge25519_p3 &P) {
    fe25519 recip, x, y;

    // Compute Z^(-1)
    fe25519_invert(recip, P.Z);

    // x = X * Z^(-1), y = Y * Z^(-1)
    fe25519_mul(x, P.X, recip);
    fe25519_mul(y, P.Y, recip);

    // Fully reduce
    fe25519_carry(x);
    fe25519_reduce(x);
    fe25519_carry(y);
    fe25519_reduce(y);

    // Encode y as little-endian 32 bytes
    // Each limb is 51 bits; pack into bytes
    uint64_t y_bits[4];
    y_bits[0] = (uint64_t)y.v[0] | ((uint64_t)y.v[1] << 51);
    y_bits[1] = ((uint64_t)y.v[1] >> 13) | ((uint64_t)y.v[2] << 38);
    y_bits[2] = ((uint64_t)y.v[2] >> 26) | ((uint64_t)y.v[3] << 25);
    y_bits[3] = ((uint64_t)y.v[3] >> 39) | ((uint64_t)y.v[4] << 12);

    // Write as little-endian bytes
    for (int i = 0; i < 8; ++i) out[i]      = (y_bits[0] >> (8 * i)) & 0xFF;
    for (int i = 0; i < 8; ++i) out[8 + i]  = (y_bits[1] >> (8 * i)) & 0xFF;
    for (int i = 0; i < 8; ++i) out[16 + i] = (y_bits[2] >> (8 * i)) & 0xFF;
    for (int i = 0; i < 8; ++i) out[24 + i] = (y_bits[3] >> (8 * i)) & 0xFF;

    // Set sign bit: bit 255 = low bit of x
    out[31] |= ((uint8_t)(x.v[0] & 1)) << 7;
}

// =============================================================================
// fe25519_invert — Modular inverse using Fermat's little theorem
//   a^(-1) = a^(p-2) mod p, where p = 2^255 - 19
//   Uses an addition chain for p-2 = 2^255 - 21
// =============================================================================
__device__ __forceinline__ void fe25519_invert(fe25519 &r, const fe25519 &z) {
    fe25519 t0, t1, t2, t3;

    // z^2
    fe25519_sqr(t0, z);

    // z^(2^2) = z^4
    fe25519_sqr(t1, t0);
    fe25519_sqr(t1, t1);

    // z^9
    fe25519_mul(t1, z, t1);     // t1 = z * z^8 = z^9

    // z^11
    fe25519_mul(t0, t0, t1);    // t0 = z^2 * z^9 = z^11

    // z^(2^1 * 11) = z^22
    fe25519_sqr(t2, t0);        // t2 = z^22

    // z^(2^1 * 11 + 9) = z^31
    fe25519_mul(t1, t1, t2);    // t1 = z^9 * z^22 = z^31

    // z^(2^5 * 31) = z^(31 * 32)
    fe25519_sqr(t2, t1);
    for (int i = 1; i < 5; ++i) fe25519_sqr(t2, t2);

    // z^(2^5 * 31 + 31) = z^(32*31 + 31) = z^(1023)
    fe25519_mul(t1, t2, t1);

    // z^(2^10 * 1023)
    fe25519_sqr(t2, t1);
    for (int i = 1; i < 10; ++i) fe25519_sqr(t2, t2);

    // z^(2^10 * 1023 + 1023) = z^(2^20 - 1)
    fe25519_mul(t2, t2, t1);

    // z^(2^20 * (2^20 - 1))
    fe25519_sqr(t3, t2);
    for (int i = 1; i < 20; ++i) fe25519_sqr(t3, t3);

    // z^(2^40 - 1)
    fe25519_mul(t2, t3, t2);

    // z^(2^10 * (2^40 - 1))
    for (int i = 0; i < 10; ++i) fe25519_sqr(t2, t2);

    // z^(2^50 - 1)
    fe25519_mul(t1, t2, t1);

    // z^(2^50 * (2^50 - 1))
    fe25519_sqr(t2, t1);
    for (int i = 1; i < 50; ++i) fe25519_sqr(t2, t2);

    // z^(2^100 - 1)
    fe25519_mul(t2, t2, t1);

    // z^(2^100 * (2^100 - 1))
    fe25519_sqr(t3, t2);
    for (int i = 1; i < 100; ++i) fe25519_sqr(t3, t3);

    // z^(2^200 - 1)
    fe25519_mul(t2, t3, t2);

    // z^(2^50 * (2^200 - 1))
    for (int i = 0; i < 50; ++i) fe25519_sqr(t2, t2);

    // z^(2^250 - 1)
    fe25519_mul(t1, t2, t1);

    // z^(2^5 * (2^250 - 1))
    for (int i = 0; i < 5; ++i) fe25519_sqr(t1, t1);

    // z^(2^255 - 32 + 11) = z^(2^255 - 21) = z^(p-2)
    fe25519_mul(r, t1, t0);     // multiply by z^11
}
