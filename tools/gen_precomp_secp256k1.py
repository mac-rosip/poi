#!/usr/bin/env python3

"""
Generate precomputed tables for secp256k1 point operations.

This script generates lookup tables for efficient point multiplication
on the secp256k1 curve.
"""

import sys

# secp256k1 parameters
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
A = 0
B = 7
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

def mod_inverse(a, m):
    """Modular inverse using extended Euclidean algorithm."""
    m0, y, x = m, 0, 1
    if m == 1:
        return 0
    while a > 1:
        q = a // m
        m, a = a % m, m
        y, x = x - q * y, y
    if x < 0:
        x += m0
    return x

def point_add(x1, y1, x2, y2, p):
    """Add two points on elliptic curve."""
    if x1 == x2 and y1 == y2:
        return point_double(x1, y1, p)
    if x1 == x2:
        return (0, 0)  # Point at infinity
    lam = ((y2 - y1) * mod_inverse(x2 - x1, p)) % p
    x3 = (lam * lam - x1 - x2) % p
    y3 = (lam * (x1 - x3) - y1) % p
    return (x3, y3)

def point_double(x, y, p):
    """Double a point on elliptic curve."""
    lam = ((3 * x * x + A) * mod_inverse(2 * y, p)) % p
    x3 = (lam * lam - 2 * x) % p
    y3 = (lam * (x - x3) - y) % p
    return (x3, y3)

def generate_precomp_table(base_x, base_y, p, size=256):
    """Generate precomputed table for fixed-base multiplication."""
    points = [(0, 0)]  # Index 0: point at infinity
    current_x, current_y = base_x, base_y
    for i in range(1, size + 1):
        points.append((current_x, current_y))
        current_x, current_y = point_double(current_x, current_y, p)
    return points

def main():
    print("Generating secp256k1 precomputed table for base point G...")
    table = generate_precomp_table(Gx, Gy, P, 256)

    print("Precomputed table:")
    print("const uint64_t SECP256K1_PRECOMP_X[257] = {")
    for i, (x, y) in enumerate(table):
        if i % 4 == 0:
            print(f"    0x{x:064X}ULL,", end="")
        else:
            print(f" 0x{x:064X}ULL,", end="")
        if (i + 1) % 4 == 0:
            print()
    print("};")
    print()
    print("const uint64_t SECP256K1_PRECOMP_Y[257] = {")
    for i, (x, y) in enumerate(table):
        if i % 4 == 0:
            print(f"    0x{y:064X}ULL,", end="")
        else:
            print(f" 0x{y:064X}ULL,", end="")
        if (i + 1) % 4 == 0:
            print()
    print("};")

if __name__ == "__main__":
    main()