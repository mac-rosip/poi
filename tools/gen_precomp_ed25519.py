#!/usr/bin/env python3

"""
Generate precomputed tables for Ed25519 point operations.

This script generates lookup tables for efficient point multiplication
on the Ed25519 curve.
"""

import sys

# Ed25519 parameters
P = 2**255 - 19
A = -1  # Actually 486662, but for Montgomery: a = 486662, but for Edwards it's -1
D = 0x52036CEE2B6FFE738CC740797779E89800700A4D4141D8AB75EB4DCA135978A3  # d = -121665/121666 mod p

# Base point
Gx = 0x216936D3CD6E53FEC0A4E231FDD6DC5C692CC7609525A7B2C9562D608F25D51A
Gy = 0x6666666666666666666666666666666666666666666666666666666666666658

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

def ed25519_point_add(x1, y1, x2, y2, p, d):
    """Add two points on Ed25519 curve (Edwards form)."""
    # Edwards addition: (x1,y1) + (x2,y2) = ((x1*y2 + y1*x2)/(1 + d*x1*y1*x2*y2), (y1*y2 - x1*x2)/(1 - d*x1*y1*x2*y2))
    denom1 = (1 + d * x1 * y1 * x2 * y2) % p
    denom2 = (1 - d * x1 * y1 * x2 * y2) % p
    x3 = ((x1 * y2 + y1 * x2) * mod_inverse(denom1, p)) % p
    y3 = ((y1 * y2 - x1 * x2) * mod_inverse(denom2, p)) % p
    return (x3, y3)

def ed25519_point_double(x, y, p, d):
    """Double a point on Ed25519 curve."""
    return ed25519_point_add(x, y, x, y, p, d)

def generate_precomp_table(base_x, base_y, p, d, size=256):
    """Generate precomputed table for fixed-base multiplication."""
    points = [(0, 1)]  # Identity: (0,1)
    current_x, current_y = base_x, base_y
    for i in range(1, size + 1):
        points.append((current_x, current_y))
        current_x, current_y = ed25519_point_double(current_x, current_y, p, d)
    return points

def main():
    print("Generating Ed25519 precomputed table for base point...")
    table = generate_precomp_table(Gx, Gy, P, D, 256)

    print("Precomputed table:")
    print("const uint64_t ED25519_PRECOMP_X[257] = {")
    for i, (x, y) in enumerate(table):
        if i % 4 == 0:
            print(f"    0x{x:064X}ULL,", end="")
        else:
            print(f" 0x{x:064X}ULL,", end="")
        if (i + 1) % 4 == 0:
            print()
    print("};")
    print()
    print("const uint64_t ED25519_PRECOMP_Y[257] = {")
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