#pragma once

// =============================================================================
// ed25519_precomp.cuh — Precomputed base point multiples for Ed25519
// =============================================================================
//
// Table of 256 precomputed points: B, 2B, 4B, 8B, ..., 2^255 * B
// Each entry i holds 2^i * B in Niels form (yPlusX, yMinusX, xy2d)
// for fast mixed addition during scalar multiplication.
//
// Used by the keygen kernel for fixed-base scalar multiplication.
//
// Format: arrays of ge25519_niels indexed by bit position [0..255].
//
// NOTE: Like the secp256k1 precomp, this table is populated at runtime
// by the host. The init routine computes the table via repeated doubling
// of the base point, converts each to Niels form, and copies to device.
//
// Dependencies: ge25519.cuh (for ge25519_niels, ge25519_p3, fe25519)
// =============================================================================

#include "ge25519.cuh"

// Number of precomputed points (one per bit of the 255-bit scalar)
#define ED25519_PRECOMP_SIZE 256

// Device constant memory for the precomputed Niels table
// Note: Defined in ed25519_keygen.cu, declared extern here for other .cu files
extern __device__ __constant__ ge25519_niels d_ed25519_precomp[ED25519_PRECOMP_SIZE];
