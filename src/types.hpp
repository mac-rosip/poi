#pragma once
#include <cstdint>

// Guard for CUDA vs host compilation
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

#define MP_WORDS 8    // 256-bit numbers as 8x 32-bit words
#define MAX_SCORE 40  // Maximum vanity score

// 256-bit number, little-endian word order
struct mp_number {
    uint32_t d[MP_WORDS];
};

// secp256k1 affine point
struct secp256k1_point {
    mp_number x;
    mp_number y;
};

// Result structure for vanity matches
struct result {
    uint32_t found;
    uint32_t foundId;
    uint8_t foundHash[32];  // 32 bytes covers both TRX (20) and Solana (32)
};

// Random seed -- 256 bits as 4x uint64_t
struct seed256 {
    uint64_t s[4];
};

// Chain types
enum class ChainType : uint8_t {
    TRON = 0,
    ETHEREUM = 1,
    SOLANA = 2,
};

// Curve types
enum class CurveType : uint8_t {
    SECP256K1 = 0,
    ED25519 = 1,
};
