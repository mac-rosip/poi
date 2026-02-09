#include "ethereum.hpp"
#include <cstring>

namespace chain {

// =============================================================================
// Host-side Keccak-256 for EIP-55 checksum computation
// This is a minimal implementation — the GPU version is in keccak.cuh.
// Uses Ethereum's 0x01 padding (not SHA-3's 0x06).
// =============================================================================

namespace {

static const uint64_t keccak_rc[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

static inline uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

void keccakf(uint64_t state[25]) {
    for (int round = 0; round < 24; ++round) {
        // Theta
        uint64_t C[5], D[5];
        for (int x = 0; x < 5; ++x)
            C[x] = state[x] ^ state[x+5] ^ state[x+10] ^ state[x+15] ^ state[x+20];
        for (int x = 0; x < 5; ++x) {
            D[x] = C[(x+4)%5] ^ rotl64(C[(x+1)%5], 1);
            for (int y = 0; y < 25; y += 5)
                state[x+y] ^= D[x];
        }

        // Rho + Pi
        uint64_t temp = state[1];
        static const int pi_lane[24] = {
            10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
            15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
        };
        static const int rho_off[24] = {
            1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
            27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
        };
        for (int i = 0; i < 24; ++i) {
            int j = pi_lane[i];
            uint64_t t2 = state[j];
            state[j] = rotl64(temp, rho_off[i]);
            temp = t2;
        }

        // Chi
        for (int y = 0; y < 25; y += 5) {
            uint64_t T[5];
            for (int x = 0; x < 5; ++x)
                T[x] = state[y+x];
            for (int x = 0; x < 5; ++x)
                state[y+x] = T[x] ^ ((~T[(x+1)%5]) & T[(x+2)%5]);
        }

        // Iota
        state[0] ^= keccak_rc[round];
    }
}

void host_keccak256(uint8_t out[32], const uint8_t* input, size_t len) {
    uint64_t state[25];
    memset(state, 0, sizeof(state));

    const size_t rate = 136; // bytes (1088 bits for Keccak-256)
    size_t offset = 0;

    // Absorb full blocks
    while (len >= rate) {
        for (size_t i = 0; i < rate / 8; ++i) {
            uint64_t word = 0;
            for (int j = 0; j < 8; ++j)
                word |= ((uint64_t)input[offset + i*8 + j]) << (8*j);
            state[i] ^= word;
        }
        keccakf(state);
        offset += rate;
        len -= rate;
    }

    // Pad remaining
    uint8_t buffer[136];
    memset(buffer, 0, sizeof(buffer));
    memcpy(buffer, input + offset, len);
    buffer[len] = 0x01;         // Ethereum Keccak padding
    buffer[rate - 1] |= 0x80;  // End bit

    for (size_t i = 0; i < rate / 8; ++i) {
        uint64_t word = 0;
        for (int j = 0; j < 8; ++j)
            word |= ((uint64_t)buffer[i*8 + j]) << (8*j);
        state[i] ^= word;
    }
    keccakf(state);

    // Squeeze 32 bytes (little-endian)
    memcpy(out, state, 32);
}

} // anonymous namespace

// =============================================================================
// EIP-55 checksummed Ethereum address from 20-byte hash
// =============================================================================
std::string ethereum_address_from_hash(const uint8_t hash[20]) {
    static const char hex_lower[] = "0123456789abcdef";

    // Step 1: Convert hash to lowercase hex string (40 chars, no "0x")
    char hex_addr[41];
    for (int i = 0; i < 20; ++i) {
        hex_addr[i*2]     = hex_lower[(hash[i] >> 4) & 0x0F];
        hex_addr[i*2 + 1] = hex_lower[hash[i] & 0x0F];
    }
    hex_addr[40] = '\0';

    // Step 2: Keccak-256 of the lowercase hex string
    uint8_t addr_hash[32];
    host_keccak256(addr_hash, reinterpret_cast<const uint8_t*>(hex_addr), 40);

    // Step 3: Apply EIP-55 mixed-case checksum
    // For each hex digit: if the corresponding nibble in the hash >= 8, uppercase it
    std::string result = "0x";
    result.reserve(42);
    for (int i = 0; i < 40; ++i) {
        uint8_t hash_nibble = (addr_hash[i / 2] >> ((1 - (i % 2)) * 4)) & 0x0F;
        char c = hex_addr[i];
        if (c >= 'a' && c <= 'f' && hash_nibble >= 8) {
            c -= 32; // to uppercase
        }
        result += c;
    }

    return result;
}

} // namespace chain
