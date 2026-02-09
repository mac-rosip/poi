// =============================================================================
// score_matching.cu — GPU scoring kernel for suffix and anywhere matching
// =============================================================================
//
// Suffix: compare pattern against the end of the address hex/bytes.
// Anywhere: search for the pattern anywhere in the address hex/bytes.
//
// Dependencies: same types as score_prefix.cu
// =============================================================================

#include <cstdint>

struct ScoringParams_d {
    uint8_t pattern[40];
    uint32_t pattern_len;
    uint32_t mode;
    uint32_t min_score;
    uint32_t case_sensitive;
};

struct ScoreResult {
    uint32_t found;
    uint32_t foundId;
    uint8_t foundHash[32];
};

__device__ __forceinline__ uint8_t nibble_to_hex_m(uint8_t n) {
    return n < 10 ? ('0' + n) : ('a' + n - 10);
}

__device__ __forceinline__ int char_match_m(uint8_t a, uint8_t b, uint32_t cs) {
    if (cs) return a == b;
    uint8_t la = (a >= 'A' && a <= 'Z') ? (a + 32) : a;
    uint8_t lb = (b >= 'A' && b <= 'Z') ? (b + 32) : b;
    return la == lb;
}

// =============================================================================
// Kernel: score_suffix_hex — Suffix matching for ETH/TRX (20-byte hash)
// =============================================================================
__global__ void score_suffix_hex(
    const uint8_t* __restrict__ hashes,
    const ScoringParams_d params,
    ScoreResult* __restrict__ results,
    uint32_t num_hashes
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hashes) return;

    const uint8_t* hash = hashes + (size_t)idx * 20;
    const uint32_t hex_len = 40; // 20 bytes = 40 hex chars

    uint32_t score = 0;
    uint32_t max_check = min(params.pattern_len, hex_len);

    // Compare from end
    for (uint32_t i = 0; i < max_check; ++i) {
        uint32_t hex_idx = hex_len - 1 - i;
        uint32_t pat_idx = params.pattern_len - 1 - i;

        uint8_t byte = hash[hex_idx / 2];
        uint8_t nibble = (hex_idx % 2 == 0) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
        uint8_t hex_char = nibble_to_hex_m(nibble);

        if (char_match_m(hex_char, params.pattern[pat_idx], params.case_sensitive)) {
            score++;
        } else {
            break;
        }
    }

    if (score >= params.min_score) {
        uint32_t old = atomicMax(&results[0].found, score);
        if (score > old) {
            results[0].foundId = idx;
            for (int i = 0; i < 20; ++i)
                results[0].foundHash[i] = hash[i];
        }
    }
}

// =============================================================================
// Kernel: score_anywhere_hex — Substring search for ETH/TRX
// =============================================================================
__global__ void score_anywhere_hex(
    const uint8_t* __restrict__ hashes,
    const ScoringParams_d params,
    ScoreResult* __restrict__ results,
    uint32_t num_hashes
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hashes) return;

    const uint8_t* hash = hashes + (size_t)idx * 20;

    // Convert hash to hex in registers
    uint8_t hex_chars[40];
    #pragma unroll
    for (int i = 0; i < 20; ++i) {
        hex_chars[i * 2]     = nibble_to_hex_m((hash[i] >> 4) & 0x0F);
        hex_chars[i * 2 + 1] = nibble_to_hex_m(hash[i] & 0x0F);
    }

    // Search for pattern anywhere in hex string
    uint32_t found = 0;
    if (params.pattern_len <= 40) {
        uint32_t search_end = 40 - params.pattern_len + 1;
        for (uint32_t start = 0; start < search_end && !found; ++start) {
            uint32_t match = 1;
            for (uint32_t j = 0; j < params.pattern_len && match; ++j) {
                if (!char_match_m(hex_chars[start + j], params.pattern[j], params.case_sensitive)) {
                    match = 0;
                }
            }
            if (match) found = 1;
        }
    }

    if (found) {
        uint32_t score = params.pattern_len;
        uint32_t old = atomicMax(&results[0].found, score);
        if (score > old) {
            results[0].foundId = idx;
            for (int i = 0; i < 20; ++i)
                results[0].foundHash[i] = hash[i];
        }
    }
}

// =============================================================================
// Kernel: score_suffix_raw — Suffix matching for Solana (32-byte pubkey)
// =============================================================================
__global__ void score_suffix_raw(
    const uint8_t* __restrict__ pubkeys,
    const ScoringParams_d params,
    ScoreResult* __restrict__ results,
    uint32_t num_keys
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_keys) return;

    const uint8_t* pubkey = pubkeys + (size_t)idx * 32;

    uint32_t score = 0;
    uint32_t max_check = min(params.pattern_len, (uint32_t)32);

    for (uint32_t i = 0; i < max_check; ++i) {
        uint32_t pk_idx = 32 - 1 - i;
        uint32_t pat_idx = params.pattern_len - 1 - i;
        if (char_match_m(pubkey[pk_idx], params.pattern[pat_idx], params.case_sensitive)) {
            score++;
        } else {
            break;
        }
    }

    if (score >= params.min_score) {
        uint32_t old = atomicMax(&results[0].found, score);
        if (score > old) {
            results[0].foundId = idx;
            for (int i = 0; i < 32; ++i)
                results[0].foundHash[i] = pubkey[i];
        }
    }
}

// =============================================================================
// Kernel: score_anywhere_raw — Substring search for Solana
// =============================================================================
__global__ void score_anywhere_raw(
    const uint8_t* __restrict__ pubkeys,
    const ScoringParams_d params,
    ScoreResult* __restrict__ results,
    uint32_t num_keys
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_keys) return;

    const uint8_t* pubkey = pubkeys + (size_t)idx * 32;

    uint32_t found = 0;
    if (params.pattern_len <= 32) {
        uint32_t search_end = 32 - params.pattern_len + 1;
        for (uint32_t start = 0; start < search_end && !found; ++start) {
            uint32_t match = 1;
            for (uint32_t j = 0; j < params.pattern_len && match; ++j) {
                if (!char_match_m(pubkey[start + j], params.pattern[j], params.case_sensitive)) {
                    match = 0;
                }
            }
            if (match) found = 1;
        }
    }

    if (found) {
        uint32_t score = params.pattern_len;
        uint32_t old = atomicMax(&results[0].found, score);
        if (score > old) {
            results[0].foundId = idx;
            for (int i = 0; i < 32; ++i)
                results[0].foundHash[i] = pubkey[i];
        }
    }
}
