// =============================================================================
// score_prefix.cu — GPU scoring kernel for prefix matching
// =============================================================================
//
// Compares each address hash against a target pattern from the start.
// Supports both hex-encoded (ETH/TRX) and raw byte (Solana) matching.
//
// For ETH/TRX: the hash is 20 bytes, compared as hex nibbles against pattern.
// For Solana: the pubkey is 32 bytes, compared as Base58 characters
//             (but on GPU we compare raw bytes; Base58 scoring on host).
//
// If the score meets the minimum threshold, the thread's index is recorded
// in the result buffer for the host to collect.
//
// Dependencies: types (ScoringParams, result)
// =============================================================================

#include <cstdint>

// ScoringParams is a POD struct copied from host
struct ScoringParams_d {
    uint8_t pattern[40];
    uint32_t pattern_len;
    uint32_t mode;
    uint32_t min_score;
    uint32_t case_sensitive;
};

// Result structure (matches host-side result struct)
struct ScoreResult {
    uint32_t found;
    uint32_t foundId;
    uint8_t foundHash[32];
};

// =============================================================================
// Device helper: convert a nibble to hex character
// =============================================================================
__device__ __forceinline__ uint8_t nibble_to_hex(uint8_t n) {
    return n < 10 ? ('0' + n) : ('a' + n - 10);
}

// =============================================================================
// Device helper: case-insensitive character comparison
// =============================================================================
__device__ __forceinline__ int char_match(uint8_t a, uint8_t b, uint32_t case_sensitive) {
    if (case_sensitive) return a == b;
    // Convert both to lowercase for comparison
    uint8_t la = (a >= 'A' && a <= 'Z') ? (a + 32) : a;
    uint8_t lb = (b >= 'A' && b <= 'Z') ? (b + 32) : b;
    return la == lb;
}

// =============================================================================
// Kernel: score_prefix_hex
//
// For ETH/TRX chains: score 20-byte hash as hex prefix match.
// Each byte produces 2 hex nibbles; compare against pattern.
//
// Inputs:
//   hashes[]   — 20-byte address hashes
//   params     — scoring parameters (pattern, min_score)
//   results    — output: best match per block
//   num_hashes — total count
// =============================================================================
__global__ void score_prefix_hex(
    const uint8_t* __restrict__ hashes,
    const ScoringParams_d params,
    ScoreResult* __restrict__ results,
    uint32_t num_hashes
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hashes) return;

    const uint8_t* hash = hashes + (size_t)idx * 20;

    // Compare hex nibbles against pattern
    uint32_t score = 0;
    uint32_t max_check = min(params.pattern_len, (uint32_t)40); // 20 bytes = 40 hex chars

    for (uint32_t i = 0; i < max_check; ++i) {
        uint8_t byte = hash[i / 2];
        uint8_t nibble = (i % 2 == 0) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
        uint8_t hex_char = nibble_to_hex(nibble);

        if (char_match(hex_char, params.pattern[i], params.case_sensitive)) {
            score++;
        } else {
            break;
        }
    }

    // Check if score meets threshold
    if (score >= params.min_score) {
        // Atomic: record this result if better than current best
        uint32_t old = atomicMax(&results[0].found, score);
        if (score > old) {
            results[0].foundId = idx;
            // Copy hash
            for (int i = 0; i < 20; ++i) {
                results[0].foundHash[i] = hash[i];
            }
        }
    }
}

// =============================================================================
// Kernel: score_prefix_raw
//
// For Solana: score 32-byte pubkey bytes as prefix match.
// Pattern bytes are compared directly against pubkey bytes.
//
// Inputs:
//   pubkeys[]  — 32-byte compressed pubkeys
//   params     — scoring parameters
//   results    — output
//   num_keys   — total count
// =============================================================================
__global__ void score_prefix_raw(
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
        if (char_match(pubkey[i], params.pattern[i], params.case_sensitive)) {
            score++;
        } else {
            break;
        }
    }

    if (score >= params.min_score) {
        uint32_t old = atomicMax(&results[0].found, score);
        if (score > old) {
            results[0].foundId = idx;
            for (int i = 0; i < 32; ++i) {
                results[0].foundHash[i] = pubkey[i];
            }
        }
    }
}

// =============================================================================
// Host-callable wrappers
// =============================================================================
extern "C" void score_prefix_hex_launch(
    const uint8_t* d_hashes,
    const ScoringParams_d* d_params,
    ScoreResult* d_results,
    uint32_t num_hashes,
    cudaStream_t stream
) {
    ScoringParams_d h_params;
    cudaMemcpy(&h_params, d_params, sizeof(ScoringParams_d), cudaMemcpyDeviceToHost);

    const uint32_t block_size = 256;
    const uint32_t grid_size = (num_hashes + block_size - 1) / block_size;

    score_prefix_hex<<<grid_size, block_size, 0, stream>>>(
        d_hashes, h_params, d_results, num_hashes
    );
}

extern "C" void score_prefix_raw_launch(
    const uint8_t* d_pubkeys,
    const ScoringParams_d* d_params,
    ScoreResult* d_results,
    uint32_t num_keys,
    cudaStream_t stream
) {
    ScoringParams_d h_params;
    cudaMemcpy(&h_params, d_params, sizeof(ScoringParams_d), cudaMemcpyDeviceToHost);

    const uint32_t block_size = 256;
    const uint32_t grid_size = (num_keys + block_size - 1) / block_size;

    score_prefix_raw<<<grid_size, block_size, 0, stream>>>(
        d_pubkeys, h_params, d_results, num_keys
    );
}
