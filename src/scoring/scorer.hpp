#pragma once

// =============================================================================
// scorer.hpp — Scorer interface for GPU-side scoring dispatch
// =============================================================================
//
// The Scorer wraps a ScoringConfig and provides methods to:
//   1. Prepare GPU-side scoring parameters (pattern bytes, length, mode)
//   2. Score addresses on the host (for verification)
//   3. Check if a score meets the minimum threshold
//
// The GPU scoring kernels (Wave 3, T28) will consume the exported parameters.
// =============================================================================

#include "mode.hpp"
#include "../chain/chain.hpp"
#include <string>
#include <cstring>

// GPU-friendly scoring parameters (plain C struct, copyable to device)
struct ScoringParams {
    uint8_t pattern[MAX_SCORE];    // Pattern bytes (ASCII)
    uint32_t pattern_len;          // Length of pattern
    uint32_t mode;                 // ScoringMode as uint32_t
    uint32_t min_score;            // Minimum score to report
    uint32_t case_sensitive;       // 1 = case sensitive, 0 = insensitive
};

class Scorer {
public:
    Scorer() = default;

    explicit Scorer(const ScoringConfig& config)
        : config_(config)
    {
        buildParams();
    }

    // Score an address string on the host
    uint32_t score(const std::string& address) const {
        return score_address(address, config_);
    }

    // Check if a score meets the minimum threshold
    bool meets_threshold(uint32_t s) const {
        return s >= config_.min_score;
    }

    // Get the GPU-friendly parameters
    const ScoringParams& params() const { return params_; }

    // Get the config
    const ScoringConfig& config() const { return config_; }

    // Is this a benchmark-only run?
    bool is_benchmark() const {
        return config_.mode == ScoringMode::BENCHMARK;
    }

private:
    ScoringConfig config_;
    ScoringParams params_{};

    void buildParams() {
        memset(&params_, 0, sizeof(params_));
        params_.pattern_len = static_cast<uint32_t>(
            std::min(config_.pattern.size(), (size_t)MAX_SCORE));
        for (uint32_t i = 0; i < params_.pattern_len; ++i) {
            params_.pattern[i] = static_cast<uint8_t>(config_.pattern[i]);
        }
        params_.mode = static_cast<uint32_t>(config_.mode);
        params_.min_score = config_.min_score;
        params_.case_sensitive = config_.case_sensitive ? 1 : 0;
    }
};
