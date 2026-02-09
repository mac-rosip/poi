#pragma once

// =============================================================================
// mode.hpp — Scoring mode definitions for vanity address matching
// =============================================================================
//
// Scoring modes determine how a generated address is evaluated against
// the user's desired pattern. Higher scores = better match.
//
// Modes:
//   PREFIX    — Match from the start of the address (most common)
//   SUFFIX    — Match from the end of the address
//   ANYWHERE  — Match anywhere in the address (contains)
//   BENCHMARK — No pattern; used for raw hashrate measurement
//
// The ScoringConfig bundles the mode with the target pattern and chain.
// =============================================================================

#include <string>
#include <cstdint>
#include "../types.hpp"

// Scoring mode enum
enum class ScoringMode : uint8_t {
    PREFIX = 0,
    SUFFIX = 1,
    ANYWHERE = 2,
    BENCHMARK = 3,
};

// Configuration for scoring
struct ScoringConfig {
    ScoringMode mode;
    ChainType chain;
    std::string pattern;        // Target pattern (case-insensitive for hex chains)
    bool case_sensitive;        // Whether to match case (false for ETH, true for TRX/SOL)
    uint32_t min_score;         // Minimum score to report as result

    ScoringConfig()
        : mode(ScoringMode::BENCHMARK)
        , chain(ChainType::ETHEREUM)
        , pattern("")
        , case_sensitive(false)
        , min_score(1)
    {}
};

// Score an address against a pattern
uint32_t score_address(const std::string& address, const ScoringConfig& config);

// Parse scoring mode from string
ScoringMode parse_scoring_mode(const std::string& str);

// Convert scoring mode to display string
const char* scoring_mode_name(ScoringMode mode);
