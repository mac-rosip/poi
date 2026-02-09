#include "mode.hpp"
#include <algorithm>
#include <cctype>

// =============================================================================
// score_address — Score a generated address against the target pattern
//
// For PREFIX:   Count matching characters from the start (after prefix like "0x" or "T")
// For SUFFIX:   Count matching characters from the end
// For ANYWHERE: Find the longest matching substring
// For BENCHMARK: Always returns 0 (no pattern matching)
//
// Returns: number of matching characters (0 to MAX_SCORE)
// =============================================================================
uint32_t score_address(const std::string& address, const ScoringConfig& config) {
    if (config.mode == ScoringMode::BENCHMARK || config.pattern.empty()) {
        return 0;
    }

    // Determine the scorable portion of the address (skip chain prefix)
    std::string addr_body;
    switch (config.chain) {
        case ChainType::ETHEREUM:
            // Skip "0x" prefix
            addr_body = (address.size() > 2 && address[0] == '0' && address[1] == 'x')
                ? address.substr(2) : address;
            break;
        case ChainType::TRON:
            // Skip "T" prefix (first char is always T for Tron)
            addr_body = (address.size() > 1 && address[0] == 'T')
                ? address.substr(1) : address;
            break;
        case ChainType::SOLANA:
            // No prefix to skip
            addr_body = address;
            break;
    }

    std::string pat = config.pattern;

    // Case-insensitive comparison if configured
    if (!config.case_sensitive) {
        std::transform(addr_body.begin(), addr_body.end(), addr_body.begin(),
            [](unsigned char c) { return std::tolower(c); });
        std::transform(pat.begin(), pat.end(), pat.begin(),
            [](unsigned char c) { return std::tolower(c); });
    }

    uint32_t score = 0;

    switch (config.mode) {
        case ScoringMode::PREFIX: {
            size_t max_len = std::min(addr_body.size(), pat.size());
            for (size_t i = 0; i < max_len; ++i) {
                if (addr_body[i] == pat[i]) {
                    ++score;
                } else {
                    break;
                }
            }
            break;
        }

        case ScoringMode::SUFFIX: {
            size_t max_len = std::min(addr_body.size(), pat.size());
            for (size_t i = 0; i < max_len; ++i) {
                size_t addr_idx = addr_body.size() - 1 - i;
                size_t pat_idx = pat.size() - 1 - i;
                if (addr_body[addr_idx] == pat[pat_idx]) {
                    ++score;
                } else {
                    break;
                }
            }
            break;
        }

        case ScoringMode::ANYWHERE: {
            // Find the pattern anywhere in the address body
            // Score = length of pattern if found, 0 otherwise
            if (addr_body.find(pat) != std::string::npos) {
                score = static_cast<uint32_t>(pat.size());
            }
            break;
        }

        case ScoringMode::BENCHMARK:
            score = 0;
            break;
    }

    return std::min(score, (uint32_t)MAX_SCORE);
}

// =============================================================================
// parse_scoring_mode — Convert string to ScoringMode enum
// =============================================================================
ScoringMode parse_scoring_mode(const std::string& str) {
    if (str == "prefix")    return ScoringMode::PREFIX;
    if (str == "suffix")    return ScoringMode::SUFFIX;
    if (str == "anywhere" || str == "contains") return ScoringMode::ANYWHERE;
    if (str == "benchmark") return ScoringMode::BENCHMARK;
    // Default to prefix
    return ScoringMode::PREFIX;
}

// =============================================================================
// scoring_mode_name — Convert ScoringMode to display string
// =============================================================================
const char* scoring_mode_name(ScoringMode mode) {
    switch (mode) {
        case ScoringMode::PREFIX:    return "prefix";
        case ScoringMode::SUFFIX:    return "suffix";
        case ScoringMode::ANYWHERE:  return "anywhere";
        case ScoringMode::BENCHMARK: return "benchmark";
        default:                     return "unknown";
    }
}
