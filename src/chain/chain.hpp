#pragma once

// =============================================================================
// chain.hpp — Unified chain address encoding interface
// =============================================================================
//
// Provides a common interface for converting raw hash/pubkey bytes to
// human-readable addresses for each supported chain.
//
// TRX:  20-byte hash → Base58Check "T..." address
// ETH:  20-byte hash → EIP-55 checksummed "0x..." address
// SOL:  32-byte pubkey → Base58 address
// BTC:  20-byte HASH160 → bech32 "bc1q..." address
// =============================================================================

#include <string>
#include <vector>
#include <cstdint>
#include <array>
#include "../types.hpp"

namespace chain {

// Forward declarations from chain-specific headers
std::string tron_address_from_hash(const uint8_t hash[20]);
std::string ethereum_address_from_hash(const uint8_t hash[20]);
std::string solana_address_from_pubkey(const uint8_t pubkey[32]);
std::string bitcoin_address_from_hash(const uint8_t hash[20]);

// Unified address encoding dispatch
inline std::string encode_address(ChainType chain, const uint8_t* data) {
    switch (chain) {
        case ChainType::TRON:
            return tron_address_from_hash(data);
        case ChainType::ETHEREUM:
            return ethereum_address_from_hash(data);
        case ChainType::SOLANA:
            return solana_address_from_pubkey(data);
        case ChainType::BITCOIN:
            return bitcoin_address_from_hash(data);
        default:
            return "";
    }
}

} // namespace chain
