#pragma once

// =============================================================================
// bitcoin.hpp -- Bitcoin bc1 (bech32) address encoding
// =============================================================================
//
// Converts a 20-byte HASH160 (RIPEMD160(SHA256(compressed_pubkey))) into
// a bech32 native SegWit address (P2WPKH, witness version 0).
//
// Output format: bc1q + 32 bech32 chars + 6 checksum chars = 42 characters
//
// Reference: BIP-173 (https://github.com/bitcoin/bips/blob/master/bip-0173.mediawiki)
// =============================================================================

#include <string>
#include <cstdint>

namespace chain {

// Convert 20-byte HASH160 to bech32 bc1q... address (P2WPKH, witness v0)
std::string bitcoin_address_from_hash(const uint8_t hash[20]);

} // namespace chain
