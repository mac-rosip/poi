#pragma once

// =============================================================================
// solana.hpp — Solana address encoding
// =============================================================================
//
// Solana address format:
//   - 32-byte Ed25519 public key → Base58 encode (no checksum)
//   - Typical length: 32-44 characters
//
// Dependencies: base58
// =============================================================================

#include <string>
#include <cstdint>

namespace chain {

// Convert 32-byte Ed25519 public key to Solana Base58 address
std::string solana_address_from_pubkey(const uint8_t pubkey[32]);

} // namespace chain
