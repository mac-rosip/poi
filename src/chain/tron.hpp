#pragma once

// =============================================================================
// tron.hpp — TRX (Tron) address encoding
// =============================================================================
//
// TRX address format:
//   1. Take 20-byte Keccak-256 hash of public key (last 20 bytes)
//   2. Prepend version byte 0x41
//   3. Compute checksum: first 4 bytes of double-SHA256(21 bytes)
//   4. Append checksum → 25 bytes
//   5. Base58 encode → "T..." address (always starts with T)
//
// Dependencies: sha256, base58
// =============================================================================

#include <string>
#include <cstdint>

namespace chain {

// Convert 20-byte address hash to TRX "T..." address
std::string tron_address_from_hash(const uint8_t hash[20]);

} // namespace chain
