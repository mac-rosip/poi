#pragma once

// =============================================================================
// ethereum.hpp — ETH address encoding with EIP-55 checksum
// =============================================================================
//
// ETH address format:
//   1. Take 20-byte Keccak-256 hash of public key (last 20 bytes)
//   2. Hex-encode as lowercase "0x" + 40 hex chars
//   3. Apply EIP-55 mixed-case checksum:
//      - Keccak-256 the lowercase hex string (without "0x")
//      - For each hex digit: if corresponding hash nibble >= 8, uppercase it
//
// Dependencies: none (Keccak implemented inline for host-side EIP-55)
// =============================================================================

#include <string>
#include <cstdint>

namespace chain {

// Convert 20-byte address hash to EIP-55 checksummed ETH address
std::string ethereum_address_from_hash(const uint8_t hash[20]);

} // namespace chain
