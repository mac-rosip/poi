#include "solana.hpp"
#include "base58.hpp"
#include <vector>

namespace chain {

std::string solana_address_from_pubkey(const uint8_t pubkey[32]) {
    // Solana addresses are simply Base58-encoded 32-byte public keys
    // No version byte, no checksum
    std::vector<uint8_t> data(pubkey, pubkey + 32);
    return base58_encode(data);
}

} // namespace chain
