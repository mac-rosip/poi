#include "tron.hpp"
#include "base58.hpp"
#include "../crypto/sha256.hpp"
#include <vector>

namespace chain {

std::string tron_address_from_hash(const uint8_t hash[20]) {
    // 1. Prepend version byte 0x41 (Tron mainnet)
    std::vector<uint8_t> payload(21);
    payload[0] = 0x41;
    for (int i = 0; i < 20; ++i) {
        payload[i + 1] = hash[i];
    }

    // 2. Base58Check encode (appends double-SHA256 checksum)
    return base58_encode_check(payload);
}

} // namespace chain
