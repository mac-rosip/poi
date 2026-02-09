// =============================================================================
// test_bitcoin_address.cpp -- Integration tests for Bitcoin bc1 address encoding
// =============================================================================

#include <gtest/gtest.h>
#include "chain/bitcoin.hpp"
#include <cstdint>
#include <cstring>
#include <string>

// Test: address starts with "bc1q"
TEST(BitcoinAddress, StartsWithBc1q) {
    uint8_t hash[20] = {0};
    std::string addr = chain::bitcoin_address_from_hash(hash);
    ASSERT_GE(addr.size(), 4u);
    EXPECT_EQ(addr.substr(0, 4), "bc1q");
}

// Test: P2WPKH addresses are exactly 42 characters
// bc1 (3) + q (1) + 32 data chars + 6 checksum chars = 42
TEST(BitcoinAddress, LengthIs42) {
    uint8_t hash[20] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    std::string addr = chain::bitcoin_address_from_hash(hash);
    EXPECT_EQ(addr.size(), 42u);
}

// Test: different hashes produce different addresses
TEST(BitcoinAddress, DifferentHashesDifferent) {
    uint8_t h1[20] = {0};
    uint8_t h2[20] = {0};
    h2[0] = 1;

    std::string addr1 = chain::bitcoin_address_from_hash(h1);
    std::string addr2 = chain::bitcoin_address_from_hash(h2);
    EXPECT_NE(addr1, addr2);
}

// Test: deterministic output
TEST(BitcoinAddress, Deterministic) {
    uint8_t hash[20] = {0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89,
                        0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89,
                        0xAB, 0xCD, 0xEF, 0x01};
    std::string addr1 = chain::bitcoin_address_from_hash(hash);
    std::string addr2 = chain::bitcoin_address_from_hash(hash);
    EXPECT_EQ(addr1, addr2);
}

// Test: all characters after "bc1" are valid bech32 characters
TEST(BitcoinAddress, ValidBech32Chars) {
    uint8_t hash[20] = {0xFF, 0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67,
                        0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC, 0xBA, 0x98,
                        0x76, 0x54, 0x32, 0x10};
    std::string addr = chain::bitcoin_address_from_hash(hash);

    const std::string bech32_chars = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";
    // Skip "bc1" prefix (first 3 chars: hrp + separator)
    for (size_t i = 3; i < addr.size(); ++i) {
        EXPECT_NE(bech32_chars.find(addr[i]), std::string::npos)
            << "Invalid bech32 character '" << addr[i] << "' at position " << i;
    }
}

// Test: known BIP-173 test vector
// Private key 1 -> compressed pubkey 0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
// HASH160 of that pubkey = 751e76e8199196d454941c45d1b3a323f1433bd6
// Expected bc1 address: bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4
TEST(BitcoinAddress, KnownTestVector) {
    uint8_t hash[20] = {
        0x75, 0x1e, 0x76, 0xe8, 0x19, 0x91, 0x96, 0xd4, 0x54, 0x94,
        0x1c, 0x45, 0xd1, 0xb3, 0xa3, 0x23, 0xf1, 0x43, 0x3b, 0xd6
    };
    std::string addr = chain::bitcoin_address_from_hash(hash);
    EXPECT_EQ(addr, "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4");
}

// Test: all-zeros hash produces valid address
TEST(BitcoinAddress, AllZeroHash) {
    uint8_t hash[20] = {0};
    std::string addr = chain::bitcoin_address_from_hash(hash);
    EXPECT_EQ(addr.size(), 42u);
    EXPECT_EQ(addr.substr(0, 4), "bc1q");
}

// Test: second known vector -- all-ones hash
// This just verifies we get a consistent, validly-formatted result
TEST(BitcoinAddress, AllOnesHash) {
    uint8_t hash[20];
    memset(hash, 0xFF, 20);
    std::string addr = chain::bitcoin_address_from_hash(hash);
    EXPECT_EQ(addr.size(), 42u);
    EXPECT_EQ(addr.substr(0, 4), "bc1q");

    // Should be different from all-zeros
    uint8_t zero_hash[20] = {0};
    std::string zero_addr = chain::bitcoin_address_from_hash(zero_hash);
    EXPECT_NE(addr, zero_addr);
}
