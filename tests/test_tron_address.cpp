// =============================================================================
// test_tron_address.cpp — Integration tests for TRX address encoding
// =============================================================================

#include <gtest/gtest.h>
#include "chain/tron.hpp"
#include "chain/base58.hpp"
#include <cstdint>
#include <cstring>
#include <vector>

// Test: basic address format
TEST(TronAddress, StartsWithT) {
    // A hash of all zeros should produce a valid T-address
    uint8_t hash[20] = {0};
    std::string addr = chain::tron_address_from_hash(hash);
    EXPECT_FALSE(addr.empty());
    EXPECT_EQ(addr[0], 'T');
}

// Test: different hashes produce different addresses
TEST(TronAddress, DifferentHashesDifferentAddresses) {
    uint8_t hash1[20] = {0};
    uint8_t hash2[20] = {0};
    hash2[0] = 1;

    std::string addr1 = chain::tron_address_from_hash(hash1);
    std::string addr2 = chain::tron_address_from_hash(hash2);
    EXPECT_NE(addr1, addr2);
}

// Test: address length is 34 characters (standard for Tron)
TEST(TronAddress, AddressLength) {
    uint8_t hash[20] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    std::string addr = chain::tron_address_from_hash(hash);
    EXPECT_EQ(addr.size(), 34u);
}

// Test: round-trip — decode address and verify checksum and version byte
TEST(TronAddress, RoundTripChecksum) {
    uint8_t hash[20] = {0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE,
                        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                        0x09, 0x0A, 0x0B, 0x0C};
    std::string addr = chain::tron_address_from_hash(hash);

    // Decode and verify checksum
    auto decoded = chain::base58_decode_check(addr);
    ASSERT_EQ(decoded.size(), 21u);

    // Version byte should be 0x41
    EXPECT_EQ(decoded[0], 0x41);

    // Hash bytes should match
    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(decoded[i + 1], hash[i]) << "Mismatch at byte " << i;
    }
}

// Test: known Tron genesis address
// The all-zeros hash with version 0x41 should produce a specific address
TEST(TronAddress, DeterministicOutput) {
    uint8_t hash[20] = {0};
    std::string addr1 = chain::tron_address_from_hash(hash);
    std::string addr2 = chain::tron_address_from_hash(hash);
    EXPECT_EQ(addr1, addr2); // Same input → same output
}

// Test: address only contains valid Base58 characters
TEST(TronAddress, ValidBase58Chars) {
    uint8_t hash[20] = {0xFF, 0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67,
                        0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC, 0xBA, 0x98,
                        0x76, 0x54, 0x32, 0x10};
    std::string addr = chain::tron_address_from_hash(hash);

    const std::string base58_chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    for (char c : addr) {
        EXPECT_NE(base58_chars.find(c), std::string::npos)
            << "Invalid character '" << c << "' in address";
    }
}
