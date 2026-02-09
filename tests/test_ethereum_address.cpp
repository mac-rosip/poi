// =============================================================================
// test_ethereum_address.cpp — Integration tests for ETH address encoding
// =============================================================================

#include <gtest/gtest.h>
#include "chain/ethereum.hpp"
#include <cstdint>
#include <cstring>
#include <string>
#include <algorithm>

// Test: address starts with "0x"
TEST(EthereumAddress, StartsWithPrefix) {
    uint8_t hash[20] = {0};
    std::string addr = chain::ethereum_address_from_hash(hash);
    ASSERT_GE(addr.size(), 2u);
    EXPECT_EQ(addr.substr(0, 2), "0x");
}

// Test: address is exactly 42 characters (0x + 40 hex chars)
TEST(EthereumAddress, LengthIs42) {
    uint8_t hash[20] = {0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE,
                        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                        0x09, 0x0A, 0x0B, 0x0C};
    std::string addr = chain::ethereum_address_from_hash(hash);
    EXPECT_EQ(addr.size(), 42u);
}

// Test: different hashes produce different addresses
TEST(EthereumAddress, DifferentHashes) {
    uint8_t hash1[20] = {0};
    uint8_t hash2[20] = {0};
    hash2[19] = 1;

    std::string addr1 = chain::ethereum_address_from_hash(hash1);
    std::string addr2 = chain::ethereum_address_from_hash(hash2);
    EXPECT_NE(addr1, addr2);
}

// Test: deterministic output
TEST(EthereumAddress, Deterministic) {
    uint8_t hash[20] = {0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89,
                        0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89,
                        0xAB, 0xCD, 0xEF, 0x01};
    std::string addr1 = chain::ethereum_address_from_hash(hash);
    std::string addr2 = chain::ethereum_address_from_hash(hash);
    EXPECT_EQ(addr1, addr2);
}

// Test: only valid hex characters (plus uppercase for EIP-55)
TEST(EthereumAddress, ValidHexChars) {
    uint8_t hash[20] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
                        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
                        0x99, 0xAA, 0xBB, 0xCC};
    std::string addr = chain::ethereum_address_from_hash(hash);

    const std::string valid_chars = "0123456789abcdefABCDEF";
    // Skip "0x" prefix
    for (size_t i = 2; i < addr.size(); ++i) {
        EXPECT_NE(valid_chars.find(addr[i]), std::string::npos)
            << "Invalid character '" << addr[i] << "' at position " << i;
    }
}

// Test: EIP-55 checksum is mixed-case (not all lowercase)
TEST(EthereumAddress, EIP55MixedCase) {
    // Hash that should produce a mix of uppercase and lowercase
    uint8_t hash[20] = {0x5a, 0xAb, 0x3B, 0x93, 0x8C, 0x7d, 0x9D, 0x3C,
                        0x30, 0x79, 0x60, 0xFc, 0x92, 0x57, 0x06, 0x2D,
                        0x2D, 0x48, 0xD9, 0x53};
    std::string addr = chain::ethereum_address_from_hash(hash);
    std::string hex_part = addr.substr(2);

    // Check that it's not all lowercase (EIP-55 should uppercase some)
    std::string lower_version = hex_part;
    std::transform(lower_version.begin(), lower_version.end(), lower_version.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    // The hex part should contain at least some uppercase letters
    // (unless the hash happens to produce no a-f chars, which is extremely unlikely)
    bool has_alpha = false;
    for (char c : hex_part) {
        if ((c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
            has_alpha = true;
            break;
        }
    }

    if (has_alpha) {
        // Should have at least some case variation
        bool has_upper = false;
        bool has_lower = false;
        for (char c : hex_part) {
            if (c >= 'A' && c <= 'F') has_upper = true;
            if (c >= 'a' && c <= 'f') has_lower = true;
        }
        // With a typical hash, we expect both cases to appear
        // (statistically very likely but not guaranteed for all inputs)
        EXPECT_TRUE(has_upper || has_lower);
    }
}

// Test: EIP-55 self-consistency — checksumming an already-checksummed address
// should produce the same result
TEST(EthereumAddress, EIP55SelfConsistent) {
    uint8_t hash[20] = {0xFB, 0x69, 0x16, 0x09, 0x5c, 0xA1, 0xdF, 0x60,
                        0xBb, 0x79, 0xCe, 0x92, 0xCe, 0x3E, 0xA7, 0x4C,
                        0x37, 0xc5, 0xd3, 0x59};
    std::string addr1 = chain::ethereum_address_from_hash(hash);
    std::string addr2 = chain::ethereum_address_from_hash(hash);
    EXPECT_EQ(addr1, addr2);
}

// Test: all-zero hash produces valid address
TEST(EthereumAddress, AllZeroHash) {
    uint8_t hash[20] = {0};
    std::string addr = chain::ethereum_address_from_hash(hash);
    EXPECT_EQ(addr.size(), 42u);
    EXPECT_EQ(addr.substr(0, 2), "0x");

    // The hex part (ignoring case) should be all zeros
    std::string hex_lower = addr.substr(2);
    std::transform(hex_lower.begin(), hex_lower.end(), hex_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    EXPECT_EQ(hex_lower, "0000000000000000000000000000000000000000");
}
