// =============================================================================
// test_solana_address.cpp — Integration tests for Solana address encoding
// =============================================================================

#include <gtest/gtest.h>
#include "chain/solana.hpp"
#include "chain/base58.hpp"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// Test: address is non-empty for valid pubkey
TEST(SolanaAddress, NonEmpty) {
    uint8_t pubkey[32] = {0};
    pubkey[0] = 1;
    std::string addr = chain::solana_address_from_pubkey(pubkey);
    EXPECT_FALSE(addr.empty());
}

// Test: address length is typically 32-44 characters
TEST(SolanaAddress, AddressLength) {
    uint8_t pubkey[32];
    for (int i = 0; i < 32; ++i) {
        pubkey[i] = static_cast<uint8_t>(i + 1);
    }
    std::string addr = chain::solana_address_from_pubkey(pubkey);
    EXPECT_GE(addr.size(), 32u);
    EXPECT_LE(addr.size(), 44u);
}

// Test: different pubkeys produce different addresses
TEST(SolanaAddress, DifferentPubkeys) {
    uint8_t pk1[32] = {0};
    uint8_t pk2[32] = {0};
    pk1[0] = 1;
    pk2[0] = 2;

    std::string addr1 = chain::solana_address_from_pubkey(pk1);
    std::string addr2 = chain::solana_address_from_pubkey(pk2);
    EXPECT_NE(addr1, addr2);
}

// Test: deterministic output
TEST(SolanaAddress, Deterministic) {
    uint8_t pubkey[32];
    for (int i = 0; i < 32; ++i) pubkey[i] = static_cast<uint8_t>(i * 7 + 3);

    std::string addr1 = chain::solana_address_from_pubkey(pubkey);
    std::string addr2 = chain::solana_address_from_pubkey(pubkey);
    EXPECT_EQ(addr1, addr2);
}

// Test: round-trip — decode address back to bytes
TEST(SolanaAddress, RoundTrip) {
    uint8_t pubkey[32];
    for (int i = 0; i < 32; ++i) pubkey[i] = static_cast<uint8_t>(i + 100);

    std::string addr = chain::solana_address_from_pubkey(pubkey);
    auto decoded = chain::base58_decode(addr);

    ASSERT_EQ(decoded.size(), 32u);
    for (int i = 0; i < 32; ++i) {
        EXPECT_EQ(decoded[i], pubkey[i]) << "Mismatch at byte " << i;
    }
}

// Test: address only contains valid Base58 characters
TEST(SolanaAddress, ValidBase58Chars) {
    uint8_t pubkey[32] = {0xFF, 0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67,
                          0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC, 0xBA, 0x98,
                          0x76, 0x54, 0x32, 0x10, 0xDE, 0xAD, 0xBE, 0xEF,
                          0xCA, 0xFE, 0xBA, 0xBE, 0x11, 0x22, 0x33, 0x44};
    std::string addr = chain::solana_address_from_pubkey(pubkey);

    const std::string base58_chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    for (char c : addr) {
        EXPECT_NE(base58_chars.find(c), std::string::npos)
            << "Invalid character '" << c << "' in address";
    }
}

// Test: known Solana system program address (all zeros = 11111...1)
TEST(SolanaAddress, AllZerosIsOnes) {
    uint8_t pubkey[32] = {0};
    std::string addr = chain::solana_address_from_pubkey(pubkey);
    // 32 zero bytes → 32 leading '1's in Base58
    // The address should be exactly 32 '1' characters
    std::string expected(32, '1');
    EXPECT_EQ(addr, expected);
}

// Test: Solana address has no checksum (unlike Tron)
// Modifying any character and decoding should NOT throw
TEST(SolanaAddress, NoChecksum) {
    uint8_t pubkey[32];
    for (int i = 0; i < 32; ++i) pubkey[i] = static_cast<uint8_t>(i + 50);

    std::string addr = chain::solana_address_from_pubkey(pubkey);

    // Modify a character
    std::string modified = addr;
    if (modified.back() == '1') {
        modified.back() = '2';
    } else {
        modified.back() = '1';
    }

    // base58_decode (not base58_decode_check) should succeed
    EXPECT_NO_THROW({
        auto decoded = chain::base58_decode(modified);
        // Result will be different bytes but shouldn't throw
    });
}
