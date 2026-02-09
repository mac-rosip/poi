// =============================================================================
// test_base58.cpp — Unit tests for Base58 / Base58Check encoding
// =============================================================================

#include <gtest/gtest.h>
#include "chain/base58.hpp"
#include <vector>
#include <string>
#include <cstdint>

// ---- Base58 Encode Tests ----

TEST(Base58, EncodeEmpty) {
    std::vector<uint8_t> data;
    EXPECT_EQ(chain::base58_encode(data), "");
}

TEST(Base58, EncodeSingleZero) {
    std::vector<uint8_t> data = {0};
    EXPECT_EQ(chain::base58_encode(data), "1");
}

TEST(Base58, EncodeMultipleLeadingZeros) {
    std::vector<uint8_t> data = {0, 0, 0, 1};
    std::string encoded = chain::base58_encode(data);
    // Should start with three '1's (leading zeros)
    EXPECT_EQ(encoded.substr(0, 3), "111");
    EXPECT_EQ(encoded, "1112");
}

TEST(Base58, EncodeHelloWorld) {
    // "Hello World" in Base58 = JxF12TrwUP45BMd
    std::string input = "Hello World";
    std::vector<uint8_t> data(input.begin(), input.end());
    EXPECT_EQ(chain::base58_encode(data), "JxF12TrwUP45BMd");
}

TEST(Base58, EncodeSmallNumber) {
    // 0x01 = "2" in Base58
    std::vector<uint8_t> data = {1};
    EXPECT_EQ(chain::base58_encode(data), "2");
}

TEST(Base58, EncodeByte57) {
    // 57 = "z" in Base58 (last char)
    std::vector<uint8_t> data = {57};
    EXPECT_EQ(chain::base58_encode(data), "Q");
}

// ---- Base58 Decode Tests ----

TEST(Base58, DecodeEmpty) {
    auto result = chain::base58_decode("");
    EXPECT_TRUE(result.empty());
}

TEST(Base58, DecodeSingle1) {
    auto result = chain::base58_decode("1");
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], 0);
}

TEST(Base58, DecodeHelloWorld) {
    auto result = chain::base58_decode("JxF12TrwUP45BMd");
    std::string decoded(result.begin(), result.end());
    EXPECT_EQ(decoded, "Hello World");
}

TEST(Base58, InvalidCharacter) {
    EXPECT_THROW(chain::base58_decode("0OIl"), std::invalid_argument);
}

// ---- Round-trip Tests ----

TEST(Base58, RoundTrip) {
    std::vector<uint8_t> original = {0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE};
    auto encoded = chain::base58_encode(original);
    auto decoded = chain::base58_decode(encoded);
    EXPECT_EQ(original, decoded);
}

TEST(Base58, RoundTripWithLeadingZeros) {
    std::vector<uint8_t> original = {0, 0, 0xDE, 0xAD};
    auto encoded = chain::base58_encode(original);
    auto decoded = chain::base58_decode(encoded);
    EXPECT_EQ(original, decoded);
}

// ---- Base58Check Tests ----

TEST(Base58Check, EncodeDecodeRoundTrip) {
    std::vector<uint8_t> data = {0x41, 0x01, 0x02, 0x03, 0x04};
    auto encoded = chain::base58_encode_check(data);
    auto decoded = chain::base58_decode_check(encoded);
    EXPECT_EQ(data, decoded);
}

TEST(Base58Check, InvalidChecksum) {
    std::vector<uint8_t> data = {0x41, 0x01, 0x02, 0x03, 0x04};
    auto encoded = chain::base58_encode_check(data);

    // Corrupt the encoded string
    std::string corrupted = encoded;
    if (corrupted.back() == '1') {
        corrupted.back() = '2';
    } else {
        corrupted.back() = '1';
    }

    EXPECT_THROW(chain::base58_decode_check(corrupted), std::invalid_argument);
}

TEST(Base58Check, TronStyleAddress) {
    // Tron addresses start with version byte 0x41
    std::vector<uint8_t> addr_bytes(21);
    addr_bytes[0] = 0x41;
    for (int i = 1; i <= 20; ++i) {
        addr_bytes[i] = static_cast<uint8_t>(i);
    }

    auto encoded = chain::base58_encode_check(addr_bytes);

    // Should start with 'T' (Tron mainnet prefix)
    EXPECT_EQ(encoded[0], 'T');

    // Decode should recover original
    auto decoded = chain::base58_decode_check(encoded);
    EXPECT_EQ(addr_bytes, decoded);
}
