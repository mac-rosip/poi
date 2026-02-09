#pragma once
#include <string>
#include <cstdint>
#include <vector>
#include <stdexcept>

// Convert a single hex character to its 4-bit value.
inline uint8_t hexCharToNibble(char c) {
    if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
    if (c >= 'a' && c <= 'f') return static_cast<uint8_t>(c - 'a' + 10);
    if (c >= 'A' && c <= 'F') return static_cast<uint8_t>(c - 'A' + 10);
    throw std::invalid_argument(std::string("Invalid hex character: ") + c);
}

// Encode a byte array to a lowercase hex string.
inline std::string toHex(const uint8_t* data, size_t len) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) {
        out.push_back(digits[(data[i] >> 4) & 0x0F]);
        out.push_back(digits[data[i] & 0x0F]);
    }
    return out;
}

// Decode a hex string to a byte vector.
inline std::vector<uint8_t> fromHex(const std::string& hex) {
    if (hex.size() % 2 != 0) {
        throw std::invalid_argument("Hex string must have an even number of characters");
    }
    std::vector<uint8_t> bytes;
    bytes.reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2) {
        uint8_t high = hexCharToNibble(hex[i]);
        uint8_t low  = hexCharToNibble(hex[i + 1]);
        bytes.push_back(static_cast<uint8_t>((high << 4) | low));
    }
    return bytes;
}
