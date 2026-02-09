#include "bitcoin.hpp"
#include <vector>
#include <string>
#include <stdexcept>

namespace {

// Bech32 character set (BIP-173)
const char* BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";

// Bech32 polymod -- GF(2^5) polynomial checksum
uint32_t bech32_polymod(const std::vector<uint8_t>& values) {
    const uint32_t GEN[5] = {
        0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3
    };
    uint32_t chk = 1;
    for (auto v : values) {
        uint32_t top = chk >> 25;
        chk = ((chk & 0x1ffffff) << 5) ^ v;
        for (int i = 0; i < 5; ++i) {
            if ((top >> i) & 1) {
                chk ^= GEN[i];
            }
        }
    }
    return chk;
}

// Expand human-readable part for checksum computation
std::vector<uint8_t> bech32_hrp_expand(const std::string& hrp) {
    std::vector<uint8_t> ret;
    ret.reserve(hrp.size() * 2 + 1);
    for (char c : hrp) {
        ret.push_back(c >> 5);
    }
    ret.push_back(0);
    for (char c : hrp) {
        ret.push_back(c & 31);
    }
    return ret;
}

// Create 6-byte bech32 checksum
std::vector<uint8_t> bech32_create_checksum(const std::string& hrp,
                                              const std::vector<uint8_t>& values) {
    auto hrp_exp = bech32_hrp_expand(hrp);
    std::vector<uint8_t> enc;
    enc.reserve(hrp_exp.size() + values.size() + 6);
    enc.insert(enc.end(), hrp_exp.begin(), hrp_exp.end());
    enc.insert(enc.end(), values.begin(), values.end());
    // Append 6 zero bytes for checksum computation
    enc.resize(enc.size() + 6, 0);

    uint32_t polymod = bech32_polymod(enc) ^ 1;

    std::vector<uint8_t> chk(6);
    for (int i = 0; i < 6; ++i) {
        chk[i] = (polymod >> (5 * (5 - i))) & 31;
    }
    return chk;
}

// Convert from 8-bit groups to 5-bit groups
// frombits=8, tobits=5, pad=true
std::vector<uint8_t> convert_bits_8_to_5(const uint8_t* data, size_t len) {
    std::vector<uint8_t> ret;
    ret.reserve((len * 8 + 4) / 5);

    uint32_t acc = 0;
    int bits = 0;

    for (size_t i = 0; i < len; ++i) {
        acc = (acc << 8) | data[i];
        bits += 8;
        while (bits >= 5) {
            bits -= 5;
            ret.push_back((acc >> bits) & 31);
        }
    }

    // Pad remaining bits
    if (bits > 0) {
        ret.push_back((acc << (5 - bits)) & 31);
    }

    return ret;
}

} // anonymous namespace

namespace chain {

std::string bitcoin_address_from_hash(const uint8_t hash[20]) {
    const std::string hrp = "bc";
    const uint8_t witness_version = 0;

    // Convert 20-byte witness program from 8-bit to 5-bit groups
    auto prog_5bit = convert_bits_8_to_5(hash, 20);

    // Prepend witness version (0 = 'q' in bech32)
    std::vector<uint8_t> values;
    values.reserve(1 + prog_5bit.size());
    values.push_back(witness_version);
    values.insert(values.end(), prog_5bit.begin(), prog_5bit.end());

    // Compute checksum
    auto checksum = bech32_create_checksum(hrp, values);

    // Build the address string
    // Format: hrp + "1" + data_chars + checksum_chars
    std::string result = hrp + "1";
    result.reserve(result.size() + values.size() + checksum.size());

    for (auto v : values) {
        result += BECH32_CHARSET[v];
    }
    for (auto v : checksum) {
        result += BECH32_CHARSET[v];
    }

    return result;
}

} // namespace chain
